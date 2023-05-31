from RVSNUpy import correlation
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from astropy.modeling import models, fitting
from RVSNUpy import continuum
import warnings
from tqdm import tqdm, notebook
from astropy.constants import c
from joblib import Parallel, delayed
import copy
c = c.value/1e+3


warnings.filterwarnings("ignore") # ignore warning

def z_finding(corr, lag, pkfrac, template_dispersion = [0,0], correlation_range=[-0.01,2]):
    '''

    Parameters
    ----------
    corr : numpy.array
        cross-correaltion signal
    lag : numpy.array
        lag in cross-correlating
    pkfrac : float
        ratio of the minimum signal to the peak to be used in the fitting
    plotting : bool, optional
        If true, plots the result of the cross-correlation and fitted curve. 
        The default is False.
    correlation_range : list of length 2, optional
        The range of the redshift estimation. The default is [-0.01,2].

    Returns
    -------
    z : float
        estimated redshift
    r : float
        estmiatedr
    error : float
        estmiated error

    '''

    correlation_range = np.array(correlation_range)*c
    _corr, _lag = corr[lag>correlation_range[0]], lag[lag>correlation_range[0]]
    _corr, _lag = _corr[_lag<correlation_range[1]], _lag[_lag<correlation_range[1]]
    
    peak = np.nanmax(_corr) # estimates a peak
    
    if math.isnan(peak): # If the peak is not found,
        z, r, error, dispersion, dispersion_err, fit_parabol, fit_gaussian, lag_fit, npeak, nrange, result  = \
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'Peak is not found'
    
    else: # If the peak is founded
        corr_fit, lag_fit = corr[corr >= pkfrac*peak], lag[corr >= pkfrac*peak] # select points >pkfrac*peak
        if np.nanmax(lag_fit[1:]-lag_fit[:-1]) > 600 or len(corr_fit) < 3: # too small number of points
            z, r, error, dispersion, dispersion_err, fit_parabol, fit_gaussian, lag_fit, npeak, nrange, result  = \
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'Too small number of points'
            
        else:
            center = lag[np.nanargmax(corr_fit)]
            c2 = (peak - peak*pkfrac)/(center-lag_fit[0])**2
            c1= 2*c2*center
            c0 = peak + (c1**2)/(4*c2)
            # fit with the parabola
            parabol = models.Polynomial1D(2, c2=c2, c1=c1, c0=c0)
            fit = fitting.LevMarLSQFitter()
            fit_parabol = fit(parabol, lag_fit, corr_fit)
            fitted_center = -fit_parabol.c1/(2*fit_parabol.c2) # find the center
            if c1 > 0 or fitted_center<correlation_range[0] or fitted_center>correlation_range[1]:
                z, r, error, dispersion, dispersion_err, fit_parabol, fit_gaussian, lag_fit, npeak, nrange, result  = \
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'Parabol is not convex'
            else:
                h = fit_parabol(fitted_center) # find the peak
                z = fitted_center/c # estimate the redshift
                # calculate sigma^2 abd r
                npeak = np.abs(lag-fitted_center).argmin() # find an index of peak
                N = int(2e+5/(lag[npeak]-lag[npeak-1]))
                left, right = max(npeak-N,0), min(npeak+N, len(corr))
                nrange = int(min(npeak-left, right-npeak))
                corr_left, corr_right = corr[npeak-nrange:npeak], np.flip(corr[npeak:npeak+nrange])
                sigma = np.sum(((corr_left - corr_right)**2))/nrange
                r = peak/(np.sqrt(sigma))
                # measure fwhm and estimate the error((3/8)*(w/1+r))
                fwhm = 2*np.sqrt(-h/(2*fit_parabol.c2))
                gaussian = models.Gaussian1D(amplitude=h, mean=fitted_center, stddev=fwhm/(2*np.sqrt(2*np.log(2))))
                fit_gaussian = fit(gaussian, lag_fit, corr_fit)
                fwhm = 2*np.sqrt(2*np.log(2))*fit_gaussian.stddev.value
                error_vel = (3/8)*fwhm/(1+r)
                # convert the error of the velocity to the redshift
                error = error_vel/c
                dispersion = np.sqrt(fit_gaussian.stddev.value**2-2*template_dispersion[0]**2)
                dispersion_err = np.sqrt(2*np.sqrt(fit_gaussian.stddev.value**2*error_vel**2
                                                   +4*template_dispersion[0]**2*template_dispersion[1]**2
                                                   +dispersion**4)-2*dispersion**2)
                result = 'Well fitted'
            
    return z, r, error, dispersion, dispersion_err, fit_parabol, fit_gaussian, lag_fit, npeak, nrange, result


class rvm:
    def __init__(self, spectrum, templates, hcutoff_scale=10, apodization_window = 0.05, spectrum_range=None, 
        template_range=None, 
        correlation_range=[-0.01,2], mask = None, continuum_subtraction = True, 
        window = None, sigma =3):
        
        self.spectrum, self.templates = spectrum, templates
        self.hcutoff_scale, self.apdoization_window = hcutoff_scale, apodization_window
        self.spectrum_range, self.template_range, self.correlation_range = spectrum_range, template_range, correlation_range
        self.mask = mask
        self.window, self.sigma = window, sigma
        
        if continuum_subtraction:
            if self.window == None:
                self.window = int(np.nanmedian(self.spectrum[0,1:]-self.spectrum[0,:-1])*35/1.4)
            self.subt_spectrum = continuum.continuum_subtraction(self.spectrum, window=self.window, sigma=self.sigma)
        else:
            self.subt_spectrum = copy.deepcopy(self.spectrum)
            
        if self.spectrum_range != None:
            self.subt_spectrum = self.subt_spectrum[:,(self.subt_spectrum[0,:]>self.spectrum_range[0])&((self.subt_spectrum[0,:]<self.spectrum_range[1]))]
            
    def redshift(self):
        template_list = list(self.templates.keys())
        n_divide = len(template_list)//4
        parallel_section = [template_list[:n_divide],template_list[n_divide:2*n_divide],
                            template_list[2*n_divide:3*n_divide], template_list[3*n_divide:]]
        
        def _redshift(temp_names):
            n_template = len(temp_names)
            template_name = np.zeros(n_template, dtype='<U32')
            z = np.zeros(n_template)
            r = np.zeros(n_template)
            error = np.zeros(n_template)
            for i, temp_name in enumerate(temp_names):
                template_name[i], temp = temp_name, copy.deepcopy(self.templates[temp_name])
                if self.template_range != None:
                    temp = temp[:,(temp[0,:]>self.template_range[0])&((temp[0,:]<self.template_range[1]))]
                lag, corr, _ = correlation.template_correlate(self.subt_spectrum, temp[0], template_type=temp[2],
                                                                              hcutoff_scale=self.hcutoff_scale,
                                                                              apodization_window = self.apdoization_window,
                                                                              mask = self.mask)
                z[i], r[i], error[i],_,_,_,_,_,_,_,_ = z_finding(corr, lag, pkfrac = temp[1], template_dispersion=temp[3],
                                                                  correlation_range=self.correlation_range)
            return template_name, z, r, error
        
        result = np.hstack(Parallel(n_jobs=4, verbose=0)(delayed(_redshift)(temp_names) for temp_names in parallel_section))
        template_name, z, r, error = result[0], result[1].astype(float), result[2].astype(float), result[3].astype(float)
        # eliminate np.nan in r and error
        nan = np.ma.masked_invalid(r).mask
        template_name, z, r, error = template_name[~nan], z[~nan], r[~nan], error[~nan]
        nan = np.ma.masked_invalid(error).mask
        template_name, z, r, error = template_name[~nan], z[~nan], r[~nan], error[~nan]
        # arange the value in the order of R
        order = np.flip(np.argsort(r))
        self.template_name = template_name[order]
        self.z = z[order]
        self.r = r[order]
        self.error = error[order]
    
        if len(self.template_name) < 5:
            top = len(self.template_name)
        else:
            top = 5
        table = np.vstack((self.template_name[0:top], self.z[0:top], self.r[0:top], self.error[0:top]))
        column_names = ['Template_name', 'Redshift', 'r-value', 'Error']
        df = pd.DataFrame(table.T, columns = column_names)
        df = df.astype({'Template_name':str, 'Redshift':float, 'r-value':float, 'Error':float})
        return df
    
    def cc(self, index, plotting=False):
        if type(index) == int:
            temp = copy.deepcopy(self.templates[self.template_name[index]])
        elif type(index) == str:
            temp = copy.deepcopy(self.templates[index])
        if self.template_range != None:
            temp = temp[:,(temp[0,:]>self.template_range[0])&((temp[0,:]<self.template_range[1]))]
        
        lag, corr, resampled_spectrum =  correlation.template_correlate(self.subt_spectrum, temp[0], template_type=temp[2], hcutoff_scale=self.hcutoff_scale,
                                              apodization_window = self.apdoization_window, mask = self.mask)
        z, _, _, _,_,self.parabol_fit,self.gaussian_fit, self.lag_fit, self.npeak, self.nrange, result =\
            z_finding(corr, lag, pkfrac = temp[1], template_dispersion=temp[3], correlation_range=self.correlation_range)
        
        if plotting:
            x = np.linspace(z*c-3000, z*c+3000,1000)
            f, ax = plt.subplots(1,3, figsize=(18,5))
            ax[0].plot(resampled_spectrum[0,:], resampled_spectrum[1,:],'k-', label='Resampled spec')
            ax[0].set_xlabel(r'Wavelength ($\AA$)')
            ax[0].set_ylabel('Flux')
            
            ax[1].plot(lag, corr,'k-',label='CC sig')
            ax[1].set_xlabel('Lags (km/s)')
            ax[1].set_ylabel('CC signal')
            ax[1].set_ylim(-0.2,1.1*np.max(corr))
            
            ax[2].plot(lag, corr, 'ko-', label='CC sig')
            ax[2].plot([z*c,z*c],[-1,5],'b--')
            ax[2].set_xlabel('Lags (km/s)')
            ax[2].set_ylabel('CC signal')
            ax[2].set_xlim(x[0],x[-1])
            ax[2].set_ylim(-0.2,1.1*np.max(corr))
            
            if self.parabol_fit != np.nan:
                ax[1].plot(x, self.parabol_fit(x), 'b-', label='Parabol')
                ax[2].plot(x, self.parabol_fit(x), 'b-', label='Parabol')
                
                ax[1].fill_between([lag[self.npeak-self.nrange],lag[self.npeak+self.nrange]], [-1,-1],[5,5], color='b',
                               alpha=0.2,ec='none', label='Range for calculating r-value')
                ax[2].fill_between([self.lag_fit[0],self.lag_fit[-1]],[-1,-1],[5,5], color='r',
                               alpha=0.2,ec='none', label='Fitting range')
                
            if self.gaussian_fit != np.nan:
                ax[1].plot(x, self.gaussian_fit(x), 'r-', label='Gaussian')
                ax[2].plot(x, self.gaussian_fit(x), 'r-', label='Gaussian')
            
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            
            ax[1].set_title('Result: '+result)
            
        return lag, corr, resampled_spectrum
    
    def vp_best(self):
        if len(self.z)==0:
            self.dispersion, self.dispersion_error = np.nan, np.nan
        else:
            self.rest_spectrum = copy.deepcopy(self.subt_spectrum)
            self.rest_spectrum[0,:] = self.subt_spectrum[0,:]/(1+self.z[0])
            temp = copy.deepcopy(self.templates[self.template_name[0]])
            if self.template_range != None:
                temp = temp[:,(temp[0,:]>self.template_range[0])&((temp[0,:]<self.template_range[1]))]
            lag, corr, observed_spectrum = correlation.template_correlate(self.rest_spectrum, temp[0], template_type=temp[2],
                                                                            hcutoff_scale=self.hcutoff_scale,
                                                                            apodization_window = self.apdoization_window,
                                                                            mask = self.mask)
            _,_,_, self.dispersion, self.dispersion_error,_,_,_,_,_,_ = z_finding(corr, lag, pkfrac = temp[1], template_dispersion=temp[3],
                                                                correlation_range=self.correlation_range)
        return self.dispersion, self.dispersion_error
    
    def vp_all(self):
        self.rest_spectrum = copy.deepcopy(self.subt_spectrum)
        self.rest_spectrum[0,:] = self.subt_spectrum[0,:]/(1+self.z[0])
        
        n_divide = len(self.template_name)//4
        parallel_section = [self.template_name[:n_divide],self.template_name[n_divide:2*n_divide],
                            self.template_name[2*n_divide:3*n_divide], self.template_name[3*n_divide:]]
        
        if len(self.z) == 0:
            dispersion, dispersion_error = np.zeros(n_divide)*np.nan, np.zeros(n_divide)*np.nan
        else:
            def _vp_all(temp_names):
                n_template = len(self.template_name)
                dispersion = np.zeros(n_template)
                dispersion_error = np.zeros(n_template)
                error = np.zeros(n_template)
                for i, temp_name in enumerate(self.template_name):
                    temp = copy.deepcopy(self.templates[temp_name])
                    if self.template_range != None:
                        temp = temp[:,(temp[0,:]>self.template_range[0])&((temp[0,:]<self.template_range[1]))]
                    lag, corr, observed_spectrum = correlation.template_correlate(self.rest_spectrum, temp[0], template_type=temp[2],
                                                                                hcutoff_scale=self.hcutoff_scale,
                                                                                apodization_window = self.apdoization_window,
                                                                                mask = self.mask)
                    _, _, _,dispersion[i],dispersion_error[i],_,_,_,_,_,_ = z_finding(corr, lag, pkfrac = temp[1], template_dispersion=temp[3],
                                                                    correlation_range=self.correlation_range)
                return dispersion, dispersion_error
        
        
            result = np.array(Parallel(n_jobs=4, verbose=0)(delayed(_vp_all)(temp_names) for temp_names in parallel_section))
            self.dispersion, self.dispersion_error = np.concatenate(result[:,0,:]), np.concatenate(result[:,1,:]).astype(float)
    
        if len(self.template_name) < 5:
            top = len(self.template_name)
        else:
            top = 5
        table = np.vstack((self.template_name[0:top], self.z[0:top], self.r[0:top], self.error[0:top],
                           self.dispersion[0:top], 
                           self.dispersion_error[0:top]))
        column_names = ['Template_name', 'Redshift', 'r-value', 'Error', 'Dispersion', 'Dispersion_error']
        df = pd.DataFrame(table.T, columns = column_names)
        df = df.astype({'Template_name':str, 'Redshift':float, 'r-value':float, 'Error':float,
                        'Dispersion':float, 'Dispersion_error':float})
        return df
    
    def cc_vp(self, index, plotting=False):
        if type(index) == int:
            temp = copy.deepcopy(self.templates[self.template_name[index]])
        elif type(index) == str:
            temp = copy.deepcopy(self.templates[index])
        if self.template_range != None:
            temp = temp[:,(temp[0,:]>self.template_range[0])&((temp[0,:]<self.template_range[1]))]
            
        lag, corr, resampled_spectrum = correlation.template_correlate(self.rest_spectrum, temp[0], template_type=temp[2],
                                                                            hcutoff_scale=self.hcutoff_scale,
                                                                            apodization_window = self.apdoization_window,
                                                                            mask = self.mask)
        z, _, _, _,_,self.parabol_fit_vp,self.gaussian_fit_vp, self.lag_fit_vp, self.npeak_vp, self.nrange_vp, result =\
            z_finding(corr, lag, pkfrac = temp[1], template_dispersion=temp[3], correlation_range=self.correlation_range)
        
        if plotting:
            x = np.linspace(z*c-3000, z*c+3000,1000)
            f, ax = plt.subplots(1,3, figsize=(18,5))
            ax[0].plot(resampled_spectrum[0,:],resampled_spectrum[1,:],'k-', label='Resampled spec')
            ax[0].set_xlabel(r'Wavelength ($\AA$)')
            ax[0].set_ylabel('Flux')
            
            ax[1].plot(lag, corr,'k-',label='CC sig')
            ax[1].set_xlabel('Lags (km/s)')
            ax[1].set_ylabel('CC signal')
            ax[1].set_ylim(-0.2,1.1*np.max(corr))
            
            ax[2].plot(lag, corr, 'ko-', label='CC sig')
            ax[2].plot([z*c,z*c],[-1,5],'b--')
            ax[2].set_xlabel('Lags (km/s)')
            ax[2].set_ylabel('CC signal')
            ax[2].set_xlim(x[0],x[-1])
            ax[2].set_ylim(-0.2,1.1*np.max(corr))
            
            if self.parabol_fit_vp != np.nan:
                ax[1].plot(x, self.parabol_fit_vp(x), 'b-', label='Parabol')
                ax[2].plot(x, self.parabol_fit_vp(x), 'b-', label='Parabol')
                
                ax[1].fill_between([lag[self.npeak_vp-self.nrange_vp],lag[self.npeak_vp+self.nrange_vp]], [-1,-1],[5,5], color='b',
                               alpha=0.2,ec='none', label='Range for calculating r-value')
                ax[2].fill_between([self.lag_fit_vp[0],self.lag_fit_vp[-1]],[-1,-1],[5,5], color='r',
                               alpha=0.2,ec='none', label='Fitting range')
                
            if self.gaussian_fit_vp != np.nan:
                ax[1].plot(x, self.gaussian_fit_vp(x), 'r-', label='Gaussian')
                ax[2].plot(x, self.gaussian_fit_vp(x), 'r-', label='Gaussian')
            
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            
            ax[1].set_title('Result: '+result)
            
        return lag, corr, resampled_spectrum
    
    
        
def multi_rvm(spectrums, template, vel_disp=False, progress=False, **kwargs):
    if progress:
        pro = tqdm
    elif progress == 'notebook':
        pro = notebook.tqdm
    else:
        def pro(x):
            return x
    
    if vel_disp:
        besttemp, Z, R, Error, Dispersion, Dispersion_err = np.zeros(len(spectrums), dtype='<U32'), np.zeros(len(spectrums)),\
            np.zeros(len(spectrums)), np.zeros(len(spectrums)), np.zeros(len(spectrums)), np.zeros(len(spectrums))
    
        for i, spectrum in enumerate(pro(spectrums)):
            RVM = rvm(spectrum, template, **kwargs)
            z_result = RVM.redshift()
            template_name = np.array(z_result['Template_name'])
            redshift = np.array(z_result['Redshift'], dtype=float)
            r = np.array(z_result['r-value'], dtype=float)
            error = np.array(z_result['Error'], dtype=float)
            dispersion, dispersion_err = RVM.vp_best()
            if len(redshift) >= 1:
                besttemp[i] = template_name[0]
                Z[i] = redshift[0]
                R[i] = r[0]
                Error[i] = float(error[0])
                Dispersion[i] = dispersion
                Dispersion_err[i] = dispersion_err
            else:
                besttemp[i] = np.nan
                Z[i] = np.nan
                R[i] = np.nan
                Error[i] = np.nan
                Dispersion[i] = np.nan
                Dispersion_err[i] = np.nan

            if Error[i] < 1e-5:
                Z[i] = np.round(Z[i], 6)
            else:
                Z[i] = np.round(Z[i], 8)
            
            Error[i] = np.round(Error[i], 7)
        
        table = np.vstack((besttemp, Z, R, Error, Dispersion, Dispersion_err))
        column_names = ['Best_template', 'Redshift', 'r-value', 'Error', 'Dispersion', 'Dispersion_error']
        df = pd.DataFrame(table.T, columns = column_names)
        df = df.astype({'Best_template':str, 'Redshift':float, 'r-value':float, 'Error':float,
                        'Dispersion':float, 'Dispersion_error':float})
    
    else: 
        besttemp, Z, R, Error = np.zeros(len(spectrums), dtype='<U32'), np.zeros(len(spectrums)), np.zeros(len(spectrums)),\
            np.zeros(len(spectrums))
        
        for i, spectrum in enumerate(pro(spectrums)):
            RVM = rvm(spectrum, template, **kwargs)
            z_result = RVM.redshift()
            template_name = np.array(z_result['Template_name'])
            redshift = np.array(z_result['Redshift'], dtype=float)
            r = np.array(z_result['r-value'], dtype=float)
            error = np.array(z_result['Error'], dtype=float)
            if len(redshift) >= 1:
                besttemp[i] = template_name[0]
                Z[i] = redshift[0]
                R[i] = r[0]
                Error[i] = float(error[0])
            else:
                besttemp[i] = np.nan
                Z[i] = np.nan
                R[i] = np.nan
                Error[i] = np.nan

            if Error[i] < 1e-5:
                Z[i] = np.round(Z[i], 6)
            else:
                Z[i] = np.round(Z[i], 8)
            
            Error[i] = np.round(Error[i], 7)
        
        table = np.vstack((besttemp, Z, R, Error))
        column_names = ['Best_template', 'Redshift', 'r-value', 'Error']
        df = pd.DataFrame(table.T, columns = column_names)
        df = df.astype({'Best_template':str, 'Redshift':float, 'r-value':float, 'Error':float})
    
    return df