import sys
sys.path.insert(0,'../')
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

def z_finding(corr, lag, pkfrac=0.65, template_dispersion = [0,0], correlation_range=[-0.01,2]):
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
    
    try:
        i_peak = np.nanargmax(_corr)
        peak = _corr[i_peak] # estimates a peak
        lag_peak = _lag[i_peak]
        corr_fit, lag_fit = corr[corr >= pkfrac*peak], lag[corr >= pkfrac*peak] # select points >pkfrac*peak
                
        if np.nanmax(lag_fit[1:]-lag_fit[:-1]) < 500:
            center = lag[np.nanargmax(corr_fit)]
            c2 = (peak - peak*pkfrac)/(center-lag_fit[0])**2
            c1= 2*c2*center
            c0 = peak + (c1**2)/(4*c2)
            # fit with the parabola
            parabol = models.Polynomial1D(2, c2=c2, c1=c1, c0=c0)
            fit = fitting.LevMarLSQFitter()
            fit_parabol = fit(parabol, lag_fit, corr_fit)
            fitted_center = -fit_parabol.c1/(2*fit_parabol.c2) # find the center
            if fit_parabol.c2 > 0 or fitted_center<correlation_range[0] or fitted_center>correlation_range[1]:
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
                try:
                    fit_gaussian = fit(gaussian, lag_fit, corr_fit)
                except:
                    print(h, fitted_center, fwhm)
                    fit_gaussian = fit(gaussian, lag_fit, corr_fit)
                fwhm = 2*np.sqrt(2*np.log(2))*fit_gaussian.stddev.value
                error_vel = (3/8)*fwhm/(1+r)
                # convert the error of the velocity to the redshift
                error = error_vel/c
                dispersion = np.sqrt(fit_gaussian.stddev.value**2-template_dispersion[0]**2)
                dispersion_err = np.sqrt(2*np.sqrt(fit_gaussian.stddev.value**2*error_vel**2
                                                   +template_dispersion[0]**2*template_dispersion[1]**2
                                                   +0.25*dispersion**4)-dispersion**2)
                result = 'Well fitted'
                
        else:
            fit_condition = (corr > pkfrac*peak) & (np.abs(lag-lag_peak)<500)
            corr_fit, lag_fit = corr[fit_condition], lag[fit_condition]
            center = lag[np.nanargmax(corr_fit)]
            ####### This must be revsied to be more efficiently
            c2 = (peak - peak*pkfrac)/(center-lag_fit[0])**2
            c1= 2*c2*center
            c0 = peak + (c1**2)/(4*c2)
            # fit with the parabola
            parabol = models.Polynomial1D(2, c2=c2, c1=c1, c0=c0)
            fit = fitting.LevMarLSQFitter()
            fit_parabol = fit(parabol, lag_fit, corr_fit)
            fitted_center = -fit_parabol.c1/(2*fit_parabol.c2) # find the center
            if fit_parabol.c2 > 0 or fitted_center<correlation_range[0] or fitted_center>correlation_range[1]:
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
                try:
                    fit_gaussian = fit(gaussian, lag_fit, corr_fit)
                except:
                    print(h, fitted_center, fwhm)
                    fit_gaussian = fit(gaussian, lag_fit, corr_fit)
                fwhm = 2*np.sqrt(2*np.log(2))*fit_gaussian.stddev.value
                error_vel = (3/8)*fwhm/(1+r)
                # convert the error of the velocity to the redshift
                error = error_vel/c
                dispersion = np.sqrt(fit_gaussian.stddev.value**2-template_dispersion[0]**2)
                dispersion_err = np.sqrt(2*np.sqrt(fit_gaussian.stddev.value**2*error_vel**2
                                                   +template_dispersion[0]**2*template_dispersion[1]**2
                                                   +0.25*dispersion**4)-dispersion**2)
                result = 'Separated peaks'
    except:
        z, r, error, dispersion, dispersion_err, fit_parabol, fit_gaussian, lag_fit, npeak, nrange, result  = \
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'Fitting failed'
            
            
    return z, r, error, dispersion, dispersion_err, fit_parabol, fit_gaussian, lag_fit, npeak, nrange, result


class rvm:
    def __init__(self, spectrum, templates, star_templates=None, clipping=False, hcutoff_scale=2, apodization_window = 0.05, 
                 spectrum_range=None, template_range=None, correlation_range=[-0.01,2], mask = None,
                 continuum_subtraction = True, window = 80, sigma =3): 
        
        self.spectrum, self.templates, self.star_templates = spectrum, templates, star_templates
        self.clipping, self.hcutoff_scale, self.apdoization_window = clipping, hcutoff_scale, apodization_window
        self.spectrum_range, self.template_range, self.correlation_range = spectrum_range, template_range, correlation_range
        self.mask = mask
        self.window, self.sigma = window, sigma
        
        if continuum_subtraction:
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
            flag = np.zeros(n_template).astype(int)
            for i, temp_name in enumerate(temp_names):
                template_name[i], temp = temp_name, copy.deepcopy(self.templates[temp_name])
                if self.template_range != None:
                    temp = temp[:,(temp[0,:]>self.template_range[0])&((temp[0,:]<self.template_range[1]))]
                lag, corr, _ = correlation.template_correlate(self.subt_spectrum, temp[0], template_type=temp[2], 
                                                              clipping=self.clipping, hcutoff_scale=self.hcutoff_scale, 
                                                              apodization_window = self.apdoization_window, mask = self.mask)
                z[i], r[i], error[i],_,_,_,_,_,_,_,result = z_finding(corr, lag, pkfrac = temp[1], template_dispersion=temp[3],
                                                                  correlation_range=self.correlation_range)
                if result == 'Well fitted':
                    flag[i] = 0
                elif result == 'Separated peaks':
                    flag[i] = 1
                else:
                    flag[i] = 99
                    
            return template_name, z, r, error, flag
        
        result = np.hstack(Parallel(n_jobs=4, verbose=0)(delayed(_redshift)(temp_names) for temp_names in parallel_section))
        template_name, z, r, error, flag = result[0], result[1].astype(float), result[2].astype(float), result[3].astype(float), result[4].astype(int)
        # eliminate np.nan in r and error
        nan = np.ma.masked_invalid(r).mask
        template_name, z, r, error, flag = template_name[~nan], z[~nan], r[~nan], error[~nan], flag[~nan]
        nan = np.ma.masked_invalid(error).mask
        template_name, z, r, error, flag = template_name[~nan], z[~nan], r[~nan], error[~nan], flag[~nan]
        # arange the value in the order of R
        order = np.flip(np.argsort(r))
        self.template_name = template_name[order]
        self.z = z[order]
        self.r = r[order]
        self.error = error[order]
        self.flag = flag[order]
    
        table = np.vstack((self.template_name, self.z, self.r, self.error, self.flag))
        column_names = ['Template_name', 'Redshift', 'r-value', 'Error', 'Flag']
        df = pd.DataFrame(table.T, columns = column_names)
        df = df.astype({'Template_name':str, 'Redshift':float, 'r-value':float, 'Error':float, 'Flag':int})
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
            if np.isfinite(z):
                zc = z*c
            else:
                zc = lag[np.argmax(corr)]
            x = np.linspace(zc-100000, zc+100000,1000)
            f, ax = plt.subplots(1,3, figsize=(30,10))
            ax[0].plot(resampled_spectrum[0,:], resampled_spectrum[1,:],'k-')
            ax[0].set_xlabel(r'Wavelength ($\AA$)', fontsize=25)
            ax[0].set_ylabel('Flux', fontsize=25)
            ax[0].tick_params(axis='both', labelsize=25)
            ax[0].set_title('Resampled spectrum', fontsize=25)
            
            ax[1].plot(lag, corr,'k-')
            ax[1].plot([lag[0]-10,lag[-1]+10],[temp[1]*np.nanmax(corr), temp[1]*np.nanmax(corr)], 'k:')
            ax[1].set_xlabel('Lags (km/s)', fontsize=25)
            ax[1].set_ylabel('CC signal', fontsize=25)
            ax[1].set_xlim(lag[0],lag[-1])
            ax[1].set_ylim(-0.2,1.1*np.max(corr))
            ax[1].tick_params(axis='both', labelsize=25)
            ax[1].set_title('Cross-correlation', fontsize=25)
            
            ax[2].plot(lag, corr, 'ko-', label='CC sig')
            ax[2].plot([x[0]-10,x[-1]+10],[temp[1]*np.nanmax(corr), temp[1]*np.nanmax(corr)], 'k:')
            ax[2].plot([zc,zc],[-1,5],'b--')
            ax[2].set_xlabel('Lags (km/s)', fontsize=25)
            ax[2].set_ylabel('CC signal', fontsize=25)
            ax[2].set_xlim(zc-3000,zc+3000)
            ax[2].set_ylim(-0.2,1.1*np.max(corr))
            ax[2].tick_params(axis='both', labelsize=25)
            ax[2].set_title('Result: '+result, fontsize=25)
            
            if self.parabol_fit != np.nan:
                ax[1].plot(x, self.parabol_fit(x), 'b-')
                ax[2].plot(x, self.parabol_fit(x), 'b-', label='Parabol')
                
                ax[1].fill_between([lag[self.npeak-self.nrange],lag[self.npeak+self.nrange-1]], [-1,-1],[5,5], color='b',
                               alpha=0.2,ec='none', label='Range for calculating r-value')
                ax[2].fill_between([self.lag_fit[0],self.lag_fit[-1]],[-1,-1],[5,5], color='r',
                               alpha=0.2,ec='none', label='Fitting range')
                
            if self.gaussian_fit != np.nan:
                ax[1].plot(x, self.gaussian_fit(x), 'r-')
                ax[2].plot(x, self.gaussian_fit(x), 'r-', label='Gaussian')
            
            ax[1].legend(fontsize=25)
            ax[2].legend(fontsize=25)
            
            f.tight_layout()
            
        return lag, corr, resampled_spectrum
    
    def vd(self):
        self.rest_spectrum = copy.deepcopy(self.subt_spectrum)
        
        template_list = list(self.star_templates.keys())
        n_divide = len(template_list)//4
        parallel_section = [template_list[:n_divide],template_list[n_divide:2*n_divide],
                            template_list[2*n_divide:3*n_divide], template_list[3*n_divide:]]
        
        if len(self.z) == 0:
            template_name, r, dispersion, dispersion_error, flag = np.zeros(n_divide).astype(str), np.zeros(n_divide)*np.nan, np.zeros(n_divide)*np.nan, np.zeros(n_divide)*np.nan, 99*np.ones(n_divide).astype(int)
        else:
            self.rest_spectrum[0,:] = self.subt_spectrum[0,:]/(1+self.z[0])
            def _vd(temp_names):
                n_template = len(temp_names)
                template_name = np.zeros(n_template, dtype='<U32')
                r = np.zeros(n_template).astype(float)
                dispersion = np.zeros(n_template)
                dispersion_error = np.zeros(n_template)
                flag = np.zeros(n_template).astype(int)
                for i, temp_name in enumerate(temp_names):
                    template_name[i], temp = temp_name, copy.deepcopy(self.star_templates[temp_name])
                    if self.template_range != None:
                        temp = temp[:,(temp[0,:]>self.template_range[0])&((temp[0,:]<self.template_range[1]))]
                    lag, corr, observed_spectrum = correlation.template_correlate(self.rest_spectrum, temp[0], template_type=temp[2],
                                                                                hcutoff_scale=0,
                                                                                apodization_window = self.apdoization_window,
                                                                                mask = self.mask)
                    _, r[i], _,dispersion[i],dispersion_error[i],_,_,_,_,_,result = z_finding(corr, lag, pkfrac = temp[1], template_dispersion=temp[3],
                                                                    correlation_range=self.correlation_range)
                    
                if result == 'Well fitted':
                    flag[i] = 0
                elif result == 'Separated peaks':
                    flag[i] = 1
                else:
                    flag[i] = 99
                    
                return template_name, r, dispersion, dispersion_error, flag
        
        
            result = np.hstack(Parallel(n_jobs=4, verbose=0)(delayed(_vd)(temp_names) for temp_names in parallel_section))
            star_template_name, r_disp, dispersion , dispersion_error, flag_disp = result[0], result[1].astype(float), result[2].astype(float), result[3].astype(float), result[4].astype(int)
            
        # eliminate np.nan in r and error
        nan = np.ma.masked_invalid(r_disp).mask
        star_template_name, r_disp, dispersion, dispersion_error, flag_disp = star_template_name[~nan], r_disp[~nan], dispersion[~nan], dispersion_error[~nan], flag_disp[~nan]
        nan = np.ma.masked_invalid(dispersion_error).mask
        star_template_name, r_disp, dispersion, dispersion_error, flag_disp = star_template_name[~nan], r_disp[~nan], dispersion[~nan], dispersion_error[~nan], flag_disp[~nan]
        # arange the value in the order of R
        order = np.flip(np.argsort(r_disp))
        self.star_template_name = star_template_name[order]
        self.r_disp = r_disp[order]
        self.dispersion = dispersion[order]
        self.dispersion_error = dispersion_error[order]
        self.flag_disp = flag_disp[order]
    
        table = np.vstack((self.star_template_name, self.r_disp, self.dispersion, self.dispersion_error, self.flag_disp))
        column_names = ['Template_name', 'r-value', 'Dispersion', 'Dispersion_error', 'Flag']
        df = pd.DataFrame(table.T, columns = column_names)
        df = df.astype({'Template_name':str, 'r-value':float, 'Dispersion':float, 'Dispersion_error':float, 'Flag':int})
        return df
    
    def cc_vd(self, index, plotting=False):
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
        z, _, _, _,_,self.parabol_fit_vd,self.gaussian_fit_vd, self.lag_fit_vd, self.npeak_vd, self.nrange_vd, result =\
            z_finding(corr, lag, pkfrac = temp[1], template_dispersion=temp[3], correlation_range=self.correlation_range)
        
        if plotting:
            if np.isfinite(z):
                zc = z*c
            else:
                zc = lag[np.argmax(corr)]
            x = np.linspace(zc-100000, zc+100000,1000)
            f, ax = plt.subplots(1,3, figsize=(30,10))
            ax[0].plot(resampled_spectrum[0,:],resampled_spectrum[1,:],'k-')
            ax[0].set_xlabel(r'Wavelength ($\AA$)', fontsize=25)
            ax[0].set_ylabel('Flux', fontsize=25)
            ax[0].tick_params(axis='both', labelsize=25)
            ax[0].set_title('Resample spectrum', fontsize=25)
            
            ax[1].plot(lag, corr,'k-')
            ax[1].plot([lag[0]-10,lag[-1]+10],[temp[1]*np.max(corr), temp[1]*np.max(corr)], 'k:')
            ax[1].set_xlabel('Lags (km/s)', fontsize=25)
            ax[1].set_ylabel('CC signal',fontsize=25)
            ax[1].set_ylim(-0.2,1.1*np.max(corr))
            ax[1].tick_params(axis='both', labelsize=25)
            ax[1].set_title('Cross-correlation', fontsize=25)
            
            ax[2].plot(lag, corr, 'ko-', label='CC sig')
            ax[2].plot([zc,zc],[-1,5],'b--')
            ax[2].plot([x[0]-10,x[-1]+10],[temp[1]*np.nanmax(corr), temp[1]*np.nanmax(corr)], 'k:')
            ax[2].set_xlabel('Lags (km/s)', fontsize=25)
            ax[2].set_ylabel('CC signal', fontsize=25)
            ax[2].set_xlim(zc-3000,zc+3000)
            ax[2].set_ylim(-0.2,1.1*np.max(corr))
            ax[2].tick_params(axis='both', labelsize=25)
            ax[2].set_title('Result: '+result, fontsize=25)
            
            if self.parabol_fit_vd != np.nan:
                ax[1].plot(x, self.parabol_fit_vd(x), 'b-')
                ax[2].plot(x, self.parabol_fit_vd(x), 'b-', label='Parabol')
                
                ax[1].fill_between([lag[self.npeak_vd-self.nrange_vd],lag[self.npeak_vd+self.nrange_vd-1]], [-1,-1],[5,5], color='b',
                               alpha=0.2,ec='none', label='Range for calculating r-value')
                ax[2].fill_between([self.lag_fit_vd[0],self.lag_fit_vd[-1]],[-1,-1],[5,5], color='r',
                               alpha=0.2,ec='none', label='Fitting range')
                
            if self.gaussian_fit_vd != np.nan:
                ax[1].plot(x, self.gaussian_fit_vd(x), 'r-')
                ax[2].plot(x, self.gaussian_fit_vd(x), 'r-', label='Gaussian')
            
            ax[1].legend(fontsize=25)
            ax[2].legend(fontsize=25)
            
            f.tight_layout()
            
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
        besttemp, Z, R, Error, Flag, besttemp_vd, Dispersion, R_vd, Dispersion_error, Flag_vd = np.zeros(len(spectrums), dtype='<U32'), np.zeros(len(spectrums)),\
        np.zeros(len(spectrums)), np.zeros(len(spectrums)), np.zeros(len(spectrums)).astype(int), \
        np.zeros(len(spectrums), dtype='<U32'), np.zeros(len(spectrums)), np.zeros(len(spectrums)), np.zeros(len(spectrums)), np.zeros(len(spectrums)).astype(int)
    
        for i, spectrum in enumerate(pro(spectrums)):
            RVM = rvm(spectrum, template, **kwargs)
            z_result = RVM.redshift()
            template_name = np.array(z_result['Template_name'])
            redshift = np.array(z_result['Redshift'], dtype=float)
            r = np.array(z_result['r-value'], dtype=float)
            error = np.array(z_result['Error'], dtype=float)
            flag = np.array(z_result['Flag'], dtype=str)
            
            vd_result = RVM.vd()
            star_template_name = np.array(vd_result['Template_name'])
            dispersion = np.array(vd_result['Dispersion'], dtype=float)
            r_vd = np.array(vd_result['r-value'], dtype=float)
            dispersion_error = np.array(vd_result['Dispersion_error'], dtype=float)
            flag_vd = np.array(vd_result['Flag'], dtype=str)
            if len(redshift) >= 1:
                besttemp[i] = template_name[0]
                Z[i] = np.round(redshift[0], 7)
                R[i] = r[0]
                Error[i] = np.round(float(error[0]), 7)
                Flag[i] = flag[0]
                
                besttemp_vd[i] = star_template_name[0]
                Dispersion[i] = np.round(dispersion[0],0)
                R_vd[i] = r_vd[0]
                Dispersion_error[i] = np.round(dispersion_error[0],0)
                Flag_vd[i] = flag_vd[0]
                
            else:
                besttemp[i] = np.nan
                Z[i] = np.nan
                R[i] = np.nan
                Error[i] = np.nan
                Flag[i] = 99
                
                besttemp_vd[i] = np.nan
                Dispersion[i] = np.nan
                R_vd[i] = np.nan
                Dispersion_error[i] = np.nan
                Flag_vd[i] = np.nan
            
        
        table = np.vstack((besttemp, Z, R, Error, Flag, besttemp_vd, Dispersion, R_vd, Dispersion_error, Flag_vd))
        column_names = ['Best_template', 'Redshift', 'r-value', 'Error', 'Flag', 'Best_template_disp', 'Dispersion', 'r-value_disp', 'Dispersion_error', 'Flag_disp']
        df = pd.DataFrame(table.T, columns = column_names)
        df = df.astype({'Best_template':str, 'Redshift':float, 'r-value':float, 'Error':float, 'Flag':int,
                        'Best_template_disp':str, 'Dispersion':float, 'r-value_disp':float, 'Dispersion_error':float,
                        'Flag_disp':int})
    
    else: 
        besttemp, Z, R, Error, Flag = np.zeros(len(spectrums), dtype='<U32'), np.zeros(len(spectrums)), np.zeros(len(spectrums)),\
            np.zeros(len(spectrums)), np.zeros(len(spectrums)).astype(int)
        
        for i, spectrum in enumerate(pro(spectrums)):
            RVM = rvm(spectrum, template, **kwargs)
            z_result = RVM.redshift()
            template_name = np.array(z_result['Template_name'])
            redshift = np.array(z_result['Redshift'], dtype=float)
            r = np.array(z_result['r-value'], dtype=float)
            error = np.array(z_result['Error'], dtype=float)
            flag = np.array(z_result['Flag'], dtype=float)
            if len(redshift) >= 1:
                besttemp[i] = template_name[0]
                Z[i] = redshift[0]
                R[i] = r[0]
                Error[i] = float(error[0])
                Flag[i] = flag[0]
            else:
                besttemp[i] = np.nan
                Z[i] = np.nan
                R[i] = np.nan
                Error[i] = np.nan
                Flag[i] = 3

            if Error[i] < 1e-5:
                Z[i] = np.round(Z[i], 6)
            else:
                Z[i] = np.round(Z[i], 8)
            
            Error[i] = np.round(Error[i], 7)
        
        table = np.vstack((besttemp, Z, R, Error, Flag))
        column_names = ['Best_template', 'Redshift', 'r-value', 'Error', 'Flag']
        df = pd.DataFrame(table.T, columns = column_names)
        df = df.astype({'Best_template':str, 'Redshift':float, 'r-value':float, 'Error':float, 'Flag':int})
    
    return df