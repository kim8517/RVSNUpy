import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sigmaclip
import copy
from scipy.interpolate import splrep, splev
from scipy.signal.windows import tukey

class continuum:
    def __init__(self, wavelengths, fluxes, weights, knots_bin = 100, thres=3, apodization_size = 0.05, plotting=False):
        self.trace_continuum(wavelengths, fluxes, weights, knots_bin, thres)
        self.normalized_fluxes = ((fluxes/self.continuum_fluxes)-1)*tukey(len(wavelengths), apodization_size)
        # self.normalized_fluxes[np.isnan(self.normalized_fluxes)] = 0
        if plotting:
            fig, ax = plt.subplots(2,1)
            ax[0].set_title('Spectrum')
            ax[0].plot(wavelengths, fluxes, 'k-')
            ax[0].plot(wavelengths, self.continuum_fluxes, '-', label='Continuum')
            ax[0].set_xlabel(r'Wavelengths $(\mathrm{\AA})$')
            ax[0].set_ylabel('Flux')
            
            ax[1].set_title('Normalized spectrum')
            ax[1].plot(wavelengths, self.normalized_fluxes, 'k-')
            ax[1].set_xlabel(r'Wavelengths $(\mathrm{\AA})$')
            ax[1].set_ylabel('Flux')
            
            fig.tight_layout()
        
    
    def trace_continuum(self, wavelengths, fluxes, weights, knots_bin = 200, thres=3, plotting=False):
        self.knots = np.arange(wavelengths[0]+1e-10, wavelengths[-1]+1e-10, knots_bin)
        
        sp_param = splrep(wavelengths, fluxes*weights**2, t=self.knots, k=5)  # k is the degree of the spline

        continuum_fluxes = splev(wavelengths, sp_param)

        if thres:
            res = fluxes-continuum_fluxes
            std = np.std(res)
            pscale = np.median(wavelengths[1:]-wavelengths[:-1])
            width = int(8/pscale) # 400 km/s at 6000 A
            detection = (np.abs(res)>thres*std)

            i_weights = []
            if len(res[detection])>0:
                # find the line center
                lends, rends = np.where(np.diff(detection.astype(int)) == 1)[0] + 1, np.where(np.diff(detection.astype(int)) == -1)[0] + 1
                if detection[-1] == True: # dectecion = (..., True, True, True)
                    rends = np.concatenate((rends, np.array([len(res)-1])))
                if detection[0] == True: # dectecion = (True, True, True, ...)
                    lends = np.concatenate((np.array([0]), lends))
                    
                centers = (lends+rends) // 2
                
                for lend, rend, center in zip(lends, rends, centers):
                    l,r = max(0, center-3*width), min(len(res)-1, center+3*width)
                    ll,rr = max(0,center-int(50/pscale)), min(len(res)-1, center+int(50/pscale))
                    if ll<l and rr>r and np.median(weights[ll:l])>3 and np.median(weights[r:rr])>3:
                        i_weights.append(np.arange(l,r+0.5,1))
            
            if len(i_weights)>0:
                i_weights = np.concatenate(i_weights).astype(int)
                weights[i_weights]=3e-15

            self.sp_param = splrep(wavelengths, fluxes, t=self.knots, k=5, w = weights)  # k is the degree of the spline
        
            self.continuum_fluxes = splev(wavelengths, self.sp_param)
            self.continuum_fluxes[self.continuum_fluxes==0] = 1e+5*np.max(np.abs(fluxes))
            
from scipy.interpolate import CubicSpline

def resampler(wavelengths, values, new_wavelengths): # spline interpolation
    cs = CubicSpline(wavelengths, values, extrapolate=True)
    new_values = cs(new_wavelengths)

    return new_values

def discrete_resampler(wavelengths, values, new_wavelengths):
    new_values = np.interp(new_wavelengths, wavelengths, values, left=0, right=0)
    new_values = np.round(new_values).astype(int)
    return new_values

from astropy.constants import c
c = c.to_value('km/s')
from scipy.signal.windows import tukey


def shift_templates(templates, z_range=[-0.1,2], apodization_size=0.05, knots_bin = 100, thres=3):
    shifted_templates = {}
    for temp_name in templates.keys():
        temp = templates[temp_name][0]
        normalize=continuum(temp[0], temp[1], np.ones_like(temp[0]), knots_bin, thres, apodization_size)
        # normalize=continuum(temp[0], temp[1], 100*np.ones_like(temp[0]))
        
        # log pixel scale
        log_wavelengths = np.log10(temp[0])
        log_bin = np.median(log_wavelengths[1:]-log_wavelengths[:-1])

        # prepare a wavelength array
        n_left, n_right = -int(np.log10(z_range[0]+1)/log_bin), int(np.log10(z_range[1]+1)/log_bin)
        n_shift = n_left+n_right+1 # 1 for the zero-redshift
        n_pixel = temp.shape[1]+n_left+n_right

        min_log_wavelengths = np.log10(temp[0,0])-n_left*log_bin
        shifted_log_wavelengths = min_log_wavelengths + np.arange(n_pixel)*log_bin
        shifted_wavelengths = pow(10,shifted_log_wavelengths)

        # construct the shifted fluxes
        shifted_fluxes = np.zeros((n_shift, n_pixel))

        for i in range(n_shift):
            shifted_fluxes[i,i:i+temp.shape[1]] = normalize.normalized_fluxes
        
        min_log_vel = -n_left*log_bin
        shifted_vels = c*(pow(10,np.ones(n_shift)*min_log_vel+np.arange(n_shift)*log_bin)-1)
        shifted_templates[temp_name] = [shifted_vels, shifted_wavelengths, shifted_fluxes]
    return shifted_templates


import copy

def clean_spectrum(spec, window, sn):
    pscale = np.median(spec[0,1:]-spec[0,:-1])
    window = int(window/pscale)

    for i_left in range(0, spec.shape[1], window):
        if np.median(np.abs(spec[1,i_left:i_left+window]/spec[2,i_left:i_left+window])) > sn:
            break
        
    for i_right in range(spec.shape[1], 0, -window):
        if np.median(np.abs(spec[1,i_right-window:i_right]/spec[2,i_right-window:i_right])) > sn:
            break
    
    if i_left < i_right:
        cspec = copy.deepcopy(spec[:,i_left:i_right])
    else:
        cspec = copy.deepcopy(spec)
        cspec[1,:] = 0
        
    return cspec

import copy

def abs_supress(wavelengths, flux, pscale, thres):
    flux = copy.deepcopy(flux)
    detection = flux < -1*thres
    
    if len(flux[detection])>0:
        # find the line center
        lends, rends = np.where(np.diff(detection.astype(int)) == 1)[0] + 1, np.where(np.diff(detection.astype(int)) == -1)[0] + 1
        if detection[-1] == True: # dectecion = (..., True, True, True)
            rends = np.concatenate((rends, np.array([len(flux)-1])))
        if detection[0] == True: # dectecion = (True, True, True, ...)
            lends = np.concatenate((np.array([0]), lends))
            
        centers = (lends+rends) // 2
        
        # smoothly replace the lines with 0
        region = np.zeros_like(flux).astype(bool)
        width = int(24/pscale) # 3x400 km/s at 6000 A
        for lend, rend, center in zip(lends, rends, centers):
            l,r = max(0, center-width), min(len(flux)-1, center+width)
            window = 1-tukey(r-l+1, alpha=0.3)
            flux[l:r+1] *= window
        
    return flux

def emi_supress(wavelengths, flux, pscale, thres):
    flux = copy.deepcopy(flux)
    detection = flux > thres
    
    if len(flux[detection])>0:
        mask = np.zeros_like(flux).astype(bool)
        flux[detection] = 0
        
        # find the line center
        lends, rends = np.where(np.diff(detection.astype(int)) == 1)[0] + 1, np.where(np.diff(detection.astype(int)) == -1)[0] + 1
        if detection[-1] == True: # dectecion = (..., True, True, True)
            rends = np.concatenate((rends, np.array([len(flux)-1])))
        if detection[0] == True: # dectecion = (True, True, True, ...)
            lends = np.concatenate((np.array([0]), lends))
            
        centers = (lends+rends) // 2
        
        # smoothly replace the lines with 0
        region = np.zeros_like(flux).astype(bool)
        width = int(24/pscale) # 3x400 km/s at 6000 A
        for lend, rend, center in zip(lends, rends, centers):
            l,r = max(0, center-width), min(len(flux)-1, center+width)
            window = 1-tukey(r-l+1, alpha=0.3)
            flux[l:r+1] *= window

    return flux


def process_spectrum(wavelengths, normalized_fluxes, pscale, temp_type, mask=None, abs_thres=3, emi_thres=3):
    if temp_type == 'absorption':
        processed_fluxes = emi_supress(wavelengths, normalized_fluxes, pscale, abs_thres*np.std(normalized_fluxes))
    elif temp_type == 'emission':
        processed_fluxes = abs_supress(wavelengths, normalized_fluxes, pscale,  emi_thres*np.std(normalized_fluxes))
    else:
        raise TypeError('The type of %s is incorrect'%temp_type)
    if type(mask)==type([]) or type(mask)==type(np.array([])):
        for i in range(len(mask)):
            left_end = abs(wavelengths- mask[i][0]).argmin()
            right_end = abs(wavelengths - mask[i][1]).argmin()
            window = 1-tukey(right_end-left_end+1, alpha=0.3)
            processed_fluxes[left_end:right_end+1] *= window
    elif mask!=None:
        raise TypeError('Type of the maks must be list or 2D numpy array')
    
    
    
    return processed_fluxes
    
    
from astropy.modeling import models, fitting

def return_nan(x):
    return np.nan

class cc_result:
    def __init__(self, spectrum, normalize, pscale, templates, shifted_templates, temp_name,
                 temp_apodization_size, temp_knots_bin, temp_line_thres,
                 z_range=[-0.01,2], peak_range=0.001, mask=None, abs_thres=3, emi_thres=3):
        self.spectrum, self.normalize, self.pscale, self.templates, self.shifted_templates = spectrum, normalize, pscale, templates, shifted_templates
        self.temp_apodization_size, self.temp_knots_bin, self.temp_line_thres = temp_apodization_size, temp_knots_bin, temp_line_thres
        self.z_range, self.peak_range = z_range, peak_range
        self.processed_fluxes = process_spectrum(self.spectrum[0], self.normalize.normalized_fluxes, self.pscale,
                                                 self.templates[temp_name][2], mask, abs_thres, emi_thres)
        self.shifted_vels, self.shifted_wavelengths, self.shifted_fluxes, self.template_spectrum = self.shifted_templates[temp_name][0], self.shifted_templates[temp_name][1], self.shifted_templates[temp_name][2], self.templates[temp_name][0]
        try:
            self.cross_correlate()
            self.z_finding(self.templates[temp_name][1])
            self.cal_chi_eff()
        except:
            self.z, self.zerr, self.r, self.result, self.chi_eff  = np.nan, np.nan, np.nan, 'Bad cc signal', np.nan
   
        
    def cz_init(self):
        overlap_wavelengths = self.shifted_wavelengths[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])]
        self.new_fluxes0, self.new_weights0, self.new_masks0 = np.zeros_like(self.shifted_wavelengths), np.ones_like(self.shifted_wavelengths), np.zeros_like(self.shifted_wavelengths)
        overlap_fluxes = resampler(self.spectrum[0], self.processed_fluxes, overlap_wavelengths)
        overlap_weights = resampler(self.spectrum[0], np.abs(self.spectrum[1]/self.spectrum[2]), overlap_wavelengths)
        overlap_masks = discrete_resampler(self.spectrum[0], self.spectrum[3], overlap_wavelengths)

        self.new_fluxes0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_fluxes
        self.new_weights0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_weights
        self.new_masks0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_masks

        self.cc0 = np.matmul(self.shifted_fluxes, self.new_masks0*self.new_weights0**2*self.new_fluxes0)

        self.cz0 = self.shifted_vels[np.nanargmax(self.cc0)]

    def cc_near_peak(self):
        # zero padd the spectrum
        left_new_wavelengths, right_new_wavelengths = self.shifted_wavelengths[self.shifted_wavelengths<self.spectrum[0,0]], self.shifted_wavelengths[self.shifted_wavelengths>self.spectrum[0,-1]]
        self.new_wavelengths = np.concatenate([left_new_wavelengths, self.spectrum[0], right_new_wavelengths])

        self.new_fluxes, self.new_weights, self.new_masks, self.new_fit_weights = np.zeros_like(self.new_wavelengths), np.ones_like(self.new_wavelengths), np.ones_like(self.new_wavelengths), np.ones_like(self.new_wavelengths)
        self.new_fluxes[(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])] = self.processed_fluxes
        self.new_fit_weights[(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])] = np.abs(self.spectrum[3]/self.spectrum[2])
        self.new_weights[(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])] = np.abs(self.spectrum[1]/self.spectrum[2])
        self.new_masks[(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])] = np.ones_like(self.spectrum[1])

        self.peak_region = np.where((self.shifted_vels>self.cz0-c*self.peak_range)&(self.shifted_vels<self.cz0+c*self.peak_range))[0]
        self.interp_fluxes = np.zeros((len(self.peak_region),len(self.new_fluxes)))
        for n, i in enumerate(self.peak_region):
            region = ((self.new_wavelengths>=self.shifted_wavelengths[i])&(self.new_wavelengths<=self.shifted_wavelengths[i+self.template_spectrum.shape[1]-1]))
            new_template_wavelengths = self.new_wavelengths[region]
            resampled_template_fluxes = resampler(self.shifted_wavelengths[i:i+self.template_spectrum.shape[1]], self.template_spectrum[1], new_template_wavelengths)
            self.interp_fluxes[n,region] = continuum(new_template_wavelengths, resampled_template_fluxes, self.new_fit_weights[region], self.temp_knots_bin,
                                                     self.temp_line_thres, self.temp_apodization_size).normalized_fluxes
            # self.interp_fluxes[n,region] = continuum_rm(new_template_wavelengths, resampled_template_fluxes, self.new_fit_weights[region]).normalized_fluxes

        self.cc = copy.deepcopy(self.cc0)
        self.cc[self.peak_region] = np.matmul(self.interp_fluxes,self.new_masks*self.new_weights**2*self.new_fluxes)

    
    def cross_correlate(self):
        self.cz_init()
        self.cc_near_peak()

    def z_finding(self, pkfrac=0.65):

        correlation_range = np.array(self.z_range)*c
        _cc = self.cc[(self.shifted_vels>correlation_range[0])&(self.shifted_vels<correlation_range[1])]
        _lags = self.shifted_vels[(self.shifted_vels>correlation_range[0])&(self.shifted_vels<correlation_range[1])]
        i_peak = np.nanargmax(_cc)
        peak = _cc[i_peak] # estimates a peak
        lag_peak = _lags[i_peak]
        cc_fit, lags_fit = self.cc[self.cc >= pkfrac*peak], self.shifted_vels[self.cc >= pkfrac*peak] # select points >pkfrac*peak
                
        if np.nanmax(lags_fit[1:]-lags_fit[:-1]) < 500:
            center = lags_fit[np.nanargmax(cc_fit)]
            fit = fitting.LevMarLSQFitter()
            
            gaussian = models.Gaussian1D(amplitude=np.nanmax(cc_fit), mean=center, stddev=lags_fit[-1]-lags_fit[0])
            self.fit_gaussian = fit(gaussian, lags_fit, cc_fit)
            fitted_center = self.fit_gaussian.mean.value
            
            if self.fit_gaussian.amplitude.value < 0 or fitted_center<correlation_range[0] or fitted_center>correlation_range[1]:
                self.z, self.zerr, self.r, self.result  = np.nan, np.nan, np.nan, 'No peak'
            else:
                self.z = fitted_center/c # estimate the redshift
                self.zerr = self.fit_gaussian.stddev.value*np.sqrt(-2*np.log(1-0.5/self.fit_gaussian.amplitude.value))/c
                self.result = 'Well fitted'
                npeak = np.abs(self.shifted_vels-fitted_center).argmin() # find an index of peak
                N = int(0.1*c/(self.shifted_vels[npeak]-self.shifted_vels[npeak-1]))
                left, right = max(npeak-N,0), min(npeak+N, len(self.cc))
                nrange = int(min(npeak-left, right-npeak))
                self.npeak, self.N, self.left, self.right, self.nrange = npeak, N, left, right, nrange
                cc_left, cc_right = self.cc[npeak-nrange:npeak], np.flip(self.cc[npeak:npeak+nrange])
                cc_left, cc_right = cc_left[(~np.isnan(cc_left))&(~np.isnan(cc_right))], cc_right[(~np.isnan(cc_left))&(~np.isnan(cc_right))]
                sigma = np.sum(((cc_left - cc_right)**2))/nrange
                self.cc_left, self.cc_right = cc_left, cc_right
                self.sigma = sigma
                self.r = peak/(np.sqrt(sigma))
                
        else:
            fit_condition = (self.cc > pkfrac*peak) & (np.abs(self.shifted_vels-lag_peak)<500)
            cc_fit, lags_fit = self.cc[fit_condition], self.shifted_vels[fit_condition]

            center = lags_fit[np.nanargmax(cc_fit)]
            fit = fitting.LevMarLSQFitter()
            
            gaussian = models.Gaussian1D(amplitude=np.nanmax(cc_fit), mean=center, stddev=lags_fit[-1]-lags_fit[0])
            self.fit_gaussian = fit(gaussian, lags_fit, cc_fit)
            fitted_center = self.fit_gaussian.mean.value
            if self.fit_gaussian.amplitude.value < 0 or fitted_center<correlation_range[0] or fitted_center>correlation_range[1]:
                self.z, self.zerr, self.r, self.result  = np.nan, np.nan, np.nan, 'No peak'
            else:
                self.z = fitted_center/c # estimate the redshift
                self.zerr = self.fit_gaussian.stddev.value*np.sqrt(-2*np.log(1-0.5/self.fit_gaussian.amplitude.value))/c
                self.result = 'subpeak exists'
                npeak = np.abs(self.shifted_vels-fitted_center).argmin() # find an index of peak
                N = int(0.1*c/(self.shifted_vels[npeak]-self.shifted_vels[npeak-1]))
                left, right = max(npeak-N,0), min(npeak+N, len(self.cc))
                nrange = int(min(npeak-left, right-npeak))
                cc_left, cc_right = self.cc[npeak-nrange:npeak], np.flip(self.cc[npeak:npeak+nrange])
                cc_left, cc_right = cc_left[(~np.isnan(cc_left))&(~np.isnan(cc_right))], cc_right[(~np.isnan(cc_left))&(~np.isnan(cc_right))]
                sigma = np.sum(((cc_left - cc_right)**2))/nrange
                self.r = peak/(np.sqrt(sigma))

    
    def cal_chi_eff(self):
        self.i_vel = np.argmin(np.abs(self.shifted_vels-c*self.z))-self.peak_region[0]
        try:
            self.T_ = self.normalize.continuum_fluxes*(self.interp_fluxes[self.i_vel,(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])]+1)
            self.chi_eff = np.sum((np.abs(self.spectrum[3]/self.spectrum[2]))**2*(self.spectrum[1]-self.T_)**2)/(np.sum(self.spectrum[3])-len(self.normalize.knots)-1)
        except:
            self.chi_eff = 1000

            
import os
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

class rvm:
    def __init__(self, templates, z_range=[-0.01,2], temp_apodization_size=0.05, temp_knots_bin = 100, temp_line_thres=3):
        self.templates = templates
        self.z_range = z_range
        self.temp_apodization_size, self.temp_knots_bin, self.temp_line_thres = temp_apodization_size, temp_knots_bin, temp_line_thres
        self.shifted_templates = shift_templates(templates, self.z_range, self.temp_apodization_size, temp_knots_bin, temp_line_thres)
    
    def z_single(self, spectrum, peak_range=0.01, mask=None, knots_bin=200, line_thres=3, abs_thres=3, emi_thres=3, apodization_size=0.05, sn_window=10, min_sn=2):
        self.pscale = np.median(spectrum[0,1:]-spectrum[0,:-1])
        spectrum = clean_spectrum(spectrum, sn_window, min_sn)
        normalize = continuum(spectrum[0], spectrum[1], np.abs(spectrum[3]/spectrum[2]), knots_bin, line_thres, apodization_size)
        # normalize = continuum_rm(spectrum[0], spectrum[1], np.abs(1/spectrum[2]))

        output = {}
        template_names = list(self.templates.keys())
        n_templates = len(template_names)
        z, zerr, r, chi_eff, flag = np.zeros(n_templates), np.zeros(n_templates), np.zeros(n_templates), np.zeros(n_templates), np.zeros(n_templates, dtype=int)
        for i, temp_name in enumerate(template_names):
            cc_spec_temp = cc_result(spectrum, normalize, self.pscale, templates=self.templates, shifted_templates=self.shifted_templates,
                                     temp_name=temp_name, temp_apodization_size=self.temp_apodization_size,
                                     temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, z_range=self.z_range,
                                     peak_range=peak_range, mask=mask, abs_thres=abs_thres, emi_thres=emi_thres)
            output[temp_name] = cc_spec_temp
        # return output
            z[i], zerr[i], r[i], chi_eff[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
            if cc_spec_temp.result == 'Well fitted':
                flag[i] = 0
            elif cc_spec_temp.result == 'subpeak exists':
                flag[i] = 1
            else:
                flag[i] = 99
                
        # remove the results with nan-redshift
        nan_check = (~np.isnan(zerr))&(~np.isnan(chi_eff))&(chi_eff<1.5)
        template_names, z, zerr, r, chi_eff, flag = np.array(template_names)[nan_check], z[nan_check], zerr[nan_check], r[nan_check], chi_eff[nan_check], flag[nan_check]
        
        # arange the value in the order of chi_eff
        order = np.flip(np.argsort(r))
        template_names, z, zerr, r, chi_eff, flag = template_names[order], z[order], zerr[order], r[order], chi_eff[order], flag[order]
        
        table = np.vstack((template_names, z, zerr, r, chi_eff, flag))
        column_names = ['template_name', 'z', 'zerr', 'r', 'chi_eff', 'flag']
        df = pd.DataFrame(table.T, columns = column_names)
        df = df.astype({'template_name':str, 'z':np.float32, 'zerr':np.float32, 'r':np.float32, 'chi_eff':np.float32, 'flag':int})
        
        return df
    
    def cc_analysis(self, spectrum, temp_name, peak_range=0.01, mask=None, knots_bin=100, line_thres=3, abs_thres=3, emi_thres=3, apodization_size=0.05, sn_window=10, min_sn=2):
        if type(temp_name) == str:
            temp_name = temp_name
        else:
            z_single_result = self.z_single(spectrum, peak_range=0.001, mask=None, knots_bin=100, line_thres=3, abs_thres=3, emi_thres=3, apodization_size=0.05, sn_window=10, min_sn=2)
            temp_name = z_single_result['template_name'][temp_name]
        
        self.cspectrum = clean_spectrum(spectrum, sn_window, min_sn)
        self.norm = continuum(self.cspectrum[0], self.cspectrum[1], np.abs(1/self.cspectrum[2]), knots_bin, line_thres, apodization_size)
        # self.norm = continuum_rm(self.cspectrum[0], self.cspectrum[1], np.abs(1/self.cspectrum[2]))

        self.cc_result = cc_result(self.cspectrum, self.norm, self.pscale, self.templates, self.shifted_templates,
                            temp_name, self.temp_apodization_size, self.temp_knots_bin, self.temp_line_thres,
                            self.z_range, peak_range, mask, abs_thres, emi_thres)
         
        
    def z_multi(self, spectrums, directory=None, multi_process=4, **kwargs):
        if directory==None:
            if len(glob.glob('z_result'))==0:
                os.system('mkdir z_result')
            save_folder = 'z_result'
        # elif ~len(glob.glob(directory)):
        #     raise KeyError('%s does not exist'%directory)
        else:
            if len(glob.glob(directory))==0:
                os.system('mkdir %s'%directory)
            save_folder = directory
        
        os.system('rm -fr %s'%save_folder+'/z_result*.txt')
        if multi_process:
            spec_number = np.arange(len(spectrums))
            spec_subset = [[n, spec_number[int(len(spec_number)*n/multi_process):int(len(spec_number)*(n+1)/multi_process)]] for n in range(multi_process)]
            def z_multi_sub(indices, subset_number, **kwargs):
                warnings.simplefilter('ignore')
                with open(save_folder+'/z_result%d.txt'%subset_number, 'w') as file:
                    file.write('number best_template z zerr chi_eff flag\n')
                    if subset_number == len(spec_subset)-1:
                        for index in tqdm(indices):
                            result = self.z_single(spectrums[index], **kwargs)
                            try:
                                file.write('%d %s %f %f %f %f %d\n'%(index, result['template_name'][0], result['z'][0],
                                                            result['zerr'][0], result['r'][0], result['chi_eff'][0], result['flag'][0]))
                            except:
                                file.write('%d nan -9 -9 -9 -9 9\n'%(index))
                    else:
                        for index in indices:
                            result = self.z_single(spectrums[index], **kwargs)
                            try:
                                file.write('%d %s %f %f %f %f %d\n'%(index, result['template_name'][0], result['z'][0],
                                                            result['zerr'][0], result['r'][0], result['chi_eff'][0], result['flag'][0]))
                            except:
                                file.write('%d nan -9 -9 -9 -9 9\n'%(index))
                                
                    warnings.resetwarnings()
            Parallel(multi_process, verbose=0)(delayed(z_multi_sub)(subset[1], subset[0],**kwargs) for subset in spec_subset)
            z_result_files = np.sort(glob.glob(save_folder+'/z_result*.txt'))
            
            with open(save_folder+'/z_result.txt', 'w') as final_file:
                final_file.write('number best_template z zerr chi_eff flag\n')
                for result_file in z_result_files:
                    with open(result_file) as file:
                        for n, line in enumerate(file):
                            if n>0:
                                final_file.write(line)
            
            
        else:
            spec_number = np.arange(len(spectrums))
            with open(save_folder+'/z_result.txt', 'w') as file:
                file.write('number best_template z zerr chi_eff flag\n')
                for index in tqdm(spec_number):
                    result = self.z_single(spectrums[index], **kwargs)
                    try:
                        file.write('%d %s %f %f %f %f %d\n'%(index, result['template_name'][0], result['z'][0],
                                                    result['zerr'][0], result['r'][0], result['chi_eff'][0], result['flag'][0]))
                    except:
                        file.write('%d nan -9 -9 -9 -9 9\n'%(index))
            
    
    