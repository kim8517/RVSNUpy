import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import copy

from RVSNUpy.spectrum_import import MMT_raw, sdss_fits
import numpy as np
from RVSNUpy.template import mmt_template_galaxy, sdss_template_galaxy
import glob
import os
import pandas as pd
from tqdm import tqdm

summary = pd.read_csv('rvsnu_py_test/summary.csv')
z_dr16, zerr_dr16 = np.array(summary['Z_DR16']), np.array(summary['Zerr_DR16']),
z_hecto, zerr_hecto = np.array(summary['Z_Hecto']), np.array(summary['Zerr_Hecto'])
specfile_dr16, specfile_hecto = np.array(summary['SPEC_DR16']), np.array(summary['SPEC_Hecto'])

# import Hectomap fits file
fits_file = np.array(glob.glob('rvsnu_py_test/HMAP_MMT/*.fits'))
hecto_fits = []
# rearange file order
for file in specfile_hecto:
    path='rvsnu_py_test/HMAP_MMT/'+file
    hecto_fits.append(fits_file[fits_file==path][0])
    if len(fits_file[fits_file==path])!=1:
        print('strange!')

# import Hectomap fits file
fits_file = np.array(glob.glob('rvsnu_py_test/HMAP_SDSS/*.fits'))
dr16_fits = []
# rearange file order
for file in specfile_dr16:
    path='rvsnu_py_test/HMAP_SDSS/'+file
    dr16_fits.append(fits_file[fits_file==path][0])

hecto_spec = []
for file in tqdm(hecto_fits):
    hecto_spec.append(MMT_raw(file))

dr16_spec = []
for file in tqdm(dr16_fits):
    dr16_spec.append(sdss_fits(file))

from RVSNUpy.rvm import resampler

def read_model(file, age):
    age_col = {'1':1, '3':2, '5':3, '7':4, '9':5, '11':6} # column corresponding to the age
    wavelength, flux = np.loadtxt(file, usecols=[0,age_col['%d'%age]], unpack=True, skiprows=6)
    error, mask = copy.deepcopy(flux)/5, np.ones_like(flux)
    return np.vstack([wavelength, flux, error, mask])

model_templates = {}
for age in [1, 3, 5, 7, 9, 11]:
    modeltemp = read_model('model_template_hr.csv', age)
    new_wavelengths = np.logspace(np.log10(modeltemp[0,0]), np.log10(modeltemp[0,-1]), len(modeltemp[0]))
    new_fluxes, new_error = resampler(modeltemp[0], modeltemp[1], new_wavelengths), resampler(modeltemp[0], modeltemp[2], new_wavelengths)
    new_modeltemp = np.vstack([new_wavelengths, new_fluxes, new_error, np.ones_like(new_wavelengths)])
    model_templates[f'{age:d}yr'] = [new_modeltemp, 0.7, 1, [9000,100]]

abs_templates = {}
emi_templates = {}
for name in sdss_template_galaxy.keys():
    temp = sdss_template_galaxy[name]
    if temp[2]==1:
        abs_templates[name] = copy.deepcopy(temp)
    else:
        emi_templates[name] = copy.deepcopy(temp)

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sigmaclip
import copy
from scipy.interpolate import splrep, splev
from scipy.signal.windows import tukey

class continuum:
    def __init__(self, wavelengths, fluxes, weights, template=False, knots_bin = 100, thres=3, apodization_size = 0.05, plotting=False):
        self.trace_continuum(wavelengths, fluxes, weights, template = template, knots_bin = knots_bin, thres = thres)
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
        
    
    def trace_continuum(self, wavelengths, fluxes, weights, template=False, knots_bin = 200, thres=3, plotting=False):
        self.knots = np.arange(wavelengths[0]+1e-10, wavelengths[-1]+1e-10, knots_bin)
        
        sp_param = splrep(wavelengths, fluxes, t=self.knots, w=weights, k=5)  # k is the degree of the spline

        self.continuum_fluxes = splev(wavelengths, sp_param)

        if thres:
            res = fluxes-self.continuum_fluxes
            std = np.std(res)
            pscale = np.median(wavelengths[1:]-wavelengths[:-1])
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
                widths = (rends-lends) // 2
                self.centers = centers
                
                for lend, rend, center, width in zip(lends, rends, centers, widths):
                    l,r = max(0, center-3*width), min(len(res)-1, center+3*width)
                    if template:
                        i_weights.append(np.arange(l,r+0.5,1))
                    else:
                        ll,rr = max(0,center-5*width), min(len(res)-1, center+5*width)
                        if np.median(fluxes[l:r]*weights[l:r]>3):
                            i_weights.append(np.arange(l,r+0.5,1))
                        elif ll<l and rr>r and np.median(fluxes[ll:l]*weights[ll:l])>3 and np.median(fluxes[r:rr]*weights[r:rr])>3:
                            i_weights.append(np.arange(l,r+0.5,1))
            
            if len(i_weights)>0:
                i_weights = np.concatenate(i_weights).astype(int)
                weights[i_weights]=3e-15
            self.sp_param = splrep(wavelengths, fluxes, t=self.knots, k=5, w = weights)  # k is the degree of the spline
        
            self.continuum_fluxes = splev(wavelengths, self.sp_param)
            self.continuum_fluxes[self.continuum_fluxes==0] = 1e+5*np.max(np.abs(fluxes))
            self.continuum_fluxes[0], self.continuum_fluxes[-1] = self.continuum_fluxes[1], self.continuum_fluxes[-2]
            
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
        normalize=continuum(temp[0], temp[1], np.ones_like(temp[1]), template=True, knots_bin=knots_bin, thres=thres, apodization_size=apodization_size)
        
        # log pixel scale
        log_wavelengths = np.log10(temp[0])
        log_bin = np.median(log_wavelengths[1:]-log_wavelengths[:-1])

        # prepare a wavelength array
        n_left, n_right = -int(np.log10(z_range[0]-0.1+1)/log_bin), int(np.log10(z_range[1]+0.1+1)/log_bin)
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
        if np.median(spec[1,i_left:i_left+window]/spec[2,i_left:i_left+window]) > sn:
            break
        
    for i_right in range(spec.shape[1], 0, -window):
        if np.median(spec[1,i_right-window:i_right]/spec[2,i_right-window:i_right]) > sn:
            break
    
    if i_left+10 < i_right:
        cspec = copy.deepcopy(spec[:,i_left:i_right])
    else:
        cspec = copy.deepcopy(spec)
        cspec[1,:] = 0
        
    return cspec

import copy
from scipy.signal import butter,filtfilt

def abs_supress(wavelengths, flux, pscale, thres):
    flux = copy.deepcopy(flux)
    detection = flux < -1*thres
    
    i_fit = np.ones_like(wavelengths).astype(bool)
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
            i_fit[l:r+1] = False
        
    return flux, i_fit

def emi_supress(wavelengths, flux, pscale, thres):
    flux = copy.deepcopy(flux)
    detection = flux > thres
    
    i_fit = np.ones_like(flux).astype(bool)
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
            i_fit[l:r+1] = False

    return flux, i_fit

def butter_highstop_filter(data, cutoff):
    nyq = 0.5
    normal_cutoff = cutoff / nyq
    b, a = butter(2, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def process_spectrum(wavelengths, sn_array, normalized_fluxes, min_sn, pscale, temp_type, hcutoff_scale=0, mask=None, abs_thres=3, emi_thres=3):
    if temp_type == 1:
        processed_fluxes, i_fit = emi_supress(wavelengths, normalized_fluxes, pscale, abs_thres*np.std(normalized_fluxes))
        if hcutoff_scale:
            hcutoff_scale = hcutoff_scale/pscale
            hcutoff = 0.5/hcutoff_scale
            processed_fluxes = butter_highstop_filter(processed_fluxes, hcutoff)
    elif temp_type == 2:
        processed_fluxes, i_fit = abs_supress(wavelengths, normalized_fluxes, pscale,  emi_thres*np.std(normalized_fluxes))
        if hcutoff_scale:
            hcutoff_scale = hcutoff_scale/pscale
            hcutoff = 0.5/hcutoff_scale
            processed_fluxes = butter_highstop_filter(processed_fluxes, hcutoff)
    elif temp_type == 0:
        processed_fluxes = copy.deepcopy(normalized_fluxes)
        i_fit = np.ones_like(wavelengths).astype(bool)
        if hcutoff_scale:
            hcutoff_scale = hcutoff_scale/pscale
            hcutoff = 0.5/hcutoff_scale
            processed_fluxes = butter_highstop_filter(processed_fluxes, hcutoff)
    else:
        raise TypeError('The type of %s is incorrect'%temp_type)
    
    processed_fluxes[sn_array<min_sn] = 0
    i_fit[sn_array<min_sn] = False
    
    
    return processed_fluxes, i_fit


from astropy.modeling import models, fitting
import warnings

def return_nan(x):
    return np.nan

class cc_result:
    def __init__(self, spectrum, normalize, processed_fluxes, i_fit, template, shifted_template,
                 temp_apodization_size, temp_knots_bin, temp_line_thres,
                 z_range=[-0.01,2], abs_thres=3, emi_thres=3):
        warnings.simplefilter('ignore')
        self.spectrum, self.normalize, self.processed_fluxes, self.i_fit = spectrum, normalize, processed_fluxes, i_fit
        self.template, self.shifted_template  = template, shifted_template
        self.temp_apodization_size, self.temp_knots_bin, self.temp_line_thres = temp_apodization_size, temp_knots_bin, temp_line_thres
        self.z_range = z_range
        self.shifted_vels, self.shifted_wavelengths, self.shifted_fluxes, self.template_spectrum = self.shifted_template[0], self.shifted_template[1], self.shifted_template[2], self.template[0]
        # self.cross_correlate()
        try:
            self.cross_correlate()
        except:
            self.z, self.zerr, self.r, self.result, self.chi_eff  = np.nan, np.nan, np.nan, 99, np.nan
        
        warnings.resetwarnings()
        
    def cross_correlate0(self):
        overlap_wavelengths = self.shifted_wavelengths[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])]
        self.new_fluxes0, self.new_weights0, self.new_masks0 = np.zeros_like(self.shifted_wavelengths), np.ones_like(self.shifted_wavelengths), np.ones_like(self.shifted_wavelengths)
        overlap_fluxes = resampler(self.spectrum[0], self.processed_fluxes, overlap_wavelengths)
        overlap_weights = resampler(self.spectrum[0], np.abs(self.spectrum[1]/self.spectrum[2]), overlap_wavelengths)
        overlap_masks = discrete_resampler(self.spectrum[0], self.spectrum[3], overlap_wavelengths)

        self.new_fluxes0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_fluxes
        self.new_weights0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_weights
        self.new_masks0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_masks

        self.cc0 = np.matmul(self.shifted_fluxes, self.new_masks0*self.new_weights0**2*self.new_fluxes0)
        # self.cc0 = np.matmul(self.shifted_fluxes, self.new_fluxes0)


        self.cz0 = self.shifted_vels[np.nanargmax(self.cc0)]

    def find_peak_region(self, pkfrac):
        cz_range = np.array(self.z_range)*c
        cc_inrange = self.cc0[(self.shifted_vels>cz_range[0])&(self.shifted_vels<cz_range[1])]
        lags_inrange = self.shifted_vels[(self.shifted_vels>cz_range[0])&(self.shifted_vels<cz_range[1])]
        
        i_peak = np.nanargmax(cc_inrange)
        h_peak, v_peak = cc_inrange[i_peak], lags_inrange[i_peak] # estimates a peak
        cc_pkfrac, lags_pkfrac = self.cc0[self.cc0 >= pkfrac*h_peak], self.shifted_vels[self.cc0 >= pkfrac*h_peak] # select points >pkfrac*peak
        if np.nanmax(lags_pkfrac[1:]-lags_pkfrac[:-1]) > 500:
            cc_pkfrac = self.cc0[(self.cc0 > pkfrac*h_peak) & (np.abs(self.shifted_vels-v_peak)<500)]
            lags_pkfrac = self.shifted_vels[(self.cc0 > pkfrac*h_peak) & (np.abs(self.shifted_vels-v_peak)<500)]
            self.result = 2
        else:
            self.result = 1
        
        self.fit_peak_range = fitting.LevMarLSQFitter()
        self.gaussian_peak_range0 = models.Gaussian1D(amplitude=h_peak, mean=v_peak, stddev=lags_pkfrac[-1]-lags_pkfrac[0])
        self.gaussian_peak_range = self.fit_peak_range(self.gaussian_peak_range0, lags_pkfrac, cc_pkfrac)
        
        if self.gaussian_peak_range.amplitude.value < 0 or self.gaussian_peak_range.mean.value < cz_range[0] or self.gaussian_peak_range.mean.value>cz_range[1]:
                self.z, self.zerr, self.r, self.chi_eff, self.result  = np.nan, np.nan, np.nan, np.nan, 9
                
        else:
            self.peak_range = [self.gaussian_peak_range.mean.value-2*self.gaussian_peak_range.stddev.value,
                               self.gaussian_peak_range.mean.value+2*self.gaussian_peak_range.stddev.value]
        
            
    def cc_near_peak(self):
        # zero padd the spectrum
        left_new_wavelengths, right_new_wavelengths = self.shifted_wavelengths[self.shifted_wavelengths<self.spectrum[0,0]], self.shifted_wavelengths[self.shifted_wavelengths>self.spectrum[0,-1]]
        self.new_wavelengths = np.concatenate([left_new_wavelengths, self.spectrum[0], right_new_wavelengths])

        self.new_fluxes, self.new_weights, self.new_masks, self.new_fit_weights = np.zeros_like(self.new_wavelengths), np.ones_like(self.new_wavelengths), np.ones_like(self.new_wavelengths), np.ones_like(self.new_wavelengths)
        self.new_fluxes[(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])] = self.processed_fluxes
        self.new_fit_weights[(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])] = np.abs(self.spectrum[3]/self.spectrum[2])
        self.new_weights[(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])] = np.abs(self.spectrum[1]/self.spectrum[2])
        self.new_masks[(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])] = np.ones_like(self.spectrum[1])

        self.peak_region = np.where((self.shifted_vels>self.peak_range[0])&(self.shifted_vels<self.peak_range[1]))[0]
        self.interp_fluxes = np.zeros((len(self.peak_region),len(self.new_fluxes)))
        for n, i in enumerate(self.peak_region):
            region = ((self.new_wavelengths>=self.shifted_wavelengths[i])&(self.new_wavelengths<=self.shifted_wavelengths[i+self.template_spectrum.shape[1]-1]))
            new_template_wavelengths = self.new_wavelengths[region]
            resampled_template_fluxes = resampler(self.shifted_wavelengths[i:i+self.template_spectrum.shape[1]], self.template_spectrum[1], new_template_wavelengths)
            self.interp_fluxes[n,region] = continuum(new_template_wavelengths, resampled_template_fluxes, self.new_fit_weights[region], template=True, knots_bin = self.temp_knots_bin,
                                                     thres = self.temp_line_thres, apodization_size=self.temp_apodization_size).normalized_fluxes
        self.cc = copy.deepcopy(self.cc0)
        self.cc[self.peak_region] = np.matmul(self.interp_fluxes,self.new_masks*self.new_weights**2*self.new_fluxes)
        # self.cc[self.peak_region] = np.matmul(self.interp_fluxes, self.new_fluxes)


    
    def cross_correlate(self):
        self.cross_correlate0()
        self.find_peak_region(pkfrac=self.template[1])
        if self.result != 9:
            self.cc_near_peak()
            self.z_finding()
        else:
            self.cc = self.cc0

    def z_finding(self):
        cc_peak = self.cc[(self.shifted_vels>self.peak_range[0])&(self.shifted_vels<self.peak_range[1])]
        lags_peak = self.shifted_vels[(self.shifted_vels>self.peak_range[0])&(self.shifted_vels<self.peak_range[1])]
        
        self.fit_peak = fitting.LevMarLSQFitter()
        self.gaussian_peak0 = models.Gaussian1D(amplitude=np.nanmax(cc_peak), mean=np.median(lags_peak), stddev=0.5*(lags_peak[-1]-lags_peak[0]))
        self.gaussian_peak = self.fit_peak(self.gaussian_peak0, lags_peak, cc_peak)
        
        self.z = self.gaussian_peak.mean.value/c
        self.cal_zerr()
        self.cal_chi_eff()
        n_peak = np.abs(self.shifted_vels-self.gaussian_peak.mean.value).argmin() # find an index of peak
        N = int(0.1*c/(self.shifted_vels[n_peak]-self.shifted_vels[n_peak-1]))
        left, right = max(n_peak-N,0), min(n_peak+N, len(self.cc))
        nrange = int(min(n_peak-left, right-n_peak))
        cc_left, cc_right = self.cc[n_peak-nrange:n_peak], np.flip(self.cc[n_peak:n_peak+nrange])
        cc_left, cc_right = cc_left[(~np.isnan(cc_left))&(~np.isnan(cc_right))], cc_right[(~np.isnan(cc_left))&(~np.isnan(cc_right))]
        sigma = np.sum(((cc_left - cc_right)**2))/nrange
        self.r = self.gaussian_peak.amplitude.value/(np.sqrt(sigma))
    
    def cal_zerr(self):
        amp, mean, std = self.gaussian_peak.amplitude.value, self.gaussian_peak.mean.value, self.gaussian_peak.stddev.value
        damp, dmean, dstd = np.sqrt(self.fit_peak.fit_info['param_cov'][0,0]), np.sqrt(self.fit_peak.fit_info['param_cov'][1,1]), np.sqrt(self.fit_peak.fit_info['param_cov'][2,2])
        self.zerr_ = std*np.sqrt(-2*np.log(1-0.5/amp))/c
        self.dzerr_ = np.sqrt((dmean+2*std*np.sqrt(-2/np.log(1-0.5/amp))*damp/(amp*(amp-0.5))+2*dstd*np.sqrt(-2*np.log(1-0.5/amp)))**2+dmean**2)/c
        self.zerr = (self.zerr_+self.dzerr_)
    
    def cal_chi_eff(self):
        z_wavelength = self.template_spectrum[0]*(1+self.z)
        self.overlap_spec = copy.deepcopy(self.spectrum)
        # self.overlap_spec[1,:] = (self.processed_fluxes+1)*self.normalize.continuum_fluxes
        self.overlap_spec = self.spectrum[:,(self.spectrum[0,:]>max(z_wavelength[0], self.spectrum[0,0]))&
                                     (self.spectrum[0,:]<min(z_wavelength[-1], self.spectrum[0,-1]))]
        self.i_fit = self.i_fit[(self.spectrum[0,:]>max(z_wavelength[0], self.spectrum[0,0]))&(self.spectrum[0,:]<min(z_wavelength[-1], self.spectrum[0,-1]))]
        T = resampler(z_wavelength, self.template_spectrum[1], self.overlap_spec[0])
        T_continuum = continuum(self.overlap_spec[0], T, np.abs(self.overlap_spec[3]/self.overlap_spec[2]), template=True)
        self.T_ = T*self.normalize.continuum_fluxes[(self.spectrum[0,:]>max(z_wavelength[0], self.spectrum[0,0]))&(self.spectrum[0,:]<min(z_wavelength[-1], self.spectrum[0,-1]))]/T_continuum.continuum_fluxes
        self.chi_eff = np.sum(((self.overlap_spec[3]/self.overlap_spec[2])**2*(self.overlap_spec[1]-self.T_)**2)[self.i_fit])/(np.sum((self.overlap_spec[3])[self.i_fit])-len(T_continuum.knots)-1)                     
        self.T = T
        self.T_conti = T_continuum
        # self.i_vel = np.argmin(np.abs(self.shifted_vels-c*self.z))-self.peak_region[0]
        # try:
        #     self.T_ = self.normalize.continuum_fluxes*(self.interp_fluxes[self.i_vel,(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])]+1)
        #     self.chi_eff = np.sum((np.abs(self.spectrum[3]/self.spectrum[2]))**2*(self.spectrum[1]-self.T_)**2)/(np.sum(self.spectrum[3])-len(self.normalize.knots)-1)
        # except:
        #     self.chi_eff = 1000
    


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
        
        self.templates1, self.templates2 = {}, {}
        for name in self.templates.keys():
            if self.templates[name][2] == 1:
                self.templates1[name] = copy.deepcopy(self.templates[name])
            if templates[name][2] == 2:
                self.templates2[name] = copy.deepcopy(self.templates[name])
        
        self.shifted_templates = shift_templates(self.templates, self.z_range, self.temp_apodization_size, temp_knots_bin, temp_line_thres)
        self.shifted_templates1 = shift_templates(self.templates1, self.z_range, self.temp_apodization_size, temp_knots_bin, temp_line_thres)
        self.shifted_templates2 = shift_templates(self.templates2, self.z_range, self.temp_apodization_size, temp_knots_bin, temp_line_thres)
    
    def z_single(self, spectrum, output='all', spectrum_range=None, min_sn=2, hcutoff_scale=0, chi_thres=2, mask=None, r_abs=2, r_em=10, 
                 knots_bin=200, line_thres=3, abs_thres=2, emi_thres=2, apodization_size=0.05, window_continuum=40, sn_continuum=2):
        spectrum = copy.deepcopy(spectrum)
        if type(spectrum_range)==type([]) or type(spectrum_range)==type(np.array([])):
            if len(np.where((spectrum[0,:]>spectrum_range[0])&(spectrum[0,:]<spectrum_range[1]))[0]) ==0:
                raise ValueError('spectrum_range should contain spectrum wavelengths')
            spectrum = spectrum[:,(spectrum[0,:]>spectrum_range[0])&(spectrum[0,:]<spectrum_range[1])]
        
        self.pscale = np.median(spectrum[0,1:]-spectrum[0,:-1])
        if type(mask)==type([]) or type(mask)==type(np.array([])):
            for i in range(len(mask)):
                left_end = abs(spectrum[0,:]- mask[i][0]).argmin()
                right_end = abs(spectrum[0,:] - mask[i][1]).argmin()
                spectrum[3,:][left_end:right_end+1] = 0
            
        elif mask!=None:
            raise TypeError('Type of the maks must be list or 2D numpy array')
        
        
        spectrum = clean_spectrum(spectrum, window_continuum, sn_continuum)

        normalize = continuum(spectrum[0], spectrum[1], np.abs(spectrum[3]/spectrum[2]), knots_bin = knots_bin,
                              thres=line_thres, apodization_size=apodization_size)

        # absorption tempaltes
        processed_fluxes1, i_fit1 = process_spectrum(spectrum[0], sn_array=spectrum[1]/spectrum[2], normalized_fluxes = normalize.normalized_fluxes, min_sn=min_sn, pscale = self.pscale,
                                                 temp_type = 1, hcutoff_scale=hcutoff_scale, mask=mask, abs_thres=abs_thres, emi_thres = emi_thres)
        template_names1 = list(self.templates1.keys())
        n_templates1 = len(template_names1)
        z1, zerr1, r1, chi_eff1, flag1 = np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1, dtype=int)
        for i, temp_name in enumerate(template_names1):
            cc_spec_temp = cc_result(spectrum, normalize=normalize, processed_fluxes=processed_fluxes1, i_fit=i_fit1,
                                     template=self.templates1[temp_name], shifted_template=self.shifted_templates1[temp_name],
                                     temp_apodization_size=self.temp_apodization_size,
                                     temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, z_range=self.z_range,
                                     abs_thres=abs_thres, emi_thres=emi_thres)
            z1[i], zerr1[i], r1[i], chi_eff1[i], flag1[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff, cc_spec_temp.result 
                
        # remove the results with nan-redshift
        if chi_thres:
            nan_check = (~np.isnan(zerr1))&(~np.isnan(chi_eff1))&(chi_eff1<chi_thres)
        else:
            nan_check = (~np.isnan(zerr1))&(~np.isnan(chi_eff1))
        template_names1, z1, zerr1, r1, chi_eff1, flag1 = np.array(template_names1)[nan_check], z1[nan_check], zerr1[nan_check], r1[nan_check], chi_eff1[nan_check], flag1[nan_check]
        
        # best result among absorption templates
        if len(r1):
            i_best1 = np.nanargmax(r1)
            best_templates_name1, best_z1, best_zerr1, best_r1, best_chi_eff1, best_flag1 = template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1], flag1[i_best1]
        else:
            best_r1 = 0
            
        
        # emission tempaltes
        processed_fluxes2, i_fit2 = process_spectrum(spectrum[0],sn_array=spectrum[1]/spectrum[2], normalized_fluxes = normalize.normalized_fluxes, min_sn=min_sn, pscale = self.pscale,
                                                 temp_type = 2, hcutoff_scale=hcutoff_scale, mask=mask, abs_thres=abs_thres, emi_thres = emi_thres)
        template_names2 = list(self.templates2.keys())
        n_templates2 = len(template_names2)
        z2, zerr2, r2, chi_eff2, flag2 = np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2, dtype=int)
        for i, temp_name in enumerate(template_names2):
            cc_spec_temp = cc_result(spectrum, normalize=normalize, processed_fluxes=processed_fluxes2, i_fit=i_fit2, 
                                     template=self.templates2[temp_name], shifted_template=self.shifted_templates2[temp_name],
                                     temp_apodization_size=self.temp_apodization_size,
                                     temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, z_range=self.z_range,
                                     abs_thres=abs_thres, emi_thres=emi_thres)
        # return output
            z2[i], zerr2[i], r2[i], chi_eff2[i], flag2[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff, cc_spec_temp.result 
                
        # remove the results with nan-redshift
        nan_check = (~np.isnan(zerr2))&(~np.isnan(chi_eff2))
        template_names2, z2, zerr2, r2, chi_eff2, flag2 = np.array(template_names2)[nan_check], z2[nan_check], zerr2[nan_check], r2[nan_check], chi_eff2[nan_check], flag2[nan_check]

        # best result among absorption templates
        if len(r2):
            i_best2 = np.nanargmax(r2)
            best_templates_name2, best_z2, best_zerr2, best_r2, best_chi_eff2, best_flag2 = template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2], flag2[i_best2]     
        else:
            best_r2 = 0

        # choose the best result
        if best_r1 > r_abs:
            if best_r2 > r_em:
                if np.abs(best_z1-best_z2)/np.sqrt(best_zerr1**2+best_zerr2**2) > 3:
                    i_best = i_best1
                    best = (best_templates_name1, best_z1, best_zerr1, best_r1, best_chi_eff1, best_flag1)
                else:
                    i_best = i_best2 + len(r1)
                    best = (best_templates_name2, best_z2, best_zerr2, best_r2, best_chi_eff2, best_flag2)
            else:
                i_best = i_best1
                best = (best_templates_name1, best_z1, best_zerr1, best_r1, best_chi_eff1, best_flag1)
        else:
            if best_r2 > r_em:
                i_best = i_best2 + len(r1)
                best = (best_templates_name2, best_z2, best_zerr2, best_r2, best_chi_eff2, best_flag2)
            else:
                i_best = None
                best = ('No template', -9,-9,-9,-9,99)
        
        if output=='best':
            result = best
        
        if output=='all':
            # concatenate the results from absorption and emission templates
            template_names, z, zerr, r, chi_eff, flag = np.concatenate([template_names1, template_names2]), np.concatenate([z1, z2]), np.concatenate([zerr1, zerr2]), np.concatenate([r1,r2]), np.concatenate([chi_eff1, chi_eff2]), np.concatenate([flag1, flag2])
            if i_best != None:
                flag[i_best] = 0
            # arange the value in the order of chi_eff
            order = np.flip(np.argsort(r))
            template_names, z, zerr, r, chi_eff, flag,  = template_names[order], z[order], zerr[order], r[order], chi_eff[order], flag[order]
            
            table = np.vstack((template_names, z, zerr, r, chi_eff, flag))
            column_names = ['template_name', 'z', 'zerr', 'r', 'chi_eff', 'flag']
            result = pd.DataFrame(table.T, columns = column_names)
            result = result.astype({'template_name':str, 'z':np.float128, 'zerr':np.float128, 'r':np.float64, 'chi_eff':np.float64, 'flag':int})
            
            
                    
        
        return result
    
    def cc_analysis(self, spectrum, temp_name, spectrum_range=None, min_sn=3, hcutoff_scale=0, mask=None, knots_bin=100,
                    line_thres=3, abs_thres=2, emi_thres=2, apodization_size=0.05, window_continuum=10, sn_continuum=3):
        if type(temp_name) == str:
            temp_name = temp_name
        else:
            z_single_result = self.z_single(spectrum, output='all', spectrum_range=None, min_sn=2, hcutoff_scale=0, chi_thres=2, mask=None, r_abs=2, r_em=10, 
                 knots_bin=200, line_thres=3, abs_thres=2, emi_thres=2, apodization_size=0.05, window_continuum=40, sn_continuum=2)
            temp_name = z_single_result['template_name'][temp_name]
        
        spectrum = copy.deepcopy(spectrum)
        self.pscale = np.median(spectrum[0,1:]-spectrum[0,:-1])
        if type(spectrum_range)==type([]) or type(spectrum_range)==type(np.array([])):
            if len(np.where((spectrum[0,:]>spectrum_range[0])&(spectrum[0,:]<spectrum_range[1]))[0]) ==0:
                raise ValueError('spectrum_range should contain spectrum wavelengths')
            spectrum = spectrum[:,(spectrum[0,:]>spectrum_range[0])&(spectrum[0,:]<spectrum_range[1])]
            
        if type(mask)==type([]) or type(mask)==type(np.array([])):
            for i in range(len(mask)):
                left_end = abs(spectrum[0,:]- mask[i][0]).argmin()
                right_end = abs(spectrum[0,:] - mask[i][1]).argmin()
                spectrum[3,:][left_end:right_end+1] = 0
            
        elif mask!=None:
            raise TypeError('Type of the maks must be list or 2D numpy array')
        
        self.cspectrum = clean_spectrum(spectrum, window_continuum, sn_continuum)
        self.norm = continuum(self.cspectrum[0], self.cspectrum[1], np.abs(self.cspectrum[3]/self.cspectrum[2]),
                              knots_bin=knots_bin, thres=line_thres, apodization_size=apodization_size)
        spectrum[3,:][spectrum[1,:]/spectrum[2,:]<min_sn] = 0
        self.processed_fluxes, self.i_fit = process_spectrum(self.cspectrum[0], self.cspectrum[1]/self.cspectrum[2], self.norm.normalized_fluxes, min_sn, self.pscale, self.templates[temp_name][2], hcutoff_scale, mask, abs_thres, emi_thres)
        # self.norm = continuum_rm(self.cspectrum[0], self.cspectrum[1], np.abs(1/self.cspectrum[2]))

        self.cc_result = cc_result(self.cspectrum, normalize = self.norm, processed_fluxes=self.processed_fluxes, i_fit=self.i_fit,
                                   template = self.templates[temp_name], shifted_template = self.shifted_templates[temp_name],
                                   temp_apodization_size = self.temp_apodization_size, temp_knots_bin = self.temp_knots_bin, temp_line_thres = self.temp_line_thres,
                                   z_range = self.z_range, abs_thres = abs_thres, emi_thres = emi_thres)
         
        
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
                with open(save_folder+'/z_result%d.txt'%subset_number, 'w') as file:
                    file.write('number best_template z zerr chi_eff flag\n')
                    if subset_number == len(spec_subset)-1:
                        for index in tqdm(indices):
                            result = self.z_single(spectrums[index], output='best', **kwargs)
                            file.write('%d %s %f %f %f %f %d\n'%(index, result[0], result[1],
                                                        result[2], result[3], result[4], result[5]))
                    else:
                        for index in tqdm(indices):
                            result = self.z_single(spectrums[index], output='best', **kwargs)
                            file.write('%d %s %f %f %f %f %d\n'%(index, result[0], result[1],
                                                        result[2], result[3], result[4], result[5]))
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
                    result = self.z_single(spectrums[index], output='best', **kwargs)
                    file.write('%d %s %f %f %f %f %d\n'%(index, result[0], result[1],
                                                        result[2], result[3], result[4], result[5]))






badlines = [[4030.0, 4070.0],
            [4333.0, 4383.0],
            [5436.0, 5496.0],
            [5542.0, 5612.0],
            [6275.0, 6325.0],
            [6339.0, 6389.0]]

# run = rvm(model_templates, z_range=[0,1])
# run.z_multi(dr16_spec, directory='z_model', multi_process=8, min_sn=2, hcutoff_scale=3)
# print('z_model is done')

# run = rvm(model_templates, z_range=[0,1])
# run.z_multi(hecto_spec, directory='z_model_hecto', mask=badlines, multi_process=8, min_sn=3, hcutoff_scale=3)
# print('z_model_hecto is done')

# run = rvm(sdss_template_galaxy, z_range=[0,2])
# run.z_multi(dr16_spec, directory='z_sdss', multi_process=8, min_sn=3, hcutoff_scale=3)
# print('z_sdss is done')


run = rvm(abs_templates, z_range=[0,1])
run.z_multi(dr16_spec, directory='z_abs', multi_process=8,
            knots_bin=100, min_sn=1, chi_thres=2, hcutoff_scale=3, window_continuum=100, sn_continuum=1.,)
print('z_abs is done')

# run = rvm(emi_templates, z_range=[0,1])
# run.z_multi(dr16_spec, directory='z_emi', multi_process=8, min_sn=3, hcutoff_scale=3)
# print('z_emi is done')

run = rvm(abs_templates, z_range=[0,1])
run.z_multi(hecto_spec, directory='z_abs_hecto', multi_process=8,
            knots_bin=100, min_sn=1, chi_thres=2, hcutoff_scale=3, window_continuum=100, sn_continuum=1., mask=badlines)
print('z_abs_hecto is done')

# run = rvm(emi_templates, z_range=[0,1])
# run.z_multi(hecto_spec, directory='z_emi_hecto', mask=badlines, multi_process=8, min_sn=3, hcutoff_scale=3)
# print('z_emi_hecto is done')

# run = rvm(sdss_template_galaxy, z_range=[0,1])
# run.z_multi(hecto_spec, directory='z_sdss_hecto', mask=badlines, multi_process=8, min_sn=3, hcutoff_scale=3)
# print('z_model is done')

# from RVSNUpy.template import mmt_template_galaxy
# badlines = [[4030.0, 4070.0],
#             [4333.0, 4383.0],
#             [5436.0, 5496.0],
#             [5542.0, 5612.0],
#             [6275.0, 6325.0],
#             [6339.0, 6389.0]]
# rvn = rvm(mmt_template_galaxy, z_range=[-0.001,2])
# rvn.z_multi(hecto_spec, directory='z_hecto', mask=badlines, multi_process=8, min_sn=2)
