from scipy.optimize import curve_fit

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sigmaclip
import copy
from scipy.interpolate import splrep, splev
from scipy.signal.windows import tukey
import warnings
import pandas as pd

class process_template:
    warnings.simplefilter('ignore')
    def __init__(self, wavelengths, fluxes, weights, line_identified=False, resolution=3, knots_bin = 100, thres=3, apodization_size = 0.05):
        if line_identified:
            self.line_identified_trace_continuum(wavelengths, fluxes, weights, knots_bin = knots_bin, thres = thres)
        else:
            self.trace_continuum(wavelengths,fluxes, weights, resolution=resolution, knots_bin = knots_bin, thres = thres)
        self.normalized_fluxes = ((fluxes/self.continuum_fluxes)-1)*tukey(len(fluxes), apodization_size)
        self.normalized_fluxes[np.isnan(self.normalized_fluxes)] = 0
        
    def line_identified_trace_continuum(self, wavelengths, fluxes, weights, knots_bin = 200, thres=3):
        self.knots = np.arange(wavelengths[0]+1e-10, wavelengths[-1]+1e-10, knots_bin)
        concat_edge=[0]
        while len(concat_edge) > 0:
            mask_ratio = np.histogram(wavelengths[weights==0], bins=self.knots)[0]/np.histogram(wavelengths, bins=self.knots)[0]
            concat_edge = np.where(mask_ratio>0.25)[0]+1
            self.knots = np.delete(self.knots, concat_edge)
        
        self.sp_param = splrep(wavelengths, fluxes, t=self.knots, k=5, w = weights)  # k is the degree of the spline
    
        self.continuum_fluxes = splev(wavelengths, self.sp_param)
        self.continuum_fluxes[self.continuum_fluxes==0] = 1e+5*np.max(np.abs(fluxes))
        self.continuum_fluxes[0], self.continuum_fluxes[-1] = self.continuum_fluxes[1], self.continuum_fluxes[-2]
        
    
    def trace_continuum(self, wavelengths, fluxes, weights, resolution, knots_bin = 200, thres=3):
        self.knots = np.arange(wavelengths[0]+1e-10, wavelengths[-1]+1e-10, knots_bin)
        
        sp_param = splrep(wavelengths, fluxes, t=self.knots, w=np.ones_like(fluxes), k=5)  # k is the degree of the spline

        self.continuum_fluxes = splev(wavelengths, sp_param)
        self.new_weights = copy.deepcopy(weights)
        self.new_masks = copy.deepcopy(weights)
        self.new_masks[self.new_masks!=0] = 1
        self.lines = []

        if thres:
            res = fluxes-self.continuum_fluxes
            std = np.nanstd(res)
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
                for rend, lend, center in zip(rends, lends, centers):
                    if rend-lend+1>3:
                        try:
                            line_weights = weights[lend:rend+1]
                            line_wavelengths = wavelengths[lend:rend+1]
                            line_res = res[lend:rend+1]
                            line_weights = 1/line_weights[line_weights!=0]
                            popt, _ = curve_fit(gaussian, line_wavelengths, line_res,
                                                p0 = [res[center], wavelengths[center], (wavelengths[rend]-wavelengths[lend]+2)*0.5])
                            stddev = np.abs(popt[2])
                            width = 2*int(stddev)
                        except:
                            continue
                        if stddev*2*np.sqrt(2*np.log(2))>resolution:
                            
                            self.new_weights[max(0,center-width):min(len(res),center+width+1)]=0
                            self.lines.append([wavelengths[max(0,center-width)], wavelengths[min(len(res),center+width+1)]])
            
            concat_edge=[0]
            while len(concat_edge) > 0:
                mask_ratio = np.histogram(wavelengths[self.new_weights==0], bins=self.knots)[0]/np.histogram(wavelengths, bins=self.knots)[0]
                concat_edge = np.where(mask_ratio>0.25)[0]+1
                self.knots = np.delete(self.knots, concat_edge)
            
            self.sp_param = splrep(wavelengths, fluxes, t=self.knots, k=5, w = self.new_weights)  # k is the degree of the spline
        
            self.continuum_fluxes = splev(wavelengths, self.sp_param)
            self.continuum_fluxes[self.continuum_fluxes==0] = 1e+5*np.max(np.abs(fluxes))
            self.continuum_fluxes[0], self.continuum_fluxes[-1] = self.continuum_fluxes[1], self.continuum_fluxes[-2]

from astropy.modeling import models, fitting

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sigmaclip
import copy
from scipy.interpolate import splrep, splev
from scipy.signal.windows import tukey
import warnings
import pandas as pd

class process_spectrum:
    warnings.simplefilter('ignore')
    def __init__(self, wavelengths, fluxes, weights, temp_type, resolution=3, knots_bin = 100, thres=3, apodization_size = 0.05):
        self.wavelengths, self.fluxes, self.weights, self.temp_type = wavelengths, fluxes, weights, temp_type
        self.resolution, self.knots_bin, self.thres = resolution, knots_bin, thres
        self.trace_continuum()
        self.gen_mask()
        self.normalized_fluxes = ((fluxes/self.continuum_fluxes)-1)*tukey(len(fluxes), apodization_size)
        self.normalized_fluxes[np.isnan(self.normalized_fluxes)] = 0
        
    
    def trace_continuum(self):
        self.knots = np.arange(self.wavelengths[0]+1e-10, self.wavelengths[-1]+1e-10, self.knots_bin)
        
        sp_param = splrep(self.wavelengths, self.fluxes, t=self.knots, w=np.ones_like(self.fluxes), k=5)  # k is the degree of the spline

        self.continuum_fluxes = splev(self.wavelengths, sp_param)
        self.new_weights = copy.deepcopy(self.weights)

        if self.thres:
            res = self.fluxes-self.continuum_fluxes
            std = np.nanstd(res)
            pscale = np.median(self.wavelengths[1:]-self.wavelengths[:-1])
            detection = (np.abs(res)>self.thres*std)

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
                for rend, lend, center in zip(rends, lends, centers):
                    if rend-lend+1>3:
                        try:
                            line_weights = self.weights[lend:rend+1]
                            line_wavelengths = self.wavelengths[lend:rend+1][line_weights!=0]
                            line_res = res[lend:rend+1][line_weights!=0]
                            line_weights = 1/line_weights[line_weights!=0]
                            popt, _ = curve_fit(gaussian, line_wavelengths, line_res, sigma=line_weights,
                                                p0 = [res[center], self.wavelengths[center], (self.wavelengths[rend]-self.wavelengths[lend]+2)*0.5])
                            stddev = np.abs(popt[2])
                            width = 2*int(stddev)
                        except:
                            continue
                        if stddev*2*np.sqrt(2*np.log(2))>self.resolution:
                            self.new_weights[max(0,center-width):min(len(res),center+width+1)]=0
                            # if temp_type==1:
                            #     if fit_lines.amplitude.value>0:
                            #         self.new_masks[max(0,center-width):min(len(res),center+width+1)]=0
                            # elif temp_type==2:
                            #     if fit_lines.amplitude.value<0:
                            #         self.new_masks[max(0,center-width):min(len(res),center+width+1)]=0
            
            concat_edge=[0]
            while len(concat_edge) > 0:
                mask_ratio = np.histogram(self.wavelengths[self.new_weights==0], bins=self.knots)[0]/np.histogram(self.wavelengths, bins=self.knots)[0]
                concat_edge = np.where(mask_ratio>0.25)[0]+1
                self.knots = np.delete(self.knots, concat_edge)
            
            self.sp_param = splrep(self.wavelengths, self.fluxes, t=self.knots, k=5, w = self.new_weights)  # k is the degree of the spline
        
            self.continuum_fluxes = splev(self.wavelengths, self.sp_param)
            self.continuum_fluxes[self.continuum_fluxes==0] = 1e+5*np.max(np.abs(self.fluxes))
            self.continuum_fluxes[0], self.continuum_fluxes[-1] = self.continuum_fluxes[1], self.continuum_fluxes[-2]
            
    def gen_mask(self):
        self.new_masks = copy.deepcopy(self.weights)
        self.new_masks[self.new_masks!=0] = 1
        if self.thres:
            res = self.fluxes-self.continuum_fluxes
            std = np.nanstd(res)
            pscale = np.median(self.wavelengths[1:]-self.wavelengths[:-1])
            detection = (np.abs(res)>self.thres*std)
            
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
                for rend, lend, center in zip(rends, lends, centers):
                    if rend-lend+1>3:
                        try:
                            line_weights = self.weights[lend:rend+1]
                            line_wavelengths = self.wavelengths[lend:rend+1][line_weights!=0]
                            line_res = res[lend:rend+1][line_weights!=0]
                            line_weights = 1/line_weights[line_weights!=0]
                            popt, _ = curve_fit(gaussian, line_wavelengths, line_res, sigma=line_weights,
                                                p0 = [res[center], self.wavelengths[center], (self.wavelengths[rend]-self.wavelengths[lend]+2)*0.5],
                                                bounds = [[-np.inf,0,0], [np.inf, np.inf, np.inf]])
                            stddev = np.abs(popt[2])
                            width=2*int(stddev)
                        except:
                            continue
                        if stddev*2*np.sqrt(2*np.log(2))>self.resolution:
                            if self.temp_type==1:
                                if popt[0]>0:
                                    self.new_masks[max(0,center-width):min(len(res),center+width+1)]=0
                            elif self.temp_type==2:
                                if popt[0]<0: 
                                    self.new_masks[max(0,center-width):min(len(res),center+width+1)]=0
                            

        
        
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


def shift_templates(templates, z_range=[-0.1,2], apodization_size=0.05, knots_bin = 100, thres=3, normalziation=True):
    shifted_templates = {}
    for temp_name in templates.keys():
        temp = templates[temp_name][0]
        if normalziation:
            normalize=process_template(temp[0], temp[1], np.ones_like(temp[1]), knots_bin=knots_bin, thres=thres, apodization_size=apodization_size)
            
        # log pixel scale
        log_wavelengths = np.log10(temp[0])
        log_bin = np.median(log_wavelengths[1:]-log_wavelengths[:-1])

        # prepare a wavelength array
        n_left, n_right = -int(np.log10(z_range[0]-0.1*(1+z_range[0])+1)/log_bin), int(np.log10(z_range[1]+0.1*(1+z_range[1])+1)/log_bin)
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
        if (np.median(spec[1,i_left:i_left+window]/spec[2,i_left:i_left+window]) > sn) & (np.sum(spec[3,i_left:i_left+window])>window*0.9):
            break
        
    for i_right in range(spec.shape[1], 0, -window):
        if (np.median(spec[1,i_right-window:i_right]/spec[2,i_right-window:i_right]) > sn) & (np.sum(spec[3,i_right-window:i_right])>window*0.9):
            break
    
    if i_left+10 < i_right:
        cspec = copy.deepcopy(spec[:,i_left:i_right])
    else:
        cspec = copy.deepcopy(spec)
        cspec[1,:] = 0
        
    return cspec

from astropy.modeling import models, fitting
import warnings

def return_nan(x):
    return np.nan

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting, Fittable1DModel, Parameter
    
class RedshiftedGaussian(Fittable1DModel):
    redshift = Parameter(default=0.01)
    amplitude = Parameter(default=1.0)
    stddev = Parameter(default=1.0)
    
    def __init__(self, rest_wavelength, **kwargs):
        self.rest_wavelength = rest_wavelength  # Fixed rest-frame wavelength for each line
        super().__init__(**kwargs)

    def evaluate(self, x, redshift, amplitude, stddev):
        shifted_mean = self.rest_wavelength * (1 + redshift)  # Redshifted mean wavelength
        return amplitude * np.exp(-0.5 * ((x - shifted_mean) / stddev) ** 2)


def fit_multiline(wavelengths, fluxes, weights, lines, z, zerr, resolution):
    nlines, ndetect = 0,0
    model_lines0 = []
    
    # Wavelength range of the input spectrum
    observed_wavelength_min, observed_wavelength_max = wavelengths[0], wavelengths[-1]
    z_min, z_max = z-3*zerr, z+3*zerr
    
    for line in lines:
        if isinstance(line, float):
            # Redshifted wavelength
            redshifted_wavelength = line * (1 + z)
            
            # Check if the redshifted line is within the observed wavelength range
            if observed_wavelength_min <= redshifted_wavelength <= observed_wavelength_max:
                # Single line within range
                model_line0 = RedshiftedGaussian(rest_wavelength=line,
                                                 amplitude=fluxes[np.argmin(np.abs(wavelengths - redshifted_wavelength))],
                                                 redshift=z, stddev=resolution)
                model_line0.amplitude.bounds = (0, None)
                # model_line0.redshift.bounds = (z_min, z_max)
                model_lines0.append(model_line0)
                nlines += 1
            
        elif isinstance(line, list):
            # Handle lines with amplitude ratio constraint
            wavelengths_list, ratio_list = line
            ratio_reference = ratio_list[0]  # The first ratio should be 1 (reference line)
            
            valid_lines = []
            
            for j, wl in enumerate(wavelengths_list):
                redshifted_wavelength = wl * (1 + z)
                if observed_wavelength_min <= redshifted_wavelength <= observed_wavelength_max:
                    valid_lines.append((wl, ratio_list[j]))  # Store valid lines and their ratios
                    nlines += 1

            if valid_lines:
                # If at least one valid line exists, create a Gaussian model
                # First line (reference) for this group
                first_wl, _ = valid_lines[0]
                model_line0 = RedshiftedGaussian(rest_wavelength=first_wl,
                                                 amplitude=fluxes[np.argmin(np.abs(wavelengths - (1 + z) * first_wl))],
                                                 redshift=z, stddev=resolution)
                # model_line0.redshift.bounds = (z_min, z_max)
                model_line0.amplitude.bounds = (0, None)
                model_lines0.append(model_line0)
                
                # Additional lines with tied amplitudes
                for wl, ratio in valid_lines[1:]:
                    model_line_ratio = RedshiftedGaussian(rest_wavelength=wl,
                                                          amplitude=fluxes[np.argmin(np.abs(wavelengths - (1 + z) * wl))],
                                                          redshift=z, stddev=resolution)
                    model_line_ratio.amplitude.tied = lambda m: ratio * m.amplitude_0  # Tie amplitude
                    # model_line_ratio.redshift.bounds = (z_min, z_max)
                    model_lines0.append(model_line_ratio)

    if not model_lines0:
        return 0, 0

    # Combine the models into a single compound model
    multi_lines0 = model_lines0[0]
    
    # Constrain to have similar standard deviations
    for i, model_line0 in enumerate(model_lines0[1:]):
        f = 3 * model_lines0[0].rest_wavelength / model_line0.rest_wavelength
        model_line0.stddev.bounds = (model_lines0[0].stddev.value / f, model_lines0[0].stddev.value * f)
        model_line0.redshift.tied = lambda model: model.redshift_0
        multi_lines0 += model_line0
    
    fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)
    multi_lines = fitter(multi_lines0, wavelengths, fluxes, weights=weights)
    chi_eff = np.sum((multi_lines(wavelengths)-fluxes) ** 2 * weights ** 2) / (len(wavelengths) - 2 * nlines - 1)
    
    for i, amp_name in enumerate(multi_lines.param_names):
        if 'amplitude' in amp_name and getattr(multi_lines, amp_name).value > 0:
            ndetect += 1
    
    fdetect = ndetect / nlines
    
    if 'redshift' in multi_lines.param_names:
        z_line = multi_lines.redshift.value
    else:
        z_line = multi_lines.redshift_0.value
    
    
    return ndetect, fdetect, chi_eff, z_line

class cc_result:
    def __init__(self, spectrum, normalize, processed_fluxes, template, shifted_template,
                 temp_apodization_size=0.05, temp_knots_bin=100, temp_line_thres=3, weight=True,
                 z_range=[-0.01,2],
                 em_lines=[3727.30, 4861.33, 4958.91, 5006.84, [[6548.06, 6562.82, 6583.57], [1,10,5]], 6716.440, 6730.815],
                 resolution=3):
        warnings.simplefilter('ignore')
        self.spectrum, self.normalize, self.processed_fluxes = spectrum, normalize, processed_fluxes
        self.template, self.shifted_template  = template, shifted_template
        self.temp_apodization_size, self.temp_knots_bin, self.temp_line_thres = temp_apodization_size, temp_knots_bin, temp_line_thres
        self.z_range = z_range
        self.em_lines, self.resolution = em_lines, resolution
        self.shifted_vels, self.shifted_wavelengths, self.shifted_fluxes, self.template_spectrum = self.shifted_template[0], self.shifted_template[1], self.shifted_template[2], self.template[0]
        # self.cross_correlate(weight)
        try:
            self.cross_correlate(weight)
        except:
            self.z, self.zerr, self.r, self.pkratio, self.chi_eff  = np.nan, np.nan, np.nan, 99, np.nan
     
        
    def cross_correlate0(self, weight=True):
        if weight:
            overlap_wavelengths = self.shifted_wavelengths[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])]
            self.new_fluxes0, self.new_weights0, self.new_masks0 = np.zeros_like(self.shifted_wavelengths), np.ones_like(self.shifted_wavelengths), np.ones_like(self.shifted_wavelengths)
            overlap_fluxes = resampler(self.spectrum[0], self.processed_fluxes, overlap_wavelengths)
            overlap_weights = resampler(self.spectrum[0], np.abs(self.normalize.continuum_fluxes/self.spectrum[2]), overlap_wavelengths)
            overlap_masks = discrete_resampler(self.spectrum[0], self.spectrum[3], overlap_wavelengths)

            self.new_fluxes0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_fluxes
            self.new_weights0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_weights
            self.new_masks0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_masks

            self.cc0 = np.matmul(self.shifted_fluxes, self.new_masks0*self.new_weights0**2*self.new_fluxes0)
        else:
            overlap_wavelengths = self.shifted_wavelengths[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])]
            self.new_fluxes0, self.new_masks0 = np.zeros_like(self.shifted_wavelengths), np.ones_like(self.shifted_wavelengths)
            overlap_fluxes = resampler(self.spectrum[0], self.processed_fluxes, overlap_wavelengths)
            overlap_masks = discrete_resampler(self.spectrum[0], self.spectrum[3], overlap_wavelengths)

            self.new_fluxes0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_fluxes
            self.new_masks0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_masks

            self.cc0 = np.matmul(self.shifted_fluxes, self.new_masks0*self.new_weights0*self.new_fluxes0)


        self.cz0 = self.shifted_vels[np.nanargmax(self.cc0)]

    def find_peak_region(self):
        cz_range = np.array(self.z_range)*c
        cc_inrange = self.cc0[(self.shifted_vels>cz_range[0])&(self.shifted_vels<cz_range[1])]
        lags_inrange = self.shifted_vels[(self.shifted_vels>cz_range[0])&(self.shifted_vels<cz_range[1])]
        
        i_max = np.nanargmax(cc_inrange)
        cc_max, lags_max = cc_inrange[i_max], lags_inrange[i_max] # estimates a peak
        
        # find peaks
        detection = (self.cc0 >= 0.5*cc_max) | (self.cc0 >= 3*np.std(self.cc0))
        lends, rends = np.where(np.diff(detection.astype(int)) == 1)[0] + 1, np.where(np.diff(detection.astype(int)) == -1)[0] + 1
        if detection[-1] == True: # dectecion = (..., True, True, True)
            rends = np.concatenate((rends, np.array([len(self.cc0)-1])))
        if detection[0] == True: # dectecion = (True, True, True, ...)
            lends = np.concatenate((np.array([0]), lends))
        rends -= 1
        
        centers = (lends+rends) // 2

        self.peak_ranges = []
        self.pkratio = 0
        n = 0
        for i_peak in np.argsort(self.cc0[centers])[-1::-1]:
            if n>5:
                break
            v_pcenter, v_pwidth = self.shifted_vels[centers[i_peak]], 0.5*(self.shifted_vels[rends[i_peak]]-self.shifted_vels[lends[i_peak]])
            lv_pcenter, rv_pcenter = v_pcenter-3*v_pwidth, v_pcenter+3*v_pwidth
            cc_peak, lags_peak = self.cc0[(self.shifted_vels>lv_pcenter)&(self.shifted_vels<rv_pcenter)], self.shifted_vels[(self.shifted_vels>lv_pcenter)&(self.shifted_vels<rv_pcenter)]
            try:
                popt, pcov = curve_fit(gaussian, lags_peak, cc_peak,
                                       p0 = [self.cc0[centers[i_peak]], self.shifted_vels[centers[i_peak]], self.shifted_vels[centers[i_peak]]-self.shifted_vels[lends[i_peak]-1]],
                                       bounds = ([-np.inf,0,0], [np.inf, np.inf, np.inf]))
                prange = [popt[1]-2*popt[2], popt[1]+2*popt[2]]
            except:
                continue
                
            if (popt[0] < 0) | (popt[1] < cz_range[0]) | (popt[1]>cz_range[1]) | (len(np.where((self.shifted_vels>prange[0])&(self.shifted_vels<prange[1]))[0])<1):
                    continue


            self.peak_ranges.append(prange)
            n +=1
        
            
    def cc_near_peak(self):
        left_new_wavelengths, right_new_wavelengths = self.shifted_wavelengths[self.shifted_wavelengths<self.spectrum[0,0]], self.shifted_wavelengths[self.shifted_wavelengths>self.spectrum[0,-1]]
        self.new_wavelengths = np.concatenate([left_new_wavelengths, self.spectrum[0], right_new_wavelengths])

        self.new_fluxes, self.new_weights, self.new_masks, self.new_fit_weights = np.zeros_like(self.new_wavelengths), np.ones_like(self.new_wavelengths), np.ones_like(self.new_wavelengths), np.ones_like(self.new_wavelengths)
        self.new_fluxes[(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])] = self.processed_fluxes
        self.new_fit_weights[(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])] = np.abs(self.spectrum[3]/self.spectrum[2])
        self.new_weights[(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])] = np.abs(self.normalize.continuum_fluxes/self.spectrum[2])
        self.new_masks[(self.new_wavelengths>=self.spectrum[0,0])&(self.new_wavelengths<=self.spectrum[0,-1])] = self.spectrum[3]
        

        self.cc = copy.deepcopy(self.cc0)
        max_peaks, max_rs = np.zeros(len(self.peak_ranges)), np.zeros(len(self.peak_ranges))
        for m, prange in enumerate(self.peak_ranges):
            peak_region = np.where((self.shifted_vels>prange[0])&(self.shifted_vels<prange[1]))[0]
            interp_fluxes = np.zeros((len(peak_region),len(self.new_fluxes)))
            
            ########################################################################################################
            i = int(np.median(peak_region))
            region = ((self.new_wavelengths>=self.shifted_wavelengths[i])&(self.new_wavelengths<=self.shifted_wavelengths[i+self.template_spectrum.shape[1]-1]))
            new_template_wavelengths = self.new_wavelengths[region]
            resampled_template_fluxes = resampler(self.shifted_wavelengths[i:i+self.template_spectrum.shape[1]], self.template_spectrum[1], new_template_wavelengths)
            template_lines = process_template(new_template_wavelengths, resampled_template_fluxes, self.new_fit_weights[region], knots_bin = self.temp_knots_bin,
                                                    thres = self.temp_line_thres, apodization_size=self.temp_apodization_size).lines
            template_lines0 = []
            for line in template_lines:
                template_lines0.append([line[0]/(1+self.shifted_vels[i]/c), line[1]/(1+self.shifted_vels[i]/c)])
            ########################################################################################################
            
            for n, i in enumerate(peak_region):
                region = ((self.new_wavelengths>=self.shifted_wavelengths[i])&(self.new_wavelengths<=self.shifted_wavelengths[i+self.template_spectrum.shape[1]-1]))
                new_template_wavelengths = self.new_wavelengths[region]
                resampled_template_fluxes = resampler(self.shifted_wavelengths[i:i+self.template_spectrum.shape[1]], self.template_spectrum[1], new_template_wavelengths)
                new_fit_weights_at_v = copy.deepcopy(self.new_fit_weights[region])
                for line in template_lines0:
                    new_fit_weights_at_v[(new_template_wavelengths>line[0]*(1+self.shifted_vels[i]/c))&(new_template_wavelengths<line[1]*(1+self.shifted_vels[i]/c))] = 0
                interp_fluxes[n,region] = process_template(new_template_wavelengths, resampled_template_fluxes, new_fit_weights_at_v, line_identified=True, knots_bin = self.temp_knots_bin,
                                                        thres = self.temp_line_thres, apodization_size=self.temp_apodization_size).normalized_fluxes
                # self.interp_fluxes[n,region] = process_template(new_template_wavelengths, resampled_template_fluxes, self.new_fit_weights[region],  knots_bin = self.temp_knots_bin,
                #                                         thres = self.temp_line_thres, apodization_size=self.temp_apodization_size).normalized_fluxes
            weight_cc = np.matmul(interp_fluxes,self.new_masks*self.new_weights**2*self.new_fluxes)
            max_peaks[m] = np.nanmax(weight_cc)
            self.cc[peak_region] = weight_cc
        
        peak_order = np.argsort(max_peaks)
        self.n_peakmax = peak_order[-1]
        if len(peak_order)>1:
            self.pkratio = max_peaks[peak_order][-2]/max_peaks[peak_order][-1]
        else:
            self.pkratio = 0
    
    def cross_correlate(self, weight=True):
        self.cross_correlate0(weight)
        self.find_peak_region()
        if self.pkratio != 9:
            self.cc_near_peak()
            self.z_finding()
        else:
            self.cc = self.cc0

    def z_finding(self):
        fit_params, fit_covs= [],[]
        for i in range(len(self.peak_ranges)):
            self.max_peak_range = self.peak_ranges[i]
            cc_peak = self.cc[(self.shifted_vels>self.max_peak_range[0])&(self.shifted_vels<self.max_peak_range[1])]
            lags_peak = self.shifted_vels[(self.shifted_vels>self.max_peak_range[0])&(self.shifted_vels<self.max_peak_range[1])]
            nan_filtering = (~np.isnan(cc_peak))&((~np.isnan(lags_peak)))
            cc_peak, lags_peak = cc_peak[nan_filtering], lags_peak[nan_filtering]
            
            try:
                popt, pcov = curve_fit(gaussian, lags_peak, cc_peak,
                                        p0 = [np.nanmax(cc_peak), np.median(lags_peak), 0.5*(lags_peak[-1]-lags_peak[0])],
                                        bounds = ([-np.inf, 0, 0], [np.inf, np.inf, np.inf]))
                fit_params.append(popt)
                fit_covs.append(pcov)
            except:
                continue
                

                
        zs, zerrs, r_values, dr_values = np.zeros(len(fit_params)), np.zeros(len(fit_params)), np.zeros(len(fit_params)), np.zeros(len(fit_params))
        
        for i in range(len(zs)):
            # extract best fit parameters
            amp, mean, std = fit_params[i][0], fit_params[i][1], fit_params[i][2]
            damp, dmean, dstd = np.sqrt(fit_covs[i][0,0]), np.sqrt(fit_covs[i][1,1]), np.sqrt(fit_covs[i][2,2])
            # redshift
            zs[i]  = mean/c
            # redshift uncertainty
            zerr_ = std*np.sqrt(-2*np.log(1-0.5/amp))/c
            dzerr_ = np.sqrt((dmean+2*std*np.sqrt(-2/np.log(1-0.5/amp))*damp/(amp*(amp-0.5))+2*dstd*np.sqrt(-2*np.log(1-0.5/amp)))**2+dmean**2)/c
            zerrs[i] = zerr_+dzerr_
            # r-value
            n_peak = np.abs(self.shifted_vels-mean).argmin() # find an index of peak
            N = int((0.1*(c+mean))/(self.shifted_vels[n_peak]-self.shifted_vels[n_peak-1]))
            left, right = max(n_peak-N,0), min(n_peak+N, len(self.cc))
            nrange = int(min(n_peak-left, right-n_peak))
            cc_left, cc_right = self.cc[n_peak-nrange:n_peak], np.flip(self.cc[n_peak:n_peak+nrange])
            cc_left, cc_right = cc_left[(~np.isnan(cc_left))&(~np.isnan(cc_right))], cc_right[(~np.isnan(cc_left))&(~np.isnan(cc_right))]
            sigma = np.sum(((cc_left - cc_right)**2))/nrange
            r_values[i] = amp/(np.sqrt(sigma))
            dr_values[i] = damp/(np.sqrt(sigma))
        
        if (len(r_values) == 0) | (np.max(r_values) ==0) :
            self.z, self.zerr, self.r, self.chi_eff, self.pkratio  = np.nan, np.nan, np.nan, np.nan, 9
            
        elif r_values.shape[0] == 1:
            self.z, self.zerr, self.r = zs[0], zerrs[0], r_values[0]
            self.bestfit_gaussian, self.bestfitter = fit_params[0], fit_covs[0]
            self.cal_chi_eff()
    
        else:
            sort = np.argsort(r_values)[::-1]
            zs, zerrs, r_values, dr_values = zs[sort], zerrs[sort], r_values[sort], dr_values[sort]
            
            candidate = np.where((r_values[:-1]-r_values[1:])/np.sqrt(dr_values[:-1]**2+dr_values[1:])>3)[0]
            # if len(candidate) > 0:
            #     candidate = candidate[0]+1
            # else:
            #     candidate = zs.shape[0]
            # zs, zerrs, r_values, dr_values = zs[:candidate], zerrs[:candidate], r_values[:candidate], dr_values[:candidate]
            candidate = (r_values>5) & (r_values/r_values[0]>0.5)
            # zs, zerrs, r_values = zs, zerrs, r_values = zs[candidate], zerrs[candidate], r_values[candidate]
    
            if zs.shape[0] == 1:
                self.z, self.zerr, self.r = zs[0], zerrs[0], r_values[0]
                self.bestfit_gaussian, self.bestfitter = fit_params[0], fit_covs[0]
            elif self.template[1] == 2:
                # self.z, self.zerr, self.r = zs[0], zerrs[0], r_values[0]
                # self.bestfit_gaussian, self.bestfitter = gaussian_fits[sort[0]], fitters[sort[0]]
                fdetect, chi_eff, z_fit = np.zeros_like(zs), np.zeros_like(zs), np.zeros_like(zs)
                for i in range(len(zs)):
                    _, fdetect[i], chi_eff[i], z_fit[i] = fit_multiline(self.spectrum[0], self.processed_fluxes, 
                                                     self.spectrum[3]*self.normalize.continuum_fluxes/self.spectrum[2],
                                                     lines=self.em_lines, z=zs[i], zerr=zerrs[i], resolution=self.resolution)

                best_z = np.argmin(chi_eff)
                self.z, self.zerr, self.r = zs[best_z], zerrs[best_z], r_values[best_z]
            else:

                self.z, self.zerr, self.r = zs[0], zerrs[0], r_values[0]
                self.bestfit_gaussian, self.bestfitter = fit_params[0], fit_covs[0]

                
            self.cal_chi_eff()
                
            
    def cal_chi_eff(self):
        z_wavelength = self.template_spectrum[0]*(1+self.z)
        self.overlap_spec = copy.deepcopy(self.spectrum)
        # self.overlap_spec[1,:] = (self.processed_fluxes+1)*self.normalize.continuum_fluxes
        self.overlap_spec = self.spectrum[:,(self.spectrum[0,:]>max(z_wavelength[0], self.spectrum[0,0]))&
                                     (self.spectrum[0,:]<min(z_wavelength[-1], self.spectrum[0,-1]))]
        self.T = resampler(z_wavelength, self.template_spectrum[1], self.overlap_spec[0])
        self.T_continuum = process_template(self.overlap_spec[0], self.T, np.abs(self.overlap_spec[3]/self.overlap_spec[2]), knots_bin = self.temp_knots_bin,
                                                        thres = self.temp_line_thres, apodization_size=self.temp_apodization_size)
        self.T_ = self.T*self.normalize.continuum_fluxes[(self.spectrum[0,:]>max(z_wavelength[0], self.spectrum[0,0]))&(self.spectrum[0,:]<min(z_wavelength[-1], self.spectrum[0,-1]))]/self.T_continuum.continuum_fluxes
        self.chi_eff = np.sum(((self.overlap_spec[3]/self.overlap_spec[2])**2*(self.overlap_spec[1]-self.T_)**2))/(np.sum((self.overlap_spec[3]))-len(self.T_continuum.knots)-1)                     
        # self.chi_eff = np.sum(((self.overlap_spec[3]/self.overlap_spec[2])**2*(self.overlap_spec[1]-self.T_)**2))/(np.sum((self.overlap_spec[3]))-len(T_continuum.knots)-1)                     

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
from multiprocessing import Manager, get_start_method, set_start_method

import warnings

class rvm:
    def __init__(self, templates, z_range=[-0.01,2], temp_apodization_size=0.05, temp_knots_bin = 100, temp_line_thres=3):
        self.templates = templates
        self.z_range = z_range
        self.temp_apodization_size, self.temp_knots_bin, self.temp_line_thres = temp_apodization_size, temp_knots_bin, temp_line_thres
        
        self.templates1, self.templates2 = {}, {}
        for name in self.templates.keys():
            if self.templates[name][1] == 1:
                self.templates1[name] = copy.deepcopy(self.templates[name])
            if templates[name][1] == 2:
                self.templates2[name] = copy.deepcopy(self.templates[name])
        
        self.shifted_templates = shift_templates(self.templates, self.z_range, self.temp_apodization_size, temp_knots_bin, temp_line_thres)
        self.shifted_templates1 = shift_templates(self.templates1, self.z_range, self.temp_apodization_size, temp_knots_bin, temp_line_thres)
        self.shifted_templates2 = shift_templates(self.templates2, self.z_range, self.temp_apodization_size, temp_knots_bin, temp_line_thres)
    
    
    
    def z_single(self, spectrum, weight=True, output='all', prior='abs', normalization=True, spectrum_range=None, resolution=3, chi_thres=2, mask=None, r_abs=2, r_em=10, 
                 pkratio_abs=1, pkratio_em=1, knots_bin=100, line_thres=3, apodization_size=0.05, window_continuum=100, sn_continuum=1,
                 em_lines=[2798.00, 3727.30, 4861.33, [4958.91, 5006.84, [1,3]], [[6548.06, 6562.82, 6583.57], [0.1,1,0.5]], 6716.440, 6730.815]):
        
        spectrum = copy.deepcopy(spectrum)
        
        if type(spectrum_range)==type([]) or type(spectrum_range)==type(np.array([])):
            if len(np.where((spectrum[0,:]>spectrum_range[0])&(spectrum[0,:]<spectrum_range[1]))[0]) ==0:
                raise ValueError('spectrum_range should contain spectrum wavelengths')
            spectrum = spectrum[:,(spectrum[0,:]>spectrum_range[0])&(spectrum[0,:]<spectrum_range[1])]
        self.pscale = np.median(spectrum[0,1:]-spectrum[0,:-1]) 
        
        spectrum = clean_spectrum(spectrum, window_continuum, sn_continuum)
        scale = np.median(spectrum[2])
        spectrum[1] /= scale
        spectrum[2] /= scale
        abs_spectrum, em_spectrum = copy.deepcopy(spectrum), copy.deepcopy(spectrum)

        
        if output=='best':
            if prior == 'abs':
                # absorption tempaltes
                normalize = process_spectrum(abs_spectrum[0], abs_spectrum[1], np.abs(abs_spectrum[3]/abs_spectrum[2]), resolution=resolution, temp_type=1, knots_bin = knots_bin,
                                            thres=line_thres, apodization_size=apodization_size)
                abs_spectrum[3] = normalize.new_masks
                
                processed_fluxes1 = copy.deepcopy(normalize.normalized_fluxes)
                    
                if type(mask)==type([]) or type(mask)==type(np.array([])):
                    for i in range(len(mask)):
                        left_end = abs(abs_spectrum[0,:]- mask[i][0]).argmin()
                        right_end = abs(abs_spectrum[0,:] - mask[i][1]).argmin()
                        processed_fluxes1[left_end:right_end+1] = 0
                    
                elif mask!=None:
                    raise TypeError('Type of the maks must be list or 2D numpy array')
                
                template_names1 = list(self.templates1.keys())
                n_templates1 = len(template_names1)
                z1, zerr1, r1, chi_eff1, pkratio1 = np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1)
                for i, temp_name in enumerate(template_names1):
                    cc_spec_temp = cc_result(abs_spectrum, weight=weight, normalize=normalize, processed_fluxes=processed_fluxes1,
                                            template=self.templates1[temp_name], shifted_template=self.shifted_templates1[temp_name], 
                                            temp_apodization_size=self.temp_apodization_size,
                                            temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres,
                                            z_range=self.z_range, em_lines=em_lines, resolution=resolution)
                    z1[i], zerr1[i], r1[i], chi_eff1[i], pkratio1[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff, cc_spec_temp.pkratio 
                
                # remove the results with nan-redshift
                if chi_thres:
                    nan_check = (~np.isnan(zerr1))&(~np.isnan(r1))&(~np.isnan(chi_eff1))&(chi_eff1<chi_thres)
                else:
                    nan_check = (~np.isnan(zerr1))&(~np.isnan(r1))&(~np.isnan(chi_eff1))
                template_names1, z1, zerr1, r1, chi_eff1, pkratio1 = np.array(template_names1)[nan_check], z1[nan_check], zerr1[nan_check], r1[nan_check], chi_eff1[nan_check], pkratio1[nan_check]
                # best result among absorption templates
                if len(r1):
                    i_best1 = np.nanargmax(r1)
                    best_templates_name1, best_z1, best_zerr1, best_r1, best_chi_eff1, best_pkratio1 = template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1], pkratio1[i_best1]
                else:
                    best_r1, best_pkratio1 = 0, 99
                    
                if (best_r1 > r_abs) & (best_pkratio1 < pkratio_abs):
                    result = (best_templates_name1, best_z1, best_zerr1, best_r1, best_chi_eff1, best_pkratio1)
                    
                else:
                    # emission tempaltes
                    normalize = process_spectrum(em_spectrum[0], em_spectrum[1], np.abs(em_spectrum[3]/em_spectrum[2]), resolution=resolution, temp_type=2, knots_bin = knots_bin,
                                                thres=line_thres, apodization_size=apodization_size)
                    em_spectrum[3] = normalize.new_masks
                    
                    processed_fluxes2 = copy.deepcopy(normalize.normalized_fluxes)
                    
                    if type(mask)==type([]) or type(mask)==type(np.array([])):
                        for i in range(len(mask)):
                            left_end = abs(em_spectrum[0,:]- mask[i][0]).argmin()
                            right_end = abs(em_spectrum[0,:] - mask[i][1]).argmin()
                            processed_fluxes2[left_end:right_end+1] = 0
                    
                        
                    elif mask!=None:
                        raise TypeError('Type of the maks must be list or 2D numpy array')
                    
                    template_names2 = list(self.templates2.keys())
                    n_templates2 = len(template_names2)
                    z2, zerr2, r2, chi_eff2, pkratio2 = np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2)
                    for i, temp_name in enumerate(template_names2):
                        cc_spec_temp = cc_result(em_spectrum, weight=weight, normalize=normalize, processed_fluxes=processed_fluxes2, 
                                                template=self.templates2[temp_name], shifted_template=self.shifted_templates2[temp_name],
                                                temp_apodization_size=self.temp_apodization_size,
                                                temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres,
                                                z_range=self.z_range, em_lines=em_lines, resolution=resolution)
                    # return output
                        z2[i], zerr2[i], r2[i], chi_eff2[i], pkratio2[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff, cc_spec_temp.pkratio 
                    
                    # remove the results with nan-redshift
                    nan_check = (~np.isnan(zerr2))&(~np.isnan(r2))&(~np.isnan(chi_eff2))
                    template_names2, z2, zerr2, r2, chi_eff2, pkratio2 = np.array(template_names2)[nan_check], z2[nan_check], zerr2[nan_check], r2[nan_check], chi_eff2[nan_check], pkratio2[nan_check]
                            
                    # best result among emssion templates
                    if len(r2):
                        i_best2 = np.nanargmax(r2)
                        best_templates_name2, best_z2, best_zerr2, best_r2, best_chi_eff2, best_pkratio2 = template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2], pkratio2[i_best2]     
                    else:
                        best_r2, best_pkratio2 = 0, 99
                               
                    if (best_r2 > r_em) & (best_pkratio2 < pkratio_em):
                        result = (best_templates_name2, best_z2, best_zerr2, best_r2, best_chi_eff2, best_pkratio2)
                    else:
                        result = ('No_template', -9,-9,-9,-9,99)
                        
            if prior == 'em':
                # emission tempaltes
                normalize = process_spectrum(em_spectrum[0], em_spectrum[1], np.abs(em_spectrum[3]/em_spectrum[2]), resolution=resolution, temp_type=2, knots_bin = knots_bin,
                                            thres=line_thres, apodization_size=apodization_size)
                em_spectrum[3] = normalize.new_masks
                
                processed_fluxes2 = copy.deepcopy(normalize.normalized_fluxes)
                
                if type(mask)==type([]) or type(mask)==type(np.array([])):
                    for i in range(len(mask)):
                        left_end = abs(em_spectrum[0,:]- mask[i][0]).argmin()
                        right_end = abs(em_spectrum[0,:] - mask[i][1]).argmin()
                        processed_fluxes2[left_end:right_end+1] = 0
                
                    
                elif mask!=None:
                    raise TypeError('Type of the maks must be list or 2D numpy array')
                
                template_names2 = list(self.templates2.keys())
                n_templates2 = len(template_names2)
                z2, zerr2, r2, chi_eff2, pkratio2 = np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2)
                for i, temp_name in enumerate(template_names2):
                    cc_spec_temp = cc_result(em_spectrum, weight=weight, normalize=normalize, processed_fluxes=processed_fluxes2, 
                                            template=self.templates2[temp_name], shifted_template=self.shifted_templates2[temp_name],
                                            temp_apodization_size=self.temp_apodization_size,
                                            temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres,
                                            z_range=self.z_range, em_lines=em_lines, resolution=resolution)
                # return output
                    z2[i], zerr2[i], r2[i], chi_eff2[i], pkratio2[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff, cc_spec_temp.pkratio 
                    
                # remove the results with nan-redshift
                nan_check = (~np.isnan(zerr2))&(~np.isnan(r2))&(~np.isnan(chi_eff2))
                template_names2, z2, zerr2, r2, chi_eff2, pkratio2 = np.array(template_names2)[nan_check], z2[nan_check], zerr2[nan_check], r2[nan_check], chi_eff2[nan_check], pkratio2[nan_check]
                
                # best result among emssion templates
                if len(r2):
                    i_best2 = np.nanargmax(r2)
                    best_templates_name2, best_z2, best_zerr2, best_r2, best_chi_eff2, best_pkratio2 = template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2], pkratio2[i_best2]     
                else:
                    best_r2, best_pkratio2 = 0, 99
                
                if (best_r2 > r_em) & (best_pkratio2 < pkratio_em):
                    result = (best_templates_name2, best_z2, best_zerr2, best_r2, best_chi_eff2, best_pkratio2)
                    
                else:
                    # absorption tempaltes
                    normalize = process_spectrum(abs_spectrum[0], abs_spectrum[1], np.abs(abs_spectrum[3]/abs_spectrum[2]), resolution=resolution, temp_type=1, knots_bin = knots_bin,
                                                thres=line_thres, apodization_size=apodization_size)
                    abs_spectrum[3] = normalize.new_masks
                    
                    processed_fluxes1 = copy.deepcopy(normalize.normalized_fluxes)
                        
                    if type(mask)==type([]) or type(mask)==type(np.array([])):
                        for i in range(len(mask)):
                            left_end = abs(abs_spectrum[0,:]- mask[i][0]).argmin()
                            right_end = abs(abs_spectrum[0,:] - mask[i][1]).argmin()
                            processed_fluxes1[left_end:right_end+1] = 0
                        
                    elif mask!=None:
                        raise TypeError('Type of the maks must be list or 2D numpy array')
                    
                    template_names1 = list(self.templates1.keys())
                    n_templates1 = len(template_names1)
                    z1, zerr1, r1, chi_eff1, pkratio1 = np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1)
                    for i, temp_name in enumerate(template_names1):
                        cc_spec_temp = cc_result(abs_spectrum, weight=weight, normalize=normalize, processed_fluxes=processed_fluxes1,
                                                template=self.templates1[temp_name], shifted_template=self.shifted_templates1[temp_name], 
                                                temp_apodization_size=self.temp_apodization_size,
                                                temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres,
                                                z_range=self.z_range, em_lines=em_lines, resolution=resolution)
                        z1[i], zerr1[i], r1[i], chi_eff1[i], pkratio1[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff, cc_spec_temp.pkratio 
                    
                    # remove the results with nan-redshift
                    if chi_thres:
                        nan_check = (~np.isnan(zerr1))&(~np.isnan(r1))&(~np.isnan(chi_eff1))&(chi_eff1<chi_thres)
                    else:
                        nan_check = (~np.isnan(zerr1))&(~np.isnan(r1))&(~np.isnan(chi_eff1))
                    template_names1, z1, zerr1, r1, chi_eff1, pkratio1 = np.array(template_names1)[nan_check], z1[nan_check], zerr1[nan_check], r1[nan_check], chi_eff1[nan_check], pkratio1[nan_check]
                    # best result among absorption templates
                    if len(r1):
                        i_best1 = np.nanargmax(r1)
                        best_templates_name1, best_z1, best_zerr1, best_r1, best_chi_eff1, best_pkratio1 = template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1], pkratio1[i_best1]
                    else:
                        best_r1, best_pkratio1 = 0, 99
                        
                    if (best_r1 > r_abs) & (best_pkratio1 < pkratio_abs):
                        result = (best_templates_name1, best_z1, best_zerr1, best_r1, best_chi_eff1, best_pkratio1)
                        
                    else:
                        result = ('No_template', -9,-9,-9,-9,99)
                    
        if output=='all':
                    
                
            # absorption tempaltes
            normalize = process_spectrum(abs_spectrum[0], abs_spectrum[1], np.abs(abs_spectrum[3]/abs_spectrum[2]), resolution=resolution, temp_type=1, knots_bin = knots_bin,
                                        thres=line_thres, apodization_size=apodization_size)
            abs_spectrum[3] = normalize.new_masks
            
            processed_fluxes1 = copy.deepcopy(normalize.normalized_fluxes)
                
            if type(mask)==type([]) or type(mask)==type(np.array([])):
                for i in range(len(mask)):
                    left_end = abs(abs_spectrum[0,:]- mask[i][0]).argmin()
                    right_end = abs(abs_spectrum[0,:] - mask[i][1]).argmin()
                    processed_fluxes1[left_end:right_end+1] = 0
                
            elif mask!=None:
                raise TypeError('Type of the maks must be list or 2D numpy array')
            
            template_names1 = list(self.templates1.keys())
            n_templates1 = len(template_names1)
            z1, zerr1, r1, chi_eff1, pkratio1 = np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1)
            for i, temp_name in enumerate(template_names1):
                cc_spec_temp = cc_result(abs_spectrum, weight=weight, normalize=normalize, processed_fluxes=processed_fluxes1,
                                        template=self.templates1[temp_name], shifted_template=self.shifted_templates1[temp_name], 
                                        temp_apodization_size=self.temp_apodization_size,
                                        temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres,
                                        z_range=self.z_range, em_lines=em_lines, resolution=resolution)
                z1[i], zerr1[i], r1[i], chi_eff1[i], pkratio1[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff, cc_spec_temp.pkratio 
                    
            # remove the results with nan-redshift
            if chi_thres:
                nan_check = (~np.isnan(zerr1))&(~np.isnan(r1))&(~np.isnan(chi_eff1))&(chi_eff1<chi_thres)
            else:
                nan_check = (~np.isnan(zerr1))&(~np.isnan(r1))&(~np.isnan(chi_eff1))
            template_names1, z1, zerr1, r1, chi_eff1, pkratio1 = np.array(template_names1)[nan_check], z1[nan_check], zerr1[nan_check], r1[nan_check], chi_eff1[nan_check], pkratio1[nan_check]
            
            # best result among absorption templates
            if len(r1):
                i_best1 = np.nanargmax(r1)
                best_templates_name1, best_z1, best_zerr1, best_r1, best_chi_eff1, best_pkratio1 = template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1], pkratio1[i_best1]
            else:
                best_r1, best_pkratio1 = 0, 99
                
            # emission tempaltes
            normalize = process_spectrum(em_spectrum[0], em_spectrum[1], np.abs(em_spectrum[3]/em_spectrum[2]), resolution=resolution, temp_type=2, knots_bin = knots_bin,
                                        thres=line_thres, apodization_size=apodization_size)
            em_spectrum[3] = normalize.new_masks
            
            processed_fluxes2 = copy.deepcopy(normalize.normalized_fluxes)
            
            if type(mask)==type([]) or type(mask)==type(np.array([])):
                for i in range(len(mask)):
                    left_end = abs(em_spectrum[0,:]- mask[i][0]).argmin()
                    right_end = abs(em_spectrum[0,:] - mask[i][1]).argmin()
                    processed_fluxes2[left_end:right_end+1] = 0
            
                
            elif mask!=None:
                raise TypeError('Type of the maks must be list or 2D numpy array')
            
            template_names2 = list(self.templates2.keys())
            n_templates2 = len(template_names2)
            z2, zerr2, r2, chi_eff2, pkratio2 = np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2)
            for i, temp_name in enumerate(template_names2):
                cc_spec_temp = cc_result(em_spectrum, weight=weight, normalize=normalize, processed_fluxes=processed_fluxes2, 
                                        template=self.templates2[temp_name], shifted_template=self.shifted_templates2[temp_name],
                                        temp_apodization_size=self.temp_apodization_size,
                                        temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres,
                                        z_range=self.z_range, em_lines=em_lines, resolution=resolution)
            # return output
                z2[i], zerr2[i], r2[i], chi_eff2[i], pkratio2[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff, cc_spec_temp.pkratio 
                    
            # remove the results with nan-redshift
            nan_check = (~np.isnan(zerr2))&(~np.isnan(r2))&(~np.isnan(chi_eff2))
            template_names2, z2, zerr2, r2, chi_eff2, pkratio2 = np.array(template_names2)[nan_check], z2[nan_check], zerr2[nan_check], r2[nan_check], chi_eff2[nan_check], pkratio2[nan_check]

            # best result among absorption templates
            if len(r2):
                i_best2 = np.nanargmax(r2)
                best_templates_name2, best_z2, best_zerr2, best_r2, best_chi_eff2, best_pkratio2 = template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2], pkratio2[i_best2]     
            else:
                best_r2, best_pkratio2 = 0, 99

            # choose the best result
            if prior=='abs':
                if (best_r1 > r_abs) & (best_pkratio1 < pkratio_abs):
                    i_best = i_best1
                    best = (best_templates_name1, best_z1, best_zerr1, best_r1, best_chi_eff1, best_pkratio1)
                else:
                    if (best_r2 > r_em) & (best_pkratio2 < pkratio_em):
                        i_best = i_best2 + len(r1)
                        best = (best_templates_name2, best_z2, best_zerr2, best_r2, best_chi_eff2, best_pkratio2)
                    else:
                        i_best = None
                        best = ('No_template', -9,-9,-9,-9,99)
            elif prior=='em':
                if (best_r2 > r_em) & (best_pkratio2 < pkratio2):
                    i_best = i_best2+len(r1)
                    best = (best_templates_name2, best_z2, best_zerr2, best_r2, best_chi_eff2, best_pkratio2)
                else:
                    if (best_r1 > r_abs) & (best_pkratio2 < pkratio1):
                        i_best = i_best1
                        best = (best_templates_name1, best_z1, best_zerr1, best_r1, best_chi_eff1, best_pkratio1)
                    else:
                        i_best = None
                        best = ('No_template', -9,-9,-9,-9,99)

            
            if output=='all':
                # concatenate the results from absorption and emission templates
                template_names, z, zerr, r, chi_eff, pkratio = np.concatenate([template_names1, template_names2]), np.concatenate([z1, z2]), np.concatenate([zerr1, zerr2]), np.concatenate([r1,r2]), np.concatenate([chi_eff1, chi_eff2]), np.concatenate([pkratio1, pkratio2])
                note = np.zeros_like(r).astype(str)
                note[:] = ' '
                if i_best != None:
                    note[i_best] = 'best'
                # arange the value in the order of chi_eff
                order = np.flip(np.argsort(r))
                template_names, z, zerr, r, chi_eff, pkratio, note = template_names[order], z[order], zerr[order], r[order], chi_eff[order], pkratio[order], note[order]
                
                table = np.vstack((template_names, z, zerr, r, chi_eff, pkratio, note))
                column_names = ['template_name', 'z', 'zerr', 'r', 'chi_eff', 'pkratio', 'note']
                result = pd.DataFrame(table.T, columns = column_names)
                result = result.astype({'template_name':str, 'z':np.float32, 'zerr':np.float32, 'r':np.float32, 'chi_eff':np.float32, 'pkratio':np.float32, 'note':str})
            
            
                    
        
        return result
    
    def cc_analysis(self, spectrum, temp_name, weight=True, normalization=True, spectrum_range=None, resolution=3, chi_thres=2, mask=None, knots_bin=100,
                    line_thres=3, apodization_size=0.05, window_continuum=100, sn_continuum=1,
                    em_lines=[2798.00, 3727.30, 4861.33, 4958.91, 5006.84, [[6548.06, 6562.82, 6583.57], [0.1,1,0.5]], 6716.440, 6730.815]):
        if type(temp_name) == str:
            temp_name = temp_name
        else:
            z_single_result = self.z_single(spectrum, output='all', spectrum_range=spectrum_range,
                                            chi_thres=chi_thres, resolution=resolution, mask=mask, r_abs=2, r_em=10, 
                 knots_bin=knots_bin, line_thres=line_thres, apodization_size=apodization_size, window_continuum=window_continuum, sn_continuum=sn_continuum)
            temp_name = z_single_result['template_name'][temp_name]
        
        spectrum = copy.deepcopy(spectrum)
        
        self.pscale = np.median(spectrum[0,1:]-spectrum[0,:-1])
        if type(spectrum_range)==type([]) or type(spectrum_range)==type(np.array([])):
            if len(np.where((spectrum[0,:]>spectrum_range[0])&(spectrum[0,:]<spectrum_range[1]))[0]) ==0:
                raise ValueError('spectrum_range should contain spectrum wavelengths')
            spectrum = spectrum[:,(spectrum[0,:]>spectrum_range[0])&(spectrum[0,:]<spectrum_range[1])]
            
        
        self.cspectrum = clean_spectrum(spectrum, window_continuum, sn_continuum)
        scale = np.median(self.cspectrum[2])
        self.cspectrum[1] /= scale
        self.cspectrum[2] /= scale
        self.norm = process_spectrum(self.cspectrum[0], self.cspectrum[1], np.abs(self.cspectrum[3]/self.cspectrum[2]), resolution=resolution, temp_type=self.templates[temp_name][1], knots_bin = knots_bin,
                                    thres=line_thres, apodization_size=apodization_size)
        self.cspectrum[3] = self.norm.new_masks

        self.processed_fluxes = copy.deepcopy(self.norm.normalized_fluxes)
        
        if type(mask)==type([]) or type(mask)==type(np.array([])):
            for i in range(len(mask)):
                left_end = abs(self.cspectrum[0,:]- mask[i][0]).argmin()
                right_end = abs(self.cspectrum[0,:] - mask[i][1]).argmin()
                self.processed_fluxes[left_end:right_end+1] = 0
            
        elif mask!=None:
            raise TypeError('Type of the maks must be list or 2D numpy array')

        self.cc_result = cc_result(self.cspectrum, weight=weight, normalize = self.norm, processed_fluxes=self.processed_fluxes,
                                   template = self.templates[temp_name], shifted_template = self.shifted_templates[temp_name],
                                   temp_apodization_size = self.temp_apodization_size, temp_knots_bin = self.temp_knots_bin, temp_line_thres = self.temp_line_thres,
                                   z_range = self.z_range, em_lines=em_lines, resolution=resolution)


    def z_multi(self, spectrums, directory=None, multi_process=4, **kwargs):
        if multi_process:
            if get_start_method() != 'fork':
                set_start_method('fork', force=True)
            
            def z_multi_sub(indices, subset_number, progress, **kwargs):
                sub_result = []
                for index in tqdm(indices, position=progress, leave=False, desc=f'Progress {subset_number + 1}'):
                    single_result = self.z_single(spectrums[index], output='best', **kwargs)
                    sub_result.append(single_result)
                return np.vstack(sub_result)
                
            spec_number = np.arange(len(spectrums))
            spec_subset = [[n, spec_number[int(len(spec_number) * n / multi_process):int(len(spec_number) * (n + 1) / multi_process)]] for n in range(multi_process)]
            
            manager = Manager()
            progress = manager.list([0] * multi_process)

            def parallel_subroutine(i, indices, subset_number, progress):
                result = z_multi_sub(indices, subset_number, i, **kwargs)
                progress[i] = 1 
                return result 

            result = Parallel(n_jobs=multi_process, verbose=0)(delayed(parallel_subroutine)(i, subset[1], subset[0], progress) for i, subset in enumerate(spec_subset))
            result = pd.DataFrame(np.concatenate(result), columns=['best_template', 'z', 'zerr', 'r', 'chi_eff', 'pkratio'])
            result.astype({'best_template':str, 'z':np.float32, 'zerr':np.float32, 'r':np.float32, 'chi_eff':np.float32, 'pkratio':np.float32})
            return result
            
            
        else:
            spec_number = np.arange(len(spectrums))
            result = []
            for index in tqdm(spec_number, desc='Single Process Progress'):
                singe_result = self.z_single(spectrums[index], output='best', **kwargs)
                result.append(singe_result)
            result = pd.DataFrame(result, columns=['best_template', 'z', 'zerr', 'r', 'chi_eff', 'pkratio'])
            result.astype({'best_template':str, 'z':np.float32, 'zerr':np.float32, 'r':np.float32, 'chi_eff':np.float32, 'pkratio':np.float32})
            return result
            
def rvsnupy(spec_files, spec_import, templates, chunk=5000, directory=None, multi_process=4, z_range=[-0.01,2],
            temp_apodization_size=0.05, temp_knots_bin = 100, temp_line_thres=3, **kwargs):
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
        
        os.system('rm -fr %s'%save_folder+'/*.txt')
        file = open(save_folder + '/z_result.txt', 'w')
        file.write('spectrum_path besttemp z zerr r chi_eff pkratio\n')
        file.close()
        run_rvm = rvm(templates, z_range, temp_apodization_size, temp_knots_bin , temp_line_thres)
        
        n_subs = len(spec_files)/chunk
        if n_subs > int(n_subs):
            n_subs = int(n_subs)+1
        else:
            n_subs = int(n_subs)
        for i in range(n_subs):
            sub_spec_files = spec_files[i*chunk:min((i+1)*chunk, len(spec_files))]
            spectra = []
            print(f'importing spectra for {i+1:d}-th chunk...')
            for file in tqdm(sub_spec_files, leave=False):
                spectra.append(spec_import(file))
            print('done')
            
            print(f'measuring redshifts for {i+1:d}-th chunk...')
            measured = run_rvm.z_multi(spectra, multi_process=multi_process, **kwargs)
            measured = measured.values
            file = open(save_folder + '/z_result.txt', 'a')
            for j in range(measured.shape[0]):
                file.write(f'{sub_spec_files[j]} {measured[j][0]} {measured[j][1]} {measured[j][2]} {measured[j][3]} {measured[j][4]} {measured[j][5]}\n')
            file.close()
            print('done')