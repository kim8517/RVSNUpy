from scipy.optimize import curve_fit

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

from joblib import Parallel, delayed

def fit_gaussian_segment(wavelengths, res, center, rend, lend, resolution, weights):
    try:
        line_wavelengths, line_res, line_weights = wavelengths[lend:rend+1], res[lend:rend+1], weights[lend:rend+1]
        weight_filter = (line_weights!=0)
        line_wavelengths, line_res, line_weights = line_wavelengths[weight_filter], line_res[weight_filter], line_weights[weight_filter]
        line_weights = 1/line_weights
        popt, _ = curve_fit(gaussian, line_wavelengths, line_res, sigma=line_weights,
                            p0 = [res[center], line_wavelengths[len(line_wavelengths)//2], (line_wavelengths[-1]-line_wavelengths[0])*0.5])
        amp = popt[0]
        stddev = np.abs(popt[2])
        width = 2 * int(stddev)
        return True, amp, center, width, stddev
    except Exception as e:
        return False, lend, center, rend

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
            res = fluxes - self.continuum_fluxes
            std = np.nanstd(res)
            detection = np.abs(res) > thres * std

            # Detect line regions
            lends, rends = np.where(np.diff(detection.astype(int)) == 1)[0] + 1, np.where(np.diff(detection.astype(int)) == -1)[0] + 1
            if detection[-1]:  # If the last value is True
                rends = np.concatenate((rends, [len(res) - 1]))
            if detection[0]:  # If the first value is True
                lends = np.concatenate(([0], lends))
            centers = (lends + rends) // 2
            
            # Fit the detected lines in parallel
            line_results = Parallel(n_jobs=-1)(
                delayed(fit_gaussian_segment)(wavelengths, res, center, rend, lend, resolution, weights)
                for lend, rend, center in zip(lends, rends, centers)
                if rend - lend + 1 > 3
            )
            
            # Process the results of the parallel fitting
            for result in line_results:
                if not result[0]:
                    continue
                _, amp, center, width, stddev = result
                if stddev * 2 * np.sqrt(2 * np.log(2)) > resolution:
                    self.new_weights[max(0, center - width):min(len(res)-1, center + width + 1)] = 0
                    self.lines.append([wavelengths[max(0, center - width)], wavelengths[min(len(res)-1, center + width + 1)]])
            
            # Remove regions where too much masking occurred
            concat_edge = [0]
            while len(concat_edge) > 0:
                mask_ratio = np.histogram(wavelengths[self.new_weights == 0], bins=self.knots)[0] / np.histogram(wavelengths, bins=self.knots)[0]
                concat_edge = np.where(mask_ratio > 0.25)[0] + 1
                self.knots = np.delete(self.knots, concat_edge)

            # Recalculate the spline with the new weights
            self.sp_param = splrep(wavelengths, fluxes, t=self.knots, k=5, w=self.new_weights)  # k is the degree of the spline
            self.continuum_fluxes = splev(wavelengths, self.sp_param)
            self.continuum_fluxes[self.continuum_fluxes == 0] = 1e+5 * np.max(np.abs(fluxes))
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
    
    def __init__(self, wavelengths, fluxes, weights, temp_type, resolution=3, knots_bin=100, thres=3, apodization_size=0.05):
        self.wavelengths, self.fluxes, self.weights, self.temp_type = wavelengths, fluxes, weights, temp_type
        self.resolution, self.knots_bin, self.thres = resolution, knots_bin, thres
        self.trace_continuum()
        self.gen_mask()
        self.normalized_fluxes = ((fluxes/self.continuum_fluxes)-1) * tukey(len(fluxes), apodization_size)
        self.normalized_fluxes[np.isnan(self.normalized_fluxes)] = 0
    
    def trace_continuum(self):
        self.knots = np.arange(self.wavelengths[0] + 1e-10, self.wavelengths[-1] + 1e-10, self.knots_bin)
        sp_param = splrep(self.wavelengths, self.fluxes, t=self.knots, w=np.ones_like(self.fluxes), k=5)
        self.continuum_fluxes = splev(self.wavelengths, sp_param)
        self.new_weights = copy.deepcopy(self.weights)

        if self.thres:
            res = self.fluxes - self.continuum_fluxes
            std = np.nanstd(res)
            detection = np.abs(res) > self.thres * std

            if len(res[detection]) > 0:
                # Find line regions
                lends, rends = np.where(np.diff(detection.astype(int)) == 1)[0] + 1, np.where(np.diff(detection.astype(int)) == -1)[0] + 1
                if detection[-1]:
                    rends = np.concatenate((rends, [len(res) - 1]))
                if detection[0]:
                    lends = np.concatenate(([0], lends))
                centers = (lends + rends) // 2

                # Parallel fitting of detected lines
                line_results = Parallel(n_jobs=-1)(
                    delayed(fit_gaussian_segment)(
                        self.wavelengths,
                        res,
                        center,
                        rend,
                        lend,
                        self.resolution,
                        self.weights
                    )
                    for lend, rend, center in zip(lends, rends, centers) if rend - lend + 1 > 3
                )

                # Process the parallel results
                for result in line_results:
                    if not result[0]:
                        continue
                    _, amp, center, width, stddev = result
                    if stddev * 2 * np.sqrt(2 * np.log(2)) > self.resolution:
                        self.new_weights[max(0, center - width):min(len(res)-1, center + width + 1)] = 0
            
            concat_edge = [0]
            while len(concat_edge) > 0:
                mask_ratio = np.histogram(self.wavelengths[self.new_weights == 0], bins=self.knots)[0] / np.histogram(self.wavelengths, bins=self.knots)[0]
                concat_edge = np.where(mask_ratio > 0.25)[0] + 1
                self.knots = np.delete(self.knots, concat_edge)

            self.sp_param = splrep(self.wavelengths, self.fluxes, t=self.knots, k=5, w=self.new_weights)
            self.continuum_fluxes = splev(self.wavelengths, self.sp_param)
            self.continuum_fluxes[self.continuum_fluxes == 0] = 1e+5 * np.max(np.abs(self.fluxes))
            self.continuum_fluxes[0], self.continuum_fluxes[-1] = self.continuum_fluxes[1], self.continuum_fluxes[-2]

    def gen_mask(self):
        self.new_masks = copy.deepcopy(self.weights)
        self.new_masks[self.new_masks != 0] = 1
        if self.thres:
            res = self.fluxes - self.continuum_fluxes
            std = np.nanstd(res)
            detection = np.abs(res) > self.thres * std

            if len(res[detection]) > 0:
                # Find line regions
                lends, rends = np.where(np.diff(detection.astype(int)) == 1)[0], np.where(np.diff(detection.astype(int)) == -1)[0] + 2
                if detection[-1]:
                    rends = np.concatenate((rends, [len(res) - 1]))
                if detection[0]:
                    lends = np.concatenate(([0], lends))
                centers = (lends + rends) // 2

                # Parallel fitting of detected lines
                line_results = Parallel(n_jobs=-1)(
                    delayed(fit_gaussian_segment)(
                        self.wavelengths,
                        res,
                        center,
                        rend,
                        lend,
                        self.resolution,
                        self.weights
                    )
                    for lend, rend, center in zip(lends, rends, centers) if rend - lend + 1 > 3
                )

                # Process the parallel results
                for result in line_results:
                    if not result[0]:
                        _, lend, center, rend = result
                        if np.abs(res[center]*self.weights[center])>3:
                            self.new_masks[max(0, lend):min(len(res)-1, rend)] = 0
                    else:
                        _, amp, center, width, stddev = result
                        if stddev * 2 * np.sqrt(2 * np.log(2)) > self.resolution:
                            if self.temp_type == 1 and amp > 0:
                                self.new_masks[max(0, center - width):min(len(res)-1, center + width + 1)] = 0
                            elif self.temp_type == 2 and amp < 0:
                                self.new_masks[max(0, center - width):min(len(res)-1, center + width + 1)] = 0
                        else:
                            if np.abs(res[center]*self.weights[center])>3:
                                self.new_masks[max(0, center - width):min(len(res)-1, center + width + 1)] = 0
                            

        
        
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
        normalize=process_template(temp[0], temp[1], np.ones_like(temp[1]), knots_bin=knots_bin, thres=thres, apodization_size=apodization_size)
            
        # log pixel scale
        log_wavelengths = np.log10(temp[0])
        log_bin = np.median(log_wavelengths[1:]-log_wavelengths[:-1])

        # prepare a wavelength array
        n_left, n_right = -int(np.log10(z_range[0]-0.2*(1+z_range[0])+1)/log_bin), int(np.log10(z_range[1]+0.2*(1+z_range[1])+1)/log_bin)
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
        if (np.median(np.abs(spec[1,i_left:i_left+window]/spec[2,i_left:i_left+window])) > sn) & (np.sum(spec[3,i_left:i_left+window])>window*0.9):
            break
        
    for i_right in range(spec.shape[1], 0, -window):
        if (np.median(np.abs(spec[1,i_right-window:i_right]/spec[2,i_right-window:i_right])) > sn) & (np.sum(spec[3,i_right-window:i_right])>window*0.9):
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
from scipy.optimize import curve_fit as cf

def multiple_gaussian(x, *params):
    n_gaussians = len(params) //3
    y = np.zeros_like(x)
    for i in range(n_gaussians):
        amp = params[i*3]
        mu = params[i*3+1]
        sigma = params[i*3+2]
        y+= amp*np.exp(-0.5*(x-mu)**2/sigma**2)
        
    return y

def sub_fit_multiline(filter_weights, observed_wavelength_min, observed_wavelength_max,
                      wavelengths, fluxes, weights, line, z, z_min, z_max,
                      stddev_resolution, temp_type, cc_width):
    nlines, ndetect = 0,0
    model_fluxes = np.zeros_like(wavelengths)
    z_lines = []
    if isinstance(line, float):
        # Redshifted wavelength
        redshifted_wavelength = line * (1 + z)
        redshifted_wavelength_min, redshifted_wavelength_max = line*(1+z_min), line*(1+z_max)
        # Check if the redshifted line is within the observed wavelength range
        if observed_wavelength_min <= redshifted_wavelength <= observed_wavelength_max:
            nlines += 1
            wave_left, wave_right = redshifted_wavelength*(1-cc_width/c), redshifted_wavelength*(1+cc_width/c)
            fit_range = (wavelengths>wave_left)&(wavelengths<wave_right)&filter_weights
            std0 = np.sqrt((redshifted_wavelength*cc_width/c)**2-2*stddev_resolution**2)
            # Single line within range
            p0 = [fluxes[np.argmin(np.abs(wavelengths - redshifted_wavelength))], redshifted_wavelength, std0]
            bounds = ([-np.inf, redshifted_wavelength_min, stddev_resolution],
                        [np.inf, redshifted_wavelength_max, 2*std0])
            try:
                popt, pcov = cf(multiple_gaussian, wavelengths[fit_range] , fluxes[fit_range], sigma=weights[fit_range], p0=p0,
                                bounds=bounds, absolute_sigma=True)
                if popt[0]>0:
                # plt.plot(wavelengths[fit_range], multiple_gaussian(wavelengths[fit_range], *popt))
                    model_fluxes += multiple_gaussian(wavelengths, *popt)
                    z_line = (popt[1]-line)/line
                    z_lines.append(z_line)
                    ndetect += 1
            except Exception as e:
                pass
    elif isinstance(line, list):
        valid_lines = []
        for l in line:
            redshifted_wavelength = l*(1+z)
            if observed_wavelength_min <= redshifted_wavelength <= observed_wavelength_max:
                valid_lines.append(l)
                nlines += 1
        valid_lines = np.array(valid_lines)
        while(len(valid_lines)):
            wave_left, wave_right = valid_lines[0]*(1+z)*(1-5*cc_width/c), valid_lines[-1]*(1+z)*(1+5*cc_width/c)
            fit_range = (wavelengths>wave_left)&(wavelengths<wave_right)&filter_weights
            p0 = []
            low_bounds, up_bounds = [], []
            for l in valid_lines:
                redshifted_wavelength = l*(1+z)
                redshifted_wavelength_min, redshifted_wavelength_max = l*(1+z_min), l*(1+z_max)
                std0 = np.sqrt((redshifted_wavelength*cc_width/c)**2-2*stddev_resolution**2)
                p0.append(fluxes[np.argmin(np.abs(wavelengths - redshifted_wavelength))])
                p0.append(redshifted_wavelength)
                p0.append(std0)
                low_bounds.append(-np.inf)
                low_bounds.append(redshifted_wavelength_min)
                low_bounds.append(stddev_resolution)
                up_bounds.append(np.inf)
                up_bounds.append(redshifted_wavelength_max)
                up_bounds.append(2*std0)
            bounds = (low_bounds, up_bounds)
            try:
                popt, pcov = cf(multiple_gaussian, wavelengths[fit_range] , fluxes[fit_range], sigma=weights[fit_range], p0=p0,
                                bounds=bounds)
                
                if temp_type==2:
                    detect = popt[np.arange(len(valid_lines))*3]>0
                elif temp_type==1:
                    detect = popt[np.arange(len(valid_lines))*3]<0
                if len(np.where(~detect)[0]):
                    valid_lines = valid_lines[detect]
                else:
                    ndetect += len(valid_lines)
                    model_fluxes += multiple_gaussian(wavelengths, *popt)
                    z_lines.append(z_line)
                    break
                
            except Exception as e:
                break
    if len(z_lines):
        z_line = np.mean(np.array(z_lines))
    else:
        z_line = np.nan   
    return nlines, ndetect, z_line, model_fluxes

def fit_multiline(wavelengths, fluxes, weights, lines, z, zerr, cc_width, resolution, temp_type):
    filter_weights = weights!=0
    # wavelengths, fluxes, weights = wavelengths[filter_weights], fluxes[filter_weights], weights[filter_weights]
    observed_wavelength_min, observed_wavelength_max = wavelengths[0], wavelengths[-1]
    z_min, z_max = z-3*zerr, z+3*zerr
    stddev_resolution = resolution/(2*np.sqrt(np.log(2)*2))
    
    results = Parallel(n_jobs=-1)(
        delayed(sub_fit_multiline)(filter_weights, observed_wavelength_min,
                                   observed_wavelength_max,
                                   wavelengths, fluxes, weights, line, z, z_min, z_max,
                                   stddev_resolution, temp_type, cc_width) for line in lines)
    
    ndetect = sum(res[1] for res in results)
    model_fluxes = np.sum([res[3] for res in results], axis=0)
    chi_eff = np.sum(((fluxes[filter_weights]-model_fluxes[filter_weights])/weights[filter_weights])**2)/(len(fluxes[filter_weights])-3*ndetect)
    
    return chi_eff, model_fluxes

def sub_interp_fluxes(i_pixel, spectrum, normalize, new_wavelengths, shifted_wavelengths, shifted_vels,
                      template_spectrum, template_lines, new_fit_weights, temp_knots_bin,
                      temp_line_thres, temp_apodization_size):
    region = ((new_wavelengths>=shifted_wavelengths[i_pixel])
              &(new_wavelengths<=shifted_wavelengths[i_pixel+template_spectrum.shape[1]-1]))
    new_template_wavelengths = new_wavelengths[region]
    resampled_template_fluxes = resampler(shifted_wavelengths[i_pixel:i_pixel+template_spectrum.shape[1]],
                                          template_spectrum[1], new_template_wavelengths)
    new_fit_weights_at_v = copy.deepcopy(new_fit_weights[region])
    for line in template_lines:
        new_fit_weights_at_v[(new_template_wavelengths>line[0]*(1+shifted_vels[i_pixel]/c))
                             &(new_template_wavelengths<line[1]*(1+shifted_vels[i_pixel]/c))] = 0
    
    pro_temp = process_template(new_template_wavelengths, resampled_template_fluxes,
                                     new_fit_weights_at_v, line_identified=True, knots_bin=temp_knots_bin,
                                     thres = temp_line_thres, apodization_size=temp_apodization_size)
    
    lamb_min, lamb_max = max(spectrum[0,0], new_template_wavelengths[0]), min(spectrum[0,-1], new_template_wavelengths[-1])
    overlap_spec_range = (spectrum[0]>=lamb_min)&(spectrum[0]<=lamb_max)
    overlap_spec = spectrum[:,overlap_spec_range]
    overlap_temp_range = (new_template_wavelengths>=lamb_min)&(new_template_wavelengths<=lamb_max)
    T_ = resampled_template_fluxes[overlap_temp_range]*normalize.continuum_fluxes[overlap_spec_range]/pro_temp.continuum_fluxes[overlap_temp_range]
    n_knots_eff = len(pro_temp.knots[(pro_temp.knots>=lamb_min)&(pro_temp.knots<lamb_max)])
    overlap_res = (overlap_spec[3]/overlap_spec[2])**2*(overlap_spec[1]-T_)**2
    nonoverlap_res = (spectrum[1]-normalize.continuum_fluxes)*spectrum[3]/spectrum[2]
    res = np.concatenate([nonoverlap_res[spectrum[0]<lamb_min], overlap_res, nonoverlap_res[spectrum[1]>lamb_max]])
    chi_eff = np.sum(res)/(np.sum((overlap_spec[3]))-n_knots_eff-1)
    
    return region, pro_temp.normalized_fluxes, chi_eff

class cc_result:
    def __init__(self, spectrum, normalize, processed_fluxes, template, shifted_template,
                 temp_apodization_size=0.05, temp_knots_bin=100, temp_line_thres=3, weight=True,
                 z_range=[-0.01,2], r_thres = 5, line_fit = True,
                 em_lines=[2799.117, 3727.30, 4102.89, 4341.68, 4861.33, [4958.91, 5006.84], [6548.06, 6562.82, 6583.57], [6716.440, 6730.815]],
                 abs_lines = [3933.66, 3968.47, 4304.40, 5175.26, 5893.97, 8498.03, 8542.09, 8662.14], 
                 resolution=3):
        warnings.simplefilter('ignore')
        self.spectrum, self.normalize, self.processed_fluxes = spectrum, normalize, processed_fluxes
        self.template, self.shifted_template  = template, shifted_template
        self.temp_apodization_size, self.temp_knots_bin, self.temp_line_thres = temp_apodization_size, temp_knots_bin, temp_line_thres
        self.z_range = z_range
        self.r_thres, self.line_fit = r_thres, line_fit
        self.em_lines, self.abs_lines, self.resolution = em_lines, abs_lines, resolution
        self.shifted_vels, self.shifted_wavelengths, self.shifted_fluxes, self.template_spectrum = self.shifted_template[0], self.shifted_template[1], self.shifted_template[2], self.template[0]
        # self.cross_correlate()
        try:
            self.cross_correlate()
        except:
            self.z, self.zerr, self.r, self.chi_eff  = np.nan, np.nan, np.nan, np.nan
     
        
    def cross_correlate0(self):
        overlap_wavelengths = self.shifted_wavelengths[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])]
        self.new_fluxes0, self.new_weights0, self.new_masks0 = np.zeros_like(self.shifted_wavelengths), np.ones_like(self.shifted_wavelengths), np.ones_like(self.shifted_wavelengths)
        overlap_fluxes = resampler(self.spectrum[0], self.processed_fluxes, overlap_wavelengths)
        overlap_weights = resampler(self.spectrum[0], np.abs(self.normalize.continuum_fluxes/self.spectrum[2]), overlap_wavelengths)
        overlap_masks = discrete_resampler(self.spectrum[0], self.spectrum[3], overlap_wavelengths)

        self.new_fluxes0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_fluxes
        self.new_weights0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_weights
        self.new_masks0[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_masks

        self.cc0 = np.matmul(self.shifted_fluxes, self.new_masks0*self.new_weights0**2*self.new_fluxes0)


        self.cz0 = self.shifted_vels[np.nanargmax(self.cc0)]

    def find_peak_region(self):
        cz_range = np.array(self.z_range)*c
        cc_inrange = self.cc0[(self.shifted_vels>cz_range[0])&(self.shifted_vels<cz_range[1])]
        lags_inrange = self.shifted_vels[(self.shifted_vels>cz_range[0])&(self.shifted_vels<cz_range[1])]
        
        i_max = np.nanargmax(cc_inrange)
        cc_max, lags_max = cc_inrange[i_max], lags_inrange[i_max] # estimates a peak
        
        # find peaks
        detection = (((self.cc0 >= 0.5*cc_max))| (self.cc0 >= 3*np.std(self.cc0)))&((self.shifted_vels>=cz_range[0])&(self.shifted_vels<=cz_range[1]))
        # detection = ((self.cc0 >= 3*np.std(self.cc0)))&((self.shifted_vels>=cz_range[0])&(self.shifted_vels<=cz_range[1]))

        lends, rends = np.where(np.diff(detection.astype(int)) == 1)[0] + 1, np.where(np.diff(detection.astype(int)) == -1)[0] + 1
        if detection[-1] == True: # dectecion = (..., True, True, True)
            rends = np.concatenate((rends, np.array([len(self.cc0)-1])))
        if detection[0] == True: # dectecion = (True, True, True, ...)
            lends = np.concatenate((np.array([0]), lends))
        rends -= 1
        
        centers = (lends+rends) // 2

        self.peak_ranges = []
        n = 0
        for i_peak in np.argsort(self.cc0[centers])[-1::-1]:
            if n>5:
                break
            v_pcenter, v_pwidth = self.shifted_vels[centers[i_peak]], 0.5*(self.shifted_vels[rends[i_peak]]-self.shifted_vels[lends[i_peak]])
            lv_pcenter, rv_pcenter = v_pcenter-3*v_pwidth, v_pcenter+3*v_pwidth
            cc_peak, lags_peak = self.cc0[(self.shifted_vels>lv_pcenter)&(self.shifted_vels<rv_pcenter)], self.shifted_vels[(self.shifted_vels>lv_pcenter)&(self.shifted_vels<rv_pcenter)]
            try:
                self.fit_peak_range = fitting.LevMarLSQFitter()
                self.gaussian_peak_range0 = models.Gaussian1D(amplitude=self.cc0[centers[i_peak]], mean=self.shifted_vels[centers[i_peak]], stddev=self.shifted_vels[centers[i_peak]]-self.shifted_vels[lends[i_peak]-1])
                self.gaussian_peak_range = self.fit_peak_range(self.gaussian_peak_range0, lags_peak, cc_peak)
                prange = [self.gaussian_peak_range.mean.value-2*self.gaussian_peak_range.stddev.value,
                                    self.gaussian_peak_range.mean.value+2*self.gaussian_peak_range.stddev.value]
            except:
                continue
                
            if (self.gaussian_peak_range.amplitude.value < 0) | (self.gaussian_peak_range.mean.value < cz_range[0]) | (self.gaussian_peak_range.mean.value>cz_range[1]) | (len(np.where((self.shifted_vels>prange[0])&(self.shifted_vels<prange[1]))[0])<1):
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
        self.chi_effs = np.zeros_like(self.cc0)*np.nan
        max_peaks, max_rs = np.zeros(len(self.peak_ranges)), np.zeros(len(self.peak_ranges))
        for m, prange in enumerate(self.peak_ranges):
            self.peak_region = np.where((self.shifted_vels>prange[0])&(self.shifted_vels<prange[1]))[0]
            self.interp_fluxes = np.zeros((len(self.peak_region),len(self.new_fluxes)))
            self.chi_effs_ = np.zeros(len(self.peak_region))
            
            ########################################################################################################
            i = int(np.median(self.peak_region))
            region = ((self.new_wavelengths>=self.shifted_wavelengths[i])&(self.new_wavelengths<=self.shifted_wavelengths[i+self.template_spectrum.shape[1]-1]))
            new_template_wavelengths = self.new_wavelengths[region]
            resampled_template_fluxes = resampler(self.shifted_wavelengths[i:i+self.template_spectrum.shape[1]], self.template_spectrum[1], new_template_wavelengths)
            template_lines = process_template(new_template_wavelengths, resampled_template_fluxes, self.new_fit_weights[region], knots_bin = self.temp_knots_bin,
                                                    thres = self.temp_line_thres, apodization_size=self.temp_apodization_size).lines
            template_lines0 = []
            for line in template_lines:
                template_lines0.append([line[0]/(1+self.shifted_vels[i]/c), line[1]/(1+self.shifted_vels[i]/c)])
            ########################################################################################################
            
            # for n, i in enumerate(self.peak_region):
            #     region = ((self.new_wavelengths>=self.shifted_wavelengths[i])&(self.new_wavelengths<=self.shifted_wavelengths[i+self.template_spectrum.shape[1]-1]))
            #     new_template_wavelengths = self.new_wavelengths[region]
            #     resampled_template_fluxes = resampler(self.shifted_wavelengths[i:i+self.template_spectrum.shape[1]], self.template_spectrum[1], new_template_wavelengths)
            #     new_fit_weights_at_v = copy.deepcopy(self.new_fit_weights[region])
            #     for line in template_lines0:
            #         new_fit_weights_at_v[(new_template_wavelengths>line[0]*(1+self.shifted_vels[i]/c))&(new_template_wavelengths<line[1]*(1+self.shifted_vels[i]/c))] = 0
            #     self.interp_fluxes[n,region] = process_template(new_template_wavelengths, resampled_template_fluxes, new_fit_weights_at_v, line_identified=True, knots_bin = self.temp_knots_bin,
            #                                             thres = self.temp_line_thres, apodization_size=self.temp_apodization_size).normalized_fluxes
            
            interp_fluxes_ = Parallel(n_jobs=-1)(delayed(sub_interp_fluxes)(i, self.spectrum, self.normalize,
                                                                            self.new_wavelengths, self.shifted_wavelengths,
                                                                           self.shifted_vels, self.template_spectrum,
                                                                           template_lines0, self.new_fit_weights, self.temp_knots_bin,
                                                                           self.temp_line_thres, self.temp_apodization_size) for i in self.peak_region)
            for n, (region, interp_fluxes_at_shift, chi_eff_) in enumerate(interp_fluxes_):
                self.interp_fluxes[n, region] = interp_fluxes_at_shift
                self.chi_effs_[n] = chi_eff_

            weight_cc = np.matmul(self.interp_fluxes,self.new_masks*self.new_weights**2*self.new_fluxes)
            max_peaks[m] = np.nanmax(weight_cc)
            self.cc[self.peak_region] = weight_cc
            self.chi_effs[self.peak_region] = self.chi_effs_
    
    def cross_correlate(self):
        self.cross_correlate0()
        self.find_peak_region()
        if len(self.peak_ranges):
            self.cc_near_peak()
            self.z_finding()
        else:
            self.cc = self.cc0
            self.z, self.zerr, self.r, self.chi_eff  = np.nan, np.nan, np.nan, np.nan

    def z_finding(self):
        gaussian_fits, fitters= [],[]
        for i in range(len(self.peak_ranges)):
            self.max_peak_range = self.peak_ranges[i]
            cc_peak = self.cc[(self.shifted_vels>self.max_peak_range[0])&(self.shifted_vels<self.max_peak_range[1])]
            lags_peak = self.shifted_vels[(self.shifted_vels>self.max_peak_range[0])&(self.shifted_vels<self.max_peak_range[1])]
            nan_filtering = (~np.isnan(cc_peak))&((~np.isnan(lags_peak)))
            cc_peak, lags_peak = cc_peak[nan_filtering], lags_peak[nan_filtering]
            
            fit_peak = fitting.LevMarLSQFitter(calc_uncertainties=True)
            gaussian_peak0 = models.Gaussian1D(amplitude=np.nanmax(cc_peak), mean=np.median(lags_peak), stddev=0.5*(lags_peak[-1]-lags_peak[0]))
            gaussian_peak = fit_peak(gaussian_peak0, lags_peak, cc_peak)
            if type(fit_peak.fit_info['param_cov']) != type(None):
                gaussian_fits.append(gaussian_peak)
                fitters.append(fit_peak)
                n_peak = np.abs(self.shifted_vels-gaussian_peak.mean.value).argmin() # find an index of peak
                
        self.bestfit_gaussians = gaussian_fits[0]
        for gauss in gaussian_fits[1:]:
            self.bestfit_gaussians+=gauss
        
                
        zs, zerrs, r_values, pwidth = np.zeros(len(gaussian_fits)), np.zeros(len(gaussian_fits)), np.zeros(len(gaussian_fits)), np.zeros(len(gaussian_fits))
        pscale = np.median(self.spectrum[0,1:]-self.spectrum[0,:-1])
        presol = self.resolution/pscale
        for i in range(len(zs)):
            # extract best fit parameters
            amp, mean, std = gaussian_fits[i].amplitude.value, gaussian_fits[i].mean.value, gaussian_fits[i].stddev.value
            damp, dmean, dstd = np.sqrt(fitters[i].fit_info['param_cov'][0,0]), np.sqrt(fitters[i].fit_info['param_cov'][1,1]), np.sqrt(fitters[i].fit_info['param_cov'][2,2])
            # redshift
            zs[i]  = mean/c
            # redshift uncertainty
            zerr_ = std*np.sqrt(-2*np.log(1-0.5/amp))/c
            dzerr_ = np.sqrt((dmean+2*std*np.sqrt(-2/np.log(1-0.5/amp))*damp/(amp*(amp-0.5))+2*dstd*np.sqrt(-2*np.log(1-0.5/amp)))**2+dmean**2)/c
            zerrs[i] = zerr_+dzerr_
            # r-value
            n_peak = np.abs(self.shifted_vels-mean).argmin() # find an index of peak
            N = int((0.1*(c+mean))/(self.shifted_vels[n_peak]-self.shifted_vels[n_peak-1]))
            # N = int((5*std)/(self.shifted_vels[n_peak]-self.shifted_vels[n_peak-1]))
            left, right = max(n_peak-N,0), min(n_peak+N, len(self.cc)-1)
            nrange = int(min(n_peak-left, right-n_peak))
            cc_left, cc_right = self.cc[n_peak-nrange:n_peak], np.flip(self.cc[n_peak+1:n_peak+nrange+1])
            cc_left, cc_right = cc_left[(~np.isnan(cc_left))&(~np.isnan(cc_right))], cc_right[(~np.isnan(cc_left))&(~np.isnan(cc_right))]
            sigma = np.sum(((cc_left - cc_right)**2))/nrange
            r_values[i] = self.cc[n_peak]/(np.sqrt(sigma))
            pwidth[i] = std

        if (len(r_values) == 0) | (np.max(r_values) ==0) :
            self.z, self.zerr, self.r, self_chi_eff = np.nan, np.nan, np.nan, np.nan
    
        else:
            sort = np.argsort(r_values)[::-1]
            zs, zerrs, r_values, pwidth = zs[sort], zerrs[sort], r_values[sort], pwidth[sort]
            r_filt = (r_values>self.r_thres)
            self.zs, self.r_values = zs, r_values
            zs, zerrs, r_values, pwidth = zs[r_filt], zerrs[r_filt], r_values[r_filt], pwidth[r_filt]
            if len(zs):
                if self.line_fit:
                    self.line_chi_eff, self.model_fluxes = 100, np.zeros_like(self.spectrum[0])
                    self.z, self.zerr, self.r = np.nan, np.nan, np.nan
                    if self.template[1] == 2:
                        for i in range(len(zs)):
                            line_chi_eff_, model_fluxes_ = fit_multiline(self.spectrum[0], self.processed_fluxes, 
                                                            self.spectrum[3]*self.spectrum[2]/self.normalize.continuum_fluxes,
                                                            lines=self.em_lines, z=zs[i], zerr=zerrs[i], cc_width=pwidth[i],
                                                            resolution=self.resolution, temp_type=self.template[1])
                            if line_chi_eff_ < self.line_chi_eff:
                                self.line_chi_eff, self.model_fluxes = line_chi_eff_, model_fluxes_
                                self.z, self.zerr, self.r = zs[i], zerrs[i], r_values[i]
                        self.chi_eff = self.line_chi_eff
                    else:
                        chi_min = 100
                        self.z, self.zerr, self.r = np.nan, np.nan, np.nan
                        for i in range(len(zs)):
                            chi_min_ = self.chi_effs[np.argmin(np.abs(zs[i]*c-self.shifted_vels))]
                            if chi_min_<chi_min:
                                chi_min = chi_min_
                                self.z, self.zerr, self.r = zs[i], zerrs[i], r_values[i]
                        self.chi_eff = chi_min
                    # elif self.template[1] == 1:
                    #     for i in range(len(zs)):
                    #         line_chi_eff_, model_fluxes_ = fit_multiline(self.spectrum[0], self.processed_fluxes, 
                    #                                         self.spectrum[3]*self.spectrum[2]/self.normalize.continuum_fluxes,
                    #                                         lines=self.em_lines, z=zs[i], zerr=zerrs[i], cc_width=pwidth[i],
                    #                                         resolution=self.resolution, temp_type=self.template[1])
                    #         if line_chi_eff_ < self.line_chi_eff:
                    #             self.line_chi_eff, self.model_fluxes = line_chi_eff_, model_fluxes_
                    #             self.z, self.zerr, self.r = zs[i], zerrs[i], r_values[i]
                else:
                    self.z, self.zerr, self.r = zs[0], zerrs[0], r_values[0]
                    self.line_chi_eff, self.model_fluxes = np.nan, np.zeros_like(self.spectrum[0])
                    if np.isfinite(self.z):
                        self.cal_chi_eff()
                    else:
                        self.chi_eff = np.nan
            else:
                self.z, self.zerr, self.r, self.chi_eff = np.nan, np.nan, np.nan, np.nan
                

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

import os
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Manager, get_start_method, set_start_method

import warnings
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
    
    
    
    def z_single(self, spectrum, weight=True, output='all', prior='abs', normalization=True, spectrum_range=None, resolution=3, chi_thres=2, mask=None, r_thres=3, r_confi=5,
                 knots_bin=100, line_thres=3, apodization_size=0.05, window_continuum=100, sn_continuum=0.5,
                 line_fit = True,
                 em_lines=[2798.00, 3727.30, 4861.33, [4958.91, 5006.84], [6548.06, 6562.82, 6583.57], [6716.440, 6730.815]],
                 abs_lines = [[3933.66, 3968.47], 4304.40, 5175.26, 5893.97, 8498.03, 8542.09, 8662.14]):
        
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
        # spectrum[3,spectrum[1]/spectrum[2]<-3] = 0
        if type(mask)==type([]) or type(mask)==type(np.array([])):
            for i in range(len(mask)):
                left_end = abs(spectrum[0,:]- mask[i][0]).argmin()
                right_end = abs(spectrum[0,:] - mask[i][1]).argmin()
                spectrum[3,left_end:right_end+1] = 0
                
        abs_spectrum, em_spectrum = copy.deepcopy(spectrum), copy.deepcopy(spectrum)

        
        if output=='best':
            if prior == 'abs':
                # absorption tempaltes
                normalize = process_spectrum(abs_spectrum[0], abs_spectrum[1], np.abs(abs_spectrum[3]/abs_spectrum[2]), resolution=resolution, temp_type=1, knots_bin = knots_bin,
                                            thres=line_thres, apodization_size=apodization_size)
                abs_spectrum[3] = normalize.new_masks
                
                processed_fluxes1 = copy.deepcopy(normalize.normalized_fluxes)
                
                template_names1 = list(self.templates1.keys())
                n_templates1 = len(template_names1)
                z1, zerr1, r1, chi_eff1 = np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1)
                for i, temp_name in enumerate(template_names1):
                    cc_spec_temp = cc_result(abs_spectrum, weight=weight, normalize=normalize, processed_fluxes=processed_fluxes1,
                                            template=self.templates1[temp_name], shifted_template=self.shifted_templates1[temp_name], 
                                            temp_apodization_size=self.temp_apodization_size,
                                            temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres,
                                            z_range=self.z_range, r_thres=r_thres, line_fit=line_fit,
                                            em_lines=em_lines, abs_lines=abs_lines, resolution=resolution)
                    z1[i], zerr1[i], r1[i], chi_eff1[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                
                
                
                # remove the results with nan-redshift
                if chi_thres:
                    nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(chi_eff1<chi_thres)
                else:
                    nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))
                template_names1, z1, zerr1, r1, chi_eff1 = np.array(template_names1)[nan_check], z1[nan_check], zerr1[nan_check], r1[nan_check], chi_eff1[nan_check]
                # best result among absorption templates
                if (len(r1)>0)&(r1[np.argmin(chi_eff1)]>r_confi):
                    i_best1 = np.nanargmin(chi_eff1)
                    result = (template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1])
                    
                else:
                    # emission tempaltes
                    normalize = process_spectrum(em_spectrum[0], em_spectrum[1], np.abs(em_spectrum[3]/em_spectrum[2]), resolution=resolution, temp_type=2, knots_bin = knots_bin,
                                                thres=line_thres, apodization_size=apodization_size)
                    em_spectrum[3] = normalize.new_masks
                    
                    processed_fluxes2 = copy.deepcopy(normalize.normalized_fluxes)
                    
                    template_names2 = list(self.templates2.keys())
                    n_templates2 = len(template_names2)
                    z2, zerr2, r2, chi_eff2 = np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2)
                    for i, temp_name in enumerate(template_names2):
                        cc_spec_temp = cc_result(em_spectrum, weight=weight, normalize=normalize, processed_fluxes=processed_fluxes2, 
                                                template=self.templates2[temp_name], shifted_template=self.shifted_templates2[temp_name],
                                                temp_apodization_size=self.temp_apodization_size,
                                                temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres,
                                                z_range=self.z_range, r_thres=r_thres, line_fit=line_fit,
                                                em_lines=em_lines, abs_lines=abs_lines, resolution=resolution)
                    # return output
                        z2[i], zerr2[i], r2[i], chi_eff2[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                    
                    # remove the results with nan-redshift
                    if chi_thres:
                        nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(chi_eff2<chi_thres)
                    else:
                        nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))
                    template_names2, z2, zerr2, r2, chi_eff2 = np.array(template_names2)[nan_check], z2[nan_check], zerr2[nan_check], r2[nan_check], chi_eff2[nan_check]
                            
                    # best result among emssion templates
                    if len(r2):
                        i_best2 = np.nanargmin(chi_eff2)
                        if r2[i_best2]>r_confi:
                            result = (template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2]) 
                        else:
                            if len(r1):
                                i_best1 = np.nanargmin(chi_eff1)
                                result = (template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1])
                            else:
                                result = (template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2])
                    else:
                        if len(r1):
                            i_best1 = np.nanargmin(chi_eff1)
                            result = (template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1])
                        else:
                            result = ('No_template', -9,-9,-9,-9)
                        
            if prior == 'em':
                # emission tempaltes
                normalize = process_spectrum(em_spectrum[0], em_spectrum[1], np.abs(em_spectrum[3]/em_spectrum[2]), resolution=resolution, temp_type=2, knots_bin = knots_bin,
                                            thres=line_thres, apodization_size=apodization_size)
                em_spectrum[3] = normalize.new_masks
                
                processed_fluxes2 = copy.deepcopy(normalize.normalized_fluxes)
                
                template_names2 = list(self.templates2.keys())
                n_templates2 = len(template_names2)
                z2, zerr2, r2, chi_eff2 = np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2)
                for i, temp_name in enumerate(template_names2):
                    cc_spec_temp = cc_result(em_spectrum, weight=weight, normalize=normalize, processed_fluxes=processed_fluxes2, 
                                            template=self.templates2[temp_name], shifted_template=self.shifted_templates2[temp_name],
                                            temp_apodization_size=self.temp_apodization_size,
                                            temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres,
                                            z_range=self.z_range, line_fit=line_fit,
                                            em_lines=em_lines, abs_lines=abs_lines, resolution=resolution)
                # return output
                    z2[i], zerr2[i], r2[i], chi_eff2[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                    
                # remove the results with nan-redshift
                if chi_thres:
                    nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(chi_eff2<chi_thres)
                else:
                    nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))
                template_names2, z2, zerr2, r2, chi_eff2 = np.array(template_names2)[nan_check], z2[nan_check], zerr2[nan_check], r2[nan_check], chi_eff2[nan_check]
                
                # best result among emssion templates
                if len(r2):
                    i_best2 = np.nanargmin(chi_eff2)
                    result = (template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2])
                    
                else:
                    # absorption tempaltes
                    normalize = process_spectrum(abs_spectrum[0], abs_spectrum[1], np.abs(abs_spectrum[3]/abs_spectrum[2]), resolution=resolution, temp_type=1, knots_bin = knots_bin,
                                                thres=line_thres, apodization_size=apodization_size)
                    abs_spectrum[3] = normalize.new_masks
                    
                    processed_fluxes1 = copy.deepcopy(normalize.normalized_fluxes)
                    
                    template_names1 = list(self.templates1.keys())
                    n_templates1 = len(template_names1)
                    z1, zerr1, r1, chi_eff1 = np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1)
                    for i, temp_name in enumerate(template_names1):
                        cc_spec_temp = cc_result(abs_spectrum, weight=weight, normalize=normalize, processed_fluxes=processed_fluxes1,
                                                template=self.templates1[temp_name], shifted_template=self.shifted_templates1[temp_name], 
                                                temp_apodization_size=self.temp_apodization_size,
                                                temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres,
                                                z_range=self.z_range, r_thres=r_thres, line_fit=line_fit,
                                                em_lines=em_lines, abs_lines=abs_lines, resolution=resolution)
                        z1[i], zerr1[i], r1[i], chi_eff1[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                    
                    # remove the results with nan-redshift
                    if chi_thres:
                        nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(chi_eff1<chi_thres)
                    else:
                        nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))
                    template_names1, z1, zerr1, r1, chi_eff1 = np.array(template_names1)[nan_check], z1[nan_check], zerr1[nan_check], r1[nan_check], chi_eff1[nan_check]
                    # best result among absorption templates
                    if len(r1):
                        i_best1 = np.nanargmax(chi_eff1)
                        result = (template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1])  
                    else:
                        result = ('No_template', -9,-9,-9,-9)
                    
        elif output=='all':
            self.cc_result = {}
            # absorption tempaltes
            normalize = process_spectrum(abs_spectrum[0], abs_spectrum[1], np.abs(abs_spectrum[3]/abs_spectrum[2]), resolution=resolution, temp_type=1, knots_bin = knots_bin,
                                        thres=line_thres, apodization_size=apodization_size)
            abs_spectrum[3] = normalize.new_masks
            
            processed_fluxes1 = copy.deepcopy(normalize.normalized_fluxes)
            
            template_names1 = list(self.templates1.keys())
            n_templates1 = len(template_names1)
            z1, zerr1, r1, chi_eff1 = np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1)
            for i, temp_name in enumerate(template_names1):
                cc_spec_temp = cc_result(abs_spectrum, weight=weight, normalize=normalize, processed_fluxes=processed_fluxes1,
                                        template=self.templates1[temp_name], shifted_template=self.shifted_templates1[temp_name], 
                                        temp_apodization_size=self.temp_apodization_size,
                                        temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres,
                                        z_range=self.z_range, r_thres=r_thres, line_fit=line_fit,
                                        em_lines=em_lines, abs_lines=abs_lines, resolution=resolution)
                self.cc_result[temp_name] = cc_spec_temp
                z1[i], zerr1[i], r1[i], chi_eff1[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                    
            # remove the results with nan-redshift
            if chi_thres:
                nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(chi_eff1<chi_thres)
            else:
                nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))
            template_names1, z1, zerr1, r1, chi_eff1 = np.array(template_names1)[nan_check], z1[nan_check], zerr1[nan_check], r1[nan_check], chi_eff1[nan_check]
            
            # best result among absorption templates
            if len(r1):
                i_best1 = np.nanargmin(chi_eff1)
                best_templates_name1, best_z1, best_zerr1, best_r1, best_chi_eff1 = template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1]
            else:
                best_r1 = np.nan
                
            # emission tempaltes
            normalize = process_spectrum(em_spectrum[0], em_spectrum[1], np.abs(em_spectrum[3]/em_spectrum[2]), resolution=resolution, temp_type=2, knots_bin = knots_bin,
                                        thres=line_thres, apodization_size=apodization_size)
            em_spectrum[3] = normalize.new_masks
            
            processed_fluxes2 = copy.deepcopy(normalize.normalized_fluxes)
            
            template_names2 = list(self.templates2.keys())
            n_templates2 = len(template_names2)
            z2, zerr2, r2, chi_eff2 = np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2)
            for i, temp_name in enumerate(template_names2):
                cc_spec_temp = cc_result(em_spectrum, weight=weight, normalize=normalize, processed_fluxes=processed_fluxes2, 
                                        template=self.templates2[temp_name], shifted_template=self.shifted_templates2[temp_name],
                                        temp_apodization_size=self.temp_apodization_size,
                                        temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres,
                                        z_range=self.z_range, r_thres = r_thres, line_fit=line_fit,
                                        em_lines=em_lines, abs_lines=abs_lines,
                                        resolution=resolution)
                self.cc_result[temp_name] = cc_spec_temp
                z2[i], zerr2[i], r2[i], chi_eff2[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                    
            # remove the results with nan-redshift
            if chi_thres:
                nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(chi_eff2<chi_thres)
            else:
                nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))
            template_names2, z2, zerr2, r2, chi_eff2 = np.array(template_names2)[nan_check], z2[nan_check], zerr2[nan_check], r2[nan_check], chi_eff2[nan_check]

            # best result among absorption templates
            if len(r2):
                i_best2 = np.nanargmin(chi_eff2)
                best_templates_name2, best_z2, best_zerr2, best_r2, best_chi_eff2 = template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2] 
            else:
                best_r2 = np.nan

            # choose the best result
            if prior=='abs':
                if np.isfinite(best_r1):
                    i_best = i_best1
                else:
                    if np.isfinite(best_r2):
                        i_best = i_best2 + len(r1)
                    else:
                        i_best = None
            elif prior=='em':
                if np.isfinite(best_r2):
                    i_best = i_best2+len(r1)
                else:
                    if np.isfinite(best_r1):
                        i_best = i_best1
                    else:
                        i_best = None

            # concatenate the results from absorption and emission templates
            template_names, z, zerr, r, chi_eff = np.concatenate([template_names1, template_names2]), np.concatenate([z1, z2]), np.concatenate([zerr1, zerr2]), np.concatenate([r1,r2]), np.concatenate([chi_eff1, chi_eff2])
            note = np.zeros_like(r).astype(str)
            note[:] = ' '
            if i_best != None:
                note[i_best] = 'best'
            # arange the value in the order of chi_eff
            order = np.flip(np.argsort(r))
            template_names, z, zerr, r, chi_eff, note = template_names[order], z[order], zerr[order], r[order], chi_eff[order], note[order]
            
            table = np.vstack((template_names, z, zerr, r, chi_eff, note))
            column_names = ['template_name', 'z', 'zerr', 'r', 'chi_eff', 'note']
            result = pd.DataFrame(table.T, columns = column_names)
            result = result.astype({'template_name':str, 'z':np.float32, 'zerr':np.float32, 'r':np.float32, 'chi_eff':np.float32, 'note':str})
            

        return result

    def z_multi(self, spectrums, multi_process=4, **kwargs):
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
            result = pd.DataFrame(np.concatenate(result), columns=['best_template', 'z', 'zerr', 'r', 'chi_eff'])
            result.astype({'best_template':str, 'z':np.float32, 'zerr':np.float32, 'r':np.float32, 'chi_eff':np.float32})
            return result
            
            
        else:
            spec_number = np.arange(len(spectrums))
            result = []
            for index in tqdm(spec_number, leave=False, desc='Single Process Progress'):
                singe_result = self.z_single(spectrums[index], output='best', **kwargs)
                result.append(singe_result)
            result = pd.DataFrame(result, columns=['best_template', 'z', 'zerr', 'r', 'chi_eff'])
            result.astype({'best_template':str, 'z':np.float32, 'zerr':np.float32, 'r':np.float32, 'chi_eff':np.float32})
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
            file.write(f'{sub_spec_files[j]} {measured[j][0]} {measured[j][1]} {measured[j][2]} {measured[j][3]} {measured[j][4]}\n')
        file.close()
        print('done')