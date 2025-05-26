import warnings
import os
import glob
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit as cf
from scipy.stats import sigmaclip
from scipy.interpolate import splrep, splev
from scipy.signal.windows import tukey
from scipy.interpolate import CubicSpline
from joblib import Parallel, delayed
from astropy.modeling import models, fitting
from astropy.constants import c
c = c.to_value('km/s')

def return_nan(x):
    return np.nan

# Gaussian function
def gaussian(x, amplitude, mean, stddev):
    '''
    x: input x-values
    amplitude: height of the peak
    mean: position of the center of the peak
    stddev: standard deviation (width) of the peak
    
    Returns the value of the Gaussian function at x
    '''
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

# Multiple Gaussian function
def multiple_gaussian(x, *params):
    '''
    x: input x-values
    params: parameters for the Gaussian functions, in the order of amplitude, mean, and standard deviation
    '''
    n_gaussians = len(params) //3
    y = np.zeros_like(x)
    for i in range(n_gaussians):
        amp = params[i*3]
        mu = params[i*3+1]
        sigma = params[i*3+2]
        y+= amp*np.exp(-0.5*(x-mu)**2/sigma**2)
        
    return y

# Gaussian fit to detect spectral lines
def fit_gaussian_segment(wavelengths, res, center, rend, lend, resolution, weights):
    '''
    wavelengths: wavelengths of a spectrum
    res: residuals of the subtraction of the spectrum and the continuum
    center: center index of a spectral line candidate
    rend: right index of a spectral line candidate
    lend: left index of a spectral line candidate
    
    return True with best-fit Gaussian amplitude, cneter, width, and standard deviation if the fit is successful, False otherwise
    '''
    try:
        line_wavelengths, line_res, line_weights = wavelengths[lend:rend+1], res[lend:rend+1], weights[lend:rend+1]
        weight_filter = (line_weights!=0)
        line_wavelengths, line_res, line_weights = line_wavelengths[weight_filter], line_res[weight_filter], line_weights[weight_filter]
        line_weights = 1/line_weights
        popt, _ = cf(gaussian, line_wavelengths, line_res, sigma=line_weights,
                            p0 = [res[center], line_wavelengths[len(line_wavelengths)//2], (line_wavelengths[-1]-line_wavelengths[0])*0.5])
        amp = popt[0]
        stddev = np.abs(popt[2])
        width = 2 * int(stddev)
        return True, amp, center, width, stddev
    except Exception as e:
        return False, lend, center, rend

# Normalize the template
class process_template:
    warnings.simplefilter('ignore')
    def __init__(self, wavelengths, fluxes, weights, line_identified=False, resolution=3, knots_bin = 100, thres=3, apodization_size = 0.05, n_jobs=-1):
        '''
        wavelengths: wavelengths of a template
        fluxes: fluxes of a template
        weights: weights for the fit, i.e., mask/eflux
        line_identified: if the wavelengths of spectral lines are already specified in the weights
        resolution: resolution of a template
        knots_bin: bin size for the knots
        thres: threshold for line detection
        apodization_size: size of the Tukey window for the apodization
        n_jobs: the number of jobs for parallel processing. If n_jobs = -1, all available resources are used
        '''
        if line_identified: # if the wavelengths of spectral lines are already specified in the weights
            # determine the continuum value
            self.line_identified_trace_continuum(wavelengths, fluxes, weights, knots_bin = knots_bin)
        else:
            # determine the continuum value
            self.trace_continuum(wavelengths,fluxes, weights, resolution=resolution, knots_bin = knots_bin, thres = thres, n_jobs=n_jobs)
        # compute the normalized fluxes
        self.normalized_fluxes = ((fluxes/self.continuum_fluxes)-1)*tukey(len(fluxes), apodization_size)
        self.normalized_fluxes[~np.isfinite(self.normalized_fluxes)] = 0 # replace NaN with 0
        
    def line_identified_trace_continuum(self, wavelengths, fluxes, weights, knots_bin = 200):
        '''
        wavelengths: wavelengths of a template
        fluxes: fluxes of a template
        weights: weights for the fit, i.e., mask/eflux
        knots_bin: bin size for the knots
        '''
        # Knots for the spline fit
        self.knots = np.arange(wavelengths[0]+1e-10, wavelengths[-1]+1e-10, knots_bin)
        
        # Concatenate adjacent knots if one has too many masked points
        concat_edge=[0]
        while len(concat_edge) > 0:
            mask_ratio = np.histogram(wavelengths[weights==0], bins=self.knots)[0]/np.histogram(wavelengths, bins=self.knots)[0]
            concat_edge = np.where(mask_ratio>0.25)[0]+1
            self.knots = np.delete(self.knots, concat_edge)
        
        # Fit the spline to the data
        self.sp_param = splrep(wavelengths, fluxes, t=self.knots, k=5, w = weights)  # k is the degree of the spline

        # Determine the continuum based on the spline fit
        self.continuum_fluxes = splev(wavelengths, self.sp_param)
        # Replace zero continum fluxes with very large number. Otherwise, it will result in NaN in the normalized fluxes
        self.continuum_fluxes[self.continuum_fluxes==0] = 1e+5*np.max(np.abs(fluxes))
        # The end points of the continuum fluxes are replaced with the second and second last points to avoid edge effects
        self.continuum_fluxes[0], self.continuum_fluxes[-1] = self.continuum_fluxes[1], self.continuum_fluxes[-2]
        
    
    def trace_continuum(self, wavelengths, fluxes, weights, resolution, knots_bin = 200, thres=3, n_jobs=-1):
        '''
        wavelengths: wavelengths of a template
        fluxes: fluxes of a template
        weights: weights for the fit, i.e., mask/eflux
        knots_bin: bin size for the knots
        thres: threshold for line detection
        n_jobs: the number of jobs for parallel processing. If n_jobs = -1, all available resources are used
        '''
        # Knots for the spline fit
        self.knots = np.arange(wavelengths[0]+1e-10, wavelengths[-1]+1e-10, knots_bin)
        # Initial spline fit to trace the rough continuum
        sp_param = splrep(wavelengths, fluxes, t=self.knots, w=np.ones_like(fluxes), k=5)  # k is the degree of the spline
        self.continuum_fluxes = splev(wavelengths, sp_param)
        
        # new weights and mask arrays that will be determined based on detected lines
        self.new_weights = copy.deepcopy(weights)
        self.new_masks = copy.deepcopy(weights)
        self.new_masks[self.new_masks!=0] = 1
        # list for the detected lines
        self.lines = []

        # identify the spectral lines and mask them for the fit
        if thres:
            # detect pixels with (flux-continum) larger than thres*std(flux-continum)
            res = fluxes - self.continuum_fluxes
            std = np.nanstd(res)
            detection = np.abs(res) > thres * std

            # Detect spectral line candidates consisting of adjacent detected pixels
            lends, rends = np.where(np.diff(detection.astype(int)) == 1)[0] + 1, np.where(np.diff(detection.astype(int)) == -1)[0] + 1
            if detection[-1]:  # If the last value is True
                rends = np.concatenate((rends, [len(res) - 1]))
            if detection[0]:  # If the first value is True
                lends = np.concatenate(([0], lends))
            centers = (lends + rends) // 2
            
            # Fit the detected line candidates in parallel
            line_results = Parallel(n_jobs=n_jobs)(
                delayed(fit_gaussian_segment)(wavelengths, res, center, rend, lend, resolution, weights)
                for lend, rend, center in zip(lends, rends, centers)
                if rend - lend + 1 > 3
            )
            
            # append spectral lines if the best-fit Gaussian stddev is larger than the resolution
            for result in line_results:
                if not result[0]:
                    continue
                _, amp, center, width, stddev = result
                if stddev * 2 * np.sqrt(2 * np.log(2)) > resolution:
                    self.new_weights[max(0, center - width):min(len(res)-1, center + width + 1)] = 0
                    self.lines.append([wavelengths[max(0, center - width)], wavelengths[min(len(res)-1, center + width + 1)]])
            
            # Concatenate adjacent knots if one has too many masked points
            concat_edge = [0]
            while len(concat_edge) > 0:
                mask_ratio = np.histogram(wavelengths[self.new_weights == 0], bins=self.knots)[0] / np.histogram(wavelengths, bins=self.knots)[0]
                concat_edge = np.where(mask_ratio > 0.25)[0] + 1
                self.knots = np.delete(self.knots, concat_edge)

            # Refit spline with the new weights and determine the final continuum values
            self.sp_param = splrep(wavelengths, fluxes, t=self.knots, k=5, w=self.new_weights)  # k is the degree of the spline
            self.continuum_fluxes = splev(wavelengths, self.sp_param)
            self.continuum_fluxes[self.continuum_fluxes == 0] = 1e+5 * np.max(np.abs(fluxes))
            self.continuum_fluxes[0], self.continuum_fluxes[-1] = self.continuum_fluxes[1], self.continuum_fluxes[-2]

# Normalize the template and contruct a mask for a cross-correlation
class process_spectrum:
    warnings.simplefilter('ignore')
    def __init__(self, wavelengths, fluxes, weights, temp_type, resolution=3, knots_bin=100, thres=3, apodization_size=0.05, n_jobs=-1):
        '''
        wavelengths: wavelengths of a spectrum
        fluxes: fluxes of a spectrum
        weights: weights for the fit, i.e., mask/eflux
        temp_type: type of the template to be cross-correlated. 1 if absorption, 2 if emission
        resolution: resolution of a template
        knots_bin: bin size for the knots
        thres: threshold for line detection
        apodization_size: size of the Tukey window for the apodization
        n_jobs: the number of jobs for parallel processing. If n_jobs = -1, all available resources are used
        '''
        self.wavelengths, self.fluxes, self.weights, self.temp_type = wavelengths, fluxes, weights, temp_type
        self.resolution, self.knots_bin, self.thres = resolution, knots_bin, thres
        self.n_jobs = n_jobs
        # Determine the continuum value
        self.trace_continuum()
        # Contruct a mask for the cross-correlation
        self.gen_mask()
        # Compute the normalized fluxes
        self.normalized_fluxes = ((fluxes/self.continuum_fluxes)-1) * tukey(len(fluxes), apodization_size)
        self.normalized_fluxes[np.isnan(self.normalized_fluxes)] = 0
    
    def trace_continuum(self):
        # Knots for the spline fit
        self.knots = np.arange(self.wavelengths[0] + 1e-10, self.wavelengths[-1] + 1e-10, self.knots_bin)
        # Initial spline fit to trace the rough continuum
        sp_param = splrep(self.wavelengths, self.fluxes, t=self.knots, w=np.ones_like(self.fluxes), k=5)
        self.continuum_fluxes = splev(self.wavelengths, sp_param)
        
        # New weights that will be determined based on detected lines
        self.new_weights = copy.deepcopy(self.weights)

        # Identify the spectral lines and mask them for the fit
        if self.thres:
            # Detect pixels with (flux-continum) larger than thres*std(flux-continum)
            res = self.fluxes - self.continuum_fluxes
            std = np.nanstd(res)
            detection = np.abs(res) > self.thres * std

            if len(res[detection]) > 0:
                # Detect spectral line candidates consisting of adjacent detected pixels
                lends, rends = np.where(np.diff(detection.astype(int)) == 1)[0] + 1, np.where(np.diff(detection.astype(int)) == -1)[0] + 1
                if detection[-1]:
                    rends = np.concatenate((rends, [len(res) - 1]))
                if detection[0]:
                    lends = np.concatenate(([0], lends))
                centers = (lends + rends) // 2

                # Fit the detected line candidates in parallel
                line_results = Parallel(n_jobs=self.n_jobs)(
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

                # Indentify spectral lines if best-fit Gaussian stddev is larger than the resolution
                for result in line_results:
                    if not result[0]:
                        continue
                    _, amp, center, width, stddev = result
                    if stddev * 2 * np.sqrt(2 * np.log(2)) > self.resolution:
                        self.new_weights[max(0, center - width):min(len(res)-1, center + width + 1)] = 0
            
            # Concatenate adjacent knots if one has too many masked points
            concat_edge = [0]
            while len(concat_edge) > 0:
                mask_ratio = np.histogram(self.wavelengths[self.new_weights == 0], bins=self.knots)[0] / np.histogram(self.wavelengths, bins=self.knots)[0]
                concat_edge = np.where(mask_ratio > 0.25)[0] + 1
                self.knots = np.delete(self.knots, concat_edge)

            # Refit spline with the new weights and determine the final continuum values
            self.sp_param = splrep(self.wavelengths, self.fluxes, t=self.knots, k=5, w=self.new_weights)
            self.continuum_fluxes = splev(self.wavelengths, self.sp_param)
            self.continuum_fluxes[self.continuum_fluxes == 0] = 1e+5 * np.max(np.abs(self.fluxes))
            self.continuum_fluxes[0], self.continuum_fluxes[-1] = self.continuum_fluxes[1], self.continuum_fluxes[-2]

    def gen_mask(self):
        # New mask for the cross-correlation
        self.new_masks = copy.deepcopy(self.weights)
        self.new_masks[self.new_masks != 0] = 1
        
        # Identify the spectral lines and mask them for the cross-correlation
        if self.thres:
            # Detect pixels with (flux-continum) larger than thres*std(flux-continum)
            res = self.fluxes - self.continuum_fluxes
            std = np.nanstd(res)
            detection = np.abs(res) > self.thres * std

            if len(res[detection]) > 0:
                # Detect spectral line candidates consisting of adjacent detected pixels
                lends, rends = np.where(np.diff(detection.astype(int)) == 1)[0], np.where(np.diff(detection.astype(int)) == -1)[0] + 2
                if detection[-1]:
                    rends = np.concatenate((rends, [len(res) - 1]))
                if detection[0]:
                    lends = np.concatenate(([0], lends))
                centers = (lends + rends) // 2

                # Fit the detected line candidates in parallel
                line_results = Parallel(n_jobs=self.n_jobs)(
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

                # Identify spectral lines if the best-fit Gaussian width is larger than the resolution and mask them if their type differs from the template
                # or if the amplitude is larger than 3*std but width smaller than the resoltuion (which may indicate a bad pixel due to a cosmic ray)
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
                            

        
        


# Interpolation based on spline
def resampler(wavelengths, values, new_wavelengths):
    cs = CubicSpline(wavelengths, values, extrapolate=True)
    new_values = cs(new_wavelengths)

    return new_values

# Interpolation for the discrete value (e.g. mask)
def discrete_resampler(wavelengths, values, new_wavelengths):
    new_values = np.interp(new_wavelengths, wavelengths, values, left=0, right=0)
    new_values = np.round(new_values).astype(int)
    return new_values

# Normalize the templates with the unifrom weights and shifts them along the redshift
def shift_templates(templates, z_range=[-0.1,2], apodization_size=0.05, knots_bin = 100, thres=3, resolution=3, n_jobs=-1):
    '''
    templates: dictionary of a set of templates to be shifted
    z_range: redshift range for the template to be shifted
    apodization_size: size of the Tukey window for the apodizating the templates
    knots_bin: bin size for the knots
    thres: threshold for line detection for the templates
    n_jobs: the number of jobs for parallel processing. If n_jobs = -1, all available resources are used
    
    return: dictionary of the [radial velocity correspond to shift (km/s) (N),
                                wvaelength of the shifted templates (M),
                                flux of shifted templates (N, M)
    '''
    shifted_templates = {}
    for temp_name in templates.keys():
        # Normalize the template
        temp = templates[temp_name][0]
        normalize=process_template(temp[0], temp[1], np.ones_like(temp[1]), knots_bin=knots_bin, thres=thres, apodization_size=apodization_size, resolution=resolution, n_jobs=n_jobs)
            
        # Define log pixel scale
        log_wavelengths = np.log10(temp[0])
        log_bin = np.median(log_wavelengths[1:]-log_wavelengths[:-1])

        # Prepare a wavelength array
        n_left, n_right = -int(np.log10(z_range[0]-0.2*(1+z_range[0])+1)/log_bin), int(np.log10(z_range[1]+0.2*(1+z_range[1])+1)/log_bin)
        n_shift = n_left+n_right+1 # 1 for the zero-redshift
        n_pixel = temp.shape[1]+n_left+n_right

        # Construct the shifted wavelength array
        min_log_wavelengths = np.log10(temp[0,0])-n_left*log_bin
        shifted_log_wavelengths = min_log_wavelengths + np.arange(n_pixel)*log_bin
        shifted_wavelengths = pow(10,shifted_log_wavelengths)

        # Construct the shifted fluxes
        shifted_fluxes = np.zeros((n_shift, n_pixel))
        for i in range(n_shift):
            shifted_fluxes[i,i:i+temp.shape[1]] = normalize.normalized_fluxes
        
        # Contruct the radial velocity array corresponding to the shift
        min_log_vel = -n_left*log_bin
        shifted_vels = c*(pow(10,np.ones(n_shift)*min_log_vel+np.arange(n_shift)*log_bin)-1)
        shifted_templates[temp_name] = [shifted_vels, shifted_wavelengths, shifted_fluxes]
    return shifted_templates

import copy

# remove the range of the spectrum with low S/N
def clean_spectrum(spec, window, sn):
    '''
    spec: spectrum array (4, n_pixel)
    window: size of the window for the S/N calculation
    sn: S/N threshold for the spectrum
    '''
    pixel_scale = np.median(spec[0,1:]-spec[0,:-1])
    window = int(window/pixel_scale)

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

# Fit the emission lines with gaussians
def sub_fit_multiline(used_pixels, observed_wavelength_min, observed_wavelength_max,
                      wavelengths, fluxes, weights, line, z, z_min, z_max,
                      stddev_resolution, temp_type, cc_width):
    '''
    used_pixels: indication for the pixels to be used in the fit
    observed_wavelength_min, observed_wavelength_max: min and max wavelengths of the spectrum
    wavelengths: wavelengths of a spectrum
    fluxes: fluxes of a spectrum
    weights: weights for the fit, i.e., mask*eflux
    line: spectral line to be fitted
    z: redshift candidate of a spectrum
    z_min, z_max: min and max redshift candidates of a spectrum
    stddev_resolution: standard deviation of the resolution
    temp_type: type of the template to be cross-correlated. 1 if absorption, 2 if emission
    cc_width: width of a cross-correlation signal peak
    
    return the number of detected lines, redshift of spectrl lines, model line fluxes
    '''
    ndetect = 0 # number of detected lines
    model_fluxes = np.zeros_like(wavelengths) # model line fluxes
    z_lines = [] # list of lines' redshifts
    
    if isinstance(line, float): # if 'line' is a single line (no other spectral lines around it)
        # Redshifted line wavelength
        redshifted_wavelength = line * (1 + z)
        redshifted_wavelength_min, redshifted_wavelength_max = line*(1+z_min), line*(1+z_max)
        
        # If the redshifted line wavelength is within the observed range,
        # fit a Gaussian to the spectrum around the redshifted line wavelength
        if observed_wavelength_min <= redshifted_wavelength <= observed_wavelength_max:
            # pixels used for the fit (~width of the cross-correlation signal peak)
            wavelength_left, wavelength_right = redshifted_wavelength*(1-cc_width/c), redshifted_wavelength*(1+cc_width/c)
            pix4fit = (wavelengths>wavelength_left)&(wavelengths<wavelength_right)&used_pixels
            
            # Initial guess and bounds for the Gaussian parameters 
            # amplitude: F((1+z)*\lambda_{line}); (-inf,inf)
            # mean: (1+z)*\lambda_{line}; (1+z_min, 1+z_max)*\lambda_{\line} 
            # stddev: ~broadening due to stellar velocity dispersion; (spectral resoltuion, 2*initial guess) 
            std0 = np.sqrt((redshifted_wavelength*cc_width/c)**2-2*stddev_resolution**2)
            p0 = [fluxes[np.argmin(np.abs(wavelengths - redshifted_wavelength))], redshifted_wavelength, std0]
            bounds = ([-np.inf, redshifted_wavelength_min, stddev_resolution],
                        [np.inf, redshifted_wavelength_max, 2*std0])
            
            # fit a Gaussian
            try:
                popt, pcov = cf(multiple_gaussian, wavelengths[pix4fit] , fluxes[pix4fit], sigma=weights[pix4fit], p0=p0,
                                bounds=bounds, absolute_sigma=True)
                # if the type of the template is emission,
                # and the best fit amplitude is larger than zero (emission)
                # we regard the emission line is detected
                if popt[0]>0: 
                    model_fluxes += multiple_gaussian(wavelengths, *popt)
                    z_line = (popt[1]-line)/line
                    z_lines.append(z_line)
                    ndetect += 1
                # if the type of the template is absorption,
                # and the best fit amplitude is smaller than zero (emission)
                # we regard the absorption line is detected
                elif (temp_type==1)&(popt[0])<0:
                    model_fluxes += multiple_gaussian(wavelengths, *popt)
                    z_line = (popt[1]-line)/line
                    z_lines.append(z_line)
                    ndetect += 1
            except Exception as e:
                pass
            
    elif isinstance(line, list): # if 'line' is a list of adjacent spectral lines
        # lines that are within the observed range
        valid_lines = []
        for l in line:
            redshifted_wavelength = l*(1+z)
            if observed_wavelength_min <= redshifted_wavelength <= observed_wavelength_max:
                valid_lines.append(l)
        valid_lines = np.array(valid_lines)
        
        # If there are valid lines, fit a Gaussian to the spectrum around the redshifted line wavelengths
        # If one of the lines is not detected, we remove it from the list and fit the remaining lies
        while(len(valid_lines)):
            # pixels for the fit (~width of the cross-correlation signal peak)
            wavelength_left, wavelength_right = valid_lines[0]*(1+z)*(1-5*cc_width/c), valid_lines[-1]*(1+z)*(1+5*cc_width/c)
            pix4fit = (wavelengths>wavelength_left)&(wavelengths<wavelength_right)&used_pixels
            
            # Initial guess and bounds for the Gaussian parameters 
            # amplitude: F((1+z)*\lambda_{line}); (-inf,inf)
            # mean: (1+z)*\lambda_{line}; (1+z_min, 1+z_max)*\lambda_{\line} 
            # stddev: ~broadening due to stellar velocity dispersion; (spectral resoltuion, 2*initial guess) 
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
            
            # fit multtiple Gaussians
            try:
                popt, pcov = cf(multiple_gaussian, wavelengths[pix4fit] , fluxes[pix4fit], sigma=weights[pix4fit], p0=p0,
                                bounds=bounds)
                
                # if the type of the template is emission,
                # and the best fit amplitudes are larger than zero (emission)
                # we regard the emission lines are detected
                if temp_type==2:
                    detect = popt[np.arange(len(valid_lines))*3]>0
                # if the type of the template is absorption,
                # and the best fit amplitudes are smaller than zero (absorption)
                # we regard the absorption lines are detected
                elif temp_type==1:
                    detect = popt[np.arange(len(valid_lines))*3]<0
                    
                # if one of the lines is not detected, we remove it from the list and go back to the first step
                if len(np.where(~detect)[0]):
                    valid_lines = valid_lines[detect]
                # otherwise, we regard all the lines are detected
                # and break the loop
                else:
                    ndetect += len(valid_lines)
                    model_fluxes += multiple_gaussian(wavelengths, *popt)
                    z_lines.append(z_line)
                    break
                
            except Exception as e:
                break
            
    # redshift of the spectral lines
    if len(z_lines):
        z_line = np.mean(np.array(z_lines))
    else:
        z_line = np.nan   
    return ndetect, z_line, model_fluxes

def fit_multiline(wavelengths, fluxes, weights, lines, z, zerr, cc_width, resolution, temp_type, n_jobs=-1):
    '''
    wavelengths: wavelengths of a spectrum
    fluxes: fluxes of a spectrum
    weights: weights for the fit, i.e., mask*eflux
    lines: list of spectral lines to be fitted
    z: redshift candidate of a spectrum
    zerr: uncertainty of the redshift candidate
    cc_width: width of a cross-correlation signal peak
    resolution: resolution of a template
    temp_type: type of the template to be cross-correlated. 1 if absorption, 2 if emission
    return: effective chi square of the fit, model line fluxes
    n_jobs: the number of jobs for parallel processing. If n_jobs = -1, all available resources are used
    '''
    used_pixels = (weights!=0) # not-masked pixels
    
    observed_wavelength_min, observed_wavelength_max = wavelengths[0], wavelengths[-1]
    z_min, z_max = z-3*zerr, z+3*zerr
    stddev_resolution = resolution/(2*np.sqrt(np.log(2)*2))
    
    # fit the lines in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(sub_fit_multiline)(used_pixels, observed_wavelength_min,
                                   observed_wavelength_max,
                                   wavelengths, fluxes, weights, line, z, z_min, z_max,
                                   stddev_resolution, temp_type, cc_width) for line in lines)
    
    # sum the results
    ndetect = sum(res[0] for res in results) # detected lines
    model_fluxes = np.sum([res[2] for res in results], axis=0) # model line fluxes
    chi_eff = np.sum(((fluxes[used_pixels]-model_fluxes[used_pixels])/weights[used_pixels])**2)/(len(fluxes[used_pixels])-3*ndetect) # effective chi square
    
    return chi_eff, model_fluxes

# Zoom-in cross-correlation
def sub_interp_fluxes(idx_cc, spectrum, proc_spec, ext_wavelengths, shifted_wavelengths, shifted_vels,
                      template_spectrum, template_lines, ext_fit_weights, temp_knots_bin, temp_resolution,
                      temp_line_thres, temp_apodization_size, n_jobs=-1):
    '''
    idx_shift: index of the cross-correlation signals to be computed
    spectrum: a spectrum array (4, n_pixel)
    proc_spec: class of process_spectrum for the spectrum
    ext_wavelengths: wavelengths of the extended spectrum
    shifted_wavelengths: wavelengths of the shifted template
    shifted_vels: radial velocities of the shifted template
    template_spectrum: template spectrum (3, n_pixel)
    template_lines: the list of blue- and red-end of spectral lines' wavelenegths in the template
    ext_fit_weights: weights for tracing the continuum of the template, i.e., mask/eflux of the extedned spectrum
    temp_resolution: the spectral resolution of the template
    temp_knots_bin: the size of the knots for the template
    temp_line_thres: threshold for line detection
    temp_apodization_size: size of the Tukey window for the apodization for the template
    
    return
    pix_overlap: the pixels of the spectrum overlapped with the template.
    normalized resampled template fluxes: normalized resampled template fluxes
    chi_eff: effective chi square between the spectrum and the shifted template at idx_cc
    '''
    
    # Resample the template spectrum where the template and the spectrum overlap
    pix_overlap = ((ext_wavelengths>=shifted_wavelengths[idx_cc])&(ext_wavelengths<=shifted_wavelengths[idx_cc+template_spectrum.shape[1]-1]))
    new_template_wavelengths = ext_wavelengths[pix_overlap]
    resampled_template_fluxes = resampler(shifted_wavelengths[idx_cc:idx_cc+template_spectrum.shape[1]],
                                          template_spectrum[1], new_template_wavelengths)
    
    # Construct the weights for the fit, which similar to the weights used for processing the spectrum, but mask the blue/redshifted spectral lines,
    # to trace the continuum of the template (process_template)
    new_fit_weights_at_idx_cc = copy.deepcopy(ext_fit_weights[pix_overlap])
    for line in template_lines:
        new_fit_weights_at_idx_cc[(new_template_wavelengths>line[0]*(1+shifted_vels[idx_cc]/c))
                                  &(new_template_wavelengths<line[1]*(1+shifted_vels[idx_cc]/c))] = 0
    
    # Process the resampled template (normalization) based on the new weights (new_fit_weights_at_idx_cc)
    pro_temp = process_template(new_template_wavelengths, resampled_template_fluxes,
                                     new_fit_weights_at_idx_cc, line_identified=True, resolution=temp_resolution, 
                                     knots_bin=temp_knots_bin, thres = temp_line_thres,
                                     apodization_size=temp_apodization_size, n_jobs=n_jobs)
    
    # Compute the effective chi square between the spectrum and the shifted template
    # The spectrum overlapped with the template
    lamb_min, lamb_max = max(spectrum[0,0], new_template_wavelengths[0]), min(spectrum[0,-1], new_template_wavelengths[-1])
    overlap_spec_region = (spectrum[0]>=lamb_min)&(spectrum[0]<=lamb_max)
    overlap_spec = spectrum[:,overlap_spec_region]
    # The template overlapped with the spectrum
    overlap_temp_region = (new_template_wavelengths>=lamb_min)&(new_template_wavelengths<=lamb_max)
    
    # Residual in the overlapped region: (M/dF)^2*(G-T*G_conti/T_conti)^2 (see Kim et al. 2025)
    T_ = resampled_template_fluxes[overlap_temp_region]*proc_spec.continuum_fluxes[overlap_spec_region]/pro_temp.continuum_fluxes[overlap_temp_region]
    overlap_res = ((overlap_spec[3]/overlap_spec[2])**2)*((overlap_spec[1]-T_)**2)
    # Residual in the non-overlapped region: (M/dF)^2*(G-G_conti)^2 (i.e., T/T_conti-1=0)
    # the normalized spectrum flux is cross-correlated with the array 
    # whose values are normalized template flux in the region where the template and the spectrum overlap and zeros in non-overlapped region
    nonoverlap_res = ((spectrum[3]/spectrum[2])**2)*((spectrum[1]-proc_spec.continuum_fluxes)**2)
    
    # effective chi square (=sum(residual)/(N-3*n_knots_eff-1))
    # where N is the number of overlapped pixels in the spectrum
    # n_knots_eff is the number of knots used in process_template within the overlapped region
    # see Kim et al. 2025
    res = np.concatenate([nonoverlap_res[spectrum[0]<lamb_min], overlap_res, nonoverlap_res[spectrum[1]>lamb_max]])
    n_knots_eff = len(pro_temp.knots[(pro_temp.knots>=lamb_min)&(pro_temp.knots<lamb_max)])
    chi_eff = np.sum(res)/(np.sum((overlap_spec[3]))-n_knots_eff-1)
    
    return pix_overlap, pro_temp.normalized_fluxes, chi_eff

# Cross-correlate the spectrum with the template
class cc_result:
    def __init__(self, spectrum, proc_spec, template, shifted_template,
                 temp_apodization_size=0.05, temp_knots_bin=100, temp_line_thres=3,
                 temp_resolution=3, z_range=[-0.01,2], line_fit = True,
                 em_lines=[2799.117, 3727.30, 4102.89, 4341.68, 4861.33, [4958.91, 5006.84], [6548.06, 6562.82, 6583.57], [6716.440, 6730.815]], 
                 resolution=3,
                 n_jobs=-1):
        '''
        spectrum: a spectrum array (4, n_pixel)
        proc_spec: class of process_spectrum for the spectrum
        template: template spectrum (3, n_pixel)
        shifted_template: the dictionary of the shifted template constructed by shift_templates
        temp_apodization_size: size of the Tukey window for the apodization for the template
        temp_knots_bin: the size of the knots for the template
        temp_line_thres: threshold for line detection for the template
        temp_resolution: the spectral resoltuion of the template
        z_range: redshift range for the shifted template
        r_thres: threshold for the cross-correlation signal
        line_fit: if True, line fitting is performed for redshift measurement for the emission template
        em_lines: the list of emission lines used for the line fitting for the redshift measurement
        resolution: resolution of the template
        n_jobs: the number of jobs for parallel processing. If n_jobs = -1, all available resources are used
        '''
        warnings.simplefilter('ignore')
        self.spectrum, self.proc_spec = spectrum, proc_spec
        self.template, self.shifted_template  = template, shifted_template
        self.temp_apodization_size, self.temp_knots_bin, self.temp_line_thres = temp_apodization_size, temp_knots_bin, temp_line_thres
        self.temp_resolution, self.z_range = temp_resolution, z_range
        self.line_fit = line_fit
        self.em_lines, self.resolution = em_lines, resolution
        self.shifted_vels, self.shifted_wavelengths, self.shifted_fluxes, self.template_spectrum = self.shifted_template[0], self.shifted_template[1], self.shifted_template[2], self.template[0]
        self.n_jobs = n_jobs
        # self.cross_correlate()
        try: # perfrom cross-correlation
            self.cross_correlate()
        except: # if the cross-correlation fails, set the results to NaN
            self.z, self.zerr, self.r, self.chi_eff  = np.nan, np.nan, np.nan, np.nan
    
    # Perfrom an inverse-variance weighted cross-correlation
    def cross_correlate(self):
        self.initial_broad_cross_correlation()
        self.define_zoomed_in_cross_correlation_region()
        if len(self.peak_ranges):
            self.zoomed_in_cross_correlation()
            self.determine_z()
        else:
            self.cc = self.cc0
            self.z, self.zerr, self.r, self.chi_eff  = np.nan, np.nan, np.nan, np.nan
    
    # initial broad cross-correlation
    def initial_broad_cross_correlation(self):
        # overlapped wvavelength range between the spectrum and the template
        overlap_template_wavelengths = self.shifted_wavelengths[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])]
        
        # resample the spectrum to math its pixel scale with that of the template
        # arrays for fluxes, weights, and massks for the resampled spectrum
        self.resampled_spectrum_fluxes, self.resampled_spectrum_weights, self.resampled_spectrum_masks = np.zeros_like(self.shifted_wavelengths), np.ones_like(self.shifted_wavelengths), np.ones_like(self.shifted_wavelengths)
        # resample the spectrum in the overlapped region
        overlap_spectrum_fluxes = resampler(self.spectrum[0], self.proc_spec.normalized_fluxes, overlap_template_wavelengths)
        overlap_spectrum_weights = resampler(self.spectrum[0], np.abs(self.proc_spec.continuum_fluxes/self.spectrum[2]), overlap_template_wavelengths)
        overlap_spectrum_masks = discrete_resampler(self.spectrum[0], self.spectrum[3], overlap_template_wavelengths)
        # concatenate the resampled spectrum with the non-overlapped region
        self.resampled_spectrum_fluxes[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_spectrum_fluxes
        self.resampled_spectrum_weights[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_spectrum_weights
        self.resampled_spectrum_masks[(self.shifted_wavelengths>self.spectrum[0,0])&(self.shifted_wavelengths<self.spectrum[0,-1])] = overlap_spectrum_masks

        # cross-correlate the resample spectrum with the shifted template (initial broad cross-correlation)
        self.cc0 = np.matmul(self.shifted_fluxes, self.resampled_spectrum_masks*self.resampled_spectrum_weights**2*self.resampled_spectrum_fluxes)
        

    # find the peaks in the initial broad cross-correlation signals
    def define_zoomed_in_cross_correlation_region(self):
        # radial velocity lags and initial broad cross-correlation signals within the input redshift range 
        cz_range = np.array(self.z_range)*c
        cc0_within_cz_range = self.cc0[(self.shifted_vels>cz_range[0])&(self.shifted_vels<cz_range[1])]
        lags_within_cz_range = self.shifted_vels[(self.shifted_vels>cz_range[0])&(self.shifted_vels<cz_range[1])]
        
        # find the maximum of the initial broad cross-correlation signals and the corresponding lag within the input redshift range
        idx_max = np.nanargmax(cc0_within_cz_range)
        cc0_max, lags_max = cc0_within_cz_range[idx_max], lags_within_cz_range[idx_max] # estimates a peak
        
        # find lags where the cross-correlation signals are larger than 0.5*cc0_max or 3*std(cc0)
        # and the radial velocities are within the input redshift range
        possible_peak_lags = (((self.cc0 >= 0.5*cc0_max))| (self.cc0 >= 3*np.std(self.cc0)))&((self.shifted_vels>=cz_range[0])&(self.shifted_vels<=cz_range[1]))

        # find peak candidates by merging adjacent possible_peak_lags==True
        # and find the left and right edges of the peak candidates
        idxs_peak_candidate_left_ends, idxs_peak_candidate_right_ends = np.where(np.diff(possible_peak_lags.astype(int)) == 1)[0] + 1, np.where(np.diff(possible_peak_lags.astype(int)) == -1)[0] + 1
        if possible_peak_lags[-1] == True: # possible_peak_lags = (..., True, True, True)
            idxs_peak_candidate_right_ends = np.concatenate((idxs_peak_candidate_right_ends, np.array([len(self.cc0)-1])))
        if possible_peak_lags[0] == True: # possible_peak_lags = (True, True, True, ...)
            idxs_peak_candidate_left_ends = np.concatenate((np.array([0]), idxs_peak_candidate_left_ends))
        idxs_peak_candidate_right_ends -= 1
        idxs_peak_candidate_centers = (idxs_peak_candidate_left_ends+idxs_peak_candidate_right_ends) // 2

        # define the region where the zoomed-in cross-correlation will be performed by fitting the Gaussian to the cross-correlation signals for the peak candidates
        self.peak_ranges = [] # left end right ends of the peak candidates will be stored here
        n = 0 # number of peaks for the zoomed-in cross-correlation
        for i_peak in np.argsort(self.cc0[idxs_peak_candidate_centers])[-1::-1]: # sort the peak candidates in descending order in heights
            if n>5: # the number of peaks for the zoomed-in cross-correlation is limited to 5
                break
            
            # define the range for the gaussian fitting
            center_lag4fit, width_lag4fit = self.shifted_vels[idxs_peak_candidate_centers[i_peak]], 0.5*(self.shifted_vels[idxs_peak_candidate_right_ends[i_peak]]-self.shifted_vels[idxs_peak_candidate_left_ends[i_peak]])
            leftend_lag4fit, rightend_lag4fit = center_lag4fit-3*width_lag4fit, center_lag4fit+3*width_lag4fit
            cc_in_peak, lags_in_peak = self.cc0[(self.shifted_vels>leftend_lag4fit)&(self.shifted_vels<rightend_lag4fit)], self.shifted_vels[(self.shifted_vels>leftend_lag4fit)&(self.shifted_vels<rightend_lag4fit)]
            
            # fit a Gaussian to the initial broad cross-correlation signals
            try:
                self.fit_peak_range = fitting.LevMarLSQFitter()
                self.gaussian_peak_range0 = models.Gaussian1D(amplitude=self.cc0[idxs_peak_candidate_centers[i_peak]], mean=self.shifted_vels[idxs_peak_candidate_centers[i_peak]], stddev=self.shifted_vels[idxs_peak_candidate_centers[i_peak]]-self.shifted_vels[idxs_peak_candidate_left_ends[i_peak]-1])
                self.gaussian_peak_range = self.fit_peak_range(self.gaussian_peak_range0, lags_in_peak, cc_in_peak)
                peak_range = [self.gaussian_peak_range.mean.value-2*self.gaussian_peak_range.stddev.value,
                                    self.gaussian_peak_range.mean.value+2*self.gaussian_peak_range.stddev.value]
            except:
                continue
            
            # If the best-fit Gaussian is not a real peak, i.e., the amplitude is negative or the mean is outside the input redshift range,
            # or the number of pixels in the range is less than 1, skip this peak candidate
            if (self.gaussian_peak_range.amplitude.value < 0) | (self.gaussian_peak_range.mean.value < cz_range[0]) | (self.gaussian_peak_range.mean.value>cz_range[1]) | (len(np.where((self.shifted_vels>peak_range[0])&(self.shifted_vels<peak_range[1]))[0])<1):
                    continue
            
            # If the best-fit Gaussian is a real peak, add the left and right ends of the peak candidate to the list
            # and increase the number of peaks for the zoomed-in cross-correlation
            self.peak_ranges.append(peak_range)
            n +=1
        
    
    # zoomed-in cross-correlation
    def zoomed_in_cross_correlation(self):
        # Extend the spectrum to match its wavelength range with that of the shifted template
        # The wavelengths of the extended spectrum outside the spectrum is the same as those of the shifted template
        # Fluxes, weights, and masks in the blue and red regions outside the spectrum are filled with zeros, ones, and ones, respectively
        blue_wavelengths_outside_the_spectrum, red_wavelengths_outside_the_spectrum = self.shifted_wavelengths[self.shifted_wavelengths<self.spectrum[0,0]], self.shifted_wavelengths[self.shifted_wavelengths>self.spectrum[0,-1]]
        self.ext_wavelengths = np.concatenate([blue_wavelengths_outside_the_spectrum, self.spectrum[0], red_wavelengths_outside_the_spectrum])
        self.ext_fluxes, self.ext_weights, self.ext_masks, self.ext_fit_weights = np.zeros_like(self.ext_wavelengths), np.ones_like(self.ext_wavelengths), np.ones_like(self.ext_wavelengths), np.ones_like(self.ext_wavelengths)
        self.ext_fluxes[(self.ext_wavelengths>=self.spectrum[0,0])&(self.ext_wavelengths<=self.spectrum[0,-1])] = copy.deepcopy(self.proc_spec.normalized_fluxes)
        self.ext_fit_weights[(self.ext_wavelengths>=self.spectrum[0,0])&(self.ext_wavelengths<=self.spectrum[0,-1])] = np.abs(self.spectrum[3]/self.spectrum[2])
        self.ext_weights[(self.ext_wavelengths>=self.spectrum[0,0])&(self.ext_wavelengths<=self.spectrum[0,-1])] = np.abs(self.proc_spec.continuum_fluxes/self.spectrum[2])
        self.ext_masks[(self.ext_wavelengths>=self.spectrum[0,0])&(self.ext_wavelengths<=self.spectrum[0,-1])] = self.spectrum[3]
        
        # Perform the zoomed-in cross-correlation
        self.cc = copy.deepcopy(self.cc0)
        self.chi_effs = np.zeros_like(self.cc0)*np.nan # arrays to store effective chi squares between the spectrum and the template at each lag where the zoomed-in cross-correlation is performed
        for m, peak_range in enumerate(self.peak_ranges):
            # Indices of the radial velocity lags where the zoomed-in cross-correlation will be performed
            idxs_within_peak = np.where((self.shifted_vels>peak_range[0])&(self.shifted_vels<peak_range[1]))[0]
            
            # Arrays to store the normalized resampled template fluxes and the effective chi squares at each lag
            # where the zoomed-in cross-correlation is performed
            normalized_resampled_template_fluxes_fluxes = np.zeros((len(idxs_within_peak),len(self.ext_fluxes))) # 
            chi_effs_ = np.zeros(len(idxs_within_peak))
            
            # Define spectral lines in the template based on the wieght from the extended spectrum
            # by using the process_template function at a certain lag within the peak
            i = int(np.median(idxs_within_peak))
            overlap_region_in_spectrum = ((self.ext_wavelengths>=self.shifted_wavelengths[i])&(self.ext_wavelengths<=self.shifted_wavelengths[i+self.template_spectrum.shape[1]-1]))
            overlap_template_wavelengths = self.ext_wavelengths[overlap_region_in_spectrum]
            resampled_template_fluxes = resampler(self.shifted_wavelengths[i:i+self.template_spectrum.shape[1]], self.template_spectrum[1], overlap_template_wavelengths)
            template_lines = process_template(overlap_template_wavelengths, resampled_template_fluxes, self.ext_fit_weights[overlap_region_in_spectrum],
                                              resolution = self.temp_resolution, knots_bin = self.temp_knots_bin,
                                              thres = self.temp_line_thres, apodization_size=self.temp_apodization_size, n_jobs=self.n_jobs).lines
            # Store the blue- and red- end of spectral lines in the template
            # The wavelengths of the spectral lines in the template are shifted to rest frame
            template_lines_at_rest = []
            for line in template_lines:
                template_lines_at_rest.append([line[0]/(1+self.shifted_vels[i]/c), line[1]/(1+self.shifted_vels[i]/c)])
            
            # resample and process the template at each lag within the peak in parallel
            process_temp_result = Parallel(n_jobs=-1)(delayed(sub_interp_fluxes)(idx_cc=i, spectrum=self.spectrum, proc_spec=self.proc_spec,
                                                                            ext_wavelengths = self.ext_wavelengths, shifted_wavelengths=self.shifted_wavelengths,
                                                                            shifted_vels = self.shifted_vels, template_spectrum = self.template_spectrum,
                                                                            template_lines = template_lines_at_rest, ext_fit_weights = self.ext_fit_weights,
                                                                            temp_knots_bin = self.temp_knots_bin, temp_resolution = self.temp_resolution,
                                                                            temp_line_thres = self.temp_line_thres, temp_apodization_size = self.temp_apodization_size) for i in idxs_within_peak)
            for n, (pix_overlap_at_shift, normalized_resampled_template_flux_at_shift, chi_eff_at_shift) in enumerate(process_temp_result):
                normalized_resampled_template_fluxes_fluxes[n, pix_overlap_at_shift] = normalized_resampled_template_flux_at_shift
                chi_effs_[n] = chi_eff_at_shift

            # Perform the zoomed-in cross-correlation and replace the initial broad cross-correlation signals
            # with the zoomed-in cross-correlation signals at each lag within the peak
            weight_cc = np.matmul(normalized_resampled_template_fluxes_fluxes,self.ext_masks*self.ext_weights**2*self.ext_fluxes)
            self.cc[idxs_within_peak] = weight_cc
            # Also add the effective chi squares at each lag within the peak
            self.chi_effs[idxs_within_peak] = chi_effs_
    
    # Determine the redshift of the spectrum based on the zoomed-in cross-correlation signals
    # by fitting a Gaussian to the cross-correlation signals within the peak ranges
    def determine_z(self):
        # Fit a Gaussian to the cross-correlation signals within each peak range
        self.bestfit_gaussians_cc, self.fitters= [],[]
        for i in range(len(self.peak_ranges)):
            # Find the indices of the radial velocity lags within the peak range
            peak_range = self.peak_ranges[i]
            idxs_lags_within_peak = np.where((self.shifted_vels>peak_range[0])&(self.shifted_vels<peak_range[1]))[0]
            # If the number of indices is less than 5, extend the range by adding one index
            # to the left and right ends for the Gaussian fitting
            # to avoid the fitting failure
            if len(idxs_lags_within_peak) <5:
                idxs_lags_within_peak = np.insert(idxs_lags_within_peak, 0, max(idxs_lags_within_peak[0]-1,0))
                idxs_lags_within_peak = np.insert(idxs_lags_within_peak, -1, min(idxs_lags_within_peak[-1]+1,len(self.cc)-1))
            
            # Determine cross-correlation signals and radial velocity lags within the peak range
            cc_in_peak = self.cc[idxs_lags_within_peak]
            lags_in_peak = self.shifted_vels[idxs_lags_within_peak]
            # Remove NaN values from the cross-correlation signals and lags
            nan_filtering = (~np.isnan(cc_in_peak))&((~np.isnan(lags_in_peak)))
            cc_in_peak, lags_in_peak = cc_in_peak[nan_filtering], lags_in_peak[nan_filtering]
            
            # Fit a Gaussian
            fit_peak = fitting.LevMarLSQFitter(calc_uncertainties=True)
            initial_gaussian = models.Gaussian1D(amplitude=np.nanmax(cc_in_peak), mean=np.median(lags_in_peak), stddev=0.5*(lags_in_peak[-1]-lags_in_peak[0]))
            bestfit_gaussian = fit_peak(initial_gaussian, lags_in_peak, cc_in_peak)
            # If the Gaussian fitting is successful, i.e., the covariance matrix is not None,
            # store the best-fit Gaussian and the fitting result
            if type(fit_peak.fit_info['param_cov']) != type(None):
                self.bestfit_gaussians_cc.append(bestfit_gaussian)
                self.fitters.append(fit_peak)        
        
        # Determine the redshift candidates and thier uncertainties and r-values based on the best-fit Gaussians
        # peak_width_in_rv is the width of the cross-correlation signal peak in radial velocity (km/s)
        # It will be used for the line fitting for the emission template
        self.zs, self.zerrs, self.r_values, peak_width_in_rv = np.zeros(len(self.bestfit_gaussians_cc)), np.zeros(len(self.bestfit_gaussians_cc)), np.zeros(len(self.bestfit_gaussians_cc)), np.zeros(len(self.bestfit_gaussians_cc))
        for i in range(len(self.zs)):
            # extract best fit parameters
            amp, mean, std = self.bestfit_gaussians_cc[i].amplitude.value, self.bestfit_gaussians_cc[i].mean.value, self.bestfit_gaussians_cc[i].stddev.value
            damp, dmean, dstd = np.sqrt(self.fitters[i].fit_info['param_cov'][0,0]), np.sqrt(self.fitters[i].fit_info['param_cov'][1,1]), np.sqrt(self.fitters[i].fit_info['param_cov'][2,2])
            # redshift
            self.zs[i]  = mean/c
            # redshift uncertainty
            zerr_ = std*np.sqrt(-2*np.log(1-0.5/amp))/c
            dzerr_ = np.sqrt((dmean+2*std*np.sqrt(-2/np.log(1-0.5/amp))*damp/(amp*(amp-0.5))+2*dstd*np.sqrt(-2*np.log(1-0.5/amp)))**2+dmean**2)/c
            self.zerrs[i] = zerr_+dzerr_
            # peak width in radial velocity
            peak_width_in_rv[i] = std
            # r-value
            # We calculate the r-value based on the cross-correlation signals within (z-0.1*(1+z),z+0.1*(1+z))
            idx_peak = np.abs(self.shifted_vels-mean).argmin() # index of the peak
            N = min(int((0.1*(c+mean))/(self.shifted_vels[idx_peak]-self.shifted_vels[idx_peak-1])), idx_peak, len(self.cc) - 1 - idx_peak) # The number of pixels used for computing the r-value, considering the boundary of the cross-correlation signals
            cc_left, cc_right = self.cc[idx_peak-N:idx_peak], np.flip(self.cc[idx_peak+1:idx_peak+N+1]) # cross-correlation signals in the left and right in N pixels with respect to the peak
            cc_left, cc_right = cc_left[(~np.isnan(cc_left))&(~np.isnan(cc_right))], cc_right[(~np.isnan(cc_left))&(~np.isnan(cc_right))] # remove NaN values from the cross-correlation signals
            sigma = np.sum(((cc_left - cc_right)**2))/N # sigma
            self.r_values[i] = self.cc[idx_peak]/(np.sqrt(sigma)) # r-value
            
        # Now, we determine the final redshift measurement result from the redshift candidates
        # If there is at least one redshift candidate with the r-value larger than 0,
        if (len(self.zs)>0)&(np.max(self.r_values)>0):
            # sort the redshift candidates in descending order of the r-value
            sort = np.argsort(self.r_values)[::-1]
            self.zs, self.zerrs, self.r_values, peak_width_in_rv = self.zs[sort], self.zerrs[sort], self.r_values[sort], peak_width_in_rv[sort]
            self.bestfit_gaussians_cc = [self.bestfit_gaussians_cc[i] for i in sort]
            self.fitters = [self.fitters[i] for i in sort]
            # For the absorption template, we select the redshift candidate with the lowest effective chi square value as the final redshift measurement result
            if self.template[1] == 1:
                chi_min = 100000000
                self.z, self.zerr, self.r = np.nan, np.nan, np.nan
                for i in range(len(self.zs)):
                    chi_min_ = self.chi_effs[np.argmin(np.abs(self.zs[i]*c-self.shifted_vels))]
                    if chi_min_<chi_min:
                        chi_min = chi_min_
                        self.z, self.zerr, self.r = self.zs[i], self.zerrs[i], self.r_values[i]
                        self.bestfit_gaussian_cc_at_z = self.bestfit_gaussians_cc[i]
                self.chi_eff = chi_min
            # For the emission template, we perform the line fitting for the redshift measurement if line_fit is True,
            # and select the redshift candidate with the lowest effective chi square values 'from the line fitting' as the final redshift measurement result
            # If line_fit is False, we select the redshift candidate with the lowest effective chi square value as the final redshift measurement result
            elif self.template[1] == 2:
                if self.line_fit:
                    self.line_chi_effs, self.line_models = [], []
                    self.line_chi_eff, self.model_fluxes = 100, np.zeros_like(self.spectrum[0])
                    self.z, self.zerr, self.r = np.nan, np.nan, np.nan
                    if self.template[1] == 2:
                        for i in range(len(self.zs)):
                            line_chi_eff_, model_fluxes_ = fit_multiline(self.spectrum[0], self.proc_spec.normalized_fluxes, 
                                                            self.spectrum[3]*self.spectrum[2]/self.proc_spec.continuum_fluxes,
                                                            lines=self.em_lines, z=self.zs[i], zerr=self.zerrs[i], cc_width=peak_width_in_rv[i],
                                                            resolution=self.resolution, temp_type=self.template[1], n_jobs=self.n_jobs)
                            self.line_chi_effs.append(line_chi_eff_)
                            self.line_models.append(model_fluxes_)
                            if line_chi_eff_ < self.line_chi_eff:
                                self.line_chi_eff, self.model_fluxes = line_chi_eff_, model_fluxes_
                                self.z, self.zerr, self.r = self.zs[i], self.zerrs[i], self.r_values[i]
                                self.bestfit_gaussian_cc_at_z = self.bestfit_gaussians_cc[i]
                        self.chi_eff = self.line_chi_eff
                else:
                    chi_min = 100000000
                    self.z, self.zerr, self.r = np.nan, np.nan, np.nan
                    for i in range(len(self.zs)):
                        chi_min_ = self.chi_effs[np.argmin(np.abs(self.zs[i]*c-self.shifted_vels))]
                        if chi_min_<chi_min:
                            chi_min = chi_min_
                            self.z, self.zerr, self.r = self.zs[i], self.zerrs[i], self.r_values[i]
                            self.bestfit_gaussian_cc_at_z = self.bestfit_gaussians_cc[i]
                    self.chi_eff = chi_min
            else:
                raise ValueError('template[1] should be 1 or 2')
        # If there is no redshift candidate with the r-value larger than 0,
        # set the redshift measurement result to NaN
        else:
            self.z, self.zerr, self.r, self.chi_eff = np.nan, np.nan, np.nan, np.nan
            self.bestfit_gaussian_cc_at_z = return_nan
                

    # def cal_chi_eff(self):
    #     z_wavelength = self.template_spectrum[0]*(1+self.z)
    #     self.overlap_spec = copy.deepcopy(self.spectrum)
    #     self.overlap_spec = self.spectrum[:,(self.spectrum[0,:]>max(z_wavelength[0], self.spectrum[0,0]))&
    #                                  (self.spectrum[0,:]<min(z_wavelength[-1], self.spectrum[0,-1]))]
    #     self.T = resampler(z_wavelength, self.template_spectrum[1], self.overlap_spec[0])
    #     self.T_continuum = process_template(self.overlap_spec[0], self.T, np.abs(self.overlap_spec[3]/self.overlap_spec[2]), knots_bin = self.temp_knots_bin,
    #                                                     thres = self.temp_line_thres, apodization_size=self.temp_apodization_size)
    #     self.T_ = self.T*self.proc_spec.continuum_fluxes[(self.spectrum[0,:]>max(z_wavelength[0], self.spectrum[0,0]))&(self.spectrum[0,:]<min(z_wavelength[-1], self.spectrum[0,-1]))]/self.T_continuum.continuum_fluxes
    #     self.chi_eff = np.sum(((self.overlap_spec[3]/self.overlap_spec[2])**2*(self.overlap_spec[1]-self.T_)**2))/(np.sum((self.overlap_spec[3]))-len(self.T_continuum.knots)-1)




# radial velocity measurement
class rvm:
    def __init__(self, templates, z_range=[-0.01,2], temp_apodization_size=0.05, temp_knots_bin = 100, temp_line_thres=3, temp_resolution=3, n_jobs=-1):
        '''
        templates: a dictionary of templates; templates[name] = [template_spectrum, template_type]
        z_range: redshift range for the shifted template
        temp_apodization_size: size of the Tukey window for the apodization for the template
        temp_knots_bin: the size of the knots for the template
        temp_line_thres: threshold for line detection
        temp_resolution: the spectral resoltuion of the template
        n_jobs: the number of jobs for parallel processing. If n_jobs = -1, all available resources are used
        '''
        self.templates = templates
        self.z_range = z_range
        self.temp_apodization_size, self.temp_knots_bin, self.temp_line_thres = temp_apodization_size, temp_knots_bin, temp_line_thres
        self.temp_resolution = temp_resolution
        self.n_jobs = n_jobs
        
        self.templates1, self.templates2 = {}, {}
        for name in self.templates.keys():
            if self.templates[name][1] == 1:
                self.templates1[name] = copy.deepcopy(self.templates[name])
            if templates[name][1] == 2:
                self.templates2[name] = copy.deepcopy(self.templates[name])
        
        self.shifted_templates = shift_templates(templates=self.templates, z_range=self.z_range, apodization_size=self.temp_apodization_size, knots_bin=self.temp_knots_bin, thres = self.temp_line_thres, resolution=self.temp_resolution, n_jobs=self.n_jobs)
        self.shifted_templates1 = shift_templates(templates=self.templates1, z_range=self.z_range, apodization_size=self.temp_apodization_size, knots_bin=self.temp_knots_bin, thres=self.temp_line_thres, resolution=self.temp_resolution, n_jobs=self.n_jobs)
        self.shifted_templates2 = shift_templates(templates=self.templates2, z_range=self.z_range, apodization_size=self.temp_apodization_size, knots_bin=self.temp_knots_bin, thres=self.temp_line_thres, resolution=self.temp_resolution, n_jobs=self.n_jobs)
    
    def z_single(self, spectrum, output='all', prior='abs', spectrum_range=None, mask=None,  window_continuum=100, sn_continuum=0.5, **kwargs):
        '''
        spectrum: spectrum (wavelength, flux, ivar, mask)
        output: whether to show the results for all templates or the best template
        prior: the type of template to be used for the redshift measurement in prior
        spectrum_range: the wavelength range of the spectrum used for the redshift measurement
        mask: [[mask_left, mask_right], ...], the wavelength range of the spectrum to be masked
        window_continuum: the size of the window for cleaning the spectrum using clean_spectrum
        sn_continuum: the signal-to-noise ratio for cleaning the spectrum using clean_spectrum
        kwargs: other parameters for the redshift measurement
        
        returns:
        if output == 'all':
            return the results for all templates (best template, redshift, redshift error, r-value, effective chi square)
        if output == 'best':
            return the best template (best template, redshift, redshift error, r-value, effective chi square)
        '''
        # Cut the spectrum based on the spectrm_range
        spectrum = copy.deepcopy(spectrum)
        if type(spectrum_range)==type([]) or type(spectrum_range)==type(np.array([])):
            if len(np.where((spectrum[0,:]>spectrum_range[0])&(spectrum[0,:]<spectrum_range[1]))[0]) ==0:
                raise ValueError('spectrum_range should contain spectrum wavelengths')
            spectrum = spectrum[:,(spectrum[0,:]>spectrum_range[0])&(spectrum[0,:]<spectrum_range[1])]
        
        # Clean the spectrum by removing the left/right end where S/N is low
        spectrum = clean_spectrum(spectrum, window_continuum, sn_continuum)
        # Resacle the spectrum for flux uncertainty not to be too larger than 1
        scale = np.median(spectrum[2])
        spectrum[1] /= scale
        spectrum[2] /= scale
        # Explicitly mask the spectrum based on 'mask'
        if type(mask)==type([]) or type(mask)==type(np.array([])):
            for i in range(len(mask)):
                left_end = abs(spectrum[0,:]- mask[i][0]).argmin()
                right_end = abs(spectrum[0,:] - mask[i][1]).argmin()
                spectrum[3,left_end:right_end+1] = 0
                
        if output == 'all':
            return self.z_all_templates(spectrum=spectrum, prior=prior, **kwargs)
        elif output == 'best':
            if prior == 'abs':
                return self.z_abs_em(spectrum=spectrum, **kwargs)
            elif prior == 'em':
                return self.z_em_abs(spectrum=spectrum, **kwargs)
            else:
                raise ValueError('prior should be abs or em')
        else:
            raise ValueError('output should be all or best')
    
    def z_abs_em(self, spectrum, resolution=3, chi_thres=4, r_thres=5, 
                 knots_bin=100, line_thres=3, apodization_size=0.05,
                 line_fit = True,
                 em_lines=[2798.00, 3727.30, 4861.33, [4958.91, 5006.84], [6548.06, 6562.82, 6583.57], [6716.440, 6730.815]]):
        
        # Spectrum for the cross-correlation with absorption and emission templates
        abs_spectrum, em_spectrum = copy.deepcopy(spectrum), copy.deepcopy(spectrum)
        
        # Cross-correate with the absorption templates first
        # Normalize and mask the spectrum
        abs_proc_spec = process_spectrum(abs_spectrum[0], abs_spectrum[1], np.abs(abs_spectrum[3]/abs_spectrum[2]), resolution=resolution, temp_type=1, knots_bin = knots_bin,
                                    thres=line_thres, apodization_size=apodization_size, n_jobs=self.n_jobs)
        abs_spectrum[3] = abs_proc_spec.new_masks
        
        # Cross-correalte with the shifted absorption templates
        # and obtain the redshift measurement results for each absorption template
        template_names1 = list(self.templates1.keys())
        n_templates1 = len(template_names1)
        z1, zerr1, r1, chi_eff1 = np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1)
        for i, temp_name in enumerate(template_names1):
            cc_spec_temp = cc_result(abs_spectrum, proc_spec=abs_proc_spec,
                                    template=self.templates1[temp_name], shifted_template=self.shifted_templates1[temp_name], 
                                    temp_apodization_size=self.temp_apodization_size,
                                    temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, temp_resolution=self.temp_resolution,
                                    z_range=self.z_range, line_fit=line_fit,
                                    em_lines=em_lines, resolution=resolution, n_jobs=self.n_jobs)
            z1[i], zerr1[i], r1[i], chi_eff1[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
        
        # remove the results with nan-redshift, chi_eff>chi_thres, and r<r_thres
        if chi_thres:
            nan_thres_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(chi_eff1<chi_thres)&(r1>r_thres)
        else:
            nan_thres_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(r1>r_thres)
        template_names1, z1, zerr1, r1, chi_eff1 = np.array(template_names1)[nan_thres_check], z1[nan_thres_check], zerr1[nan_thres_check], r1[nan_thres_check], chi_eff1[nan_thres_check]
        # Select the best result among absorption templates if there are any
        if len(r1):
            i_best1 = np.nanargmin(chi_eff1)
            return (template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1])
        
        # If there is no absorption template with chi_eff<chi_thres and r>r_thres,
        # cross-correate with the emission templates
        else:
            # Normalize and mask the spectrum 
            em_proc_spec = process_spectrum(em_spectrum[0], em_spectrum[1], np.abs(em_spectrum[3]/em_spectrum[2]), resolution=resolution, temp_type=2, knots_bin = knots_bin,
                                        thres=line_thres, apodization_size=apodization_size, n_jobs=self.n_jobs)
            em_spectrum[3] = em_proc_spec.new_masks
            
            # Cross-correalte with the shifted emission templates
            # and obtain the redshift measurement results for each emission template
            template_names2 = list(self.templates2.keys())
            n_templates2 = len(template_names2)
            z2, zerr2, r2, chi_eff2 = np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2)
            for i, temp_name in enumerate(template_names2):
                cc_spec_temp = cc_result(em_spectrum, proc_spec=em_proc_spec, 
                                        template=self.templates2[temp_name], shifted_template=self.shifted_templates2[temp_name],
                                        temp_apodization_size=self.temp_apodization_size,
                                        temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, temp_resolution=self.temp_resolution,
                                        z_range=self.z_range, line_fit=line_fit,
                                        em_lines=em_lines, resolution=resolution, n_jobs=self.n_jobs)
                z2[i], zerr2[i], r2[i], chi_eff2[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
            
            # remove the results with nan-redshift, chi_eff>chi_thres, and r<r_thres
            if chi_thres:
                nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(chi_eff2<chi_thres)&(r2>r_thres)
            else:
                nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(r2>r_thres)
            template_names2, z2, zerr2, r2, chi_eff2 = np.array(template_names2)[nan_check], z2[nan_check], zerr2[nan_check], r2[nan_check], chi_eff2[nan_check]
                    
            # Select the best result among absorption templates if there are any
            if len(r2):
                i_best2 = np.nanargmin(chi_eff2)
                return (template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2])     
            # If there is no emission template with chi_eff<chi_thres and r>r_thres,
            # set the final redshift measurement result to NaN
            else:
                return ('No_template', -9,-9,-9,-9)
    
    def z_em_abs(self, spectrum, resolution=3, chi_thres=4, r_thres=5, 
                 knots_bin=100, line_thres=3, apodization_size=0.05,
                 line_fit = True,
                 em_lines=[2798.00, 3727.30, 4861.33, [4958.91, 5006.84], [6548.06, 6562.82, 6583.57], [6716.440, 6730.815]]):
        # Spectrum for the cross-correlation with absorption and emission templates
        abs_spectrum, em_spectrum = copy.deepcopy(spectrum), copy.deepcopy(spectrum)
        
        # Cross-correate with the emission templates first
        # Normalize and mask the spectrum
        em_proc_spec = process_spectrum(em_spectrum[0], em_spectrum[1], np.abs(em_spectrum[3]/em_spectrum[2]), resolution=resolution, temp_type=2, knots_bin = knots_bin,
                                    thres=line_thres, apodization_size=apodization_size, n_jobs=self.n_jobs)
        em_spectrum[3] = em_proc_spec.new_masks
        
        # Cross-correalte with the shifted emission templates
        # and obtain the redshift measurement results for each emission template
        template_names2 = list(self.templates2.keys())
        n_templates2 = len(template_names2)
        z2, zerr2, r2, chi_eff2 = np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2)
        for i, temp_name in enumerate(template_names2):
            cc_spec_temp = cc_result(em_spectrum, proc_spec=em_proc_spec, 
                                    template=self.templates2[temp_name], shifted_template=self.shifted_templates2[temp_name],
                                    temp_apodization_size=self.temp_apodization_size,
                                    temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, temp_resolution=self.temp_resolution,
                                    z_range=self.z_range, line_fit=line_fit,
                                    em_lines=em_lines, resolution=resolution, n_jobs=self.n_jobs)
            z2[i], zerr2[i], r2[i], chi_eff2[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
            
        # Remove the results with nan-redshift, chi_eff>chi_thres, and r<r_thres
        if chi_thres:
            nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(chi_eff2<chi_thres)&(r2>r_thres)
        else:
            nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(r2>r_thres)
        template_names2, z2, zerr2, r2, chi_eff2 = np.array(template_names2)[nan_check], z2[nan_check], zerr2[nan_check], r2[nan_check], chi_eff2[nan_check]
        
        # Select the best result among absorption templates if there are any
        if len(r2):
            i_best2 = np.nanargmin(chi_eff2)
            return (template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2])
        
        # If there is no emission template with chi_eff<chi_thres and r>r_thres,
        # cross-correate with the absorption templates   
        else:
            # Normalize and mask the spectrum 
            abs_proc_spec = process_spectrum(abs_spectrum[0], abs_spectrum[1], np.abs(abs_spectrum[3]/abs_spectrum[2]), resolution=resolution, temp_type=1, knots_bin = knots_bin,
                                        thres=line_thres, apodization_size=apodization_size, n_jobs=self.n_jobs)
            abs_spectrum[3] = abs_proc_spec.new_masks
            
            # Cross-correalte with the shifted absorption templates
            # and obtain the redshift measurement results for each absorption templat
            template_names1 = list(self.templates1.keys())
            n_templates1 = len(template_names1)
            z1, zerr1, r1, chi_eff1 = np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1)
            for i, temp_name in enumerate(template_names1):
                cc_spec_temp = cc_result(abs_spectrum, proc_spec=abs_proc_spec,
                                        template=self.templates1[temp_name], shifted_template=self.shifted_templates1[temp_name], 
                                        temp_apodization_size=self.temp_apodization_size,
                                        temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, temp_resolution=self.temp_resolution,
                                        z_range=self.z_range, line_fit=line_fit,
                                        em_lines=em_lines, resolution=resolution, n_jobs=self.n_jobs)
                z1[i], zerr1[i], r1[i], chi_eff1[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
            
            # Remove the results with nan-redshift, chi_eff>chi_thres, and r<r_thres
            if chi_thres:
                nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(chi_eff1<chi_thres)&(r1>r_thres)
            else:
                nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(r1>r_thres)
            template_names1, z1, zerr1, r1, chi_eff1 = np.array(template_names1)[nan_check], z1[nan_check], zerr1[nan_check], r1[nan_check], chi_eff1[nan_check]
            
            # Select the best result among absorption templates if there are any
            if len(r1):
                i_best1 = np.nanargmin(chi_eff1)
                return (template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1])  
            # If there is no emission template with chi_eff<chi_thres and r>r_thres,
            # set the final redshift measurement result to NaN
            else:
                return ('No_template', -9,-9,-9,-9)
            
    def z_all_templates(self, spectrum, prior='abs', output='all', resolution=3, chi_thres=4, r_thres=5, 
                 knots_bin=100, line_thres=3, apodization_size=0.05,
                 line_fit = True,
                 em_lines=[2798.00, 3727.30, 4861.33, [4958.91, 5006.84], [6548.06, 6562.82, 6583.57], [6716.440, 6730.815]]):
        # Spectrum for the cross-correlation with absorption and emission templates
        abs_spectrum, em_spectrum = copy.deepcopy(spectrum), copy.deepcopy(spectrum)
        
        self.cc_result = {} # to store the cross-correlation and redshift measurement results for each template
        
        # Absorption tempaltes
        abs_proc_spec = process_spectrum(abs_spectrum[0], abs_spectrum[1], np.abs(abs_spectrum[3]/abs_spectrum[2]), resolution=resolution, temp_type=1, knots_bin = knots_bin,
                                    thres=line_thres, apodization_size=apodization_size, n_jobs=self.n_jobs)
        abs_spectrum[3] = abs_proc_spec.new_masks
        template_names1 = list(self.templates1.keys())
        n_templates1 = len(template_names1)
        z1, zerr1, r1, chi_eff1 = np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1)
        for i, temp_name in enumerate(template_names1):
            cc_spec_temp = cc_result(abs_spectrum, proc_spec=abs_proc_spec,
                                    template=self.templates1[temp_name], shifted_template=self.shifted_templates1[temp_name], 
                                    temp_apodization_size=self.temp_apodization_size,
                                    temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, temp_resolution=self.temp_resolution,
                                    z_range=self.z_range, line_fit=line_fit,
                                    em_lines=em_lines, resolution=resolution, n_jobs=self.n_jobs)
            self.cc_result[temp_name] = cc_spec_temp
            z1[i], zerr1[i], r1[i], chi_eff1[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                
        # Remove the results with nan-redshift, chi_eff>chi_thres, and r<r_thres
        if chi_thres:
            nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(chi_eff1<chi_thres)&(r1>r_thres)
        else:
            nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(r1>r_thres)
        template_names1, z1, zerr1, r1, chi_eff1 = np.array(template_names1)[nan_check], z1[nan_check], zerr1[nan_check], r1[nan_check], chi_eff1[nan_check]
        
        # Select the best result among absorption templates if there are any
        if len(r1):
            i_best1 = np.nanargmin(chi_eff1)
            best_templates_name1, best_z1, best_zerr1, best_r1, best_chi_eff1 = template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1]
        else:
            best_r1 = np.nan
            
        # Emission tempaltes
        em_proc_spec = process_spectrum(em_spectrum[0], em_spectrum[1], np.abs(em_spectrum[3]/em_spectrum[2]), resolution=resolution, temp_type=2, knots_bin = knots_bin,
                                    thres=line_thres, apodization_size=apodization_size, n_jobs=self.n_jobs)
        em_spectrum[3] = em_proc_spec.new_masks
        template_names2 = list(self.templates2.keys())
        n_templates2 = len(template_names2)
        z2, zerr2, r2, chi_eff2 = np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2)
        for i, temp_name in enumerate(template_names2):
            cc_spec_temp = cc_result(em_spectrum, proc_spec=em_proc_spec,
                                    template=self.templates2[temp_name], shifted_template=self.shifted_templates2[temp_name],
                                    temp_apodization_size=self.temp_apodization_size,
                                    temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, temp_resolution=self.temp_resolution,
                                    z_range=self.z_range, line_fit=line_fit,
                                    em_lines=em_lines, resolution=resolution)
            self.cc_result[temp_name] = cc_spec_temp
            z2[i], zerr2[i], r2[i], chi_eff2[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                
        # Remove the results with nan-redshift, chi_eff>chi_thres, and r<r_thres
        if chi_thres:
            nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(chi_eff2<chi_thres)&(r2>r_thres)
        else:
            nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(r2>r_thres)
        template_names2, z2, zerr2, r2, chi_eff2 = np.array(template_names2)[nan_check], z2[nan_check], zerr2[nan_check], r2[nan_check], chi_eff2[nan_check]

        # Select the best result among emission templates if there are any
        if len(r2):
            i_best2 = np.nanargmin(chi_eff2)
            best_templates_name2, best_z2, best_zerr2, best_r2, best_chi_eff2 = template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2] 
        else:
            best_r2 = np.nan

        # Choose the best result
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

        # Concatenate the results from absorption and emission templates
        template_names, z, zerr, r, chi_eff = np.concatenate([template_names1, template_names2]), np.concatenate([z1, z2]), np.concatenate([zerr1, zerr2]), np.concatenate([r1,r2]), np.concatenate([chi_eff1, chi_eff2])
        note = np.zeros_like(r).astype(str)
        note[:] = ' '
        if i_best != None:
            note[i_best] = 'best'
        # Arange the value in the order of chi_eff
        order = np.flip(np.argsort(r))
        template_names, z, zerr, r, chi_eff, note = template_names[order], z[order], zerr[order], r[order], chi_eff[order], note[order]
        
        table = np.vstack((template_names, z, zerr, r, chi_eff, note))
        column_names = ['template_name', 'z', 'zerr', 'r', 'chi_eff', 'note']
        result = pd.DataFrame(table.T, columns = column_names)
        result = result.astype({'template_name':str, 'z':np.float32, 'zerr':np.float32, 'r':np.float32, 'chi_eff':np.float32, 'note':str})
        
        return result
        
    
    # def z_single(self, spectrum, output='all', prior='abs', spectrum_range=None, resolution=3, chi_thres=4, mask=None, r_thres=5, 
    #              knots_bin=100, line_thres=3, apodization_size=0.05, window_continuum=100, sn_continuum=0.5,
    #              line_fit = True,
    #              em_lines=[2798.00, 3727.30, 4861.33, [4958.91, 5006.84], [6548.06, 6562.82, 6583.57], [6716.440, 6730.815]]):
        
    #     spectrum = copy.deepcopy(spectrum)
        
    #     if type(spectrum_range)==type([]) or type(spectrum_range)==type(np.array([])):
    #         if len(np.where((spectrum[0,:]>spectrum_range[0])&(spectrum[0,:]<spectrum_range[1]))[0]) ==0:
    #             raise ValueError('spectrum_range should contain spectrum wavelengths')
    #         spectrum = spectrum[:,(spectrum[0,:]>spectrum_range[0])&(spectrum[0,:]<spectrum_range[1])]
        
    #     spectrum = clean_spectrum(spectrum, window_continuum, sn_continuum)
    #     scale = np.median(spectrum[2])
    #     spectrum[1] /= scale
    #     spectrum[2] /= scale
    #     # spectrum[3,spectrum[1]/spectrum[2]<-3] = 0
    #     if type(mask)==type([]) or type(mask)==type(np.array([])):
    #         for i in range(len(mask)):
    #             left_end = abs(spectrum[0,:]- mask[i][0]).argmin()
    #             right_end = abs(spectrum[0,:] - mask[i][1]).argmin()
    #             spectrum[3,left_end:right_end+1] = 0
                
    #     abs_spectrum, em_spectrum = copy.deepcopy(spectrum), copy.deepcopy(spectrum)
        
    #     if output=='best':
    #         if prior == 'abs':
    #             # absorption tempaltes
    #             normalize = process_spectrum(abs_spectrum[0], abs_spectrum[1], np.abs(abs_spectrum[3]/abs_spectrum[2]), resolution=resolution, temp_type=1, knots_bin = knots_bin,
    #                                         thres=line_thres, apodization_size=apodization_size)
    #             abs_spectrum[3] = normalize.new_masks
                
    #             template_names1 = list(self.templates1.keys())
    #             n_templates1 = len(template_names1)
    #             z1, zerr1, r1, chi_eff1 = np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1)
    #             for i, temp_name in enumerate(template_names1):
    #                 cc_spec_temp = cc_result(abs_spectrum, proc_spec=normalize,
    #                                         template=self.templates1[temp_name], shifted_template=self.shifted_templates1[temp_name], 
    #                                         temp_apodization_size=self.temp_apodization_size,
    #                                         temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, temp_resolution=self.temp_resolution,
    #                                         z_range=self.z_range, line_fit=line_fit,
    #                                         em_lines=em_lines, resolution=resolution)
    #                 z1[i], zerr1[i], r1[i], chi_eff1[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                
                
                
    #             # remove the results with nan-redshift
    #             if chi_thres:
    #                 nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(chi_eff1<chi_thres)&(r1>r_thres)
    #             else:
    #                 nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(r1>r_thres)
    #             template_names1, z1, zerr1, r1, chi_eff1 = np.array(template_names1)[nan_check], z1[nan_check], zerr1[nan_check], r1[nan_check], chi_eff1[nan_check]
    #             # best result among absorption templates
    #             if len(r1):
    #                 i_best1 = np.nanargmin(chi_eff1)
    #                 result = (template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1])
                    
    #             else:
    #                 # emission tempaltes
    #                 normalize = process_spectrum(em_spectrum[0], em_spectrum[1], np.abs(em_spectrum[3]/em_spectrum[2]), resolution=resolution, temp_type=2, knots_bin = knots_bin,
    #                                             thres=line_thres, apodization_size=apodization_size)
    #                 em_spectrum[3] = normalize.new_masks
                    
    #                 template_names2 = list(self.templates2.keys())
    #                 n_templates2 = len(template_names2)
    #                 z2, zerr2, r2, chi_eff2 = np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2)
    #                 for i, temp_name in enumerate(template_names2):
    #                     cc_spec_temp = cc_result(em_spectrum, proc_spec=normalize, 
    #                                             template=self.templates2[temp_name], shifted_template=self.shifted_templates2[temp_name],
    #                                             temp_apodization_size=self.temp_apodization_size,
    #                                             temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, temp_resolution=self.temp_resolution,
    #                                             z_range=self.z_range, line_fit=line_fit,
    #                                             em_lines=em_lines, resolution=resolution)
    #                 # return output
    #                     z2[i], zerr2[i], r2[i], chi_eff2[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                    
    #                 # remove the results with nan-redshift
    #                 if chi_thres:
    #                     nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(chi_eff2<chi_thres)&(r2>r_thres)
    #                 else:
    #                     nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(r2>r_thres)
    #                 template_names2, z2, zerr2, r2, chi_eff2 = np.array(template_names2)[nan_check], z2[nan_check], zerr2[nan_check], r2[nan_check], chi_eff2[nan_check]
                            
    #                 # best result among emssion templates
    #                 if len(r2):
    #                     i_best2 = np.nanargmin(chi_eff2)
    #                     result = (template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2])     
    #                 else:
    #                     result = ('No_template', -9,-9,-9,-9)
                        
    #         if prior == 'em':
    #             # emission tempaltes
    #             normalize = process_spectrum(em_spectrum[0], em_spectrum[1], np.abs(em_spectrum[3]/em_spectrum[2]), resolution=resolution, temp_type=2, knots_bin = knots_bin,
    #                                         thres=line_thres, apodization_size=apodization_size)
    #             em_spectrum[3] = normalize.new_masks
                
    #             template_names2 = list(self.templates2.keys())
    #             n_templates2 = len(template_names2)
    #             z2, zerr2, r2, chi_eff2 = np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2)
    #             for i, temp_name in enumerate(template_names2):
    #                 cc_spec_temp = cc_result(em_spectrum, normalize=normalize, proc_spec=normalize, 
    #                                         template=self.templates2[temp_name], shifted_template=self.shifted_templates2[temp_name],
    #                                         temp_apodization_size=self.temp_apodization_size,
    #                                         temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, temp_resolution=self.temp_resolution,
    #                                         z_range=self.z_range, line_fit=line_fit,
    #                                         em_lines=em_lines, resolution=resolution)
    #             # return output
    #                 z2[i], zerr2[i], r2[i], chi_eff2[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                    
    #             # remove the results with nan-redshift
    #             if chi_thres:
    #                 nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(chi_eff2<chi_thres)&(r2>r_thres)
    #             else:
    #                 nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(r2>r_thres)
    #             template_names2, z2, zerr2, r2, chi_eff2 = np.array(template_names2)[nan_check], z2[nan_check], zerr2[nan_check], r2[nan_check], chi_eff2[nan_check]
                
    #             # best result among emssion templates
    #             if len(r2):
    #                 i_best2 = np.nanargmin(chi_eff2)
    #                 result = (template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2])
                    
    #             else:
    #                 # absorption tempaltes
    #                 normalize = process_spectrum(abs_spectrum[0], abs_spectrum[1], np.abs(abs_spectrum[3]/abs_spectrum[2]), resolution=resolution, temp_type=1, knots_bin = knots_bin,
    #                                             thres=line_thres, apodization_size=apodization_size)
    #                 abs_spectrum[3] = normalize.new_masks
                    
    #                 template_names1 = list(self.templates1.keys())
    #                 n_templates1 = len(template_names1)
    #                 z1, zerr1, r1, chi_eff1 = np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1)
    #                 for i, temp_name in enumerate(template_names1):
    #                     cc_spec_temp = cc_result(abs_spectrum, proc_spec=normalize,
    #                                             template=self.templates1[temp_name], shifted_template=self.shifted_templates1[temp_name], 
    #                                             temp_apodization_size=self.temp_apodization_size,
    #                                             temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, temp_resolution=self.temp_resolution,
    #                                             z_range=self.z_range, line_fit=line_fit,
    #                                             em_lines=em_lines, resolution=resolution)
    #                     z1[i], zerr1[i], r1[i], chi_eff1[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                    
    #                 # remove the results with nan-redshift
    #                 if chi_thres:
    #                     nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(chi_eff1<chi_thres)&(r1>r_thres)
    #                 else:
    #                     nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(r1>r_thres)
    #                 template_names1, z1, zerr1, r1, chi_eff1 = np.array(template_names1)[nan_check], z1[nan_check], zerr1[nan_check], r1[nan_check], chi_eff1[nan_check]
    #                 # best result among absorption templates
    #                 if len(r1):
    #                     i_best1 = np.nanargmax(chi_eff1)
    #                     result = (template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1])  
    #                 else:
    #                     result = ('No_template', -9,-9,-9,-9)
                    
    #     elif output=='all':
    #         self.cc_result = {}
    #         # absorption tempaltes
    #         normalize = process_spectrum(abs_spectrum[0], abs_spectrum[1], np.abs(abs_spectrum[3]/abs_spectrum[2]), resolution=resolution, temp_type=1, knots_bin = knots_bin,
    #                                     thres=line_thres, apodization_size=apodization_size)
    #         abs_spectrum[3] = normalize.new_masks
            
    #         template_names1 = list(self.templates1.keys())
    #         n_templates1 = len(template_names1)
    #         z1, zerr1, r1, chi_eff1 = np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1), np.zeros(n_templates1)
    #         for i, temp_name in enumerate(template_names1):
    #             cc_spec_temp = cc_result(abs_spectrum, proc_spec=normalize,
    #                                     template=self.templates1[temp_name], shifted_template=self.shifted_templates1[temp_name], 
    #                                     temp_apodization_size=self.temp_apodization_size,
    #                                     temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, temp_resolution=self.temp_resolution,
    #                                     z_range=self.z_range, line_fit=line_fit,
    #                                     em_lines=em_lines, resolution=resolution)
    #             self.cc_result[temp_name] = cc_spec_temp
    #             z1[i], zerr1[i], r1[i], chi_eff1[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                    
    #         # remove the results with nan-redshift
    #         if chi_thres:
    #             nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(chi_eff1<chi_thres)&(r1>r_thres)
    #         else:
    #             nan_check = (np.isfinite(zerr1))&(np.isfinite(r1))&(np.isfinite(chi_eff1))&(r1>r_thres)
    #         template_names1, z1, zerr1, r1, chi_eff1 = np.array(template_names1)[nan_check], z1[nan_check], zerr1[nan_check], r1[nan_check], chi_eff1[nan_check]
            
    #         # best result among absorption templates
    #         if len(r1):
    #             i_best1 = np.nanargmin(chi_eff1)
    #             best_templates_name1, best_z1, best_zerr1, best_r1, best_chi_eff1 = template_names1[i_best1], z1[i_best1], zerr1[i_best1], r1[i_best1], chi_eff1[i_best1]
    #         else:
    #             best_r1 = np.nan
                
    #         # emission tempaltes
    #         normalize = process_spectrum(em_spectrum[0], em_spectrum[1], np.abs(em_spectrum[3]/em_spectrum[2]), resolution=resolution, temp_type=2, knots_bin = knots_bin,
    #                                     thres=line_thres, apodization_size=apodization_size)
    #         em_spectrum[3] = normalize.new_masks
            
    #         template_names2 = list(self.templates2.keys())
    #         n_templates2 = len(template_names2)
    #         z2, zerr2, r2, chi_eff2 = np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2), np.zeros(n_templates2)
    #         for i, temp_name in enumerate(template_names2):
    #             cc_spec_temp = cc_result(em_spectrum, proc_spec=normalize,
    #                                     template=self.templates2[temp_name], shifted_template=self.shifted_templates2[temp_name],
    #                                     temp_apodization_size=self.temp_apodization_size,
    #                                     temp_knots_bin=self.temp_knots_bin, temp_line_thres=self.temp_line_thres, temp_resolution=self.temp_resolution,
    #                                     z_range=self.z_range, line_fit=line_fit,
    #                                     em_lines=em_lines, resolution=resolution)
    #             self.cc_result[temp_name] = cc_spec_temp
    #             z2[i], zerr2[i], r2[i], chi_eff2[i] = cc_spec_temp.z, cc_spec_temp.zerr, cc_spec_temp.r, cc_spec_temp.chi_eff
                    
    #         # remove the results with nan-redshift
    #         if chi_thres:
    #             nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(chi_eff2<chi_thres)&(r2>r_thres)
    #         else:
    #             nan_check = (np.isfinite(zerr2))&(np.isfinite(r2))&(np.isfinite(chi_eff2))&(r2>r_thres)
    #         template_names2, z2, zerr2, r2, chi_eff2 = np.array(template_names2)[nan_check], z2[nan_check], zerr2[nan_check], r2[nan_check], chi_eff2[nan_check]

    #         # best result among absorption templates
    #         if len(r2):
    #             i_best2 = np.nanargmin(chi_eff2)
    #             best_templates_name2, best_z2, best_zerr2, best_r2, best_chi_eff2 = template_names2[i_best2], z2[i_best2], zerr2[i_best2], r2[i_best2], chi_eff2[i_best2] 
    #         else:
    #             best_r2 = np.nan

    #         # choose the best result
    #         if prior=='abs':
    #             if np.isfinite(best_r1):
    #                 i_best = i_best1
    #             else:
    #                 if np.isfinite(best_r2):
    #                     i_best = i_best2 + len(r1)
    #                 else:
    #                     i_best = None
    #         elif prior=='em':
    #             if np.isfinite(best_r2):
    #                 i_best = i_best2+len(r1)
    #             else:
    #                 if np.isfinite(best_r1):
    #                     i_best = i_best1
    #                 else:
    #                     i_best = None

    #         # concatenate the results from absorption and emission templates
    #         template_names, z, zerr, r, chi_eff = np.concatenate([template_names1, template_names2]), np.concatenate([z1, z2]), np.concatenate([zerr1, zerr2]), np.concatenate([r1,r2]), np.concatenate([chi_eff1, chi_eff2])
    #         note = np.zeros_like(r).astype(str)
    #         note[:] = ' '
    #         if i_best != None:
    #             note[i_best] = 'best'
    #         # arange the value in the order of chi_eff
    #         order = np.flip(np.argsort(r))
    #         template_names, z, zerr, r, chi_eff, note = template_names[order], z[order], zerr[order], r[order], chi_eff[order], note[order]
            
    #         table = np.vstack((template_names, z, zerr, r, chi_eff, note))
    #         column_names = ['template_name', 'z', 'zerr', 'r', 'chi_eff', 'note']
    #         result = pd.DataFrame(table.T, columns = column_names)
    #         result = result.astype({'template_name':str, 'z':np.float32, 'zerr':np.float32, 'r':np.float32, 'chi_eff':np.float32, 'note':str})
            

    #     return result

    def z_speclist(self, spectrums, **kw4z_single):  
        spec_number = np.arange(len(spectrums))
        result = []
        for index in tqdm(spec_number, leave=False):
            singe_result = self.z_single(spectrums[index], output='best', **kw4z_single)
            result.append(singe_result)
        result = pd.DataFrame(result, columns=['best_template', 'z', 'zerr', 'r', 'chi_eff'])
        result.astype({'best_template':str, 'z':np.float32, 'zerr':np.float32, 'r':np.float32, 'chi_eff':np.float32})
        return result
    
    def z_multi(self, spec_files, spec_import, chunk=5000, directory=None, **kw4z_speclist):
        if directory==None:
            if len(glob.glob('z_result'))==0:
                os.system('mkdir z_result')
            save_folder = 'z_result'
        else:
            if len(glob.glob(directory))==0:
                os.system('mkdir %s'%directory)
            save_folder = directory
        
        os.system('rm -fr %s'%save_folder+'/*.txt')
        file = open(save_folder + '/z_result.txt', 'w')
        file.write('spectrum_path besttemp z zerr r chi_eff pkratio\n')
        file.close()
        
        n_subs = len(spec_files)/chunk
        if n_subs > int(n_subs):
            n_subs = int(n_subs)+1
        else:
            n_subs = int(n_subs)
        for i in range(n_subs):
            sub_spec_files = spec_files[i*chunk:min((i+1)*chunk, len(spec_files))]
            spectra = []
            
            if (i+1%10) == 1:
                ordinal = f'{i+1:d}st'
            elif (i+1%10) == 2:
                ordinal = f'{i+1:d}nd'
            elif (i+1%10) == 3:
                ordinal = f'{i+1:d}rd'
            else:
                ordinal = f'{i+1:d}th'
            
            print(f'importing spectra for the {ordinal} chunk...')
            for file in tqdm(sub_spec_files, leave=False):
                spectra.append(spec_import(file))
            print('done')
            
            print(f'measuring redshifts for the {ordinal} chunk...')
            measured = self.z_speclist(spectra, **kw4z_speclist)
            measured = measured.values
            file = open(save_folder + '/z_result.txt', 'a')
            for j in range(measured.shape[0]):
                file.write(f'{sub_spec_files[j]} {measured[j][0]} {measured[j][1]} {measured[j][2]} {measured[j][3]} {measured[j][4]}\n')
            file.close()
            print('done')
        
            
# def rvsnupy(spec_files, spec_import, templates, chunk=5000, directory=None, z_range=[-0.01,2],
#             temp_apodization_size=0.05, temp_knots_bin = 100, temp_line_thres=3, **kwargs):
#     if directory==None:
#         if len(glob.glob('z_result'))==0:
#             os.system('mkdir z_result')
#         save_folder = 'z_result'
#     else:
#         if len(glob.glob(directory))==0:
#             os.system('mkdir %s'%directory)
#         save_folder = directory
    
#     os.system('rm -fr %s'%save_folder+'/*.txt')
#     file = open(save_folder + '/z_result.txt', 'w')
#     file.write('spectrum_path besttemp z zerr r chi_eff pkratio\n')
#     file.close()
#     run_rvm = rvm(templates, z_range, temp_apodization_size, temp_knots_bin , temp_line_thres)
    
#     n_subs = len(spec_files)/chunk
#     if n_subs > int(n_subs):
#         n_subs = int(n_subs)+1
#     else:
#         n_subs = int(n_subs)
#     for i in range(n_subs):
#         sub_spec_files = spec_files[i*chunk:min((i+1)*chunk, len(spec_files))]
#         spectra = []
#         print(f'importing spectra for {i+1:d}-th chunk...')
#         for file in tqdm(sub_spec_files, leave=False):
#             spectra.append(spec_import(file))
#         print('done')
        
#         print(f'measuring redshifts for {i+1:d}-th chunk...')
#         measured = run_rvm.z_multi(spectra, **kwargs)
#         measured = measured.values
#         file = open(save_folder + '/z_result.txt', 'a')
#         for j in range(measured.shape[0]):
#             file.write(f'{sub_spec_files[j]} {measured[j][0]} {measured[j][1]} {measured[j][2]} {measured[j][3]} {measured[j][4]}\n')
#         file.close()
#         print('done')