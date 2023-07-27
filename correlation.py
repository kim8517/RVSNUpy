import numpy as np
from astropy import constants as const
from astropy.units import Quantity
from scipy.signal.windows import tukey
import copy
from scipy.signal import butter,filtfilt
from astropy.constants import c
c = c.value/1e+3

from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

__all__ = ['template_correlate', 'template_logwl_resample']

def resampler(orig_spectrum, resample_lambda):
    """
    Call interpolation, repackage new spectra


    Parameters
    ----------
    orig_spectrum : 3xn array
        The original 1D spectrum.
    fin_spec_axis : 1xn array
        The desired spectral axis array.

    Returns
    -------
    resample_spectrum : 3xn
        resampled spectrum
    """
    # resampled flux
    new_flux = np.interp(resample_lambda, orig_spectrum[0,:], orig_spectrum[1,:], left=0, right=0)
    # resampled uncertainty
    new_unc = np.interp(resample_lambda, orig_spectrum[0,:], orig_spectrum[2,:], left=0, right=0)
    # resampled mask
    new_mask = np.interp(resample_lambda, orig_spectrum[0,:], orig_spectrum[3,:], left=0, right=0)
    new_mask = np.round(new_mask).astype(int)
    
    return np.vstack([resample_lambda, new_flux, new_unc, new_mask])

def interp(flux, region):
    x = np.arange(len(flux))
    y = np.ma.masked_array(flux, region)

    # Perform spline interpolation
    interp_func = InterpolatedUnivariateSpline(x[~region], y[~region])
#     interp_func = interp1d(x[~mask], y[~mask], kind='linear', fill_value='extrapolate')
    interpolated_flux = interp_func(x)
    return interpolated_flux

def abs_supress(wavelength, flux, thres):
    detection = flux < -1*thres
    
    # find the line center
    lends, rends = np.where(np.diff(detection.astype(int)) == 1)[0] + 1, np.where(np.diff(detection.astype(int)) == -1)[0] + 1
    if detection[-1] == True: # dectecion = (..., True, True, True)
        rends = np.concatenate((rends, np.array([len(flux)-1])))
    if detection[0] == True: # dectecion = (True, True, True, ...)
        lends = np.concatenate((np.array([0]), lends))
        
    centers = (lends+rends) // 2
    
    # smoothly replace the lines with 0
    region = np.zeros_like(flux).astype(bool)
    width = int(24/np.median(wavelength[1:]-wavelength[:-1])) # 3x400 km/s at 6000 A
    for lend, rend, center in zip(lends, rends, centers):
        l,r = max(0, center-width), min(len(flux)-1, center+width)
        window = 1-tukey(r-l+1, alpha=0.3)
        flux[l:r+1] *= window
        
    return flux

def emi_supress(wavelength, flux, thres):
    detection = flux > thres
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
    width = int(24/np.median(wavelength[1:]-wavelength[:-1])) # 3x400 km/s at 6000 A
    for lend, rend, center in zip(lends, rends, centers):
        l,r = max(0, center-width), min(len(flux)-1, center+width)
        window = 1-tukey(r-l+1, alpha=0.3)
        flux[l:r+1] *= window
        
    return flux
    
        

def template_correlate2(observed_spectrum, template_spectrum, template_type, clipping=False, hcutoff_scale=10,
                       fs=1, order=2, apodization_window=0.05, mask = None):
    """
    Compute cross-correlation of the observed and template spectra.


    After re-sampling into log-wavelength, both observed and template
    spectra are apodized by a Tukey window in order to minimize edge
    and consequent non-periodicity effects and thus decrease
    high-frequency power in the correlation function. To turn off the
    apodization, use alpha=0.

    Parameters
    ----------
    observed_spectrum : :3xn array
        The observed spectrum.
    template_spectrum : 3xn array
        The template spectrum, which will be correlated with
        the observed spectrum.
    apodization_window: float, callable, or None
        If a callable, will be treated as a window function for apodization of
        the cross-correlation (should behave like a `~scipy.signal.windows`
        window function, with ``sym=True``). If a float, will be treated as the
        ``alpha`` parameter for a Tukey window (`~scipy.signal.windows.tukey`),
        in units of pixels. If None, no apodization will be performed
    mask : list, optional
        Regions which is not used in cross-correlation.
        If None, the full range of the given spectrum is used. The defulat is None.

    Returns
    -------
    lag, corr: 2xn array
        1st row: lag
        2nd row: correlation signals at lags
    """
    _observed_spectrum = copy.deepcopy(observed_spectrum)
    filtered_flux = _observed_spectrum[1,:]
    hcutoff = 1/(2*hcutoff_scale)
    if template_type == 'emission':
        if clipping:
            filtered_flux[filtered_flux>8*np.std(filtered_flux)] = 0
        filtered_flux[filtered_flux<-2*np.std(filtered_flux)] = 0
    elif template_type == 'absorption':
        if clipping:
            filtered_flux[filtered_flux<-5*np.std(filtered_flux)] = 0
        filtered_flux = butter_highstop_filter(filtered_flux, hcutoff, fs, order)
        filtered_flux[filtered_flux>2*np.std(filtered_flux)] = 0
        
    _observed_spectrum[1,:] = filtered_flux
    # resample if the user requested to log wavelength
    
    log_spectrum, log_template = template_logwl_resample(_observed_spectrum,
                                                             template_spectrum)

    # apodize (might be a no-op if apodization_window is None)
    observed_log_spectrum, template_log_spectrum = _apodize(log_spectrum,
                                                            log_template,
                                                            apodization_window)
    # Normalize template
    normalization = _normalize(observed_log_spectrum, template_log_spectrum)

    # Not sure if we need to actually normalize the template. Depending
    # on the specific data uncertainty, the normalization factor
    # may turn out negative. That causes a flip of the correlation function,
    # in which the maximum (correlation peak) is no longer meaningful.
    wave_l = observed_log_spectrum[0,:]
    if normalization > 0.:
        pass
    else:
        normalization = 1.
    wave_l = observed_log_spectrum[0,:]
    
    # masking
    observed_log_spectrum[1,:][observed_log_spectrum[3,:].astype(bool)] = 0
    observed_log_spectrum[2,:][observed_log_spectrum[3,:].astype(bool)] = 0
    
    if mask != None:
        masked_indices = []
        for i in range(len(mask)):
            left_end = abs(observed_log_spectrum[1,:]- mask[i][0]).argmin()
            right_end = abs(observed_log_spectrum[1,:] - mask[i][1]).argmin()
            masked_index = np.arange(left_end, right_end+1)
            masked_indices.append(masked_index)
        masked_indices = np.concatenate(masked_indices)
        observed_log_spectrum[1,:][masked_indices] = 0
        observed_log_spectrum[2,:][masked_indices] = 0
        
    # Correlate
    corr = np.correlate(observed_log_spectrum[1,:],
                        (template_log_spectrum[1,:] * normalization),
                        mode='full')/(np.linalg.norm(observed_log_spectrum[1,:])
                                      *np.linalg.norm(template_log_spectrum[1,:]*normalization))
                        
    wave_l = observed_log_spectrum[0,:]

    # Compute lag
    # wave_l is the wavelength array equally spaced in log space.
    wave_l = observed_log_spectrum[0,:]
    delta_log_wave = np.log10(wave_l[1]) - np.log10(wave_l[0])
    deltas = (np.array(range(len(corr))) - len(corr)/2 + 0.5) * delta_log_wave
    lags = (np.power(10., deltas) - 1)*c

    return lags, corr, observed_log_spectrum

def template_correlate(observed_spectrum, template_spectrum, template_type, clipping=False, hcutoff_scale=10,
                       fs=1, order=2, apodization_window=0.05, mask = None):
    """
    Compute cross-correlation of the observed and template spectra.


    After re-sampling into log-wavelength, both observed and template
    spectra are apodized by a Tukey window in order to minimize edge
    and consequent non-periodicity effects and thus decrease
    high-frequency power in the correlation function. To turn off the
    apodization, use alpha=0.

    Parameters
    ----------
    observed_spectrum : :3xn array
        The observed spectrum.
    template_spectrum : 3xn array
        The template spectrum, which will be correlated with
        the observed spectrum.
    apodization_window: float, callable, or None
        If a callable, will be treated as a window function for apodization of
        the cross-correlation (should behave like a `~scipy.signal.windows`
        window function, with ``sym=True``). If a float, will be treated as the
        ``alpha`` parameter for a Tukey window (`~scipy.signal.windows.tukey`),
        in units of pixels. If None, no apodization will be performed
    mask : list, optional
        Regions which is not used in cross-correlation.
        If None, the full range of the given spectrum is used. The defulat is None.

    Returns
    -------
    lag, corr: 2xn array
        1st row: lag
        2nd row: correlation signals at lags
    """
    _observed_spectrum = copy.deepcopy(observed_spectrum)
    if template_type == 'emission':
        if clipping:
            std = np.std(_observed_spectrum[1,:])
            _observed_spectrum[1,:] = abs_supress(_observed_spectrum[0,:], _observed_spectrum[1,:], 2*std)
            _observed_spectrum[1,:] = emi_supress(_observed_spectrum[0,:], _observed_spectrum[1,:], 8*std)
        else:
            std = np.std(_observed_spectrum[1,:])
            _observed_spectrum[1,:] = abs_supress(_observed_spectrum[0,:], _observed_spectrum[1,:], 2*std)
    elif template_type == 'absorption':
        if hcutoff_scale:
            hcutoff_scale = hcutoff_scale/np.median(_observed_spectrum[0,1:]-_observed_spectrum[0,:-1])
            hcutoff = 1/(2*hcutoff_scale)
            _observed_spectrum[1,:] = butter_highstop_filter(_observed_spectrum[1,:], hcutoff, fs, order)
        if clipping:
            std = np.std(_observed_spectrum[1,:])
            _observed_spectrum[1,:] = emi_supress(_observed_spectrum[0,:], _observed_spectrum[1,:], 2*std)
            _observed_spectrum[1,:] = abs_supress(_observed_spectrum[0,:], _observed_spectrum[1,:], 5*std)
        else:
            std = np.std(_observed_spectrum[1,:])
            _observed_spectrum[1,:] = emi_supress(_observed_spectrum[0,:], _observed_spectrum[1,:], 2*std)
            
    # resample if the user requested to log wavelength
    
    log_spectrum, log_template = template_logwl_resample(_observed_spectrum,
                                                             template_spectrum)

    # apodize (might be a no-op if apodization_window is None)
    observed_log_spectrum, template_log_spectrum = _apodize(log_spectrum,
                                                            log_template,
                                                            apodization_window)
    # Normalize template
    normalization = _normalize(observed_log_spectrum, template_log_spectrum)

    # Not sure if we need to actually normalize the template. Depending
    # on the specific data uncertainty, the normalization factor
    # may turn out negative. That causes a flip of the correlation function,
    # in which the maximum (correlation peak) is no longer meaningful.
    wave_l = observed_log_spectrum[0,:]
    if normalization > 0.:
        pass
    else:
        normalization = 1.
    wave_l = observed_log_spectrum[0,:]
    
    # masking
    observed_log_spectrum[1,:][observed_log_spectrum[3,:].astype(bool)] = 0
    observed_log_spectrum[2,:][observed_log_spectrum[3,:].astype(bool)] = 0
    
    if mask != None:
        for i in range(len(mask)):
            left_end = abs(observed_log_spectrum[0,:]- mask[i][0]).argmin()
            right_end = abs(observed_log_spectrum[0,:] - mask[i][1]).argmin()
            observed_log_spectrum[1,:][left_end:right_end+1] = 0
#             window = 1-tukey(right_end-left_end+1, alpha=0.3)
#             observed_log_spectrum[1,:][left_end:right_end+1] *= window
        
    # Correlate
    corr = np.correlate(observed_log_spectrum[1,:],
                        (template_log_spectrum[1,:] * normalization),
                        mode='full')/(np.linalg.norm(observed_log_spectrum[1,:])
                                      *np.linalg.norm(template_log_spectrum[1,:]*normalization))
                        
    wave_l = observed_log_spectrum[0,:]

    # Compute lag
    # wave_l is the wavelength array equally spaced in log space.
    wave_l = observed_log_spectrum[0,:]
    delta_log_wave = np.log10(wave_l[1]) - np.log10(wave_l[0])
    deltas = (np.array(range(len(corr))) - len(corr)/2 + 0.5) * delta_log_wave
    lags = (np.power(10., deltas) - 1)*c

    return lags, corr, observed_log_spectrum



def _apodize(spectrum, template, apodization_window):
    # Apodization. Must be performed after resampling.
    clean_spectrum = spectrum
    clean_template = template
    
    if apodization_window is not None:
        if callable(apodization_window):
            window = apodization_window
        else:
            def window(wlen):
                return tukey(wlen, alpha=apodization_window)
        clean_spectrum[1,:] = spectrum[1,:] * window(len(spectrum[0,:]))
        clean_spectrum[2,:] = spectrum[2,:] * window(len(spectrum[0,:]))
        clean_template[1,:] = template[1,:] * window(len(template[0,:]))
        clean_template[2,:] = template[2,:] * window(len(template[0,:]))

    return clean_spectrum, clean_template


def template_logwl_resample(spectrum, template, wblue=None, wred=None,
                            delta_log_wavelength=None):
    """
    Resample a spectrum and template onto a common log-spaced spectral grid.

    If wavelength limits are not provided, the function will use
    the limits of the merged (observed+template) wavelength scale
    for building the log-wavelength scale.

    For the wavelength step, the function uses either the smallest wavelength
    interval found in the *observed* spectrum, or takes it from the
    ``delta_log_wavelength`` parameter.

    Parameters
    ----------
    observed_spectrum : 4xn array
        The observed spectrum.
    template_spectrum : 4xn array
        The template spectrum.
    wblue, wred: float
        Wavelength limits to include in the correlation.
    delta_log_wavelength: float
        Log-wavelength step to use to build the log-wavelength
        scale. If None, use limits defined as explained above.
    resampler
        A specutils resampler to use to actually do the resampling.  Defaults to
        using a `~specutils.manipulation.LinearInterpolatedResampler`.
    mask : list, optional
        Regions which is not used in cross-correlation.
        If None, the full range of the given spectrum is used. The defulat is None./

    Returns
    -------
    resampled_observed : 4xn arrauy
        The observed spectrum resampled to a common spectral_axis.
    resampled_template: 4xn array
        The template spectrum resampled to a common spectral_axis.
    """

    # Build an equally-spaced log-wavelength array based on
    # the input and template spectrum's limit wavelengths and
    # smallest sampling interval. Consider only the observed spectrum's
    # sampling, since it's the one that counts for the final accuracy
    # of the correlation. Alternatively, use the wred and wblue limits,
    # and delta log wave provided by the user.
    #
    # We work with separate float and units entities instead of Quantity
    # instances, due to the profusion of log10 and power function calls
    # (they only work on floats)
    
    
    if wblue:
        w0 = np.log10(wblue)
    else:
        ws0 = np.log10(spectrum[0,0])
        wt0 = np.log10(template[0,0])
        w0 = min(ws0, wt0)

    if wred:
        w1 = np.log10(wred)
    else:
        ws1 = np.log10(spectrum[0,-1])
        wt1 = np.log10(template[0,-1])
        w1 = max(ws1, wt1)

    if delta_log_wavelength is None:
        ds = np.log10(template[0,1:]) - np.log10(template[0,:-1])
        dw = ds[np.argmin(ds)]
    else:
        dw = delta_log_wavelength

    nsamples = int((w1 - w0) / dw)

    log_wave_array = np.ones(nsamples) * w0
    for i in range(nsamples):
        log_wave_array[i] += dw * i

    # Build the corresponding wavelength array
    wave_array = np.power(10., log_wave_array)

    # Resample spectrum and template into wavelength array so built
    resampled_spectrum = resampler(spectrum, wave_array)
    resampled_template = resampler(template, wave_array)

    # Resampler leaves Nans on flux bins that aren't touched by it.
    # We replace with zeros. This has the net effect of zero-padding
    # the spectrum and/or template so they exactly match each other,
    # wavelengthwise.
    resampled_spectrum[1,:] = np.nan_to_num(resampled_spectrum[1,:])
    resampled_template[1,:]  = np.nan_to_num(resampled_template[1,:])

    return resampled_spectrum, resampled_template

def _normalize(observed_spectrum, template_spectrum):
    """
    Calculate a scale factor to be applied to the template spectrum so the
    total flux in both spectra will be the same.

    Parameters
    ----------
    observed_spectrum : :class:`~specutils.Spectrum1D`
        The observed spectrum.
    template_spectrum : :class:`~specutils.Spectrum1D`
        The template spectrum, which needs to be normalized in order to be
        compared with the observed spectrum.

    Returns
    -------
    `float`
        A float which will normalize the template spectrum's flux so that it
        can be compared to the observed spectrum.
    """
    num = np.nansum((observed_spectrum[1,:]*template_spectrum[1,:])/(observed_spectrum[2,:]**2))
    denom = np.nansum((template_spectrum[1,:]/observed_spectrum[2,:]**2))

    return num/denom

def butter_highstop_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_lowstop_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y