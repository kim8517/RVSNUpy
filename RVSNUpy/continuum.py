import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sigmaclip
import copy
from joblib import Parallel, delayed

def median_filter(flux, I, window, sigma):
    flux_length = len(flux)
    continuum_flux = np.zeros(flux_length)
    for i in I:
        left, right = max(0, i-window), min(i+window+1, flux_length)
        _window = sigmaclip(flux[left:right], low=sigma, high=sigma)[0]
        continuum_flux[i] = np.ma.median(_window)
    return continuum_flux

def continuum_finding(spectrum, sigma=3, window = 35, plotting = False):
    '''

    Parameters
    ----------
    spectrum : 3xn array
    sigma : integer, optional
        The sigma will be used in sigma_clip. The pixel which is separted from the mean
        more that sigma will be masekd. The default is 3.
    window : Integer, optional
        The size of the window for running median The default is 60.
    plotting : bool, optional
        If true, plot the continuum's spectrum. The default is False.

    Returns
    -------
    continuum : 3xn array
    
    '''
    flux = spectrum[1,:]
    flux_length = len(flux) 
    # create the continuum_flux by running median
    section = [np.arange(int(flux_length*0.25)), np.arange(int(flux_length*0.25), int(flux_length*0.5)),
              np.arange(int(flux_length*0.5), int(flux_length*0.75)), np.arange(int(flux_length*0.75), flux_length)]
    continuum_flux = Parallel(n_jobs=4, verbose=0)(delayed(median_filter)(flux, I, window, sigma) for I in section)
    # creae the continuum for the given spectrum
    continuum = np.vstack([spectrum[0,:], sum(continuum_flux), spectrum[2,:], spectrum[3,:]])
    
    if plotting:
        plt.figure()
        plt.step(spectrum[0,:], spectrum[1,:], 'k-', label='original spectrum')
        plt.step(continuum[0,:], continuum[1,:], 'r-', label='continuum')
        plt.xlabel('Wavelength (AA)')
        plt.ylabel('Flux (\'erg cm-2 s-1 AA-1\')')
        plt.legend()
        plt.show()
    return continuum

def continuum_subtraction(spectrum, sigma=3, window = 35, plotting = False):
    '''

    Parameters
    ----------
    spectrum : 3xn array
    sigma : integer, optional
        The sigma will be used in sigma_clip. The pixel which is separted from the mean
        more that sigma will be masekd. The default is 3.
    window : Integer, optional
        The size of the window for running median The default is 60.
    plotting : bool, optional
        If true, plot the continuum's spectrum. The default is False.

    Returns
    -------
    continuum_subtracted : 3xn array
        continuum_subtracted spectrum

    '''
    continuum = continuum_finding(spectrum, sigma=sigma, window=window)
    continuum_subtracted = np.vstack([spectrum[0,:], spectrum[1,:] - continuum[1,:], spectrum[2,:], spectrum[3,:]]) 
    
    if plotting:
        plt.figure()
        plt.step(continuum_subtracted[0,:], continuum_subtracted[1,:], 'k-')
        plt.title('continuum_subtracted spectrum')
        plt.xlabel('Wavelength (AA)')
        plt.ylabel('Flux (\'erg cm-2 s-1 AA-1\')')
        plt.show()
        
    return continuum_subtracted