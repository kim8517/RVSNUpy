import sys
sys.path.insert(0,'../')
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from astroquery.sdss import SDSS
from astropy.table import Table
from RVSNUpy.correlation import resampler


def vac2air(wavelength):
    sigma = 1e+4/wavelength
    factor = 6.4328e-5+2.9481e-2/(146-sigma**2)+2.5540e-4/(41-sigma**2)
    return wavelength/(1+factor)

def sdss_fits(file, dr=14, plotting=False):
    '''
    Parameters
    ----------
    file : str
        Directory of the file
    dr : integer
        The version of the data release of SDSS
    plotting : bool, optional
        If true, plot the spectrum. The default is False.

    Returns
    -------
    spectrum : 4xn array
        spectrum
        0th row: wavelength (AA), 1st row: flux (erg cm-2 s-1 AA-1), 2nd row: uncertainty (erg cm-2 s-1 AA-1), 3rd row: mask (0 or 1)

    '''
    f = fits.open(file)
    if dr>7:
        data = f[1].data
        f.close()
        data = Table(data)
        try:
            flux, wavelength, ivar, mask = np.array(data['flux']), np.array(data['loglam']), np.array(data['ivar']), np.array(data['and_mask'])
        except:
            flux, wavelength, ivar, mask = np.array(data['FLUX']), np.array(data['LOGLAM']), np.array(data['IVAR']), np.array(data['AND_MASK'])
    else:
        data = f[0].data # import data
        header = f[0].header # import header
        f.close()
        st = header['COEFF0'] # the lowest wavelength
        bi = header['COEFF1'] # the bin between wavelength
        # create the wavelength range
        npix = len(data[0])
        wavelength = np.zeros(npix)
        for i in range(npix):
            wavelength[i] = st + (i+1)*bi
        flux = data[0]
        ivar = data[2]
        mask = data[3]
        
    wavelength = vac2air(10**wavelength)
    
    # For cross-correlation, uncertainty is processed as a standard deviation
    uncertainty = (1/np.sqrt(ivar))*10**-17
    # create spectrum1D
    _spectrum = np.vstack([wavelength, flux*10**-17, uncertainty, mask])
    
    linear_wave = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
    spectrum = resampler(_spectrum,linear_wave)
    
    if plotting:
        plt.figure()
        plt.step(spectrum[0,:], spectrum[1,:],'k-')
        plt.xlabel('Wavelength (AA)')
        plt.ylabel('Flux (\'erg cm-2 s-1 AA-1\')')
        plt.show()
        
    return spectrum

def MMT_raw(file, plotting=False):
    '''
    Parameters
    ----------
    file : str
        Directory of the file
    plotting : bool, optional
        If true, plot the spectrum. The default is False.

    Returns
    -------
    spectrum : Spectrum1D
        spectrum : 4xn array
        spectrum
        0th row: wavelength (AA), 1st row: relative flux (no unit), 2nd row: uncertainty (no unit), 3rd row: mask (0 or 1)
    '''
    f = fits.open(file)
    header = f[0].header
    flux = (f[0].data)[0][0]
    flux = flux
    uncertainty = ((f[0].data)[3][0])
    uncertainty = (1/np.sqrt(uncertainty))
    
    sz = len(flux)
    crval = header['CRVAL1']
    
    if 'CDELT1' in header:
        cdelt = header['CDELT1']
    
    else:
        cdelt = header['CD1_1']
        
    crpix = header['CRPIX1']
        
    wavelength = (np.arange(sz) - crpix + 1)*cdelt+crval
    wavelength = wavelength
        
    mask = np.zeros(len(wavelength))
        
    spectrum = np.vstack([wavelength, flux, uncertainty, mask])
    
    if plotting:
        plt.figure()
        plt.step(spectrum[0,:], spectrum[1,:],'k-')
        plt.xlabel('Wavelength (AA)')
        plt.ylabel('Flux (\'erg cm-2 s-1 AA-1\')')
        plt.show()
    
    return spectrum

def MMT_flux(file):
    f = fits.open(file)
    header = f[0].header
    flux = (f[0].data)[0]
    uncertainty = ((f[0].data)[1])
    uncertainty = (1/np.sqrt(uncertainty))
    
    sz = len(flux)
    crval = header['CRVAL1']
    
    if 'CDELT1' in header:
        cdelt = header['CDELT1']
    
    else:
        cdelt = header['CD1_1']
        
    crpix = header['CRPIX1']
        
    wavelength = (np.arange(sz) - crpix + 1)*cdelt+crval
    wavelength = wavelength
        
    mask = np.zeros(len(wavelength))
        
    spectrum = np.vstack([wavelength, flux, uncertainty, mask])
    return spectrum
