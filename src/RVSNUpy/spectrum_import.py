import sys
sys.path.insert(0,'../')
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from astroquery.sdss import SDSS
from astropy.table import Table

# Conversion function from vacuum to air wavelengths
def vac2air(wavelength):
    sigma = 1e+4/wavelength
    factor = 5.792105E-2/(238.0185 - sigma**2) + 1.67917E-3/(57.362 - sigma**2)
    return wavelength/(1+factor)

def sdss_fits(file, dr=14, calibration="air"):
    '''
    Parameters
    ----------
    file : directory of the spectrum fits file
    calibartion: wavelength calibration system
        "air" or "vacuum"
    dr : the version of the data release of SDSS

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
        data = f[0].data
        header = f[0].header
        f.close()
        st = header['COEFF0']
        bi = header['COEFF1']
        npix = len(data[0])
        wavelength = np.zeros(npix)
        for i in range(npix):
            wavelength[i] = st + (i+1)*bi
        flux = data[0]
        ivar = data[2]
        mask = data[3]
    
    ivar[ivar==0] = 1e-10
    mask = (~mask.astype(bool)).astype(int)
    if calibration=="air":
        wavelength = vac2air(10**wavelength)
    elif calibration=="vacuum":
        wavelength = 10**wavelength
    else:
        raise ValueError("calibration must be 'air' or 'vacuum'")
    
    uncertainty = (1/np.sqrt(ivar))
    # create spectrum1D
    spectrum = np.vstack([wavelength, flux, uncertainty, mask])
    f.close()
        
    return spectrum

def MMT_raw(file):
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
    uncertainty = (f[0].data)[3][0]
    f.close()
    
    sz = len(flux)
    crval = header['CRVAL1']
    
    if 'CDELT1' in header:
        cdelt = header['CDELT1']
    
    else:
        cdelt = header['CD1_1']
        
    crpix = header['CRPIX1']
        
    wavelength = (np.arange(sz) - crpix + 1)*cdelt+crval
    wavelength = wavelength
        
    mask = np.ones(len(wavelength))
        
    spectrum = np.vstack([wavelength, flux, uncertainty, mask])
    
    f.close()
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
        
    mask = np.ones(len(wavelength))
        
    spectrum = np.vstack([wavelength, flux, uncertainty, mask])
    f.close()
    return spectrum



