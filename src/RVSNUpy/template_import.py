from astropy.io import fits
import numpy as np
import pandas as pd
import os
# from continuum import subtract_continuum
from rvm import resampler, discrete_resampler
from astropy.modeling import models, fitting
import warnings
import glob
from spectrum_import import vac2air


path = os.path.join(os.path.dirname(__file__), 'template_files')
if len(glob.glob(os.path.join(path, 'csv', 'sdss')))==0:
    folder = os.path.join(path, 'csv', 'sdss')
    os.system(f'mkdir {folder}')
else:
    exist_files = os.path.join(path, 'csv', 'sdss', '*')
    os.system(f'rm -fr {exist_files}')
    
template_name = ['Early_type_galaxy', 'Galaxy','Late_type_galaxy', 'Luminous_red_galaxy']
sdss_template_galaxy = {}
for k, i in enumerate([23,24,27,28]):
    # import a fits file
    fits_path = os.path.join(path,'fits','spectemplatesDR2','spDR2-0%d.fit'%i)
    f = fits.open(fits_path)
    sdss_template_data = f[0].data
    sdss_template_header = f[0].header
    f.close()
    st = sdss_template_header['COEFF0']
    bi = sdss_template_header['COEFF1']
    npix = len(sdss_template_data[0])
    log_wavelength = np.zeros(npix)
    wave = st
    for j in range(0,npix):
        log_wavelength[j] = wave
        wave += bi
    
    wavelength_vac = 10**log_wavelength
    wavelength_air = vac2air(10**log_wavelength)
    flux = sdss_template_data[0]
    uncertainty = 1e-4*flux
    
    left, right = 0,-1
    while flux[left]==0:
        left+=1
    while flux[right]==0:
        right-=1
    flux = flux[left:right]
    uncertainty = uncertainty[left:right]
    wavelength_vac = wavelength_vac[left:right]
    wavelength_air = wavelength_air[left:right]
                
    try:
        wavelength_vac = wavelength_vac/(1+float(sdss_template_header['Z']))
        wavelength_air = wavelength_air/(1+float(sdss_template_header['Z']))
    except:
        pass
    
    template_vac = np.vstack([wavelength_vac, flux, uncertainty, np.zeros(len(wavelength_vac))])
    
    # resample the wavleneght_air loagarithmically
    template_air = np.vstack([wavelength_air, flux, uncertainty, np.zeros(len(wavelength_air))])
    
    ds = np.log10(template_air[0,1:]) - np.log10(template_air[0,:-1])
    dw = ds[np.argmin(ds)]

    nsamples = int((np.log10(template_air[0,-1]) - np.log10(template_air[0,0])) / dw)

    log_new_wavelengths = np.ones(nsamples) * np.log10(template_air[0,0])
    for j in range(nsamples):
        log_new_wavelengths[j] += dw * j

    # Build the corresponding wavelength array
    new_wavelengths = pow(10,log_new_wavelengths)
    template_air = np.vstack([new_wavelengths, resampler(template_air[0],template_air[1], new_wavelengths), 
                                    resampler(template_air[0], template_air[2], new_wavelengths),
                                    discrete_resampler(template_air[0], template_air[3], new_wavelengths)])
    # template = subtract_continuum(template)
    
    # save the template data
    df_template_vac = pd.DataFrame({'wave':template_vac[0,:], 'flux':template_vac[1,:],
                                'uncertainty':template_vac[2,:]})
    csv_path_vac = os.path.join(path, 'csv','sdss',template_name[k]+'_vacuum.csv')
    df_template_vac.to_csv(csv_path_vac, index=False)
    
    df_template_air = pd.DataFrame({'wave':template_air[0,:], 'flux':template_air[1,:],
                                'uncertainty':template_air[2,:]})
    csv_path_air = os.path.join(path, 'csv','sdss',template_name[k]+'_air.csv')
    df_template_air.to_csv(csv_path_air, index=False)