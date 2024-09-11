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
    
template_name = ['Early_type_galaxy', 'Late_type_galaxy', 'Luminous_red_galaxy']
sdss_template_galaxy = {}
for k, i in enumerate([23,27,28]):
    # import a fits file
    fits_path = os.path.join(path,'fits','spectemplatesDR2','spDR2-0%d.fit'%i)
    f = fits.open(fits_path)
    sdss_template_data = f[0].data
    sdss_template_header = f[0].header
    f.close()
    st = sdss_template_header['COEFF0']
    bi = sdss_template_header['COEFF1']
    npix = len(sdss_template_data[0])
    wavelength = np.zeros(npix)
    wave = st
    for j in range(0,npix):
        wavelength[j] = wave
        wave += bi
        
    wavelength = vac2air(10**wavelength)
    flux = sdss_template_data[0]
    uncertainty = 1e-4*flux
    
    flux = flux[(wavelength>3800) & (wavelength<8900)]
    uncertainty = uncertainty[(wavelength>3800) &( wavelength<8900)]
    wavelength = wavelength[(wavelength>3800) & (wavelength<8900)]
                
    try:
        wavelength = wavelength/(1+float(sdss_template_header['Z']))
    except:
        pass
    
    # construct a continuum-subtracted template
    template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(wavelength))])
    
    ds = np.log10(template[0,1:]) - np.log10(template[0,:-1])
    dw = ds[np.argmin(ds)]

    nsamples = int((np.log10(template[0,-1]) - np.log10(template[0,0])) / dw)

    log_new_wavelengths = np.ones(nsamples) * np.log10(template[0,0])
    for j in range(nsamples):
        log_new_wavelengths[j] += dw * j

    # Build the corresponding wavelength array
    new_wavelengths = pow(10,log_new_wavelengths)
    template = np.vstack([new_wavelengths, resampler(template[0],template[1], new_wavelengths), 
                                    resampler(template[0], template[2], new_wavelengths),
                                    discrete_resampler(template[0], template[3], new_wavelengths)])
    # template = subtract_continuum(template)
    
    # save the template data
    df_template = pd.DataFrame({'wave':template[0,:], 'flux':template[1,:],
                                'uncertainty':template[2,:]})
    csv_path = os.path.join(path, 'csv','sdss',template_name[k]+'.csv')
    df_template.to_csv(csv_path, index=False)
    

# ##########################################################################################
# # sdss template
# ##########################################################################################

# template_name = ['O_star', 'OB_transition star', 'B_star',
#                'A_star1', 'A_star2', 'FA_transition_star',
#                'F_star1', 'F_star2', 'G_star0', 'G_star1',
#                'K_star', 'M1_star', 'M3_star1',
#                'M5_star2', 'M8_star', 'L1_star',
#                'Magnetic_white_dwarf', 'Carbon_star1', 'Carbon_star2', 'Carbon_star3',
#                'White_dwarf', 'B_white_dwarf', 'Low_metallicity_K_subdwarf', 'Early_type_galaxy',
#                'Galaxy1', 'Galaxy2', 'Galaxy3', 'Late_type_galaxy', 'Luminous_red_galaxy',
#                'QSO', 'QSO_with_some_BAL_activity1', 'Qso_with_some_BAL_activity2', 'High_luminosity_QSO']

# sdss_template_star = {}
# for i in range(0,10):
#     # import a fits file
#     fits_path = os.path.join(path,'fits','spectemplatesDR2','spDR2-00{}.fit'.format(i))
#     f = fits.open(fits_path)
#     sdss_template_data = f[0].data
#     sdss_template_header = f[0].header
#     f.close()
#     st = sdss_template_header['COEFF0']
#     bi = sdss_template_header['COEFF1']
#     npix = len(sdss_template_data[0])
#     wavelength = np.zeros(npix)
#     wave = st
#     for j in range(0,npix):
#         wavelength[j] = wave
#         wave += bi
        
#     wavelength = vac2air(10**wavelength)
#     flux = sdss_template_data[1]
#     ivar = sdss_template_data[2]
#     ivar[ivar==0] = 1e-10
#     uncertainty = 1/np.sqrt(ivar)
    
#     flux = flux[(wavelength>3800) & (wavelength<8900)]
#     uncertainty = uncertainty[(wavelength>3800) &( wavelength<8900)]
#     wavelength = wavelength[(wavelength>3800) & (wavelength<8900)]
    
#     try:
#         wavelength = wavelength/(1+float(sdss_template_header['Z']))
#     except:
#         pass
        
#     # construct a continuum-subtracted template
#     template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(wavelength))])

#     ds = np.log10(template[0,1:]) - np.log10(template[0,:-1])
#     dw = ds[np.argmin(ds)]

#     nsamples = int((np.log10(template[0,-1]) - np.log10(template[0,0])) / dw)

#     log_new_wavelengths = np.ones(nsamples) * np.log10(template[0,0])
#     for j in range(nsamples):
#         log_new_wavelengths[j] += dw * j

#     # Build the corresponding wavelength array
#     new_wavelengths = pow(10,log_new_wavelengths)
#     template = np.vstack([new_wavelengths, resampler(template[0],template[1], new_wavelengths), 
#                                     resampler(template[0], template[2], new_wavelengths),
#                                     discrete_resampler(template[0], template[3], new_wavelengths)])
    
#     # save the template data
#     df_template = pd.DataFrame({'wave':template[0,:], 'flux':template[1,:],
#                                 'uncertainty':template[2,:]})
#     csv_path = os.path.join(path, 'csv','sdss','star',template_name[i]+'.csv')
#     df_template.to_csv(csv_path, index=False)

# for i in range(10,23):
#     # import a fits file
#     fits_path = os.path.join(path,'fits','spectemplatesDR2','spDR2-0{}.fit'.format(i))
#     f = fits.open(fits_path)
#     sdss_template_data = f[0].data
#     sdss_template_header = f[0].header
#     f.close()
#     st = sdss_template_header['COEFF0']
#     bi = sdss_template_header['COEFF1']
#     npix = len(sdss_template_data[0])
#     wavelength = np.zeros(npix)
#     wave = st
#     for j in range(0,npix):
#         wavelength[j] = wave
#         wave += bi
        
#     wavelength = vac2air(10**wavelength)
#     flux = sdss_template_data[0]
#     ivar = sdss_template_data[2]
#     ivar[ivar==0] = 1e-10
#     uncertainty = 1/np.sqrt(ivar)
    
#     flux = flux[(wavelength>3800) & (wavelength<8900)]
#     uncertainty = uncertainty[(wavelength>3800) &( wavelength<8900)]
#     wavelength = wavelength[(wavelength>3800) & (wavelength<8900)]
                
#     try:
#         wavelength = wavelength/(1+float(sdss_template_header['Z']))
#     except:
#         pass
    
#     # construct a continuum-subtracted template
#     template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(wavelength))])
    
#     ds = np.log10(template[0,1:]) - np.log10(template[0,:-1])
#     dw = ds[np.argmin(ds)]

#     nsamples = int((np.log10(template[0,-1]) - np.log10(template[0,0])) / dw)

#     log_new_wavelengths = np.ones(nsamples) * np.log10(template[0,0])
#     for j in range(nsamples):
#         log_new_wavelengths[j] += dw * j

#     # Build the corresponding wavelength array
#     new_wavelengths = pow(10,log_new_wavelengths)
#     template = np.vstack([new_wavelengths, resampler(template[0],template[1], new_wavelengths), 
#                                     resampler(template[0], template[2], new_wavelengths),
#                                     discrete_resampler(template[0], template[3], new_wavelengths)])
#     # template = subtract_continuum(template)
    
#     # save the template data
#     df_template = pd.DataFrame({'wave':template[0,:], 'flux':template[1,:],
#                                 'uncertainty':template[2,:]})
#     csv_path = os.path.join(path, 'csv','sdss','star',template_name[i]+'.csv')
#     df_template.to_csv(csv_path, index=False)

# sdss_template_galaxy = {}
# for i in range(23,29):
#     # import a fits file
#     fits_path = os.path.join(path,'fits','spectemplatesDR2','spDR2-0{}.fit'.format(i))
#     f = fits.open(fits_path)
#     sdss_template_data = f[0].data
#     sdss_template_header = f[0].header
#     f.close()
#     st = sdss_template_header['COEFF0']
#     bi = sdss_template_header['COEFF1']
#     npix = len(sdss_template_data[0])
#     wavelength = np.zeros(npix)
#     wave = st
#     for j in range(0,npix):
#         wavelength[j] = wave
#         wave += bi
        
#     wavelength = vac2air(10**wavelength)
#     flux = sdss_template_data[0]
#     ivar = sdss_template_data[2]
#     ivar[ivar==0] = 1e-10
#     uncertainty = 1/np.sqrt(ivar)
    
#     flux = flux[(wavelength>3800) & (wavelength<8900)]
#     uncertainty = uncertainty[(wavelength>3800) &( wavelength<8900)]
#     wavelength = wavelength[(wavelength>3800) & (wavelength<8900)]
                
#     try:
#         wavelength = wavelength/(1+float(sdss_template_header['Z']))
#     except:
#         pass
    
#     # construct a continuum-subtracted template
#     template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(wavelength))])
    
#     ds = np.log10(template[0,1:]) - np.log10(template[0,:-1])
#     dw = ds[np.argmin(ds)]

#     nsamples = int((np.log10(template[0,-1]) - np.log10(template[0,0])) / dw)

#     log_new_wavelengths = np.ones(nsamples) * np.log10(template[0,0])
#     for j in range(nsamples):
#         log_new_wavelengths[j] += dw * j

#     # Build the corresponding wavelength array
#     new_wavelengths = pow(10,log_new_wavelengths)
#     template = np.vstack([new_wavelengths, resampler(template[0],template[1], new_wavelengths), 
#                                     resampler(template[0], template[2], new_wavelengths),
#                                     discrete_resampler(template[0], template[3], new_wavelengths)])
#     # template = subtract_continuum(template)
    
#     # save the template data
#     df_template = pd.DataFrame({'wave':template[0,:], 'flux':template[1,:],
#                                 'uncertainty':template[2,:]})
#     csv_path = os.path.join(path, 'csv','sdss','galaxy',template_name[i]+'.csv')
#     df_template.to_csv(csv_path, index=False)
    
# sdss_template_QSO = {}
# for i in range(29,33):
#     fits_path = os.path.join(path,'fits','spectemplatesDR2','spDR2-0{}.fit'.format(i))
#     f = fits.open(fits_path)
#     sdss_template_data = f[0].data
#     sdss_template_header = f[0].header
#     f.close()
#     st = sdss_template_header['COEFF0']
#     bi = sdss_template_header['COEFF1']
#     npix = len(sdss_template_data[0])
#     wavelength = np.zeros(npix)
#     wave = st
#     for j in range(0,npix):
#         wavelength[j] = wave
#         wave += bi
        
#     wavelength = vac2air(10**wavelength)
#     flux = sdss_template_data[0]
#     ivar = sdss_template_data[2]
#     ivar[ivar==0] = 1e-10
#     uncertainty = 1/np.sqrt(ivar)
    
#     if i == 30:
#         flux = flux[(wavelength>1000) & (wavelength<7500)]
#         uncertainty = uncertainty[(wavelength>1000) &(wavelength<7500)]
#         wavelength = wavelength[(wavelength>1000) & (wavelength<7500)]
        
#     elif i == 32:
#         flux = flux[(wavelength>1300) & (wavelength<6200)]
#         uncertainty = uncertainty[(wavelength>1300) &(wavelength<6200)]
#         wavelength = wavelength[(wavelength>1300) & (wavelength<6200)]
    
#     else:
#         flux = flux[(wavelength>1200) & (wavelength<3500)]
#         uncertainty = uncertainty[(wavelength>1200) &( wavelength<3500)]
#         wavelength = wavelength[(wavelength>1200) & (wavelength<3500)]
                
#     try:
#         wavelength = wavelength/(1+float(sdss_template_header['Z']))
#     except:
#         pass
    
#     # construct a continuum-subtracted template
#     template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(wavelength))])
#     template = template[:,np.argsort(template[0,:])] # some spectra is not sorted in the increaseing sequence
    
#     ds = np.log10(template[0,1:]) - np.log10(template[0,:-1])
#     dw = ds[np.argmin(ds)]

#     nsamples = int((np.log10(template[0,-1]) - np.log10(template[0,0])) / dw)

#     log_new_wavelengths = np.ones(nsamples) * np.log10(template[0,0])
#     for j in range(nsamples):
#         log_new_wavelengths[j] += dw * j

#     # Build the corresponding wavelength array
#     new_wavelengths = pow(10,log_new_wavelengths)
#     template = np.vstack([new_wavelengths, resampler(template[0],template[1], new_wavelengths), 
#                                     resampler(template[0], template[2], new_wavelengths),
#                                     discrete_resampler(template[0], template[3], new_wavelengths)])
# #     template = subtract_continuum(template)
    
#     # save the template data
#     df_template = pd.DataFrame({'wave':template[0,:], 'flux':template[1,:],
#                                 'uncertainty':template[2,:]})
#     csv_path = os.path.join(path, 'csv','sdss','QSO',template_name[i]+'.csv')
#     df_template.to_csv(csv_path, index=False)

# ##########################################################################################

# ##########################################################################################
# # MMT template
# ##########################################################################################

# from astropy import units as u
# from astropy.constants import c
# c = c.value/1e+3

# template_name = ['eatemp', 'eltemp', 'hemtemp0.0', 'habtemp90', 'm31_a_temp', 'm31_f_temp',
#                'm31_k_temp', 'sptemp']

# MMT_template = {}

# for name in template_name:
#     fits_path = os.path.join(path,'fits','MMTtemplate', name+'.fits')
#     f = fits.open(fits_path)
#     header = f[0].header
#     flux = f[0].data
#     f.close()
#     sz = len(flux)
#     crval = header['CRVAL1']
#     if 'CDELT1' in header:
#         cdelt = header['CDELT1']
#     else:
#         cdelt = header['CD1_1']
#     crpix = header['CRPIX1']
#     wavelength = ((np.arange(sz) - crpix + 1)*cdelt+crval)
#     z = header['VELOCITY']/c
#     wavelength = wavelength/(1+z)
#     uncertainty = np.zeros(sz)
    
#     template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(flux))])
#     ds = np.log10(template[0,1:]) - np.log10(template[0,:-1])
#     dw = ds[np.argmin(ds)]

#     nsamples = int((np.log10(template[0,-1]) - np.log10(template[0,0])) / dw)

#     log_new_wavelengths = np.ones(nsamples) * np.log10(template[0,0])
#     for i in range(nsamples):
#         log_new_wavelengths[i] += dw * i

#     # Build the corresponding wavelength array
#     new_wavelengths = pow(10,log_new_wavelengths)
#     template = np.vstack([new_wavelengths, resampler(template[0],template[1], new_wavelengths), 
#                                     resampler(template[0], template[2], new_wavelengths),
#                                     discrete_resampler(template[0], template[3], new_wavelengths)])
    
#     # save the template data
#     df_template = pd.DataFrame({'wave':template[0,:], 'flux':template[1,:],
#                                 'uncertainty':template[2,:]})
#     csv_path = os.path.join(path, 'csv','MMT', name+'.csv')
#     df_template.to_csv(csv_path, index=False)