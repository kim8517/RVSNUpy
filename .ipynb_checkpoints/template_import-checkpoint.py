from astropy.io import fits
import numpy as np
import pandas as pd
import os
from continuum import continuum_subtraction
from correlation import template_correlate, resampler
from rvm import z_finding
from astropy.modeling import models, fitting
import warnings
warnings.filterwarnings("ignore")

from spectrum_import import vac2air


path = os.path.join(os.path.dirname(__file__), 'template_files')
##########################################################################################
# sdss template
##########################################################################################

template_name = ['O star', 'OB transition star', 'B star',
               'A star0', 'A star1', 'FA transition star',
               'F star0', 'F star1', 'G star0', 'G star1',
               'K star', 'M1 star', 'M3 star0',
               'M5 star1', 'M8 star', 'L1 star',
               'Magnetic white dwarf', 'Carbon star0', 'Carbon star1', 'Carbon star2',
               'White dwarf', 'B white dwarf', 'Low-metallicity K subdwarf', 'Early-type galaxy',
               'Galaxy0', 'Galaxy1', 'Galaxy2', 'Late-type galaxy', 'Luminous Red Galaxy',
               'QSO', 'QSO with some BAL activity0', 'QSO with some BAL activity1', 'High-luminosity QSO']

dispersion_csv = open(os.path.join(path, 'csv','dispersion', 'sdss.csv'), 'w')

sdss_template_star = {}
for i in range(0,10):
    # import a fits file
    fits_path = os.path.join(path,'fits','spectemplatesDR2','spDR2-00{}.fit'.format(i))
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
    flux = sdss_template_data[1]
    uncertainty = 1/np.sqrt(sdss_template_data[2])
    
    flux = flux[(wavelength>3800) & (wavelength<8900)]
    uncertainty = uncertainty[(wavelength>3800) &( wavelength<8900)]
    wavelength = wavelength[(wavelength>3800) & (wavelength<8900)]
    
    try:
        wavelength = wavelength/(1+float(sdss_template_header['Z']))
    except:
        pass
        
    # construct a continuum-subtracted template
    template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(wavelength))])
    linear_wave = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
    linear_template = resampler(template,linear_wave)
    template = continuum_subtraction(linear_template, window=90)

    # measure the dispersion of template (refer z_finding in RVM)
    # fit to the correlation signal
    lag, corr,_ = template_correlate(template, template, template_type='nothing', hcutoff_scale=2.5, lcutoff_scale=110,
                                  apodization_window=0.05, mask=None)
    z, r, _,_,gaussian_fit, _, _, _, _ = z_finding(corr, lag, pkfrac = 0.55)
    err_dispersion = (3/8)*2*np.sqrt(2*np.log(2))*gaussian_fit.stddev.value/(1+r)
    
    # save the template data
    df_template = pd.DataFrame({'wave':template[0,:], 'flux':template[1,:],
                                'uncertainty':template[2,:]})
    csv_path = os.path.join(path, 'csv','sdss','star',template_name[i]+'.csv')
    df_template.to_csv(csv_path, index=False)
    dispersion_csv.write(template_name[i]+',{},{}\n'.format(gaussian_fit.stddev.value, err_dispersion))

for i in range(10,23):
    # import a fits file
    fits_path = os.path.join(path,'fits','spectemplatesDR2','spDR2-0{}.fit'.format(i))
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
    uncertainty = 1/np.sqrt(sdss_template_data[2])
    
    flux = flux[(wavelength>3800) & (wavelength<8900)]
    uncertainty = uncertainty[(wavelength>3800) &( wavelength<8900)]
    wavelength = wavelength[(wavelength>3800) & (wavelength<8900)]
                
    try:
        wavelength = wavelength/(1+float(sdss_template_header['Z']))
    except:
        pass
    
    # construct a continuum-subtracted template
    template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(wavelength))])
    linear_wave = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
    linear_template = resampler(template,linear_wave)
    template = continuum_subtraction(linear_template, window=90)

    # measure the dispersion of template (refer z_finding in RVM)
    # fit to the correlation signal
    lag, corr,_ = template_correlate(template, template, template_type='nothing', hcutoff_scale=2.5, lcutoff_scale=110,
                                  apodization_window=0.05, mask=None)
    z, r, _,_,gaussian_fit, _, _, _, _ = z_finding(corr, lag, pkfrac = 0.55)
    err_dispersion = (3/8)*2*np.sqrt(2*np.log(2))*gaussian_fit.stddev.value/(1+r)
    
    # save the template data
    df_template = pd.DataFrame({'wave':template[0,:], 'flux':template[1,:],
                                'uncertainty':template[2,:]})
    csv_path = os.path.join(path, 'csv','sdss','star',template_name[i]+'.csv')
    df_template.to_csv(csv_path, index=False)
    dispersion_csv.write(template_name[i]+',{},{}\n'.format(gaussian_fit.stddev.value, err_dispersion))

sdss_template_galaxy = {}
for i in range(23,29):
    # import a fits file
    fits_path = os.path.join(path,'fits','spectemplatesDR2','spDR2-0{}.fit'.format(i))
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
    uncertainty = 1/np.sqrt(sdss_template_data[2])
    
    flux = flux[(wavelength>3800) & (wavelength<8900)]
    uncertainty = uncertainty[(wavelength>3800) &( wavelength<8900)]
    wavelength = wavelength[(wavelength>3800) & (wavelength<8900)]
                
    try:
        wavelength = wavelength/(1+float(sdss_template_header['Z']))
    except:
        pass
    
    # construct a continuum-subtracted template
    template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(wavelength))])
    linear_wave = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
    linear_template = resampler(template,linear_wave)
    template = continuum_subtraction(linear_template, window=90)

    # measure the dispersion of template (refer z_finding in RVM)
    # fit to the correlation signal
    lag, corr,_ = template_correlate(template, template, template_type='nothing', hcutoff_scale=0,
                                  apodization_window=0.05, mask=None)
    z, r, _,_,gaussian_fit, _, _, _, _ = z_finding(corr, lag, pkfrac = 0.7)
    err_dispersion = (3/8)*2*np.sqrt(2*np.log(2))*gaussian_fit.stddev.value/(1+r)
    
    # save the template data
    df_template = pd.DataFrame({'wave':template[0,:], 'flux':template[1,:],
                                'uncertainty':template[2,:]})
    csv_path = os.path.join(path, 'csv','sdss','galaxy',template_name[i]+'.csv')
    df_template.to_csv(csv_path, index=False)
    dispersion_csv.write(template_name[i]+',{},{}\n'.format(gaussian_fit.stddev.value, err_dispersion))
    
sdss_template_QSO = {}
for i in range(29,33):
    fits_path = os.path.join(path,'fits','spectemplatesDR2','spDR2-0{}.fit'.format(i))
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
    uncertainty = 1/np.sqrt(sdss_template_data[2])
    
    if i == 30:
        flux = flux[(wavelength>1000) & (wavelength<7500)]
        uncertainty = uncertainty[(wavelength>1000) &(wavelength<7500)]
        wavelength = wavelength[(wavelength>1000) & (wavelength<7500)]
        
    elif i == 32:
        flux = flux[(wavelength>1300) & (wavelength<6200)]
        uncertainty = uncertainty[(wavelength>1300) &(wavelength<6200)]
        wavelength = wavelength[(wavelength>1300) & (wavelength<6200)]
    
    else:
        flux = flux[(wavelength>1200) & (wavelength<3500)]
        uncertainty = uncertainty[(wavelength>1200) &( wavelength<3500)]
        wavelength = wavelength[(wavelength>1200) & (wavelength<3500)]
                
    try:
        wavelength = wavelength/(1+float(sdss_template_header['Z']))
    except:
        pass
    
    # construct a continuum-subtracted template
    template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(wavelength))])
    linear_wave = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
    linear_template = resampler(template,linear_wave)
    template = continuum_subtraction(linear_template, window=90)

    # measure the dispersion of template (refer z_finding in RVM)
    # fit to the correlation signal
    lag, corr,_ = template_correlate(template, template, template_type='nothing', hcutoff_scale=0,
                                  apodization_window=0.05, mask=None)
    z, r, _,_,gaussian_fit, _, _, _, _ = z_finding(corr, lag, pkfrac = 0.55)
    err_dispersion = (3/8)*2*np.sqrt(2*np.log(2))*gaussian_fit.stddev.value/(1+r)
    
    # save the template data
    df_template = pd.DataFrame({'wave':template[0,:], 'flux':template[1,:],
                                'uncertainty':template[2,:]})
    csv_path = os.path.join(path, 'csv','sdss','QSO',template_name[i]+'.csv')
    df_template.to_csv(csv_path, index=False)
    dispersion_csv.write(template_name[i]+',{},{}\n'.format(gaussian_fit.stddev.value, err_dispersion))
    
dispersion_csv.close()

##########################################################################################

##########################################################################################
# MMT template
##########################################################################################

from specutils import Spectrum1D
from astropy import units as u
import continuum
from astropy.constants import c
c = c.value/1e+3

template_name = ['eatemp', 'eltemp', 'hemtemp0.0', 'habtemp90', 'm31_a_temp', 'm31_f_temp',
               'm31_k_temp', 'sptemp']

MMT_template = {}
dispersion_csv = open(os.path.join(path, 'csv','dispersion', 'MMT.csv'), 'w')

for name in template_name:
    fits_path = os.path.join(path,'fits','MMTtemplate', name+'.fits')
    f = fits.open(fits_path)
    header = f[0].header
    flux = f[0].data
    f.close()
    sz = len(flux)
    crval = header['CRVAL1']
    if 'CDELT1' in header:
        cdelt = header['CDELT1']
    else:
        cdelt = header['CD1_1']
    crpix = header['CRPIX1']
    wavelength = ((np.arange(sz) - crpix + 1)*cdelt+crval)
    z = header['VELOCITY']/c
    wavelength = wavelength/(1+z)
    uncertainty = np.zeros(sz)
    
    template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(flux))])
    template = continuum.continuum_subtraction(template, window=35)
    
    # measure the dispersion of template (refer z_finding in RVM)
    lag, corr,_ = template_correlate(template, template, template_type='nothing', hcutoff_scale=0,
                                  apodization_window=0.05, mask=None)
    z, r,_,_,gaussian_fit, _, _, _, _ = z_finding(corr, lag, pkfrac = 0.7)
    err_dispersion = (3/8)*2*np.sqrt(2*np.log(2))*gaussian_fit.stddev.value/(1+r)
    
    # save the template data
    df_template = pd.DataFrame({'wave':template[0,:], 'flux':template[1,:],
                                'uncertainty':template[2,:]})
    csv_path = os.path.join(path, 'csv','MMT', name+'.csv')
    df_template.to_csv(csv_path, index=False)
    dispersion_csv.write(name+',{},{}\n'.format(gaussian_fit.stddev.value, err_dispersion))
    
dispersion_csv.close()