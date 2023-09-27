import sys
sys.path.insert(0,'../')
import glob
import numpy as np
import copy
import pandas as pd
import os
from RVSNUpy.correlation import resampler

##########################################################################################
# sdss template
##########################################################################################
folder_name = os.path.join(os.path.dirname(__file__), 'template_files','csv')
sdss_template_star = {}
sdss_star = glob.glob(os.path.join(folder_name, 'sdss','star','*.csv'))
Template_names, temp_dispersion, temp_dispersion_err  = np.loadtxt(os.path.join(folder_name, 'dispersion', 'sdss.csv'),
                                                                   unpack=True, delimiter=',', dtype=str)
for path in sdss_star:
    template_data = pd.read_csv(path)
    wavelength = np.array(template_data['wave'])
    flux =  np.array(template_data['flux'])*10**-17
    uncertainty = np.array(template_data['uncertainty'])*10**-17
    template_name = os.path.basename(path).replace('.csv','')
    
    template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(flux))])
    
#     linear_wavelength = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
#     template = resampler(_template, linear_wavelength)
    
    pkfrac = 0.5
    sdss_template_star[template_name] = [template, pkfrac, 'absorption',
                                        [float(temp_dispersion[Template_names==template_name]),
                                        float(temp_dispersion_err[Template_names==template_name])]]
    
sdss_template_galaxy = {}
sdss_galaxy = glob.glob(os.path.join(folder_name, 'sdss', 'galaxy', '*.csv'))
for path in sdss_galaxy:
    template_data = pd.read_csv(path)
    wavelength = np.array(template_data['wave'])
    flux =  np.array(template_data['flux'])*10**-17
    uncertainty = np.array(template_data['uncertainty'])*10**-17
    template_name = os.path.basename(path).replace('.csv','')
    
    template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(flux))])
    
#     linear_wavelength = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
#     template = resampler(_template, linear_wavelength)
                        
    pkfrac = 0.7
    if os.path.basename(path) in ['Early-type galaxy.csv', 'Luminous Red Galaxy.csv']:
        template_type = 'absoprtion'
    else:
        template_type = 'emission'
    sdss_template_galaxy[template_name] = [template, pkfrac, template_type,
                                          [float(temp_dispersion[Template_names==template_name]),
                                           float(temp_dispersion_err[Template_names==template_name])]]
    
sdss_template_QSO = {}
sdss_QSO = glob.glob(os.path.join(folder_name, 'sdss','QSO','*.csv'))
for path in sdss_QSO:
    template_data = pd.read_csv(path)
    wavelength = np.array(template_data['wave'])
    flux =  np.array(template_data['flux'])*10**-17
    uncertainty = np.array(template_data['uncertainty'])*10**-17
    template_name = os.path.basename(path).replace('.csv','')
    
    template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(flux))])
    
#     linear_wavelength = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
#     template = resampler(_template, linear_wavelength)
                        
    pkfrac = 0.7
    sdss_template_QSO[template_name] = [template, pkfrac, 'emission',
                                       [float(temp_dispersion[Template_names==template_name]),
                                        float(temp_dispersion_err[Template_names==template_name])]]

sdss_template_QSO['High-luminosity QSO'][1] = 0.9

sdss_template = copy.deepcopy(sdss_template_star)
sdss_template.update(sdss_template_galaxy)
sdss_template.update(sdss_template_QSO)

sdss_temp_wt_qso = copy.deepcopy(sdss_template_star)
sdss_temp_wt_qso.update(sdss_template_galaxy)

##########################################################################################

##########################################################################################
# MMT template
##########################################################################################
mmt_template = {}
mmt_template_star = {}
mmt_template_galaxy = {}

MMT = glob.glob(os.path.join(folder_name,'MMT','*.csv'))
Template_names, temp_dispersion, temp_dispersion_err = np.loadtxt(os.path.join(folder_name, 'dispersion', 'MMT.csv'), unpack=True,
                                                delimiter=',', dtype=str)
for path in MMT:
    template_data = pd.read_csv(path)
    wavelength = np.array(template_data['wave'])[5:-5]
    flux =  np.array(template_data['flux'])[5:-5]
    uncertainty = np.array(template_data['uncertainty'])[5:-5]
    template_name = os.path.basename(path).replace('.csv','')
    
    if os.path.basename(path) in ['sptemp.csv', 'hemtemp0.0.csv']:
        template_type = 'emission'
        pkfrac = 0.65
    else:
        template_type = 'absorption'
        pkfrac = 0.65
        
    template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(flux))])
    mmt_template[template_name] = [template, pkfrac, template_type,
                                  [float(temp_dispersion[Template_names==template_name]),
                                   float(temp_dispersion_err[Template_names==template_name])]]
    
    if template_name in ['m31_a_temp', 'm31_f_temp', 'm31_k_temp']:
        mmt_template_star[template_name] = [template, pkfrac, template_type,
                                  [float(temp_dispersion[Template_names==template_name]),
                                   float(temp_dispersion_err[Template_names==template_name])]]
    elif template_name in ['eatemp', 'eltemp', 'habtemp90', 'hemtemp0.0', 'sptemp']:
        mmt_template_galaxy[template_name] = [template, pkfrac, template_type,
                                  [float(temp_dispersion[Template_names==template_name]),
                                   float(temp_dispersion_err[Template_names==template_name])]]
    
    
##########################################################################################
# RVSNUpy_template
##########################################################################################
    
RVSNUpy_template = copy.deepcopy(sdss_template)
RVSNUpy_template.update(mmt_template)
