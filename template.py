import sys
sys.path.insert(0,'../')
import numpy as np
import copy
import pandas as pd
import os
from RVSNUpy.rvm import resampler, process_spectrum, rvm

# ##########################################################################################
# # model template
# ##########################################################################################

# def read_model(age):
#     age_col = {'1':1, '3':2, '5':3, '7':4, '9':5, '11':6} # column corresponding to the age
#     wavelength, flux = np.loadtxt(os.path.join(os.path.dirname(__file__), 'model_template_hr.csv'), usecols=[0,age_col['%d'%age]], unpack=True, skiprows=6)
#     error, mask = copy.deepcopy(flux)/5, np.ones_like(flux)
#     return np.vstack([wavelength, flux, error, mask])

# model_templates = {}
# for age in [3, 5, 7, 9, 11]:
#     modeltemp = read_model(age)
#     new_wavelengths = np.logspace(np.log10(modeltemp[0,0]), np.log10(modeltemp[0,-1]), len(modeltemp[0]))
#     new_fluxes, new_error = resampler(modeltemp[0], modeltemp[1], new_wavelengths), resampler(modeltemp[0], modeltemp[2], new_wavelengths)
#     new_modeltemp = np.vstack([new_wavelengths, new_fluxes, new_error, np.ones_like(new_wavelengths)])
#     model_templates[f'{age:d}yr'] = [new_modeltemp, 0.7, 1, [9000,100]]

##########################################################################################
# synthetic template
##########################################################################################

syn_abstemplates = {}
for i, age in enumerate([3, 5, 7, 9, 11]):
    temp_file = pd.read_csv(os.path.join(os.path.dirname(__file__), 'fsps_absspec.txt'), delimiter=' ')
    wavelength, fluxes = np.array(temp_file['wavelengths']), np.array(temp_file[f'{age:d}'])
    error, mask = np.ones_like(fluxes), np.ones_like(fluxes)
    temp = np.vstack([wavelength, fluxes, error, mask])
    new_wavelengths = np.logspace(np.log10(temp[0,0]), np.log10(temp[0,-1]), len(temp[0]))
    new_fluxes, new_error = resampler(temp[0], temp[1], new_wavelengths), resampler(temp[0], temp[2], new_wavelengths)
    new_temp = np.vstack([new_wavelengths, new_fluxes, new_error, np.ones_like(new_wavelengths)])
    syn_abstemplates[f'{age:d}Gyr'] = [new_temp, 1]

syn_emtemplates = {}
for i, age in enumerate([0.01]):
    temp_file = pd.read_csv(os.path.join(os.path.dirname(__file__), 'fsps_emspec.txt'), delimiter=' ')
    wavelength, fluxes = np.array(temp_file['wavelengths']), np.array(temp_file[f'{age:.2f}'])
    error, mask = np.ones_like(fluxes), np.ones_like(fluxes)
    temp = np.vstack([wavelength, fluxes, error, mask])
    new_wavelengths = np.logspace(np.log10(temp[0,0]), np.log10(temp[0,-1]), len(temp[0]))
    new_fluxes, new_error = resampler(temp[0], temp[1], new_wavelengths), resampler(temp[0], temp[2], new_wavelengths)
    new_temp = np.vstack([new_wavelengths, new_fluxes, new_error, np.ones_like(new_wavelengths)])
    syn_emtemplates[f'{age:.2f}Gyr'] = [new_temp, 2]

syn_templates = {**copy.deepcopy(syn_abstemplates), **copy.deepcopy(syn_emtemplates)}

##########################################################################################
# sdss template
##########################################################################################

folder_name = os.path.join(os.path.dirname(__file__), 'template_files','csv')
sdss_galaxy_templates = {}
abs_templates = ['Early_type_galaxy.csv', 'Luminous_red_galaxy.csv']
em_templates = ['Late_type_galaxy.csv']

for temp_file in abs_templates:
    path = os.path.join(folder_name, 'sdss', temp_file)
    template_data = pd.read_csv(path)
    wavelength = np.array(template_data['wave'])
    flux =  np.array(template_data['flux'])
    uncertainty = np.array(template_data['uncertainty'])
    template_name = os.path.basename(path).replace('.csv','')
    
    template = np.vstack([wavelength, flux, uncertainty, np.ones(len(flux))])
    
#     linear_wavelength = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
#     template = resampler(_template, linear_wavelength)
    
    
    sdss_galaxy_templates[template_name] = [template, 1]
    
for temp_file in em_templates:
    path = os.path.join(folder_name, 'sdss', temp_file)
    template_data = pd.read_csv(path)
    wavelength = np.array(template_data['wave'])
    flux =  np.array(template_data['flux'])
    uncertainty = np.array(template_data['uncertainty'])
    template_name = os.path.basename(path).replace('.csv','')
    
    template = np.vstack([wavelength, flux, uncertainty, np.ones(len(flux))])
    
#     linear_wavelength = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
#     template = resampler(_template, linear_wavelength)
    
    sdss_galaxy_templates[template_name] = [template, 2]

# ##########################################################################################
# # calibate templates
# ##########################################################################################

def zp_calib(templates):
    abs_templates = {}
    em_templates = {}
    for name in templates.keys():
        if templates[name][1] == 1:
            abs_templates[name] = copy.deepcopy(templates[name])
        elif templates[name][1] == 2:
            em_templates[name] = copy.deepcopy(templates[name])

    zp_templates = {}
    abssyn = rvm(syn_abstemplates, z_range=[-0.001,0.001])
    for name in abs_templates.keys():
        temp0 = templates[name][0]
        calib = abssyn.z_single(temp0, chi_thres=0, line_fit=False)
        dz = np.sum(calib['z']/calib['zerr']**2)/np.sum(1/calib['zerr']**2)
        # calib = abssyn.z_single(temp0, chi_thres=0, line_fit=False, output='best')
        # dz = calib[1]
        zero_wavelengths = temp0[0]/(1+dz)
        
        new_wavelengths = np.logspace(np.log10(zero_wavelengths[0]), np.log10(zero_wavelengths[-1]), len(zero_wavelengths))
        new_fluxes, new_error = resampler(zero_wavelengths, temp0[1], new_wavelengths), resampler(zero_wavelengths, temp0[2], new_wavelengths)
        new_temp = np.vstack([new_wavelengths, new_fluxes, new_error, np.ones_like(new_wavelengths)])
        zp_templates[name] = [new_temp, sdss_galaxy_templates[name][1]]
        
    for name in em_templates.keys():
        zp_templates[name] = [templates[name][0], templates[name][1]]
    return zp_templates

# ##########################################################################################
# # sdss template
# ##########################################################################################
# folder_name = os.path.join(os.path.dirname(__file__), 'template_files','csv')
# sdss_template_star = {}
# sdss_star = glob.glob(os.path.join(folder_name, 'sdss','star','*.csv'))

# for path in sdss_star:
#     template_data = pd.read_csv(path)
#     wavelength = np.array(template_data['wave'])
#     flux =  np.array(template_data['flux'])
#     uncertainty = np.array(template_data['uncertainty'])
#     template_name = os.path.basename(path).replace('.csv','')
    
#     template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(flux))])
    
# #     linear_wavelength = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
# #     template = resampler(_template, linear_wavelength)
    
#     pkfrac = 0.5
#     sdss_template_star[template_name] = [template, pkfrac, 1]
    
# sdss_template_galaxy = {}
# sdss_galaxy = glob.glob(os.path.join(folder_name, 'sdss', 'galaxy', '*.csv'))
# for path in sdss_galaxy:
#     template_data = pd.read_csv(path)
#     wavelength = np.array(template_data['wave'])
#     flux =  np.array(template_data['flux'])
#     uncertainty = np.array(template_data['uncertainty'])
#     template_name = os.path.basename(path).replace('.csv','')
    
#     template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(flux))])
    
# #     linear_wavelength = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
# #     template = resampler(_template, linear_wavelength)
                        
#     pkfrac = 0.7
#     if os.path.basename(path) in ['Early_type_galaxy.csv', 'Luminous_red_galaxy.csv']:
#         template_type = 1
#     else:
#         template_type = 2
#     sdss_template_galaxy[template_name] = [template, template_type]
    
# sdss_template_QSO = {}
# sdss_QSO = glob.glob(os.path.join(folder_name, 'sdss','QSO','*.csv'))
# for path in sdss_QSO:
#     template_data = pd.read_csv(path)
#     wavelength = np.array(template_data['wave'])
#     flux =  np.array(template_data['flux'])
#     uncertainty = np.array(template_data['uncertainty'])
#     template_name = os.path.basename(path).replace('.csv','')
    
#     template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(flux))])
    
# #     linear_wavelength = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
# #     template = resampler(_template, linear_wavelength)
                        
#     pkfrac = 0.7
#     sdss_template_QSO[template_name] = [template, 2]

# sdss_template_QSO['High_luminosity_QSO'][1] = 0.9

# sdss_template = copy.deepcopy(sdss_template_star)
# sdss_template.update(sdss_template_galaxy)
# sdss_template.update(sdss_template_QSO)

# sdss_temp_wt_qso = copy.deepcopy(sdss_template_star)
# sdss_temp_wt_qso.update(sdss_template_galaxy)

# ##########################################################################################

# ##########################################################################################
# # MMT template
# ##########################################################################################
# mmt_template = {}
# mmt_template_star = {}
# mmt_template_galaxy = {}

# MMT = glob.glob(os.path.join(folder_name,'MMT','*.csv'))
# for path in MMT:
#     template_data = pd.read_csv(path)
#     wavelength = np.array(template_data['wave'])[5:-5]
#     flux =  np.array(template_data['flux'])[5:-5]
#     uncertainty = np.array(template_data['uncertainty'])[5:-5]
#     template_name = os.path.basename(path).replace('.csv','')
    
#     if os.path.basename(path) in ['sptemp.csv', 'hemtemp0.0.csv']:
#         template_type = 2
#         pkfrac = 0.65
#     else:
#         template_type = 1
#         pkfrac = 0.65
        
#     template = np.vstack([wavelength, flux, uncertainty, np.zeros(len(flux))])
#     mmt_template[template_name] = [template, template_type]
    
#     if template_name in ['m31_a_temp', 'm31_f_temp', 'm31_k_temp']:
#         mmt_template_star[template_name] = [template, template_type]
#     elif template_name in ['eatemp', 'eltemp', 'habtemp90', 'hemtemp0.0', 'sptemp']:
#         mmt_template_galaxy[template_name] = [template, template_type]
    
    
# ##########################################################################################
# # RVSNUpy_template
# ##########################################################################################
    
# RVSNUpy_template = copy.deepcopy(sdss_template)
# RVSNUpy_template.update(mmt_template)
