import sys
sys.path.insert(0,'../')
import numpy as np
import copy
import pandas as pd
import os
from RVSNUpy.rvm import resampler, process_spectrum, rvm

def syn_abstemplates(type="air"):
    syn_abstemp = {}
    for i, age in enumerate([3, 5, 7, 9, 11]):
        temp_file = pd.read_csv(os.path.join(os.environ['rvsnupy'], 'template_files', 'csv', 'fsps', f'fsps_absspec_{type}.txt'), delimiter=' ')
        wavelength, fluxes = np.array(temp_file['wavelengths']), np.array(temp_file[f'{age:d}'])
        error, mask = np.ones_like(fluxes), np.ones_like(fluxes)
        temp = np.vstack([wavelength, fluxes, error, mask])
        if type=="air":
            new_wavelengths = np.logspace(np.log10(temp[0,0]), np.log10(temp[0,-1]), len(temp[0]))
            new_fluxes, new_error = resampler(temp[0], temp[1], new_wavelengths), resampler(temp[0], temp[2], new_wavelengths)
            new_temp = np.vstack([new_wavelengths, new_fluxes, new_error, np.ones_like(new_wavelengths)])
        elif type=="vacuum":
            new_temp = temp
        syn_abstemp[f'{age:d}Gyr'] = [new_temp, 1]
    return syn_abstemp

def syn_emtemplates(type="air"):
    syn_emtemp = {}
    for i, age in enumerate([0.01]):
        temp_file = pd.read_csv(os.path.join(os.environ['rvsnupy'], 'template_files', 'csv', 'fsps', f'fsps_emspec_{type}.txt'), delimiter=' ')
        wavelength, fluxes = np.array(temp_file['wavelengths']), np.array(temp_file[f'{age:.2f}'])
        error, mask = np.ones_like(fluxes), np.ones_like(fluxes)
        temp = np.vstack([wavelength, fluxes, error, mask])
        if type=="air":
            new_wavelengths = np.logspace(np.log10(temp[0,0]), np.log10(temp[0,-1]), len(temp[0]))
            new_fluxes, new_error = resampler(temp[0], temp[1], new_wavelengths), resampler(temp[0], temp[2], new_wavelengths)
            new_temp = np.vstack([new_wavelengths, new_fluxes, new_error, np.ones_like(new_wavelengths)])
        elif type=="vacuum":
            new_temp = temp
        syn_emtemp[f'{age:.2f}Gyr'] = [new_temp, 2]
    return syn_emtemp

def syn_templates(type="air"):
    return {**syn_abstemplates(type), **syn_emtemplates(type)}

##########################################################################################
# sdss template
##########################################################################################

folder_name = os.path.join(os.environ['rvsnupy'], 'template_files','csv')

def sdss_galaxy_templates(type="air"):

    sdss_galaxy_temps = {}
    abs_templates = ['Early_type_galaxy', 'Luminous_red_galaxy']
    em_templates = ['Late_type_galaxy']

    for temp_name in abs_templates:
        path = os.path.join(folder_name, 'sdss', temp_name+f'_{type}.csv')
        template_data = pd.read_csv(path)
        wavelength = np.array(template_data['wave'])
        flux =  np.array(template_data['flux'])
        uncertainty = np.array(template_data['uncertainty'])
        
        template = np.vstack([wavelength, flux, uncertainty, np.ones(len(flux))])
        
    #     linear_wavelength = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
    #     template = resampler(_template, linear_wavelength)
        
        
        sdss_galaxy_temps[temp_name] = [template, 1]
        
    for temp_name in em_templates:
        path = os.path.join(folder_name, 'sdss', temp_name+f'_{type}.csv')
        template_data = pd.read_csv(path)
        wavelength = np.array(template_data['wave'])
        flux =  np.array(template_data['flux'])
        uncertainty = np.array(template_data['uncertainty'])
        
        template = np.vstack([wavelength, flux, uncertainty, np.ones(len(flux))])
        
    #     linear_wavelength = np.linspace(wavelength[0], wavelength[-1], len(wavelength))
    #     template = resampler(_template, linear_wavelength)
        
        sdss_galaxy_temps[temp_name] = [template, 2]
    return sdss_galaxy_temps

# ##########################################################################################
# # calibate templates
# ##########################################################################################

def zp_calib(templates, type="air"):
    abs_templates = {}
    em_templates = {}
    for name in templates.keys():
        if templates[name][1] == 1:
            abs_templates[name] = copy.deepcopy(templates[name])
        elif templates[name][1] == 2:
            em_templates[name] = copy.deepcopy(templates[name])

    zp_templates = {}
    abssyn = rvm(syn_abstemplates(type), z_range=[-0.001,0.001])
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
        zp_templates[name] = [new_temp, templates[name][1]]
        
    for name in em_templates.keys():
        zp_templates[name] = [templates[name][0], templates[name][1]]
    return zp_templates