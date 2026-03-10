import sys
import os
import glob
import argparse
import numpy as np
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{file_dir}')
asys_path = os.path.join(os.path.dirname(file_dir), "Analyze_HAMR_Data")
sys.path.insert(0, f'{asys_path}')
import compute_variables as cv
import hamr_read as hr
import theory as theory

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import cmasher as cmr 
import matplotlib.style as style
matplotlib.rcParams['text.usetex'] = True
style.use('seaborn-v0_8-bright')
matplotlib.rcParams['figure.dpi'] = 300
import figs

MBH     = 10.
figsize = [42.0, 24.0]
FS      = 40
plt.rcParams['xtick.labelsize'] = FS
plt.rcParams['ytick.labelsize'] = FS

opt = dict(ha='left', va='center', color='white',
           bbox=dict(facecolor='black', edgecolor='none', alpha=0.4))
def main(**kwargs):
    sim            = kwargs.get('sim')
    file_number    = int(kwargs.get('file_number'))
    start_file     = int(kwargs.get('start_file'))
    end_file       = int(kwargs.get('end_file'))
    time_average   = kwargs.get('time_average')
    reduced        = kwargs.get('reduced')
    radiation      = kwargs.get('radiation')
    two_temp       = kwargs.get('two_temp')
    r_max          = kwargs.get('r_max')
    plot_title     = kwargs.get('plot_title')
    plot_vec       = kwargs.get('plot_vec')

    cgs = True if (two_temp or radiation) else False
    Analytic = theory.Analytic(hr.CONSTANTS, cgs=cgs)

    if (two_temp or radiation):
        Msun = hr.CONSTANTS['Msun']
        c    = hr.CONSTANTS['c']
        G    = hr.CONSTANTS['G']
        Rg   = G * MBH * Msun / (c * c)
    else:
        c, G, Rg = 1., 1., 1.

    if (two_temp or radiation):
        kB   = hr.CONSTANTS['kB']
        mp   = hr.CONSTANTS['mp'] 

        x_AB = hr.CONSTANTS['X_AB']
        y_AB = hr.CONSTANTS['Y_AB']

        # Equation 11 in Sadowski+ (2017)
        mu_i = 4.0 / (4.0 * x_AB + y_AB)
        # Equation 10 in Sadowski+ (2017)
        mu_e = 2.0 / (1.0 + x_AB)
        mi   = (x_AB + y_AB) * mp  
        me   = hr.CONSTANTS['me'] 

        # Given in Kaaz et al. (2024)
        mu_g = 4.0 / (6.0 * x_AB + y_AB + 2.)

    # Determine slice directory
    averages_dir = cv.user_path + f'/Averages/{sim}'
    save_dir = cv.user_path +  '/PDF'
  
    keys = [
        'qdiss',         'qdiss_ion',         'qie',         'qrad',     
        'qdiss_ie',      'qdiss_ion_ie',      'qie_ie',      'qrad_ie',
        'qdiss_thin_ie', 'qdiss_ion_thin_ie', 'qie_thin_ie', 'qrad_thin_ie'
    ]

    norms     = dict.fromkeys(keys)
    vols      = dict.fromkeys(keys)
    logs      = dict.fromkeys(keys)
    linthresh = dict.fromkeys(keys) 
    extend    = dict.fromkeys(keys)

    for key in keys: 
        vols[     f'{key}'] = get_vols(key)
        norms[    f'{key}'] = get_norms(key, cgs=cgs, MBH=MBH, Rg=Rg)
        logs[     f'{key}'] = get_logs(key)
        linthresh[f'{key}'] = get_linthresh(key)
        extend[   f'{key}'] = get_extend(key)

    limits = dict.fromkeys(keys)
    for key in keys: 
        limits[f'{key}'] = [1.e-8, 1.]

    color_maps = dict.fromkeys(keys)
    for key in keys: 
        color_maps[f'{key}'] = cmr.torch
    
        #if ('qdiss' in key or 'qrad' in key):
        #    limits[f'{key}']     = [-1., 1.]
        #    color_maps[f'{key}'] = cmr.prinsenvlag_r
    
    plot_labels = {
        'qdiss'             : r'$\mathcal{Q}_{\rm diss} R_g^3 / L_{\rm Edd}$',
        'qdiss_ion'         : r'$\left(1 - \delta_e \right) \mathcal{Q}_{\rm diss} R_g^3 / L_{\rm Edd}$',
        'qie'               : r'$\mathcal{Q}_{\rm Coul} R_g^3 / L_{\rm Edd}$',
        'qrad'              : r'$\mathcal{Q}_{\rm rad} R_g^3 / L_{\rm Edd}$',
        'qdiss_ie'          : r'$\mathcal{Q}^{ie}_{\rm diss} R_g^3 / L_{\rm Edd}$',
        'qdiss_ion_ie'      : r'$\left(1 - \delta_e \right) \mathcal{Q}^{ie}_{\rm diss} R_g^3 / L_{\rm Edd}$',
        'qie_ie'            : r'$\mathcal{Q}^{ie}_{\rm Coul} R_g^3 / L_{\rm Edd}$',
        'qrad_ie'           : r'$\mathcal{Q}^{ie}_{\rm rad} R_g^3 / L_{\rm Edd}$',
        'qdiss_thin_ie'     : r'$\mathcal{Q}^{ie}_{\rm diss} R_g^3 / L_{\rm Edd}$',
        'qdiss_ion_thin_ie' : r'$\left(1 - \delta_e \right) \mathcal{Q}^{ie}_{\rm diss} R_g^3 / L_{\rm Edd}$',
        'qie_thin_ie'       : r'$\mathcal{Q}^{ie}_{\rm Coul} R_g^3 / L_{\rm Edd}$',
        'qrad_thin_ie'      : r'$\mathcal{Q}^{ie}_{\rm rad} R_g^3 / L_{\rm Edd}$',
    }
    
    label_colors = dict.fromkeys(keys)
    for key in keys:
        label_colors[f'{key}'] = 'white'
   
    axes_ids = {
        'qdiss'             : 0,
        'qdiss_ion'         : 1, 
        'qie'               : 2, 
        'qrad'              : 3,
        'qdiss_ie'          : 4,
        'qdiss_ion_ie'      : 5,
        'qie_ie'            : 6,
        'qrad_ie'           : 7,
        'qdiss_thin_ie'     : 8,
        'qdiss_ion_thin_ie' : 9,
        'qie_thin_ie'       : 10,
        'qrad_thin_ie'      : 11
    }
    
    file_info = {
        'averages_dir' : averages_dir,
        'file_number'  : file_number,
        'start_file'   : start_file,
        'end_file'     : end_file,
        'reduced'      : reduced
    }

    Slice  = cv.Heating_Slices()
    fnums  = Slice.get_fnums(**file_info)
    single_file  = True if (len(fnums) == 1) else False
    time_average = True if (len(fnums) != 1) else False
    if (single_file):
        pkeys  = keys + ['Vol', 'r', 'theta', 'a']
        slices = Slice.load_slices(averages_dir, fnums, keys=pkeys, reduced=reduced)
    elif (time_average):
        pkeys = keys
        slices = Slice.load_slices_time_averaged(averages_dir, fnums, 
            keys=pkeys, vols=vols, reduced=reduced)
        
    Cart = figs.Cartesian([])
    fig  = plt.figure(figsize = figsize)
    gs   = gridspec.GridSpec(3, 4, figure = fig)
    axes = []
    for i in range(3):
        for j in range(4):
            axes.append(fig.add_subplot(gs[i, j]))

    for key in keys:
        id        = int(axes_ids[f'{key}'])
        vol       = vols[f'{key}']
        norm      = norms[f'{key}']
        lims      = limits[f'{key}']
        log       = logs[f'{key}']
        cmap      = color_maps[f'{key}']
        if (single_file):
            r     = slices[f'{fnums[0]}']['r'][None, :, None, None]
            theta = slices[f'{fnums[0]}']['theta'][None, None, :, None]
            R     = {
                'r' :  (r * np.sin(theta))[0, :, :, 0],
                'l' : -(r * np.sin(theta))[0, :, :, 0],
            }
            Z = {
                'r' : (r * np.cos(theta))[0, :, :, 0],
                'l' : (r * np.cos(theta))[0, :, :, 0],
            }
            Volume = slices[f'{fnums[0]}']['Vol'] if vol else 1.
            Q  = {
                'r' : norm * slices[f'{fnums[0]}'][f'{key}'] / Volume
            }        
        elif (time_average):
            r     = slices['r'][None, :, None, None]
            theta = slices['theta'][None, None, :, None]
            R     = {
                'r' :  (r * np.sin(theta))[0, :, :, 0],
                'l' : -(r * np.sin(theta))[0, :, :, 0],
            }
            Z = {
                'r' : (r * np.cos(theta))[0, :, :, 0],
                'l' : (r * np.cos(theta))[0, :, :, 0],
            }
            Q  = {
                'r' : norm * slices[f'{key}']
            }

        plt_args = {
            'lim' : lims,
            'log' : log,
            'sym' : False if not linthresh[f'{key}'] else linthresh[f'{key}'],
            'linthresh' : linthresh[f'{key}'],
            'extend'    : extend[f'{key}']
        }

        if id >= 0:
            Cart.plot_Rz(fig, axes[id], R, Z, Q, cmap, FS, **plt_args) 
            opt['color'] = label_colors[f'{key}'] 
            axes[id].text(0.05, 0.90, plot_labels[f'{key}'], 
                          fontsize=FS+4, transform=axes[id].transAxes, **opt)
            if 'thin' in key:
                tau_label = r'$\tau_{\rm es} < 1$'
                axes[id].text(0.05, 0.10, tau_label, 
                              fontsize=FS+4, transform=axes[id].transAxes, **opt)

    for ax in axes:
        if (r_max != 0.):
            ax.set_xlim([0., r_max]) 
            ax.set_ylim([-r_max/2., r_max/2.])   
        ax.set_xlabel(r'$R \: (R_g)$', fontsize=FS)
        ax.set_ylabel(r'$z \: (R_g)$', fontsize=FS)
        ax.set_aspect('equal')

    print("Saving figure ...", flush=True)
    fig.tight_layout()
    plt.savefig(f'{save_dir}/{plot_title}.png', format='png')
    plt.close()

def get_vols(key):
    if 'Mdot' in key or 'Edot' in key:
        return False
    elif (key == 'Phi_BH' or key == 'Phi_mid' or 
          key == 'Phi_sph_upp' or key == 'Phi_sph_low'):
        return False
    else:
        return True

def get_norms(key, **kwargs):
    cgs = kwargs.get('cgs', False)
    if (cgs):
        if 'MBH' not in kwargs:
            raise ValueError("Must supply MBH for cgs=True")
        MBH      = kwargs.get('MBH', 1.0)
        Analytic = theory.Analytic(hr.CONSTANTS, cgs=cgs)
        L_Edd    = Analytic.calc_LEdd(MBH=MBH)
        Rg = kwargs.get('Rg', 1.)
        return (Rg * Rg * Rg / L_Edd)
    else:
        return 1.

def get_logs(key):
    if (key == 'vR' or key == 'vz' or key == 'bphi' or key == 'bz'):
        return False
    else:
        return True

def get_linthresh(key):
    if key == 'vR':
        return 1.e-3
    elif (key == 'bphi' or key == 'bz'):
        return 1.e7
    elif (key == 'Be' or key == 'Be_Rad'):
        return 1.e-2
    elif 'qrad' in key or 'qdiss' in key:
        return False #1.e-12
    else:
        return False
    
def get_extend(key):
    if key == 'Pmag':
        return 'min'
    else:
        return 'both'
    
if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Plot profiles.""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('--file_number', type=int, required=False, 
                        default=-1, help = """dump number of profile plot""")
    parser.add_argument('--start_file', type=int, required=False,
                        default=-1, help = """dump number
                        of start file in profile average/evolution""")
    parser.add_argument('--end_file', type=int, required=False,
                        default=-1, help = """dump number
                        of end file in profile average/evolution""")
    parser.add_argument('--time_average', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""boolean to determine if profiles from
                        --start_file to --end_file should be averaged.""")
    parser.add_argument('--r_max', type=float, default=0.,
                        required=False,
                        help = """radial extent in Rg (default: R_max)""")
    parser.add_argument('--plot_vec', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if vector streamlines 
                        should be plotted.""")   
    parser.add_argument('--radiation', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if radiation quantities
                        should be included in profile extraction.""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if two-temperature 
                        physics should be included in slice extraction.""")   
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if profile extraction
                        should be done on reduced data.""")
    parser.add_argument('--plot_title', type=str, required=False, 
                        default='Slices', 
                        help="""title of PDF output""")
    args = parser.parse_args()
    main(**vars(args))

'''
            if vec != None:
                npts = 100
                R1D = np.linspace(0., r_max, npts)
                z1D = np.linspace(-r_max/2, r_max/2, npts)
                R_uni, Z_uni = np.meshgrid(R1D, z1D)

            if vec == 'Mass_Flux':
                # Enthalpy-weighted velocity
                Volume = slices[f'{fnums[0]}']['Vol']
                rho    = slices[f'{fnums[0]}']['rho'] / Volume
                Pmag   = slices[f'{fnums[0]}']['Pmag'] / Volume
                Pthm   = slices[f'{fnums[0]}']['Pthm'] / Volume
                Prad   = slices[f'{fnums[0]}']['Prad'] / Volume
                Eion   = slices[f'{fnums[0]}']['Eion'] / Volume
                Eelc   = slices[f'{fnums[0]}']['Eelc'] / Volume
                w      = rho*c*c + Pthm + Eion + Eelc + 2.*Pmag + 4.*Prad
                wuR    = slices[f'{fnums[0]}']['wuR'] / Volume
                wuz    = slices[f'{fnums[0]}']['wuz'] / Volume
                uR     = wuR / w
                uz     = wuz / w
                Vec_R  = {'r' : uR}
                Vec_Z  = {'r' : uz} 
            elif vec == 'Magnetic_Field':
                Volume = slices[f'{fnums[0]}']['Vol']
                bR     = slices[f'{fnums[0]}']['bR'] / Volume
                bz     = slices[f'{fnums[0]}']['bz'] / Volume
                Vec_R  = {'r' : bR}
                Vec_Z  = {'r' : bz} 
            elif vec == 'Total_Energy_Flux':
                Volume = slices[f'{fnums[0]}']['Vol']
                THD_tR = slices[f'{fnums[0]}']['THD_tR'] / Volume
                THD_tz = slices[f'{fnums[0]}']['THD_tz'] / Volume
                TEM_tR = slices[f'{fnums[0]}']['TEM_tR'] / Volume
                TEM_tz = slices[f'{fnums[0]}']['TEM_tz'] / Volume
                Vec_R  = {'r' : -(THD_tR + TEM_tR)}
                Vec_Z  = {'r' : -(THD_tz + TEM_tz)}
            elif vec == 'Radiation_Flux':
                Volume = slices[f'{fnums[0]}']['Vol']
                R_tR   = slices[f'{fnums[0]}']['R_tR'] / Volume
                R_tz   = slices[f'{fnums[0]}']['R_tz'] / Volume
                Vec_R  = {'r' : -R_tR}
                Vec_Z  = {'r' : -R_tz}

            if vec != None:
                npts = 100
                R1D = np.linspace(0., r_max, npts)
                z1D = np.linspace(-r_max/2, r_max/2, npts)
                R_uni, Z_uni = np.meshgrid(R1D, z1D)

            if vec == 'Mass_Flux':
                # Enthalpy-weighted velocity
                rho    = slices['rho']
                Pmag   = slices['Pmag']
                Pthm   = slices['Pthm']
                Prad   = slices['Prad']
                Eion   = slices['Eion']
                Eelc   = slices['Eelc']
                w      = rho*c*c + Pthm + Eion + Eelc + 2.*Pmag + 4.*Prad
                wuR    = slices['wuR']
                wuz    = slices['wuz']
                uR     = wuR / w
                uz     = wuz / w
                Vec_R  = {'r' : uR}
                Vec_Z  = {'r' : uz}
            elif vec == 'Magnetic_Field':
                bR     = slices['bR']
                bz     = slices['bz']
                Vec_R  = {'r' : bR}
                Vec_Z  = {'r' : bz} 
            elif vec == 'Total_Energy_Flux':
                THD_tR = slices['THD_tR']
                THD_tz = slices['THD_tz']
                TEM_tR = slices['TEM_tR']
                TEM_tz = slices['TEM_tz']
                Vec_R  = {'r' : -(THD_tR + TEM_tR)}
                Vec_Z  = {'r' : -(THD_tz + TEM_tz)}
            elif vec == 'Radiation_Flux':
                R_tR  = slices['R_tR']
                R_tz  = slices['R_tz']
                Vec_R = {'r' : -R_tR}
                Vec_Z = {'r' : -R_tz}                
                
        vec_args = {
            'dens'               : 6.0,
            'linewidth'          : 0.6,
            'color'              : 'black',
            'broken_streamlines' : True,
            'arrowsize'          : 1.
        }

            if vec != None:
                Cart.plot_Rz_vec(fig, axes[id], R, Z, R_uni, Z_uni, 
                                 Vec_R, Vec_Z, **vec_args)

  

        if key == 'Eion':
            plt_args['lim']    = [1., 1.e3]
            plt_args['extend'] = 'max'
            Cart.plot_Rz(fig, axes[3], R, Z, Ti_Te, cmr.torch, FS, **plt_args)
            opt['color'] = 'white'
            axes[3].text(0.05, 0.90, r'$T_i/T_e$', fontsize=FS+4, 
                         transform=axes[3].transAxes, **opt)
        if key == 'qie':
            plt_args['lim'] = [1.e-2, 1.e2]
            Cart.plot_Rz(fig, axes[10], R, Z, qie_qIC, cmr.prinsenvlag_r, FS, **plt_args)
            opt['color'] = 'black'
            axes[10].text(0.05, 0.90, 
                          r'$\mathcal{Q}_{\rm ie} / \mathcal{Q}_{\rm IC}$', 
                          fontsize=FS+4, transform=axes[10].transAxes, **opt)
'''
