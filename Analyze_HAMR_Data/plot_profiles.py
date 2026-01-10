import sys
import os
import glob
import argparse
import numpy as np
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{file_dir}')
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
figsize = [40.0, 26.0]
FS      = 48
plt.rcParams['xtick.labelsize'] = FS
plt.rcParams['ytick.labelsize'] = FS

def main(**kwargs):
    sim            = kwargs.get('sim')
    file_number    = int(kwargs.get('file_number'))
    start_file     = int(kwargs.get('start_file'))
    end_file       = int(kwargs.get('end_file'))
    time_average   = kwargs.get('time_average')
    time_evolution = kwargs.get('time_evolution')
    reduced        = kwargs.get('reduced')
    radiation      = kwargs.get('radiation')
    two_temp       = kwargs.get('two_temp')
    r_max          = kwargs.get('r_max')
    plot_title     = kwargs.get('plot_title')

    cgs = True if (two_temp or radiation) else False
    Analytic = theory.Analytic(hr.CONSTANTS, cgs=cgs)

    if (two_temp or radiation):
        Msun = hr.CONSTANTS['Msun']
        c    = hr.CONSTANTS['c']
        G    = hr.CONSTANTS['G']
        Rg   = G * MBH * Msun / (c * c)
    else:
        c, G, Rg = 1., 1., 1.

    # Determine profile directory
    averages_dir = cv.user_path + f'/Averages/{sim}'
    save_dir = cv.user_path +  '/PDF'
    
    keys = [
        'Mdot_sigma', 'rho', 'Ti', 'Te', 'Pthm', 'Prad', 'Pmag',
        'Edot_jet', 'Edot_Poyt', 'Edot_Fluid', 'Edot_Rad', 
        'Dtheta2', 'Phi_BH', 'Phi_mid', 'Phi_sph_upp', 'Phi_sph_low',
        'vR', 'vphi', 'vz', 'bphi', 'bz'
    ]

    norms     = dict.fromkeys(keys)
    vols      = dict.fromkeys(keys)
    logs      = dict.fromkeys(keys)
    linthresh = dict.fromkeys(keys) 

    for key in keys: 
        vols[     f'{key}'] = get_vols(key)
        norms[    f'{key}'] = get_norms(key, cgs=cgs, MBH=MBH)
        logs[     f'{key}'] = get_logs(key)
        linthresh[f'{key}'] = get_linthresh(key)

    limits = {
        'Mdot_sigma' : [1.e-2, 1.e1],
        'rho'        : [1.e-5, 1.e0],
        'Ti'         : [1.e4, 1.e12],
        'Te'         : [1.e4, 1.e12],
        'Pthm'       : [1.e10, 1.e18],
        'Prad'       : [1.e10, 1.e18],
        'Pmag'       : [1.e10, 1.e18],
        'vR'         : [-1, 0.1],
        'vphi'       : [0.1, 1.1],
        'vz'         : [-0.01, 0.01],
        'bphi'       : [-1.e9, 1.e9],
        'bz'         : [-1.e9, 1.e9],
        'Edot_jet'   : [1.e-2, 2.e2],
        'Edot_Poyt'  : [1.e-2, 2.e2],
        'Edot_Fluid' : [1.e-2, 2.e2],
        'Edot_Rad'   : [1.e-2, 2.e2],
        'Dtheta2'    : [1.e-2, 2.],
        'Phi_BH'     : [1., 1.e3],
        'Phi_mid'    : [1., 1.e3],
        'Phi_sph_upp': [1., 1.e3],
        'Phi_sph_low': [1., 1.e3]
    }

    colors = {
        'Mdot_sigma' : 'black',
        'rho'        : 'black',
        'Ti'         : 'darkorange',
        'Te'         : 'dodgerblue',
        'Pthm'       : 'darkorange',
        'Prad'       : 'crimson',
        'Pmag'       : 'dodgerblue',
        'vR'         : 'black',
        'vphi'       : 'black',
        'vz'         : 'black',
        'bphi'       : 'black',
        'bz'         : 'black',
        'Edot_jet'   : 'darkorange',
        'Edot_Poyt'  : 'dodgerblue',
        'Edot_Fluid' : 'goldenrod',
        'Edot_Rad'   : 'crimson',
        'Dtheta2'    : 'black',
        'Phi_BH'     : 'black',
        'Phi_mid'    : 'darkorange',
        'Phi_sph_upp': 'dodgerblue',
        'Phi_sph_low': 'dodgerblue'
    }

    labels = {
        'Mdot_sigma' : None,
        'rho'        : None,
        'Ti'         : r'$\rm{Ions}$',
        'Te'         : r'$\rm{Electrons}$',
        'Pthm'       : r'$\rm{Thermal}$',
        'Prad'       : r'$\rm{Radiation}$',
        'Pmag'       : r'$\rm{Magnetic}$',
        'vR'         : None,
        'vphi'       : None,
        'vz'         : None,
        'bphi'       : None,
        'bz'         : None,
        'Edot_jet'   : r'$\rm{Jet}$',
        'Edot_Poyt'  : r'$\rm{Poynting}$',
        'Edot_Fluid' : r'$\rm{Enthalpy}$',
        'Edot_Rad'   : r'$\rm{Radiation}$',
        'Dtheta2'    : None,
        'Phi_BH'     : r'$\rm{BH}$',
        'Phi_mid'    : r'$\rm{mid}$',
        'Phi_sph_upp': r'$\rm{sph}$',
        'Phi_sph_low': None
    }

    plot_labels = {
        'Mdot_sigma' : r'$\dot{M}/\dot{M}_{\rm Edd}$',
        'rho'        : r'$\langle \rho \rangle_{\rho} \: (\rm{g/cm}^3)$',
        'Ti'         : r'$\rm{Temperature} \: (\rm{K})$',
        'Te'         : r'$\rm{Temperature} \: (\rm{K})$',
        'Pthm'       : r'$\rm{Pressure} \: (\rm{erg/cm}^3)$',
        'Prad'       : r'$\rm{Pressure} \: (\rm{erg/cm}^3)$',
        'Pmag'       : r'$\rm{Pressure} \: (\rm{erg/cm}^3)$',
        'vR'         : r'$\langle v^R / c \rangle_{\rho}$',
        'vphi'       : r'$\langle \Omega / \Omega_{\rm K} \rangle_{\rho}$',
        'vz'         : r'$\langle v^z / c \rangle_{\rho}$',
        'bphi'       : r'$\langle b^{\varphi} \rangle_{\rho} \: (\rm{G})$',
        'bz'         : r'$\langle b^z \rangle_{\rho} \: (\rm{G})$',
        'Edot_jet'   : r'$\dot{E} \: (L_{\rm Edd})$',
        'Edot_Poyt'  : r'$\dot{E} \: (L_{\rm Edd})$',
        'Edot_Fluid' : r'$\dot{E} \: (L_{\rm Edd})$',
        'Edot_Rad'   : r'$\dot{E} \: (L_{\rm Edd})$',
        'Dtheta2'    : r'$\langle H_{\rho} / R \rangle_{\rho}$',
        'Phi_BH'     : r'$\phi \equiv \Phi_{\rm B} / \sqrt{\langle \dot{M} \rangle_t R_g^2 c}$',
        'Phi_mid'    : r'$\phi \equiv \Phi_{\rm B} / \sqrt{\langle \dot{M} \rangle_t R_g^2 c}$',
        'Phi_sph_upp': r'$\phi \equiv \Phi_{\rm B} / \sqrt{\langle \dot{M} \rangle_t R_g^2 c}$',
        'Phi_sph_low': r'$\phi \equiv \Phi_{\rm B} / \sqrt{\langle \dot{M} \rangle_t R_g^2 c}$'
    }

    axes_ids = {
        'Mdot_sigma' : 0,
        'rho'        : 1,
        'Ti'         : 2,
        'Te'         : 2,
        'Pthm'       : 6,
        'Prad'       : 6,
        'Pmag'       : 6,
        'vR'         : 3,
        'vphi'       : 7,
        'vz'         : 11,
        'bphi'       : 9,
        'bz'         : 10,
        'Edot'       : 4,
        'Edot_jet'   : 4,
        'Edot_Poyt'  : 4,
        'Edot_Fluid' : 4,
        'Edot_Rad'   : 4,
        'Dtheta2'    : 5,
        'Phi_BH'     : 8,
        'Phi_mid'    : 8,
        'Phi_sph_upp': 8,
        'Phi_sph_low': 8
    }

    file_info = {
        'averages_dir' : averages_dir,
        'file_number'  : file_number,
        'start_file'   : start_file,
        'end_file'     : end_file,
        'reduced'      : reduced
    }

    Profs = cv.Profiles()
    fnums = Profs.get_fnums(**file_info)
    pkeys = keys + ['Vol', 'r', 'a']
    profiles = Profs.load_profiles(averages_dir, fnums, keys=pkeys, reduced=reduced)
    single_file = True if (len(fnums) == 1) else False

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
        color     = colors[f'{key}']
        linestyle = '-'
        a         = profiles[f'{fnums[0]}']['a']
        if (single_file):
            r  = profiles[f'{fnums[0]}']['r']
            if ('Phi' in key):
                r5_idx = np.argmin(np.abs(r - 5.))
                Mdot_avg = profiles[f'{fnums[0]}']['Mdot_sigma'][r5_idx]
                norm = 1. / np.sqrt(Mdot_avg * Rg * Rg * c)
                if (key == 'Phi_mid'):
                    norm *= -1
            if (vol):
                y  = norm * profiles[f'{fnums[0]}'][f'{key}'] / profiles[f'{fnums[0]}']['Vol']
            else:
                y  = norm * profiles[f'{fnums[0]}'][f'{key}']
            if (key == 'Dtheta2'): 
                y = np.sqrt(y)
            if (key == 'vphi'):
                vphi = c * profiles[f'{fnums[0]}'][f'{key}'] / profiles[f'{fnums[0]}']['Vol']
                OmegaK = Analytic.calc_OmegaK(MBH, a, r)
                Omega  = vphi / (r * Rg)
                y = Omega / OmegaK

            if ('Mdot' in key  or 'Edot' in key):
                axes[id].plot(r, -y, lw=2., color=color, linestyle='--', alpha=0.5)
            if (key == 'Prad'):
                Trad = (3. * y / hr.CONSTANTS['aRAD']) ** (1./4.)
                axes[axes_ids['Te']].plot(r, Trad, color=colors['Prad'], lw=2., label=r'$\rm{Radiation}$')

            if (key == 'Phi_sph_upp'):
                rH     = Analytic.calc_R_Horizon(a)
                rH_idx = np.argmin(np.abs(r - rH))
                phi_Disk_upp = norm * (profiles[f'{fnums[0]}']['Phi_sph_upp'] - profiles[f'{fnums[0]}']['Phi_BH'][rH_idx])
                axes[id].plot(r, phi_Disk_upp, lw=2., color='crimson', linestyle='-', label=r'$\rm{sph} - \rm{BH}$')
            if (key == 'Phi_sph_low'):
                linestyle = '--'
                rH     = Analytic.calc_R_Horizon(a)
                rH_idx = np.argmin(np.abs(r - rH))
                phi_Disk_low = norm * (profiles[f'{fnums[0]}']['Phi_sph_low'] - profiles[f'{fnums[0]}']['Phi_BH'][rH_idx])
                axes[id].plot(r, phi_Disk_low, lw=2., color='crimson', linestyle='--', alpha=0.5)

            axes[id].plot(r, y, lw=2., color=color, linestyle=linestyle, label=labels[f'{key}'])
        
        elif (time_average):
            Profs.check_AMR(fnums, profiles)
            r  = profiles[f'{Profs.fnums0}']['r']
            if ('Phi' in key):
                r5_idx = np.argmin(np.abs(r - 5.))
                Mdot_avg = Profs.time_avg('Mdot_sigma', profiles, fnums=fnums)[r5_idx]
                norm = 1. / np.sqrt(Mdot_avg * Rg * Rg * c)
                if (key == 'Phi_mid'):
                    norm *= -1
            y = norm * Profs.time_avg(key, profiles, fnums=fnums, vol=vol)
            if (key == 'Dtheta2'): y = np.sqrt(y)
            if (key == 'vphi'):
                vphi = c * Profs.time_avg(key, profiles, fnums=fnums, vol=True)
                OmegaK = Analytic.calc_OmegaK(MBH, a, r)
                Omega  = vphi / (r * Rg)
                y = Omega / OmegaK
            if ('Mdot' in key  or 'Edot' in key):
                axes[id].plot(r, -y, lw=2., color=color, linestyle='--', alpha=0.5)
            if (key == 'Prad'):
                Trad = (3. * y / hr.CONSTANTS['aRAD']) ** (1./4.)
                axes[axes_ids['Te']].plot(r, Trad, color=colors['Prad'], lw=2., label=r'$\rm{Radiation}$')

            if (key == 'Phi_sph_upp'):
                rH     = Analytic.calc_R_Horizon(a)
                rH_idx = np.argmin(np.abs(r - rH))
                phi_Disk_upp = norm * (
                    Profs.time_avg('Phi_sph_upp', profiles, fnums=fnums) - 
                    Profs.time_avg('Phi_BH',      profiles, fnums=fnums)[rH_idx])
                axes[id].plot(r, phi_Disk_upp, lw=2., color='crimson', linestyle='-', label=r'$\rm{sph} - \rm{BH}$')
            if (key == 'Phi_sph_low'):
                linestyle = '--'
                rH     = Analytic.calc_R_Horizon(a)
                rH_idx = np.argmin(np.abs(r - rH))
                phi_Disk_low = norm * (
                    Profs.time_avg('Phi_sph_low', profiles, fnums=fnums) - 
                    Profs.time_avg('Phi_BH',      profiles, fnums=fnums)[rH_idx])
                axes[id].plot(r, phi_Disk_low, lw=2., color='crimson', linestyle='--', alpha=0.5)

            axes[id].plot(r, y, lw=2., color=color, linestyle=linestyle, label=labels[f'{key}'])

        if (log):
            axes[id].set_yscale('log')
        if ('Mdot' in key):
            axes[id].set_ylabel(plot_labels['Mdot_sigma'], fontsize=FS)
        elif ('Edot' in key):  
            axes[id].set_ylabel(plot_labels['Edot_Poyt'], fontsize=FS)
        else:
            axes[id].set_ylabel(plot_labels[f'{key}'], fontsize=FS)

        if (linthresh[f'{key}']):
            axes[id].set_yscale('symlog', linthresh=linthresh[f'{key}'])

        axes[id].set_ylim(lims)
        axes[id].set_xlabel(r'$r/R_g$', fontsize=FS)
        if (r_max != 0.):
            axes[id].set_xlim([1., r_max])

    legend_keys = ['Edot_Poyt', 'Ti', 'Pthm', 'Phi_BH']
    for key in legend_keys:
        axes[axes_ids[f'{key}']].legend(loc='upper right', fontsize=FS-24, frameon=False, ncols=2)
    
    for ax in axes:
        ax.set_xscale('log')
        cv.grid(ax, alpha=0.2, color='black')
    
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
    if (cgs and 'Mdot' in key or 'Edot' in key):
        if 'MBH' not in kwargs:
            raise ValueError("Must supply MBH for cgs=True")
        MBH      = kwargs.get('MBH', 1.0)
        Analytic = theory.Analytic(hr.CONSTANTS, cgs=cgs)
        L_Edd    = Analytic.calc_LEdd(MBH=MBH)
        Mdot_Edd = Analytic.calc_Mdot_Edd(MBH=MBH, eta=0.179)
        if ('Mdot' in key and cgs):
            return (1. / Mdot_Edd)
        elif ('Edot' in key and cgs):
            return -(1. / L_Edd)
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
    else:
        return False
    
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
    parser.add_argument('--time_evolution', 
                        action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""boolean to determine if evolution of profiles 
                        from --start_file to --end_file should be shown.""")
    parser.add_argument('--r_max', type=float, default=0.,
                        required=False,
                        help = """radial extent in Rg (default: R_max)""")
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
                        default='Profiles', 
                        help="""title of PDF output""")
    args = parser.parse_args()
    main(**vars(args))