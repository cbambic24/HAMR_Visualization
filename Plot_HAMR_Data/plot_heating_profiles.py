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
    MBH      = 10.
    L_Edd    = Analytic.calc_LEdd(MBH=MBH)

    # Radiation trapping radius
    Rt = 0.

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
        'qdiss',         'qrad',           'qie',         'qdiss_ion',
        'qdiss_ie',      'qrad_ie',        'qie_ie',      'qdiss_ion_ie',
        'qdiss_thin',    'qrad_thin',      'qie_thin',    'qdiss_ion_thin',
        'qdiss_thin_ie', 'qrad_thin_ie',   'qie_thin_ie', 'qdiss_ion_thin_ie'
    ]
    
    keys += [
        'qdiss_out',         'qrad_out',           'qie_out',         'qdiss_ion_out',
        'qdiss_ie_out',      'qrad_ie_out',        'qie_ie_out',      'qdiss_ion_ie_out',
        'qdiss_thin_out',    'qrad_thin_out',      'qie_thin_out',    'qdiss_ion_thin_out',
        'qdiss_thin_ie_out', 'qrad_thin_ie_out',   'qie_thin_ie_out', 'qdiss_ion_thin_ie_out'
    ]

    limits = {
        'qdiss'     : [1.e-5, 1.e0],
        'qdiss_ion' : [1.e-5, 1.e0],
        'qrad'      : [1.e-5, 1.e0],
        'qie'       : [1.e-5, 1.e0]
    }

    plot_labels = {
        'qdiss'     : r'$\dot{E}_{\rm{diss}}$', 
        'qdiss_ion' : r'$\dot{E}^{\rm ion}_{\rm diss}$',
        'qrad'      : r'$\dot{E}_{\rm rad}$',
        'qie'       : r'$\dot{E}_{ie}$'
    }

    axes_ids = {
        'qdiss'     : 0,
        'qdiss_ion' : 1,
        'qrad'      : 2,
        'qie'       : 3
    }
    
    file_info = {
        'averages_dir' : averages_dir,
        'file_number'  : file_number,
        'start_file'   : start_file,
        'end_file'     : end_file,
        'reduced'      : reduced
    }

    Profs    = cv.Heating_Profiles()
    fnums    = Profs.get_fnums(**file_info)
    profiles = Profs.load_profiles(averages_dir, fnums, 
        keys=(keys + ['dx1', 'r']), reduced=reduced)
    single_file = True if (len(fnums) == 1) else False

    fig  = plt.figure(figsize = figsize)
    gs   = gridspec.GridSpec(2, 2, figure = fig)
    axes = []
    for i in range(2):
        for j in range(2):
            axes.append(fig.add_subplot(gs[i, j]))

    for key in ['qdiss', 'qdiss_ion', 'qrad', 'qie']: 
        id        = int(axes_ids[f'{key}'])
        lims      = limits[f'{key}']
        linestyle = '-'
        norm      = (Rg * Rg * Rg / L_Edd)

        if (single_file):
            r     = profiles[f'{fnums[0]}']['r']
            dx1   = profiles[f'{fnums[0]}']['dx1']
            r_plt = r[r > Rt]
            if (key == 'qdiss'):
                qdiss         = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss'][        r > Rt] * dx1)
                qdiss_thin    = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_thin'][   r > Rt] * dx1)
                qdiss_ie      = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_ie'][     r > Rt] * dx1)
                qdiss_thin_ie = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_thin_ie'][r > Rt] * dx1)
                
                qdiss         = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_out'][        r > Rt] * dx1)
                qdiss_thin    = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_thin_out'][   r > Rt] * dx1)
                qdiss_ie      = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_ie_out'][     r > Rt] * dx1)
                qdiss_thin_ie = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_thin_ie_out'][r > Rt] * dx1)
            elif (key == 'qdiss_ion'):
                qdiss_ion         = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_ion'][        r > Rt] * dx1)
                qdiss_ion_thin    = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_ion_thin'][   r > Rt] * dx1)
                qdiss_ion_ie      = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_ion_ie'][     r > Rt] * dx1)
                qdiss_ion_thin_ie = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_ion_thin_ie'][r > Rt] * dx1)
                
                qdiss_ion         = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_ion_out'][        r > Rt] * dx1)
                qdiss_ion_thin    = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_ion_thin_out'][   r > Rt] * dx1)
                qdiss_ion_ie      = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_ion_ie_out'][     r > Rt] * dx1)
                qdiss_ion_thin_ie = norm * np.cumsum(profiles[f'{fnums[0]}']['qdiss_ion_thin_ie_out'][r > Rt] * dx1)
            elif (key == 'qrad'):
                qrad         = norm * np.cumsum(profiles[f'{fnums[0]}']['qrad'][        r > Rt] * dx1)
                qrad_thin    = norm * np.cumsum(profiles[f'{fnums[0]}']['qrad_thin'][   r > Rt] * dx1)
                qrad_ie      = norm * np.cumsum(profiles[f'{fnums[0]}']['qrad_ie'][     r > Rt] * dx1)
                qrad_thin_ie = norm * np.cumsum(profiles[f'{fnums[0]}']['qrad_thin_ie'][r > Rt] * dx1)
                
                qrad         = norm * np.cumsum(profiles[f'{fnums[0]}']['qrad_out'][        r > Rt] * dx1)
                qrad_thin    = norm * np.cumsum(profiles[f'{fnums[0]}']['qrad_thin_out'][   r > Rt] * dx1)
                qrad_ie      = norm * np.cumsum(profiles[f'{fnums[0]}']['qrad_ie_out'][     r > Rt] * dx1)
                qrad_thin_ie = norm * np.cumsum(profiles[f'{fnums[0]}']['qrad_thin_ie_out'][r > Rt] * dx1)
            elif (key == 'qie'):
                qie         = norm * np.cumsum(profiles[f'{fnums[0]}']['qie'][        r > Rt] * dx1)
                qie_thin    = norm * np.cumsum(profiles[f'{fnums[0]}']['qie_thin'][   r > Rt] * dx1)
                qie_ie      = norm * np.cumsum(profiles[f'{fnums[0]}']['qie_ie'][     r > Rt] * dx1)
                qie_thin_ie = norm * np.cumsum(profiles[f'{fnums[0]}']['qie_thin_ie'][r > Rt] * dx1)
                
                qie         = norm * np.cumsum(profiles[f'{fnums[0]}']['qie_out'][        r > Rt] * dx1)
                qie_thin    = norm * np.cumsum(profiles[f'{fnums[0]}']['qie_thin_out'][   r > Rt] * dx1)
                qie_ie      = norm * np.cumsum(profiles[f'{fnums[0]}']['qie_ie_out'][     r > Rt] * dx1)
                qie_thin_ie = norm * np.cumsum(profiles[f'{fnums[0]}']['qie_thin_ie_out'][r > Rt] * dx1)

            if (key == 'qdiss'):
                axes[id].plot(r_plt, qdiss, lw=2., color='black', 
                              linestyle=linestyle, label=r'$\rm{Total}$')
                axes[id].plot(r_plt, qdiss_thin, lw=2., color='dodgerblue', 
                              linestyle=linestyle, label=r'$\tau_{\rm es} < 1$')
                axes[id].plot(r_plt, qdiss_ie, lw=2., color='violet', 
                              linestyle=linestyle, label=r'$\rm{Ion-Electron}$')
                axes[id].plot(r_plt, qdiss_thin_ie, lw=2., color='darkorange', 
                              linestyle=linestyle, label=r'$\rm{Corona}$')
            elif (key == 'qdiss_ion'):
                axes[id].plot(r_plt, qdiss_ion, lw=2., color='black', 
                              linestyle=linestyle, label=r'$\rm{Total}$')
                axes[id].plot(r_plt, qdiss_ion_thin, lw=2., color='dodgerblue', 
                              linestyle=linestyle, label=r'$\tau_{\rm es} < 1$')
                axes[id].plot(r_plt, qdiss_ion_ie, lw=2., color='violet', 
                              linestyle=linestyle, label=r'$\rm{Ion-Electron}$')
                axes[id].plot(r_plt, qdiss_ion_thin_ie, lw=2., color='darkorange', 
                              linestyle=linestyle, label=r'$\rm{Corona}$')
            elif (key == 'qrad'):
                axes[id].plot(r_plt, qrad, lw=2., color='black', 
                              linestyle=linestyle, label=r'$\rm{Total}$')
                axes[id].plot(r_plt, qrad_thin, lw=2., color='dodgerblue', 
                              linestyle=linestyle, label=r'$\tau_{\rm es} < 1$')
                axes[id].plot(r_plt, qrad_ie, lw=2., color='violet', 
                              linestyle=linestyle, label=r'$\rm{Ion-Electron}$')
                axes[id].plot(r_plt, qrad_thin_ie, lw=2., color='darkorange', 
                              linestyle=linestyle, label=r'$\rm{Corona}$')
            elif (key == 'qie'):
                axes[id].plot(r_plt, qie, lw=2., color='black', 
                              linestyle=linestyle, label=r'$\rm{Total}$')
                axes[id].plot(r_plt, qie_thin, lw=2., color='dodgerblue', 
                              linestyle=linestyle, label=r'$\tau_{\rm es} < 1$')
                axes[id].plot(r_plt, qie_ie, lw=2., color='violet', 
                              linestyle=linestyle, label=r'$\rm{Ion-Electron}$')
                axes[id].plot(r_plt, qie_thin_ie, lw=2., color='darkorange', 
                              linestyle=linestyle, label=r'$\rm{Corona}$')
        elif (time_average):
            Profs.check_AMR(fnums, profiles)
            r  = profiles[f'{Profs.fnums0}']['r']
            qdiss = Profs.time_avg('qdiss', profiles, fnums=fnums)
            
        #axes[id].set_yscale('log')
        axes[id].set_ylabel(plot_labels[f'{key}'], fontsize=FS)

        #axes[id].set_ylim(lims)
        axes[id].set_xlabel(r'$r/R_g$', fontsize=FS)
        if (r_max != 0.):
            axes[id].set_xlim([1., r_max])

        axes[id].legend(loc='upper right', fontsize=FS-24, frameon=False, ncols=1)
    
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