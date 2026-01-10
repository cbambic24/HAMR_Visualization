import sys
import os
import glob
import argparse
import numpy as np
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{file_dir}')
import compute_variables as cv
import hamr_read as hr
import analysis as ana
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
figsize = [20.0, 26.0]
FS      = 48
plt.rcParams['xtick.labelsize'] = FS
plt.rcParams['ytick.labelsize'] = FS

def main(**kwargs):
    sim            = kwargs.get('sim')
    reduced        = kwargs.get('reduced')
    radiation      = kwargs.get('radiation')
    two_temp       = kwargs.get('two_temp')
    t_start        = kwargs.get('t_start')
    t_end          = kwargs.get('t_end')
    r_Mdot         = kwargs.get('r_Mdot')
    r_Hrho         = kwargs.get('r_Hrho')
    r_Jet          = kwargs.get('r_Jet')
    r_Rad          = kwargs.get('r_Rad')
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
    
    keys = ['Mdot_sigma', 'Edot_jet', 'Edot_Rad', 'Dtheta2', 'Phi_BH']

    limits = {
        'Mdot_sigma' : [1.e-2, 1.e1],
        'Edot_jet'   : [1.e-2, 2.],
        'Edot_Rad'   : [1.e-2, 2.],
        'Dtheta2'    : [1.e-2, 2.],
        'Phi_BH'     : [1., 3.e2]
    }

    colors = {
        'Mdot_sigma' : 'black',
        'Edot_jet'   : 'darkorange',
        'Edot_Rad'   : 'crimson',
        'Dtheta2'    : 'black',
        'Phi_BH'     : 'black'
    }

    labels = {
        'Mdot_sigma' : None,
        'Edot_jet'   : r'$\rm{Jet}$',
        'Edot_Rad'   : r'$\rm{Radiation}$',
        'Dtheta2'    : None,
        'Phi_BH'     : r'$\rm{BH}$'
    }

    plot_labels = {
        'Mdot_sigma' : r'$\dot{M}/\dot{M}_{\rm Edd}$',
        'Edot_jet'   : r'$\dot{E} \: (L_{\rm Edd})$',
        'Edot_Rad'   : r'$\eta = \dot{E} / \langle \dot{M} c^2 \rangle_t$',
        'Dtheta2'    : r'$\langle H_{\rho} / R \rangle_{\rho}$',
        'Phi_BH'     : r'$\phi \equiv \Phi_{\rm B} / \sqrt{\langle \dot{M} \rangle_t R_g^2 c}$'
    }

    axes_ids = {
        'Mdot_sigma' : 0,
        'Edot_jet'   : 2,
        'Edot_Rad'   : 3,
        'Dtheta2'    : 4,
        'Phi_BH'     : 1
    }

    file_info = {
        'averages_dir' : averages_dir,
        'file_number'  : -1,
        'start_file'   : -1,
        'end_file'     : -1,
        'reduced'      : reduced
    }

    TS = cv.Time_Series()
    fnums = TS.get_fnums(**file_info)
    pkeys = keys + ['t', 'r', 'a', 'Vol', 'k_max']
    profiles = TS.load_profiles(averages_dir, fnums, keys=pkeys, reduced=reduced)

    if (cgs):
        L_Edd    = Analytic.calc_LEdd(MBH=MBH)
        Mdot_Edd = Analytic.calc_Mdot_Edd(MBH=MBH, eta=0.179)    
    else:
        L_Edd    = 1.
        Mdot_Edd = 1.
    # Load all time series
    time, Mdot     = TS.calc_time_series('Mdot_sigma', profiles, r0=r_Mdot)
    time, Phi_BH   = TS.calc_time_series('Phi_BH',     profiles, r0=0.)
    time, Edot_jet = TS.calc_time_series('Edot_jet',   profiles, r0=r_Jet)
    time, HoR2     = TS.calc_time_series('Dtheta2',    profiles, r0=r_Hrho, vol=True)
    HoR = np.sqrt(HoR2)
    if (cgs):
        time, Edot_Rad = TS.calc_time_series('Edot_Rad', profiles, r0=r_Rad)

    dT          = time[-1]  - time[0]
    Mdot_smooth = ana.Tools.smooth(Mdot, 1.e2, dT, edge='constant')
    phi_BH      = Phi_BH / np.sqrt(Mdot_smooth * Rg * Rg * c)
    Mdotc2      = Mdot_smooth * c * c

    fig  = plt.figure(figsize = figsize)
    gs   = gridspec.GridSpec(5, 1, figure = fig)
    axes = []
    for i in range(5):
        axes.append(fig.add_subplot(gs[i, 0]))
    
    axes[0].plot(time/1.e4,  Mdot/Mdot_Edd,   lw=2, color=colors['Mdot_sigma'])
    axes[0].plot(time/1.e4, -Mdot/Mdot_Edd,   lw=2, linestyle='--', color=colors['Mdot_sigma'])
    axes[1].plot(time/1.e4, phi_BH,           lw=2, color=colors['Phi_BH'])
    axes[2].plot(time/1.e4, -Edot_jet/L_Edd,  lw=2, color=colors['Edot_jet'])
    axes[2].plot(time/1.e4, -Edot_Rad/L_Edd,  lw=2, color=colors['Edot_Rad'])
    axes[3].plot(time/1.e4, -Edot_jet/Mdotc2, lw=2, color=colors['Edot_jet'])
    axes[3].plot(time/1.e4, -Edot_Rad/Mdotc2, lw=2, color=colors['Edot_Rad'])
    axes[4].plot(time/1.e4, HoR,              lw=2, color=colors['Dtheta2'])

    for key in keys:
        id = int(axes_ids[f'{key}'])
        axes[id].set_ylabel(plot_labels[f'{key}'], fontsize=FS)
        axes[id].set_ylim(limits[f'{key}'])
        axes[id].set_yscale('log')

    for ax in axes:
        cv.grid(ax, alpha=0.2, color='black') 
        ax.set_xlabel(r'${\rm{Time}} \: (10^4 R_g/c)$', fontsize=FS)

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
    parser.add_argument('sim', type=str, help = """simulation name in 
                        Averages directory""")
    parser.add_argument('--t_start', type=float, default=0.,
                        required=False, help = """starting time (Rg/c)""")
    parser.add_argument('--t_end', type=float, default=-1.,
                        required=False, help = """ending time (Rg/c)""")
    parser.add_argument('--r_Mdot', type=float, default=5.,
                        required=False, help = """radius for measuring 
                        Mdot in Rg (default: 5)""")
    parser.add_argument('--r_Hrho', type=float, default=10.,
                        required=False, help = """radius for measuring 
                        scaleheight in Rg (default: 10)""")
    parser.add_argument('--r_Jet', type=float, default=10.,
                        required=False, help = """radius for measuring 
                        jet power in Rg (default: 10)""")
    parser.add_argument('--r_Rad', type=float, default=100.,
                        required=False, help = """radius for measuring 
                        Edot_Rad in Rg (default: 100)""")
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
                        default='Time_Series', 
                        help="""title of PDF output""")
    args = parser.parse_args()
    main(**vars(args))