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
import analysis as ana
import theory as theory

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.style as style
matplotlib.rcParams['text.usetex'] = True
style.use('seaborn-v0_8-bright')
matplotlib.rcParams['figure.dpi'] = 300

figsize = [20.0, 26.0]
FS      = 48
FS      = 28
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
    show_HoR       = kwargs.get('show_HoR')
    show_rho0      = kwargs.get('show_rho0')
    disable_TeX    = kwargs.get('disable_TeX')
    MBH            = kwargs.get('MBH')
    
    matplotlib.rcParams['text.usetex'] = False if disable_TeX else True

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

    plot_keys = [
        'Mdot',
        'PhiBH',
        'Edot',
        'eta'
    ]

    if show_HoR:  plot_keys += ['scaleheight']
    if show_rho0: plot_keys += ['rho0']

    plot_info = dict.fromkeys(plot_keys)
    for key in plot_keys:
        plot_info[key] = dict.fromkeys(['quant', 'colors', 'axis_id', 'labels', 'title', 'log'])

    # Limits for each figure
    user_limits = {
        'Mdot'        : None,
        'PhiBH'       : [0., 60.],
        'Edot'        : None,
        'eta'         : None,
        'scaleheight' : [0.01, 2],
        'rho0'        : None
    }

    # Quantities for each figure
    plot_info['Mdot'][ 'quant'] = 'Mdot_sigma'
    plot_info['PhiBH']['quant'] = 'Phi_BH'
    plot_info['Edot'][ 'quant'] = ['Edot_jet', 'Edot_Rad'] if radiation else ['Edot_jet']
    if show_HoR:
        plot_info['scaleheight']['quant'] = 'Dtheta2'
    if show_rho0:
        plot_info['rho0']['quant']        = 'rho0'

    # Log-scale
    for key in plot_keys: plot_info[key]['log'] = True
    plot_info['PhiBH']['log'] = False

    # Colors for each figure
    for key in plot_keys: plot_info[key]['colors'] = 'black'
    plot_info['Edot']['colors'] = ['darkorange', 'crimson'] if radiation else ['darkorange']

    # Labels for each figure
    for key in plot_keys: plot_info[key]['labels'] = None
    plot_info['Edot']['labels'] = [r'$\rm{Jet}$', r'$\rm{Radiation}$'] if radiation else [r'$\rm{Jet}$']

    # Axis ID
    for id, key in enumerate(plot_keys):
        plot_info[key]['axis_id'] = id
    
    # Determine valid profile files
    file_info = {
        'averages_dir' : averages_dir,
        'file_number'  : -1,
        'start_file'   : -1,
        'end_file'     : -1,
        'reduced'      : reduced
    }

    TS    = cv.Time_Series()
    fnums = TS.get_fnums(**file_info)

    if radiation:
        data_keys = ['Mdot_sigma', 'Edot_jet', 'Edot_Rad', 'Dtheta2', 'Phi_BH']
    else:
        data_keys = ['Mdot_sigma', 'Edot_jet', 'Dtheta2', 'Phi_BH']

    prof_args = {
        'averages_dir' : file_info['averages_dir'],
        'fnums'        : fnums,
        'keys'         : data_keys + ['t', 'r', 'a', 'Vol', 'k_max', 'rho0'],
        'reduced'      : reduced
    }

    profiles = TS.load_profiles(**prof_args)

    L_Edd    = Analytic.calc_LEdd(MBH=MBH)                if cgs else 1.
    Mdot_Edd = Analytic.calc_Mdot_Edd(MBH=MBH, eta=0.179) if cgs else 1.

    # Load all time series
    time, Mdot     = TS.calc_time_series('Mdot_sigma', profiles, r0=r_Mdot)
    time, Phi_BH   = TS.calc_time_series('Phi_BH',     profiles, r0=0.)
    time, Edot_jet = TS.calc_time_series('Edot_jet',   profiles, r0=r_Jet)
    time, HoR2     = TS.calc_time_series('Dtheta2',    profiles, r0=r_Hrho, vol=True)
    time, rho0     = TS.calc_time_series('rho0',       profiles)

    HoR = np.sqrt(HoR2)
    if radiation:
        time, Edot_Rad = TS.calc_time_series('Edot_Rad', profiles, r0=r_Rad)

    dT          = time[-1]  - time[0]
    Mdot_smooth = ana.Tools.smooth(Mdot, 1.e2, dT, edge='constant')
    phi_BH      = Phi_BH / np.sqrt(Mdot_smooth * Rg * Rg * c)
    Mdotc2      = Mdot_smooth * c * c
    
    fig  = plt.figure(figsize = figsize)
    gs   = gridspec.GridSpec(len(plot_keys), 1, figure = fig)
    axes = []
    for i in range(len(plot_keys)):
        axes.append(fig.add_subplot(gs[i, 0]))

    id = plot_info['Mdot']['axis_id']
    axes[id].plot(time/1.e4,  Mdot / Mdot_Edd, color=plot_info['Mdot']['colors'], lw=2)
    axes[id].plot(time/1.e4, -Mdot / Mdot_Edd, color=plot_info['Mdot']['colors'], lw=2, linestyle='--')
    
    id = plot_info['PhiBH']['axis_id']
    axes[id].plot(time/1.e4, phi_BH, color=plot_info['PhiBH']['colors'], lw=2)

    id_Edot = plot_info['Edot']['axis_id']
    id_eta  = plot_info['eta']['axis_id']
    plt_args = {
        'color' : plot_info['Edot']['colors'][0],
        'label' : plot_info['Edot']['labels'][0],
        'lw'    : 2.
    }
    axes[id_Edot].plot(time/1.e4, -Edot_jet / L_Edd,  **plt_args)
    axes[id_eta].plot( time/1.e4, -Edot_jet / Mdotc2, **plt_args)
    
    if radiation:
        plt_args['color'] = plot_info['Edot']['colors'][1]
        plt_args['label'] = plot_info['Edot']['labels'][1]
        axes[id_Edot].plot(time/1.e4, -Edot_Rad/L_Edd,  **plt_args)
        axes[id_eta].plot( time/1.e4, -Edot_Rad/Mdotc2, **plt_args)
    
    if show_HoR and not show_rho0:
        id = plot_info['HoR']['axis_id']
        axes[id].plot(time/1.e4, HoR, color=plot_info['scaleheight']['colors'], lw=2)
    
    if show_rho0 and not show_HoR:
        id = plot_info['rho0']['axis_id']
        axes[id].plot(time/1.e4, rho0, color=plot_info['rho0']['colors'], lw=2)
    
    if show_HoR and show_rho0:
        id_HoR  = plot_info['HoR'][ 'axis_id']
        id_rho0 = plot_info['rho0']['axis_id']
        axes[id_HoR].plot(time/1.e4, HoR,  color=plot_info['scaleheight']['colors'], lw=2)
        axes[id_rho0].plot(time/1.e4, rho0, color=plot_info['rho0']['colors'], lw=2)

    axes[id_Edot].legend(loc='lower left', fontsize=FS-4, frameon=False)
    axes[id_eta].legend( loc='lower left', fontsize=FS-4, frameon=False)

    plotted_data = {
        'Mdot'  : [Mdot/Mdot_Edd],
        'PhiBH' : [phi_BH],
        'Edot'  : [-Edot_jet/L_Edd,  -Edot_Rad/L_Edd]  if radiation else [-Edot_jet/L_Edd],
        'eta'   : [-Edot_jet/Mdotc2, -Edot_Rad/Mdotc2] if radiation else [-Edot_jet/Mdotc2],
        'scaleheight' : [HoR],
        'rho0'  : [rho0]
    }

    if (t_start != 0. or t_end != -1.):
        start_time = t_start
        end_time   = t_end if t_end != -1 else time[-1]
        for ax in axes:
            ax.set_xlim([start_time/1.e4, end_time/1.e4])
    
        id_start = np.argmin(np.abs(time - start_time))
        id_end   = np.argmin(np.abs(time - end_time))
        for key in plot_keys:
            if user_limits[key] is None:
                new_curves = []
                for curve in plotted_data[key]:
                    new_curves.append(curve[id_start:id_end+1])
                plotted_data[key] = new_curves

    for key in plot_keys:
        id = int(plot_info[key]['axis_id'])
        axes[id].set_ylabel(get_plot_title(key, cgs), fontsize=FS)
        limits = get_limits(key, user_limits, plotted_data, plot_info)
        axes[id].set_ylim(limits)
        if plot_info[key]['log']:
            axes[id].set_yscale('log')
        
    for ax in axes:
        cv.grid(ax, alpha=0.2, color='black') 
        ax.set_xlabel(r'${\rm{Time}} \: (10^4 R_g/c)$', fontsize=FS)

    print("Saving figure ...", flush=True)
    fig.tight_layout()
    plt.savefig(f'{save_dir}/{plot_title}.png', format='png')
    plt.close()

def get_plot_title(key, cgs):
    plot_titles = {
        'Mdot'       : r'$\dot{M}/\dot{M}_{\rm Edd}$'                    if cgs else r'$\dot{M} \: (\rho_0 R_g^2 c)$',
        'Edot'       : r'$\dot{E} \: (L_{\rm Edd})$' if cgs else r'$\dot{E} \: (\rho_0 c^3 R_g^2)$',
        'eta'        : r'$\eta = \dot{E} / \langle \dot{M} c^2 \rangle_t$', 
        'scaleheight': r'$\langle H_{\rho} / R \rangle_{\rho}$',
        'PhiBH'      : r'$\phi \equiv \Phi_{\rm B} / \sqrt{\langle \dot{M} \rangle_t R_g^2 c}$',
        'rho0'       : r'$\rho_0 \: (\rm{g/cm}^3)$'
    }
    return plot_titles[key]

def get_limits(key, user_limits, plotted_data, plot_info):
    if user_limits[key] == None:
        all_data = np.concatenate(plotted_data[key])
        if plot_info[key]['log']:
            pos_data = all_data[all_data > 0]
            log_min = np.floor(np.amin(np.log10(np.abs(pos_data))))
            log_max = np.floor(np.amax(np.log10(np.abs(pos_data))))
            return [10 ** log_min, 10 ** (log_max + 1)]
        else:
            return [np.amin(all_data), np.amax(all_data)]
    else:
        return user_limits[key]

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Plot time series.""")
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
                        should be included in figure.""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if two-temperature 
                        physics should be assumed.""")   
    parser.add_argument('--MBH', type=float, default=10.,
                        required=False,
                        help = """black hole mass in Msun (default: 10)""")
    parser.add_argument('--show_HoR', action='store_true', required=False,
                        help="""boolean to determine if scaleheight should
                        be plotted in time series figure.""")
    parser.add_argument('--show_rho0', action='store_true', required=False,
                        help="""boolean to determine if density scale (rho0) 
                        should be plotted in time series figure.""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if reduced data
                        should be used for plotting.""")
    parser.add_argument('--disable_TeX', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""boolean to disable TeX in plotting""")
    parser.add_argument('--plot_title', type=str, required=False, 
                        default='Time_Series', 
                        help="""title of PDF output""")
    args = parser.parse_args()
    main(**vars(args))