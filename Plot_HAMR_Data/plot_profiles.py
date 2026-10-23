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
import matplotlib.style as style
style.use('seaborn-v0_8-bright')
matplotlib.rcParams['figure.dpi'] = 300

figsize = [40.0, 26.0]
FS      = 40
plt.rcParams['xtick.labelsize'] = FS
plt.rcParams['ytick.labelsize'] = FS

def main(**kwargs):
    sim            = kwargs.get('sim')
    file_number    = int(kwargs.get('file_number'))
    start_file     = int(kwargs.get('start_file'))
    end_file       = int(kwargs.get('end_file'))
    reduced        = kwargs.get('reduced')
    radiation      = kwargs.get('radiation')
    two_temp       = kwargs.get('two_temp')
    r_max          = kwargs.get('r_max')
    plot_title     = kwargs.get('plot_title')
    disable_TeX    = kwargs.get('disable_TeX')
    MBH            = kwargs.get('MBH')

    matplotlib.rcParams['text.usetex'] = False if disable_TeX else True

    cgs = True if (two_temp or radiation) else False
    Theory = theory.Analytic(hr.CONSTANTS, cgs=cgs)

    Msun = hr.CONSTANTS['Msun']     if cgs else 1.
    c    = hr.CONSTANTS['c']        if cgs else 1.
    G    = hr.CONSTANTS['G']        if cgs else 1.
    Rg   = G * MBH * Msun / (c * c) if cgs else 1.

    # Determine profile directory
    averages_dir = cv.user_path + f'/Averages/{sim}'
    save_dir = cv.user_path +  '/PDF'

    plot_keys = [
        'Mdot',     'rho',        'Temp',     'vR',
        'Edot',  'scaleheight', 'pressures', 'vphi',
        'PhiBH',    'bphi',        'bz',      'vz'
    ]

    plot_info = dict.fromkeys(plot_keys)
    for key in plot_keys:
        plot_info[key] = dict.fromkeys(['quant', 'colors', 'axis_id', 'labels', 'title', 'log', 'linthresh'])
    
    # Limits for each figure
    user_limits = {
        'Mdot'        : None,
        'rho'         : None,
        'Temp'        : [1.e4,  1.e12] if cgs else [1.e-5, 1.],
        'vR'          : [-1, 0.1],
        'vphi'        : [0.1, 1.1],
        'vz'          : [-0.01, 0.01],
        'Edot'        : None,
        'scaleheight' : [0.01, 2],
        'pressures'   : None,
        'PhiBH'       : [1., 1.e3],
        'bphi'        : None,
        'bz'          : None
    } 

    # Quantities for each figure
    plot_info['Mdot'][       'quant'] = 'Mdot_sigma'
    plot_info['rho'][        'quant'] = 'rho'
    plot_info['Temp'][       'quant'] = ['Ti', 'Te'] if two_temp else 'Ti'
    plot_info['vR'][         'quant'] = 'vR'
    plot_info['vphi'][       'quant'] = 'vphi'
    plot_info['vz'][         'quant'] = 'vz'
    plot_info['bphi'][       'quant'] = 'bphi'
    plot_info['bz'][         'quant'] = 'bz'
    plot_info['Edot'][       'quant'] = ['Edot_jet', 'Edot_Poyt', 'Edot_Fluid', 'Edot_Rad'] if radiation else ['Edot_jet', 'Edot_Poyt', 'Edot_Fluid']
    plot_info['scaleheight']['quant'] = 'Dtheta2'
    plot_info['pressures'][  'quant'] = ['Pthm', 'Pmag', 'Prad'] if radiation else ['Pthm', 'Pmag']
    plot_info['PhiBH'][      'quant'] = ['Phi_BH', 'Phi_mid', 'Phi_sph_upp', 'Phi_sph_low']

    # Log-scale and linthresh if symlog
    for key in plot_keys: plot_info[key]['log']       = True
    for key in plot_keys: plot_info[key]['linthresh'] = None

    plot_info['vz']['log'] = False

    plot_info['bphi']['linthresh'] = 'infer'
    plot_info['bz'][  'linthresh'] = 'infer'
    plot_info['vR'][  'linthresh'] = 1.e-3
    
    # Colors for each figure
    for key in plot_keys: plot_info[key]['colors'] = 'black'
    plot_info['Temp']['colors']      = ['darkorange', 'dodgerblue'] if two_temp else 'darkorange'
    plot_info['Edot']['colors']      = ['darkorange', 'dodgerblue', 'goldenrod', 'crimson'] if radiation else ['darkorange', 'dodgerblue', 'goldenrod']
    plot_info['pressures']['colors'] = ['darkorange', 'dodgerblue', 'crimson'] if radiation else ['darkorange', 'dodgerblue']
    plot_info['PhiBH']['colors']     = ['black', 'darkorange', 'dodgerblue', 'dodgerblue']

    # Labels for each figure
    for key in plot_keys: plot_info[key]['labels'] = None
    plot_info['Temp']['labels']      = [r'$\rm{Ions}$', r'$\rm{Electrons}$'] if cgs else None
    plot_info['Edot']['labels']      = [r'$\rm{Jet}$', r'$\rm{Poynting}$', r'$\rm{Enthalpy}$', r'$\rm{Radiation}$'] if radiation else [r'$\rm{Jet}$', r'$\rm{Poynting}$', r'$\rm{Enthalpy}$']
    plot_info['pressures']['labels'] = [r'$\rm{Thermal}$', r'$\rm{Magnetic}$', r'$\rm{Radiation}$'] if radiation else [r'$\rm{Thermal}$', r'$\rm{Magnetic}$']
    plot_info['PhiBH']['labels']     = [r'$\rm{BH}$', r'$\rm{mid}$', r'$\rm{sph}$', None]

    # Axis ID
    for id, key in enumerate(plot_keys):
        plot_info[key]['axis_id'] = id

    # Determine valid profile files
    file_info = {
        'averages_dir' : averages_dir,
        'file_number'  : file_number,
        'start_file'   : start_file,
        'end_file'     : end_file,
        'reduced'      : reduced
    }

    Profs = cv.Profiles()
    fnums = Profs.get_fnums(**file_info)

    # Load in data
    data_keys = [
        'Mdot_sigma', 'rho',       'Ti',          'Pthm',        'Pmag',
        'Edot_jet',   'Edot_Poyt', 'Edot_Fluid',  'Dtheta2',
        'Phi_BH',     'Phi_mid',   'Phi_sph_upp', 'Phi_sph_low',
        'vR',         'vphi',      'vz',          'bphi',        'bz'
    ]
    if two_temp:
        data_keys += ['Te']
    if radiation:
        data_keys += ['Prad', 'Edot_Rad']

    prof_args = {
        'averages_dir' : file_info['averages_dir'],
        'fnums'        : fnums,
        'keys'         : data_keys + ['Vol', 'r', 'a', 't'],
        'reduced'      : reduced
    }

    profiles = Profs.load_profiles(**prof_args)
    single_file = True if (len(fnums) == 1) else False

    # Determine radius for plotting 
    if not single_file:
        Profs.check_AMR(fnums, profiles)
    fnums0 = f'{fnums[0]}' if single_file else f'{Profs.fnums0}'
    r = profiles[fnums0]['r']

    # Calculate average mass-accretion rate for Phi_BH normalization
    r5_idx = np.argmin(np.abs(r - 5.))
    if single_file:
        Mdot_avg = profiles[fnums0]['Mdot_sigma'][r5_idx]
    else:
        Mdot_avg = Profs.time_avg('Mdot_sigma', profiles)[r5_idx]

    # Produce figure
    fig  = plt.figure(figsize = figsize)
    gs   = gridspec.GridSpec(3, 4, figure = fig)
    axes = []
    for i in range(3):
        for j in range(4):
            axes.append(fig.add_subplot(gs[i, j]))

    for key in plot_keys:
        id = plot_info[key]['axis_id']
        plotted_data = []

        # Normalizations
        a    = profiles[fnums0]['a']
        norm = get_norm(key, MBH, a, Theory, Mdot_avg=Mdot_avg, r=r)

        # Calculate profile
        vol = False if key == 'Mdot' or key == 'Edot' or key == 'PhiBH' else True

        if single_file:
            def fetch(quant, _vol=vol):
                return profiles[fnums0][quant] / profiles[fnums0]['Vol'] if _vol else profiles[fnums0][quant]
        else:
            def fetch(quant, _vol=vol):
                return Profs.time_avg(quant, profiles, vol=_vol)

        if isinstance(plot_info[key]['quant'], list):
            for quant, color, label in zip(plot_info[key]['quant'], plot_info[key]['colors'], plot_info[key]['labels']):
                q = fetch(quant)
                q_norm = norm * q
                if quant == 'Phi_mid': q_norm *= -1
                if quant in ('Phi_sph_upp', 'Phi_sph_low'):
                    plot_phi_sph_disk(axes, id, r, quant, q_norm, norm, a, profiles, fnums0, Theory)

                axes[id].plot(r, q_norm, lw=2., color=color, linestyle='-', label=label)
                plotted_data.append(q_norm)

                if key == 'Edot':
                    axes[id].plot(r, -q_norm, lw=2., color=color, linestyle='--', alpha=0.5)
             
        else:
            quant = plot_info[key]['quant']
            q     = fetch(quant)
            q_norm= norm * q
            if key == 'scaleheight':
                q_norm = np.sqrt(q_norm) # Apply square root to the scale height
            axes[id].plot(r, q_norm, lw=2., color=plot_info[key]['colors'], linestyle='-', label=plot_info[key]['labels'])
            plotted_data.append(q_norm)
            
            if key == 'Mdot':
                axes[id].plot(r, -q_norm, lw=2., color=plot_info[key]['colors'], linestyle='--', alpha=0.5)

        # Plot radiation temperature if radiation = True 
        if key == 'Temp':
            if radiation:
                Prad = fetch('Prad')
                Trad = (3. * Prad / hr.CONSTANTS['aRAD']) ** (1./4.)
                axes[id].plot(r, Trad, color='crimson', lw=2., label=r'$\rm{Radiation}$')

        if plot_info[key]['log']:
            axes[id].set_yscale('log')

        axes[id].set_ylabel(get_plot_title(key, cgs), fontsize=FS)
        
        if (r_max != 0.):
            axes[id].set_xlim([1., r_max])
            for i, plotted_curve in enumerate(plotted_data):
                plotted_data[i] = plotted_curve[(r < r_max)]
                
        if plot_info[key]['linthresh']:
            linthresh = get_linthresh(plot_info[key]['linthresh'], plotted_data)
            axes[id].set_yscale('symlog', linthresh=linthresh)

        limits = cv.get_limits(key, user_limits, plotted_data, plot_info)
        axes[id].set_ylim(limits)
        
        axes[id].set_xlabel(r'$r/R_g$', fontsize=FS)

        legend_keys = ['Temp', 'Edot', 'pressures', 'PhiBH'] if two_temp else ['Edot', 'pressures', 'PhiBH'] 
        if key in legend_keys:
            axes[id].legend(loc='upper right', fontsize=FS-18, frameon=False, ncols=2)
    
    for ax in axes:
        ax.set_xscale('log')
        cv.grid(ax, alpha=0.2, color='black')

    opt = dict(ha='left', va='center', color='black', bbox=dict(facecolor='none', edgecolor='none'))
    tstring = get_tstring(profiles, fnums, single_file, ndec=3)
    axes[0].text(0.05, 0.90, r'$t = $' + tstring + r' $R_g/c$', fontsize=FS-2, transform=axes[0].transAxes, **opt)  
    
    os.makedirs(save_dir, exist_ok=True)
    print("Saving figure ...", flush=True)
    fig.tight_layout()
    plt.savefig(f'{save_dir}/{plot_title}.png', format='png')
    plt.close()

def get_plot_title(key, cgs):
    plot_titles = {
        'Mdot'       : r'$\dot{M}/\dot{M}_{\rm Edd}$'                    if cgs else r'$\dot{M} \: (\rho_0 R_g^2 c)$',
        'rho'        : r'$\langle \rho \rangle_{\rho} \: (\rm{g/cm}^3)$' if cgs else r'$\langle \rho /\rho_0 \rangle_{\rho}$',
        'Temp'       : r'$\rm{Temperature} \: (\rm{K})$'                 if cgs else r'${\rm{Temperature}} \: (m_i c^2)$',
        'pressures'  : r'$\rm{Pressure} \: (\rm{erg/cm}^3)$'             if cgs else r'$\rm{Pressure} \: (\rho_0 c^2)$',
        'vR'         : r'$\langle v^R / c \rangle_{\rho}$',
        'vphi'       : r'$\langle \Omega / \Omega_{\rm K} \rangle_{\rho}$',
        'vz'         : r'$\langle v^z / c \rangle_{\rho}$',
        'bphi'       : r'$\langle b^{\varphi} \rangle_{\rho} \: (\rm{G})$' if cgs else r'$\langle b^{\varphi} / B_0 \rangle_{\rho}$',
        'bz'         : r'$\langle b^z \rangle_{\rho} \: (\rm{G})$'         if cgs else r'$\langle b^z / B_0 \rangle_{\rho}$',
        'Edot'       : r'$\dot{E} \: (L_{\rm Edd})$' if cgs else r'$\dot{E} \: (\rho_0 c^3 R_g^2)$',
        'scaleheight': r'$\langle H_{\rho} / R \rangle_{\rho}$',
        'PhiBH'      : r'$\phi \equiv \Phi_{\rm B} / \sqrt{\langle \dot{M} \rangle_t R_g^2 c}$'
    }
    return plot_titles[key]

def plot_phi_sph_disk(axes, id, r, quant, q_norm, norm, a, profiles, fnums0, Theory):
    rH     = Theory.calc_R_Horizon(a)
    rH_idx = np.argmin(np.abs(r - rH))
    phi_BH = norm * profiles[fnums0]['Phi_BH'][rH_idx]
    q_diff = q_norm - phi_BH
    linestyle  = '--' if quant == 'Phi_sph_low' else '-'
    alpha      = 0.5  if quant == 'Phi_sph_low' else 1.
    flux_label = r'$\rm{sph} - \rm{BH}$' if quant == 'Phi_sph_upp' else None
    axes[id].plot(r, q_diff, lw=2., color='crimson', linestyle=linestyle, alpha=alpha, label=flux_label)

def get_norm(key, MBH, a, Theory, **kwargs):
    Msun = hr.CONSTANTS['Msun']     if Theory.cgs else 1.
    c    = hr.CONSTANTS['c']        if Theory.cgs else 1.
    G    = hr.CONSTANTS['G']        if Theory.cgs else 1.
    Rg   = G * MBH * Msun / (c * c) if Theory.cgs else 1.

    norm = 1.
    if key == 'Mdot':
        if Theory.cgs:
            Mdot_Edd = Theory.calc_Mdot_Edd(MBH=MBH, eta=0.179)
            norm = 1. / Mdot_Edd
    if key == 'Edot':
        L_Edd = Theory.calc_LEdd(MBH=MBH) if Theory.cgs else 1.
        norm  = -1. / L_Edd
    if key == 'PhiBH':
        Mdot_avg = kwargs.get('Mdot_avg')
        if Mdot_avg is None:
            raise ValueError("Must supply Mdot_avg for phi normalization.")
        norm = 1. / np.sqrt(Mdot_avg * Rg * Rg * c)
    if key == 'vphi':
        r = kwargs.get('r')
        if r is None:
            raise ValueError("Must supply r for Omega normalization.")
        OmegaK = Theory.calc_OmegaK(MBH, a, r)
        norm = (1. / OmegaK) * (c / (r * Rg)) # This is now Omega/OmegaK
    return norm

def get_linthresh(linthresh, quants):
    if linthresh == 'infer':
        all_data    = np.concatenate(quants)
        log_thresh  = np.floor(np.amax(np.log10(np.abs(all_data)))) - 2
        linthresh = 10 ** log_thresh
    return linthresh

def get_tstring(profiles, fnums, single_file, ndec=2):
    if single_file:
        time = profiles[f'{fnums[0]}']['t']
        return cv.get_nice_time(time, prec=ndec)

    tstrt = profiles[f'{fnums[0]}']['t']
    tend  = profiles[f'{fnums[-1]}']['t']
    coeff_strt, oom_strt = cv.get_time_coeff_expon(tstrt)
    coeff_end,  oom_end  = cv.get_time_coeff_expon(tend)

    if oom_strt == oom_end:
        return (rf'$%.{ndec}f-%.{ndec}f$' % (coeff_strt, coeff_end) + r'$\times$' + r'$10^{%d}$' % oom_end)
    return cv.get_nice_time(tstrt, prec=ndec) + '-' + cv.get_nice_time(tend, prec=ndec)
    
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
    parser.add_argument('--r_max', type=float, default=0.,
                        required=False,
                        help = """radial extent in Rg (default: R_max)""")
    parser.add_argument('--radiation', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if radiation quantities
                        should be included in profile figure.""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if two-temperature 
                        physics was used to produce profiles.""")   
    parser.add_argument('--MBH', type=float, default=10.,
                        required=False,
                        help = """black hole mass in Msun (default: 10)""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if profile plotting
                        should be done on reduced data.""")
    parser.add_argument('--disable_TeX', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""boolean to disable TeX in plotting""")
    parser.add_argument('--plot_title', type=str, required=False, 
                        default='Profiles', 
                        help="""title of PDF output""")
    args = parser.parse_args()
    main(**vars(args))