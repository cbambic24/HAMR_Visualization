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
FS      = 32
plt.rcParams['xtick.labelsize'] = FS
plt.rcParams['ytick.labelsize'] = FS

opt = dict(ha='left', va='center', color='black',
           bbox=dict(facecolor='none', edgecolor='none'))

def main(**kwargs):
    sim            = kwargs.get('sim')
    file_number    = int(kwargs.get('file_number'))
    start_file     = int(kwargs.get('start_file'))
    end_file       = int(kwargs.get('end_file'))
    reduced        = kwargs.get('reduced')
    radiation      = kwargs.get('radiation')
    two_temp       = kwargs.get('two_temp')
    disable_TeX    = kwargs.get('disable_TeX')
    MBH            = kwargs.get('MBH')
    gamma_adi      = kwargs.get('gamma_adi')
    r0             = kwargs.get('r0')
    z_plot         = kwargs.get('z_plot')
    plot_title     = kwargs.get('plot_title')
    b_spherical    = kwargs.get('b_spherical')
    b_only         = kwargs.get('b_only')

    theta_plot = True if not z_plot else False

    matplotlib.rcParams['text.usetex'] = False if disable_TeX else True

    cgs = True if (two_temp or radiation) else False
    Theory = theory.Analytic(hr.CONSTANTS, cgs=cgs)

    Msun = hr.CONSTANTS['Msun']     if cgs else 1.
    c    = hr.CONSTANTS['c']        if cgs else 1.
    G    = hr.CONSTANTS['G']        if cgs else 1.
    Rg   = G * MBH * Msun / (c * c) if cgs else 1.

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

    plot_keys = [
         'rho',         'Temp',          'pressures', 
        'Omega',        'qcool',       'pressure_ratio',
       'velocity',  'magnetic_field',    'energy_flux'
    ]

    plot_info = dict.fromkeys(plot_keys)
    for key in plot_keys:
        plot_info[key] = dict.fromkeys(['quant', 'colors', 'axis_id', 'labels', 'title', 'log', 'linthresh'])

    # Quantities for each figure
    plot_info['rho'][            'quant'] = 'rho'
    plot_info['Temp'][           'quant'] = ['Eion', 'Eelc'] if two_temp else 'Pthm'
    plot_info['pressures'][      'quant'] = ['Pthm', 'Pmag', 'Prad'] if radiation else ['Pthm', 'Pmag']
    plot_info['qcool'][          'quant'] = ['qie', 'qIC', 'qemi', 'qabs']
    plot_info['pressure_ratio'][ 'quant'] = 'Pmag'
    plot_info['Omega'][          'quant'] = ['wuphi', 'wut']
    plot_info['velocity'][       'quant'] = ['wuR', 'wuphi', 'wuz', 'wut']
    plot_info['magnetic_field'][ 'quant'] = ['bR', 'bphi', 'bz']
    plot_info['energy_flux'][    'quant'] = ['THD_t1', 'TEM_t1', 'R_t1'] if radiation else ['THD_t1', 'TEM_t1']

    # Log-scale and linthresh if symlog
    for key in plot_keys: plot_info[key]['log']       = True
    for key in plot_keys: plot_info[key]['linthresh'] = None

    plot_info['velocity']['log'] = False
    plot_info['Omega'][   'log'] = False

    plot_info['magnetic_field']['linthresh'] = 'infer'
    plot_info['energy_flux'][   'linthresh'] = 'infer'

    # Colors for each figure
    for key in plot_keys: plot_info[key]['colors'] = 'black'
    plot_info['Temp'][           'colors'] = ['darkorange', 'dodgerblue'] if two_temp else 'darkorange'
    plot_info['pressures'][      'colors'] = ['darkorange', 'dodgerblue', 'crimson'] if radiation else ['darkorange', 'dodgerblue']
    plot_info['qcool'][          'colors'] = ['darkorange', 'dodgerblue', 'crimson']
    plot_info['pressure_ratio'][ 'colors'] = ['dodgerblue', 'crimson'] if cgs else 'dodgerblue'
    plot_info['velocity'][       'colors'] = ['crimson', 'black', 'dodgerblue']
    plot_info['magnetic_field'][ 'colors'] = ['crimson', 'black', 'dodgerblue']
    plot_info['energy_flux'][    'colors'] = ['darkorange', 'dodgerblue', 'crimson'] if radiation else ['darkorange', 'dodgerblue']

    # Legend labels for each figure
    for key in plot_keys: plot_info[key]['labels'] = None
    plot_info['Temp'][           'labels'] = [r'$\rm{Ions}$', r'$\rm{Electrons}$'] if two_temp else None
    plot_info['pressures'][      'labels'] = [r'$\rm{Thermal}$', r'$\rm{Magnetic}$', r'$\rm{Radiation}$'] if radiation else [r'$\rm{Thermal}$', r'$\rm{Magnetic}$']
    plot_info['qcool'][          'labels'] = [r'$\rm{Coulomb}$', r'$\rm{Compton}$', r'$\rm{Emission}$']
    plot_info['velocity'][       'labels'] = [r'$R$', r'$\varphi$', r'$z$']
    plot_info['magnetic_field'][ 'labels'] = [r'$R$', r'$\varphi$', r'$z$']
    plot_info['energy_flux'][    'labels'] = [r'$-\mathcal{T}^{r}_{t, {\rm HD}}$', r'$-\mathcal{T}^{r}_{t, {\rm EM}}$', r'$-\mathcal{R}^{r}_t$'] if radiation else [r'$-\mathcal{T}^{r}_{t, {\rm HD}}$', r'$-\mathcal{T}^{r}_{t, {\rm EM}}$']

    # Axis ID
    for id, key in enumerate(plot_keys):
        plot_info[key]['axis_id'] = id

    # Limits for each figure
    user_limits = {
        'rho'            : None,
        'Temp'           : [1.e4,  1.e12] if cgs else [1.e-5, 1.],
        'pressures'      : None,
        'Omega'          : [0., 3.],
        'qcool'          : None,
        'pressure_ratio' : None,
        'velocity'       : [-1., 1.],
        'magnetic_field' : None,
        'energy_flux'    : [-1.e2, 1.e2]
    }

    # Legend location for each figure
    legend_loc = {
        'rho'            : False,
        'Temp'           : 'lower left',
        'pressures'      : 'upper left',
        'Omega'          : False,
        'qcool'          : 'lower left',
        'pressure_ratio' : 'lower left',
        'velocity'       : 'lower left',
        'magnetic_field' : 'center left',
        'energy_flux'    : 'lower left'
    }

    # Spherical field components in place of the cylindrical ones
    if b_spherical:
        plot_info['magnetic_field']['quant']  = ['br', 'bth', 'bphi']
        plot_info['magnetic_field']['labels'] = [r'$r$', r'$\theta$', r'$\varphi$']

    # Only the magnetic field panel
    if b_only:
        plot_keys = ['magnetic_field']
        for id, key in enumerate(plot_keys):
            plot_info[key]['axis_id'] = id

    # Construct the necessary keys here from plot_info
    keys = []
    for key in plot_keys:
        quant = plot_info[key]['quant']
        keys += quant if isinstance(quant, list) else [quant]
    keys += ['rho', 'Pmag']     # sigma, for the jet shading
    keys = list(dict.fromkeys(keys))
    keys += ['w']
    if not cgs:
        keys += ['wur', 'wuth', 'wut']

    file_info = {
        'averages_dir' : averages_dir,
        'file_number'  : file_number,
        'start_file'   : start_file,
        'end_file'     : end_file,
        'reduced'      : reduced
    }

    vols = dict.fromkeys(keys)
    for key in keys:
        vols[key] = True
    for key in ['TEM_t1', 'THD_t1']:
        vols[key] = False
    if radiation:
        vols['R_t1'] = False
    vols['Vol'] = False

    Profile = cv.Vertical_Profiles()
    fnums   = Profile.get_fnums(**file_info)
    single_file  = True if (len(fnums) == 1) else False

    prof_args = {
        'reduced'      : reduced, 
        'r0'           : r0,
        'time_average' : False if single_file else True,
        'vols'         : vols
    }

    prof_args['keys'] = keys + ['Vol', 'r', 'theta', 'a'] if single_file else keys + ['Vol']

    if theta_plot:
        profiles = Profile.load_theta_profiles(averages_dir, fnums, **prof_args)
    elif z_plot:
        profiles = Profile.load_z_profiles(averages_dir, fnums, **prof_args)

    fnums0 = f'{fnums[0]}' if single_file else f'{Profile.fnums0}'

    if single_file:
        profiles = profiles[fnums0]
        t_coords = Profile.load_slices(averages_dir, fnums, keys=['t'],
                                        reduced=reduced, show_loading=False)
        profiles['t'] = t_coords[fnums0]['t']

    if z_plot:
        rprof_args = prof_args.copy()
        HoR = calc_HoR(averages_dir, fnums, **rprof_args)
    
    Cart = figs.Cartesian([])
    fig  = plt.figure(figsize = [14.0, 9.0] if b_only else figsize)
    nrow, ncol = (1, 1) if b_only else (3, 3)
    gs   = gridspec.GridSpec(nrow, ncol, figure = fig)
    axes = []
    for i in range(nrow):
        for j in range(ncol):
            axes.append(fig.add_subplot(gs[i, j]))

    x     = profiles['theta'] if theta_plot else profiles['z_r'] / HoR
    Pmag  = profiles['Pmag']
    rho   = profiles['rho']
    a     = profiles['a']
    sigma = 2. * Pmag / (rho * c * c)

    if theta_plot:
        theta_min, theta_max = get_jet_bounds(x, sigma, sigma_cut=1.)
    elif z_plot:
        z_min, z_max = get_jet_bounds(x, sigma, sigma_cut=1, z=True)

    axes_ids = {key: plot_info[key]['axis_id'] for key in plot_keys}

    plot_kwargs = {
        'cgs' : cgs, 'radiation' : radiation, 'two_temp' : two_temp,
        'c' : c, 'r0' : r0, 'Rg' : Rg, 'MBH' : MBH, 'a' : a,
        'kB' : kB, 'mp' : mp, 'mu_i' : mu_i, 'mu_e' : mu_e,
        'gamma_adi' : gamma_adi, 'axes_ids' : axes_ids, 'Analytic' : Theory,
        'single_file' : single_file, 'b_spherical' : b_spherical
    }

    for key in plot_keys:
        id     = plot_info[key]['axis_id']
        quant0 = plot_info[key]['quant']
        quant0 = quant0[0] if isinstance(quant0, list) else quant0
        vol    = vols[quant0]
        Volume = profiles['Vol'] if vol else 1.

        if single_file:
            def fetch(quant, _Volume=Volume):
                return profiles[quant] / _Volume
        else:
            def fetch(quant):
                return profiles[quant]

        plotted_data = PLOT_FUNCS[key](axes, id, x, fetch, colors=plot_info[key]['colors'], **plot_kwargs)

        axes[id].set_ylabel(get_plot_title(key, cgs), fontsize=FS)

        if plot_info[key]['linthresh']:
            linthresh = get_linthresh(plot_info[key]['linthresh'], plotted_data)
            axes[id].set_yscale('symlog', linthresh=linthresh)
        elif plot_info[key]['log']:
            axes[id].set_yscale('log')

        limits = cv.get_limits(key, user_limits, plotted_data, plot_info)
        axes[id].set_ylim(limits)

        if legend_loc[key]:
            axes[id].legend(loc=legend_loc[key], fontsize=FS-4, frameon=False)

    for ax in axes:
        if theta_plot:
            xticklabels = [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
            ax.set_xlabel(r'$\theta$', fontsize=FS)
            ax.set_xticks([0., (1./4.)*np.pi, (1./2.)*np.pi, (3./4.)*np.pi, np.pi])
            ax.set_xticklabels(xticklabels, fontsize=FS)
        elif z_plot:
            ax.set_xlabel(r'$z/H_{\rho}$', fontsize=FS)
            ax.set_xlim([-10, 10])
            #ax.set_xlim([-6, 6])
        cv.grid(ax, alpha=0.2, color='black')

        ymin, ymax = ax.get_ylim()
        if theta_plot:
            ax.fill_between(x, ymin, ymax, where=x < theta_min, facecolor='gray', alpha=0.2)
            ax.fill_between(x, ymin, ymax, where=x > theta_max, facecolor='gray', alpha=0.2)
        elif z_plot:
            ax.fill_between(x, ymin, ymax, where=x < z_min, facecolor='gray', alpha=0.2)
            ax.fill_between(x, ymin, ymax, where=x > z_max, facecolor='gray', alpha=0.2)

    tstring = get_tstring(profiles, single_file, Profile, ndec=3)
    axes[0].text(0.05, 0.10, r'$t = $' + tstring + r' $R_g/c$', fontsize=FS-2,
        transform=axes[0].transAxes, **opt)

    print("Saving figure ...", flush=True)
    fig.tight_layout()
    plt.savefig(f'{save_dir}/{plot_title}.png', format='png')
    plt.close()

def calc_HoR(averages_dir, fnums, **kwargs):
    r0 = kwargs.get('r0')
    Radial_Profile = cv.Profiles()
    rprof_args     = kwargs.copy()
    rprof_args['keys'] = ['Dtheta2', 'Vol', 'r', 't', 'a']
    rprofiles = Radial_Profile.load_profiles(averages_dir, fnums, **rprof_args)
    Radial_Profile.check_AMR(fnums, rprofiles)
    HoR_prof  = np.sqrt(Radial_Profile.time_avg('Dtheta2', rprofiles, vol=True))
    r_prof    = rprofiles[f'{Radial_Profile.fnums0}']['r']
    idx       = np.argmin(np.abs(r_prof - r0))
    HoR       = HoR_prof[idx]
    return HoR

def plot_rho(axes, id, x, fetch, **kwargs):
    plt_args = dict(lw=2., linestyle='-')
    colors = kwargs['colors']
    rho = fetch('rho')
    axes[id].plot(x, rho, color=colors, **plt_args)
    return [rho]

def plot_Temp(axes, id, x, fetch, **kwargs):
    plt_args  = dict(lw=2., linestyle='-')
    two_temp  = kwargs['two_temp']
    radiation = kwargs['radiation']
    colors    = kwargs['colors']

    if two_temp:
        kB, mp, mu_i, mu_e = kwargs['kB'], kwargs['mp'], kwargs['mu_i'], kwargs['mu_e']
        gamma = kwargs['gamma_adi']
        Eion = fetch('Eion')
        Eelc = fetch('Eelc')
        rho  = fetch('rho')
        Ti = (gamma-1.) * (1./kB) * Eion / (rho / (mu_i*mp))
        Te = (gamma-1.) * (1./kB) * Eelc / (rho / (mu_e*mp))
        axes[id].plot(x, Ti, color=colors[0], label=r'$\rm{Ions}$',      **plt_args)
        axes[id].plot(x, Te, color=colors[1], label=r'$\rm{Electrons}$', **plt_args)
        plotted_data = [Ti, Te]
    else:
        Pthm = fetch('Pthm')
        rho  = fetch('rho')
        Tgas = Pthm / rho
        axes[id].plot(x, Tgas, color=colors, label=r'$\rm{Gas}$', **plt_args)
        plotted_data = [Tgas]

    if radiation:
        Prad = fetch('Prad')
        Tr = (3. * Prad / hr.CONSTANTS['aRAD']) ** (1./4.)
        axes[id].plot(x, Tr, color='crimson', label=r'$\rm{Radiation}$', **plt_args)
        plotted_data += [Tr]

    return plotted_data

def plot_pressures(axes, id, x, fetch, **kwargs):
    plt_args  = dict(lw=2., linestyle='-')
    radiation = kwargs['radiation']
    colors    = kwargs['colors']

    Pthm = fetch('Pthm')
    Pmag = fetch('Pmag')
    axes[id].plot(x, Pthm, color=colors[0], label=r'$\rm{Thermal}$',  **plt_args)
    axes[id].plot(x, Pmag, color=colors[1], label=r'$\rm{Magnetic}$', **plt_args)
    plotted_data = [Pthm, Pmag]
    if radiation:
        Prad = fetch('Prad')
        axes[id].plot(x, Prad, color=colors[2], label=r'$\rm{Radiation}$', **plt_args)
        plotted_data += [Prad]
    return plotted_data

def plot_pressure_ratio(axes, id, x, fetch, **kwargs):
    plt_args  = dict(lw=2., linestyle='-')
    radiation = kwargs['radiation']
    colors    = kwargs['colors']
    color0    = colors[0] if isinstance(colors, list) else colors

    Pthm = fetch('Pthm')
    Pmag = fetch('Pmag')
    if radiation:
        Prad = fetch('Prad')
        Pmag_Ptot  = Pmag / (Pthm + Prad)
        ptot_label = r'$P_{\rm mag}/(P_{\rm rad} + P_{\rm thm})$'
    else:
        Pmag_Ptot  = Pmag / Pthm
        ptot_label = r'$P_{\rm mag}/P_{\rm thm}$'
    axes[id].plot(x, Pmag_Ptot, color=color0, label=ptot_label, **plt_args)
    plotted_data = [Pmag_Ptot]
    if radiation:
        Prad_Pthm = Prad / Pthm
        axes[id].plot(x, Prad_Pthm, color=colors[1], label=r'$P_{\rm rad}/P_{\rm thm}$', **plt_args)
        plotted_data += [Prad_Pthm]
    return plotted_data

def plot_velocity(axes, id, x, fetch, **kwargs):
    plt_args = dict(lw=2., linestyle='-')
    cgs, axes_ids, c = kwargs['cgs'], kwargs['axes_ids'], kwargs['c']
    colors = kwargs['colors']

    wuR   = fetch('wuR')
    wuphi = fetch('wuphi')
    wuz   = fetch('wuz')
    wut   = fetch('wut')
    vR    = wuR   / wut
    vphi  = wuphi / wut
    vz    = wuz   / wut
    axes[id].plot(x, vR,   color=colors[0], label=r'$R$',       **plt_args)
    axes[id].plot(x, vphi, color=colors[1], label=r'$\varphi$', **plt_args)
    axes[id].plot(x, vz,   color=colors[2], label=r'$z$',       **plt_args)
    if not cgs:
        vr  = fetch('wur')  / wut
        vth = fetch('wuth') / wut
        # No cooling without CGS units so use for spherical velocities
        sph_id = int(axes_ids['qcool'])
        axes[sph_id].plot(x, vr/c,   color='crimson',    label=r'$r$',       **plt_args)
        axes[sph_id].plot(x, vphi/c, color='black',      label=r'$\varphi$', **plt_args)
        axes[sph_id].plot(x, vth/c,  color='dodgerblue', label=r'$\theta$',  **plt_args)
    return [vR, vphi, vz]

def plot_Omega(axes, id, x, fetch, **kwargs):
    plt_args = dict(lw=2., linestyle='-')
    r0, Rg, MBH = kwargs['r0'], kwargs['Rg'], kwargs['MBH'] 
    a, c        = kwargs['a'], kwargs['c']
    Analytic = kwargs['Analytic']
    colors = kwargs['colors']

    wuphi  = fetch('wuphi')
    wut    = fetch('wut')
    vphi   = wuphi / wut
    Omega  = c * vphi / (r0 * Rg)
    Omega0 = Analytic.calc_OmegaK(MBH, a, r0)
    axes[id].plot(x, Omega/Omega0, color=colors, **plt_args)
    return [Omega/Omega0]

def plot_qcool(axes, id, x, fetch, **kwargs):
    plt_args = dict(lw=2., linestyle='-')
    cgs, MBH, radiation, single_file = kwargs['cgs'], kwargs['MBH'], kwargs['radiation'], kwargs['single_file']
    colors = kwargs['colors']

    if cgs:
        Analytic = theory.Analytic(hr.CONSTANTS, cgs=True)
        Msun  = hr.CONSTANTS['Msun']
        c     = hr.CONSTANTS['c']
        G     = hr.CONSTANTS['G']
        Rg    = G * MBH * Msun / (c * c)
        L_Edd = Analytic.calc_LEdd(MBH=MBH)
        norm  = -(Rg * Rg * Rg / L_Edd)
    else:
        norm = 1.

    qie = norm * fetch('qie')
    axes[id].plot(x, qie, color=colors[0], label=r'$\rm{Coulomb}$', **plt_args)
    plotted_data = [qie]
    if radiation:
        qIC  = norm * fetch('qIC')
        qemi = norm * fetch('qemi')
        axes[id].plot(x, qIC,  color=colors[1], label=r'$\rm{Compton}$',  **plt_args)
        axes[id].plot(x, qemi, color=colors[2], label=r'$\rm{Emission}$', **plt_args)
        plotted_data += [qIC, qemi]
        if single_file:
            qabs = norm * fetch('qabs')
            axes[id].plot(x, qabs, color='purple', label=r'$\rm{Absorption}$', **plt_args)
            plotted_data += [qabs]
    return plotted_data

def plot_magnetic_field(axes, id, x, fetch, **kwargs):
    plt_args = dict(lw=2., linestyle='-')
    colors = kwargs['colors']
    if kwargs['b_spherical']:
        br   = fetch('br')
        bth  = fetch('bth')
        bphi = fetch('bphi')
        axes[id].plot(x, br,   color=colors[0], label=r'$r$',       **plt_args)
        axes[id].plot(x, bth,  color=colors[1], label=r'$\theta$',  **plt_args)
        axes[id].plot(x, bphi, color=colors[2], label=r'$\varphi$', **plt_args)
        return [br, bth, bphi]
    bR   = fetch('bR')
    bphi = fetch('bphi')
    bz   = fetch('bz')
    axes[id].plot(x, bR,   color=colors[0], label=r'$R$',       **plt_args)
    axes[id].plot(x, bphi, color=colors[1], label=r'$\varphi$', **plt_args)
    axes[id].plot(x, bz,   color=colors[2], label=r'$z$',       **plt_args)
    return [bR, bphi, bz]

def plot_energy_flux(axes, id, x, fetch, **kwargs):
    plt_args = dict(lw=2., linestyle='-')
    cgs, MBH, radiation = kwargs['cgs'], kwargs['MBH'], kwargs['radiation']
    colors = kwargs['colors']

    if cgs:
        Analytic = theory.Analytic(hr.CONSTANTS, cgs=True)
        Msun  = hr.CONSTANTS['Msun']
        c     = hr.CONSTANTS['c']
        G     = hr.CONSTANTS['G']
        Rg    = G * MBH * Msun / (c * c)
        L_Edd = Analytic.calc_LEdd(MBH=MBH)
        norm  = -Rg * Rg * c / L_Edd
    else:
        norm = 1.

    TEM_tr = norm * fetch('TEM_t1')
    THD_tr = norm * fetch('THD_t1')
    axes[id].plot(x, THD_tr, color=colors[0], label=r'$-\mathcal{T}^{r}_{t, {\rm HD}}$', **plt_args)
    axes[id].plot(x, TEM_tr, color=colors[1], label=r'$-\mathcal{T}^{r}_{t, {\rm EM}}$', **plt_args)
    plotted_data = [THD_tr, TEM_tr]
    if radiation:
        R_tr = norm * fetch('R_t1')
        axes[id].plot(x, R_tr, color=colors[2], label=r'$-\mathcal{R}^{r}_t$', **plt_args)
        plotted_data += [R_tr]
    return plotted_data

PLOT_FUNCS = {
    'rho'            : plot_rho,
    'Temp'           : plot_Temp,
    'pressures'      : plot_pressures,
    'pressure_ratio' : plot_pressure_ratio,
    'Omega'          : plot_Omega,
    'qcool'          : plot_qcool,
    'velocity'       : plot_velocity,
    'magnetic_field' : plot_magnetic_field,
    'energy_flux'    : plot_energy_flux,
}

def get_plot_title(key, cgs):
    plot_titles = {
        'rho'            : r'$\rho \: (\rm{g/cm}^3)$' if cgs else r'$\rho / \rho_0$',
        'Temp'           : r'$\rm{Temperature} \: (\rm{K})$' if cgs else r'${\rm{Temperature}} \: (m_i c^2)$',
        'pressures'      : r'$\rm{Pressure} \: (\rm{erg/cm}^3)$' if cgs else r'${\rm{Pressure}} \: (\rho_0 c^2)$',
        'qcool'          : r'$\mathcal{Q} R_g^3 / L_{\rm Edd}$',
        'pressure_ratio' : r'$\rm{Pressure} \: \rm{Ratio}$',
        'Omega'          : r'$\Omega / \Omega_{\rm K}$',
        'velocity'       : r'$\langle w u^{\mu} \rangle_{\varphi} / \langle w u^t \rangle_{\varphi}$',
        'magnetic_field' : r'$\rm{Magnetic} \: \rm{Field} \: (\rm{G})$' if cgs else r'${\rm{Magnetic}} \: {\rm{Field}} \: (B_0)$',
        'energy_flux'    : r'$\rm{Energy} \: \rm{Flux} \: (L_{\rm Edd})$' if cgs else r'${\rm{Energy}} \: {\rm{Flux}} \: (\rho_0 c^3 R_g^2)$'
    }
    return plot_titles[key]

def get_linthresh(linthresh, quants):
    if linthresh == 'infer':
        all_data   = np.concatenate(quants)
        log_thresh = np.floor(np.amax(np.log10(np.abs(all_data)))) - 2
        linthresh  = 10 ** log_thresh
    return linthresh

def get_jet_bounds(x, sigma, **kwargs):
    sigma_cut = kwargs.get('sigma_cut', 1.)
    x_j = x[sigma > sigma_cut]
    if (len(x_j) == 0):
        return x[0], x[-1]
    if kwargs.get('z', False):
        return x_j[x_j < 0.][-1], x_j[x_j > 0.][0]
    else:
        return x_j[x_j < np.pi/2.][-1], x_j[x_j > np.pi/2.][0]

def get_tstring(profiles, single_file, Profile, ndec=2):
    if single_file:
        return cv.get_nice_time(profiles['t'], prec=ndec)

    tstrt, tend = Profile.t[0], Profile.t[-1]
    coeff_strt, oom_strt = cv.get_time_coeff_expon(tstrt)
    coeff_end,  oom_end  = cv.get_time_coeff_expon(tend)

    if oom_strt == oom_end:
        return (rf'$%.{ndec}f-%.{ndec}f$' % (coeff_strt, coeff_end) +
            r'$\times$' + r'$10^{%d}$' % oom_end)
    return cv.get_nice_time(tstrt, prec=ndec) + '-' + cv.get_nice_time(tend, prec=ndec)

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Plot vertical profiles.""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('--file_number', type=int, required=False, 
                        default=-1, help = """dump number of profile plot""")
    parser.add_argument('--start_file', type=int, required=False,
                        default=-1, help = """dump number
                        of start file in profile average/evolution""")
    parser.add_argument('--end_file', type=int, required=False,
                        default=-1, help = """dump number
                        of end file in profile average/evolution""")
    parser.add_argument('--r0', type=float, default=10.,
                        required=False,
                        help = """measurement radius in Rg (default: 10)""")
    parser.add_argument('--radiation', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="""boolean to determine if radiation quantities
                        should be included in profile extraction.""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="""boolean to determine if two-temperature
                        physics should be included in slice extraction.""")
    parser.add_argument('--MBH', type=float, default=10.,
                        required=False,
                        help = """black hole mass in Msun (default: 10)""")
    parser.add_argument('--gamma_adi', type=float, default=5./3.,
                        required=False,
                        help = """adiabatic index gamma""")
    parser.add_argument('--z_plot', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="""boolean to determine if x-axis should be
                        z/R or theta (default).""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if profile extraction
                        should be done on reduced data.""")
    parser.add_argument('--disable_TeX', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="""boolean to disable TeX in plotting""")
    parser.add_argument('--b_spherical', action='store_true', required=False,
                        help="""plot B^r, B^theta, B^phi in the magnetic
                        field panel instead of the cylindrical components.""")
    parser.add_argument('--b_only', action='store_true', required=False,
                        help="""plot only the magnetic field panel.""")
    parser.add_argument('--plot_title', type=str, required=False,
                        default='Vertical_Profiles',
                        help="""title of PDF output""")
    args = parser.parse_args()
    main(**vars(args))