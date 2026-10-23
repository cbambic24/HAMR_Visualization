import sys
import os
import argparse
import numpy as np
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{file_dir}')
asys_path = os.path.join(os.path.dirname(file_dir), "Analyze_HAMR_Data")
sys.path.insert(0, f'{asys_path}')
import compute_variables as cv
import hamr_read as hr
import theory as theory
import figs
import efficiencies_timescales as eft

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.style as style
import cmasher as cmr # colormaps I like
matplotlib.rcParams['text.usetex'] = True
style.use('seaborn-v0_8-bright')
matplotlib.rcParams['figure.dpi'] = 300

def main(**kwargs):
    sim       = kwargs.get('sim')
    reduced   = kwargs.get('reduced')
    overwrite = kwargs.get('overwrite')
    plot_data = kwargs.get('plot_data')
    
    averages_dir = f'{cv.user_path}/Averages/{sim}'
    mdot_dir = cv.get_save_dir(sim, 'Averages', 'Mass_Fluxes',   reduced=reduced)
    edot_dir = cv.get_save_dir(sim, 'Averages', 'Energy_Fluxes', reduced=reduced)
    time_dir = cv.get_save_dir(sim, 'Averages', 'Timescales',    reduced=reduced)

def plot_data(sim, Mdot, Edot, Timescales, **kwargs):
    movie_title     = kwargs.get('movie_title')
    fluxes_only     = kwargs.get('fluxes_only', False)
    timescales_only = kwargs.get('timescales_only', False)
    show_2D         = kwargs.get('show_2D', True)
    if fluxes_only or timescales_only: show_2D = False
    radiation = kwargs.get('radiation')
    two_temp  = kwargs.get('two_temp')
    MBH       = kwargs.get('MBH')
    Theory    = theory.Analytic(hr.CONSTANTS, cgs=True)
    L_Edd     = Theory.calc_LEdd(MBH)
    Mdot_Edd  = Theory.calc_Mdot_Edd(MBH, eta=0.179)
    
    # Define constants
    cgs = True if (two_temp or radiation) else False
    if (two_temp or radiation):
        Msun = hr.CONSTANTS['Msun']
        c    = hr.CONSTANTS['c']
        G    = hr.CONSTANTS['G']
        Rg   = G * MBH * Msun / (c * c)
    else:
        c, G, Rg = 1., 1., 1.
    
    units = {
        'L_Edd'    : L_Edd,
        'Mdot_Edd' : Mdot_Edd,
        'Rg'       : Rg,
        'c'        : c
    }
 
    fig_info = {
        'fluxes_only'     : fluxes_only,
        'timescales_only' : timescales_only,
        'show_2D'         : show_2D
    }

    meta = get_figure_meta(**fig_info)
    fig  = plt.figure(figsize = meta['figsize'])
    gs   = gridspec.GridSpec(*meta['dims'], figure = fig)
    plt.rcParams['xtick.labelsize'] = meta['fontsize']
    plt.rcParams['ytick.labelsize'] = meta['fontsize']
    if show_2D:
        axes = []
        for i in range(4):
            axes.append(fig.add_subplot(gs[0, i]))
        eax = fig.add_subplot(gs[1, :])
        tax = fig.add_subplot(gs[2, :])
    else:
        if fluxes_only:
            eax = fig.add_subplot(gs[0, 0])
        if timescales_only:
            tax = fig.add_subplot(gs[0, 0])

    if show_2D:
        plot_2D(fig, gs, axes, units, fontsize=meta['fontsize'], **kwargs)

    if fluxes_only or show_2D:
        plot_1D('Mdot', Mdot, fig, gs, eax, units, fontsize=meta['fontsize'], **kwargs)
        plot_1D('Edot', Edot, fig, gs, eax, units, fontsize=meta['fontsize'], Mdot=Mdot, **kwargs)
    if timescales_only or show_2D:
        plot_1D('Timescales', Timescales, fig, gs, tax, units, fontsize=meta['fontsize'], **kwargs)

    print("Saving figure ...", flush=True)
    fig.tight_layout()
    movie_dir = cv.get_save_dir(sim, 'Images', f'{movie_title}', subtype='QuickView', reduced=kwargs.get('reduced'))
    os.makedirs(movie_dir, exist_ok=True)
    fnum_str = kwargs.get('fnum_str')
    fname    = f'{movie_title}_{fnum_str}' if fnum_str else movie_title
    plt.savefig(f'{movie_dir}/{fname}.png', format='png')
    plt.close()

def plot_1D(plot_type, data, fig, gs, ax, units, **kwargs):
    slices      = kwargs.get('slices')
    low_Mdot    = kwargs.get('low_Mdot')
    hi_res      = kwargs.get('hi_res')
    FS          = kwargs.get('fontsize')

    colors = {
        'blue'    : "#0077BB",
        'cyan'    : "#33BBEE",
        'orange'  : "#EE7733",
        'red'     : "#CC3311",
        'magenta' : "#EE3377",
        'teal'    : "#009988",
        'grey'    : "#BBBBBB"
    }

    if plot_type == 'Mdot':
        # Mass fluxes
        Mdot_Edd   = units['Mdot_Edd']
        acc_label  = r'$\rm{Accretion}$'
        wind_label = r'$\rm{Wind}$'
        ax.plot(data['r'], data['Acc']/Mdot_Edd,  label=acc_label,  color='black', lw=2)
        ax.plot(data['r'], data['Wind']/Mdot_Edd, label=wind_label, color='black', lw=2, linestyle='--')
    
    if plot_type == 'Edot':
        Mdot  = kwargs.get('Mdot')
        # Energy fluxes
        L_Edd = units['L_Edd']
        c     = units['c']
        
        jet_label            = r'$\rm{Jet}$'
        wind_enthalpy_label  = r'$\rm{Wind} \: \rm{Thermal}$'
        wind_poynting_label  = r'$\rm{Wind} \: \rm{Poynting}$'
        wind_total_label     = r'$\rm{Wind} \: \rm{Total}$'
        radiation_label      = r'$\rm{Radiation}$'
        radiation_wind_label = r'$\rm{Radiation} \: \rm{carried} \: \rm{by} \: \rm{Wind}$'
        radiation_disk_label = r'$\rm{Radiation} \: \rm{from} \: \rm{Photosphere}$'
        
        ax.plot(data['r'], data['Jet']/L_Edd,           label=jet_label,           color=colors['orange'], lw=2)
        Edot_Wind_Thermal = data['Wind_Enthalpy'] - Mdot['Wind'] * c * c
        ax.plot(data['r'], Edot_Wind_Thermal/L_Edd,     label=wind_enthalpy_label, color=colors['magenta'],lw=2)
        ax.plot(data['r'], data['Wind_Poynting']/L_Edd, label=wind_poynting_label, color=colors['cyan'],   lw=2)
        Edot_Wind_Total = data['Wind_Enthalpy'] + data['Wind_Poynting']
        ax.plot(data['r'], Edot_Wind_Total/L_Edd,       label=wind_total_label,    color=colors['grey'],   lw=2)
        ax.plot(data['r'], data['Radiation']/L_Edd,     label=radiation_label,     color=colors['red'],    lw=2)
        
        to_infinity = (data['r'] > 3.)
        Edot_Rad_Disk = np.cumsum(data['Radiation_Flux_Disk'] * to_infinity)
        ax.plot(data['r'], Edot_Rad_Disk/L_Edd,          label=radiation_disk_label, color=colors['teal'], lw=2)
        ax.plot(data['r'], data['Radiation_Wind']/L_Edd, label=radiation_wind_label, color=colors['blue'], lw=2)

        ax.set_ylim([1.e-9, 1.e-2]) if low_Mdot else ax.set_ylim([1.e-3, 1.e3])
        ax.set_ylabel(r'${\rm{Flux}} \: (\dot{M}_{\rm Edd} \: {\rm{or}} \: L_{\rm Edd})$', fontsize=FS)

    if plot_type == 'Timescales':
        inf_label    = r'$\rm{Inflow}$'
        diff_label   = r'$\rm{Diffusion}$'
        approx_label = r'$\rm{Diffusion} \: \rm{Approximation}$'
        cool_label   = r'$\rm{Cooling}$'
        thm_label    = r'$\rm{Thermal}$'
        Coul_label   = r'$\rm{Coulomb} \: \rm{Collisions}$'
        Compt_label  = r'$\rm{Inverse} \: \rm{Compton}$'

        t_inf  = data['Inflow']
        t_diff = data['Diffusion']
        t_cool = data['Cooling']
        t_ie   = data['Coulomb']
        t_IC   = data['Compton']
        t_thm  = data['Thermal']
        t_diff_approx = data['Diffusion_Approximation']

        # Timescale profiles
        ax.plot(data['r'], t_inf,         label=inf_label,    color='black',          lw=2)
        ax.plot(data['r'], t_diff,        label=diff_label,   color=colors['red'],    lw=2)
        ax.plot(data['r'], t_cool,        label=cool_label,   color=colors['cyan'],   lw=2)
        ax.plot(data['r'], t_ie,          label=Coul_label,   color=colors['orange'], lw=2)
        ax.plot(data['r'], t_IC,          label=Compt_label,  color=colors['blue'],   lw=2)
        ax.plot(data['r'], t_thm,         label=thm_label,    color=colors['teal'],   lw=2)
        ax.plot(data['r'], t_diff_approx, label=approx_label, color=colors['grey'],   lw=2)
        
        ax.set_ylim([1.e-4, 1.e8]) if low_Mdot else ax.set_ylim([1.e-3, 1.e4])
        ax.set_ylim([1.e-2, 1.e3])
        ax.set_ylabel(r'$\rm{Timescale} \: (\rm{s})$', fontsize=FS)

    cv.grid(ax, color='black', alpha=0.3)
    ax.set_xlim([1., 100.])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=FS-4, frameon=False, ncols=2)
    ax.set_xlabel(r'$r / R_g$', fontsize=FS)

def plot_2D(fig, gs, axes, units, **kwargs):
    slices   = kwargs.get('slices')
    extra    = kwargs.get('extra')
    disk     = kwargs.get('disk')
    tau_es   = kwargs.get('tau_es')
    Be       = kwargs.get('Be')
    MBH      = kwargs.get('MBH')
    cyl      = kwargs.get('use_cyl')
    low_Mdot = kwargs.get('low_Mdot')
    hi_res   = kwargs.get('hi_res')
    fontsize = kwargs.get('fontsize')
    if slices is None or extra is None:
        raise ValueError("Cannot plot 2D quantities without slices and extra.")
    
    plt_args = {
        'log' : True,
        'sym' : False,
        'FS'  : fontsize
    }
    
    vec_args = {
        'dens'               : 4.0,
        'linewidth'          : 0.6,
        'color'              : 'black',
        'broken_streamlines' : True,
        'arrowsize'          : 1.0
    }
    if hi_res:
        vec_args['dens'] = 10.

    if cyl:
        z_photo = eft.calc_surface(tau_es, slices, extra, photosphere=True,  Bernoulli=False)
        z_bound = eft.calc_surface(Be,     slices, extra, photosphere=False, Bernoulli=True)
    else:
        th_photo = eft.calc_surface(tau_es, slices, extra, photosphere=True,  Bernoulli=False)
        th_bound = eft.calc_surface(Be,     slices, extra, photosphere=False, Bernoulli=True)
        
    R2D, Z2D, tau_cyl = eft.interpolate_sph_to_cyl(tau_es, slices, extra)
    
    # Normalization for cooling rate
    L_Edd  = units['L_Edd']
    Rg     = units['Rg']
    qcool0 = L_Edd / (Rg * Rg * Rg)
    rho0   = extra['rho0']
    rho    = slices['rho'] / slices['Vol']

    if cyl: 
        q_emi  = -slices['qemi'] / slices['Vol']
        q_abs  = -slices['qabs'] / slices['Vol']
        q_IC   = -slices['qIC']  / slices['Vol']
        q_cool = q_IC + q_emi - q_abs
        if low_Mdot:
            q_ie   = -slices['qie'] / slices['Vol']
            q_cool = q_ie
        _, _, rho_cyl   = eft.interpolate_sph_to_cyl(rho,    slices, extra) 
        _, _, tau_cyl   = eft.interpolate_sph_to_cyl(tau_es, slices, extra) 
        _, _, Be_cyl    = eft.interpolate_sph_to_cyl(    Be, slices, extra) 
        _, _, qcool_cyl = eft.interpolate_sph_to_cyl(q_cool, slices, extra) 
         
        R         = {'r' : R2D}
        Z         = {'r' : Z2D}
        rho_plt   = {'r' : (rho_cyl[:, :, 0] / rho0) * disk}
        tau_plt   = {'r' : tau_cyl[:, :, 0]} 
        Be_plt    = {'r' : Be_cyl[:, :, 0]}
        qcool_plt = {'r' : qcool_cyl[:, :, 0] / qcool0}
    else:
        # Factor of sqrt(-g) maintained (g) for integration
        gq_emi  = -slices['qemi']
        gq_abs  = -slices['qabs']
        gq_IC   = -slices['qIC']
        gq_cool = gq_IC + gq_emi - gq_abs
        if low_Mdot:
            gq_ie   = -slices['qie']
            gq_cool = gq_ie
        R = {'r' : slices['r'][:, None] * np.sin(slices['theta'][None, :])}
        Z = {'r' : slices['r'][:, None] * np.cos(slices['theta'][None, :])}
        rho_plt   = {'r' : (rho / rho0) * disk}
        tau_plt   = {'r' : tau_es} 
        Be_plt    = {'r' : Be}
        qcool_plt = {'r' : (gq_cool / slices['Vol']) / qcool0}

    # Information for vector plots
    R1D = np.linspace(0., 240, 100)
    z1D = np.linspace(-170, 170, 100)
    if hi_res:
        R1D = np.linspace(0., 100, 100)
        z1D = np.linspace(-100, 100, 100)
    R_uni, Z_uni = np.meshgrid(R1D, z1D)
    vec_args['R_uni'] = R_uni
    vec_args['Z_uni'] = Z_uni
            
    if cyl:
        # Velocity vectors
        w   = (slices['w']   / slices['Vol'])
        wuR = (slices['wuR'] / slices['Vol'])
        wuz = (slices['wuz'] / slices['Vol'])
        # Radiative energy flux vectors
        R_tR = (slices['R_tR'] / slices['Vol'])
        R_tz = (slices['R_tz'] / slices['Vol']) 
        # Velocity vectors in cylindrical coordinates 
        _, _, w_cyl   = eft.interpolate_sph_to_cyl(w,   slices, extra)
        _, _, wuR_cyl = eft.interpolate_sph_to_cyl(wuR, slices, extra)
        _, _, wuz_cyl = eft.interpolate_sph_to_cyl(wuz, slices, extra)
        uR, uz = (wuR_cyl / w_cyl)[:, :, 0], (wuz_cyl / w_cyl)[:, :, 0]
        # Radiative fluxes in cylindrical coordinates
        _, _, R_tR_cyl = eft.interpolate_sph_to_cyl(R_tR, slices, extra)
        _, _, R_tz_cyl = eft.interpolate_sph_to_cyl(R_tz, slices, extra)
    else:
        # Factor of sqrt(-g) maintained (g) for integration
        # Velocity vectors in spherical coordinates
        gw   = slices['w']
        gwur = slices['wur']
        gwuR = slices['wuR']
        gwuz = slices['wuz']
        # Radiative energy flux vectors in spherical coordinates
        R_tR = slices['R_tR'] / slices['Vol']
        R_tz = slices['R_tz'] / slices['Vol']
        uR, uz = gwuR / gw, gwuz / gw

    Cart = figs.Cartesian([])
    plt_args['log']       = True
    plt_args['sym']       = False
    plt_args['linthresh'] = False
    plt_args['cmap']      = cmr.torch_r
    plt_args['lim']       = [1.e-6, 1.e3]
    plt_args['extend']    = 'min' 

    vec_args['R']  = R
    vec_args['Z']  = Z
    vec_args['VR'] = {'r' : -R_tR_cyl} if cyl else {'r' : -R_tR}  
    vec_args['VZ'] = {'r' : -R_tz_cyl} if cyl else {'r' : -R_tz}
    
    Cart.plot_Rz(fig, axes[0], R, Z, rho_plt, **plt_args) 
    Cart.plot_Rz_vec(fig, axes[0], **vec_args)

    if cyl:
        plot_surfaces(axes[0], z_photo, z_bound, **kwargs)
    else: 
        plot_surfaces(axes[0], th_photo, th_bound, **kwargs)
    
    plt_args['cmap']   = cmr.prinsenvlag_r
    plt_args['lim']    = [1.e-3, 1.e3]
    plt_args['extend'] = 'both' 
    Cart.plot_Rz(fig, axes[1], R, Z, tau_plt, **plt_args) 
    Cart.plot_Rz_vec(fig, axes[1], **vec_args) 
    if cyl:
        plot_surfaces(axes[1], z_photo, z_bound, only_photo=True, **kwargs)
    else: 
        plot_surfaces(axes[1], th_photo, th_bound, only_photo=True, **kwargs) 
     
    plt_args['log']       = True
    plt_args['sym']       = True
    plt_args['lim']       = [-1., 1.]
    plt_args['linthresh'] = 1.e-3 
    
    vec_args['R']  = R
    vec_args['Z']  = Z
    vec_args['VR'] = {'r' : uR}
    vec_args['VZ'] = {'r' : uz}
    
    Cart.plot_Rz(fig, axes[2], R, Z, Be_plt, **plt_args) 
    Cart.plot_Rz_vec(fig, axes[2], **vec_args)

    plt_args['log']       = True
    plt_args['sym']       = False
    plt_args['linthresh'] = False
    plt_args['cmap']      = cmr.torch
    plt_args['lim']       = [1.e-14, 1.e-7] if low_Mdot else [1.e-8, 1.e-1]
    plt_args['extend']    = 'min' 
    Cart.plot_Rz(fig, axes[3], R, Z, qcool_plt, **plt_args) 

    labels = [
        r'$\rho/\rho_0$', 
        r'$\tau_{\rm es}$',
        r'$\rm{Be}$',
        r'$q_{\rm cool} R_g^3 / L_{\rm Edd}$',
    ]
    if low_Mdot:
        labels[-1] = r'$q_{ie} R_g^3 / L_{\rm Edd}$'

    opt = dict(ha='right', va='center', color='white',
        bbox=dict(facecolor='none', edgecolor='none'))

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.set_xlim([0, 200])
        ax.set_ylim([-100, 100])
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x \: (R_g)$', fontsize=plt_args['FS'])
        ax.set_ylabel(r'$z \: (R_g)$', fontsize=plt_args['FS'])
        opt['color'] = 'white' if i == 3 else 'black'
        ax.text(0.95, 0.90, label, fontsize=plt_args['FS']+2, 
                transform=ax.transAxes, **opt)    
        if (hi_res):
            #ax.set_xlim([0, 100])
            #ax.set_ylim([-50, 50])
            ax.set_xlim([0, 40])
            ax.set_ylim([-20, 20])

def plot_surfaces(ax, photo, bound, **kwargs):
    slices     = kwargs.get('slices')
    surface    = kwargs.get('surface')
    cyl        = kwargs.get('use_cyl')
    opt_thin   = kwargs.get('opt_thin')
    only_photo = kwargs.get('only_photo')
    
    if cyl:
        ax.plot(slices['r'], photo['top'], color='black', lw=2)
        ax.plot(slices['r'], photo['bot'], color='black', lw=2)
        if not only_photo:
            ax.plot(slices['r'], bound['top'], color='limegreen', lw=2)
            ax.plot(slices['r'], bound['bot'], color='limegreen', lw=2)
        if not opt_thin and not only_photo:
            ax.plot(slices['r'], surface['top'], color='cyan', lw=2)
            ax.plot(slices['r'], surface['bot'], color='cyan', lw=2)
    else:
        Rphoto = dict.fromkeys(['top', 'bot'])
        zphoto = dict.fromkeys(['top', 'bot'])
        
        Rbound = dict.fromkeys(['top', 'bot'])
        zbound = dict.fromkeys(['top', 'bot'])

        for key in ['top', 'bot']:
            Rphoto[key] = slices['r'] * np.sin(photo[key])
            zphoto[key] = slices['r'] * np.cos(photo[key]) 
            ax.plot(Rphoto[key], zphoto[key], color='magenta', lw=2)
            if not only_photo:
                Rbound[key] = slices['r'] * np.sin(bound[key])
                zbound[key] = slices['r'] * np.cos(bound[key])
                ax.plot(Rbound[key], zbound[key], color='limegreen', lw=2)
            if not opt_thin and not only_photo: 
                ax.plot(slices['r'], surface[key], color='cyan', lw=2)
            
def get_figure_meta(**kwargs):
    meta = {}
    meta['fontsize'] = 24
    if kwargs.get('show_2D'):
        meta['figsize'] = [24., 18.]
        meta['dims']    = [3, 4]
    else:
        if kwargs.get('timescales_only') or kwargs.get('fluxes_only'):
            meta['figsize'] = [16., 8.]
            meta['dims']    = [1, 1]
        else:
            meta['figsize'] = [16., 12.]
            meta['dims']    = [2, 1]
    return meta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description="""Plot 1D timescale and flux profiles
                        for a single dump, from the saved Mass_Fluxes,
                        Energy_Fluxes and Timescales .npz files produced
                        by efficiencies_timescales.py.""")
    parser.add_argument('sim', type=str, help="""simulation name""")
    parser.add_argument('file_number', type=int,
                        help="""dump number to plot""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if reduced-resolution
                        profiles should be plotted.""")
    parser.add_argument('--low_Mdot', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="""use axis limits appropriate for a
                        low accretion rate simulation""")
    parser.add_argument('--outfile', type=str, required=False, default=None,
                        help="""output PNG path (default:
                        Images/<sim>/.../QuickView/Timescales_<file_number>.png)""")
    args = parser.parse_args()
    main(**vars(args))
