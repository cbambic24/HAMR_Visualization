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
style.use('seaborn-v0_8-bright')
matplotlib.rcParams['figure.dpi'] = 300
import figs

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
    gamma_adi      = kwargs.get('gamma_adi')
    r0             = kwargs.get('r0')
    r_max          = kwargs.get('r_max')
    z_plot         = kwargs.get('z_plot')
    plot_title     = kwargs.get('plot_title')
    disable_TeX    = kwargs.get('disable_TeX')
    MBH            = kwargs.get('MBH')

    theta_plot = True if not z_plot else False
    
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

    # Determine slice directory
    averages_dir = cv.user_path + f'/Averages/{sim}'
    save_dir = cv.user_path +  '/PDF'

    keys = ['total', 'mean', 'turb']

    # Stresses (Poloidal Slices -> Radial Profiles + Vertical Profiles)
    stress_keys = [
        'TRphi_Rey_mean', 'TRphi_Rey_turb', 'TRphi_Max_mean', 'TRphi_Max_turb', 
        'Tzphi_Rey_mean', 'Tzphi_Rey_turb', 'Tzphi_Max_mean', 'Tzphi_Max_turb', 
        'Trphi_Rey_mean', 'Trphi_Rey_turb', 'Trphi_Max_mean', 'Trphi_Max_turb', 'w' 
    ]

    # Radial Profiles
    profile_keys = ['rho', 'Dtheta2']

    # Poloidal Slices + Vertical Profiles
    slice_keys   = ['rho', 'Pthm', 'Pmag']

    if (radiation):
        stress_keys += ['Rrphi', 'RRphi', 'Rzphi']
        slice_keys  += ['Prad']

    plot_info = {
        'total' : {'row_label' : r'$\rm{Total}$',     'ylabel' : r'$\alpha_{\rm Total}$'},
        'mean'  : {'row_label' : r'$\rm{Mean}$',      'ylabel' : r'$\alpha_{\rm Mean}$'},
        'turb'  : {'row_label' : r'$\rm{Turbulent}$', 'ylabel' : r'$\alpha_{\rm Turb}$'}
    }
    for id, key in enumerate(keys):
        plot_info[key]['axis_id'] = id
    
    file_info = {
        'averages_dir' : averages_dir,
        'file_number'  : file_number,
        'start_file'   : start_file,
        'end_file'     : end_file,
        'reduced'      : reduced
    }

    Profile  = cv.Profiles()
    Slice    = cv.Slices()
    Stress   = cv.Stresses()
    VProfile = cv.Vertical_Profiles()
    VStress  = cv.Vertical_Stress_Profiles()

    # Use the vertical profiles for the file numbers
    fnums = VProfile.get_fnums(**file_info)
    single_file = True if (len(fnums) == 1) else False

    # rstress: radial stress profiles (2D)
    # vstress: vertical stress profiles (1D, at r0)
    # rprof:   radial profiles
    # vprof:   vertical profiles (1D, at r0)
    # slice:   poloidal slices (2D)
    common_args = {
        'averages_dir' : averages_dir,
        'fnums'        : fnums,
        'reduced'      : reduced
    }

    # Every requested stress/slice quantity is already an appropriately
    # weighted (sqrt(-g)-type) quantity, so none of them get divided by
    # volume again during time-averaging.
    vols       = None if single_file else dict.fromkeys(stress_keys + ['Vol'], False)
    slice_vols = None if single_file else dict.fromkeys(slice_keys  + ['Vol'], False)

    rstress_args = common_args.copy()
    rstress_args['keys'] = stress_keys + ['Vol', 'r', 'theta', 'a', 'dx2'] if single_file else stress_keys + ['Vol']
    if not single_file:
        rstress_args['vols'] = vols

    vstress_args = rstress_args.copy()
    vstress_args['r0'] = r0
    if not single_file:
        vstress_args['time_average'] = True

    rprof_args = common_args.copy()
    rprof_args['keys'] = profile_keys + ['Vol', 'r'] if single_file else profile_keys + ['Vol', 'r', 'a']
    if not single_file:
        rprof_args['vols'] = vols

    vprof_args = common_args.copy()
    vprof_args['keys'] = slice_keys + ['Vol', 'r', 'theta', 'a'] if single_file else slice_keys + ['Vol']
    vprof_args['r0'] = r0
    if not single_file:
        vprof_args['vols']         = slice_vols
        vprof_args['time_average'] = True

    slice_args = common_args.copy()
    slice_args['keys'] = slice_keys + ['Vol'] if single_file else slice_keys
    if not single_file:
        slice_args['vols'] = slice_vols

    slice_loader  = Slice.load_slices  if single_file else Slice.load_slices_time_averaged
    stress_loader = Stress.load_slices if single_file else Stress.load_slices_time_averaged

    # Slices and stresses (2D): needed to average over midplane
    slices   = slice_loader(**slice_args)
    stresses = stress_loader(**rstress_args)

    # Radial profiles
    profiles = Profile.load_profiles(**rprof_args)
    if not single_file:
        Profile.check_AMR(fnums, profiles)

    # Define loader functions
    vstress_loader  = VStress.load_theta_profiles  if theta_plot else VStress.load_z_profiles
    vprof_loader    = VProfile.load_theta_profiles if theta_plot else VProfile.load_z_profiles

    # Load in stress and vertical profiles (density, pressure, etc.)
    stress_profiles = vstress_loader(**vstress_args)
    vert_profiles   = vprof_loader(**vprof_args)

    fnum0 = f'{fnums[0]}'

    def get(d, key):
        return d[fnum0][key] if single_file else d[key]

    # Stresses (2D)
    Trphi = dict.fromkeys(keys)
    TRphi = dict.fromkeys(keys)
    Tzphi = dict.fromkeys(keys)

    # Vertical Stress Profiles (1D)
    Trphi_prof = dict.fromkeys(keys)
    TRphi_prof = dict.fromkeys(keys)
    Tzphi_prof = dict.fromkeys(keys)

    rphi_args = {'direction' : 'rphi', 'radiation' : radiation}
    Rphi_args = rphi_args.copy()
    Rphi_args['direction'] = 'Rphi'
    zphi_args = rphi_args.copy()
    zphi_args['direction'] = 'zphi'
    if single_file:
        rphi_args['fnum'] = fnums[0]
        Rphi_args['fnum'] = fnums[0]
        zphi_args['fnum'] = fnums[0]

    fig  = plt.figure(figsize = figsize)
    gs   = gridspec.GridSpec(6, 3, figure = fig)
    axes = []
    for i in range(3):
        for j in range(3):
            axes.append(fig.add_subplot(gs[2*i:2*(i+1), j]))

    x       = get(vert_profiles, 'theta') if theta_plot else get(vert_profiles, 'z_r')
    vertVol = get(vert_profiles, 'Vol')
    Pmag_v  = get(vert_profiles, 'Pmag') / vertVol
    rho_v   = get(vert_profiles, 'rho')  / vertVol
    sigma   = 2. * Pmag_v / (rho_v * c * c)

    # P_tot = sqrt(-g) * (Pthm + Pmag + [Prad])
    Pmag = get(slices, 'Pmag')
    Pthm = get(slices, 'Pthm')
    Ptot = Pthm.copy()
    if radiation:
        Ptot += get(slices, 'Prad')
    Ptot += Pmag

    # Calculate disk mask (2x density scaleheight)
    if single_file:
        fnums0  = f'{file_number}'
        Dtheta2 = get(profiles, 'Dtheta2') / get(profiles, 'Vol')
    else:
        Profile.check_AMR(fnums, profiles)
        fnums0  = f'{Profile.fnums0}'
        Dtheta2 = Profile.time_avg('Dtheta2', profiles, fnums=fnums, vol=True)
    HoR = np.sqrt(Dtheta2)
    th0 = np.asin(HoR)
    
    r_prof = profiles[fnums0]['r']

    r     = get(stresses, 'r')
    theta = get(stresses, 'theta')
    dx2   = get(stresses, 'dx2')
    ones  = np.ones_like(Ptot)

    if (len(r_prof) > len(r)): th0 = th0[:len(r)]

    # Use pi/2 for midplane if single file, otherwise use the peak density
    if single_file:
        th_mid = np.pi / 2.
    elif theta_plot:
        th_mid = theta[np.argmax(rho_v)]
    else:
        th_mid = np.acos(x[np.argmax(rho_v)])

    disk = ((theta * ones > (th_mid * ones - th0[:, None] * ones)) *
            (theta * ones < (th_mid * ones + th0[:, None] * ones)))

    # sqrt(-g)
    Vol      = get(stresses, 'Vol')
    Vol_vert = get(stress_profiles, 'Vol')

    # Midplane pressure -- also independent of `key`
    Ptot_avg  = (Ptot * disk * dx2).sum(-1)
    Ptot_of_r = Ptot_avg / (Vol * disk * dx2).sum(-1)
    idx       = np.argmin(np.abs(r - r0))
    Ptot_mid  = Ptot_of_r[idx]

    # Rescale z/R -> z/H_rho using the density scale height at r0
    if z_plot:
        HoR0 = HoR[np.argmin(np.abs(r_prof - r0))]
        x    = x / HoR0

    for key in keys:
        id = plot_info[key]['axis_id']

        # sqrt(-g) * Trphi
        rphi_args['stype']    = key
        rphi_args['stresses'] = stresses
        Trphi[key]            = get_stress(**rphi_args)
        rphi_args['stresses'] = stress_profiles
        Trphi_prof[key]       = get_stress(**rphi_args)

        # sqrt(-g) * TRphi
        Rphi_args['stype']    = key
        Rphi_args['stresses'] = stresses
        TRphi[key]            = get_stress(**Rphi_args)
        Rphi_args['stresses'] = stress_profiles
        TRphi_prof[key]       = get_stress(**Rphi_args)

        # sqrt(-g) * Tzphi
        zphi_args['stype']    = key
        zphi_args['stresses'] = stresses
        Tzphi[key]            = get_stress(**zphi_args)
        zphi_args['stresses'] = stress_profiles
        Tzphi_prof[key]       = get_stress(**zphi_args)

        # Trphi (theta) or Trphi (z)
        Trphi_prof_All = Trphi_prof[key]['All'] / Vol_vert
        Trphi_prof_Rey = Trphi_prof[key]['Rey'] / Vol_vert
        Trphi_prof_Max = Trphi_prof[key]['Max'] / Vol_vert
        if radiation and key == 'total':
            Trphi_prof_Rad = Trphi_prof[key]['Rad'] / Vol_vert
        # Tzphi (theta) or Tzphi (z)
        Tzphi_prof_All = Tzphi_prof[key]['All'] / Vol_vert
        Tzphi_prof_Rey = Tzphi_prof[key]['Rey'] / Vol_vert
        Tzphi_prof_Max = Tzphi_prof[key]['Max'] / Vol_vert
        if radiation and key == 'total':
            Tzphi_prof_Rad = Tzphi_prof[key]['Rad'] / Vol_vert

        plt_args = {'lw' : 2, 'linestyle' : '-'}

        axes[id].plot(x, Trphi_prof_All/Ptot_mid, color='black',      label=r'$\rm{Total}$',    **plt_args)
        axes[id].plot(x, Trphi_prof_Rey/Ptot_mid, color='darkorange', label=r'$\rm{Reynolds}$', **plt_args)
        axes[id].plot(x, Trphi_prof_Max/Ptot_mid, color='dodgerblue', label=r'$\rm{Maxwell}$',  **plt_args)
        if radiation and key == 'total':
            axes[id].plot(x, Trphi_prof_Rad/Ptot_mid, color='crimson', label=r'$\rm{Diffusion}$', **plt_args)
        if theta_plot:
            axes[id].axvline(x=(th_mid - th0[idx]), linestyle='--', color='black')
            axes[id].axvline(x=(th_mid + th0[idx]), linestyle='--', color='black')
        else:
            # z/R = cot(theta), then rescale to z/H_rho like x
            zor_max = 1. / np.tan(th_mid - th0[idx]) / HoR0
            zor_min = 1. / np.tan(th_mid + th0[idx]) / HoR0
            axes[id].axvline(x=zor_min, linestyle='--', color='black')
            axes[id].axvline(x=zor_max, linestyle='--', color='black')

        axes[3 + id].plot(x, Tzphi_prof_All/Ptot_mid, color='black',      label=r'$\rm{Total}$',    **plt_args)
        axes[3 + id].plot(x, Tzphi_prof_Rey/Ptot_mid, color='darkorange', label=r'$\rm{Reynolds}$', **plt_args)
        axes[3 + id].plot(x, Tzphi_prof_Max/Ptot_mid, color='dodgerblue', label=r'$\rm{Maxwell}$',  **plt_args)
        if radiation and key == 'total':
            axes[3 + id].plot(x, Tzphi_prof_Rad/Ptot_mid, color='crimson', label=r'$\rm{Diffusion}$', **plt_args)

        Trphi_All = Trphi[key]['All']
        Trphi_Rey = Trphi[key]['Rey']
        Trphi_Max = Trphi[key]['Max']

        alpha_All = (Trphi_All * disk * dx2).sum(-1) / Ptot_avg
        alpha_Rey = (Trphi_Rey * disk * dx2).sum(-1) / Ptot_avg
        alpha_Max = (Trphi_Max * disk * dx2).sum(-1) / Ptot_avg

        axes[6 + id].plot(r, alpha_All, color='black',      label=r'$\rm{Total}$',    **plt_args)
        axes[6 + id].plot(r, alpha_Rey, color='darkorange', label=r'$\rm{Reynolds}$', **plt_args)
        axes[6 + id].plot(r, alpha_Max, color='dodgerblue', label=r'$\rm{Maxwell}$',  **plt_args)
        if radiation and key == 'total':
            Trphi_Rad = Trphi[key]['Rad']
            alpha_Rad = (Trphi_Rad * disk * dx2).sum(-1) / Ptot_avg
            axes[6 + id].plot(r, alpha_Rad, color='crimson', label=r'$\rm{Diffusion}$', **plt_args)

        for i in range(3):
            axes[3*i + id].text(0.05, 0.10, plot_info[key]['row_label'],
                fontsize=FS+12, transform=axes[3*i + id].transAxes, **opt)

    for ax in axes[:6]:
        if theta_plot:
            xticklabels = [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
            ax.set_xlabel(r'$\theta$', fontsize=FS)
            ax.set_xticks([0., (1./4.)*np.pi, (1./2.)*np.pi, (3./4.)*np.pi, np.pi])
            ax.set_xticklabels(xticklabels, fontsize=FS)
        elif z_plot:
            #ax.set_xlabel(r'$z/R$', fontsize=FS)
            ax.set_xlabel(r'$z/H_{\rho}$', fontsize=FS)
            ax.set_xlim([-10, 10])
        cv.grid(ax, alpha=0.2, color='black')

    Ptot_label = r'$\langle$'
    if radiation:
        Ptot_label += r'$\left( P_{\rm gas} + P_{\rm rad} + P_{\rm mag} \right)$'
    else:
        Ptot_label += r'$P_{\rm gas} + P_{\rm mag}$'
    Ptot_label += r'$\rangle_{\varphi H}$'

    for ax in axes[:3]:
        ax.set_ylim([1.e-4, 20.])
        ax.set_yscale('log')
        Ti_phi_label = r'$\langle \mathcal{T}^r_{\varphi} \rangle_{\varphi}$'
        ax.set_ylabel(f'{Ti_phi_label}/{Ptot_label}', fontsize=FS)

    for ax in axes[3:6]:
        ax.set_ylim([-20., 20.])
        ax.set_yscale('symlog', linthresh=1.e-3)
        Ti_phi_label = r'$\langle \mathcal{T}^z_{\varphi} \rangle_{\varphi}$'
        ax.set_ylabel(f'{Ti_phi_label}/{Ptot_label}', fontsize=FS)

    for ax in axes[:6]:
        # Draw jet boundaries
        if theta_plot:
            theta_min, theta_max = get_jet_bounds(x, sigma, sigma_cut=1.)
        elif z_plot:
            z_min, z_max = get_jet_bounds(x, sigma, sigma_cut=1, z=True)

        ymin, ymax = ax.get_ylim()
        if theta_plot:
            ax.fill_between(x, ymin, ymax, where=x < theta_min, facecolor='gray', alpha=0.2)
            ax.fill_between(x, ymin, ymax, where=x > theta_max, facecolor='gray', alpha=0.2)

    for ax in axes[6:]:
        ax.set_ylim([1.e-4, 100.])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$r / R_g$', fontsize=FS)
        Ti_phi_label  = r'$\alpha \equiv$'
        Ti_phi_label += r'$\langle \mathcal{T}^r_{\varphi} \rangle_{\varphi H}$'
        ax.set_ylabel(f'{Ti_phi_label}/{Ptot_label}', fontsize=FS)
        cv.grid(ax, alpha=0.2, color='black')
        ax.legend(loc='upper right', fontsize=FS, frameon=False)
        if r_max != 0.:
            ax.set_xlim([1., r_max])

    print("Saving figure ...", flush=True)
    fig.tight_layout()
    plt.savefig(f'{save_dir}/{plot_title}.png', format='png')
    plt.close()

def get_stress(direction, stype, stresses, **kwargs):
    single    = bool(kwargs.get('fnum', False))
    fnum      = kwargs.get('fnum')
    radiation = kwargs.get('radiation', 0)

    skey = f'T{direction}'
    Tiphi = dict.fromkeys(['All', 'Rey', 'Max'])
    d = stresses[f'{fnum}'] if single else stresses

    if stype == 'total':
        Tiphi['Rey'] = d[f'{skey}_Rey_turb']
        Tiphi['Max'] = d[f'{skey}_Max_mean'] + d[f'{skey}_Max_turb']
        if radiation:
            Tiphi['Rad'] = d[f'R{direction}']
        else:
            Tiphi['Rad'] = 0.
        Tiphi['All'] = Tiphi['Rey'] + Tiphi['Max'] + radiation * Tiphi['Rad']
    elif stype == 'mean':
        Tiphi['Rey'] = d[f'{skey}_Rey_mean']
        Tiphi['Max'] = d[f'{skey}_Max_mean']
        Tiphi['All'] = Tiphi['Rey'] + Tiphi['Max']
    elif stype == 'turb':
        Tiphi['Rey'] = d[f'{skey}_Rey_turb']
        Tiphi['Max'] = d[f'{skey}_Max_turb']
        Tiphi['All'] = Tiphi['Rey'] + Tiphi['Max']

    return Tiphi

def get_jet_bounds(x, sigma, **kwargs):
    sigma_cut = kwargs.get('sigma_cut', 1.)
    x_j = x[sigma > sigma_cut]
    if (len(x_j) == 0):
        return x[0], x[-1]
    if kwargs.get('z', False):
        return x_j[x_j < 0.][-1], x_j[x_j > 0.][0]
    else:
        return x_j[x_j < np.pi/2.][-1], x_j[x_j > np.pi/2.][0]
    
if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Plot vertical and radial 
                        profiles of stresses.""")
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
    parser.add_argument('--plot_title', type=str, required=False, 
                        default='Stress_Profiles', 
                        help="""title of PDF output""")
    args = parser.parse_args()
    main(**vars(args))