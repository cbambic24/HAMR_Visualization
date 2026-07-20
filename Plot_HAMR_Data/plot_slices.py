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
FS      = 40
plt.rcParams['xtick.labelsize'] = FS
plt.rcParams['ytick.labelsize'] = FS

opt = dict(ha='left', va='center', color='white',
           bbox=dict(facecolor='none', edgecolor='none'))

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
    plot_vec       = kwargs.get('plot_vec')
    disable_TeX    = kwargs.get('disable_TeX')
    MBH            = kwargs.get('MBH')

    matplotlib.rcParams['text.usetex'] = False if disable_TeX else True

    cgs = True if (two_temp or radiation) else False
    Analytic = theory.Analytic(hr.CONSTANTS, cgs=cgs)
    L_Edd    = Analytic.calc_LEdd(MBH=MBH) if cgs else 1.

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

    plot_keys = [
        'rho',   'Ti',      'Te',        'Ti_Te',
        'Be',   'kappa',  'Pmag_Ptot',   'sigma',
        'Qie',   'QIC',    'Qie_QIC',   'QIC_Qemi'
    ]

    plot_info = dict.fromkeys(plot_keys)
    for key in plot_keys:
        plot_info[key] = dict.fromkeys(['cmap', 'axis_id', 'label', 'label_color', 'log', 'linthresh', 'vecs'])

    # Limits for each figure
    user_limits = {
        'rho'        : None,
        'Ti'         : [1.e4,  1.e12],
        'Te'         : [1.e4,  1.e12],
        'Ti_Te'      : [1.e0,  1.e3],
        'Be'         : [-1.,   1.],
        'kappa'      : [1.e-3, 1.e3],
        'Pmag_Ptot'  : [1.e-3, 1.e3],
        'sigma'      : [1.e-3, 1.e3],
        'Qie'        : [1.e-12, 1.],
        'QIC'        : [1.e-12, 1.],
        'Qie_QIC'    : [1.e-2, 1.e2],
        'QIC_Qemi'   : [1.e-2, 1.e2]
    }

    if plot_vec:
        vec_args = {
            'dens'               : 2.0,
            'linewidth'          : 0.6,
            'broken_streamlines' : True,
            'arrowsize'          : 1.
        }    

    # Log-scale and linthresh if symlog
    for key in plot_keys: plot_info[key]['log']       = True
    for key in plot_keys: plot_info[key]['linthresh'] = None
    
    plot_info['Be']['linthresh'] = 1.e-3

    # Color map for each figure
    plot_info['rho'][      'cmap'] = cmr.torch
    plot_info['Ti'][       'cmap'] = cmr.prinsenvlag_r
    plot_info['Te'][       'cmap'] = cmr.prinsenvlag_r
    plot_info['Ti_Te'][    'cmap'] = cmr.torch
    plot_info['Be'][       'cmap'] = cmr.prinsenvlag
    plot_info['kappa'][    'cmap'] = cmr.prinsenvlag
    plot_info['Pmag_Ptot']['cmap'] = cmr.prinsenvlag
    plot_info['sigma'][    'cmap'] = cmr.prinsenvlag
    plot_info['Qie'][      'cmap'] = cmr.torch
    plot_info['QIC'][      'cmap'] = cmr.torch
    plot_info['Qie_QIC'][  'cmap'] = cmr.prinsenvlag_r
    plot_info['QIC_Qemi'][ 'cmap'] = cmr.prinsenvlag_r

    # Labels for each figure
    plot_info['rho'][      'label'] = r'$\rho \: (\rm{g/cm}^3)$'
    plot_info['Ti'][       'label'] = r'$T_i \: (\rm{K})$'
    plot_info['Te'][       'label'] = r'$T_e \: (\rm{K})$'
    plot_info['Ti_Te'][    'label'] = r'$T_i / T_e$'
    plot_info['Be'][       'label'] = r'$\rm{Be}$'
    plot_info['kappa'][    'label'] = r'$\kappa_{\rm eff} / \kappa_{\rm es}$'
    plot_info['Pmag_Ptot']['label'] = r'$P_{\rm mag} / P_{\rm rad + gas}$'
    plot_info['sigma'][    'label'] = r'$\sigma$'
    plot_info['Qie'][      'label'] = r'$\mathcal{Q}_{\rm ie} R_g^3 / L_{\rm Edd}$'
    plot_info['QIC'][      'label'] = r'$\mathcal{Q}_{\rm IC} R_g^3 / L_{\rm Edd}$'
    plot_info['Qie_QIC'][  'label'] = r'$\mathcal{Q}_{\rm ie} / \mathcal{Q}_{\rm IC}$'
    plot_info['QIC_Qemi'][ 'label'] = r'$\mathcal{Q}_{\rm IC} / \mathcal{Q}_{\rm emi}$'

    # Label colors
    for key in plot_keys: plot_info[key]['label_color'] = 'black'
    plot_info['rho'][  'label_color'] = 'white'
    plot_info['Ti_Te']['label_color'] = 'white'
    plot_info['Qie'][  'label_color'] = 'white'
    plot_info['QIC'][  'label_color'] = 'white'

    # Axis ID
    for id, key in enumerate(plot_keys):
        plot_info[key]['axis_id'] = id

    # Vector plots corresponding to figure panel
    for key in plot_keys: plot_info[key]['vecs'] = None
    if plot_vec:
        plot_info['Pmag_Ptot']['vecs'] = 'Radiation_Flux'
        plot_info['sigma'][    'vecs'] = 'Magnetic_Field'
        plot_info['kappa'][    'vecs'] = 'Total_Energy_Flux'
        plot_info['Be'][       'vecs'] = 'Mass_Flux'

    file_info = {
        'averages_dir' : averages_dir,
        'file_number'  : file_number,
        'start_file'   : start_file,
        'end_file'     : end_file,
        'reduced'      : reduced
    }

    Slice  = cv.Slices()
    fnums  = Slice.get_fnums(**file_info)
    single_file  = True if (len(fnums) == 1) else False
    time_average = True if (len(fnums) != 1) else False

    # Keys to load in
    data_keys = [
        'rho', 'Eion', 'Eelc', 'Pthm', 'Prad', 'Pmag', 'rhokappa', 'Be_Rad', 
        'qie', 'qIC', 'qemi'
    ]
    vec_keys = [
        'w', 'wuR', 'wuz', 'bR', 'bz',
        'THD_tR', 'THD_tz', 'TEM_tR', 'TEM_tz', 'R_tR', 'R_tz'
    ]
        
    slice_args = {
        'averages_dir' : file_info['averages_dir'],
        'fnums'        : fnums,
        'reduced'      : reduced
    }

    if (single_file):
        slice_args['keys'] = data_keys + vec_keys + ['Vol', 'Vol_sigma_cut', 'r', 'theta', 'a']
        slices = Slice.load_slices(**slice_args)
    elif (time_average):
        slice_args['keys'] = data_keys + vec_keys + ['Vol_sigma_cut']
        vols = dict.fromkeys(data_keys + vec_keys, True)
        vols['rhokappa']      = False
        vols['Vol_sigma_cut'] = False
        slice_args['vols']    = vols
        slices = Slice.load_slices_time_averaged(**slice_args)

    Cart = figs.Cartesian([])
    fig  = plt.figure(figsize = figsize)
    gs   = gridspec.GridSpec(3, 4, figure = fig)
    axes = []
    for i in range(3):
        for j in range(4):
            axes.append(fig.add_subplot(gs[i, j]))
        
    fnum_str = f'{fnums[0]}'
    r     = slices[fnum_str]['r'][None, :, None, None] if single_file else slices['r'][None, :, None, None]
    theta = slices[fnum_str]['theta'][None, None, :, None] if single_file else slices['theta'][None, None, :, None]
    # Coordinate arrays
    R = {
        'r' :  (r * np.sin(theta))[0, :, :, 0],
        'l' : -(r * np.sin(theta))[0, :, :, 0],
    }
    Z = {
        'r' : (r * np.cos(theta))[0, :, :, 0],
        'l' : (r * np.cos(theta))[0, :, :, 0],
    }

    for key in plot_keys:
        # Create Q dictionary for scalar plotting.
        if key == 'rho':
            Vol = slices[fnum_str]['Vol']     if single_file else 1.
            rho = slices[fnum_str]['rho']/Vol if single_file else slices['rho']
            Q  = {'r' : rho}
        if key == 'Ti':
            Vol  = slices[fnum_str]['Vol']      if single_file else 1.
            rho  = slices[fnum_str]['rho']/Vol  if single_file else slices['rho']
            Eion = slices[fnum_str]['Eion']/Vol if single_file else slices['Eion']
            gamma_adi = 5./3.
            Ti   = (gamma_adi-1.) * (1./kB) * Eion / (rho / (mu_i*mp))
            Q    = {'r' : Ti}
        if key == 'Te':
            Vol  = slices[fnum_str]['Vol']      if single_file else 1.
            rho  = slices[fnum_str]['rho']/Vol  if single_file else slices['rho']
            Eelc = slices[fnum_str]['Eelc']/Vol if single_file else slices['Eelc']
            Te   = (gamma_adi-1.) * (1./kB) * Eelc / (rho / (mu_e*mp))
            Q    = {'r' : Te}
        if key == 'Ti_Te':
            gamma_adi = 5./3.
            Vol  = slices[fnum_str]['Vol']      if single_file else 1.
            rho  = slices[fnum_str]['rho']/Vol  if single_file else slices['rho']
            Eion = slices[fnum_str]['Eion']/Vol if single_file else slices['Eion']
            Eelc = slices[fnum_str]['Eelc']/Vol if single_file else slices['Eelc']
            Ti   = (gamma_adi-1.) * (1./kB) * Eion / (rho / (mu_i*mp))
            Te   = (gamma_adi-1.) * (1./kB) * Eelc / (rho / (mu_e*mp))
            Q    = {'r' : Ti / Te}
        if key == 'Be':
            Vol  = slices[fnum_str]['Vol']      if single_file else 1.
            rho  = slices[fnum_str]['rho']/Vol  if single_file else slices['rho']
            wu_t = (1./c) * slices[fnum_str]['Be_Rad']/Vol if single_file else (1./c) * slices['Be_Rad']
            Q    = {'r' : (-wu_t / (rho * c * c)) - 1.}
        if key == 'kappa':
            Vol  = slices[fnum_str]['Vol']      if single_file else 1.
            rho  = slices[fnum_str]['rho']/Vol  if single_file else slices['rho']
            Vol_sigma = slices[fnum_str]['Vol_sigma_cut'] if single_file else slices['Vol_sigma_cut']
            rhokappa  = slices[fnum_str]['rhokappa']/Vol_sigma if single_file else slices['rhokappa']/Vol_sigma
            kappa_eff = np.nan_to_num(rhokappa / rho)
            kappa_es  = 0.2 * (1 + hr.CONSTANTS['X_AB'])
            Q = {'r' : kappa_eff / kappa_es}
        if key == 'Pmag_Ptot':
            Vol  = slices[fnum_str]['Vol']      if single_file else 1.
            Pthm = slices[fnum_str]['Pthm']/Vol if single_file else slices['Pthm']
            Pmag = slices[fnum_str]['Pmag']/Vol if single_file else slices['Pmag']
            Prad = slices[fnum_str]['Prad']/Vol if single_file else slices['Prad']
        if key == 'sigma':
            Vol   = slices[fnum_str]['Vol']      if single_file else 1.
            rho   = slices[fnum_str]['rho']/Vol  if single_file else slices['rho']
            Pmag  = slices[fnum_str]['Pmag']/Vol if single_file else slices['Pmag']
            sigma = 2. * Pmag / (rho * c * c) 
            Q     = {'r' : sigma}
        if key == 'Qie':
            Vol   = slices[fnum_str]['Vol']      if single_file else 1.
            qie   = slices[fnum_str]['qie']/Vol  if single_file else slices['qie']
            Q     = {'r' : -qie * Rg * Rg * Rg / L_Edd}
        if key == 'QIC':
            Vol   = slices[fnum_str]['Vol']      if single_file else 1.
            qIC   = slices[fnum_str]['qIC']/Vol  if single_file else slices['qIC']
            Q     = {'r' : -qIC * Rg * Rg * Rg / L_Edd}
        if key == 'Qie_QIC':
            Vol   = slices[fnum_str]['Vol']      if single_file else 1.
            qie   = slices[fnum_str]['qie']/Vol  if single_file else slices['qie']
            qIC   = slices[fnum_str]['qIC']/Vol  if single_file else slices['qIC']
            Q     = {'r' : qie / qIC}
        if key == 'QIC_Qemi':
            Vol   = slices[fnum_str]['Vol']      if single_file else 1.
            qIC   = slices[fnum_str]['qIC']/Vol  if single_file else slices['qIC']
            qemi  = slices[fnum_str]['qemi']/Vol if single_file else slices['qemi']
            Q     = {'r' : qIC / qemi}

        # Create Vec_R, Vec_Z dictionaries for vector plotting.
        if plot_info[key]['vecs']:
            npts = 100
            R1D = np.linspace(0., r_max, npts)
            z1D = np.linspace(-r_max/2, r_max/2, npts)
            R_uni, Z_uni = np.meshgrid(R1D, z1D)
            
            vec = plot_info[key]['vecs']
            if vec == 'Mass_Flux':
                # Enthalpy-weighted velocity
                Vol = slices[fnum_str]['Vol']     if single_file else 1.
                w   = slices[fnum_str]['w']/Vol   if single_file else slices['w']
                wuR = slices[fnum_str]['wuR']/Vol if single_file else slices['wuR']
                wuz = slices[fnum_str]['wuz']/Vol if single_file else slices['wuz']
                Vec_R = {'r' : wuR / w}
                Vec_Z = {'r' : wuz / w}
            if vec == 'Magnetic_Field':
                # Magnetic field
                Vol = slices[fnum_str]['Vol']    if single_file else 1.
                bR  = slices[fnum_str]['bR']/Vol if single_file else slices['bR']
                bz  = slices[fnum_str]['bz']/Vol if single_file else slices['bz']
                Vec_R = {'r' : bR}
                Vec_Z = {'r' : bz} 
            if vec == 'Total_Energy_Flux':
                # Energy flux (Enthalpy + Electromagnetic)
                Vol    = slices[fnum_str]['Vol']        if single_file else 1.
                THD_tR = slices[fnum_str]['THD_tR']/Vol if single_file else slices['THD_tR']
                THD_tz = slices[fnum_str]['THD_tz']/Vol if single_file else slices['THD_tz']
                TEM_tR = slices[fnum_str]['TEM_tR']/Vol if single_file else slices['TEM_tR']
                TEM_tz = slices[fnum_str]['TEM_tz']/Vol if single_file else slices['TEM_tz']
                Vec_R  = {'r' : -(THD_tR + TEM_tR)}
                Vec_Z  = {'r' : -(THD_tz + TEM_tz)}
            if vec == 'Radiation_Flux':
                # Radiation flux
                Vol  = slices[fnum_str]['Vol']      if single_file else 1.
                R_tR = slices[fnum_str]['R_tR']/Vol if single_file else slices['R_tR']
                R_tz = slices[fnum_str]['R_tz']/Vol if single_file else slices['R_tz']
                Vec_R = {'r' : -R_tR}
                Vec_Z = {'r' : -R_tz}

        # Plot scalars.
        id   = plot_info[key]['axis_id']
        cmap = plot_info[key]['cmap']
        plt_args = {
            'lim' : user_limits[key],
            'log' : plot_info[key]['log'],
            'sym' : True if plot_info[key]['linthresh'] else False,
            'linthresh' : plot_info[key]['linthresh'],
            'extend'    : cv.check_extend(Q['r'], user_limits[key]) if user_limits[key] else 'both'
        }
        Cart.plot_Rz(fig, axes[id], R, Z, Q, cmap, FS, **plt_args) 
        opt['color'] = plot_info[key]['label_color']
        axes[id].text(0.05, 0.90, plot_info[key]['label'], 
                        fontsize=FS+4, transform=axes[id].transAxes, **opt)

        # Plot vector streamlines.
        if plot_info[key]['vecs']:
            vec_args['color'] = plot_info[key]['label_color']

            Cart.plot_Rz_vec(fig, axes[id], R, Z, R_uni, Z_uni, Vec_R, Vec_Z, **vec_args)

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
    parser.add_argument('--MBH', type=float, default=10.,
                        required=False,
                        help = """black hole mass in Msun (default: 10)""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if profile extraction
                        should be done on reduced data.""")
    parser.add_argument('--disable_TeX', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""boolean to disable TeX in plotting""")
    parser.add_argument('--plot_title', type=str, required=False, 
                        default='Slices', 
                        help="""title of PDF output""")
    args = parser.parse_args()
    main(**vars(args))