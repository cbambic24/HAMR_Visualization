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

opt = dict(ha='left', va='center', color='white',
           bbox=dict(facecolor='none', edgecolor='none'))

def main(**kwargs):
    sim            = kwargs.get('sim')
    file_number    = int(kwargs.get('file_number'))
    start_file     = int(kwargs.get('start_file'))
    end_file       = int(kwargs.get('end_file'))
    time_average   = kwargs.get('time_average')
    reduced        = kwargs.get('reduced')
    radiation      = kwargs.get('radiation')
    two_temp       = kwargs.get('two_temp')
    gamma_adi      = kwargs.get('gamma_adi')
    r0             = kwargs.get('r0')
    z_plot         = kwargs.get('z_plot')
    plot_title     = kwargs.get('plot_title')

    theta_plot = True if not z_plot else False

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
        'rho', 'Pthm', 'Pmag', 'wuR', 'bR', 'TEM_t1'
    ]
    vec_keys = [
        'w', 'wur', 'wuth', 'wuphi', 'wuz',
        'THD_t1', 'bphi', 'bz'
    ]
    if (two_temp):
        keys     += ['Eion', 'qie']
        vec_keys += ['Eelc']
    if (radiation):
        keys     += ['Prad']
        vec_keys += ['R_t1', 'qIC', 'qemi']

    norms     = dict.fromkeys(keys)
    vols      = dict.fromkeys(keys)
    logs      = dict.fromkeys(keys)
    linthresh = dict.fromkeys(keys) 
    extend    = dict.fromkeys(keys)

    for key in keys: 
        vols[     f'{key}'] = get_vols(key)
        norms[    f'{key}'] = get_norms(key, cgs=cgs, MBH=MBH, Rg=Rg)
        logs[     f'{key}'] = get_logs(key)
        linthresh[f'{key}'] = get_linthresh(key, cgs=cgs)
        extend[   f'{key}'] = get_extend(key)

    limits = {
        'rho'        : [1.e-7,  1.e-1] if cgs else [1.e-4, 1.e2],
        'Eion'       : [1.e4,   1.e12] if cgs else [1.e-8, 1.],
        'Pthm'       : [1.e10,  1.e15] if cgs else [1.e-6, 1.],
        'Pmag'       : [1.e-3,  1.e3],
        'wuR'        : [-1.,    1.],
        'qie'        : [1.e-12, 1.],
        'bR'         : [-1.e9,  1.e9] if cgs else [-1., 1.],
        'TEM_t1'     : [-1.e2,  1.e2]
    }

    plot_labels = {
        'rho'        : r'$\rho \: (\rm{g/cm}^3)$' if cgs else r'$\rho / \rho_0$',
        'Eion'       : r'$\rm{Temperature} \: (\rm{K})$' if cgs else r'${\rm{Temperature}} \: (m_i c^2)$',
        'Pthm'       : r'$\rm{Pressure} \: (\rm{erg/cm}^3)$' if cgs else r'${\rm{Pressure}} \: (\rho_0 c^2)$',
        'Pmag'       : r'$\rm{Pressure} \: \rm{Ratio}$',
        'wuR'        : r'$\langle w u^{\mu} \rangle_{\varphi} / \langle w c \rangle_{\varphi}$',
        'qie'        : r'$\mathcal{Q} R_g^3 / L_{\rm Edd}$',
        'bR'         : r'$\rm{Magnetic} \: \rm{Field} \: (\rm{G})$' if cgs else r'${\rm{Magnetic}} \: {\rm{Field}} \: (B_0)$',
        'TEM_t1'     : r'$\rm{Energy} \: \rm{Flux} \: (L_{\rm Edd})$' if cgs else r'${\rm{Energy}} \: {\rm{Flux}} \: (\rho_0 c^3 R_g^2)$'
    }

    axes_ids = {
        'rho'        : 0,
        'Eion'       : 1,
        'Pthm'       : 2,
        'Pmag'       : 5,
        'wuR'        : 6,
        'qie'        : 4,
        'bR'         : 7,
        'TEM_t1'     : 8
    }
    
    legend_loc = {
        'rho'        : False,
        'Eion'       : 'lower left',
        'Pthm'       : 'upper left',
        'Pmag'       : 'lower left',
        'wuR'        : 'lower left',
        'qie'        : 'lower left',
        'bR'         : 'center left',
        'TEM_t1'     : 'lower left'
    }

    file_info = {
        'averages_dir' : averages_dir,
        'file_number'  : file_number,
        'start_file'   : start_file,
        'end_file'     : end_file,
        'reduced'      : reduced
    }

    Profile = cv.Vertical_Profiles()
    fnums   = Profile.get_fnums(**file_info)
    single_file  = True if (len(fnums) == 1) else False
    time_average = True if (len(fnums) != 1) else False
    if (single_file):
        pkeys  = keys + vec_keys + ['Vol', 'r', 'theta', 'a']
        prof_args = {
            'keys'    : pkeys,
            'reduced' : reduced,
            'r0'      : r0
        }
        if (theta_plot):
            profiles = Profile.load_theta_profiles(averages_dir, fnums, **prof_args)
        elif (z_plot):
            profiles = Profile.load_z_profiles(averages_dir, fnums, **prof_args)

    elif (time_average):
        pkeys = keys + vec_keys + ['Vol_sigma_cut']
        # Include volume for time-averaging of vectors
        for key in vec_keys: vols[f'{key}'] = True
        vols['TEM_t1']        = False
        vols['THD_t1']        = False
        vols['R_t1']          = False
        vols['rhokappa']      = False
        vols['Vol_sigma_cut'] = False
        prof_args = {
            'keys'         : pkeys,
            'reduced'      : reduced,
            'r0'           : r0,
            'vols'         : vols,
            'time_average' : True
        }
        if (theta_plot):
            profiles = Profile.load_theta_profiles(averages_dir, fnums, **prof_args)
        elif (z_plot):
            profiles = Profile.load_z_profiles(averages_dir, fnums, **prof_args)
        
    Cart = figs.Cartesian([])
    fig  = plt.figure(figsize = figsize)
    gs   = gridspec.GridSpec(3, 3, figure = fig)
    axes = []
    for i in range(3):
        for j in range(3):
            axes.append(fig.add_subplot(gs[i, j]))

    for key in keys:
        id        = int(axes_ids[f'{key}'])
        vol       = vols[f'{key}']
        norm      = norms[f'{key}']
        lims      = limits[f'{key}']
        log       = logs[f'{key}']
        label     = plot_labels[f'{key}']
        if (single_file):
            if (theta_plot):
                x  = profiles[f'{fnums[0]}']['theta']
            elif (z_plot):
                x  = profiles[f'{fnums[0]}']['z_r']

            Volume = profiles[f'{fnums[0]}']['Vol'] if vol else 1.
            Pmag   = profiles[f'{fnums[0]}']['Pmag']
            rho    = profiles[f'{fnums[0]}']['rho']
            sigma  = 2. * Pmag / (rho * c * c)
            if key == 'rho':
                rho = profiles[f'{fnums[0]}'][f'{key}'] / Volume
                axes[id].plot(x, rho, lw=2., color='black', linestyle='-')
            elif key == 'Eion':
                Eion = profiles[f'{fnums[0]}']['Eion'] / Volume
                Eelc = profiles[f'{fnums[0]}']['Eelc'] / Volume
                Prad = profiles[f'{fnums[0]}']['Prad'] / Volume
                rho  = profiles[f'{fnums[0]}']['rho']  / Volume
                gamma = gamma_adi
                Ti = (gamma-1.) * (1./kB) * Eion / (rho / (mu_i*mp))
                Te = (gamma-1.) * (1./kB) * Eelc / (rho / (mu_e*mp))
                Tr = (3. * Prad / hr.CONSTANTS['aRAD']) ** (1./4.)
                axes[id].plot(x, Ti, lw=2., color='darkorange', 
                              linestyle='-', label=r'$\rm{Ions}$')
                axes[id].plot(x, Te, lw=2., color='dodgerblue', 
                              linestyle='-', label=r'$\rm{Electrons}$')
                axes[id].plot(x, Tr, lw=2., color='crimson',    
                              linestyle='-', label=r'$\rm{Radiation}$')
            elif key == 'Pthm':
                Pthm = profiles[f'{fnums[0]}']['Pthm'] / Volume
                Pmag = profiles[f'{fnums[0]}']['Pmag'] / Volume
                axes[id].plot(x, Pthm, lw=2., color='darkorange', 
                              linestyle='-', label=r'$\rm{Thermal}$')
                axes[id].plot(x, Pmag, lw=2., color='dodgerblue', 
                              linestyle='-', label=r'$\rm{Magnetic}$')
                if (cgs):
                    Prad = profiles[f'{fnums[0]}']['Prad'] / Volume
                    axes[id].plot(x, Prad, lw=2., color='crimson', 
                        linestyle='-', label=r'$\rm{Radiation}$')
                if not cgs:
                    rho  = profiles[f'{fnums[0]}']['rho'] / Volume
                    Tgas = Pthm / rho
                    temp_id = axes_ids['Eion']
                    axes[temp_id].plot(x, Tgas, lw=2., color='darkorange', 
                        linestyle='-', label=r'$\rm{Gas}$')
            elif key == 'Pmag':
                Pthm = profiles[f'{fnums[0]}']['Pthm'] / Volume
                Pmag = profiles[f'{fnums[0]}']['Pmag'] / Volume
                if (radiation):
                    Pmag_Ptot = Pmag / (Pthm + Prad)
                else:
                    Pmag_Ptot = Pmag / Pthm

                if (radiation):
                    ptot_label = r'$P_{\rm mag}/(P_{\rm rad} + P_{\rm thm})$'
                else:
                    ptot_label = r'$P_{\rm mag}/P_{\rm thm}$'

                axes[id].plot(x, Pmag_Ptot, lw=2., 
                              color='dodgerblue', linestyle='-', 
                              label=ptot_label)
                if (cgs):
                    Prad = profiles[f'{fnums[0]}']['Prad'] / Volume
                    Prad_Pthm = Prad / Pthm
                    axes[id].plot(x, Prad_Pthm, lw=2., color='crimson', 
                        linestyle='-', label=r'$P_{\rm rad}/P_{\rm thm}$')
            elif key == 'wuR':
                wuR   = profiles[f'{fnums[0]}']['wuR']   / Volume
                wuphi = profiles[f'{fnums[0]}']['wuphi'] / Volume
                wuz   = profiles[f'{fnums[0]}']['wuz']   / Volume
                w     = profiles[f'{fnums[0]}']['w']     / Volume
                uR    = wuR   / w
                uphi  = wuphi / w
                uz    = wuz   / w
                a     = profiles[f'{fnums[0]}']['a']
                Omega = uphi / (r0 * Rg)
                Analytic = theory.Analytic(hr.CONSTANTS, cgs=cgs)
                Omega0   = Analytic.calc_OmegaK(MBH, a, r0)
                axes[id].plot(x, uR/c,   lw=2., color='crimson',    
                              linestyle='-', label=r'$R$')
                axes[id].plot(x, uphi/c, lw=2., color='black',      
                              linestyle='-', label=r'$\varphi$')
                axes[id].plot(x, uz/c,   lw=2., color='dodgerblue', 
                              linestyle='-', label=r'$z$')
                axes[3].plot(x, Omega/Omega0,   lw=2., color='black', 
                             linestyle='-', label=r'$z$')
                axes[3].set_ylabel(r'$\Omega / \Omega_{\rm K}$', fontsize=FS)
                axes[3].set_ylim([0, 3])
                if not cgs:
                    theta     = profiles[f'{fnums[0]}']['theta']
                    wur       = profiles[f'{fnums[0]}']['wur']  / Volume
                    wuth      = profiles[f'{fnums[0]}']['wuth'] / Volume
                    if (theta_plot):
                        wuphi_sph = wuphi * np.sin(theta)
                    elif (z_plot):
                        sintheta = np.sin(np.acos(x))
                        wuphi_sph = wuphi * sintheta
                    ur    = wur   / w
                    uphi  = wuphi / w
                    uth   = wuth  / w
                    qieID = int(axes_ids['qie'])
                    axes[qieID].plot(x, ur/c,   lw=2., color='crimson', 
                                linestyle='-', label=r'$r$')
                    axes[qieID].plot(x, uphi/c, lw=2., color='black',      
                                linestyle='-', label=r'$\varphi$')
                    axes[qieID].plot(x, uth/c,   lw=2., color='dodgerblue', 
                                linestyle='-', label=r'$\theta$')
            elif key == 'qie':
                qie  = norm * profiles[f'{fnums[0]}']['qie']  / Volume
                qIC  = norm * profiles[f'{fnums[0]}']['qIC']  / Volume
                qemi = norm * profiles[f'{fnums[0]}']['qemi'] / Volume
                axes[id].plot(x, qie,  lw=2., color='darkorange', 
                              linestyle='-', label=r'$\rm{Coulomb}$')
                axes[id].plot(x, qIC,  lw=2., color='dodgerblue', 
                              linestyle='-', label=r'$\rm{Compton}$')
                axes[id].plot(x, qemi, lw=2., color='crimson',    
                              linestyle='-', label=r'$\rm{Emission}$')
            elif key == 'bR':
                bR   = profiles[f'{fnums[0]}']['bR']   / Volume
                bphi = profiles[f'{fnums[0]}']['bphi'] / Volume
                bz   = profiles[f'{fnums[0]}']['bz']   / Volume
                axes[id].plot(x, bR,   lw=2., color='crimson',    
                              linestyle='-', label=r'$R$')
                axes[id].plot(x, bphi, lw=2., color='black',      
                              linestyle='-', label=r'$\varphi$')
                axes[id].plot(x, bz,   lw=2., color='dodgerblue', 
                              linestyle='-', label=r'$z$')
            elif key == 'TEM_t1':
                TEM_tr = norm * profiles[f'{fnums[0]}']['TEM_t1']
                THD_tr = norm * profiles[f'{fnums[0]}']['THD_t1']
                axes[id].plot(x, THD_tr, lw=2., color='darkorange',      
                              linestyle='-', label=r'$-\mathcal{T}^{r}_{t, {\rm HD}}$')
                axes[id].plot(x, TEM_tr, lw=2., color='dodgerblue', 
                              linestyle='-', label=r'$-\mathcal{T}^{r}_{t, {\rm EM}}$')
                if (cgs):
                    R_tr = norm * profiles[f'{fnums[0]}']['R_t1'] 
                    axes[id].plot(x, R_tr,   lw=2., color='crimson',    
                                linestyle='-', label=r'$-\mathcal{R}^{r}_t$')

        elif (time_average):
            if (theta_plot):
                x = profiles['theta']
            elif (z_plot):
                x = profiles['z_r']
            Pmag   = profiles['Pmag']
            rho    = profiles['rho']
            sigma  = 2. * Pmag / (rho * c * c)
            if key == 'rho':
                rho = profiles['rho']
                axes[id].plot(x, rho, lw=2., color='black', linestyle='-')
            elif key == 'Eion':
                Eion = profiles['Eion']
                Eelc = profiles['Eelc']
                Prad = profiles['Prad']
                rho  = profiles['rho']
                gamma = gamma_adi
                Ti = (gamma-1.) * (1./kB) * Eion / (rho / (mu_i*mp))
                Te = (gamma-1.) * (1./kB) * Eelc / (rho / (mu_e*mp))
                Tr = (3. * Prad / hr.CONSTANTS['aRAD']) ** (1./4.)
                axes[id].plot(x, Ti, lw=2., color='darkorange', 
                              linestyle='-', label=r'$\rm{Ions}$')
                axes[id].plot(x, Te, lw=2., color='dodgerblue', 
                              linestyle='-', label=r'$\rm{Electrons}$')
                axes[id].plot(x, Tr, lw=2., color='crimson',    
                              linestyle='-', label=r'$\rm{Radiation}$')
            elif key == 'Pthm':
                Pthm = profiles['Pthm']
                Pmag = profiles['Pmag']
                axes[id].plot(x, Pthm, lw=2., color='darkorange', 
                              linestyle='-', label=r'$\rm{Thermal}$')
                axes[id].plot(x, Pmag, lw=2., color='dodgerblue', 
                              linestyle='-', label=r'$\rm{Magnetic}$')
                if (cgs):
                    Prad = profiles['Prad']
                    axes[id].plot(x, Prad, lw=2., color='crimson',    
                                linestyle='-', label=r'$\rm{Radiation}$')
                if not cgs:
                    rho  = profiles['rho']
                    Tgas = Pthm / rho
                    temp_id = axes_ids['Eion']
                    axes[temp_id].plot(x, Tgas, lw=2., color='darkorange', 
                        linestyle='-', label=r'$\rm{Gas}$')
            elif key == 'Pmag':
                Pthm = profiles['Pthm']
                Pmag = profiles['Pmag']
                if (radiation):
                    Pmag_Ptot = Pmag / (Pthm + Prad)
                else:
                    Pmag_Ptot = Pmag / Pthm

                if (radiation):
                    ptot_label = r'$P_{\rm mag}/(P_{\rm rad} + P_{\rm thm})$'
                else:
                    ptot_label = r'$P_{\rm mag}/P_{\rm thm}$'
                axes[id].plot(x, Pmag_Ptot, lw=2., 
                              color='dodgerblue', linestyle='-', 
                              label=ptot_label)
                if (cgs):
                    Prad = profiles['Prad']
                    Prad_Pthm = Prad / Pthm
                    axes[id].plot(x, Prad_Pthm, lw=2., color='crimson',    
                        linestyle='-', label=r'$P_{\rm rad}/P_{\rm thm}$')
            elif key == 'wuR':
                wuR   = profiles['wuR']
                wuphi = profiles['wuphi']
                wuz   = profiles['wuz']
                w     = profiles['w']
                uR    = wuR   / w
                uphi  = wuphi / w
                uz    = wuz   / w
                a     = profiles['a']
                Omega = uphi / (r0 * Rg)
                Analytic = theory.Analytic(hr.CONSTANTS, cgs=cgs)
                Omega0   = Analytic.calc_OmegaK(MBH, a, r0)
                axes[id].plot(x, uR/c,   lw=2., color='crimson',    
                              linestyle='-', label=r'$R$')
                axes[id].plot(x, uphi/c, lw=2., color='black',      
                              linestyle='-', label=r'$\varphi$')
                axes[id].plot(x, uz/c,   lw=2., color='dodgerblue', 
                              linestyle='-', label=r'$z$')
                axes[3].plot(x, Omega/Omega0,   lw=2., color='black', 
                             linestyle='-', label=r'$z$')
                axes[3].set_ylabel(r'$\Omega / \Omega_{\rm K}$', fontsize=FS)
                axes[3].set_ylim([0, 3])
                if not cgs:
                    theta     = profiles['theta']
                    wur       = profiles['wur']  / Volume
                    wuth      = profiles['wuth'] / Volume
                    wuphi_sph = wuphi * np.sin(theta)
                    ur    = wur   / w
                    uphi  = wuphi / w
                    uth   = wuth  / w
                    qieID = int(axes_ids['qie'])
                    axes[qieID].plot(x, ur/c,   lw=2., color='crimson', 
                                linestyle='-', label=r'$r$')
                    axes[qieID].plot(x, uphi/c, lw=2., color='black',      
                                linestyle='-', label=r'$\varphi$')
                    axes[qieID].plot(x, uth/c,   lw=2., color='dodgerblue', 
                                linestyle='-', label=r'$\theta$')
            elif key == 'qie':
                qie  = norm * profiles['qie']
                qIC  = norm * profiles['qIC']
                qemi = norm * profiles['qemi']
                axes[id].plot(x, qie,  lw=2., color='darkorange', 
                              linestyle='-', label=r'$\rm{Coulomb}$')
                axes[id].plot(x, qIC,  lw=2., color='dodgerblue', 
                              linestyle='-', label=r'$\rm{Compton}$')
                axes[id].plot(x, qemi, lw=2., color='crimson',    
                              linestyle='-', label=r'$\rm{Emission}$')
            elif key == 'bR':
                bR   = profiles['bR']
                bphi = profiles['bphi']
                bz   = profiles['bz']
                axes[id].plot(x, bR,   lw=2., color='crimson',    
                              linestyle='-', label=r'$R$')
                axes[id].plot(x, bphi, lw=2., color='black',      
                              linestyle='-', label=r'$\varphi$')
                axes[id].plot(x, bz,   lw=2., color='dodgerblue', 
                              linestyle='-', label=r'$z$')
            elif key == 'TEM_t1':
                TEM_tr = norm * profiles['TEM_t1']
                THD_tr = norm * profiles['THD_t1']
                axes[id].plot(x, THD_tr, lw=2., color='darkorange',      
                              linestyle='-', label=r'$-\mathcal{T}^{r}_{t, {\rm HD}}$')
                axes[id].plot(x, TEM_tr, lw=2., color='dodgerblue', 
                              linestyle='-', label=r'$-\mathcal{T}^{r}_{t, {\rm EM}}$')
                if (cgs):
                    R_tr = norm * profiles['R_t1'] 
                    axes[id].plot(x, R_tr,   lw=2., color='crimson',    
                                linestyle='-', label=r'$-\mathcal{R}^{r}_t$')
            
        axes[id].set_ylim(lims)
        axes[id].set_ylabel(label, fontsize=FS)
        if not cgs and key == 'rho':
            # Set temperature plot when key == 'rho' since 'Eion' not in keys
            temp_id = axes_ids['Eion']
            axes[temp_id].set_ylim(limits['Eion'])
            axes[temp_id].set_yscale('log')
            axes[temp_id].set_ylabel(plot_labels['Eion'], fontsize=FS)
        if not cgs and key == 'wuR':
            # Set spherical-coordinate velocity plot when key == 'wuR'
            qie_id = axes_ids['qie']
            axes[qie_id].set_ylim(limits['wuR'])
            axes[qie_id].set_ylabel(plot_labels['wuR'], fontsize=FS)
            axes[qie_id].legend(loc='lower left', fontsize=FS-4, frameon=False)

        if (theta_plot):
            theta_min, theta_max = get_jet_bounds(x, sigma, sigma_cut=1.)
        elif (z_plot):
            z_min, z_max = get_jet_bounds(x, sigma, sigma_cut=1, z=True)
        
        if (legend_loc[f'{key}']):
            axes[id].legend(loc=legend_loc[f'{key}'], fontsize=FS-4, frameon=False)

        if (log and not linthresh[f'{key}']):
            axes[id].set_yscale('log')
        elif (log and linthresh[f'{key}']):
            axes[id].set_yscale('symlog', linthresh=linthresh[f'{key}'])

    for ax in axes:
        if (theta_plot):
            xticklabels = [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
            ax.set_xlabel(r'$\theta$', fontsize=FS)
            ax.set_xticks([0., (1./4.)*np.pi, (1./2.)*np.pi, (3./4.)*np.pi, np.pi])
            ax.set_xticklabels(xticklabels, fontsize=FS)
        elif (z_plot):
            ax.set_xlabel(r'$z/R$', fontsize=FS)
        cv.grid(ax, alpha=0.2, color='black')

        ymin, ymax = ax.get_ylim()
        if (theta_plot):
            ax.fill_between(x, ymin, ymax, where=x < theta_min, 
                            facecolor='gray', alpha=0.2)
            ax.fill_between(x, ymin, ymax, where=x > theta_max, 
                            facecolor='gray', alpha=0.2)

    print("Saving figure ...", flush=True)
    fig.tight_layout()
    plt.savefig(f'{save_dir}/{plot_title}.png', format='png')
    plt.close()

def get_vols(key):
    if 'Mdot' in key or 'Edot' in key:
        return False
    elif key == 'TEM_t1' or key == 'THD_t1' or key == 'R_t1':
        return False
    elif (key == 'Phi_BH' or key == 'Phi_mid' or 
          key == 'Phi_sph_upp' or key == 'Phi_sph_low'):
        return False
    else:
        return True

def get_norms(key, **kwargs):
    cgs      = kwargs.get('cgs', False)
    MBH      = kwargs.get('MBH', 1.0)
    Analytic = theory.Analytic(hr.CONSTANTS, cgs=cgs)
    L_Edd    = Analytic.calc_LEdd(MBH=MBH) if cgs else 1.
    Rg       = kwargs.get('Rg', 1.)        if cgs else 1.
    c        = hr.CONSTANTS['c']           if cgs else 1.
    if cgs and 'MBH' not in kwargs:
        raise ValueError("Must supply MBH for cgs=True")
    if (key == 'TEM_t1'):
        return -Rg * Rg * c / L_Edd
    elif (key == 'qie'):
        return -(Rg * Rg * Rg / L_Edd)
    else:
        return 1.

def get_logs(key):
    if (key == 'wuR'):
        return False
    else:
        return True

def get_linthresh(key, **kwargs):
    cgs = kwargs.get('cgs', False)
    if (key == 'bR'):
        return 1.e7 if cgs else 1.e-3
    elif (key == 'TEM_t1'):
        return 1.e-1
    else:
        return False
    
def get_extend(key):
    if key == 'Pmag':
        return 'min'
    else:
        return 'both'
    
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
    parser.add_argument('--time_average', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""boolean to determine if profiles from
                        --start_file to --end_file should be averaged.""")
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
    parser.add_argument('--plot_title', type=str, required=False, 
                        default='Vertical_Profiles', 
                        help="""title of PDF output""")
    args = parser.parse_args()
    main(**vars(args))