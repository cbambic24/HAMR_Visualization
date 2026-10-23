import sys
import os
import glob
import argparse
import numpy as np
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{file_dir}')

import compute_variables as cv
import analysis as ana
import coordinates as coordinates
import radiation as rad
import hamr_read as hr
import theory as theory
sys.path.insert(0, os.path.join(cv.vis_path, "Plot_HAMR_Data"))
from mpi4py import MPI
from mpi4py.util import pkl5
  
# Mass-accretion and energy-accretion rates
Mdot_keys = ['Wind', 'Acc']

Edot_keys = [
    'Jet', 'Wind_Enthalpy', 'Wind_Poynting', 
    'Radiation', 'Radiation_Flux_Disk', 'Radiation_Wind'
]

Mdot = dict.fromkeys(Mdot_keys)
Edot = dict.fromkeys(Edot_keys)

# Radiative Fluxes
Fluxes = dict.fromkeys(['Obs', 'Fluid'])
 
timescale_keys = [
    'Inflow', 'Thermal', 'Diffusion', 'Diffusion_Approximation',
    'Cooling', 'Coulomb', 'Compton'
]

Timescales = dict.fromkeys(timescale_keys)

keys = [
    'rho', 'w', 'Eion', 'Eelc', 
    'rhou1', 'THD_t1', 'TEM_t1', 
    'R_tt',  'R_tR',   'R_tz',    'R_t1',
    'FRad_R_Fluid', 'FRad_z_Fluid',
    'wur', 'wuR',  'wuz', 'wut',  'Be',  'Be_Rad',
    'Te',  'Pmag', 'Prad', 'Vol', 'Vol_sigma_cut',
    'qie', 'qemi', 'qabs', 'qIC'
]

def main(**kwargs):
    sim       = kwargs.get('sim')
    reduced   = kwargs.get('reduced')
    overwrite = kwargs.get('overwrite')
    plot_data = kwargs.get('plot_data')

    averages_dir = cv.user_path + f'/Averages/{sim}'
    disk_dir = cv.get_save_dir(sim, 'Averages', 'Disk',          reduced=reduced)
    mdot_dir = cv.get_save_dir(sim, 'Averages', 'Mass_Fluxes',   reduced=reduced)
    edot_dir = cv.get_save_dir(sim, 'Averages', 'Energy_Fluxes', reduced=reduced)
    time_dir = cv.get_save_dir(sim, 'Averages', 'Timescales',    reduced=reduced)
    os.makedirs(disk_dir, exist_ok=True)
    os.makedirs(mdot_dir, exist_ok=True)
    os.makedirs(edot_dir, exist_ok=True)
    os.makedirs(time_dir, exist_ok=True)
    save_dirs = {'Mdot': mdot_dir, 'Edot': edot_dir, 
                 'time': time_dir, 'disk': disk_dir}
    
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    size = comm.Get_size()
    rank = comm.Get_rank()

    mode, fnums, last_file = select_mode(**kwargs)
    load_args = {
        'keys'      : keys,
        'fnums'     : fnums,
        'mode'      : mode,
        'last_file' : last_file
    }
    if mode == 'file_list' or mode == 'file_num':
        file_list_args = {
            'file_list'  : fnums,
            'size'       : size,
            'rank'       : rank, 
            'data_dir'   : averages_dir,
            'file_type'  : 'Slices',
            'reduced'    : reduced,
            'stride'     : kwargs.get('stride')
        }
        file_list_local = cv.get_file_list_local(**file_list_args)

        if not overwrite:
            file_list_local = get_unwritten(file_list_local, **file_list_args)

        for file_num in file_list_local:
            load_args['fnums'] = [file_num]
            slices, extra = load_slices_by_mode(**load_args, **kwargs)
            disk, masks = calc_disk_mask(slices, extra, fnums=fnums, **kwargs)
            masks['fnum_str'] = str(file_num).zfill(len(str(last_file)))
            calc_Mdot_Edot( slices, extra, disk, save_dirs, **masks, **kwargs)
            calc_timescales(slices, extra, disk, save_dirs, **masks, **kwargs)
            if plot_data:
                import plot_timescales as pt
                plt_args = {
                    'slices'      : slices,
                    'extra'       : extra,
                    'disk'        : disk,
                    'surface'     : Fluxes['Obs']['surface'],
                    'movie_title' : 'Timescales'
                }
                pt.plot_data(Mdot=Mdot, Edot=Edot, Timescales=Timescales, 
                            **masks, **plt_args, **kwargs)
    else:
        # file_average or use_file
        slices, extra = load_slices_by_mode(**load_args, **kwargs)
        disk, masks = calc_disk_mask(slices, extra, fnums=fnums, **kwargs)
        calc_Mdot_Edot( slices, extra, disk, save_dirs, **masks, **kwargs)
        calc_timescales(slices, extra, disk, save_dirs, **masks, **kwargs)
        if plot_data:
            import plot_timescales as pt
            plt_args = {
                'slices'      : slices,
                'extra'       : extra,
                'disk'        : disk,
                'surface'     : Fluxes['Obs']['surface'],
                'movie_title' : 'Timescales_Averaged'
            }
            pt.plot_data(Mdot=Mdot, Edot=Edot, Timescales=Timescales, 
                         **masks, **plt_args, **kwargs)

def calc_Mdot_Edot(slices, extra, disk, save_dirs, **kwargs):
    """Mass and energy accretion-outflow rates.
    Calculates mass and energy outflow rates based on 2D slices and
    disk definition. Must be called before calc_timescales.
    """
    # Photosphere/Zero-Bernoulli surface masks
    tau_es = kwargs.get('tau_es')
    Be     = kwargs.get('Be')
    sigma  = kwargs.get('sigma')
    # Options for overriding "disk" definition
    use_cyl   = kwargs.get('use_cyl')
    use_photo = kwargs.get('use_photo')
    use_bound = kwargs.get('use_bound')
    use_HoR   = kwargs.get('use_HoR')
    # For optically thin flows
    opt_thin  = kwargs.get('opt_thin')

    # Constants
    Rg   = kwargs.get('Rg')
    c    = kwargs.get('c')
    Rg2  = Rg * Rg
    Rg2c = Rg * Rg * c
    
    # Inflow vs. outflow
    Vol = slices['Vol']
    wur = slices['wur'] / Vol
    wut = slices['wut'] / Vol
    vr  = wur / wut
    
    # Mass-accretion and mass-outflow rates
    dx2         = slices['dx2']
    outflow     = (vr > 0.)
    bound       = (Be < 0.)
    unbound     = (Be > 0.)
    opt_thick   = (tau_es > 1.)
    jet         = unbound * (sigma > 1.) * outflow
    wind        = unbound * (sigma < 1.) * outflow
    not_jet     = (sigma < 1.) 

    Mdot['Acc']  = Rg2 * (-slices['rhou1'] * bound  * not_jet * dx2).sum(-1)
    Mdot['Wind'] = Rg2 * (slices['rhou1']  * wind             * dx2).sum(-1)

    # Energy-outflow rates
    gTHD_t1 = slices['THD_t1']
    gTEM_t1 = slices['TEM_t1']
    gR_t1   = slices['R_t1']
    Edot['Jet']           = Rg2c * (-gTEM_t1 * jet            * dx2).sum(-1)
    Edot['Wind_Enthalpy'] = Rg2c * (-gTHD_t1 * wind           * dx2).sum(-1)
    Edot['Wind_Poynting'] = Rg2c * (-gTEM_t1 * wind           * dx2).sum(-1)
    Edot['Radiation']     = Rg2c * (-gR_t1                    * dx2).sum(-1)
    Edot['Radiation_Wind']= Rg2c * (-gR_t1 * wind * opt_thick * dx2).sum(-1)
    
    # Radiative efficiency integrated over the disk surface
    flux_args = {
        'slices'    : slices,
        'extra'     : extra,
        'disk'      : disk,
        'cyl'       : use_cyl
    }
    if use_cyl:
        R2D, Z2D, _ = interpolate_sph_to_cyl(slices['rho'], slices, extra)
        flux_args['R_cyl'] = R2D
        flux_args['Z_cyl'] = Z2D
    flux_args['flux_R'] = -(slices['R_tR'] / slices['Vol']) * c
    flux_args['flux_Z'] = -(slices['R_tz'] / slices['Vol']) * c

    if opt_thin: 
        Edot['Radiation_Flux_Disk'] = np.zeros_like(slices['r'])
    else:
        if use_HoR:
            flux_args['use_photo'] = True
            flux_args['tau']       = tau_es
            flux_args['Be']        = Be
        Fluxes['Obs'] = calc_radiative_flux(**flux_args)
        Flux_total    = Rg2 * (Fluxes['Obs']['top'] + Fluxes['Obs']['bot'])
        Edot['Radiation_Flux_Disk'] = Flux_total
        
    flux_args['flux_R'] = slices['FRad_R_Fluid'] / slices['Vol']
    flux_args['flux_Z'] = slices['FRad_z_Fluid'] / slices['Vol']
    if use_HoR:
        flux_args['use_photo'] = False
    Fluxes['Fluid'] = calc_radiative_flux(**flux_args)
     
    Mdot['r'] = slices['r']
    Edot['r'] = slices['r']

    # Save Mdot and Edot
    if Mdot['t'] != 'Averaged' and Edot['t'] != 'Averaged':
        fnum_str = kwargs.get('fnum_str')
        np.savez(f"{save_dirs['Mdot']}/Mass_Fluxes_{fnum_str}",   **Mdot)
        np.savez(f"{save_dirs['Edot']}/Energy_Fluxes_{fnum_str}", **Edot)
    else:
        np.savez(f"{save_dirs['Mdot']}/Mass_Fluxes_Averaged",   **Mdot)
        np.savez(f"{save_dirs['Edot']}/Energy_Fluxes_Averaged", **Edot)

def calc_timescales(slices, extra, disk, save_dirs, **kwargs): 
    """Thermodynamic timescales averaged over the disk.
    Calculates all thermodynamic and radiative timescales based on
    disk definition. Must be called after calc_Mdot_Edot.
    """
    # Photosphere/Zero-Bernoulli surface masks
    tau_es = kwargs.get('tau_es')
    Be     = kwargs.get('Be')
    sigma  = kwargs.get('sigma')
    # Options for overriding "disk" definition
    use_cyl   = kwargs.get('use_cyl')
    use_photo = kwargs.get('use_photo')
    use_bound = kwargs.get('use_bound')
    use_HoR   = kwargs.get('use_HoR')
    # For optically thin flows
    opt_thin  = kwargs.get('opt_thin')
    # Use qrad from the heating slices to calculate tcool
    heating   = kwargs.get('heating')
    
    # Constants
    Rg   = kwargs.get('Rg')
    c    = kwargs.get('c')
    Rg2  = Rg * Rg
    Rg2c = Rg * Rg * c
    
    timescale_args = {
        'slices' : slices,
        'extra'  : extra,
        'disk'   : disk,
        'cyl'    : use_cyl,
        'Rg'     : Rg,
        'c'      : c
    }
    
    # Inflow timescale
    t_inf  = calc_tinflow(**timescale_args)

    # Thermal (cooling) timescale (based on cooling rates)
    timescale_args['heating'] = True if heating else False
    t_cool = calc_tcool(rate='Total', **timescale_args)

    # Inverse Compton (cooling) timescale 
    t_IC = calc_tcool(rate='Compton', **timescale_args)

    # Coulomb (cooling) timescale 
    t_ie = calc_tcool(rate='Coulomb', **timescale_args)

    if opt_thin:
        t_thm = t_cool
    else:
        timescale_args['Flux_dS'] = Fluxes['Obs']
        t_thm = calc_tthm(**timescale_args)
    
    if opt_thin:
        t_diff = t_cool
    else:
        timescale_args['Flux_dS'] = Fluxes['Fluid']
        t_diff = calc_tthm(**timescale_args)

    # Approximation for the radiative diffusion timescale 
    nth_2   = int(tau_es.shape[1]/2)
    tau_mid = 0.5 * (tau_es[:, nth_2-1] + tau_es[:, nth_2])
    Hz      = slices['r'] * calc_HoR(slices, extra)
    tdiff_approx = 3. * tau_mid * Hz * Rg / c

    Timescales['Inflow']    = t_inf
    Timescales['Thermal']   = t_thm
    Timescales['Cooling']   = t_cool
    Timescales['Diffusion'] = t_diff
    #Timescales['Diffusion'] = np.zeros_like(tdiff_approx)
    Timescales['Diffusion_Approximation'] = tdiff_approx
    Timescales['Compton']   = t_IC
    Timescales['Coulomb']   = t_ie

    Timescales['r'] = slices['r']
    disk_mask = {
        'disk'   : disk, 
        'tau_es' : tau_es,
        'Be'     : Be,
        'sigma'  : sigma,
        'r'      : slices['r'], 
        'theta'  : slices['theta']
    }
 
    # Save Timescales
    if Timescales['t'] != 'Averaged':
        fnum_str = kwargs.get('fnum_str')
        np.savez(f"{save_dirs['time']}/Timescales_{fnum_str}", **Timescales)
        np.savez(f"{save_dirs['disk']}/Disk_{fnum_str}",       **disk_mask)
    else:
        np.savez(f"{save_dirs['time']}/Timescales_Averaged",   **Timescales)
        np.savez(f"{save_dirs['disk']}/Disk_Averaged",         **disk_mask)
    
def calc_disk_mask(slices, extra, **kwargs):
    """Calculate disk.
    Disk is defined as the deepest of the photosphere or 
    zero-Bernoulli surface. Allows for use of photosphere (use_photo)
    or zero-Bernoulli surface (use_bound) or scaleheight (use_HoR) as
    an override option.
    """
    use_cyl   = kwargs.get('use_cyl')
    use_photo = kwargs.get('use_photo')
    use_bound = kwargs.get('use_bound')
    use_HoR   = kwargs.get('use_HoR')
    radiation = kwargs.get('radiation')
    two_temp  = kwargs.get('two_temp')
    MBH       = kwargs.get('MBH')
    fnums     = kwargs.get('fnums')
    
    # Define constants
    if (two_temp or radiation):
        Msun = np.float32(hr.CONSTANTS['Msun'])
        c    = np.float32(hr.CONSTANTS['c'])
        G    = np.float32(hr.CONSTANTS['G'])
        Rg   = G * MBH * Msun / (c * c)
    else:
        c, G, Rg = 1., 1., 1.

    # Calculate photosphere
    photo_args = {
        'slices' : slices,
        'extra'  : extra,
        'cyl'    : use_cyl, 
        'Rg'     : Rg,
        'c'      : c
    }
    tau_es = calc_photosphere(**photo_args)

    # Calculate Bernoulli parameter
    Vol  = slices['Vol']
    rho  = slices['rho'] / Vol
    wu_t = (1. / c) * slices['Be_Rad'] / Vol
    Be   = (-wu_t / (rho * c * c)) - 1.

    # Calculate sigma for jet/wind
    sigma = calc_sigma(slices, c=c)

    # Disk is demarcated by the deepest of the scattering photosphere
    # or the zero-Bernoulli surface
    disk_args              = photo_args.copy()
    disk_args['tau']       = tau_es
    disk_args['Be']        = Be
    disk_args['use_photo'] = use_photo
    disk_args['use_bound'] = use_bound
    if use_HoR:
        th0  = calc_HoR(slices, extra)
        th1D = slices['theta']
        disk = ((th1D[None, :] > (np.pi/2. - th0[:, None])) *
                (th1D[None, :] < (np.pi/2. + th0[:, None])))
        if use_cyl:
            _, _, disk_cyl = interpolate_sph_to_cyl(disk, slices, extra)
            disk = disk_cyl[:, :, 0]
            disk[disk != 0] = 1
    else:
        disk = calc_disk(**disk_args)

    masks = {
        'tau_es' : tau_es,
        'Be'     : Be,
        'sigma'  : sigma,
        'Rg'     : Rg,
        'c'      : c
    }
    return disk, masks

def load_slices_by_mode(keys, fnums, mode, last_file, **kwargs):
    sim          = kwargs.get('sim')
    reduced      = kwargs.get('reduced')
    time_average = kwargs.get('time_average')
    heating      = kwargs.get('heating')
    # Load slices
    Slice        = cv.Slices()
    res_str      = 'Full_Resolution' if not reduced else 'Reduced_Resolution'
    averages_dir = f'{cv.user_path}/Averages/{sim}'
    file_num = fnums[0] if fnums else -1

    file_info = {
        'averages_dir' : averages_dir,
        'file_number'  : -1,
        'start_file'   : max(file_num - int(time_average/2), 0),
        'end_file'     : min(file_num + int(time_average/2), last_file),
        'reduced'      : reduced
    }

    if mode == 'use_file':
        file_name   = kwargs.get('use_file', 'Slices_Time_Averaged_All')
        load_file   = f'{averages_dir}/{res_str}/{file_name}/{file_name}.npz'
        extra_keys  = ['phi', 'dx3', 'rho0']
        slices      = dict.fromkeys(keys)
        extra       = dict.fromkeys(extra_keys)
        with np.load(load_file) as f:
            for key in keys + ['r', 'theta', 'a', 'dx1', 'dx2']:
                slices[key] = f[key]
            for key in extra_keys:
                extra[key] = f[key]
        if heating:
            file_name = 'Heating_Slices_Time_Averaged'
            load_file = f'{averages_dir}/{res_str}/{file_name}/{file_name}.npz'
            with np.load(load_file) as f:
                for key in ['qrad']:
                    slices[key] = f[key]
        Mdot['t']       = 'Averaged'
        Edot['t']       = 'Averaged'
        Timescales['t'] = 'Averaged'
        return slices, extra

    # file_num, file_list, or file_average
    if mode == 'file_num' or mode == 'file_list':
        fnums = Slice.get_fnums(**file_info)

    vols     = dict.fromkeys(keys, False)
    slice_args = {
        'averages_dir' : averages_dir,
        'fnums'        : fnums,
        'keys'         : keys,
        'reduced'      : reduced,
        'vols'         : vols
    }
    slices = Slice.load_slices_time_averaged(**slice_args)

    slice_args['keys'] = ['phi', 'dx3', 't', 'rho0']
    extra = Slice.load_slices(**slice_args)[str(file_num)]

    t = extra['t'] if mode in ('file_num', 'file_list') else 'Averaged'
    Mdot['t']       = t
    Edot['t']       = t
    Timescales['t'] = t
    return slices, extra

def select_mode(**kwargs):
    """
    4 modes available 
    use_file: use the file specified by "--use_file" in Averages/sim/res/
    file_num: use the dump number file_num with or without moving time window
    file_average: average fstart to fend (or last file if not specified)
    file_list: process dumps from start_file to end_file in parallel
    """
    sim          = kwargs.get('sim')
    reduced      = kwargs.get('reduced')
    use_file     = kwargs.get('use_file')
    file_num     = int(kwargs.get('file_num'))
    start_file   = int(kwargs.get('start_file'))
    end_file     = int(kwargs.get('end_file'))
    fstart       = kwargs.get('fstart')
    fend         = kwargs.get('fend')
    modes = []
    if (use_file):                           modes.append('use_file')
    if (file_num != -1):                     modes.append('file_num')
    if (fstart != -1 or fend != -1):         modes.append('file_average')
    if (start_file != -1 or end_file != -1): modes.append('file_list')
    if (len(modes) > 1):
        raise ValueError("Only one averaging mode may be specified at a "
                          f"time; got: {', '.join(modes)}.")
    if (fstart == -1 and fend != -1):
        raise ValueError("Do not specify fend without setting fstart.")
    if (len(modes) == 0):
        raise ValueError("Must specify a mode (use_file, file_num, etc.).")
 
    averages_dir = f'{cv.user_path}/Averages/{sim}'
    res_str   = 'Full_Resolution' if not reduced else 'Reduced_Resolution'
    slice_dir = f'{averages_dir}/{res_str}/Slices'
    last_file = get_last_file_num(slice_dir)
    fend      = min(fend, last_file)

    mode = modes[0]
    match mode:
        case 'use_file':
            fnums = False
        case 'file_num':
            fnums = [file_num]
        case 'file_average':
            fnums = list(range(fstart, fend + 1))
        case 'file_list':
            fnums = list(range(start_file, end_file + 1))
        case _:
            raise ValueError("Invalid mode option.")

    return mode, fnums, last_file

def get_unwritten(list_of_files, **kwargs):
    # Use get_file_list_local as if on rank 0 with 1 core
    exist_args = {
        'file_list' : list_of_files,
        'size'      : 1,
        'rank'      : 0,
        'data_dir'  : kwargs.get('data_dir'),
        'file_type' : 'Timescales',
        'reduced'   : kwargs.get('reduced')
    }
    existing_profiles = cv.get_file_list_local(**exist_args)
    existing_set = set(existing_profiles)
    file_list    = [f for f in list_of_files if f not in existing_set]
    return file_list
 
def calc_disk(tau, Be, slices, extra, **kwargs):
    """Calculate Disk based on optical depth and Bernoulli parameter
    Disk is defined as the deepest of the photosphere or 
    zero-Bernoulli surface. Allows for use of photosphere (use_photo)
    or zero-Bernoulli surface (use_bound) as override options.
    """
    # Whether to remap to cylindrical coordinates
    cyl       = kwargs.get('cyl', False)
    use_photo = kwargs.get('use_photo', False)
    use_bound = kwargs.get('use_bound', False)
    
    if cyl: 
        z_photo = calc_surface(tau, slices, extra, photosphere=True, cyl=True)
        z_bound = calc_surface(Be,  slices, extra, Bernoulli=True,   cyl=True)
        if use_photo:
            z_bound = z_photo
        elif use_bound:
            z_photo = z_bound
    else:
        th_photo = calc_surface(tau, slices, extra, photosphere=True, cyl=False)
        th_bound = calc_surface(Be,  slices, extra, Bernoulli=True,   cyl=False)
        if use_photo:
            th_bound = th_photo
        elif use_bound:
            th_photo = th_bound

    # Correct for gaps
    if cyl:
        disk = z_photo['all'] * z_bound['all']
        disk[disk != 0] = 1
        coords = np.indices(disk.shape)
        # Get indices in z-direction (poloidal direction)
        masked_indices = np.where(disk, coords[0], np.nan)
        idx_top = np.nanmax(masked_indices, axis=0)
        idx_bot = np.nanmin(masked_indices, axis=0)
    else: 
        disk = th_photo['all'] * th_bound['all']
        disk[disk != 0] = 1
        disk[th_bound['top'] < 0, :] = 0
        disk[th_bound['bot'] < 0, :] = 0
        coords = np.indices(disk.shape)
        # Get indices in th-direction (poloidal direction)
        masked_indices = np.where(disk, coords[1], np.nan)
        idx_bot = np.nanmax(masked_indices, axis=1)
        idx_top = np.nanmin(masked_indices, axis=1)

    idx_top_int = np.nan_to_num(idx_top).astype(int)
    idx_bot_int = np.nan_to_num(idx_bot).astype(int)

    if cyl:
        z_indices = coords[0]
        disk = ((z_indices >= idx_bot_int[None, :]) & 
                (z_indices <= idx_top_int[None, :]))
    else:
        th_indices = coords[1]
        disk = ((th_indices >= idx_top_int[:, None]) & 
                (th_indices <= idx_bot_int[:, None]))

    return disk.astype(int)

def calc_surface(q, slices, extra, **kwargs):
    """Calculate photosphere or zero-Bernoulli surface.
    If photosphere = True: photosphere, if Bernoulli=True, zero-Bernoulli
    """
    photo = kwargs.get('photosphere', True)
    Bernu = kwargs.get('Bernoulli',  False)
    if (Bernu):
        photo = False

    r1D   = slices['r']
    r_max = np.amax(r1D)
    surf = dict.fromkeys(['top', 'bot'])
    for side in ['top', 'bot']:
        surf[side] = np.zeros_like(r1D)

    # Whether to find surface height in z (cyl = True) 
    # or in theta (cyl = False)
    cyl = kwargs.get('cyl', False)

    # Define coordinates
    if cyl:
        R2D, Z2D, q_cyl = interpolate_sph_to_cyl(q, slices, extra)
        dz = Z2D[1:, 0] - Z2D[:-1, 0]
        dz = np.insert(dz, -1, dz[-1])
    else:
        th = (slices['theta'][None, :]) * np.ones_like(slices['r'][:, None])
    
    if cyl:
        # optically thick if photo else bound outflow 
        condition = (q_cyl[:, :, 0] > 1.) if photo else (q_cyl[:, :, 0] < 0.)
    else:
        condition = (q > 1.) if photo else (q < 0.)

    if cyl:
        # Get indices in z-direction (poloidal direction)
        pol_indices = np.indices(condition.shape)[0]
        masked_indices = np.where(condition, pol_indices, np.nan)
        idx_top  = np.nanmax(masked_indices, axis=0)
        idx_bot  = np.nanmin(masked_indices, axis=0)
        ridx     = np.arange(condition.shape[1]) 
    else: 
        # Get indices in th-direction (poloidal direction)
        pol_indices = np.indices(condition.shape)[1]
        masked_indices = np.where(condition, pol_indices, np.nan)
        idx_bot  = np.nanmax(masked_indices, axis=1)
        idx_top  = np.nanmin(masked_indices, axis=1)
        ridx     = np.arange(condition.shape[0]) 

    # Check for fully empty columns
    is_empty = np.isnan(idx_top)
    idx_top_int = np.nan_to_num(idx_top).astype(int)
    idx_bot_int = np.nan_to_num(idx_bot).astype(int)

    if cyl :
        surf['all'] = q_cyl[:, :, 0] * condition
        surf['top'] = Z2D[idx_top_int, ridx]
        surf['bot'] = Z2D[idx_bot_int, ridx]
    else:
        out_str = 'photosphere' if photo else 'zero-Bernoulli'
        surf['all']   = q * condition
        surf['top']   = th[ridx, idx_top_int]
        surf['bot']   = th[ridx, idx_bot_int]

    # If no disk, assume midplane for surface
    if photo:
        surf['top'][is_empty] = 0.0 if cyl else np.pi/2.
        surf['bot'][is_empty] = 0.0 if cyl else np.pi/2.
    else:
        surf['top'][is_empty] = r_max if cyl else -1
        surf['bot'][is_empty] = r_max if cyl else -1

    return surf

def calc_photosphere(slices, extra, **kwargs):
    # Photosphere  
    # Whether to calculate photosphere in cylindrical coordinates
    cyl = kwargs.get('cyl', False)
    Rg  = kwargs.get('Rg', 1.)
    c   = kwargs.get('c',  1.)

    # Calculate metric
    Coords      = get_Coords(slices, extra, cgs=False)
    Kerr_Schild = get_KerrSchild(slices, extra, cgs=False)
    gcov_KS     = Kerr_Schild.calc_gcov()
    gcov_int    = Coords.sph_to_int_cotnsr(gcov_KS)
    if cyl:
        gcov_cyl = Coords.sph_to_cyl_cotnsr(gcov_KS)
        gcovZZ   = gcov_cyl[3, 3]

    # Variables needed for sigma cut
    sigma  = calc_sigma(slices, c=c)

    # Calculate electron-scattering opacity
    kappa_es = rad.Opacity.calc_kappa_es(hr.CONSTANTS['X_AB'])

    # Optical depth to electron scattering
    Integral = get_Integral(slices, extra)
    rho      = slices['rho'] / slices['Vol']
    rho_cut  = np.copy(rho)
    rho_cut[sigma > 3.] = 0.
    if cyl:
        # z-integrated photosphere
        tau_es = Rg * Integral.cum_integrate_z(
            kappa_es * rho_cut[None,:,:,None], gcovZZ)[0,:,:,0]
    else:
        # theta-integrated photosphere
        tau_es = Rg * Integral.cum_integrate_th(
            kappa_es * rho_cut[None,:,:,None], gcov_int[2, 2])[0,:,:,0]
    return tau_es
        
def calc_HoR(slices, extra):
    theta   = slices['theta'][None, :]
    grhodx3 = slices['rho']
    dx2     = slices['dx2']
    HoR2 = ((np.square(theta - np.pi/2.) * grhodx3 * dx2).sum(-1) / 
            (grhodx3 * dx2).sum(-1))
    return np.sqrt(HoR2)

def get_Coords(slices, extra, **kwargs):
    coord_args  = {
        'r'         : slices['r'][None, :, None, None],
        'theta'     : slices['theta'][None, None, :, None],
        'phi'       : extra['phi'][None, None, None, :],
        'pars'      : get_params(slices, extra),
        'CONSTANTS' : hr.CONSTANTS,
        'cgs'       : kwargs.get('cgs', False)
    }
    return coordinates.Coordinates(**coord_args)

def get_KerrSchild(slices, extra, **kwargs):
    coord_args  = {
        'r'         : slices['r'][None, :, None, None],
        'theta'     : slices['theta'][None, None, :, None],
        'phi'       : extra['phi'][None, None, None, :],
        'pars'      : get_params(slices, extra),
        'CONSTANTS' : hr.CONSTANTS,
        'cgs'       : kwargs.get('cgs', False)
    }
    return coordinates.Kerr_Schild(**coord_args)

def get_Integral(slices, extra):
    # Coordinate arrays
    r      = slices['r'][None, :, None, None]
    theta  = slices['theta'][None, None, :, None]
    phi    = extra['phi'][None, None, None, 0:1]

    # Parameters typically contained in H.params
    params = get_params(slices, extra)

    # Initialize Integrals class
    integral_args = {
        'r'         : r, 
        'theta'     : theta, 
        'phi'       : phi, 
        'pars'      : params, 
        'CONSTANTS' : hr.CONSTANTS
    }
    return coordinates.Integrals(**integral_args)

def interpolate_sph_to_cyl(q, slices, extra):
    Integral = get_Integral(slices, extra)
    
    # Coordinate arrays
    r1D   = slices['r']
    th1D  = slices['theta']
    
    dth = th1D[1] - th1D[0]
    interp_args = {
        'q'    : q[None, :, :, None],
        'rdth' : r1D[0] * dth
    }
    return Integral.interpolate_sph_to_cyl_3D(**interp_args)

def get_params(slices, extra):
    params = {
        'dx1' : slices['dx1'],
        'dx2' : slices['dx2'],
        'dx3' : extra['dx3'],
        'a'   : slices['a']
    }
    return params

def calc_sigma(slices, **kwargs):
    # Speed of light
    c = kwargs.get('c', 1.)

    # Variables needed for sigma cut
    Vol    = slices['Vol']
    rho    = (slices['rho']  / Vol)
    Pmag   = (slices['Pmag'] / Vol)
    sigma  = 2. * Pmag / (rho * c * c)
    return sigma

def calc_tinflow(slices, extra, disk, **kwargs):
    """Inflow timescale
    If Rg supplied, then CGS units applied.
    """
    Rg = kwargs.get('Rg', 1.)
    c  = kwargs.get('c',  1.)
    
    # Whether to perform calculation in cylindrical vs. spherical coordinates
    cyl = kwargs.get('cyl', False)
    r1D = slices['r']
    
    if cyl:
        w   = slices['w'] / slices['Vol']
        wur = slices['wur'] / slices['Vol']
        wut = slices['wut'] / slices['Vol']
        R2D, Z2D, w_cyl = interpolate_sph_to_cyl(w, slices, extra)
        _, _, wur_cyl   = interpolate_sph_to_cyl(wur, slices, extra)
        _, _, wut_cyl   = interpolate_sph_to_cyl(wut, slices, extra)
        dz = Z2D[1:, 0] - Z2D[:-1, 0]
        dz = np.insert(dz, -1, dz[-1])
        vr = (np.sum(wur_cyl * disk * dz[:, None], axis=0) / 
              np.sum(wut_cyl * disk * dz[:, None], axis=0))
    else:
        # Factor of sqrt(-g) maintained (g) for integration
        gwut = slices['wut']
        gwur = slices['wur']
        dx2  = slices['dx2'] 
        vr   = (gwur * disk * dx2).sum(-1) / (gwut * disk * dx2).sum(-1)
    tinf = (Rg / c) * r1D / (-vr)
    return tinf

def calc_tcool(rate, slices, extra, disk, **kwargs):
    """Thermal (cooling) timescale (based on cooling rates)
    Cooling rates measured as q u_t
    Minus sign accounts for coordinate frame cooling rate
    """
    # Whether to perform calculation in cylindrical vs. spherical coordinates
    cyl     = kwargs.get('cyl', False)
    heating = kwargs.get('heating', False)

    if rate == 'Coulomb':
        # Coulomb collisional energy exchange rate
        if cyl:
            q_ie   = -slices['qie'] / slices['Vol']
            q_cool = q_ie
            E_cool = ((slices['Eion'] / slices['Vol']) - 
                      (slices['Eelc'] / slices['Vol']))
        else:
            gq_ie   = -slices['qie']
            gq_cool = gq_ie
            gE_cool = slices['Eion'] - slices['Eelc']

    elif rate == 'Compton':
        # Compton cooling rate
        if cyl:
            q_IC   = -slices['qIC'] / slices['Vol']
            q_cool = q_IC
            E_cool = slices['Eelc'] / slices['Vol']
            if heating:
                q_cool = slices['qrad'] / slices['Vol']
        else:
            gq_IC   = -slices['qIC']
            gq_cool = gq_IC
            gE_cool = slices['Eelc']
            if heating:
                gq_cool = slices['qrad']

    elif rate == 'Total':
        # Emission + Compton - Absorption
        if cyl:
            q_emi  = -slices['qemi'] / slices['Vol']
            q_abs  = -slices['qabs'] / slices['Vol']
            q_IC   = -slices['qIC']  / slices['Vol']
            q_cool = q_emi + q_IC - q_abs
            if heating:
                q_cool = slices['qrad'] / slices['Vol']
            E_cool = -slices['R_tt']  / slices['Vol']
        else:
            gq_emi  = -slices['qemi']
            gq_abs  = -slices['qabs']
            gq_IC   = -slices['qIC']
            gq_cool = gq_emi + gq_IC - gq_abs
            if heating:
                gq_cool = slices['qrad']
            gE_cool = -slices['R_tt']

    if cyl:
        R2D, Z2D, qcool_cyl = interpolate_sph_to_cyl(q_cool, slices, extra)
        qcool_cyl[qcool_cyl < 0.] = 0.
        # Energy density
        _, _, Ecool_cyl = interpolate_sph_to_cyl(E_cool, slices, extra) 
        dz = Z2D[1:, 0] - Z2D[:-1, 0]
        dz = np.insert(dz, -1, dz[-1])
        # Final cooling time
        tcool = np.sum(Ecool_cyl[:, :, 0] * disk * dz[:, None], axis=0) / (
                np.sum(qcool_cyl[:, :, 0] * disk * dz[:, None], axis=0)
        )
    else:
        gq_cool[gq_cool < 0.] = 0.
        dx2   = slices['dx2']
        # Final cooling time
        tcool = (gE_cool * disk * dx2).sum(-1) / (gq_cool * disk * dx2).sum(-1)
    return tcool

def calc_tthm(slices, extra, Flux_dS, disk, **kwargs):
    """Thermal (radiative) timescale (based on -R^i_t)"""
    
    # Whether to perform calculation in cylindrical vs. spherical coordinates
    cyl = kwargs.get('cyl', False)
    Rg  = kwargs.get('Rg', 1.)
    c   = kwargs.get('c',  1.)
    Rg2 = Rg * Rg
    Rg3 = Rg * Rg * Rg

    if cyl:
        R_tt = slices['R_tt']  / slices['Vol']
        R2D, Z2D, R_tt_cyl = interpolate_sph_to_cyl(R_tt, slices, extra) 
        dz = Z2D[1:, 0] - Z2D[:-1, 0]
        dz = np.insert(dz, -1, dz[-1])
    else:
        gR_tt = slices['R_tt']
        
    Flux_total = Rg * Rg * (Flux_dS['top'] + Flux_dS['bot'])

    if cyl:
        dR   = Flux_dS['dR']
        r1D  = slices['r']
        tthm = (2. * np.pi * Rg3 * r1D * dR * (
            np.sum(-R_tt_cyl[:, :, 0] * disk * dz[:, None], axis=0))/Flux_total
        )  
    else:
        gR_tt = slices['R_tt']
        dx1   = slices['dx1']
        dx2   = slices['dx2']
        dR    = Flux_dS['dR']
        tthm  = Rg3 * (-gR_tt * disk * dx1 * dx2).sum(-1) / Flux_total
    
    return tthm

def calc_radiative_flux(flux_R, flux_Z, disk, slices, extra, **kwargs):
    """Radiative flux through the disk surface."""
    # Radiative flux calculation must be done in cylindrical coordinates
    cyl       = kwargs.get('cyl',       False)
    use_photo = kwargs.get('use_photo', False)

    if use_photo:
        tau = kwargs.get('tau')
        Be  = kwargs.get('Be')
        disk = calc_disk(tau, Be, slices, extra, cyl=cyl, use_photo=use_photo)

    if not cyl:
        # Interpolate disk mask to cylindrical coordinates and
        # fix any errors from interpolation
        disk[disk != 1] = 0
        R2D, Z2D, disk_cyl = interpolate_sph_to_cyl(disk,   slices, extra)
        _, _, flux_R_cyl   = interpolate_sph_to_cyl(flux_R, slices, extra)
        _, _, flux_Z_cyl   = interpolate_sph_to_cyl(flux_Z, slices, extra)

        disk_cyl[disk_cyl != 0] = 1
        disk = disk_cyl[:, :, 0]
        # Do not trust data beyond 200 Rg or near jet cone
        disk[np.sqrt(R2D * R2D + Z2D * Z2D) > 200.] = 0
        th2D = np.acos(Z2D / np.sqrt(R2D * R2D + Z2D * Z2D))
        disk[th2D < np.pi/60.] = 0
        disk[th2D > np.pi - np.pi/60.] = 0

        flux_R = flux_R_cyl
        flux_Z = flux_Z_cyl

    if cyl:
        if 'R_cyl' not in kwargs:
            raise ValueError("Missing argument R_cyl for radiative flux")
        if 'Z_cyl' not in kwargs:
            raise ValueError("Missing argument Z_cyl for radiative flux")
        R2D = kwargs.get('R_cyl')
        Z2D = kwargs.get('Z_cyl')

    r1D     = slices['r']
    flux_dS = dict.fromkeys(['top', 'bot'])
    for side in flux_dS.keys():
        flux_dS[side] = np.zeros(len(r1D))
    
    disk_surface = calc_disk_surface(R2D, Z2D, disk)
    dR = R2D[0, :] * slices['dx1']
    dz = dict.fromkeys(['top', 'bot'])
    for side in ['top', 'bot']:
        dz[side] = dR * ddR(disk_surface[side], R2D[0, :], slices['dx1'])

    # Rad_Flux is in cylindrical coordinates 
    Rad_Flux = [flux_R[:, :, 0], flux_Z[:, :, 0]]

    # Get indices in z-direction
    row_indices = np.indices(disk.shape)[0]
    # Set all disk=False indices to NaN
    masked_indices = np.where(disk, row_indices, np.nan)
    idx_top  = np.nanmax(masked_indices, axis=0)
    idx_bot  = np.nanmin(masked_indices, axis=0)
    # Check for fully empty columns
    is_empty = np.isnan(idx_top)
    idx_top_int = np.nan_to_num(idx_top).astype(int)
    idx_bot_int = np.nan_to_num(idx_bot).astype(int)

    # If empty, set to midplane
    nz = disk.shape[0]
    idx_top_int[is_empty] = int(nz/2)
    idx_bot_int[is_empty] = int(nz/2) - 1

    ridx  = np.arange(disk.shape[1]) 

    RFlux_top = Rad_Flux[0][idx_top_int, ridx]
    ZFlux_top = Rad_Flux[1][idx_top_int, ridx]

    RFlux_bot = Rad_Flux[0][idx_bot_int, ridx]
    ZFlux_bot = Rad_Flux[1][idx_bot_int, ridx]

    Rnhat_dS_top  = -dz['top']
    Znhat_dS_top  = dR
        
    Rnhat_dS_bot  = dz['bot'] 
    Znhat_dS_bot  = -dR
    
    flux_dS['top'] = ((RFlux_top * Rnhat_dS_top) + (ZFlux_top * Znhat_dS_top))
    flux_dS['bot'] = ((RFlux_bot * Rnhat_dS_bot) + (ZFlux_bot * Znhat_dS_bot))

    # Correct by Jacobian factor
    flux_dS['top'] *= 2. * np.pi * R2D[0, :] 
    flux_dS['bot'] *= 2. * np.pi * R2D[0, :] 
    
    flux_dS['surface'] = disk_surface
    flux_dS['dR']      = dR
    flux_dS['dz']      = dz

    return flux_dS

def calc_disk_surface(R, z, disk):
    z_top = np.zeros_like(disk[0, :])
    z_bot = np.zeros_like(disk[0, :])

    # Get indices in z-direction
    row_indices = np.indices(disk.shape)[0]
    # Set all disk=False indices to NaN
    masked_indices = np.where(disk, row_indices, np.nan)
    idx_top  = np.nanmax(masked_indices, axis=0)
    idx_bot  = np.nanmin(masked_indices, axis=0)
    # Check for fully empty columns
    is_empty = np.isnan(idx_top)
    idx_top_int = np.nan_to_num(idx_top).astype(int)
    idx_bot_int = np.nan_to_num(idx_bot).astype(int)
    
    ridx  = np.arange(disk.shape[1]) 
    z_top = z[idx_top_int, ridx]
    z_bot = z[idx_bot_int, ridx]

    # If no disk, assume midplane for surface
    z_top[is_empty] = 0.0
    z_bot[is_empty] = 0.0
    # 4 for the low-resolution runs
    z_top_smth = ana.Tools.smooth(z_top, 4, len(R[0, :]))
    z_bot_smth = ana.Tools.smooth(z_bot, 4, len(R[0, :]))
    surf = dict.fromkeys(['top', 'bot'])
    surf['top'] = z_top_smth
    surf['bot'] = z_bot_smth
    
    return surf

def ddR(q, R, dx1):
    dq_dR = (q[2:] - q[:-2]) / (2. * R[1:-1] * dx1)
    dq_dR = np.concatenate([[dq_dR[0]], dq_dR, [dq_dR[-1]]])
    return dq_dR

def get_last_file_num(data_dir):
    files = glob.glob('Slices*', root_dir=data_dir)
    file_list = [int(f.strip('/Slices_').strip('.npz')) for f in files]
    return max(file_list)

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Calculate 1D timescale profiles.""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('--use_file', type=str,
                        help="""name of file containing slice averages 
                        located in Averages/../use_file/use_file.npz.""")
    parser.add_argument('--file_num', type=int, required=False, 
                        default=-1, help = """dump number""")
    parser.add_argument('--start_file', type=int, required=False,
                        default=-1, help = """dump number
                        of end file in timescales extraction.""")
    parser.add_argument('--end_file', type=int, required=False,
                        default=-1, help = """dump number
                        of end file in stresses extraction.""")
    parser.add_argument('--fstart', type=int, default=-1,
                        required=False,
                        help = """starting dump number for
                        computing mean stresses""")
    parser.add_argument('--fend', type=int, default=-1,
                        required=False,
                        help = """ending dump number for
                        computing mean stresses""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if timescale profiles
                        should be calculated for reduced data.""")
    parser.add_argument('--stride', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in dump number
                        only relevant if --end_file is supplied.""")
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
    parser.add_argument('--heating', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""boolean to determine if radiation source term
                        from heating analysis should be used to calculate
                        cooling time.""")   
    parser.add_argument('--use_cyl', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""boolean to determine if 
                        cylindrical coordinates should be used to integrate
                        photosphere, calculate zero-Bernoulli surface, 
                        calculate thermal/diffusion time etc.""")   
    parser.add_argument('--use_photo', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""ignore zero-Bernoulli surface and 
                        force the use of the photosphere""")   
    parser.add_argument('--use_bound', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""ignore photosphere and 
                        force the use of the zero-Bernoulli surface""")   
    parser.add_argument('--use_HoR', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""ignore photosphere and the zero-Bernoulli 
                        surface. Just use the disk thickness HoR < 1.""")   
    parser.add_argument('--optically_thin', 
                        action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""if flow is everywhere optically thin, 
                        do not calculate thermal time from radiation field
                        and just use t_cool (efficient).""")   
    parser.add_argument('--time_average', type=int, default=1,
                        required=False,
                        help = """number of slices (enumerated by dump number)
                        to average in time when computing mean and 
                        turbulent/fluctuating stresses""")
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="""boolean to determine if existing profile file
                        should be overwritten.""")
    parser.add_argument('--plot_data', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""make PNG figure""")   
    parser.add_argument('--hi_res', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""make high-resolution figure (zoomed in)""")   
    args = parser.parse_args()
    main(**vars(args))