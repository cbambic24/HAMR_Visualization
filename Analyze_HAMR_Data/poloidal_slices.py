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
import coordinates as coords
import radiation as rad
from mpi4py import MPI
from mpi4py.util import pkl5
import gc
import time as timeit

# Numerical constants for performance
ONE         = np.float32(1.)
HALF        = np.float32(0.5)
ONE_THIRD   = np.float32(1. / 3.)
FOUR_THIRDS = np.float32(4. / 3.)
FOURPI      = np.float32(4. * np.pi)
EIGHTPI     = np.float32(8. * np.pi)

def main(**kwargs):
    sim       = kwargs.get('sim')
    file_num  = kwargs.get('file_num')
    nslices   = kwargs.get('nslices')
    lowres1   = kwargs.get('lowres1')
    lowres2   = kwargs.get('lowres2')
    lowres3   = kwargs.get('lowres3')
    r_max     = kwargs.get('r_max')
    reduced   = kwargs.get('reduced')
    restart   = kwargs.get('restart')
    shared    = kwargs.get('shared')
    title     = kwargs.get('title')
    radiation = kwargs.get('radiation')
    two_temp  = kwargs.get('two_temp')
    cooling   = kwargs.get('cooling')
    vargamma  = kwargs.get('vargamma')

    # Define constants
    MBH = np.float32(10.0)
    cgs = True if (two_temp or radiation) else False
    if (two_temp or radiation):
        Msun = np.float32(hr.CONSTANTS['Msun'])
        c    = np.float32(hr.CONSTANTS['c'])
        G    = np.float32(hr.CONSTANTS['G'])
        Rg   = G * MBH * Msun / (c * c)
    else:
        c, G, Rg = ONE, ONE, ONE

    if (two_temp or radiation):
        kB   = np.float32(hr.CONSTANTS['kB'])
        mp   = np.float32(hr.CONSTANTS['mp'])

        x_AB = np.float32(hr.CONSTANTS['X_AB'])
        y_AB = np.float32(hr.CONSTANTS['Y_AB'])

        # Equation 11 in Sadowski+ (2017)
        mu_i = np.float32(4. / (4. * x_AB + y_AB))
        # Equation 10 in Sadowski+ (2017)
        mu_e = np.float32(2. / (1. + x_AB))
        mi   = (x_AB + y_AB) * mp  
        me   = np.float32(hr.CONSTANTS['me'])

        # Given in Kaaz et al. (2024)
        mu_g = np.float32(4. / (6. * x_AB + y_AB + 2.))

    # Ensure CONSTANTS are 32-bit floats
    CONSTANTS = {key : np.float32(value) 
                 for (key, value) in hr.CONSTANTS.items()}
    
    # Determine data extraction and save directories
    data_dir = cv.get_data_dir(sim, reduced, shared)
    save_dir = cv.get_save_dir(sim if title == None else title, 
                               'Averages', 'Slices', reduced=reduced)

    nzeros = len(str(len(glob.glob(os.path.join(data_dir, 'dumps*')))))
 
    # Quantities (quants) for slice averaging
    keys = [
        'rho', 'Ti', 'Pmag', 'Pthm', 'Vol', 'w'
    ]
    if (two_temp and radiation):
        keys += ['Te', 'Prad', 'Eion', 'Eelc', 'rhokappa']
    elif (radiation and not two_temp):
        keys += ['Prad', 'rhokappa']
    elif (two_temp and not radiation):
        keys += ['Te', 'Eion', 'Eelc']

    # Energy fluxes (vectors)
    flux_keys = [
        'THD_tR', 'THD_tz', 'THD_t1', 
        'TEM_tR', 'TEM_tz', 'TEM_t1', 'Be'
    ]

    # Vectors (4-velocity, 3-velocity, 4-magnetic field, and 
    # radiation 4-momentum) all necessary for stress profiles
    vec_keys = [
        'uR',   'uphi',  'uz',
        'wuR',  'wuphi', 'wuz', 'wu_phi', 'wur', 'wuth',
        'vR',   'vphi',  'vz', 
        'bR',   'bphi',  'bz',  'b_phi',  'br',  'bth'
    ]

    if (radiation):
        flux_keys += ['R_tR', 'R_tz', 'R_tt', 'R_t1', 
                      'Be_Rad', 'R_tR_Fluid', 'R_tz_Fluid']
    
    rhovec_keys = ['rhouR', 'rhouphi', 'rhouz', 'rhou1']

    # Calculate phi integral over gdet including sigma cut
    keys += ['Vol_sigma_cut']

    # Cooling rates (emission, absorption, scattering source terms)
    # Expensive and can be disabled with --no-cooling
    if (cooling and two_temp and radiation):
        keys += ['qie', 'qIC', 'qemi', 'qabs']
    elif (cooling and two_temp and not radiation):
        keys += ['qie']
    elif (cooling and not two_temp and radiation):
        keys += ['qIC', 'qemi', 'qabs']
    
    # Ensure that nodes don't all attempt to make directory at same time
    timeit.sleep(np.random.rand())
    os.makedirs(save_dir, exist_ok=True)

    read_args = {
        'cgs_grid' : cgs,
        'cgs_data' : cgs,
        'coords1D' : True,
        'MBH'      : MBH,
        'lowres1'  : lowres1,
        'lowres2'  : lowres2,
        'lowres3'  : lowres3
    }

    if (r_max != 0.):
        read_args['r_max'] = r_max

    slice_keys = keys + flux_keys + vec_keys + rhovec_keys + ['B1']
    
    slices, nphi, iphi_stp = init_slices(data_dir, file_num, slice_keys,
        save_dir, nslices, restart, nzeros, **read_args)
    slices['done'] = False

    # Loop over whole data slice that fits on single node.
    # Process slivers of data slice assigned to each core in parallel.
    for slice_num, iphi in enumerate(range(0, nphi, iphi_stp)):
        if (restart):
            if slice_num < slices['slice_num_restart']:
                continue
     
        slices['saved'] = False
        k_min = int(iphi)
        k_max = int(iphi + iphi_stp)

        comm = pkl5.Intracomm(MPI.COMM_WORLD)	
        size = comm.Get_size()
        rank = comm.Get_rank()

        slice_loc = {}

        delta_k   = (k_max - k_min) / size
        k_min_loc = int(k_min + delta_k * rank)
        k_max_loc = int(k_min + delta_k * (rank + 1))

        read_args['k_min'] = k_min_loc
        read_args['k_max'] = k_max_loc

        H = hr.Read(data_dir, **read_args)

        # Staggered start to take pressure off of filesystem
        timeit.sleep(0.05 * rank + np.random.rand())
    
        # Read grid and data
        H.read_grid(file_num, direct=True)
        H.read_prims(file_num, skip_read_params=True)

        if 'tr1' in H.data.keys(): H.data['tr1'] = None
        if 'tr2' in H.data.keys(): H.data['tr2'] = None
        if 'dissipation' in H.data.keys(): H.data['dissipation'] = None
        if 'rad_src'     in H.data.keys(): H.data['rad_src']     = None

        acc_class_args = {
            'grid'      : H.grid,
            'data'      : H.data,
            'pars'      : H.params,
            'CONSTANTS' : CONSTANTS, # float32 constants
            'cgs'       : cgs,
            'MBH'       : MBH,
            'radiation' : radiation
        }

        class_args = {
            'pars'      : H.params,
            'CONSTANTS' : CONSTANTS, # float32 constants
            'cgs'       : cgs
        }
        
        coords_args = class_args.copy()
        coords_args['MBH'] = MBH

        # Initialize Accretion class
        Acc = ana.Accretion(**acc_class_args)
        # Initialize Coordinates class
        Coords = coords.Coordinates(**H.grid, **coords_args)
        # Initialize Fluid class
        Fluid = ana.Fluid(**class_args)
        # Initialize Field class
        Field = ana.Field(**class_args)
        # Initialize Radiation class
        Radi  = rad.Radiation(**class_args)

        # Calculate sigma (3D)
        b  = Field.calc_b(**H.data, **H.grid, cpp=True)
        b2 = Field.calc_b2(b=b, **H.data, **H.grid, cpp=True)
        if (cgs):
            sigma  = (b2 / FOURPI) 
            sigma /= (H.data['rho'] * c * c)
        else:
            sigma = b2 / H.data['rho']

        # Calculate gas pressure
        if (two_temp and vargamma):
            pgas= Fluid.calc_pgas(**H.data)
        else:
            pgas      = hr.Tools.zeroNd(H.data['u_gas'].shape, 1)
            gamma_adi = np.float32(5. / 3.)
            np.multiply(H.data['u_gas'], (gamma_adi - 1.), out=pgas)

        # Cut on sigma
        sigma_cut_args = {
            'sigma_cut' : np.float32(3.),
            'sigma'     : sigma
        }

        # Energy fluxes
        # Calculate T^i_t streamlines/fluxes
        stress_energy_HD_10 = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas,
            upper_lower=True, index=[1, 0], cpp=True)
        stress_energy_HD_20 = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas,
            upper_lower=True, index=[2, 0], cpp=True)
        THD_vec    = np.zeros_like(H.data['uu'])
        THD_vec[1] = stress_energy_HD_10
        THD_vec[2] = stress_energy_HD_20
        THD_cyl    = Coords.int_to_cyl_vec(THD_vec, cpp=True)
        slice_loc['THD_tR'] = Acc.ph_sum(THD_cyl[1])[0]
        slice_loc['THD_tz'] = Acc.ph_sum(THD_cyl[3])[0]
        slice_loc['THD_t1'] = Acc.ph_sum(THD_vec[1])[0]
        
        del THD_cyl, THD_vec, stress_energy_HD_10, stress_energy_HD_20

        stress_energy_EM_10 = Field.calc_TEM(**H.data, **H.grid, b2=b2, b=b, 
            upper_lower=True, index=[1, 0], cpp=True)
        stress_energy_EM_20 = Field.calc_TEM(**H.data, **H.grid, b2=b2, b=b, 
            upper_lower=True, index=[2, 0], cpp=True)
        TEM_vec    = np.zeros_like(H.data['uu'])
        TEM_vec[1] = stress_energy_EM_10
        TEM_vec[2] = stress_energy_EM_20
        TEM_cyl    = Coords.int_to_cyl_vec(TEM_vec, cpp=True)
        slice_loc['TEM_tR'] = Acc.ph_sum(TEM_cyl[1])[0]
        slice_loc['TEM_tz'] = Acc.ph_sum(TEM_cyl[3])[0]
        slice_loc['TEM_t1'] = Acc.ph_sum(TEM_vec[1])[0]
        
        del TEM_cyl, TEM_vec, stress_energy_EM_10, stress_energy_EM_20

        # Calculate R^i_t streamlines/fluxes
        if (radiation):
            stress_energy_RAD_00 = Radi.calc_TRAD(**H.data, **H.grid, 
                upper_lower=True, index=[0, 0], cpp=True)
            stress_energy_RAD_10 = Radi.calc_TRAD(**H.data, **H.grid, 
                upper_lower=True, index=[1, 0], cpp=True)
            stress_energy_RAD_20 = Radi.calc_TRAD(**H.data, **H.grid, 
                upper_lower=True, index=[2, 0], cpp=True)
            R_vec    = np.zeros_like(H.data['uu'])
            R_vec[0] = stress_energy_RAD_00
            R_vec[1] = stress_energy_RAD_10
            R_vec[2] = stress_energy_RAD_20
            R_cyl    = Coords.int_to_cyl_vec(R_vec, cpp=True)
            # R^t_t necessary for thermal and radative diffusion timescales
            slice_loc['R_tt'] = Acc.ph_sum(R_cyl[0])[0]
            slice_loc['R_tR'] = Acc.ph_sum(R_cyl[1])[0]
            slice_loc['R_tz'] = Acc.ph_sum(R_cyl[3])[0]
            slice_loc['R_t1'] = Acc.ph_sum(R_vec[1])[0]
            
            del R_cyl, R_vec, stress_energy_RAD_00 
            del stress_energy_RAD_10, stress_energy_RAD_20

        if (radiation):
            # Radiation stress-tensor in the fluid frame necessary
            # for the calculation of the radiative diffusion timescale
            uu_rad     = ana.Tools.contract_vector(
                H.data['uu_rad'], H.grid['gcov'], cpp=True)
            uu_rad_fld = Coords.boost_vector(uu_rad, H.data['uu'] / c, 
                                             inverse=True, cpp=True)
            uu_t_rad_fld = uu_rad_fld[0]
            del uu_rad, uu_rad_fld

            uurad_fld = Coords.boost_vector(H.data['uu_rad'], 
                                            H.data['uu'] / c, cpp=True)
            u_rad    = H.data['u_rad']
            R_vec    = np.zeros_like(H.data['uu'])
            R_vec[0] = (
                FOUR_THIRDS * u_rad * (uurad_fld[0]/c) * (uu_t_rad_fld/c) + 
                ONE_THIRD * u_rad
            )
            R_vec[1] = FOUR_THIRDS * u_rad * (uurad_fld[1]/c)*(uu_t_rad_fld/c)
            R_vec[2] = FOUR_THIRDS * u_rad * (uurad_fld[2]/c)*(uu_t_rad_fld/c)
            R_cyl    = Coords.int_to_cyl_vec(R_vec)
            slice_loc['R_tR_Fluid'] = Acc.ph_sum(R_cyl[1])[0]
            slice_loc['R_tz_Fluid'] = Acc.ph_sum(R_cyl[3])[0]
            del R_cyl, R_vec, u_rad, uurad_fld, uu_t_rad_fld
            H.data['uu_rad'] = None

        # Opacities
        if (radiation):
            # Calculate opacities including all processes
            # Consider testing the C++ acceleration of this
            opacities = dict.fromkeys(['electron_scattering', 'bound_free', 
                'CHIANTI', 'free_free', 'hydrogen', 'molecular', 'synchrotron'])
            if (two_temp):
                Opac = rad.Opacity(opacities, **H.data, pars=H.params, 
                    CONSTANTS=hr.CONSTANTS, b2=b2, XRB=True, cpp=True)
            else:
                Opac = rad.Opacity(opacities, **H.data, 
                    Te=(mu_g * mp / kB) * (pgas / H.data['rho']), 
                    pars=H.params, CONSTANTS=hr.CONSTANTS, 
                    b2=b2, XRB=True, cpp=True)
            kappa_abs = Opac.opacities['absorption_total']
            kappa_es  = Opac.opacities['electron_scattering']
            kappa_eff = np.sqrt(kappa_abs * (kappa_abs + kappa_es))
            del Opac.opacities
        
        uu_cov = ana.Tools.contract_vector_cpp(H.data['uu'], H.grid['gcov'])
        b_cov  = ana.Tools.contract_vector_cpp(b,            H.grid['gcov'])

        # Bernoulli parameter for outflows
        # Numerator of Bernoulli parameter
        if (cgs):
            w = (H.data['rho']*c*c + pgas + H.data['u_gas'] + b2/FOURPI)
        else:
            w = (H.data['rho'] + pgas + H.data['u_gas'] + b2) 
        slice_loc['Be'] = Acc.ph_sum(w * uu_cov[0])[0] 
        if (radiation):
            w = (H.data['rho'] * c * c + pgas + H.data['u_gas'] 
                    + b2/FOURPI + FOUR_THIRDS * H.data['u_rad']) 
            slice_loc['Be_Rad'] = Acc.ph_sum(w * uu_cov[0])[0]

        # B^1 (useful for calculating vector potential)
        slice_loc['B1'] = Acc.ph_sum(H.data['B'][1])[0]

        # Deallocate magnetic field in the lab frame and radiation 4-momentum
        # as well as metric tensor and inverse
        H.data['B']    = None
        H.grid['gcov'] = None
        H.grid['gcon'] = None
        
        # Magnetic and Thermodynamic Structure
        if (two_temp):
            if (vargamma):
                # Calculate gamma (ratio of specific heats)
                # Equation 28 in Sadowski+ (2017)
                Theta_e = (kB * H.data['Te']) / (me * c * c)
                Theta_i = (kB * H.data['Ti']) / (mi * c * c)

                gam_e = np.float32((10. + 20.*Theta_e) / (6. + 15.*Theta_e))
                gam_i = np.float32((10. + 20.*Theta_i) / (6. + 15.*Theta_i))
            else:
                gam_e = np.float32(5./3.)
                gam_i = np.float32(5./3.)
        
        ones = np.ones_like(H.data['rho'])

        # Calculate magnetic pressure
        pmag = b2 / EIGHTPI if cgs else HALF * b2

        if (two_temp and radiation):
            Eion, Eelc = hr.Tools.zeroNd(H.data['rho'].shape, 2)
            np.multiply(H.data['rho'], H.data['Ti'], out=Eion)
            np.multiply(H.data['rho'], H.data['Te'], out=Eelc)
            Eion *= np.float32(1./(gam_i-1.) * (kB / (mu_i * mp)))
            Eelc *= np.float32(1./(gam_e-1.) * (kB / (mu_e * mp)))
            prad  = ONE_THIRD * H.data['u_rad']
            quants = [
                H.data['rho'], # rho
                H.data['Ti'],  # Ti
                pmag,          # Pmag = b2/8pi
                pgas,          # Pthm
                ones,          # Vol
                w,             # w (enthalpy)
                H.data['Te'],  # Te 
                prad,          # Prad = 1/3 u_rad
                Eion,          # Eion (ion energy density)
                Eelc           # Eelc (electron energy density)
            ]
        elif (radiation and not two_temp):
            T_gas = pgas / H.data['rho']
            T_gas *= mu_g * mp / kB
            prad  = ONE_THIRD * H.data['u_rad']
            quants = [
                H.data['rho'], # rho
                T_gas,         # Ti (gas temperature when no 2T)
                pmag,          # Pmag = b2/8pi
                pgas,          # Pthm
                ones,          # Vol
                w,             # w (enthalpy)
                H.data['Te'],  # Te 
                prad           # Prad = 1/3 u_rad
            ]
        elif (two_temp and not radiation):
            Eion, Eelc = hr.Tools.zeroNd(H.data['rho'].shape, 2)
            np.multiply(H.data['rho'], H.data['Ti'], out=Eion)
            np.multiply(H.data['rho'], H.data['Te'], out=Eelc)
            Eion *= np.float32(1./(gam_i-1.) * (kB / (mu_i * mp)))
            Eelc *= np.float32(1./(gam_e-1.) * (kB / (mu_e * mp)))
            quants = [
                H.data['rho'], # rho
                H.data['Ti'],  # Ti
                pmag,          # Pmag = b2/8pi
                pgas,          # Pthm
                ones,          # Vol
                w,             # w (enthalpy)
                H.data['Te'],  # Te 
                Eion,          # Eion (ion energy density)
                Eelc           # Eelc (electron energy density)
            ]
        else:
            T_gas = pgas / H.data['rho']
            quants = [
                H.data['rho'], # rho
                T_gas,         # Ti (gas temperature when no 2T)
                pmag,          # Pmag = b2/8pi
                pgas,          # Pthm
                ones,          # Vol
                w              # w (enthalpy)
            ]

        if (radiation and two_temp):
            for (key, quant) in zip(keys[:-6], quants):
                slice_loc[f'{key}'] = Acc.ph_sum(quant)[0]
        elif (radiation and not two_temp):
            for (key, quant) in zip(keys[:-5], quants):
                slice_loc[f'{key}'] = Acc.ph_sum(quant)[0]
        elif (two_temp and not radiation):
            for (key, quant) in zip(keys[:-2], quants):
                slice_loc[f'{key}'] = Acc.ph_sum(quant)[0]
        else:
            for (key, quant) in zip(keys[:-1], quants):
                slice_loc[f'{key}'] = Acc.ph_sum(quant)[0]

        # Opacities
        if (radiation):
            slice_loc['rhokappa'] = Acc.ph_sum(kappa_eff * H.data['rho'], 
                                               **sigma_cut_args)[0]

        # Coulomb collisional and radiative cooling
        if (cooling and two_temp):
            # Exclude relativistic temperatures in qie
            non_rel = (H.data['Ti'] < 1.e12) * (H.data['Te'] < 1.e10) * (
                      (H.data['Te'] > 1.e4))
            qie  = Fluid.calc_qie(**H.data, cpp=True, cooling=True) * non_rel
            slice_loc['qie']  = Acc.ph_sum(qie, weight=uu_cov[0]/c)[0]
            del qie, non_rel
        if (cooling and radiation):
            qIC  = Radi.calc_qIC(  **H.data)
            qemi = Radi.calc_q_emi(**H.data, b2=b2, XRB=True, cpp=True) 
            qabs = Radi.calc_q_abs(**H.data, b2=b2, XRB=True, cpp=True)
            slice_loc['qIC']  = Acc.ph_sum(qIC,  weight=uu_cov[0]/c)[0]
            slice_loc['qemi'] = Acc.ph_sum(qemi, weight=uu_cov[0]/c)[0]
            slice_loc['qabs'] = Acc.ph_sum(qabs, weight=uu_cov[0]/c)[0]
            del qIC, qemi, qabs

        # phi integral over gdet including sigma cut
        slice_loc['Vol_sigma_cut'] = Acc.ph_sum(ones, **sigma_cut_args)[0]            

        H.data['u_gas'] = None
        if (two_temp and vargamma): 
            del Theta_e, Theta_i, gam_e, gam_i, pgas
        elif (two_temp and not vargamma): 
            del gam_e, gam_i, pgas
        else: 
            del pgas, T_gas

        if (radiation): 
            del kappa_eff, kappa_abs, b2, pmag
            H.data['u_rad'] = None
        else:           
            del b2, pmag
        del ones

        # Deallocate temperatures and energy densities 
        # in preparation for large transformation matrices
        if (two_temp):
            H.data['Ti'] = None
            H.data['Te'] = None
        
        # Compute theta and phi averages of vector quantities
        uu_cyl     = Coords.int_to_cyl_vec(H.data['uu'], cpp=True)
        uu_sph     = Coords.int_to_ortho_sph_vec(H.data['uu'], cpp=True)
        uu_cov_cyl = Coords.int_to_cyl_one_form(uu_cov, cpp=True)
        b_cyl      = Coords.int_to_cyl_vec(b, cpp=True)
        b_sph      = Coords.int_to_ortho_sph_vec(b, cpp=True)
        b_cov_cyl  = Coords.int_to_cyl_one_form(b_cov, cpp=True)
        del b, b_cov
        np32 = np.float32
        quants = [
            uu_cyl[1],     uu_cyl[2],     uu_cyl[3],
            w * uu_cyl[1], w * uu_cyl[2], w * uu_cyl[3], 
            w * uu_cov_cyl[2], w * uu_sph[1], w * uu_sph[2],
            np.nan_to_num(uu_cyl[1] / uu_cyl[0], copy=False).astype(np32),
            np.nan_to_num(uu_cyl[2] / uu_cyl[0], copy=False).astype(np32),
            np.nan_to_num(uu_cyl[3] / uu_cyl[0], copy=False).astype(np32),
            b_cyl[1], b_cyl[2], b_cyl[3], b_cov_cyl[2], b_sph[1], b_sph[2]
        ]

        for (key, quant) in zip(vec_keys, quants):
            slice_loc[f'{key}'] = Acc.ph_sum(quant)[0]

        # Mass fluxes
        quants = [
            H.data['rho'] * uu_cyl[1], 
            H.data['rho'] * uu_cyl[2], 
            H.data['rho'] * uu_cyl[3],
            H.data['rho'] * H.data['uu'][1]
        ] 
        for (key, quant) in zip(rhovec_keys, quants):
            slice_loc[f'{key}'] = Acc.ph_sum(quant, **sigma_cut_args)[0]

        # Deallocate memory before MPI gather
        H.data['uu']   = None
        H.grid['gdet'] = None
        H.data['rho']  = None
        del w, uu_cov, uu_cyl, b_cyl, uu_sph, b_sph
        del uu_cov_cyl, b_cov_cyl, sigma

        nr  = len(H.grid['r'][    0, :, 0, 0])
        nth = len(H.grid['theta'][0, 0, :, 0])
        
        local_buffer = np.zeros((len(slice_keys), nr, nth), dtype=np.float64)
        for i, key in enumerate(slice_keys):
            if key in slice_loc:
                local_buffer[i, :, :] = slice_loc[key]
        global_buffer = None
        for key in slice_loc:
            slice_loc[key] = None

        if (rank == 0):
            # Put spherical radius in dictionary 
            slices['r']     = H.grid['r'][0, :, 0, 0] / Rg
            slices['theta'] = H.grid['theta'][0, 0, :, 0]
            slices['phi']   = H.grid['phi'][0, 0, 0, :]
            slices['dx1']   = H.params['dx1']
            slices['dx2']   = H.params['dx2']
            slices['dx3']   = H.params['dx3']
    
            # Density scale
            slices['rho0']  = H.params['density_scale']

            # Initialize global buffer to hold MPI sum
            global_buffer = np.zeros_like(local_buffer)

        # Sum all local buffers to global buffer
        comm.Reduce(local_buffer, global_buffer, op=MPI.SUM, root=0)
        
        print(f'Rank {rank:d} done analyzing range of k: ' + 
            f'({k_min_loc:d}, {k_max_loc:d}).', flush=True)

        if (rank == 0):
            for i, key in enumerate(slice_keys):
                slices[f'{key}'] += global_buffer[i, :, :]

            slices['k_max'] = k_max
            slices['slice_num_restart'] = slice_num + 1
            slices['saved'] = True
            fnum_str = str(file_num).zfill(nzeros)
            if (slices['k_max'] == nphi): 
                slices['done'] = True
            np.savez(save_dir + f'/Slices_{fnum_str}', **slices)

            snum = int(slice_num + 1)
            print(f'Finished with slice {snum:d} of {nslices:d}', flush=True)
        comm.Barrier()
        gc.collect()

    print(f'Done with all slices for file number {file_num}.')

def init_slices(data_dir, file_num, keys, 
    save_dir, nslices, restart, nzro, **kwargs):
    slices = {}
    
    # First, determine the limits of the slices.
    H = hr.Read(data_dir, **kwargs)
    H.read_params(file_num)
    slices['t'] = H.params['t']
    slices['a'] = H.params['a']

    nr   = int(H.params['nb1'] * H.params['bs1'] * H.upscalex1 / H.lowres1)
    nth  = int(H.params['nb2'] * H.params['bs2'] * H.upscalex2 / H.lowres2)
    nphi = int(H.params['nb3'] * H.params['bs3'] * H.upscalex3 / H.lowres3)
    if (nphi % nslices != 0):
        raise ValueError(f'nslices must evenly divide nphi = {nphi}')
    iphi_stp = int(nphi / nslices)

    if 'r_max' in kwargs:
        H.gridsizex1 = int(
            H.params['nb1'] * H.params['bs1'] * H.upscalex1 / H.lowres1)
        i_min, nr = H.get_rbox_coords()

    # Logic to allow restarts if profile extraction is interrupted.
    if not restart:
        first_call = True
    else:
        fnum_str = str(file_num).zfill(nzro)
        slices = dict(np.load(f'{save_dir}/Slices_{fnum_str}.npz'))
        first_call = False

    # Initialize slices dictionary
    if first_call:
        for key in keys:
            slices[f'{key}'] = np.zeros([nr, nth])
        first_call = False
        slices['slice_num_restart'] = 0

    return slices, nphi, iphi_stp

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Calculate 2D slices for single 
                        ultra-high-resolution simulation output.""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('file_num', type=int, help = """dump number""")
    parser.add_argument('--nslices', type=int, required=False,
                        default=1,
                        help="""number of slices in phi 
                        used to load the data""")
    parser.add_argument('--lowres1', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in x1""")
    parser.add_argument('--lowres2', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in x2""")
    parser.add_argument('--lowres3', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in x3""")
    parser.add_argument('--r_max', type=float, default=0.,
                        required=False,
                        help = """radial extent in Rg (default: R_max)""")
    parser.add_argument('--radiation', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if radiation quantities
                        should be included in slice extraction.""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if two-temperature 
                        physics should be included in slice extraction.""")    
    parser.add_argument('--cooling', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if cooling rates
                        (qie, qIC, qemi) should be included.""")    
    parser.add_argument('--vargamma', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if variable gamma 
                        should be used for ion/electron energy/pressure.""")  
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if slice extraction
                        should be done on reduced data.""")
    parser.add_argument('--restart', action='store_true', required=False, 
                        help="""boolean set to determine if slice 
                        integration should continue from last slice_num""")
    parser.add_argument('--shared', action='store_true', required=False, 
                        help="""boolean set to determine if proj-shared 
                        should be used as the location of the data directory""")
    parser.add_argument('--title', type=str, required=False, default=None, 
                        help="""title of simulation used for Averages directory
                        if different from the simulation title in proj-shared""")
    args = parser.parse_args()
    main(**vars(args))