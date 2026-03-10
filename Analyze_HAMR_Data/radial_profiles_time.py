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
import gc
import time as timeit

# Numerical constants for performance
ONE       = np.float32(1.)
HALF      = np.float32(0.5)
ONE_THIRD = np.float32(1. / 3.)
FOURPI    = np.float32(4. * np.pi)
EIGHTPI   = np.float32(8. * np.pi)
PI_2      = np.float32(np.pi / 2.)

def main(**kwargs):
    sim       = kwargs.get('sim')
    start_file= int(kwargs.get('start_file'))
    end_file  = int(kwargs.get('end_file'))
    nslices   = kwargs.get('nslices')
    lowres1   = kwargs.get('lowres1')
    lowres2   = kwargs.get('lowres2')
    lowres3   = kwargs.get('lowres3')
    reduced   = kwargs.get('reduced')
    restart   = kwargs.get('restart')
    overwrite = kwargs.get('overwrite')
    shared    = kwargs.get('shared')
    title     = kwargs.get('title')
    radiation = kwargs.get('radiation')
    two_temp  = kwargs.get('two_temp')
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
                               'Averages', 'Profiles', reduced=reduced)

    nzeros = len(str(len(glob.glob(os.path.join(data_dir, 'dumps*')))))
    
    # Quantities (quants) for profile averaging
    keys = [
        'rho', 'Ti', 'Pmag', 'Pthm', 'Vol', 'Dtheta2'
    ]
    # Accretion rates (mass and energy) + Phi_BH
    acc_keys = [
        'Mdot', 'Edot', 'Mdot_sigma', 'Edot_sigma',
        'Edot_Poyt', 'Edot_Fluid', 'Edot_jet'
    ]
    phi_keys = [
        'Phi_BH',  'Phi_sph_upp', 'Phi_sph_low', 
        'Phi_mid', 'Phi_max_upp', 'Phi_max_low'
    ]
    if (two_temp and radiation):
        keys += ['Te', 'Prad', 'Eion', 'Eelc']
        acc_keys += ['Edot_Rad']
    elif (radiation and not two_temp):
        keys += ['Prad']
        acc_keys += ['Edot_Rad']

    # Vectors (4-velocity, 3-velocity, 4-magnetic field)
    vec_keys = [
        'uR',   'uphi',  'uz',
        'vR',   'vphi',  'vz', 
        'bR',   'bphi',  'bz'
    ]

    # Ensure that nodes don't all attempt to make directory at same time
    timeit.sleep(np.random.rand())
    os.makedirs(save_dir, exist_ok=True)

    file_list = list(range(start_file, end_file + 1))
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    file_list_local = cv.get_file_list_local(file_list, size, rank, data_dir)
    if not overwrite:
        sim_title = sim if title == None else title
        averages_dir = cv.user_path + f'/Averages/{sim_title}'
        # Use get_file_list_local as if on rank 0 with 1 core
        existing_profiles = cv.get_file_list_local(file_list_local, 1, 0, 
            averages_dir, file_type='Profiles', reduced=reduced)
        existing_set = set(existing_profiles)
        file_list_local = [f for f in file_list_local if f not in existing_set]

    read_args = {
        'cgs_grid' : cgs,
        'cgs_data' : cgs,
        'coords1D' : True,
        'MBH'      : MBH,
        'lowres1'  : lowres1,
        'lowres2'  : lowres2,
        'lowres3'  : lowres3
    }

    profile_keys = keys + acc_keys + phi_keys + vec_keys

    # Outermost loop over file number (each file on single core)
    for file_num in file_list_local:
        profiles, nphi, iphi_stp = init_profiles(data_dir, file_num, 
            profile_keys, save_dir, nslices, restart, nzeros, **read_args)
        profiles['done'] = False

        # Loop over whole data slice that fits on single node.
        # Process slivers of data slice assigned to each core in parallel.
        for slice_num, iphi in enumerate(range(0, nphi, iphi_stp)):
            if (restart):
                if slice_num < profiles['slice_num_restart']:
                    continue
     
            profiles['saved'] = False
            k_min = int(iphi)
            k_max = int(iphi + iphi_stp)

            read_args['k_min'] = k_min
            read_args['k_max'] = k_max

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

            # Calculate u^mu u_nu tensor
            uu_tnsr = ana.Tools.construct_dyadic(H.data['uu'] / c, 
                H.grid['gcov'], upper_lower=True, index=[1,0], cpp=True)
            
            # Calculate gas pressure
            if (two_temp and vargamma):
                pgas= Fluid.calc_pgas(**H.data)
            else:
                pgas      = hr.Tools.zeroNd(H.data['u_gas'].shape, 1)
                gamma_adi = (5. / 3.)
                np.multiply(H.data['u_gas'], (gamma_adi - 1.), out=pgas)

            # Mass and energy accretion/outflow rates
            stress_energy_HD  = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas,
                uu_tnsr=uu_tnsr, upper_lower=True, index=[1, 0], cpp=True)
            stress_energy_EM  = Field.calc_TEM(**H.data, **H.grid, b2=b2, b=b,
                uu_tnsr=uu_tnsr, upper_lower=True, index=[1, 0], cpp=True)
            if (radiation):
                stress_energy_RAD = Radi.calc_TRAD(**H.data, **H.grid, 
                    upper_lower=True, index=[1, 0], cpp=True)
            del uu_tnsr

            stress_energy_MHD  = stress_energy_HD
            stress_energy_MHD += stress_energy_EM
            if (radiation):
                stress_energy_MHD += stress_energy_RAD
            
            acc_args = {
                'sigma_cut' : np.float32(3.),
                'sigma'     : sigma
            }

            jet_args = acc_args.copy()
            # Avoid recalculating sigma mask each time
            acc_args['sigma_mask'] = Acc.sigma_cut(**acc_args)

            profiles['Mdot']      += Acc.calc_Mdot()[0]
            profiles['Edot']      += Acc.calc_Edot(TMHD=stress_energy_MHD)[0]
            profiles['Mdot_sigma']+= Acc.calc_Mdot(**acc_args)[0]
            profiles['Edot_sigma']+= Acc.calc_Edot(TMHD=stress_energy_MHD, 
                                                   **acc_args)[0]
            profiles['Edot_Poyt'] += Acc.calc_Edot(TMHD=stress_energy_EM,  
                                                   **acc_args)[0]
            profiles['Edot_jet']  += Acc.calc_Edot(TMHD=stress_energy_EM, 
                                                   jet=True, **jet_args)[0]
            profiles['Edot_Fluid']+= Acc.calc_Edot(TMHD=stress_energy_HD,  
                                                   **acc_args)[0]
            if (radiation):
                profiles['Edot_Rad'] += Acc.calc_Edot(TMHD=stress_energy_RAD, 
                                                      **acc_args)[0]
            
            del stress_energy_MHD 
            del stress_energy_HD, stress_energy_EM
            if (radiation):
                stress_energy_RAD

            # Magnetic fluxes
            profiles['Phi_BH']      += Acc.calc_PhiBH()[0]
            profiles['Phi_sph_upp'] += Acc.calc_PhiSph(hemisphere='upper')[0]
            profiles['Phi_sph_low'] += Acc.calc_PhiSph(hemisphere='lower')[0]
            profiles['Phi_mid']     += Acc.calc_PhiMid()[0]
            profiles['Phi_max_upp'] += Acc.calc_PhiMax(hemisphere='upper')[0]
            profiles['Phi_max_low'] += Acc.calc_PhiMax(hemisphere='lower')[0]       

            # Deallocate magnetic field in the lab frame and 
            # radiation 4-momentum as well as metric tensor and inverse
            H.data['B']    = None
            H.grid['gcov'] = None
            H.grid['gcon'] = None
            if (radiation): 
                H.data['uu_rad'] = None
        
            if (two_temp):
                if (vargamma):
                    # Calculate gamma (ratio of specific heats)
                    # Equation 28 in Sadowski+ (2017)
                    Theta_e = (kB * H.data['Te']) / (me * c * c)
                    Theta_i = (kB * H.data['Ti']) / (mi * c * c)
        
                    gam_e = np.float32(
                        (10. + 20. * Theta_e) / (6. + 15. * Theta_e))
                    gam_i = np.float32(
                        (10. + 20. * Theta_i) / (6. + 15. * Theta_i))
                else:
                    gam_e = np.float32(5./3.)
                    gam_i = np.float32(5./3.)

            # Calculate (theta - pi/2)^2 for density scale height
            Dtheta  = H.grid['theta'] - PI_2
            Dtheta2 = hr.Tools.zeroNd(H.data['rho'].shape, 1)
            np.square(Dtheta, out=Dtheta2)
            ones    = np.ones_like(H.data['rho'])

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
                    Dtheta2,       # Dtheta2
                    H.data['Te'],  # Te 
                    prad,          # Prad = 1/3 u_rad
                    Eion,          # Eion (ion energy density)
                    Eelc           # Eelc (electron energy density)
                ]
            elif (radiation and not two_temp):
                T_gas  = pgas / H.data['rho']
                T_gas *= mu_g * mp / kB
                prad   = ONE_THIRD * H.data['u_rad']
                quants = [
                    H.data['rho'], # rho
                    T_gas,         # Ti (gas temperature when no 2T)
                    pmag,          # Pmag = b2/8pi
                    pgas,          # Pthm
                    ones,          # Vol
                    Dtheta2,       # Dtheta2
                    prad           # Prad = 1/3 u_rad
                ]
            else:
                T_gas  = pgas / H.data['rho']
                quants = [
                    H.data['rho'], # rho
                    T_gas,         # Ti (gas temperature when no 2T)
                    pmag,          # Pmag = b2/2
                    pgas,          # Pthm
                    ones,          # Vol
                    Dtheta2        # Dtheta2
                ]
            for (key, quant) in zip(keys, quants):
                profiles[f'{key}'] += Acc.thph_sum(quant, 
                    weight=H.data['rho'], **acc_args)[0]
            
            if (two_temp and vargamma):
                del Theta_e, Theta_i, gam_e, gam_i, pgas, b2, pmag
                del Eion, Eelc
            elif (two_temp and not vargamma):
                del gam_e, gam_i, pgas, b2, pmag
                del Eion, Eelc
            else:
                del pgas, T_gas, b2, pmag
            del ones, Dtheta, Dtheta2

            if (radiation):
                del prad
            
            # Deallocate temperatures and energy densities 
            # in preparation for large transformation matrices
            if (two_temp):
                H.data['Ti'] = None
                H.data['Te'] = None

            # Compute theta and phi averages of vector quantities
            uu_cyl = Coords.int_to_cyl_vec(H.data['uu'], cpp=True)
            H.data['uu'] = None
            b_cyl  = Coords.int_to_cyl_vec(b, cpp=True)
            np32   = np.float32
            quants = [
                uu_cyl[1], uu_cyl[2], uu_cyl[3],
                np.nan_to_num(uu_cyl[1] / uu_cyl[0], copy=False).astype(np32),
                np.nan_to_num(uu_cyl[2] / uu_cyl[0], copy=False).astype(np32),
                np.nan_to_num(uu_cyl[3] / uu_cyl[0], copy=False).astype(np32),
                b_cyl[1], b_cyl[2], b_cyl[3]
            ]

            for (key, quant) in zip(vec_keys, quants):
                profiles[f'{key}']+= Acc.thph_sum(quant, weight=H.data['rho'], 
                                                  **acc_args)[0]

            del uu_cyl, b_cyl, sigma
            del Acc, Coords, Field, Fluid
            if (radiation):
                del Radi
            acc_args['sigma']      = None
            jet_args['sigma']      = None
            acc_args['sigma_mask'] = None
            
            # Density scale
            profiles['rho0'] = H.params['density_scale']
            # Put spherical radius in dictionary 
            profiles['r']   = H.grid['r'][0, :, 0, 0] / Rg
            profiles['dx1'] = H.params['dx1']
            profiles['k_max'] = k_max

            print(f'Rank {rank:d} done analyzing range of k: ' + 
                f'({k_min:d}, {k_max:d}).', flush=True)
       
            snum = int(slice_num + 1)
            print(f'Finished with slice {snum:d} of {nslices} ' +
                f'for file {file_num}', flush=True)

        profiles['slice_num_restart'] = slice_num + 1
        profiles['saved'] = True
        if (profiles['k_max'] == nphi):
            profiles['done'] = True
        fnum_str = str(file_num).zfill(nzeros)
        np.savez(save_dir + f'/Profiles_{fnum_str}', **profiles)
        gc.collect()
        print(f'Done with all profiles for file number {file_num}.',flush=True) 

def init_profiles(data_dir, file_num, keys, 
    save_dir, nslices, restart, nzro, **kwargs):
    profiles = {}
    
    # First, determine the limits of the slices.
    H = hr.Read(data_dir, **kwargs)
    H.read_params(file_num)
    profiles['t'] = H.params['t']
    profiles['a'] = H.params['a']

    nr   = int(H.params['nb1'] * H.params['bs1'] * H.upscalex1 / H.lowres1)
    nth  = int(H.params['nb2'] * H.params['bs2'] * H.upscalex2 / H.lowres2)
    nphi = int(H.params['nb3'] * H.params['bs3'] * H.upscalex3 / H.lowres3)
    if (nphi % nslices != 0):
        raise ValueError(f'nslices must evenly divide nphi = {nphi}')   
    iphi_stp = int(nphi / nslices)

    # Logic to allow restarts if profile extraction is interrupted.
    if not restart:
        first_call = True
    else:
        fnum_str = str(file_num).zfill(nzro)
        profiles = dict(np.load(f'{save_dir}/Profiles_{fnum_str}.npz'))
        first_call = False

    # Initialize profiles dictionary
    if first_call:
        for key in keys:
            profiles[f'{key}'] = np.zeros(nr)
        first_call = False
        profiles['slice_num_restart'] = 0

    return profiles, nphi, iphi_stp

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Calculate 1D profiles for all 
                        simulation outputs.""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('start_file', type=int, help = """dump number
                        of start file in profile extraction""")
    parser.add_argument('end_file', type=int, help = """dump number
                        of end file in profile extraction""")
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
    parser.add_argument('--radiation', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if radiation quantities
                        should be included in profile extraction.""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if two-temperature 
                        physics should be included in profile extraction.""")     
    parser.add_argument('--vargamma', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if variable gamma 
                        should be used for ion/electron energy/pressure.""") 
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if profile extraction
                        should be done on reduced data.""")
    parser.add_argument('--restart', action='store_true', required=False, 
                        help="""boolean set to determine if profile 
                        integration should continue from last slice_num""")
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if existing profile file 
                        should be overwritten.""")
    parser.add_argument('--shared', action='store_true', required=False, 
                        help="""boolean set to determine if proj-shared 
                        should be used as the location of the data directory""")
    parser.add_argument('--title', type=str, required=False, default=None, 
                        help="""title of simulation used for Averages directory
                        if different from the simulation title in proj-shared""")
    args = parser.parse_args()
    main(**vars(args))