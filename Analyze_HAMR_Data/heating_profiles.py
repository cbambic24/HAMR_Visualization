import sys
import os
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
import time as timeit

def main(**kwargs):
    sim       = kwargs.get('sim')
    file_num  = kwargs.get('file_num')
    nslices   = kwargs.get('nslices')
    lowres1   = kwargs.get('lowres1')
    lowres2   = kwargs.get('lowres2')
    lowres3   = kwargs.get('lowres3')
    reduced   = kwargs.get('reduced')
    restart   = kwargs.get('restart')
    shared    = kwargs.get('shared')
    title     = kwargs.get('title')
    radiation = kwargs.get('radiation')
    two_temp  = kwargs.get('two_temp')

    # Define constants
    MBH = 10.0
    cgs = True if (two_temp or radiation) else False
    if (two_temp or radiation):
        Msun = hr.CONSTANTS['Msun']
        c    = hr.CONSTANTS['c']
        G    = hr.CONSTANTS['G']
        Rg   = G * MBH * Msun / (c * c)
    else:
        c, G, Rg = 1., 1., 1.

    if not cgs:
        raise ValueError("Cannot do heating analysis without " +
            "two-temperaure and radiation effects.")

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
    
    # Determine data extraction and save directories
    data_dir = cv.get_data_dir(sim, reduced, shared)
    save_dir = cv.get_save_dir(sim if title == None else title, 
        'Averages', 'Heating_Profiles', reduced=reduced)

    nzeros = len(str(int(os.popen(
        f'ls -d {data_dir}/dumps* | wc -l').read())))
        
    # Quantities (quants) for profile integration
    keys = [
        'qdiss',         'qrad',           'qie',     
        'qdiss_ie',      'qrad_ie',        'qie_ie',
        'qdiss_thin',    'qrad_thin',      'qie_thin', 
        'qdiss_thin_ie', 'qrad_thin_ie',   'qie_thin_ie', 
        'qdiss_ion',     'qdiss_ion_thin', 'qdiss_ion_thin_ie'
    ]

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

    profile_keys = keys
    
    profiles, nphi, iphi_stp = init_profiles(data_dir, file_num, profile_keys,
        save_dir, nslices, restart, nzeros, **read_args)

    # Loop over whole data slice that fits on single node.
    # Process slivers of data slice assigned to each core in parallel.
    for slice_num, iphi in enumerate(range(0, nphi, iphi_stp)):
        if (restart):
            if slice_num < profiles['slice_num_restart']:
                continue
     
        profiles['saved'] = False
        k_min = iphi
        k_max = iphi + iphi_stp

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        prof_loc = {}

        delta_k   = (k_max - k_min) / size
        k_min_loc = int(k_min + delta_k * rank)
        k_max_loc = int(k_min + delta_k * (rank + 1))

        read_args['k_min'] = k_min_loc
        read_args['k_max'] = k_max_loc

        H = hr.Read(data_dir, **read_args)
    
        # Read grid and data
        H.read_grid(file_num, direct=True)
        H.read_prims(file_num, skip_read_params=True)

        # Initialize Accretion class
        Acc = ana.Accretion(H.grid, H.data, H.params, hr.CONSTANTS, 
            cgs=cgs, MBH=MBH, radiation=radiation)
        # Initialize Coordinates class
        Coords = coords.Coordinates(**H.grid, pars=H.params, 
                                    CONSTANTS=hr.CONSTANTS, cgs=cgs, MBH=MBH)
        # Initialize Fluid class
        Fluid = ana.Fluid(H.params, hr.CONSTANTS, cgs=cgs)
        # Initialize Field class
        Field = ana.Field(H.params, hr.CONSTANTS, cgs=cgs)
        # Initialize Radiation class
        Radi  = rad.Radiation(H.params, hr.CONSTANTS, cgs=cgs)

        # Calculate sigma (3D)
        b  = Field.calc_b(**H.data, **H.grid, cpp=True)
        b2 = Field.calc_b2(b=b, **H.data, **H.grid, cpp=True)
        if (cgs):
            sigma = (b2 / (4. * np.pi)) / (H.data['rho'] * c * c)
        else:
            sigma = b2 / H.data['rho']
        del b

        # Calculate gas pressure
        if (two_temp):
            pgas= Fluid.calc_pgas(**H.data)
        else:
            gamma_adi = (5. / 3.)
            pgas= (gamma_adi - 1.) * H.data['u_gas']

        if (radiation):
            opacities = dict.fromkeys(['electron_scattering'])
            tau_integral_args = {
                'opacities' : opacities,
                'integral'  : 'theta',
                'sigma_cut' : 3.,
                'sigma'     : sigma,
                'theta_cut' : np.pi / 6.
            }
            if (two_temp):
                tau_integral_args['Te'] = H.data['Te']
            else:
                norm = (mu_g * mp / kB)
                tau_integral_args['Te'] = norm * (pgas / H.data['rho'])
            tau_es = Acc.calc_tau(**tau_integral_args) 
        
        if (radiation or two_temp):
            # Need uu_t / c for cooling rates
            uu_t = np.einsum(
                'ij...,j...->i...', H.grid['gcov'], H.data['uu'] / c)[0]

        # Deallocate magnetic field in the lab frame and 
        # radiation 4-momentum as well as metric tensor and inverse
        H.data['B']    = H.data['rho']
        H.grid['gcov'] = H.data['rho']
        H.grid['gcon'] = H.data['rho']
        if (radiation): 
            H.data['uu_rad'] = H.data['rho']

        if ('tr1' in H.data and 'tr2' in H.data):
            tr1, tr2 = H.data['tr1'], H.data['tr2']
            ion_frac = (tr1 > tr2) * (tr1 > 0.) * (1. - (tr2 / tr1))
        else:
            ion_frac = 1.
        
        acc_args = {
            'MBH'       : MBH,
            'cgs'       : cgs,
            'radiation' : radiation,
        }
 
        if (radiation):
            tau_args = {
                'tau_cut'   : 1.,
                'tau'       : tau_es,
            }
        
        if 'dissipation' in H.data:
            qdiss = H.data['dissipation']
            prof_loc['qdiss']         = thph_sum(qdiss, 
                H.grid, H.data, H.params, hr.CONSTANTS, 
                weight=-uu_t, **acc_args)[0]
            prof_loc['qdiss_ie']      = thph_sum(qdiss * ion_frac, 
                H.grid, H.data, H.params, hr.CONSTANTS, 
                weight=-uu_t, **acc_args)[0]
            prof_loc['qdiss_thin']    = thph_sum(qdiss, 
                H.grid, H.data, H.params, hr.CONSTANTS, weight=-uu_t,
                optically_thin=True, **acc_args, **tau_args)[0]
            prof_loc['qdiss_thin_ie'] = thph_sum(qdiss * ion_frac, 
                H.grid, H.data, H.params, hr.CONSTANTS, weight=-uu_t,
                optically_thin=True, **acc_args, **tau_args)[0]

            delta_e = Fluid.calc_delta_e(**H.data, **H.grid, b2=b2) 
            prof_loc['qdiss_ion'] = thph_sum((1. - delta_e) * qdiss, 
                H.grid, H.data, H.params, hr.CONSTANTS, 
                weight=-uu_t, **acc_args)[0]
            prof_loc['qdiss_ion_thin'] = thph_sum((1. - delta_e) * qdiss, 
                H.grid, H.data, H.params, hr.CONSTANTS, weight=-uu_t, 
                optically_thin=True, **acc_args, **tau_args)[0]
            prof_loc['qdiss_ion_thin_ie'] = thph_sum(
                (1. - delta_e) * qdiss * ion_frac, 
                H.grid, H.data, H.params, hr.CONSTANTS, weight=-uu_t, 
                optically_thin=True, **acc_args, **tau_args)[0]
            del delta_e
        else:
            zero_keys = [
                'qdiss', 'qdiss_ie', 'qdiss_thin', 'qdiss_thin_ie',
                'qdiss_ion', 'qdiss_ion_thin', 'qdiss_ion_thin_ie'
            ]
            for key in zero_keys:
                prof_loc[f'{key}'] = 0.

        if 'rad_src' in H.data:
            qrad = H.data['rad_src']
            prof_loc['qrad']         = thph_sum(-qrad, 
                H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
            prof_loc['qrad_ie']      = thph_sum(-qrad * ion_frac, 
                H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
            prof_loc['qrad_thin']    = thph_sum(-qrad, 
                H.grid, H.data, H.params, hr.CONSTANTS, 
                optically_thin=True, **acc_args, **tau_args)[0]
            prof_loc['qrad_thin_ie'] = thph_sum(-qrad * ion_frac, 
                H.grid, H.data, H.params, hr.CONSTANTS,
                optically_thin=True, **acc_args, **tau_args)[0]
        else:
            zero_keys = [
                'qrad', 'qrad_ie', 'qrad_thin', 'qrad_thin_ie'
            ]
            for key in zero_keys:
                prof_loc[f'{key}'] = 0.

        if two_temp:
            qie = Fluid.calc_qie(**H.data, cpp=True)
            prof_loc['qie']         = thph_sum(qie, 
                H.grid, H.data, H.params, hr.CONSTANTS, 
                weight=-uu_t, **acc_args)[0]
            prof_loc['qie_ie']      = thph_sum(qie * ion_frac, 
                H.grid, H.data, H.params, hr.CONSTANTS, 
                weight=-uu_t, **acc_args)[0]
            prof_loc['qie_thin']    = thph_sum(qie, 
                H.grid, H.data, H.params, hr.CONSTANTS, weight=-uu_t,
                optically_thin=True, **acc_args, **tau_args)[0]
            prof_loc['qie_thin_ie'] = thph_sum(qie * ion_frac, 
                H.grid, H.data, H.params, hr.CONSTANTS, weight=-uu_t,
                optically_thin=True, **acc_args, **tau_args)[0]
            del qie
        else:
            zero_keys = [
                'qie', 'qie_ie', 'qie_thin', 'qie_thin_ie'
            ]
            for key in zero_keys:
                prof_loc[f'{key}'] = 0.

        if (rank == 0):
            # Put spherical radius in dictionary 
            profiles['r']   = H.grid['r'][0, :, 0, 0] / Rg
            profiles['dx1'] = H.params['dx1']
            # Density scale
            profiles['rho0'] = H.params['density_scale']
            profiles['dissipation'] = True if 'dissipation' in H.data else False
            profiles['rad_source']  = True if 'rad_src'     in H.data else False
            profiles['tracer1']     = True if 'tr1'         in H.data else False
            profiles['tracer2']     = True if 'tr1'         in H.data else False

        # Gather local profiles (prof_loc) from all processors and 
        # place them in main profiles dictionary
        prof_global = comm.gather(prof_loc)
        
        print(f'Rank {rank:d} done analyzing range of k: ' + 
            f'({k_min_loc:d}, {k_max_loc:d}).', flush=True)

        if (rank == 0):
            for prof in prof_global:
                for key in profile_keys:
                    profiles[f'{key}'] += prof[f'{key}']

            profiles['k_max'] = k_min + (k_max - k_min) * size 
            profiles['slice_num_restart'] = slice_num + 1
            fnum_str = str(file_num).zfill(nzeros)
            np.savez(save_dir + f'/Heating_Profiles_{fnum_str}', **profiles)

            snum = int(slice_num + 1)
            print(f'Finished with slice {snum:d} of {nslices:d}', flush=True)
            profiles['saved'] = True

    print(f'Done with all profiles for file number {file_num}.')

def thph_sum(q, grid, data, params, CONSTANTS, **kwargs):
    w    = kwargs.get('weight', 1.0)
    acc2_args = {
        'cgs'       : kwargs.get('cgs',       False),
        'radiation' : kwargs.get('radiation', False),
        'MBH'       : kwargs.get('MBH',       1.)
    }
    Acc2 = ana.Accretion(grid, data, params, CONSTANTS, **acc2_args)
    return (q * w * Acc2.sigma_cut(**kwargs) * Acc2.tau_cut(**kwargs) * (
        grid['gdet'] * params['dx2'] * params['dx3'])).sum(-1).sum(-1) 

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
                        description="""Calculate 1D heatng profiles for single 
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
    parser.add_argument('--radiation', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if radiation quantities
                        should be included in profile extraction.""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if two-temperature 
                        physics should be included in profile extraction.""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if profile extraction
                        should be done on reduced data.""")
    parser.add_argument('--restart', action='store_true', required=False, 
                        help="""boolean set to determine if profile 
                        integration should continue from last slice_num""")
    parser.add_argument('--shared', action='store_true', required=False, 
                        help="""boolean set to determine if proj-shared 
                        should be used as the location of the data directory""")
    parser.add_argument('--title', type=str, required=False, default=None, 
                        help="""title of simulation used for Averages directory
                        if different from the simulation title in proj-shared""")
    args = parser.parse_args()
    main(**vars(args))