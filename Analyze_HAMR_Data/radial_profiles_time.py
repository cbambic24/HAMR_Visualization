import sys
import os
import argparse
import numpy as np
sys.path.insert(0, '/lustre/orion/ast178/proj-shared/' +
    'cbambic/HAMR_Visualization/NewObj_Read_HAMR_Data')
import hamr_read as hr
import analysis as ana
import coordinates as coords
import radiation as rad
from mpi4py import MPI
import time as timeit

def main(**kwargs):
    main_dir  = '/lustre/orion/ast178/proj-shared/cbambic/'
    sim       = kwargs.get('sim')
    start_file= int(kwargs.get('start_file'))
    end_file  = int(kwargs.get('end_file'))
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

    if (two_temp or radiation):
        kB   = hr.CONSTANTS['kB']
        mp   = hr.CONSTANTS['mp'] 

        x_AB = hr.CONSTANTS['X_AB']
        y_AB = hr.CONSTANTS['X_AB']

        # Equation 11 in Sadowski+ (2017)
        mu_i = 4.0 / (4.0 * x_AB + y_AB)
        # Equation 10 in Sadowski+ (2017)
        mu_e = 2.0 / (1.0 + x_AB)
        mi   = (x_AB + y_AB) * mp  
        me   = hr.CONSTANTS['me'] 

        # Given in Kaaz et al. (2024)
        mu_g = 4.0 / (6.0 * x_AB + y_AB + 2.)
    
    # Determine data extraction and save directories
    data_dir = get_data_dir(sim, reduced, shared, main_dir)
    save_dir = get_save_dir(sim, title, 'Averages', 'Profiles', main_dir)

    nzeros = len(str(int(os.popen(
        f'ls -d {data_dir}/dumps* | wc -l').read())))
        
    # Quantities (quants) for profile averaging
    keys = [
        'rho', 'Ti', 'Pmag', 'Pthm', 'Vol', 'Dtheta2'
    ]
    # Accretion rates (mass and energy) + Phi_BH
    acc_keys = [
        'Mdot', 'Edot', 'Mdot_sigma', 'Edot_sigma',
        'Edot_Poyt', 'Edot_Fluid', 'Phi_BH'
    ]
    if (two_temp and radiation):
        keys += ['Te', 'Prad', 'Eion', 'Eelc']
        acc_keys.append('Edot_Rad')
    elif (radiation and not two_temp):
        keys.append('Prad')
        acc_keys.append('Edot_Rad')

    # Cooling diagnostics
    if (two_temp and not radiation):
        cool_keys = ['qie']
    if (radiation):
        cool_keys = [
            'qIC', 'qemi', 'qabs', 
            'qIC_thin', 'qemi_thin', 'qIC_thick', 'qemi_thick'
        ]
    if (two_temp and radiation):
        cool_keys += ['qie', 'qie_thin', 'qie_thick']

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

    file_list_local = get_file_list_local(file_list, size, rank, data_dir)

    read_args = {
        'cgs_grid' : True,
        'cgs_data' : True,
        'coords1D' : True,
        'MBH'      : MBH,
        'lowres1'  : lowres1,
        'lowres2'  : lowres2,
        'lowres3'  : lowres3
    }

    if (radiation or two_temp):
        profile_keys = keys + acc_keys + cool_keys + vec_keys
    else:
        profile_keys = keys + acc_keys + vec_keys

    # Outermost loop over file number (each file on single core)
    for file_num in file_list_local:
        profiles, nphi, iphi_stp = init_profiles(data_dir, file_num, 
            profile_keys, save_dir, nslices, restart, nzeros, **read_args)

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

            # Calculate u^mu u_nu tensor
            uu_tnsr = ana.Tools.construct_dyadic(H.data['uu'] / c, H.grid['gcov'], 
                upper_lower=True, index=[1,0], cpp=True)
            
            # Calculate gas pressure
            if (two_temp):
                pgas= Fluid.calc_pgas(**H.data)
            else:
                gamma_adi = (5. / 3.)
                pgas= (gamma_adi - 1.) * H.data['u_gas']

            # Mass and energy accretion rates. Radial magnetic flux (Phi_BH).
            stress_energy_HD  = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas,
                uu_tnsr=uu_tnsr, upper_lower=True, index=[1, 0], cpp=True)
            stress_energy_EM  = Field.calc_TEM(**H.data, **H.grid, b2=b2, b=b,
                uu_tnsr=uu_tnsr, upper_lower=True, index=[1, 0], cpp=True)
            if (radiation):
                stress_energy_RAD = Radi.calc_TRAD(**H.data, **H.grid, 
                    upper_lower=True, index=[1, 0], cpp=True)
            del uu_tnsr
            
            if (radiation):
                stress_energy_MHD = (
                    stress_energy_HD + stress_energy_EM + stress_energy_RAD)
            else:
                stress_energy_MHD = stress_energy_HD + stress_energy_EM

            acc_args = {
                'sigma_cut' : 3.,
                'sigma'     : sigma
            }

            profiles['Mdot']      += Acc.calc_Mdot()[0]
            profiles['Edot']      += Acc.calc_Edot(TMHD=stress_energy_MHD)[0]
            profiles['Mdot_sigma']+= Acc.calc_Mdot(**acc_args)[0]
            profiles['Edot_sigma']+= Acc.calc_Edot(TMHD=stress_energy_MHD, **acc_args)[0]
            profiles['Edot_Poyt'] += Acc.calc_Edot(TMHD=stress_energy_EM,  **acc_args)[0]
            profiles['Edot_Fluid']+= Acc.calc_Edot(TMHD=stress_energy_HD,  **acc_args)[0]
            profiles['Phi_BH']    += Acc.calc_PhiBH()[0] 

            if (radiation):
                profiles['Edot_Rad'] += Acc.calc_Edot(TMHD=stress_energy_RAD, **acc_args)[0]
            
            del stress_energy_MHD 
            del stress_energy_HD, stress_energy_EM, stress_energy_RAD        
            
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
        
            if (two_temp):
                # Calculate gamma (ratio of specific heats)
                # Equation 28 in Sadowski+ (2017)
                Theta_e = (kB * H.data['Te']) / (me * c * c)
                Theta_i = (kB * H.data['Ti']) / (mi * c * c)
     
                gam_e = (10. + 20. * Theta_e) / (6. + 15. * Theta_e)
                gam_i = (10. + 20. * Theta_i) / (6. + 15. * Theta_i)

            if (two_temp and radiation):
                quants = [
                    H.data['rho'], H.data['Ti'], b2/(8. * np.pi), pgas,
                    1.0, np.square(H.grid['theta'] - np.pi/2.),
                    H.data['Te'], (1./3.) * H.data['u_rad'],
                    (1./(gam_i-1.)) * (
                    (H.data['rho'] / (mu_i * mp)) * kB * H.data['Ti']),
                    (1./(gam_e-1.)) * (
                    (H.data['rho'] / (mu_e * mp)) * kB * H.data['Te'])
                ]
            elif (radiation and not two_temp):
                T_gas = pgas / H.data['rho']
                quants = [
                    H.data['rho'], T_gas, b2/(8. * np.pi), pgas,
                    1.0, np.square(H.grid['theta'] - np.pi/2.),
                    (1./3.) * H.data['u_rad']
                ]
            else:
                T_gas = pgas / H.data['rho']
                quants = [
                    H.data['rho'], T_gas, b2/(8. * np.pi), pgas,
                    1.0, np.square(H.grid['theta'] - np.pi/2.)
                ]

            for (key, quant) in zip(keys, quants):
                profiles[f'{key}'] += thph_sum(quant, 
                    H.grid, H.data, H.params, hr.CONSTANTS, 
                    weight=H.data['rho'], **acc_args)[0]
            
            if (two_temp):
                del Theta_e, Theta_i, gam_e, gam_i, pgas
            else:
                del pgas, T_gas
            
            if not radiation:
                del b2
                
            if (radiation):
                tau_args = {
                    'tau_cut'   : 1.,
                    'tau'       : tau_es,
                    'sigma_cut' : 3.,
                    'sigma'     : sigma
                }

            if (two_temp):
                qie = Fluid.calc_qie(**H.data, cpp=True)
                # cooling = True (maybe)
                #qie[qie < 0.] = 0.
                profiles['qie'] += thph_sum(qie, H.grid, H.data, H.params, 
                    hr.CONSTANTS, weight=uu_t, **acc_args)[0]
                if (radiation):
                    profiles['qie_thin'] += thph_sum(qie, H.grid, H.data, H.params, 
                        hr.CONSTANTS, weight=uu_t, optically_thin = True,  **tau_args)[0]
                    profiles['qie_thick']+= thph_sum(qie, H.grid, H.data, H.params, 
                        hr.CONSTANTS, weight=uu_t, optically_thin = False, **tau_args)[0]
                del qie
            
            if (radiation):
                qIC = Radi.calc_qIC(**H.data)
                # cooling = True (maybe)
                profiles['qIC'] += thph_sum(qIC, H.grid, H.data, H.params, 
                    hr.CONSTANTS, weight=uu_t, **acc_args)[0]
                profiles['qIC_thin']  += thph_sum(qIC, H.grid, H.data, H.params, 
                    hr.CONSTANTS, weight=uu_t, optically_thin=True,  **tau_args)[0]
                profiles['qIC_thick'] += thph_sum(qIC, H.grid, H.data, H.params, 
                    hr.CONSTANTS, weight=uu_t, optically_thin=False, **tau_args)[0]
                del qIC

                qemi=Radi.calc_q_emi(**H.data, b2=b2, XRB=True, cpp=True)
                profiles['qemi'] += thph_sum(qemi, H.grid, H.data, H.params, 
                    hr.CONSTANTS, weight=uu_t, **acc_args)[0]
                profiles['qemi_thin']  += thph_sum(qemi, H.grid, H.data, H.params, 
                    hr.CONSTANTS, weight=uu_t, optically_thin=True,  **tau_args)[0]
                profiles['qemi_thick'] += thph_sum(qemi, H.grid, H.data, H.params, 
                    hr.CONSTANTS, weight=uu_t, optically_thin=False, **tau_args)[0]
                del qemi

                qabs= Radi.calc_q_abs(**H.data, b2=b2, XRB=True, cpp=True)
                profiles['qabs'] += thph_sum(qabs, H.grid, H.data, H.params, 
                    hr.CONSTANTS, weight=uu_t, **acc_args)[0]
                del qabs

            # Deallocate temperatures and energy densities 
            # in preparation for large transformation matrices
            if (two_temp and radiation):
                H.data['Ti'] = H.data['rho']
                H.data['Te'] = H.data['rho']
                del uu_t, b2, tau_es
            elif (two_temp and not radiation):
                H.data['Ti'] = H.data['rho']
                H.data['Te'] = H.data['rho']
                del uu_t

            # Compute theta and phi averages of vector quantities
            uu_cyl = Coords.int_to_cyl_vec(H.data['uu'], cpp=True)
            H.data['uu'] = H.data['rho']
            b_cyl  = Coords.int_to_cyl_vec(b, cpp=True)

            quants = [
                uu_cyl[1], uu_cyl[2], uu_cyl[3],
                np.nan_to_num(uu_cyl[1] / uu_cyl[0]),
                np.nan_to_num(uu_cyl[2] / uu_cyl[0]),
                np.nan_to_num(uu_cyl[3] / uu_cyl[0]),
                b_cyl[1], b_cyl[2], b_cyl[3]
            ]

            for (key, quant) in zip(vec_keys, quants):
                profiles[f'{key}'] += thph_sum(quant, H.grid, H.data, 
                    H.params, hr.CONSTANTS, weight=H.data['rho'], **acc_args)[0]

            del uu_cyl, b_cyl, sigma
            
            # Density scale
            profiles['rho0'] = H.params['density_scale']
            # Put spherical radius in dictionary 
            profiles['r']   = H.grid['r'][0, :, 0, 0] / Rg
            profiles['dx1'] = H.params['dx1']
            profiles['k_max'] = k_min + (k_max - k_min)

            print(f'Rank {rank:d} done analyzing range of k: ' + 
                f'({k_min:d}, {k_max:d}).', flush=True)
       
            snum = int(slice_num + 1)
            print(f'Finished with slice {snum:d} of {nslices} ' +
                f'for file {file_num}', flush=True) 

        fnum_str = str(file_num).zfill(nzeros)
        np.save(save_dir + f'/Profiles_{fnum_str}', profiles)
        profiles['saved'] = True
        profiles['slice_num_restart'] = slice_num + 1

        print(f'Done with all profiles for file number {file_num}.',flush=True) 

def thph_sum(q, grid, data, params, CONSTANTS, **kwargs):
    w    = kwargs.get('weight', 1.0)
    Acc2 = ana.Accretion(grid, data, params, CONSTANTS, 
            cgs=True, radiation=True, MBH = 10.0)
    return (q * w * Acc2.sigma_cut(**kwargs) * Acc2.tau_cut(**kwargs) * (
        grid['gdet'] * params['dx2'] * params['dx3'])).sum(-1).sum(-1) 

def read_restart(profile_file, lowres1, lowres2, lowres3):
    profiles_in = np.load(profile_file, allow_pickle=True).ravel()[0]
    if (lowres1 != profiles_in['lowres1']):
        raise ValueError("Chosen lowres1 parameter incompatible with restart")
    if (lowres2 != profiles_in['lowres2']):
        raise ValueError("Chosen lowres2 parameter incompatible with restart")
    if (lowres3 != profiles_in['lowres3']):
        raise ValueError("Chosen lowres3 parameter incompatible with restart")
    return profiles_in

def get_data_dir(sim, reduced, shared, main_dir):
    if not reduced and not shared:
        data_dir = main_dir + 'DATA/' + sim
    elif reduced and not shared:
        data_dir = main_dir + 'DATA/' + sim + '/reduced'
    elif not reduced and shared:
        data_dir = '/lustre/orion/ast178/proj-shared/' + sim
    elif reduced and shared:
        data_dir = '/lustre/orion/ast178/proj-shared/' + sim + '/reduced'
    return data_dir 

def get_save_dir(sim, title, type, diagnostic, main_dir):
    if title == None:
        save_dir = main_dir + f'{type}/' + sim + f'/{diagnostic}'
    else:
        save_dir = main_dir + f'{type}/' + title + f'/{diagnostic}'
    return save_dir

def get_file_list_local(file_list, size, rank, data_dir):
    nout             = len(file_list)
    nfiles_nproc     = int(np.floor(nout / size))
    nfiles_remaining = nout % size
    files_per_proc   = ((size - nfiles_remaining) * [nfiles_nproc] +
        nfiles_remaining * [nfiles_nproc + 1])
    file_list_local  = ((sum(files_per_proc[:rank]) +
        np.arange(0, files_per_proc[rank])) + file_list[0]).tolist()
    
    file_list_local = check_dumps_exist(data_dir, file_list_local)
    
    return file_list_local

def check_dumps_exist(path_to_dumps, file_list):
    for file_num in file_list:
        if not os.path.isdir(path_to_dumps + f'/dumps{file_num}'):
            print(f'Dumps {file_num} is missing.')
            file_list.remove(file_num)
            if os.path.isdir(path_to_dumps + f'/dumps{file_num}_'):
                found_file = path_to_dumps + f'/dumps{file_num}_'
                print(f'The file can be found as {found_file}', flush=True) 
    return file_list

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
        profile_file = f'{save_dir}/Profiles_{fnum_str}'
        profiles = read_restart(profile_file)
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
    parser.add_argument('--radiation', action='store_true', required=False,
                        default=True,
                        help="""boolean to determine if radiation quantities
                        should be included in profile extraction 
                        (default: True).""")
    parser.add_argument('--two_temp', action='store_true', required=False,
                        default=True,
                        help="""boolean to determine if two-temperature 
                        physics should be included in profile extraction 
                        (default: True).""")
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