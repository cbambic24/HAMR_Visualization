import sys
import os
import argparse
import numpy as np
sys.path.insert(0, '/lustre/orion/ast178/proj-shared/' + 
    'cbambic/HAMR_Visualization/Read_HAMR_Data')
import hamr_read as hr
import analysis as ana
import radiation as rad
from mpi4py import MPI
import tracemalloc
import time as get_time

def main(**kwargs):
    fnum    = int(kwargs.get('fnum'))
    nslices = int(kwargs.get('nslices'))
    restart = kwargs.get('restart')
    
    main_dir = '/lustre/orion/ast178/proj-shared/cbambic/'
    sim      = 'Thin_RAD_MAD_HiRes'
    profiles = {}
        
    # Quantities (quants) for profile averaging
    keys   = [
        'rho',  'Ti',   'Te', 
        'Pmag', 'Pthm', 'Prad', 
        'Eion', 'Eelc', 'Vol'
    ]
    acc_keys = [
        'Mdot', 'Edot', 'Mdot_sigma', 'Edot_sigma',
        'Edot_Poyt', 'Edot_Fluid', 'Edot_Rad', 'Phi_BH'
    ]
    cool_keys = [
        'qIC', 'qie', 'qemi', 'qabs'
    ]
    vec_keys = [
        'vR',   'vphi',  'vz', 
        'bR',   'bphi',  'bz'
    ]

    # First, determine the limits of the slices.
    H = hr.Read(main_dir + 'DATA/' + sim, 
	    cgs_data = True, cgs_grid = True, MBH = 10.0)
    H.read_params(fnum)
    profiles['t'] = hr.params['t']
    profiles['a'] = hr.params['a']

    nx = int(hr.params['nb1'] * hr.params['bs1'] * H.upscalex1)
    nz = int(hr.params['nb3'] * hr.params['bs3'] * H.upscalex3)
    phi_1D = np.zeros(nz)
    for k in range(0, nz):
        phi_1D[k] = (hr.params['startx3'] + (k + 0.5) * hr.params['dx3'])

    dphi = phi_1D[1] - phi_1D[0]

    if (len(phi_1D) % nslices != 0):
        raise ValueError(
                "nslices must evenly divide len(phi_1D) = %d" %len(phi_1D))    
    iphi_stp = int(len(phi_1D)/nslices)

    # Logic to allow restarts if profile extraction is interrupted.
    if not restart:
        iphi_start = 0
        first_call = True
    else:
        profile_file = (main_dir + 'Averages/' + sim + 
            '/Profiles_%d.npy' %fnum)
        profiles, iphi_start = read_restart(profile_file, iphi_stp)
        first_call = False

    # Initialize profiles dictionary
    if first_call:
        for key in keys + acc_keys + cool_keys + vec_keys:
            profiles['%s' %key]= np.zeros(nx)
        first_call = False

    counter = 0
    # Outer loop over whole data slice that fits on single node.
    # Inner loop over slivers of data slice assigned to each core.
    for i in range(iphi_start, len(phi_1D), iphi_stp):
        profiles['saved'] = False
        phi_min = phi_1D[i]
        phi_max = phi_1D[i + iphi_stp - 1]

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        prof_loc = {}

        delta_phi   = (phi_max - phi_min) / size
        phi_min_loc = phi_min + delta_phi * rank 
        phi_max_loc = phi_min + delta_phi * (rank + 1)
       
        if (rank == 0): 
            phi_min_loc -= dphi 

        H = hr.Read(main_dir + 'DATA/Thin_RAD_MAD_HiRes', 
	    cgs_data = True, cgs_grid = True, MBH = 10.0,
            coords1D = True,
            phi_min = phi_min_loc, 
            phi_max = phi_max_loc)
        
        # Define constants
        MBH = 10.0
        Msun = hr.CONSTANTS['Msun']
        c    = hr.CONSTANTS['c']
        G    = hr.CONSTANTS['G']
        Rg   = G*MBH*Msun/(c*c)

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

        tracemalloc.start()
        # Read grid and data
        H.read_grid(fnum, direct=True)
        H.read_prims(fnum, skip_read_params=True)
        k_min, k_max = H.block_info['k_min'], H.block_info['k_max']

        # Initialize Accretion class
        Acc = ana.Accretion(H.grid, H.data, hr.params, hr.CONSTANTS, 
            cgs=True, MBH=10.0, radiation=True)
        # Initialize Coordinates class
        Coords = ana.Coordinates(H.grid['r'], H.grid['theta'], H.grid['phi'],
            hr.CONSTANTS, cgs=True, MBH=10.0)
        # Initialize Fluid class
        Fluid = ana.Fluid(hr.params, hr.CONSTANTS, cgs=True)
        # Initialize Field class
        Field = ana.Field(hr.params, hr.CONSTANTS, cgs=True)
        # Initialize Radiation class
        Radi  = rad.Radiation(hr.params, hr.CONSTANTS, cgs=True)

        # Calculate sigma (3D)
        b     = Field.calc_b(H.data['B'], H.data['uu'], H.grid['gcov'])
        b2    = Field.calc_b2(b, H.data['B'], H.grid['gcov'])
        sigma = (b2 / (4. * np.pi)) / (H.data['rho'] * c * c)

        # Calculate u^mu u_nu tensor
        uu_tnsr = ana.Tools.construct_dyadic(H.data['uu'] / c, H.grid['gcov'], 
            upper_lower=True, index=[1,0])
        
        # Calculate gas pressure
        pgas  = Fluid.calc_pgas(H.data['u_gas'], H.data['Te'], H.data['Ti'])
        
        # Mass and energy accretion rates. Radial magnetic flux (Phi_BH).
        stress_energy_HD  = Fluid.calc_THD(H.data['rho'], H.data['uu'], 
            H.data['u_gas'], pgas, H.grid['gcov'], H.grid['gcon'], 
            uu_tnsr=uu_tnsr, upper_lower=True, index=[1, 0])
        stress_energy_EM  = Field.calc_TEM(H.data['uu'], H.data['B'], 
            H.grid['gcov'], H.grid['gcon'], b2 = b2, b = b, 
            uu_tnsr = uu_tnsr, upper_lower=True, index=[1, 0])
        stress_energy_RAD = Radi.calc_TRAD(H.data['u_rad'], H.data['uu_rad'], 
            H.grid['gcov'], H.grid['gcon'], 
            upper_lower=True, index=[1, 0])
        del uu_tnsr
        
        stress_energy_MHD = stress_energy_HD + stress_energy_EM + stress_energy_RAD

        prof_loc['Mdot']      = Acc.calc_Mdot()[0]
        prof_loc['Edot']      = Acc.calc_Edot(TMHD = stress_energy_MHD)[0]
        prof_loc['Mdot_sigma']= Acc.calc_Mdot(sigma_cut = 3.0, 
                                    sigma = sigma)[0]
        prof_loc['Edot_sigma']= Acc.calc_Edot(TMHD = stress_energy_MHD, 
                                    sigma_cut = 3.0, sigma = sigma)[0]
        prof_loc['Edot_Poyt'] = Acc.calc_Edot(TMHD = stress_energy_EM, 
                                    sigma_cut = 3.0, sigma = sigma)[0]
        prof_loc['Edot_Fluid']=Acc.calc_Edot(TMHD = stress_energy_HD, 
                                    sigma_cut = 3.0, sigma = sigma)[0]
        prof_loc['Edot_Rad']  = Acc.calc_Edot(TMHD = stress_energy_RAD, 
                                    sigma_cut = 3.0, sigma = sigma)[0]
        prof_loc['Phi_BH']    = Acc.calc_PhiBH()[0] 
        print("Completed Mdot, Edot, Phi_BH on rank %d" %rank, flush=True)
        del stress_energy_MHD 
        del stress_energy_HD, stress_energy_EM, stress_energy_RAD        
        
        # Deallocate magnetic field in the lab frame and radiation 4-momentum
        # as well as metric tensor and inverse
        H.data['B']      = H.data['rho']
        H.data['uu_rad'] = H.data['rho']
        H.grid['gcov']   = H.data['rho']
        H.grid['gcon']   = H.data['rho']
        
        # Calculate gamma (ratio of specific heats)
        # Equation 28 in Sadowski+ (2017)
        Theta_e = (kB * H.data['Te']) / (me * c * c)
        Theta_i = (kB * H.data['Ti']) / (mi * c * c)
 
        gam_e = (10. + 20. * Theta_e) / (6. + 15. * Theta_e)
        gam_i = (10. + 20. * Theta_i) / (6. + 15. * Theta_i)

        quants = [
            H.data['rho'], H.data['Ti'], H.data['Te'], 
            b2/(8. * np.pi), pgas, (1./3.) * H.data['u_rad'], 
            (1./(gam_i-1.)) * (
            (H.data['rho'] / (mu_i * mp)) * kB * H.data['Ti']),
            (1./(gam_e-1.)) * (
            (H.data['rho'] / (mu_e * mp)) * kB * H.data['Te']), 1.0
        ]

        for (key, quant) in zip(keys, quants):
            prof_loc['%s' %key] = thph_sum(quant, 
                H.grid, H.data, hr.params, hr.CONSTANTS, 
                weight=H.data['rho'], 
                sigma_cut=3.0, sigma=sigma)[0]
        
        del Theta_e, Theta_i, gam_e, gam_i, pgas

        qIC = Radi.calc_qIC(H.data['rho'], 
            H.data['Te'], H.data['u_rad'], cooling=True)
        prof_loc['qIC'] = thph_sum(qIC, H.grid, H.data, 
            hr.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma)[0]
        del qIC
        qie = Fluid.calc_qie(H.data['rho'], 
            H.data['Te'], H.data['Ti'], cooling=True)
        qie[qie < 0.] = 0.
        prof_loc['qie'] = thph_sum(qie, H.grid, H.data, 
            hr.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma)[0]
        del qie
        qemi= Radi.calc_q_emi(H.data['rho'], H.data['Te'], 
            H.data['u_rad'], b2=b2, XRB=True)
        prof_loc['qemi'] = thph_sum(qemi, H.grid, H.data, 
            hr.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma)[0]
        del qemi
        qabs= Radi.calc_q_emi(H.data['rho'], H.data['Te'], 
            H.data['u_rad'], b2=b2, XRB=True)
        prof_loc['qabs'] = thph_sum(qabs, H.grid, H.data, 
            hr.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma)[0]
        del qabs

        # Deallocate temperatures and energy densities 
        # in preparation for large transformation matrices
        H.data['Ti']    = H.data['rho']
        H.data['Te']    = H.data['rho']
        H.data['u_gas'] = H.data['rho']
        H.data['u_rad'] = H.data['rho']

        uu_cyl = Coords.int_to_cyl_vec(H.data['uu'])
        H.data['uu'] = H.data['rho']
        b_cyl  = Coords.int_to_cyl_vec(b)

        quants = [
            uu_cyl[1] / uu_cyl[0],
            uu_cyl[2] / uu_cyl[0],
            uu_cyl[3] / uu_cyl[0],
            b_cyl[1], b_cyl[2], b_cyl[3]
        ]

        for (key, quant) in zip(vec_keys, quants):
            prof_loc['%s' %key] = thph_sum(quant, 
                H.grid, H.data, hr.params, hr.CONSTANTS, 
                weight=H.data['rho'], 
                sigma_cut=3.0, sigma=sigma)[0]

        del uu_cyl, b_cyl
        if (rank == 0):
            print("End of iteration %d" %counter, flush=True)
        counter += 1

        if (rank == 0):
            # Put spherical radius in dictionary 
            profiles['r']   = H.grid['r'][0, :, 0, 0] / Rg
            profiles['dx1'] = hr.params['dx1']

        # Gather local profiles (prof_loc) from all processors and 
        # place them in main profiles dictionary
        prof_global = comm.gather(prof_loc)
        
        if (rank == 0):
            for prof in prof_global:
                for key in keys + acc_keys + cool_keys + vec_keys:
                    profiles['%s' %key] += prof_loc['%s' %key]

            profiles['k_max'] = k_min + (k_max - k_min) * size 
            print("Saving profiles ...", flush=True)
            if not os.path.exists(main_dir + 'Averages/' + sim):
                os.makedirs(main_dir + 'Averages/' + sim)

            profiles['saved'] = True
            np.save(main_dir + 'Averages/' + sim + 
                    '/Profiles_%d' %fnum, profiles)
            print("Done saving profiles on rank 0.", flush=True)

        print("Done with current slice on rank %d." %rank, flush=True)
    print("Done with all profiles.")

def thph_sum(q, grid, data, params, CONSTANTS, **kwargs):
    w    = kwargs.get('weight', 1.0)
    Acc2 = ana.Accretion(grid, data, params, CONSTANTS, cgs=True)
    return (q * w * Acc2.sigma_cut(**kwargs) * (
        grid['gdet'] * params['dx2'] * params['dx3'])).sum(-1).sum(-1) 

def read_restart(profile_file, iphi_stp):
    profiles_in = np.load(profile_file, allow_pickle=True).ravel()[0]
    iphi_end    = profiles_in['k_max'] + 1
    if (profiles_in['saved']):
        iphi_start  = iphi_end - 1
    else:
        iphi_start  = iphi_end - iphi_stp
    return profiles_in, iphi_start

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Calculate 1D profiles for single 
                        ultra-high-resolution simulation output.""")
    parser.add_argument('fnum', help = """dump number""")
    parser.add_argument('--nslices', type=str, required=False, help="""
                        number of slices in phi used to load the data""")
    parser.add_argument('--restart', action='store_true', required=False, 
                        help="""boolean set to determine if profile 
                        integration should continue from last k_max""")
    args = parser.parse_args()
    main(**vars(args))

