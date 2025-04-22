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

def main(**kwargs):
    main_dir= '/lustre/orion/ast178/proj-shared/cbambic/'
    sim     = kwargs.get('sim')
    fnum    = kwargs.get('fnum')
    nslices = kwargs.get('nslices')
    lowres1 = kwargs.get('lowres1')
    lowres2 = kwargs.get('lowres2')
    lowres3 = kwargs.get('lowres3')
    reduced = kwargs.get('reduced')
    restart = kwargs.get('restart')
    
    if not reduced:
        data_dir = main_dir + 'DATA/' + sim
    else:
        data_dir = main_dir + 'DATA/' + sim + '/reduced'

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
        'qIC', 'qie', 'qemi', 'qabs',
        'qIC_thin',  'qie_thin',  'qemi_thin',
        'qIC_thick', 'qie_thick', 'qemi_thick'
    ]
    vec_keys = [
        'vR',   'vphi',  'vz', 
        'bR',   'bphi',  'bz'
    ]

    # First, determine the limits of the slices.
    H = hr.Read(data_dir, cgs_data = True, cgs_grid = True, MBH = 10.0)
    H.read_params(fnum)
    profiles['t'] = H.params['t']
    profiles['a'] = H.params['a']

    nx = int(H.params['nb1'] * H.params['bs1'] * H.upscalex1 / lowres1)
    nz = int(H.params['nb3'] * H.params['bs3'] * H.upscalex3 / lowres3)
    phi_1D = np.zeros(nz)
    for k in range(0, nz):
        phi_1D[k] = (
            H.params['startx3'] + (k + 0.5) * H.params['dx3'] * lowres3)

    dphi = phi_1D[1] - phi_1D[0]

    if (len(phi_1D) % nslices != 0):
        raise ValueError(
            "nslices must evenly divide len(phi_1D) = %d" %len(phi_1D))    
    iphi_stp = int(len(phi_1D)/nslices)

    # Logic to allow restarts if profile extraction is interrupted.
    if not restart:
        first_call = True
    else:
        profile_file = (main_dir + 'Averages/' + sim + 
            '/Profiles_%d.npy' %fnum)
        profiles = read_restart(profile_file)
        first_call = False

    # Initialize profiles dictionary
    if first_call:
        for key in keys + acc_keys + cool_keys + vec_keys:
            profiles['%s' %key]= np.zeros(nx)
        first_call = False
        profiles['slice_num_restart'] = 0

    # Outer loop over whole data slice that fits on single node.
    # Inner loop over slivers of data slice assigned to each core.
    for slice_num, i in enumerate(range(0, len(phi_1D), iphi_stp)):
        if (restart):
            while slice_num < profiles['slice_num_restart']:
                continue
     
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

        H = hr.Read(data_dir, cgs_data = True, cgs_grid = True, 
            MBH = 10.0, coords1D = True, 
            lowres1 = lowres1, lowres2 = lowres2, lowres3 = lowres3,
            phi_min = phi_min_loc, phi_max = phi_max_loc)
        
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

        # Read grid and data
        H.read_grid(fnum, direct=True)
        H.read_prims(fnum, skip_read_params=True)
        k_min, k_max = H.block_info['k_min'], H.block_info['k_max']

        # Initialize Accretion class
        Acc = ana.Accretion(H.grid, H.data, H.params, hr.CONSTANTS, 
            cgs=True, MBH=10.0, radiation=True)
        # Initialize Coordinates class
        Coords = ana.Coordinates(H.grid['r'], H.grid['theta'], H.grid['phi'],
            H.params, hr.CONSTANTS, cgs=True, MBH=10.0)
        # Initialize Fluid class
        Fluid = ana.Fluid(H.params, hr.CONSTANTS, cgs=True)
        # Initialize Field class
        Field = ana.Field(H.params, hr.CONSTANTS, cgs=True)
        # Initialize Radiation class
        Radi  = rad.Radiation(H.params, hr.CONSTANTS, cgs=True)

        # Calculate sigma (3D)
        b     = Field.calc_b(H.data['B'], H.data['uu'], 
                             H.grid['gcov'], cpp=True)
        b2    = Field.calc_b2(b, H.data['B'], H.grid['gcov'], cpp=True)
        sigma = (b2 / (4. * np.pi)) / (H.data['rho'] * c * c)
        
        # Calculate u^mu u_nu tensor
        uu_tnsr = ana.Tools.construct_dyadic(H.data['uu'] / c, H.grid['gcov'], 
            upper_lower=True, index=[1,0], cpp=True)
        
        # Calculate gas pressure
        pgas  = Fluid.calc_pgas(H.data['u_gas'], H.data['Te'], H.data['Ti'])
        
        # Mass and energy accretion rates. Radial magnetic flux (Phi_BH).
        stress_energy_HD  = Fluid.calc_THD(H.data['rho'], H.data['uu'], 
            H.data['u_gas'], pgas, H.grid['gcov'], H.grid['gcon'], 
            uu_tnsr=uu_tnsr, upper_lower=True, index=[1, 0], cpp=True)
        stress_energy_EM  = Field.calc_TEM(H.data['uu'], H.data['B'], 
            H.grid['gcov'], H.grid['gcon'], b2 = b2, b = b, 
            uu_tnsr = uu_tnsr, upper_lower=True, index=[1, 0], cpp=True)
        stress_energy_RAD = Radi.calc_TRAD(H.data['u_rad'], H.data['uu_rad'], 
            H.grid['gcov'], H.grid['gcon'], 
            upper_lower=True, index=[1, 0], cpp=True)
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
        prof_loc['Edot_Fluid']= Acc.calc_Edot(TMHD = stress_energy_HD, 
                                    sigma_cut = 3.0, sigma = sigma)[0]
        prof_loc['Edot_Rad']  = Acc.calc_Edot(TMHD = stress_energy_RAD, 
                                    sigma_cut = 3.0, sigma = sigma)[0]
        prof_loc['Phi_BH']    = Acc.calc_PhiBH()[0] 
        
        del stress_energy_MHD 
        del stress_energy_HD, stress_energy_EM, stress_energy_RAD        
        
        opacities = dict.fromkeys(['electron_scattering'])
        tau_es    = Acc.calc_tau(opacities = opacities, integral = 'theta', 
                        sigma_cut = 3.0, sigma = sigma, Te = H.data['Te'])

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
                H.grid, H.data, H.params, hr.CONSTANTS, 
                weight=H.data['rho'], 
                sigma_cut=3.0, sigma=sigma)[0]
        
        del Theta_e, Theta_i, gam_e, gam_i, pgas
        
        qIC = Radi.calc_qIC(H.data['rho'], 
            H.data['Te'], H.data['u_rad'], cooling=True)
        prof_loc['qIC']       = thph_sum(qIC, H.grid, H.data, 
            H.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma)[0]
        prof_loc['qIC_thin']  = thph_sum(qIC, H.grid, H.data, 
            H.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma,
            tau_cut = 1.0, tau = tau_es, optically_thin = True)[0]
        prof_loc['qIC_thick'] = thph_sum(qIC, H.grid, H.data, 
            H.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma,
            tau_cut = 1.0, tau = tau_es, optically_thin = False)[0]
        del qIC
        qie = Fluid.calc_qie(H.data['rho'], 
            H.data['Te'], H.data['Ti'], cooling=True, cpp=True)
        qie[qie < 0.] = 0.
        prof_loc['qie']       = thph_sum(qie, H.grid, H.data, 
            H.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma)[0]
        prof_loc['qie_thin']  = thph_sum(qie, H.grid, H.data, 
            H.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma,
            tau_cut = 1.0, tau = tau_es, optically_thin = True)[0]
        prof_loc['qie_thick'] = thph_sum(qie, H.grid, H.data, 
            H.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma,
            tau_cut = 1.0, tau = tau_es, optically_thin = False)[0]
        del qie
        qemi= Radi.calc_q_emi(H.data['rho'], H.data['Te'], 
            H.data['u_rad'], b2=b2, XRB=True, cpp=True)
        prof_loc['qemi']       = thph_sum(qemi, H.grid, H.data, 
            H.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma)[0]
        prof_loc['qemi_thin']  = thph_sum(qemi, H.grid, H.data, 
            H.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma,
            tau_cut = 1.0, tau = tau_es, optically_thin = True)[0]
        prof_loc['qemi_thick'] = thph_sum(qemi, H.grid, H.data, 
            H.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma,
            tau_cut = 1.0, tau = tau_es, optically_thin = False)[0]
        del qemi
        qabs= Radi.calc_q_abs(H.data['rho'], H.data['Te'], 
            H.data['u_rad'], b2=b2, XRB=True, cpp=True)
        prof_loc['qabs'] = thph_sum(qabs, H.grid, H.data, 
            H.params, hr.CONSTANTS, sigma_cut=3.0, sigma=sigma)[0]
        del qabs

        # Deallocate temperatures and energy densities 
        # in preparation for large transformation matrices
        H.data['Ti']    = H.data['rho']
        H.data['Te']    = H.data['rho']
        H.data['u_gas'] = H.data['rho']
        H.data['u_rad'] = H.data['rho']

        uu_cyl = Coords.int_to_cyl_vec(H.data['uu'], cpp=True)
        H.data['uu'] = H.data['rho']
        b_cyl  = Coords.int_to_cyl_vec(b, cpp=True)

        quants = [
            uu_cyl[1] / uu_cyl[0],
            uu_cyl[2] / uu_cyl[0],
            uu_cyl[3] / uu_cyl[0],
            b_cyl[1], b_cyl[2], b_cyl[3]
        ]

        for (key, quant) in zip(vec_keys, quants):
            prof_loc['%s' %key] = thph_sum(quant, 
                H.grid, H.data, H.params, hr.CONSTANTS, 
                weight=H.data['rho'], 
                sigma_cut=3.0, sigma=sigma)[0]

        del uu_cyl, b_cyl
        
        if (rank == 0):
            # Put spherical radius in dictionary 
            profiles['r']   = H.grid['r'][0, :, 0, 0] / Rg
            profiles['dx1'] = H.params['dx1']

        # Gather local profiles (prof_loc) from all processors and 
        # place them in main profiles dictionary
        prof_global = comm.gather(prof_loc)
        
        print("Rank %d done analyzing range of k: (%d, %d)." 
            %(rank, k_min, k_max), flush=True)

        if (rank == 0):
            for prof in prof_global:
                for key in keys + acc_keys + cool_keys + vec_keys:
                    profiles['%s' %key] += prof['%s' %key]

            profiles['k_max'] = k_min + (k_max - k_min) * size 
            if not os.path.exists(main_dir + 'Averages/' + sim):
                os.makedirs(main_dir + 'Averages/' + sim)

            np.save(main_dir + 'Averages/' + sim + 
                    '/Profiles_%d' %fnum, profiles)
            print("Finished with slice %d of %d" %(slice_num + 1, nslices), 
                    flush=True)
            profiles['saved'] = True
            profiles['slice_num_restart'] = slice_num + 1

    print("Done with all profiles for file number %d." %fnum)

def thph_sum(q, grid, data, params, CONSTANTS, **kwargs):
    w    = kwargs.get('weight', 1.0)
    Acc2 = ana.Accretion(grid, data, params, CONSTANTS, 
            cgs=True, radiation=True)
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

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Calculate 1D profiles for single 
                        ultra-high-resolution simulation output.""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('fnum', type=int, help = """dump number""")
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
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if profile extraction
                        should be done on reduced data.""")
    parser.add_argument('--restart', action='store_true', required=False, 
                        help="""boolean set to determine if profile 
                        integration should continue from last slice_num""")
    args = parser.parse_args()
    main(**vars(args))

