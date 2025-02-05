import sys
import os
import argparse
import numpy as np
sys.path.insert(0, '/lustre/orion/ast178/proj-shared/' + 
        'cbambic/HAMR_Visualization/Read_HAMR_Data')
import hamr_read as hr
import analysis as ana
import tracemalloc

def main(**kwargs):
    fnum    = int(kwargs.get('fnum'))
    nslices = int(kwargs.get('nslices'))
    restart = kwargs.get('restart')
    
    main_dir = '/lustre/orion/ast178/proj-shared/cbambic/'
    sim      = 'Thin_RAD_MAD_HiRes'
    profiles = {}

    H = hr.Read(main_dir + 'DATA/' + sim, 
	    cgs_data = True, cgs_grid = True, MBH = 10.0)
    H.read_params(fnum)
    profiles['t'] = hr.params['t']
    profiles['a'] = hr.params['a']

    nz = int(hr.params['nb3'] * hr.params['bs3'] * H.upscalex3)
    phi_1D = np.zeros(nz)
    for k in range(0, nz):
        phi_1D[k] = (hr.params['startx3'] + (k + 0.5) * hr.params['dx3'])

    dphi = phi_1D[1] - phi_1D[0]

    if (len(phi_1D) % nslices != 0):
        raise ValueError(
                "nslices must evenly divide len(phi_1D) = %d" %len(phi_1D))
    
    iphi_stp = int(len(phi_1D)/nslices)

    if not restart:
        iphi_start = 0
        first_call = True
    else:
        profile_file = (main_dir + 'Averages/' + sim + 
                '/Profiles_%d.npy' %fnum)
        profiles, iphi_start = read_restart(profile_file, iphi_stp)
        first_call = False

    for i in range(iphi_start, len(phi_1D), iphi_stp):
        plasma = {}
        profiles['saved'] = False
        tracemalloc.start()
        H = hr.Read(main_dir + 'DATA/Thin_RAD_MAD_HiRes', 
	        cgs_data = True, cgs_grid = True, MBH = 10.0,
                do_box = 1,
                phi_min = phi_1D[i] - dphi, 
                phi_max = phi_1D[i + iphi_stp - 1])
        
        # Define constants
        MBH = 10.0
        Msun = hr.CONSTANTS['Msun']
        c    = hr.CONSTANTS['c']
        G    = hr.CONSTANTS['G']
        Rg   = G*MBH*Msun/(c*c)
       
        # Read grid and data
        H.read_grid(fnum, direct=True)
        print("Grid successfully read in", flush=True)
        H.read_prims(fnum, skip_read_params=True)
        print("Data successfully read in.", flush=True)
        k_min, k_max = H.block_info['k_min'], H.block_info['k_max']
        print("Analyzing k range %d to %d" %(k_min, k_max))
        
        # Initialize Accretion class
        Acc = ana.Accretion(H.grid, H.data, hr.params, hr.CONSTANTS, cgs=True)
        # Initialize Coordinates class
        Coords = ana.Coordinates(H.grid['r'], H.grid['theta'], H.grid['phi'])
        # Initialize Fluid class
        Fluid = ana.Fluid(hr.params, hr.CONSTANTS, cgs=True)
        # Initialize Field class
        Field = ana.Field(hr.params, hr.CONSTANTS, cgs=True)

        # Calculate sigma (3D)
        b     = Field.calc_b(H.data['B'], H.data['uu'], H.grid['gcov'])
        b2    = Field.calc_b2(b, H.data['B'], H.grid['gcov'])
        sigma = (b2 / (8. * np.pi)) / (H.data['rho'] * c * c)
        # Calculate gas pressure
        pgas  = Fluid.calc_pgas(H.data['u_gas'], H.data['Te'], H.data['Ti'])

        # Put spherical radius in dictionary 
        profiles['r']   = H.grid['r'][0, :, 0, 0] / Rg
        plasma['r']     = H.grid['r'][0, :, 0, 0]
        plasma['theta'] = H.grid['theta'][0, 0, :, 0]

        keys   = ['rho', 'Ti', 'Te', 'Pmag', 'Pthm', 'Prad', 'Vol']
        quants = [H.data['rho'], H.data['Ti'], H.data['Te'], 
                  b2/(8. * np.pi), pgas, (1./3.) * H.data['u_rad'], 1.0
        ]
        if first_call:
            profiles['Mdot']   = Acc.calc_Mdot(sigma_cut = 3.0, 
                                               sigma = sigma)[0]
            profiles['Edot']   = Acc.calc_Edot(sigma_cut = 3.0, 
                                               sigma = sigma, 
                                               b2 = b2, b = b)[0]
            profiles['Phi_BH'] = Acc.calc_PhiBH()[0]
            for (key, quant) in zip(keys, quants):
                profiles['%s' %key] = thph_sum(quant, 
                        H.grid, H.data, hr.params, hr.CONSTANTS, 
                        weight=H.data['rho'], sigma_cut=3.0, sigma=sigma)[0]
            first_call = False
        else:
            profiles['Mdot'] += Acc.calc_Mdot(sigma_cut = 3.0, 
                                              sigma = sigma)[0]
            profiles['Edot'] += Acc.calc_Edot(sigma_cut = 3.0, 
                                              sigma = sigma, 
                                              b2 = b2, b = b)[0]
            profiles['Phi_BH'] += Acc.calc_PhiBH()[0]
            for (key, quant) in zip(keys, quants):
                profiles['%s' %key]+= thph_sum(quant, 
                        H.grid, H.data, hr.params, hr.CONSTANTS, 
                        weight=H.data['rho'], sigma_cut=3.0, sigma=sigma)[0]
            profiles['k_max'] = k_max
        
        plasma['Ti']    = H.data['Ti'][0, :, :, 0]
        plasma['Te']    = H.data['Te'][0, :, :, 0]
        plasma['BetaI'] = ((b2 / (8. * np.pi)) / pgas)[0, :, :, 0]
        plasma['sigma'] = sigma[0, :, :, 0]
        
        # Calculate magnetic field in Cartesian coordinates
        B_cart = Coords.int_to_cart_vec(H.data['B'])
        plasma['B'] = B_cart[:, 0, :, :, 0]

        print(tracemalloc.get_traced_memory())
        tracemalloc.stop()

        print("Saving profiles ...")
        if not os.path.exists(main_dir + 'Averages/' + sim):
            os.makedirs(main_dir + 'Averages/' + sim)

        profiles['saved'] = True
        np.save(main_dir + 'Averages/' + sim + 
                '/Profiles_%d' %fnum, profiles)
        np.save(main_dir + 'Averages/' + sim + 
                '/Plasma_%d_%d' %(fnum, k_min), plasma)
        print("Done saving profiles.")

    print("Profiles done.")

def thph_sum(q, grid, data, params, CONSTANTS, **kwargs):
    w    = kwargs.get('weight')
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

