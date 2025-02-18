import sys
import os
import argparse
import numpy as np
sys.path.insert(0, '/lustre/orion/ast178/proj-shared/' + 
        'cbambic/HAMR_Visualization/Read_HAMR_Data')
import hamr_read as hr
import analysis as ana
import radiation as rad
import tracemalloc
import time as get_time

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
        profiles['saved'] = False
        tracemalloc.start()
        t0 = get_time.time()
        H = hr.Read(main_dir + 'DATA/Thin_RAD_MAD_HiRes', 
	        cgs_data = True, cgs_grid = True, MBH = 10.0,
                coords1D = True,
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
        H.read_prims(fnum, skip_read_params=True)
        k_min, k_max = H.block_info['k_min'], H.block_info['k_max']
        print("Analyzing k range %d to %d" %(k_min, k_max))
        t1 = get_time.time()
        print("Reading in data took %.4e s" %(t1 - t0), flush=True)
        
        # Initialize Accretion class
        Acc = ana.Accretion(H.grid, H.data, hr.params, hr.CONSTANTS, 
                cgs=True, MBH=10.0, radiation=True)
        # Initialize Coordinates class
        Coords = ana.Coordinates(H.grid['r'], H.grid['theta'], H.grid['phi'])
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
        # Calculate gas pressure
        pgas  = Fluid.calc_pgas(H.data['u_gas'], H.data['Te'], H.data['Ti'])
        
        # Put spherical radius in dictionary 
        profiles['r']   = H.grid['r'][0, :, 0, 0] / Rg
        
        # Quantities (quants) for profile averaging
        keys   = [
            'rho',  'Ti',   'Te', 
            'Pmag', 'Pthm', 'Prad', 
            'vr',   'vth',  'vphi', 
            'br',   'bth',  'bphi', 'Vol' 
        ]
        # Timescales for profile averaging
        tkeys   = ['teq', 'tIC', 'trad_abs', 'trad_emi']
         
        if first_call:
            profiles['Mdot']       = np.zeros(len(profiles['r']))
            profiles['Edot']       = np.zeros(len(profiles['r']))
            profiles['Mdot_sigma'] = np.zeros(len(profiles['r']))
            profiles['Edot_sigma'] = np.zeros(len(profiles['r']))
            profiles['Phi_BH']     = np.zeros(len(profiles['r']))
            for key in keys:
                profiles['%s' %key]= np.zeros(len(profiles['r']))
            for tkey in tkeys:
                profiles['%s' %tkey]     = np.zeros(len(profiles['r']))
                profiles['Vol_%s' %tkey] = np.zeros(len(profiles['r'])) 
            first_call = False

        t0 = get_time.time()
        
        # Mass and energy accretion rates. Radial magnetic flux (Phi_BH).
        stress_energy_MHD = Acc.calc_TMHD(upper_lower=True, index = [1, 0], 
                radiation=True, b2 = b2, b = b) #bb_tnsr = bb_tnsr, uu_tnsr = uu_tnsr)
        profiles['Mdot']      += Acc.calc_Mdot()[0]
        profiles['Edot']      += Acc.calc_Edot(TMHD = stress_energy_MHD)[0]
        profiles['Mdot_sigma']+= Acc.calc_Mdot(sigma_cut = 3.0, 
                sigma = sigma)[0]
        profiles['Edot_sigma']+= Acc.calc_Edot(TMHD = stress_energy_MHD,
                sigma_cut = 3.0, sigma = sigma)[0]
        profiles['Phi_BH']    += Acc.calc_PhiBH()[0] 
        del stress_energy_MHD        
        
        t1 = get_time.time()
        print("Calculating Mdot, Edot, Phi_BH took %.4e s" %(t1 - t0), 
                flush=True)
        
        t0 = get_time.time()
        
        # Profiles of primitive variables and pressures
        quants = [
            H.data['rho'], H.data['Ti'], H.data['Te'], 
            b2/(8. * np.pi), pgas, (1./3.) * H.data['u_rad'], 
            H.data['uu'][1] / H.data['uu'][0], 
            H.data['uu'][2] / H.data['uu'][0], 
            H.data['uu'][3] / H.data['uu'][0], 
            b[1], b[2], b[3], 1.0 
        ]
        
        for (key, quant) in zip(keys, quants):
            profiles['%s' %key] += thph_sum(quant, 
                    H.grid, H.data, hr.params, hr.CONSTANTS, 
                    weight=H.data['rho'], 
                    sigma_cut=3.0, sigma=sigma)[0]
        
        # Save on memory before computing cooling timescales
        del quants
        del pgas
        del b
        del_keys = ['uu', 'B', 'uu_rad']
        for key in del_keys:
            H.data['%s' %key] = H.data['rho']
        H.grid['gcov'] = H.grid['gdet']
        H.grid['gcon'] = H.grid['gdet']
        
        t1 = get_time.time()
        print("Calculating profile averages took %.4e s" %(t1 - t0), 
                flush=True)
        
        t0 = get_time.time()
        
        # Profiles of cooling timescales in plasma
        timescales = [
            Fluid.calc_teq(H.data['rho'], H.data['Te'], H.data['Ti'], 
                cooling=True), 
            Radi.calc_tIC(H.data['rho'], H.data['Te'], H.data['u_rad'], 
                cooling=True),
            Radi.calc_trad_abs(H.data['rho'], H.data['Te'], 
                H.data['u_rad'], b2=b2, XRB=True),
            Radi.calc_trad_emi(H.data['rho'], H.data['Te'], 
                H.data['u_rad'], b2=b2, XRB=True)
        ]
        
        ones = np.ones(H.data['rho'].shape)

        for (tkey, timescale) in zip(tkeys, timescales):
            profiles['%s' %tkey] += thph_sum(timescale, 
                    H.grid, H.data, hr.params, hr.CONSTANTS, 
                    weight=H.data['rho'], 
                    sigma_cut=3.0, sigma=sigma)[0]
            profiles['Vol_%s' %tkey] += thph_sum(ones, H.grid, H.data, 
                    hr.params, hr.CONSTANTS, weight=H.data['rho'], 
                    sigma_cut=3.0, sigma=sigma, time=timescale)[0]

        t1 = get_time.time()
        print("Timescales took %.4e s" %(t1 - t0), flush=True)
        
        profiles['k_max'] = k_max
        
        print(tracemalloc.get_traced_memory())
        tracemalloc.stop()

        print("Saving profiles ...")
        if not os.path.exists(main_dir + 'Averages/' + sim):
            os.makedirs(main_dir + 'Averages/' + sim)

        profiles['saved'] = True
        np.save(main_dir + 'Averages/' + sim + 
                '/Profiles_%d' %fnum, profiles)
        print("Done saving profiles.")

    print("Profiles done.")

def thph_sum(q, grid, data, params, CONSTANTS, **kwargs):
    w    = kwargs.get('weight')
    if 'time' in kwargs:
        time = kwargs.get('time')
        q[time <= 0.0] = 0.0
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


'''
        #plasma['r']     = H.grid['r'][0, :, 0, 0]
        #plasma['theta'] = H.grid['theta'][0, 0, :, 0]
        #plasma['Ti']    = H.data['Ti'][0, :, :, 0]
        #plasma['Te']    = H.data['Te'][0, :, :, 0]
        #plasma['BetaI'] = ((b2 / (8. * np.pi)) / pgas)[0, :, :, 0]
        #plasma['sigma'] = sigma[0, :, :, 0]
        
        # Calculate magnetic field in Cartesian coordinates
        #B_cart = Coords.int_to_cart_vec(H.data['B'])
        #plasma['B'] = B_cart[:, 0, :, :, 0]
        #np.save(main_dir + 'Averages/' + sim + 
        #        '/Plasma_%d_%d' %(fnum, k_min), plasma)
'''
