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
from mpi4py.util import pkl5
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
    data_dir = cv.get_data_dir(sim, reduced, shared)
    save_dir = cv.get_save_dir(sim if title == None else title, 
                               'Averages', 'Slices', reduced=reduced)

    nzeros = len(str(int(os.popen(
        f'ls -d {data_dir}/dumps* | wc -l').read())))
        
    # Quantities (quants) for slice averaging
    keys = [
        'rho', 'Ti', 'Pmag', 'Pthm', 'Vol', 'Dtheta2'
    ]
    # Accretion rates (mass and energy) + Phi_BH
    acc_keys = [
        'Mdot', 'Edot', 'Mdot_sigma', 'Edot_sigma',
        'Edot_Poyt', 'Edot_Fluid', 'Phi_BH'
    ]
    if (two_temp and radiation):
        keys += ['Te', 'Prad', 'Eion', 'Eelc', 'rhokappa']
        acc_keys.append('Edot_Rad')
    elif (radiation and not two_temp):
        keys += ['Prad', 'rhokappa']
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

    # Vectors (4-velocity, 3-velocity, 4-magnetic field, and 
    # radiation 4-momentum)
    vec_keys = [
        'uR',   'uphi',  'uz',
        'vR',   'vphi',  'vz', 
        'bR',   'bphi',  'bz',
        'rhouR', 'rhouphi', 'rhouz'
    ]
    if (radiation):
        vec_keys += ['uR_rad', 'uphi_rad', 'uz_rad']

    # Ensure that nodes don't all attempt to make directory at same time
    timeit.sleep(np.random.rand())
    os.makedirs(save_dir, exist_ok=True)

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
        slice_keys = keys + acc_keys + cool_keys + vec_keys
    else:
        slice_keys = keys + acc_keys + vec_keys
    
    slices, nphi, iphi_stp = init_slices(data_dir, file_num, slice_keys,
        save_dir, nslices, restart, nzeros, **read_args)

    # Loop over whole data slice that fits on single node.
    # Process slivers of data slice assigned to each core in parallel.
    for slice_num, iphi in enumerate(range(0, nphi, iphi_stp)):
        if (restart):
            if slice_num < slices['slice_num_restart']:
                continue
     
        slices['saved'] = False
        k_min = iphi
        k_max = iphi + iphi_stp

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

        slice_loc['Mdot']      = ph_sum(-H.data['rho'] * H.data['uu'][1], 
                                    H.grid, H.data, H.params, hr.CONSTANTS)[0]
        slice_loc['Edot']      = ph_sum(stress_energy_MHD, 
                                    H.grid, H.data, H.params, hr.CONSTANTS)[0]
        slice_loc['Mdot_sigma']= ph_sum(-H.data['rho'] * H.data['uu'][1], 
                                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
        slice_loc['Edot_sigma']= ph_sum(stress_energy_MHD, 
                                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
        slice_loc['Edot_Poyt'] = ph_sum(stress_energy_EM, 
                                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
        slice_loc['Edot_Fluid']= ph_sum(stress_energy_HD, 
                                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
        slice_loc['Phi_BH']    = ph_sum(0.5 * np.abs(H.data['B'][1]), 
                                    H.grid, H.data, H.params, hr.CONSTANTS)[0] 
        
        if (radiation):
            slice_loc['Edot_Rad'] = ph_sum(stress_energy_RAD, H.grid, H.data, 
                H.params, hr.CONSTANTS, **acc_args)[0]
                
        del stress_energy_MHD 
        del stress_energy_HD, stress_energy_EM, stress_energy_RAD        
        
        if (radiation):
            # Evaluate scattering photosphere
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

        if (radiation):
            # Calculate opacities including all processes
            opacities = dict.fromkeys(['electron_scattering', 'bound_free', 
                'CHIANTI', 'free_free', 'hydrogen', 'molecular', 'synchrotron'])
            if (two_temp):
                Opac = rad.Opacity(opacities, **H.data, pars=H.params, 
                    CONSTANTS=hr.CONSTANTS, b2=b2, XRB=True)
            else:
                Opac = rad.Opacity(opacities, **H.data, 
                    Te=(mu_g * mp / kB) * (pgas / H.data['rho']), 
                    pars=H.params, CONSTANTS=hr.CONSTANTS, b2=b2, XRB=True)
            kappa_abs = Opac.opacities['absorption_total']
            kappa_es  = Opac.opacities['electron_scattering']
            kappa_eff = np.sqrt(kappa_abs * (kappa_abs + kappa_es))
        
        if (radiation or two_temp):
            # Need uu_t / c for cooling rates
            uu_t = np.einsum(
                'ij...,j...->i...', H.grid['gcov'], H.data['uu'] / c)[0]

        # Deallocate magnetic field in the lab frame and radiation 4-momentum
        # as well as metric tensor and inverse
        H.data['B']    = H.data['rho']
        H.grid['gcov'] = H.data['rho']
        H.grid['gcon'] = H.data['rho']
    
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
                (H.data['rho'] / (mu_e * mp)) * kB * H.data['Te']),
                kappa_eff * H.data['rho']
            ]
        elif (radiation and not two_temp):
            T_gas = pgas / H.data['rho']
            quants = [
                H.data['rho'], T_gas, b2/(8. * np.pi), pgas,
                1.0, np.square(H.grid['theta'] - np.pi/2.),
                (1./3.) * H.data['u_rad'], kappa_eff * H.data['rho']
            ]
        else:
            T_gas = pgas / H.data['rho']
            quants = [
                H.data['rho'], T_gas, b2/(8. * np.pi), pgas,
                1.0, np.square(H.grid['theta'] - np.pi/2.)
            ]

        for (key, quant) in zip(keys, quants):
            slice_loc[f'{key}'] = ph_sum(quant, 
                H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
        
        if (two_temp):
            del Theta_e, Theta_i, gam_e, gam_i, pgas
        else:
            del pgas, T_gas

        if radiation:
            del kappa_eff, kappa_abs
        else:
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
            slice_loc['qie'] = ph_sum(qie, H.grid, H.data, H.params, 
                hr.CONSTANTS, weight=uu_t, **acc_args)[0]
            if (radiation):
                slice_loc['qie_thin'] = ph_sum(qie, H.grid, H.data, H.params, 
                    hr.CONSTANTS, weight=uu_t, optically_thin = True,  **tau_args)[0]
                slice_loc['qie_thick']= ph_sum(qie, H.grid, H.data, H.params, 
                    hr.CONSTANTS, weight=uu_t, optically_thin = False, **tau_args)[0]
            del qie
        
        if (radiation):
            qIC = Radi.calc_qIC(**H.data)
            # cooling = True (maybe)
            slice_loc['qIC'] = ph_sum(qIC, H.grid, H.data, H.params, 
                hr.CONSTANTS, weight=uu_t, **acc_args)[0]
            slice_loc['qIC_thin']  = ph_sum(qIC, H.grid, H.data, H.params, 
                hr.CONSTANTS, weight=uu_t, optically_thin=True, **tau_args)[0]
            slice_loc['qIC_thick'] = ph_sum(qIC, H.grid, H.data, H.params, 
                hr.CONSTANTS, weight=uu_t, optically_thin=False, **tau_args)[0]
            del qIC

            qemi=Radi.calc_q_emi(**H.data, b2=b2, XRB=True, cpp=True)
            slice_loc['qemi'] = ph_sum(qemi, H.grid, H.data, H.params, 
                hr.CONSTANTS, weight=uu_t, **acc_args)[0]
            slice_loc['qemi_thin']  = ph_sum(qemi, H.grid, H.data, H.params, 
                hr.CONSTANTS, weight=uu_t, optically_thin=True, **tau_args)[0]
            slice_loc['qemi_thick'] = ph_sum(qemi, H.grid, H.data, H.params, 
                hr.CONSTANTS, weight=uu_t, optically_thin=False, **tau_args)[0]
            del qemi

            qabs= Radi.calc_q_abs(**H.data, b2=b2, XRB=True, cpp=True)
            slice_loc['qabs'] = ph_sum(qabs, H.grid, H.data, H.params, 
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
        uu_cyl       = Coords.int_to_cyl_vec(H.data['uu'], cpp=True)
        H.data['uu'] = H.data['rho']
        b_cyl        = Coords.int_to_cyl_vec(b, cpp=True)
        b            = H.data['rho']
        uu_rad_cyl   = Coords.int_to_cyl_vec(H.data['uu_rad'], cpp=True)
        H.data['uu_rad'] = H.data['rho']

        quants = [
            uu_cyl[1], uu_cyl[2], uu_cyl[3],
            np.nan_to_num(uu_cyl[1] / uu_cyl[0]),
            np.nan_to_num(uu_cyl[2] / uu_cyl[0]),
            np.nan_to_num(uu_cyl[3] / uu_cyl[0]),
            b_cyl[1], b_cyl[2], b_cyl[3],
            uu_cyl[1], uu_cyl[2], uu_cyl[3],
            uu_rad_cyl[1], uu_rad_cyl[2], uu_rad_cyl[3]
        ]

        weights = [
            1., 1., 1., 
            1., 1., 1., 
            1., 1., 1., 
            H.data['rho'], H.data['rho'], H.data['rho'], 
            1., 1., 1.
        ]

        for (key, quant, weight) in zip(vec_keys, quants, weights):
            slice_loc[f'{key}'] = ph_sum(quant, H.grid, H.data, 
                H.params, hr.CONSTANTS, weight=weight, **acc_args)[0]

        del uu_cyl, b_cyl, uu_rad_cyl, sigma

        if (rank == 0):
            # Put spherical radius in dictionary 
            slices['r']     = H.grid['r'][0, :, 0, 0] / Rg
            slices['theta'] = H.grid['theta'][0, 0, :, 0]
            slices['dx1']   = H.params['dx1']
            slices['dx2']   = H.params['dx2']
        
        # Gather local slices (slice_loc) from all processors and 
        # place them in main slices dictionary
        slice_global = comm.gather(slice_loc)
        
        print(f'Rank {rank:d} done analyzing range of k: ' + 
            f'({k_min_loc:d}, {k_max_loc:d}).', flush=True)

        if (rank == 0):
            for slc in slice_global:
                for key in slice_keys:
                    slices['%s' %key] += slc['%s' %key]

            slices['k_max'] = k_min + (k_max - k_min) * size 

            fnum_str = str(file_num).zfill(nzeros)
            np.save(save_dir + f'/Slices_{fnum_str}', slices)

            snum = int(slice_num + 1)
            print(f'Finished with slice {snum:d} of {nslices:d}', flush=True)

            slices['saved'] = True
            slices['slice_num_restart'] = slice_num + 1

    print(f'Done with all slices for file number {file_num}.')

def thph_sum(q, grid, data, params, CONSTANTS, **kwargs):
    w    = kwargs.get('weight', 1.0)
    Acc2 = ana.Accretion(grid, data, params, CONSTANTS, 
            cgs=True, radiation=True, MBH = 10.0)
    return (q * w * Acc2.sigma_cut(**kwargs) * Acc2.tau_cut(**kwargs) * (
        grid['gdet'] * params['dx2'] * params['dx3'])).sum(-1).sum(-1) 

def ph_sum(q, grid, data, params, CONSTANTS, **kwargs):
    w    = kwargs.get('weight', 1.0)
    Acc2 = ana.Accretion(grid, data, params, CONSTANTS, 
            cgs=True, radiation=True, MBH = 10.0)
    return (q * w * Acc2.sigma_cut(**kwargs) * Acc2.tau_cut(**kwargs) * (
        grid['gdet'] * params['dx3'])).sum(-1)

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

    # Logic to allow restarts if profile extraction is interrupted.
    if not restart:
        first_call = True
    else:
        fnum_str = str(file_num).zfill(nzro)
        slice_file = f'{save_dir}/Slices_{fnum_str}'
        slices = cv.read_restart(slice_file)
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
    parser.add_argument('--radiation', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if radiation quantities
                        should be included in profile extraction 
                        (default: True).""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
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