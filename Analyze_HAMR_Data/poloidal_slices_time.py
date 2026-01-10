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
    cooling   = kwargs.get('cooling')

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
                               'Averages', 'Slices', reduced=reduced)

    nzeros = len(str(int(os.popen(
        f'ls -d {data_dir}/dumps* | wc -l').read())))
        
    # Quantities (quants) for slice averaging
    keys = [
        'rho', 'Ti', 'Pmag', 'Pthm', 'Vol'
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
    
    vec_keys += ['rhouR', 'rhouphi', 'rhouz', 'rhou1']

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

    file_list = list(range(start_file, end_file + 1))
    comm = pkl5.Intracomm(MPI.COMM_WORLD)	
    size = comm.Get_size()
    rank = comm.Get_rank()

    file_list_local = cv.get_file_list_local(file_list, size, rank, data_dir)

    read_args = {
        'cgs_grid' : cgs,
        'cgs_data' : cgs,
        'coords1D' : True,
        'MBH'      : MBH,
        'lowres1'  : lowres1,
        'lowres2'  : lowres2,
        'lowres3'  : lowres3
    }

    slice_keys = keys + flux_keys + vec_keys

    # Outermost loop over file number (each file on single core)
    for file_num in file_list_local:
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
        
            # Calculate gas pressure
            if (two_temp):
                pgas= Fluid.calc_pgas(**H.data)
            else:
                gamma_adi = (5. / 3.)
                pgas= (gamma_adi - 1.) * H.data['u_gas']

            # Arguments passed to accretion class 
            acc_args = {
                'MBH'       : MBH,
                'cgs'       : cgs,
                'radiation' : radiation
            }

            # Cut on sigma
            sigma_cut_args = acc_args.copy()
            sigma_cut_args['sigma_cut'] = 3.
            sigma_cut_args['sigma']     = sigma

            # Energy fluxes
            # Calculate T^i_t streamlines/fluxes
            stress_energy_HD_10 = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas,
                upper_lower=True, index=[1, 0], cpp=True)
            stress_energy_HD_20 = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas,
                upper_lower=True, index=[2, 0], cpp=True)
            THD_vec    = np.zeros_like(H.data['uu'])
            THD_vec[1] = stress_energy_HD_10
            THD_vec[2] = stress_energy_HD_20
            THD_cyl    = Coords.int_to_cyl_vec(THD_vec)
            slices['THD_tR'] += ph_sum(THD_cyl[1], H.grid, H.data, 
                                       H.params, hr.CONSTANTS, **acc_args)[0]
            slices['THD_tz'] += ph_sum(THD_cyl[3], H.grid, H.data, 
                                       H.params, hr.CONSTANTS, **acc_args)[0]
            slices['THD_t1'] += ph_sum(THD_vec[1], H.grid, H.data, 
                                       H.params, hr.CONSTANTS, **acc_args)[0]
            
            del THD_cyl, THD_vec, stress_energy_HD_10, stress_energy_HD_20

            stress_energy_EM_10 = Field.calc_TEM(**H.data, **H.grid, b2=b2, b=b, 
                upper_lower=True, index=[1, 0], cpp=True)
            stress_energy_EM_20 = Field.calc_TEM(**H.data, **H.grid, b2=b2, b=b, 
                upper_lower=True, index=[2, 0], cpp=True)
            TEM_vec    = np.zeros_like(H.data['uu'])
            TEM_vec[1] = stress_energy_EM_10
            TEM_vec[2] = stress_energy_EM_20
            TEM_cyl    = Coords.int_to_cyl_vec(TEM_vec)
            slices['TEM_tR'] += ph_sum(TEM_cyl[1], H.grid, H.data, 
                                       H.params, hr.CONSTANTS, **acc_args)[0]
            slices['TEM_tz'] += ph_sum(TEM_cyl[3], H.grid, H.data, 
                                       H.params, hr.CONSTANTS, **acc_args)[0]
            slices['TEM_t1'] += ph_sum(TEM_vec[1], H.grid, H.data, H.params, 
                                       hr.CONSTANTS, **acc_args)[0]
            
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
                R_cyl    = Coords.int_to_cyl_vec(R_vec)
                # R^t_t necessary for thermal and radative diffusion timescales
                slices['R_tt'] += ph_sum(R_cyl[0], H.grid, H.data, 
                                         H.params, hr.CONSTANTS, **acc_args)[0]
                slices['R_tR'] += ph_sum(R_cyl[1], H.grid, H.data, 
                                         H.params, hr.CONSTANTS, **acc_args)[0]
                slices['R_tz'] += ph_sum(R_cyl[3], H.grid, H.data, 
                                         H.params, hr.CONSTANTS, **acc_args)[0]
                slices['R_t1'] += ph_sum(R_vec[1], H.grid, H.data, 
                                         H.params, hr.CONSTANTS, **acc_args)[0]
                
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
                    (4./3.) * u_rad * (uurad_fld[0]/c) * (uu_t_rad_fld/c) + 
                    (1./3.) * u_rad
                )
                R_vec[1] = (4./3.) * u_rad * (uurad_fld[1]/c) * (uu_t_rad_fld/c)
                R_vec[2] = (4./3.) * u_rad * (uurad_fld[2]/c) * (uu_t_rad_fld/c)
                R_cyl    = Coords.int_to_cyl_vec(R_vec)
                slices['R_tR_Fluid'] += ph_sum(R_cyl[1], H.grid, H.data, 
                                               H.params, hr.CONSTANTS, **acc_args)[0]
                slices['R_tz_Fluid'] += ph_sum(R_cyl[3], H.grid, H.data, 
                                               H.params, hr.CONSTANTS, **acc_args)[0]
                del R_cyl, R_vec, u_rad, uurad_fld, uu_t_rad_fld

            # Opacities
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

            uu_cov = ana.Tools.contract_vector(
                H.data['uu'], H.grid['gcov'], cpp=True)
            b_cov  = ana.Tools.contract_vector(b, H.grid['gcov'], cpp=True)
            
            # Bernoulli parameter for outflows
            # Numerator of Bernoulli parameter
            w = (H.data['rho'] * c * c + pgas + H.data['u_gas'] + b2 / (4.*np.pi)) 
            slices['Be'] += ph_sum(w * uu_cov[0], H.grid, H.data, 
                                   H.params, hr.CONSTANTS, **acc_args)[0] 
            if (radiation):
                w = (H.data['rho'] * c * c + pgas + H.data['u_gas'] 
                     + b2 / (4.*np.pi) + (4./3.) * H.data['u_rad']) 
                slices['Be_Rad'] += ph_sum(w * uu_cov[0], H.grid, H.data, 
                                           H.params, hr.CONSTANTS, **acc_args)[0]
            
            if (cooling and (radiation or two_temp)):
                # Need uu_t / c for cooling rates
                uu_t = ana.Tools.contract_vector(
                    H.data['uu'] / c, H.grid['gcov'], cpp=True)[0]
            
            # Deallocate magnetic field in the lab frame and radiation 4-momentum
            # as well as metric tensor and inverse
            H.data['B']    = H.data['rho']
            H.grid['gcov'] = H.data['rho']
            H.grid['gcon'] = H.data['rho']
    
            # Magnetic and Thermodynamic Structure
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
                    1.0, H.data['Te'], (1./3.) * H.data['u_rad'],
                    (1./(gam_i-1.)) * (H.data['rho'] / (mu_i * mp)) * kB * H.data['Ti'],
                    (1./(gam_e-1.)) * (H.data['rho'] / (mu_e * mp)) * kB * H.data['Te']
                ]
            elif (radiation and not two_temp):
                T_gas = pgas / H.data['rho']
                quants = [
                    H.data['rho'], T_gas, b2/(8. * np.pi), pgas,
                    1.0, (1./3.) * H.data['u_rad']
                ]
            elif (two_temp and not radiation):
                quants = [
                    H.data['rho'], H.data['Ti'], b2/(8. * np.pi), pgas,
                    1.0, H.data['Te'],
                    (1./(gam_i-1.)) * (H.data['rho'] / (mu_i * mp)) * kB * H.data['Ti'],
                    (1./(gam_e-1.)) * (H.data['rho'] / (mu_e * mp)) * kB * H.data['Te']
                ]
            else:
                T_gas = pgas / H.data['rho']
                quants = [
                    H.data['rho'], T_gas, b2/(8. * np.pi), pgas, 1.0
                ]

            if (radiation and two_temp):
                for (key, quant) in zip(keys[:-6], quants):
                    slices[f'{key}'] += ph_sum(quant, H.grid, H.data, 
                                               H.params, hr.CONSTANTS, **acc_args)[0]
            elif (radiation and not two_temp):
                for (key, quant) in zip(keys[:-5], quants):
                    slices[f'{key}'] += ph_sum(quant, H.grid, H.data, 
                                               H.params, hr.CONSTANTS, **acc_args)[0]
            elif (two_temp and not radiation):
                for (key, quant) in zip(keys[:-2], quants):
                    slices[f'{key}'] += ph_sum(quant, H.grid, H.data, 
                                               H.params, hr.CONSTANTS, **acc_args)[0]
            else:
                for (key, quant) in zip(keys[:-1], quants):
                    slices[f'{key}'] += ph_sum(quant, H.grid, H.data, 
                                               H.params, hr.CONSTANTS, **acc_args)[0]

            # Opacities
            if (radiation):
                slices['rhokappa'] += ph_sum(kappa_eff * H.data['rho'], H.grid, H.data, 
                                            H.params, hr.CONSTANTS, **sigma_cut_args)[0]

            # Coulomb collisional and radiative cooling
            if (cooling and two_temp):
                qie  = Fluid.calc_qie( **H.data, cpp=True)
                slices['qie']  += ph_sum(qie,  H.grid, H.data, H.params, 
                                         hr.CONSTANTS, weight=uu_t, **acc_args)[0]
            if (cooling and radiation):
                qIC  = Radi.calc_qIC(  **H.data)
                qemi = Radi.calc_q_emi(**H.data, b2=b2, XRB=True, cpp=True) 
                qabs = Radi.calc_q_abs(**H.data, b2=b2, XRB=True, cpp=True)
                slices['qIC']  += ph_sum(qIC,  H.grid, H.data, H.params, 
                                         hr.CONSTANTS, weight=uu_t, **acc_args)[0]
                slices['qemi'] += ph_sum(qemi, H.grid, H.data, H.params, 
                                         hr.CONSTANTS, weight=uu_t, **acc_args)[0]
                slices['qabs'] += ph_sum(qabs, H.grid, H.data, H.params, 
                                         hr.CONSTANTS, weight=uu_t, **acc_args)[0]

            # phi integral over gdet including sigma cut
            slices['Vol_sigma_cut'] += ph_sum(1.0, H.grid, H.data, H.params, 
                                              hr.CONSTANTS, **sigma_cut_args)[0]

            # Averages required for stress profiles
            if (radiation):
                w = (H.data['rho'] * c * c + pgas + H.data['u_gas'] 
                     + b2 / (4.*np.pi) + (4./3.) * H.data['u_rad']) 
            else:
                w = (H.data['rho'] * c * c + pgas + H.data['u_gas'] + b2) 

            if (two_temp): del Theta_e, Theta_i, gam_e, gam_i, pgas
            else:          del pgas, T_gas

            if (radiation): del kappa_eff, kappa_abs
            else:           del b2

            # Deallocate temperatures and energy densities 
            # in preparation for large transformation matrices
            if (two_temp):
                H.data['Ti'] = H.data['rho']
                H.data['Te'] = H.data['rho']
            if (radiation):
                H.data['uu_rad'] = H.data['rho']

            # Compute theta and phi averages of vector quantities
            uu_cyl     = Coords.int_to_cyl_vec(H.data['uu'], cpp=True)
            uu_sph     = Coords.int_to_ortho_sph_vec(H.data['uu'], cpp=True)
            uu_cov_cyl = Coords.int_to_cyl_one_form(uu_cov, cpp=True)
            b_cyl      = Coords.int_to_cyl_vec(b, cpp=True)
            b_sph      = Coords.int_to_ortho_sph_vec(b, cpp=True)
            b_cov_cyl  = Coords.int_to_cyl_one_form(b_cov, cpp=True)

            quants = [
                uu_cyl[1],     uu_cyl[2],     uu_cyl[3],
                w * uu_cyl[1], w * uu_cyl[2], w * uu_cyl[3], 
                w * uu_cov_cyl[2], w * uu_sph[1], w * uu_sph[2],
                np.nan_to_num(uu_cyl[1] / uu_cyl[0]),
                np.nan_to_num(uu_cyl[2] / uu_cyl[0]),
                np.nan_to_num(uu_cyl[3] / uu_cyl[0]),
                b_cyl[1], b_cyl[2], b_cyl[3], b_cov_cyl[2], b_sph[1], b_sph[2]
            ]

            # Mass fluxes
            quants += [
                H.data['rho'] * uu_cyl[1], 
                H.data['rho'] * uu_cyl[2], 
                H.data['rho'] * uu_cyl[3],
                H.data['rho'] * H.data['uu'][1]
            ]

            for i, (key, quant) in enumerate(zip(vec_keys, quants)):
                if (i < 18):
                    slices[f'{key}'] += ph_sum(quant, H.grid, H.data, 
                        H.params, hr.CONSTANTS, **acc_args)[0]
                else:
                    slices[f'{key}'] += ph_sum(quant, H.grid, H.data, 
                        H.params, hr.CONSTANTS, **sigma_cut_args)[0]

            del uu_cyl, b_cyl, uu_sph, b_sph, uu_cov_cyl, b_cov_cyl, sigma

            # Put spherical radius in dictionary 
            slices['r']     = H.grid['r'][0, :, 0, 0] / Rg
            slices['theta'] = H.grid['theta'][0, 0, :, 0]
            slices['phi']   = H.grid['phi'][0, 0, 0, :]
            slices['dx1']   = H.params['dx1']
            slices['dx2']   = H.params['dx2']
            slices['dx3']   = H.params['dx3']
        
            # Density scale
            slices['rho0'] = H.params['density_scale']
            slices['k_max'] = k_min + (k_max - k_min)

            print(f'Rank {rank:d} done analyzing range of k: ' + 
                f'({k_min:d}, {k_max:d}).', flush=True)
       
            snum = int(slice_num + 1)
            print(f'Finished with slice {snum:d} of {nslices} ' +
                f'for file {file_num}', flush=True) 

        slices['slice_num_restart'] = slice_num + 1
        slices['saved'] = True
        slices['done']  = True
        fnum_str = str(file_num).zfill(nzeros)
        np.savez(save_dir + f'/Slices_{fnum_str}', **slices)

        print(f'Done with all slices for file number {file_num}.',flush=True) 

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

def ph_sum(q, grid, data, params, CONSTANTS, **kwargs):
    w    = kwargs.get('weight', np.ones_like(data['rho']))
    acc2_args = {
        'cgs'       : kwargs.get('cgs',       False),
        'radiation' : kwargs.get('radiation', False),
        'MBH'       : kwargs.get('MBH',       1.)
    }
    Acc2 = ana.Accretion(grid, data, params, CONSTANTS, **acc2_args)
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
                        description="""Calculate 2D slices for all 
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
                        should be included in slice extraction.""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if two-temperature 
                        physics should be included in slice extraction.""")   
    parser.add_argument('--cooling', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if cooling rates
                        (qie, qIC, qemi) should be included.""")   
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