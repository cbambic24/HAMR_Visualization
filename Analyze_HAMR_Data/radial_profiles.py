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
import block_budget as bb
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
    sim        = kwargs.get('sim')
    file_num   = int(kwargs.get('file_num'))
    end_file   = int(kwargs.get('end_file'))
    nslices    = kwargs.get('nslices')
    lowres1    = kwargs.get('lowres1')
    lowres2    = kwargs.get('lowres2')
    lowres3    = kwargs.get('lowres3')
    stride     = kwargs.get('stride')
    reduced    = kwargs.get('reduced')
    restart    = kwargs.get('restart')
    overwrite  = kwargs.get('overwrite')
    shared     = kwargs.get('shared')
    title      = kwargs.get('title')
    radiation  = kwargs.get('radiation')
    two_temp   = kwargs.get('two_temp')
    vargamma   = kwargs.get('vargamma')
    MBH        = np.float32(kwargs.get('MBH'))
    by_block   = kwargs.get('by_block')
    blocks_per_batch = kwargs.get('blocks_per_batch')

    single_file = True if end_file == -1 else False

    # Define constants
    cgs = True if (two_temp or radiation) else False
    if (two_temp or radiation):
        Msun = np.float32(hr.CONSTANTS['Msun'])
        c    = np.float32(hr.CONSTANTS['c'])
        G    = np.float32(hr.CONSTANTS['G'])
        Rg   = G * MBH * Msun / (c * c)
    else:
        c, G, Rg = ONE, ONE, ONE
    
    if cgs:
        mp   = np.float32(hr.CONSTANTS['mp'])
        x_AB = np.float32(hr.CONSTANTS['X_AB'])
        y_AB = np.float32(hr.CONSTANTS['Y_AB'])

        # Equation 11 in Sadowski+ (2017)
        mu_i = np.float32(4. / (4. * x_AB + y_AB))
        # Equation 10 in Sadowski+ (2017)
        mu_e = np.float32(2. / (1. + x_AB))
        # Given in Kaaz et al. (2024)
        mu_g = np.float32(4. / (6. * x_AB + y_AB + 2.))

        physics_args = {
            'MBH'  : MBH,
            'Rg'   : Rg,
            'c'    : c,
            'kB'   : np.float32(hr.CONSTANTS['kB']),
            'mp'   : mp,
            'x_AB' : x_AB,
            'y_AB' : y_AB,
            'mu_i' : mu_i,
            'mu_e' : mu_e,
            'mi'   : (x_AB + y_AB) * mp,
            'me'   : np.float32(hr.CONSTANTS['me']),
            'mu_g' : mu_g
        }
    else:
        physics_args = {
            'MBH' : MBH,
            'Rg'  : Rg,
            'c'   : c
        }
    physics_args['vargamma'] = vargamma

    # Ensure CONSTANTS are 32-bit floats
    CONSTANTS = {key : np.float32(value) for (key, value) in hr.CONSTANTS.items()}
    
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
    if by_block:
        # Phi_max requires full theta range so not possible by_block.
        phi_keys = [k for k in phi_keys if not k.startswith('Phi_max')]
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

    init_args = {
        'data_dir' : data_dir,
        'save_dir' : save_dir,
        'nslices'  : nslices,
        'restart'  : restart,
        'nzro'     : nzeros,
        'two_temp' : two_temp,
        'radiation': radiation,
        'by_block' : by_block,
        'blk_stp'  : blocks_per_batch
    }
 
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    MPI_info = {
        'comm' : comm,
        'size' : size,
        'rank' : rank
    }

    # Size the block batches against a real memory budget. 
    if by_block and blocks_per_batch < 0:
        def probe_batch(n_blocks):
            print(f"[CALIBRATION] probe {n_blocks} blocks", flush=True)
            process_batch(data_dir=data_dir, file_num=file_num,
                          read_args=read_args, CONSTANTS=CONSTANTS, cgs=cgs,
                          two_temp=two_temp, radiation=radiation,
                          keys=keys, vec_keys=vec_keys,
                          start_block=0, end_block=n_blocks, **physics_args)

        budget_info = bb.plan_block_batches(probe_batch, comm)
        blocks_per_batch     = budget_info['n_batch']
        init_args['blk_stp'] = blocks_per_batch
        if rank == 0:
            print(bb.summarize(budget_info), flush=True)
            if budget_info.get('fallback'):
                print(f"NOTE: {budget_info['fallback']}", flush=True)
            print("[CALIBRATION] done.", flush=True)
    elif by_block and rank == 0:
        print(f"Using --blocks_per_batch={blocks_per_batch} "
              f"(calibration overridden).", flush=True)

    if single_file:
        init_args['file_num'] = file_num

        profiles, nphi, iphi_stp = cv.init_profiles(keys=profile_keys,
                                                    **init_args, **read_args)
        if not restart:
            profiles['done'] = False

        run_args = {
            'profiles'     : profiles,
            'MPI_info'     : MPI_info,
            'read_args'    : read_args,
            'profile_keys' : profile_keys,
            'keys'         : keys,
            'vec_keys'     : vec_keys,
            'CONSTANTS'    : CONSTANTS,
            'cgs'          : cgs
        }

        if by_block:
            blk_args = {k: v for k, v in init_args.items()
                        if k not in ('nslices', 'blk_stp')}
            run_parallel_by_block(**run_args, nb=int(profiles['nb']),
                                  blk_stp=blocks_per_batch,
                                  **blk_args, **physics_args)
        else:
            run_parallel_in_phi(**run_args, nphi=nphi, iphi_stp=iphi_stp,
                                **init_args, **physics_args)
    
    else:
        file_list = list(range(file_num, end_file + 1))
        file_list_args = {
            'file_list' : file_list,
            'size'      : size,
            'rank'      : rank,
            'data_dir'  : data_dir,
            'stride'    : stride
        }

        file_list_local = cv.get_file_list_local(**file_list_args)
        if not overwrite:
            sim_title = sim if title == None else title
            averages_dir = cv.user_path + f'/Averages/{sim_title}'
            # Use get_file_list_local as if on rank 0 with 1 core
            existing_profiles = cv.get_file_list_local(file_list_local, 1, 0,
                averages_dir, file_type='Profiles', reduced=reduced)
            existing_set = set(existing_profiles)
            file_list_local = [f for f in file_list_local if f not in existing_set]
        
        run_args = {
            'file_list_local' : file_list_local,
            'read_args'       : read_args,
            'init_args'       : init_args,
            'MPI_info'        : MPI_info,
            'profile_keys'    : profile_keys,
            'keys'            : keys,
            'vec_keys'        : vec_keys,
            'CONSTANTS'       : CONSTANTS,
            'cgs'             : cgs
        }

        run_parallel_in_time(**run_args, **physics_args)

def run_parallel_by_block(profiles, nb, blk_stp, MPI_info, read_args,
                          data_dir, file_num, profile_keys, keys, vec_keys,
                          save_dir, CONSTANTS, cgs, restart, nzro,
                          two_temp, radiation, **kwargs):
    """Iterate through data in batches based on calibrated block step."""
    rank, size = MPI_info['rank'], MPI_info['size']
    nr         = len(profiles[profile_keys[0]])

    start_at = cv.resume_offset(profiles, True) if restart else 0
    nbatch   = len(range(start_at, nb, blk_stp))

    if start_at >= nb:
        if (rank == 0):
            print(f'File {file_num} already complete through block '
                  f'{start_at:d} of {nb:d}; nothing to do.', flush=True)
        return

    for batch_num, batch_sel in cv.iter_block_batches(nb, blk_stp,
                                                      start=start_at):
        profiles['saved'] = False
        rank_sel = cv.split_batch(batch_sel, rank, size)

        # Staggered start to take pressure off of filesystem
        timeit.sleep(0.05 * rank + np.random.rand())

        local_buffer = np.zeros((len(profile_keys), nr), dtype=np.float64)
        meta = None
        if rank_sel['end_block'] > rank_sel['start_block']:
            batch_args = {
                'data_dir' : data_dir,
                'file_num' : file_num,
                'read_args': read_args,
                'CONSTANTS': CONSTANTS,
                'cgs'      : cgs,
                'two_temp' : two_temp,
                'radiation': radiation,
                'keys'     : keys,
                'vec_keys' : vec_keys,
                'nr_out'   : nr
            }
            print(f"[EXTRACTION] rank {rank:d} batch {batch_num + 1:d}/"
                  f"{nbatch:d} reading blocks "
                  f"[{rank_sel['start_block']:d}, {rank_sel['end_block']:d}) "
                  f"of {nb:d}", flush=True)
            prof_loc, meta = process_batch(**batch_args, **rank_sel, **kwargs)
            for i, key in enumerate(profile_keys):
                if key in prof_loc:
                    local_buffer[i, :] = prof_loc[key][:nr]
        else:
            print(f"[EXTRACTION] rank {rank:d} batch {batch_num + 1:d}/"
                  f"{nbatch:d} has no blocks; contributing zeros.", flush=True)

        global_buffer = None
        print(f"Rank {rank:d} done analyzing blocks: "
              f"({rank_sel['start_block']:d}, {rank_sel['end_block']:d}).", flush=True)

        if (rank == 0):
            profiles['r']    = meta['r']
            profiles['dx1']  = meta['dx1']
            profiles['rho0'] = meta['rho0']
            global_buffer = np.zeros_like(local_buffer)

        MPI_info['comm'].Reduce(local_buffer, global_buffer, op=MPI.SUM, root=0)

        if (rank == 0):
            for i, key in enumerate(profile_keys):
                profiles[key] += global_buffer[i, :]

            cv.record_progress(profiles, batch_sel, True, nb,
                               blk_stp=blk_stp)
            profiles['slice_num_restart'] = batch_num + 1
            cv.save_checkpoint(save_dir, file_num, nzro, profiles)

            print(f'Finished with batch {batch_num + 1:d} of {nbatch:d} '
                  f'for file number {file_num}', flush=True)
        MPI_info['comm'].Barrier()
        gc.collect()

    print(f'Done with all profiles for file number {file_num}.', flush=True)

def run_parallel_in_time(file_list_local, read_args, init_args, MPI_info,
                         profile_keys, keys, vec_keys, CONSTANTS, cgs, **kwargs):
    data_dir = init_args['data_dir']
    save_dir = init_args['save_dir']
    for file_num in file_list_local:
        init_args['file_num'] = file_num
        profiles, nphi, iphi_stp = cv.init_profiles(keys=profile_keys,
                                                    **init_args, **read_args)
        if not init_args['restart']:
            profiles['done'] = False

        batch_args = {
            'data_dir' : data_dir,
            'file_num' : file_num,
            'read_args': read_args,
            'CONSTANTS': CONSTANTS,
            'cgs'      : cgs,
            'two_temp' : init_args['two_temp'],
            'radiation': init_args['radiation'],
            'keys'     : keys,
            'vec_keys' : vec_keys
        }

        by_block = init_args.get('by_block', False)
        nb       = int(profiles['nb'])
        blk_stp  = init_args.get('blk_stp') or nb    # None/-1 -> one batch
        nbatch   = len(range(0, nb, blk_stp)) if by_block else init_args['nslices']

        restart  = init_args['restart']
        start_at = cv.resume_offset(profiles, by_block) if restart else 0
        limit    = nb if by_block else nphi
        if start_at >= limit:
            print(f'File {file_num} already complete; skipping.', flush=True)
            continue

        batches = (cv.iter_block_batches(nb, blk_stp, start=start_at)
                   if by_block else
                   cv.iter_phi_batches(nphi, iphi_stp, start=start_at))
        for batch_num, sel in batches:
            profiles['saved'] = False

            # Staggered start to take pressure off of filesystem
            timeit.sleep(0.05 * MPI_info['rank'] + np.random.rand())

            batch_args.update(sel)

            lo, hi = ((sel['start_block'], sel['end_block']) if by_block
                      else (sel['k_min'], sel['k_max']))
            print(f"[EXTRACTION] rank {MPI_info['rank']:d} file {file_num} "
                  f"batch {batch_num + 1:d}/{nbatch} reading "
                  f"{'blocks' if by_block else 'k'} [{lo:d}, {hi:d})",
                  flush=True)
            prof_loc, meta = process_batch(**batch_args, **kwargs)

            for key in profile_keys:
                if key in prof_loc:
                    profiles[key] += prof_loc[key][:len(profiles[key])]

            profiles['rho0']   = meta['rho0']
            profiles['r']      = meta['r']
            profiles['dx1']    = meta['dx1']
            cv.record_progress(profiles, sel, by_block, nb if by_block else nphi,
                               blk_stp=blk_stp if by_block else None)

            lo, hi = (sel['start_block'], sel['end_block']) if by_block \
                     else (sel['k_min'], sel['k_max'])
            unit = 'blocks' if by_block else 'k'
            print(f"Rank {MPI_info['rank']:d} done analyzing range of "
                  f"{unit}: ({lo:d}, {hi:d}).", flush=True)

            snum = int(batch_num + 1)
            print(f"Finished with batch {snum:d} of {nbatch} "
                f"for file {file_num}", flush=True)

        profiles['slice_num_restart'] = batch_num + 1   # legacy, not read
        cv.save_checkpoint(save_dir, file_num, init_args['nzro'], profiles)
        gc.collect()
        print(f"Done with all profiles for file number {file_num}.", flush=True)

def run_parallel_in_phi(profiles, nphi, iphi_stp, MPI_info, read_args,
                        data_dir, file_num, profile_keys, keys, vec_keys,
                        save_dir, CONSTANTS, cgs, nslices, restart, nzro,
                        two_temp, radiation, **kwargs):
    start_at = cv.resume_offset(profiles, False) if restart else 0

    if start_at >= nphi:
        if (MPI_info['rank'] == 0):
            print(f'File {file_num} already complete through k={start_at:d} '
                  f'of {nphi:d}; nothing to do.', flush=True)
        return

    for slice_num, iphi in enumerate(range(start_at, nphi, iphi_stp)):
        profiles['saved'] = False
        k_min = int(iphi)
        k_max = int(iphi + iphi_stp)

        delta_k   = (k_max - k_min) / MPI_info['size']
        k_min_loc = int(k_min + delta_k * MPI_info['rank'])
        k_max_loc = int(k_min + delta_k * (MPI_info['rank'] + 1))

        # Staggered start to take pressure off of filesystem
        timeit.sleep(0.05 * MPI_info['rank'] + np.random.rand())

        batch_args = {
            'data_dir' : data_dir,
            'file_num' : file_num,
            'k_min'    : k_min_loc,
            'k_max'    : k_max_loc,
            'read_args': read_args,
            'CONSTANTS': CONSTANTS,
            'cgs'      : cgs,
            'two_temp' : two_temp,
            'radiation': radiation,
            'keys'     : keys,
            'vec_keys' : vec_keys
        }

        print(f"[EXTRACTION] rank {MPI_info['rank']:d} slice "
              f"{slice_num + 1:d}/{nslices:d} reading k "
              f"[{k_min_loc:d}, {k_max_loc:d}) of {nphi:d}", flush=True)
        prof_loc, meta = process_batch(**batch_args, **kwargs)
        nr = len(meta['r'])

        local_buffer = np.zeros((len(profile_keys), nr), dtype=np.float64)
        for i, key in enumerate(profile_keys):
            if key in prof_loc:
                local_buffer[i, :] = prof_loc[key][:nr]
        global_buffer = None

        print(f"Rank {MPI_info['rank']:d} done analyzing range of k: " +
            f"({k_min_loc:d}, {k_max_loc:d}).", flush=True)

        if (MPI_info['rank'] == 0):
            profiles['r']    = meta['r']
            profiles['dx1']  = meta['dx1']
            profiles['rho0'] = meta['rho0']
            global_buffer = np.zeros_like(local_buffer)

        MPI_info['comm'].Reduce(local_buffer, global_buffer, op=MPI.SUM, root=0)

        if (MPI_info['rank'] == 0):
            for i, key in enumerate(profile_keys):
                profiles[key] += global_buffer[i, :]

            cv.record_progress(profiles, {'k_max': k_max}, False, nphi)
            profiles['slice_num_restart'] = slice_num + 1
            cv.save_checkpoint(save_dir, file_num, nzro, profiles)

            snum = int(slice_num + 1)
            print(f'Finished with slice {snum:d} of {nslices:d} ' +
                    f'for file number {file_num}', flush=True)
        MPI_info['comm'].Barrier()
        gc.collect()    

    print(f'Done with all profiles for file number {file_num}.', flush=True)

def process_batch(data_dir, file_num, read_args,
                  CONSTANTS, cgs, two_temp, radiation, keys, vec_keys,
                  k_min=None, k_max=None, start_block=None, end_block=None,
                  **kwargs):
    MBH = kwargs.get('MBH')
    Rg  = kwargs.get('Rg')
    c   = kwargs.get('c')

    if cgs:
        kB   = kwargs.get('kB')
        mp   = kwargs.get('mp')
        x_AB = kwargs.get('x_AB')
        y_AB = kwargs.get('y_AB')
        mu_i = kwargs.get('mu_i')
        mu_e = kwargs.get('mu_e')
        mi   = kwargs.get('mi')
        me   = kwargs.get('me')
        mu_g = kwargs.get('mu_g')
    
    vargamma = kwargs.get('vargamma')
    
    # A batch is either a phi slice (k_min..k_max) or a contiguous range of
    # Local Block IDs (start_block..end_block); exactly one pair is given.
    read_args = dict(read_args) # copied not mutated
    if start_block is not None:
        by_block                 = True
        read_args['by_block']    = True
        read_args['start_block'] = start_block
        read_args['end_block']   = end_block
    else:
        by_block           = False
        read_args['k_min'] = k_min
        read_args['k_max'] = k_max

    # Read grid and data
    H = hr.Read(data_dir, **read_args)
    H.read_grid(file_num, direct=True)
    H.read_prims(file_num, skip_read_params=True)

    if 'tr1' in H.data.keys(): H.data['tr1'] = None
    if 'tr2' in H.data.keys(): H.data['tr2'] = None
    if 'dissipation' in H.data.keys(): H.data['dissipation'] = None
    if 'rad_src'     in H.data.keys(): H.data['rad_src']     = None

    # Deallocate coordinate transformation array
    H.grid['dxdxp'] = None

    H.block      = None
    H.grid['x1'] = None
    H.grid['x2'] = None
    H.grid['x3'] = None

    acc_class_args = {
        'grid'      : H.grid,
        'data'      : H.data,
        'pars'      : H.params,
        'CONSTANTS' : CONSTANTS, # float32 constants
        'cgs'       : cgs,
        'MBH'       : MBH,
        'radiation' : radiation
    }
    if by_block:
        acc_class_args['amr']      = H.amr
        acc_class_args['amr_info'] = H.amr_info
        acc_class_args['unfold']   = True
        acc_class_args['lowres1']  = H.lowres1
        acc_class_args['lowres2']  = H.lowres2
        acc_class_args['lowres3']  = H.lowres3

    class_args = {
        'pars'      : H.params,
        'CONSTANTS' : CONSTANTS, # float32 constants
        'cgs'       : cgs
    }
    
    coords_args = class_args.copy()
    coords_args['MBH'] = MBH

    # Initialize Accretion class
    Acc = (ana.Accretion(  **acc_class_args) if not by_block else 
           ana.AccretionBK(**acc_class_args))
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
    uu_tnsr = ana.Tools.construct_dyadic(H.data['uu'] / c, H.grid['gcov'], 
        upper_lower=True, index=[1, 0], cpp=True)

    # Calculate gas pressure
    if (two_temp and vargamma):
        pgas= Fluid.calc_pgas(**H.data)
    else:
        pgas      = hr.Tools.zeroNd(H.data['u_gas'].shape, 1)
        gamma_adi = np.float32(H.params['gam'])
        np.multiply(H.data['u_gas'], (gamma_adi - ONE), out=pgas)

    # Mass and energy accretion/outflow rates
    stress_energy_HD  = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas,
        uu_tnsr=uu_tnsr, upper_lower=True, index=[1, 0], cpp=True)
    stress_energy_EM  = Field.calc_TEM(**H.data, **H.grid, b2=b2, b=b,
        uu_tnsr=uu_tnsr, upper_lower=True, index=[1, 0], cpp=True)
    if (radiation):
        stress_energy_RAD = Radi.calc_TRAD(**H.data, **H.grid,
            upper_lower=True, index=[1, 0], cpp=True)
    del uu_tnsr

    H.grid['gcov'] = None
    H.grid['gcon'] = None
    if (radiation):
        H.data['uu_rad'] = None

    prof_loc = {}

    prof_loc['Phi_BH']      = Acc.calc_PhiBH()[0]
    prof_loc['Phi_sph_upp'] = Acc.calc_PhiSph(hemisphere='upper')[0]
    prof_loc['Phi_sph_low'] = Acc.calc_PhiSph(hemisphere='lower')[0]
    prof_loc['Phi_mid']     = Acc.calc_PhiMid()[0]
    if not by_block:
        prof_loc['Phi_max_upp'] = Acc.calc_PhiMax(hemisphere='upper')[0]
        prof_loc['Phi_max_low'] = Acc.calc_PhiMax(hemisphere='lower')[0]

    # Deallocate magnetic field in the lab frame
    H.data['B'] = None

    acc_args = {
        'sigma_cut' : np.float32(3.),
        'sigma'     : sigma
    }
    jet_args = acc_args.copy()

    # Avoid recalculating sigma selection each time
    acc_args['sigma_selection'] = Acc.sigma_cut(**acc_args)

    prof_loc['Edot_Fluid'] = Acc.calc_Edot(TMHD=stress_energy_HD, **acc_args)[0]

    stress_energy_MHD  = stress_energy_HD
    stress_energy_MHD += stress_energy_EM
    if (radiation):
        stress_energy_MHD += stress_energy_RAD

    # Deallocate gas internal energy
    H.data['u_gas'] = None

    prof_loc['Mdot']       = Acc.calc_Mdot()[0]
    prof_loc['Edot']       = Acc.calc_Edot(TMHD=stress_energy_MHD)[0]
    prof_loc['Mdot_sigma'] = Acc.calc_Mdot(**acc_args)[0]
    prof_loc['Edot_sigma'] = Acc.calc_Edot(TMHD=stress_energy_MHD, **acc_args)[0]
    prof_loc['Edot_Poyt']  = Acc.calc_Edot(TMHD=stress_energy_EM,  **acc_args)[0]
    prof_loc['Edot_jet']   = Acc.calc_Edot(TMHD=stress_energy_EM, jet=True, **jet_args)[0]
    if (radiation):
        prof_loc['Edot_Rad'] = Acc.calc_Edot(TMHD=stress_energy_RAD)[0]

    del sigma
    acc_args['sigma'] = None
    jet_args['sigma'] = None

    del stress_energy_MHD
    del stress_energy_HD, stress_energy_EM
    if (radiation):
        del stress_energy_RAD

    uu_cyl = Coords.int_to_cyl_vec(H.data['uu'], cpp=True)
    H.data['uu'] = None
    b_cyl  = Coords.int_to_cyl_vec(b, cpp=True)
    np32   = np.float32
    vec_quants = [
        uu_cyl[1], uu_cyl[2], uu_cyl[3],
        np.nan_to_num(uu_cyl[1] / uu_cyl[0], copy=False).astype(np32),
        np.nan_to_num(uu_cyl[2] / uu_cyl[0], copy=False).astype(np32),
        np.nan_to_num(uu_cyl[3] / uu_cyl[0], copy=False).astype(np32),
        b_cyl[1], b_cyl[2], b_cyl[3]
    ]

    for (key, quant) in zip(vec_keys, vec_quants):
        prof_loc[key] = Acc.thph_sum(quant, weight=H.data['rho'], **acc_args)[0]

    del uu_cyl, b_cyl, b

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

    # Calculate (theta - pi/2)^2 for density scale height
    Dtheta  = H.grid['theta'] - PI_2
    Dtheta2 = hr.Tools.zeroNd(H.data['rho'].shape, 1)
    np.square(Dtheta, out=Dtheta2)
    del Dtheta
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
        prof_loc[key]= Acc.thph_sum(quant, weight=H.data['rho'], **acc_args)[0]
    
    if (two_temp and vargamma): 
        del Theta_e, Theta_i, gam_e, gam_i, pgas, b2, pmag
        del Eion, Eelc
    elif (two_temp and not vargamma): 
        del gam_e, gam_i, pgas, b2, pmag
        del Eion, Eelc
    else:
        del pgas, T_gas, b2, pmag
    del ones, Dtheta2

    if (radiation):
        H.data['u_rad'] = None
        del prad

    # Deallocate temperatures and energy densities
    # in preparation for large transformation matrices
    if (two_temp):
        H.data['Ti'] = None
        H.data['Te'] = None

    del Acc, Coords, Field, Fluid
    if (radiation):
        del Radi
    acc_args['sigma_selection'] = None
    nr = len(H.grid['r'][0, :, 0, 0])

    # Deallocate remaining grid variables
    H.grid['gdet'] = None
    
    if by_block:
        r_1D, th_1D = H.global_coords1D(nr_out=kwargs.get('nr_out'))
    else:
        r_1D  = H.grid['r'][0, :, 0, 0] / Rg
        th_1D = H.grid['theta'][0, 0, :, 0]

    coords_data = {
        'r'    : r_1D,
        'dx1'  : H.params['dx1'],
        'rho0' : H.params['density_scale']
    }

    return prof_loc, coords_data

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Calculate 1D profiles. If --end_file
                        is supplied, parallelization is done in time
                        (each dump is processed by a separate MPI rank).
                        Otherwise, files are parallelized phi slices.""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('file_num', type=int, help = """dump number""")
    parser.add_argument('--end_file', type=int, required=False, 
                        default=-1, help = """dump number
                        of end file in profile extraction.""")
    parser.add_argument('--nslices', type=int, required=False,
                        default=1,
                        help="""number of slices in phi 
                        used to load the data.""")
    parser.add_argument('--lowres1', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in x1""")
    parser.add_argument('--lowres2', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in x2""")
    parser.add_argument('--lowres3', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in x3""")
    parser.add_argument('--stride', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in dump number
                        only relevant if --end_file is supplied.""")
    parser.add_argument('--by_block', action='store_true', required=False,
                        default=False,
                        help = """read the dump block-by-block at each block's
                        native refinement level, in batches, instead of
                        upscaling the whole dump onto a uniform grid. Works
                        in both single-file and time-parallel modes.""")
    parser.add_argument('--blocks_per_batch', type=int, default=-1,
                        required=False,
                        help = """blocks read per batch when --by_block is
                        set. Default -1 calibrates it automatically against
                        the per-rank memory budget via block_budget. Pass a
                        value only to override that -- to reproduce an
                        earlier run exactly, or to force a smaller batch
                        after an out-of-memory failure.""")
    parser.add_argument('--radiation', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if radiation quantities
                        should be included in profile extraction.""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if two-temperature 
                        physics should be included in profile extraction.""")
    parser.add_argument('--vargamma', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""boolean to determine if variable gamma 
                        should be used for ion/electron energy/pressure.""")  
    parser.add_argument('--MBH', type=float, default=10.,
                        required=False,
                        help = """black hole mass in Msun. (default: 10)""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if profile extraction
                        should be done on reduced data.""")
    parser.add_argument('--restart', action='store_true', required=False, 
                        help="""boolean set to determine if profile 
                        integration should continue from last slice_num.""")
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if existing profile file 
                        should be overwritten. Only relevant if --end_file
                        is supplied as a command-line argument.""")
    parser.add_argument('--shared', action='store_true', required=False, 
                        help="""boolean set to determine if proj-shared should
                        be used as the location of the data directory.""")
    parser.add_argument('--title', type=str, required=False, default=None, 
                        help="""title of simulation used for Averages directory
                        if different from the title in proj-shared.""")
    args = parser.parse_args()
    main(**vars(args))