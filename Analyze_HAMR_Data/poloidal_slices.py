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
from mpi4py.util import pkl5
import gc
import time as timeit

# Numerical constants for performance
ONE         = np.float32(1.)
HALF        = np.float32(0.5)
ONE_THIRD   = np.float32(1. / 3.)
FOUR_THIRDS = np.float32(4. / 3.)
FOURPI      = np.float32(4. * np.pi)
EIGHTPI     = np.float32(8. * np.pi)

def main(**kwargs):
    sim       = kwargs.get('sim')
    file_num  = int(kwargs.get('file_num'))
    end_file  = int(kwargs.get('end_file'))
    nslices   = kwargs.get('nslices')
    lowres1   = kwargs.get('lowres1')
    lowres2   = kwargs.get('lowres2')
    lowres3   = kwargs.get('lowres3')
    stride    = kwargs.get('stride')
    r_max     = kwargs.get('r_max')
    reduced   = kwargs.get('reduced')
    restart   = kwargs.get('restart')
    overwrite = kwargs.get('overwrite')
    shared    = kwargs.get('shared')
    title     = kwargs.get('title')
    radiation = kwargs.get('radiation')
    two_temp  = kwargs.get('two_temp')
    RIAF      = kwargs.get('RIAF')
    cooling   = kwargs.get('cooling')
    vargamma  = kwargs.get('vargamma')
    MBH       = np.float32(kwargs.get('MBH'))
    by_block  = kwargs.get('by_block')
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
    physics_args['RIAF']     = RIAF
    physics_args['cooling']  = cooling

    # Ensure CONSTANTS are 32-bit floats
    CONSTANTS = {key : np.float32(value) for (key, value) in hr.CONSTANTS.items()}

    # Determine data extraction and save directories
    data_dir = cv.get_data_dir(sim, reduced, shared)
    save_dir = cv.get_save_dir(sim if title == None else title,
                               'Averages', 'Slices', reduced=reduced)

    nzeros = len(str(len(glob.glob(os.path.join(data_dir, 'dumps*')))))

    # Quantities (quants) for slice averaging
    keys = [
        'rho', 'Ti', 'Pmag', 'Pthm', 'Vol', 'w'
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
        'wuR',  'wuphi', 'wuz', 'wu_phi', 'wur', 'wuth', 'wut',
        'vR',   'vphi',  'vz',
        'bR',   'bphi',  'bz',  'b_phi',  'br',  'bth'
    ]

    if (radiation):
        flux_keys += ['R_tR', 'R_tz', 'R_tt', 'R_t1',
                      'Be_Rad', 'FRad_R_Fluid', 'FRad_z_Fluid']

    rhovec_keys = ['rhouR', 'rhouphi', 'rhouz', 'rhou1']

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

    read_args = {
        'cgs_grid' : cgs,
        'cgs_data' : cgs,
        'coords1D' : True,
        'MBH'      : MBH,
        'lowres1'  : lowres1,
        'lowres2'  : lowres2,
        'lowres3'  : lowres3
    }

    if (r_max != 0.):
        read_args['r_max'] = r_max

    slice_keys = keys + flux_keys + vec_keys + rhovec_keys + ['B1']

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

    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    size = comm.Get_size()
    rank = comm.Get_rank()

    MPI_info = {
        'comm' : comm,
        'size' : size,
        'rank' : rank
    }

    # Calibrate the total number of batch size that can be analyzed on a node.
    if by_block and blocks_per_batch < 0:
        def probe_batch(n_blocks):
            print(f"[CALIBRATION] probe {n_blocks} blocks", flush=True)
            process_batch(data_dir=data_dir, file_num=file_num,
                          read_args=read_args, CONSTANTS=CONSTANTS, cgs=cgs,
                          two_temp=two_temp, radiation=radiation,
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

        slices, nphi, iphi_stp = cv.init_slices(keys=slice_keys,
                                                **init_args, **read_args)
        if not restart:
            slices['done'] = False

        run_args = {
            'slices'     : slices,
            'MPI_info'   : MPI_info,
            'read_args'  : read_args,
            'slice_keys' : slice_keys,
            'CONSTANTS'  : CONSTANTS,
            'cgs'        : cgs
        }

        if by_block:
            blk_args = {k: v for k, v in init_args.items()
                        if k not in ('nslices', 'blk_stp')}
            run_parallel_by_block(**run_args, nb=int(slices['nb']),
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
            existing_slices = cv.get_file_list_local(file_list_local, 1, 0,
                averages_dir, file_type='Slices', reduced=reduced)
            existing_set = set(existing_slices)
            file_list_local = [f for f in file_list_local if f not in existing_set]
        
        run_args = {
            'file_list_local' : file_list_local,
            'read_args'       : read_args,
            'init_args'       : init_args,
            'MPI_info'        : MPI_info,
            'slice_keys'      : slice_keys,
            'keys'            : keys,
            'vec_keys'        : vec_keys,
            'CONSTANTS'       : CONSTANTS,
            'cgs'             : cgs
        }

        run_parallel_in_time(**run_args, **physics_args)

def run_parallel_by_block(slices, nb, blk_stp, MPI_info, read_args,
                          data_dir, file_num, slice_keys, save_dir,
                          CONSTANTS, cgs, restart, nzro,
                          two_temp, radiation, RIAF, cooling, **kwargs):
    """Iterate through data in batches based on calibrated block step."""
    rank, size = MPI_info['rank'], MPI_info['size']
    nr, nth    = slices[slice_keys[0]].shape

    start_at = cv.resume_offset(slices, True) if restart else 0
    nbatch   = len(range(start_at, nb, blk_stp))

    if start_at >= nb:
        if (rank == 0):
            print(f'File {file_num} already complete through block '
                  f'{start_at:d} of {nb:d}; nothing to do.', flush=True)
        return

    for batch_num, batch_sel in cv.iter_block_batches(nb, blk_stp,
                                                      start=start_at):
        slices['saved'] = False
        rank_sel = cv.split_batch(batch_sel, rank, size)

        # Staggered start to take pressure off of filesystem
        timeit.sleep(0.05 * rank + np.random.rand())

        local_buffer = np.zeros((len(slice_keys), nr, nth), dtype=np.float64)
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
                'RIAF'     : RIAF,
                'cooling'  : cooling,
                'nr_out'   : nr
            }
            print(f"[EXTRACTION] rank {rank:d} batch {batch_num + 1:d}/"
                  f"{nbatch:d} reading blocks "
                  f"[{rank_sel['start_block']:d}, {rank_sel['end_block']:d}) "
                  f"of {nb:d}", flush=True)
            slice_loc, meta = process_batch(**batch_args, **rank_sel, **kwargs)
            for i, key in enumerate(slice_keys):
                if key in slice_loc:
                    local_buffer[i, :, :] = slice_loc[key][:nr, :nth]
            del slice_loc
        else:
            print(f"[EXTRACTION] rank {rank:d} batch {batch_num + 1:d}/"
                  f"{nbatch:d} has no blocks; contributing zeros.", flush=True)

        global_buffer = None
        if (rank == 0):
            slices['r']     = meta['r']
            slices['theta'] = meta['theta']
            slices['phi']   = meta['phi']
            slices['dx1']   = meta['dx1']
            slices['dx2']   = meta['dx2']
            slices['dx3']   = meta['dx3']
            slices['rho0']  = meta['rho0']
            global_buffer   = np.zeros_like(local_buffer)

        MPI_info['comm'].Reduce(local_buffer, global_buffer, op=MPI.SUM, root=0)
        del local_buffer

        if (rank == 0):
            for i, key in enumerate(slice_keys):
                slices[key] += global_buffer[i, :, :]

            cv.record_progress(slices, batch_sel, True, nb, blk_stp=blk_stp)
            cv.save_checkpoint(save_dir, file_num, nzro, slices)

            print(f'Finished with batch {batch_num + 1:d} of {nbatch:d} '
                  f'for file number {file_num}', flush=True)
        del global_buffer
        MPI_info['comm'].Barrier()
        gc.collect()

    print(f'Done with all slices for file number {file_num}.', flush=True)

def run_parallel_in_time(file_list_local, read_args, init_args, MPI_info,
                         slice_keys, CONSTANTS, cgs, RIAF, cooling, **kwargs):
    """Parallelization across dump number. Each rank receives a single dump."""
    data_dir = init_args['data_dir']
    save_dir = init_args['save_dir']
    for file_num in file_list_local:
        init_args['file_num'] = file_num
        slices, nphi, iphi_stp = cv.init_slices(keys=slice_keys,
                                                **init_args, **read_args)
        if not init_args['restart']:
            slices['done'] = False

        batch_args = {
            'data_dir' : data_dir,
            'file_num' : file_num,
            'read_args': read_args,
            'CONSTANTS': CONSTANTS,
            'cgs'      : cgs,
            'two_temp' : init_args['two_temp'],
            'radiation': init_args['radiation'],
            'RIAF'     : RIAF,
            'cooling'  : cooling
        }

        # Unfold straight into the extent these arrays actually have; this
        # path adds slice_loc in without truncating.
        batch_args['nr_out'] = slices[slice_keys[0]].shape[0]

        by_block = init_args.get('by_block', False)
        nb       = int(slices['nb'])
        blk_stp  = init_args.get('blk_stp') or nb    # None/-1 -> one batch
        nbatch   = (len(range(0, nb, blk_stp)) if by_block
                    else init_args['nslices'])

        restart  = init_args['restart']
        start_at = cv.resume_offset(slices, by_block) if restart else 0
        limit    = nb if by_block else nphi
        if start_at >= limit:
            print(f'File {file_num} already complete; skipping.', flush=True)
            continue

        batches = (cv.iter_block_batches(nb, blk_stp, start=start_at)
                   if by_block else
                   cv.iter_phi_batches(nphi, iphi_stp, start=start_at))
        for batch_num, sel in batches:
            slices['saved'] = False

            # Staggered start to take pressure off of filesystem
            timeit.sleep(0.05 * MPI_info['rank'] + np.random.rand())

            batch_args.update(sel)

            lo, hi = ((sel['start_block'], sel['end_block']) if by_block
                      else (sel['k_min'], sel['k_max']))
            print(f"[EXTRACTION] rank {MPI_info['rank']:d} file {file_num} "
                  f"batch {batch_num + 1:d}/{nbatch} reading "
                  f"{'blocks' if by_block else 'k'} [{lo:d}, {hi:d})",
                  flush=True)
            slice_loc, meta = process_batch(**batch_args, **kwargs)

            for key in slice_keys:
                if key in slice_loc:
                    slices[key] += slice_loc[key]

            slices['r']     = meta['r']
            slices['theta'] = meta['theta']
            slices['phi']   = meta['phi']
            slices['dx1']   = meta['dx1']
            slices['dx2']   = meta['dx2']
            slices['dx3']   = meta['dx3']
            slices['rho0']  = meta['rho0']
            cv.record_progress(slices, sel, by_block, nb if by_block else nphi,
                               blk_stp=blk_stp if by_block else None)

            print(f"Finished with batch {batch_num + 1:d} of {nbatch} "
                  f"for file {file_num}", flush=True)

        slices['slice_num_restart'] = batch_num + 1   # legacy, not read
        cv.save_checkpoint(save_dir, file_num, init_args['nzro'], slices)
        MPI_info['comm'].Barrier()
        gc.collect()

        print(f"Done with all slices for file number {file_num}.", flush=True)

def run_parallel_in_phi(slices, nphi, iphi_stp, MPI_info, read_args,
                        data_dir, file_num, slice_keys, save_dir,
                        CONSTANTS, cgs, nslices, restart, nzro,
                        two_temp, radiation, RIAF, cooling, **kwargs):
    """Parallel in phi: each rank receives phi-slice of full domain."""
    start_at = cv.resume_offset(slices, False) if restart else 0

    if start_at >= nphi:
        if (MPI_info['rank'] == 0):
            print(f'File {file_num} already complete through k={start_at:d} '
                  f'of {nphi:d}; nothing to do.', flush=True)
        return

    for slice_num, iphi in enumerate(range(start_at, nphi, iphi_stp)):
        slices['saved'] = False
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
            'RIAF'     : RIAF,
            'cooling'  : cooling
        }

        print(f"[EXTRACTION] rank {MPI_info['rank']:d} slice "
              f"{slice_num + 1:d}/{nslices:d} reading k "
              f"[{k_min_loc:d}, {k_max_loc:d}) of {nphi:d}", flush=True)
        slice_loc, meta = process_batch(**batch_args, **kwargs)
        nr, nth = meta['nr'], meta['nth']

        local_buffer = np.zeros((len(slice_keys), nr, nth), dtype=np.float64)
        for i, key in enumerate(slice_keys):
            if key in slice_loc:
                local_buffer[i, :, :] = slice_loc[key]
        global_buffer = None

        print(f"Rank {MPI_info['rank']:d} done analyzing range of k: " +
            f"({k_min_loc:d}, {k_max_loc:d}).", flush=True)

        if (MPI_info['rank'] == 0):
            slices['r']     = meta['r']
            slices['theta'] = meta['theta']
            slices['phi']   = meta['phi']
            slices['dx1']   = meta['dx1']
            slices['dx2']   = meta['dx2']
            slices['dx3']   = meta['dx3']
            slices['rho0']  = meta['rho0']
            global_buffer   = np.zeros_like(local_buffer)

        MPI_info['comm'].Reduce(local_buffer, global_buffer, op=MPI.SUM, root=0)
        del local_buffer

        if (MPI_info['rank'] == 0):
            for i, key in enumerate(slice_keys):
                slices[key] += global_buffer[i, :, :]

            cv.record_progress(slices, {'k_max': k_max}, False, nphi)
            # Legacy field: written for older tooling, no longer read.
            slices['slice_num_restart'] = slice_num + 1
            cv.save_checkpoint(save_dir, file_num, nzro, slices)

            snum = int(slice_num + 1)
            print(f"Finished with slice {snum:d} of {nslices:d} " +
                  f"for file number {file_num}", flush=True)
        MPI_info['comm'].Barrier()
        gc.collect()

    print(f"Done with all slices for file number {file_num}.", flush=True)

def process_batch(data_dir, file_num, read_args,
                  CONSTANTS, cgs, two_temp, radiation,
                  k_min=None, k_max=None, start_block=None, end_block=None,
                  **kwargs):
    """
    Calculates poloidal slice quantities for single slice in phi or
    set of blocks if by_block=True.
    """
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
    RIAF     = kwargs.get('RIAF')
    cooling  = kwargs.get('cooling')

    # A batch is either a phi slice (k_min..k_max) or a contiguous range of
    # Local Block IDs (start_block..end_block); exactly one pair is given.
    
    # read_args is copied, not mutated. 
    read_args = dict(read_args)
    if start_block is not None:
        by_block                 = True
        read_args['by_block']    = True
        read_args['start_block'] = start_block
        read_args['end_block']   = end_block
    else:
        by_block           = False
        read_args['k_min'] = k_min
        read_args['k_max'] = k_max

    H = hr.Read(data_dir, **read_args)
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
    if by_block:
        acc_class_args['amr']      = H.amr
        acc_class_args['amr_info'] = H.amr_info
        acc_class_args['unfold']   = True
        acc_class_args['lowres1']  = H.lowres1
        acc_class_args['lowres2']  = H.lowres2
        acc_class_args['lowres3']  = H.lowres3
        acc_class_args['nr_out']   = kwargs.get('nr_out')

    class_args = {
        'pars'      : H.params,
        'CONSTANTS' : CONSTANTS, # float32 constants
        'cgs'       : cgs
    }

    coords_args = class_args.copy()
    coords_args['MBH'] = MBH

    # Units
    rho0   = H.params['density_scale']
    rho0c2 = rho0 * c * c

    # Initialize Accretion class. Every reduction below is Acc.ph_sum, which
    # AccretionBK overrides to unfold2D onto the global (r, theta) grid.
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

    # Calculate sigma (3D). 
    b  = Field.calc_b(**H.data, **H.grid, cpp=True)
    b2 = Field.calc_b2(b=b, **H.data, **H.grid, cpp=True)

    slice_loc = {}

    # B^1 (useful for calculating vector potential)
    slice_loc['B1'] = Acc.ph_sum(H.data['B'][1])[0]
    H.data['B'] = None

    if cgs:
        sigma = (b2 / FOURPI) / (H.data['rho'] * c * c)
    else:
        sigma = b2 / H.data['rho']

    # Calculate gas pressure
    if two_temp and vargamma:
        pgas = Fluid.calc_pgas(**H.data)
    else:
        pgas      = hr.Tools.zeroNd(H.data['u_gas'].shape, 1)
        gamma_adi = np.float32(H.params['gam'])
        np.multiply(H.data['u_gas'], (gamma_adi - ONE), out=pgas)

    # Cut on sigma
    sigma_cut_args = {
        'sigma_cut' : np.float32(3.),
        'sigma'     : sigma
    }

    # Energy fluxes
    # Calculate T^i_t streamlines/fluxes
    stress_energy_HD_10 = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas,
        upper_lower=True, index=[1, 0], cpp=True)
    stress_energy_HD_20 = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas,
        upper_lower=True, index=[2, 0], cpp=True)
    THD_vec    = np.zeros_like(H.data['uu'])
    THD_vec[1] = stress_energy_HD_10
    THD_vec[2] = stress_energy_HD_20
    THD_cyl    = Coords.int_to_cyl_vec(THD_vec, cpp=True)
    slice_loc['THD_tR'] = Acc.ph_sum(THD_cyl[1])[0]
    slice_loc['THD_tz'] = Acc.ph_sum(THD_cyl[3])[0]
    slice_loc['THD_t1'] = Acc.ph_sum(THD_vec[1])[0]

    del THD_cyl, THD_vec, stress_energy_HD_10, stress_energy_HD_20

    stress_energy_EM_10 = Field.calc_TEM(**H.data, **H.grid, b2=b2, b=b,
        upper_lower=True, index=[1, 0], cpp=True)
    stress_energy_EM_20 = Field.calc_TEM(**H.data, **H.grid, b2=b2, b=b,
        upper_lower=True, index=[2, 0], cpp=True)
    TEM_vec    = np.zeros_like(H.data['uu'])
    TEM_vec[1] = stress_energy_EM_10
    TEM_vec[2] = stress_energy_EM_20
    TEM_cyl    = Coords.int_to_cyl_vec(TEM_vec, cpp=True)
    slice_loc['TEM_tR'] = Acc.ph_sum(TEM_cyl[1])[0]
    slice_loc['TEM_tz'] = Acc.ph_sum(TEM_cyl[3])[0]
    slice_loc['TEM_t1'] = Acc.ph_sum(TEM_vec[1])[0]

    del TEM_cyl, TEM_vec, stress_energy_EM_10, stress_energy_EM_20

    # Calculate R^i_t streamlines/fluxes
    if radiation:
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
        R_cyl    = Coords.int_to_cyl_vec(R_vec, cpp=True)
        # R^t_t necessary for thermal and radative diffusion timescales
        slice_loc['R_tt'] = Acc.ph_sum(R_cyl[0])[0]
        slice_loc['R_tR'] = Acc.ph_sum(R_cyl[1])[0]
        slice_loc['R_tz'] = Acc.ph_sum(R_cyl[3])[0]
        slice_loc['R_t1'] = Acc.ph_sum(R_vec[1])[0]

        del R_cyl, R_vec, stress_energy_RAD_00
        del stress_energy_RAD_10, stress_energy_RAD_20

    if radiation:
        # Radiation energy flux in the fluid frame (u^mu)
        # for the calculation of the radiative diffusion timescale.
        Gamma = -(ana.Tools.dot(H.data['uu']/c, H.data['uu_rad']/c, H.grid['gcov'], cpp=True))
        uurad_proj = (H.data['uu_rad']/c - Gamma * H.data['uu']/c) 
        Rad_Flux     = FOUR_THIRDS * H.data['u_rad'] * Gamma * uurad_proj
        Rad_Flux_cyl = Coords.int_to_cyl_vec(Rad_Flux, cpp=True)
        # Radiation energy flux has units of energy flux.
        slice_loc['FRad_R_Fluid'] = Acc.upscale64(Acc.ph_sum(Rad_Flux_cyl[1])[0], c)
        slice_loc['FRad_z_Fluid'] = Acc.upscale64(Acc.ph_sum(Rad_Flux_cyl[3])[0], c)
        del Gamma, uurad_proj, Rad_Flux, Rad_Flux_cyl
        H.data['uu_rad'] = None

    uu_cov = ana.Tools.contract_vector_cpp(H.data['uu'], H.grid['gcov'])
    b_cov  = ana.Tools.contract_vector_cpp(b,            H.grid['gcov'])
    H.grid['gcov'] = None
    H.grid['gcon'] = None

    # Opacities
    if radiation:
        # Calculate opacities including all processes
        opacities = dict.fromkeys(['electron_scattering', 'bound_free',
            'CHIANTI', 'free_free', 'hydrogen', 'molecular', 'synchrotron'])
        
        if two_temp and not RIAF:
            Opac = rad.Opacity(opacities, **H.data, pars=H.params,
                CONSTANTS=hr.CONSTANTS, b2=b2, XRB=True, cpp=True)
        elif two_temp and RIAF:
            Opac = rad.RIAF_Opacity(opacities, **H.data, pars=H.params,
                CONSTANTS=hr.CONSTANTS, b2=b2, XRB=True, cpp=True)
        elif not two_temp and RIAF:
            Opac = rad.RIAF_Opacity(opacities, **H.data,
                Te=(mu_g * mp / kB) * (pgas / H.data['rho']),
                pars=H.params, CONSTANTS=hr.CONSTANTS,
                b2=b2, XRB=True, cpp=True)
        else:
            Opac = rad.Opacity(opacities, **H.data,
                Te=(mu_g * mp / kB) * (pgas / H.data['rho']),
                pars=H.params, CONSTANTS=hr.CONSTANTS,
                b2=b2, XRB=True, cpp=True)
        kappa_abs = Opac.opacities['absorption_total']
        kappa_es  = Opac.opacities['electron_scattering']
        kappa_eff = np.sqrt(kappa_abs * (kappa_abs + kappa_es))

        slice_loc['rhokappa'] = Acc.ph_sum(kappa_eff * H.data['rho'],
                                           **sigma_cut_args)[0]
        del Opac, kappa_abs, kappa_es, kappa_eff

    # Bernoulli parameter for outflows
    # Numerator of Bernoulli parameter (de-dimensionalized to handle overflows)
    if cgs:
        w = (H.data['rho'] / rho0 + pgas / rho0c2 + H.data['u_gas'] / rho0c2 
            + (b2 / FOURPI) / rho0c2)
    else:
        w = H.data['rho'] + pgas + H.data['u_gas'] + b2
    slice_loc['Be'] = Acc.upscale64(Acc.ph_sum(w * uu_cov[0])[0], rho0c2)
    if radiation:
        w = (H.data['rho'] / rho0 + pgas / rho0c2 + H.data['u_gas'] / rho0c2 
            + (b2 / FOURPI) / rho0c2 + FOUR_THIRDS * H.data['u_rad'] / rho0c2)
        slice_loc['Be_Rad'] = Acc.upscale64(Acc.ph_sum(w * uu_cov[0])[0], rho0c2)

    uu_cyl     = Coords.int_to_cyl_vec(H.data['uu'],       cpp=True)
    uu_sph     = Coords.int_to_ortho_sph_vec(H.data['uu'], cpp=True)
    uu_cov_cyl = Coords.int_to_cyl_one_form(uu_cov,        cpp=True)
    b_cyl      = Coords.int_to_cyl_vec(b,                  cpp=True)
    b_sph      = Coords.int_to_ortho_sph_vec(b,            cpp=True)
    b_cov_cyl  = Coords.int_to_cyl_one_form(b_cov,         cpp=True)
    
    u_t = uu_cov[0]
    del b, b_cov, uu_cov
    
    np32 = np.float32
    vec_keys = [
        'uR',   'uphi',  'uz',
        'wuR',  'wuphi', 'wuz', 'wu_phi', 'wur', 'wuth', 'wut',
        'vR',   'vphi',  'vz',
        'bR',   'bphi',  'bz',  'b_phi',  'br',  'bth'
    ]
    quants = [
        uu_cyl[1],     uu_cyl[2],     uu_cyl[3],
        w * uu_cyl[1], w * uu_cyl[2], w * uu_cyl[3],
        w * uu_cov_cyl[2], w * uu_sph[1], w * uu_sph[2], w * uu_cyl[0],
        np.nan_to_num(uu_cyl[1] / uu_cyl[0], copy=False).astype(np32),
        np.nan_to_num(uu_cyl[2] / uu_cyl[0], copy=False).astype(np32),
        np.nan_to_num(uu_cyl[3] / uu_cyl[0], copy=False).astype(np32),
        b_cyl[1], b_cyl[2], b_cyl[3], b_cov_cyl[2], b_sph[1], b_sph[2]
    ]

    for (key, quant) in zip(vec_keys, quants):
        slice_loc[key] = Acc.ph_sum(quant)[0]
 
    # Mass fluxes
    rhovec_keys = ['rhouR', 'rhouphi', 'rhouz', 'rhou1']
    quants = [
        H.data['rho'] * uu_cyl[1],
        H.data['rho'] * uu_cyl[2],
        H.data['rho'] * uu_cyl[3],
        H.data['rho'] * H.data['uu'][1]
    ]
    for (key, quant) in zip(rhovec_keys, quants):
        slice_loc[key] = Acc.ph_sum(quant, **sigma_cut_args)[0]

    del uu_cyl, b_cyl, uu_sph, b_sph, uu_cov_cyl, b_cov_cyl

    # Magnetic and Thermodynamic Structure
    if two_temp:
        if vargamma:
            # Calculate gamma (ratio of specific heats)
            # Equation 28 in Sadowski+ (2017)
            Theta_e = (kB * H.data['Te']) / (me * c * c)
            Theta_i = (kB * H.data['Ti']) / (mi * c * c)

            gam_e = np.float32((10. + 20.*Theta_e) / (6. + 15.*Theta_e))
            gam_i = np.float32((10. + 20.*Theta_i) / (6. + 15.*Theta_i))
        else:
            gam_e = np.float32(5./3.)
            gam_i = np.float32(5./3.)

    ones = np.ones_like(H.data['rho'])

    # Calculate magnetic pressure
    pmag = b2 / EIGHTPI if cgs else HALF * b2

    if two_temp and radiation:
        Eion, Eelc = hr.Tools.zeroNd(H.data['rho'].shape, 2)
        np.multiply(H.data['rho'], H.data['Ti'], out=Eion)
        np.multiply(H.data['rho'], H.data['Te'], out=Eelc)
        Eion *= np.float32(1./(gam_i-1.) * (kB / (mu_i * mp)))
        Eelc *= np.float32(1./(gam_e-1.) * (kB / (mu_e * mp)))
        prad  = ONE_THIRD * H.data['u_rad']
        thermo_keys = ['rho', 'Ti', 'Pmag', 'Pthm', 'Vol', 'w',
                       'Te', 'Prad', 'Eion', 'Eelc']
        quants = [
            H.data['rho'], # rho
            H.data['Ti'],  # Ti
            pmag,          # Pmag = b2/8pi
            pgas,          # Pthm
            ones,          # Vol
            w,             # w (enthalpy)
            H.data['Te'],  # Te
            prad,          # Prad = 1/3 u_rad
            Eion,          # Eion (ion energy density)
            Eelc           # Eelc (electron energy density)
        ]
    elif radiation and not two_temp:
        # Te does not exist in H.data when two_temp is off -- there is
        # no separate electron temperature to report here.
        T_gas = pgas / H.data['rho']
        T_gas *= mu_g * mp / kB
        prad  = ONE_THIRD * H.data['u_rad']
        thermo_keys = ['rho', 'Ti', 'Pmag', 'Pthm', 'Vol', 'w', 'Prad']
        quants = [
            H.data['rho'], # rho
            T_gas,         # Ti (gas temperature when no 2T)
            pmag,          # Pmag = b2/8pi
            pgas,          # Pthm
            ones,          # Vol
            w,             # w (enthalpy)
            prad           # Prad = 1/3 u_rad
        ]
    elif two_temp and not radiation:
        Eion, Eelc = hr.Tools.zeroNd(H.data['rho'].shape, 2)
        np.multiply(H.data['rho'], H.data['Ti'], out=Eion)
        np.multiply(H.data['rho'], H.data['Te'], out=Eelc)
        Eion *= np.float32(1./(gam_i-1.) * (kB / (mu_i * mp)))
        Eelc *= np.float32(1./(gam_e-1.) * (kB / (mu_e * mp)))
        thermo_keys = ['rho', 'Ti', 'Pmag', 'Pthm', 'Vol', 'w',
                       'Te', 'Eion', 'Eelc']
        quants = [
            H.data['rho'], # rho
            H.data['Ti'],  # Ti
            pmag,          # Pmag = b2/8pi
            pgas,          # Pthm
            ones,          # Vol
            w,             # w (enthalpy)
            H.data['Te'],  # Te
            Eion,          # Eion (ion energy density)
            Eelc           # Eelc (electron energy density)
        ]
    else:
        T_gas = pgas / H.data['rho']
        thermo_keys = ['rho', 'Ti', 'Pmag', 'Pthm', 'Vol', 'w']
        quants = [
            H.data['rho'], # rho
            T_gas,         # Ti (gas temperature when no 2T)
            pmag,          # Pmag = b2/8pi
            pgas,          # Pthm
            ones,          # Vol
            w              # w (enthalpy)
        ]

    for (key, quant) in zip(thermo_keys, quants):
        slice_loc[key] = Acc.ph_sum(quant)[0]
    
    # Coulomb collisional and radiative cooling
    if cooling and two_temp:
        # Exclude relativistic temperatures in qie
        non_rel = (H.data['Ti'] < 1.e12) * (H.data['Te'] < 1.e10) * (
                  (H.data['Te'] > 1.e4))
        qie  = Fluid.calc_qie(**H.data, cpp=True, cooling=True) * non_rel
        slice_loc['qie'] = Acc.ph_sum(qie, weight=u_t/c)[0]
        del qie, non_rel
    if cooling and radiation:
        qIC  = Radi.calc_qIC(  **H.data)
        qemi = Radi.calc_q_emi(**H.data, b2=b2, XRB=True, cpp=True, RIAF=RIAF)
        qabs = Radi.calc_q_abs(**H.data, b2=b2, XRB=True, cpp=True, RIAF=RIAF)
        slice_loc['qIC']  = Acc.ph_sum(qIC,  weight=u_t/c)[0]
        slice_loc['qemi'] = Acc.ph_sum(qemi, weight=u_t/c)[0]
        slice_loc['qabs'] = Acc.ph_sum(qabs, weight=u_t/c)[0]
        del qIC, qemi, qabs

    # phi integral over gdet including sigma cut
    slice_loc['Vol_sigma_cut'] = Acc.ph_sum(ones, **sigma_cut_args)[0]
    
    # Restore units for output
    for key in ['w', 'wuR', 'wuphi', 'wuz', 'wu_phi', 'wur', 'wuth', 'wut']:
        slice_loc[key] = Acc.upscale64(slice_loc[key], rho0c2)

    H.data['u_gas'] = None
    if (two_temp and vargamma):
        del Theta_e, Theta_i, gam_e, gam_i, pgas
    elif (two_temp and not vargamma):
        del gam_e, gam_i, pgas
    else:
        del pgas, T_gas

    del b2, pmag
    if (radiation):
        H.data['u_rad'] = None
    del ones

    # Deallocate temperatures and energy densities
    # in preparation for large transformation matrices
    if (two_temp):
        H.data['Ti'] = None
        H.data['Te'] = None

    # Deallocate memory before returning
    H.data['uu']    = None
    H.grid['gdet']  = None
    H.grid['dxdxp'] = None

    H.block = None
    H.data['rho']   = None
    del w, u_t, sigma

    # grid['r'][0] is the FIRST BLOCK's axis under by_block, not the dump's,
    # but ph_sum has already unfolded onto the global grid -- so the axes
    # saved beside the data have to be the global ones.
    if by_block:
        # Already geometric: global_coords1D returns exp(x1) directly, while
        # grid['r'] carries the cgs_grid scaling that the /Rg below undoes.
        # Dividing here too applies Rg twice.
        r_1D, th_1D = H.global_coords1D(nr_out=kwargs.get('nr_out'))
    else:
        r_1D  = H.grid['r'][0, :, 0, 0] / Rg
        th_1D = H.grid['theta'][0, 0, :, 0]

    coords_data = {
        'r'     : r_1D,
        'theta' : th_1D,
        'phi'   : H.grid['phi'][0, 0, 0, :],
        'dx1'   : H.params['dx1'],
        'dx2'   : H.params['dx2'],
        'dx3'   : H.params['dx3'],
        'rho0'  : H.params['density_scale'],
        'nr'    : len(r_1D),
        'nth'   : len(th_1D)
    }

    del Acc, Coords, Fluid, Field, Radi

    return slice_loc, coords_data

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Calculate 2D slices. If --end_file
                        is supplied, parallelization is done in time
                        (each dump is processed by a separate MPI rank).
                        Otherwise, ranks parallelize phi slices of a
                        single dump.""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('file_num', type=int, help = """dump number""")
    parser.add_argument('--end_file', type=int, required=False,
                        default=-1, help = """dump number
                        of end file in slice extraction.""")
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
    parser.add_argument('--stride', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in dump number
                        only relevant if --end_file is supplied.""")
    parser.add_argument('--r_max', type=float, default=0.,
                        required=False,
                        help = """radial extent in Rg (default: R_max)""")
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
                        should be included in slice extraction.""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="""boolean to determine if two-temperature
                        physics should be included in slice extraction.""")
    parser.add_argument('--RIAF', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="""boolean to determine if RIAF opacities
                        should be used for synchrotron
                        (applies to older data sets).""")
    parser.add_argument('--cooling', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="""boolean to determine if cooling rates
                        (qie, qIC, qemi) should be included.""")
    parser.add_argument('--vargamma', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="""boolean to determine if variable gamma
                        should be used for ion/electron energy/pressure.""")
    parser.add_argument('--MBH', type=float, default=10.,
                        required=False,
                        help = """black hole mass in Msun (default: 10)""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if slice extraction
                        should be done on reduced data.""")
    parser.add_argument('--restart', action='store_true', required=False,
                        help="""boolean set to determine if slice
                        integration should continue from last slice_num""")
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="""boolean to determine if existing slice file
                        should be overwritten. Only relevant if --end_file
                        is supplied as a command-line argument.""")
    parser.add_argument('--shared', action='store_true', required=False,
                        help="""boolean set to determine if proj-shared
                        should be used as the location of the data directory""")
    parser.add_argument('--title', type=str, required=False, default=None,
                        help="""title of simulation used for Averages directory
                        if different from the simulation title in proj-shared""")
    args = parser.parse_args()
    main(**vars(args))