import sys
import os
import re
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
import glob
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
    sim          = kwargs.get('sim')
    file_num     = int(kwargs.get('file_num'))
    end_file     = int(kwargs.get('end_file'))
    nslices      = kwargs.get('nslices')
    lowres1      = kwargs.get('lowres1')
    lowres2      = kwargs.get('lowres2')
    lowres3      = kwargs.get('lowres3')
    stride       = kwargs.get('stride')
    r_max        = kwargs.get('r_max')
    reduced      = kwargs.get('reduced')
    restart      = kwargs.get('restart')
    overwrite    = kwargs.get('overwrite')
    shared       = kwargs.get('shared')
    title        = kwargs.get('title')
    radiation    = kwargs.get('radiation')
    two_temp     = kwargs.get('two_temp')
    vargamma     = kwargs.get('vargamma')
    MBH          = np.float32(kwargs.get('MBH'))
    time_average = kwargs.get('time_average')
    use_file     = kwargs.get('use_file')
    fstart       = kwargs.get('fstart')
    fend         = kwargs.get('fend')
    by_block     = kwargs.get('by_block')
    blocks_per_batch = kwargs.get('blocks_per_batch')
    dedim        = kwargs.get('dedim')

    single_file = True if end_file == -1 else False

    # Averaging modes: 
    # time_average: sliding time window over ndumps = time_average
    # fstart/fend : starting file (fstart) to ending file (fend)
    # use_file    : use a precomputed averaged file for mean quantities
    modes = []
    if (time_average != 1):          modes.append('--time_average')
    if (fstart != -1 or fend != -1): modes.append('--fstart/--fend')
    if (use_file):                   modes.append('--use_file')
    if (len(modes) > 1):
        raise ValueError("Only one averaging mode may be specified at a "
                          f"time; got: {', '.join(modes)}.")
    if (fstart == -1 and fend != -1):
        raise ValueError("Do not specify fend without setting fstart.")
    fixed_files = (fstart != -1 and fend != -1)

    # Define constants
    cgs = True if (two_temp or radiation) else False
    if (two_temp or radiation):
        Msun = np.float32(hr.CONSTANTS['Msun'])
        c    = np.float32(hr.CONSTANTS['c'])
        G    = np.float32(hr.CONSTANTS['G'])
        Rg   = G * MBH * Msun / (c * c)
    else:
        c, G, Rg = ONE, ONE, ONE

    physics_args = {
        'MBH'      : MBH,
        'Rg'       : Rg,
        'c'        : c,
        'vargamma' : vargamma,
        'dedim'    : dedim
    }

    # Ensure CONSTANTS are 32-bit floats
    CONSTANTS = {key : np.float32(value) for (key, value) in hr.CONSTANTS.items()}

    # Determine data extraction and save directories
    data_dir     = cv.get_data_dir(sim, reduced, shared)
    save_dir     = cv.get_save_dir(sim if title == None else title,
                                   'Averages', 'Stresses', reduced=reduced)
    averages_dir = cv.user_path + f'/Averages/{sim if title==None else title}'

    nzeros = len(str(len(glob.glob(os.path.join(data_dir, 'dumps*')))))

    # Quantities (quants) for slice averaging
    keys = [
        'TRphi_Rey_mean', 'TRphi_Rey_turb', 'TRphi_Max_mean', 'TRphi_Max_turb',
        'Tzphi_Rey_mean', 'Tzphi_Rey_turb', 'Tzphi_Max_mean', 'Tzphi_Max_turb',
        'Trphi_Rey_mean', 'Trphi_Rey_turb', 'Trphi_Max_mean', 'Trphi_Max_turb', 'w',
        'wdurdu_phi', 'urdwdu_phi', 'dwuru_phi', 'dwduru_phi', 'dwdurdu_phi'
    ]
    # Time-averages needed for slice averaging
    mean_keys = [
        'wuR',  'wuz', 'wu_phi', 'wur', 'wuth',
        'bR',   'bz',  'b_phi',  'br',  'bth', 'w', 'Vol'
    ]
    if (radiation):
        keys += ['Rrphi', 'RRphi', 'Rzphi']

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

    last_file = get_last_file_num(data_dir)
    if (fstart != -1 and fend == -1):
        fend = last_file

    averaging_args = {
        'use_file'     : use_file,
        'time_average' : time_average,
        'fstart'       : fstart,
        'fend'         : fend,
        'fixed_files'  : fixed_files,
        'averages_dir' : averages_dir,
        'mean_keys'    : mean_keys,
        'reduced'      : reduced,
        'last_file'    : last_file,
        'dedim'        : dedim
    }

    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    size = comm.Get_size()
    rank = comm.Get_rank()
    MPI_info = {'comm': comm, 'size': size, 'rank': rank}

    # Calibrate the total number of batch size that can be analyzed on a node.
    if by_block and blocks_per_batch < 0:
        Slice_p = cv.Slices()
        mean_p  = get_time_averaged_mean({}, file_num, Slice_p, averaging_args)
        def probe_batch(n_blocks):
            print(f"[CALIBRATION] probe {n_blocks} blocks", flush=True)
            process_batch(data_dir=data_dir, file_num=file_num,
                          read_args=read_args, CONSTANTS=CONSTANTS, cgs=cgs,
                          keys=keys, mean=mean_p,
                          mean_keys=averaging_args['mean_keys'],
                          Slice=Slice_p, last_slice=False,
                          start_block=0, end_block=n_blocks,
                          radiation=radiation, two_temp=two_temp,
                          **physics_args)

        budget_info = bb.plan_block_batches(probe_batch, comm)
        blocks_per_batch = budget_info['n_batch']
        if rank == 0:
            print(bb.summarize(budget_info), flush=True)
            if budget_info.get('fallback'):
                print(f"NOTE: {budget_info['fallback']}", flush=True)
            print("[CALIBRATION] done.", flush=True)
    elif by_block and rank == 0:
        print(f"Using --blocks_per_batch={blocks_per_batch} "
              f"(calibration overridden).", flush=True)

    if single_file:
        init_args = {
            'data_dir' : data_dir,
            'file_num' : file_num,
            'keys'     : keys,
            'save_dir' : save_dir,
            'nslices'  : nslices,
            'restart'  : restart,
            'nzro'     : nzeros,
            'file_type': 'Stresses'
        }

        stresses, nphi, iphi_stp = cv.init_slices(**init_args, **read_args)
        if not restart:
            stresses['done'] = False

        run_args = {
            'stresses'      : stresses,
            'MPI_info'      : MPI_info,
            'read_args'     : read_args,
            'CONSTANTS'     : CONSTANTS,
            'cgs'           : cgs,
            'averaging_args': averaging_args
        }

        if by_block:
            run_parallel_by_block(**run_args, nb=int(stresses['nb']),
                                  blk_stp=blocks_per_batch, **init_args,
                                  radiation=radiation, two_temp=two_temp,
                                  **physics_args)
        else:
            run_parallel_in_phi(**run_args, nphi=nphi, iphi_stp=iphi_stp,
                                **init_args, radiation=radiation,
                                two_temp=two_temp, **physics_args)

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
            sim_title      = sim if title == None else title
            averages_dir2  = cv.user_path + f'/Averages/{sim_title}'
            existing_files = cv.get_file_list_local(file_list_local, 1, 0,
                averages_dir2, file_type='Stresses', reduced=reduced)
            existing_set    = set(existing_files)
            file_list_local = [f for f in file_list_local if f not in existing_set]

        init_args = {
            'data_dir' : data_dir,
            'keys'     : keys,
            'save_dir' : save_dir,
            'nslices'  : nslices,
            'restart'  : restart,
            'nzro'     : nzeros,
            'file_type': 'Stresses',
            'by_block' : by_block,
            'blk_stp'  : blocks_per_batch
        }

        run_args = {
            'file_list_local' : file_list_local,
            'init_args'       : init_args,
            'read_args'       : read_args,
            'MPI_info'        : MPI_info,
            'keys'            : keys,
            'CONSTANTS'       : CONSTANTS,
            'cgs'             : cgs,
            'averaging_args'  : averaging_args
        }

        run_parallel_in_time(**run_args, radiation=radiation, two_temp=two_temp,
                             vargamma=vargamma, MBH=MBH, c=c, Rg=Rg, dedim=dedim)

def get_time_averaged_mean(state, file_num, Slice, averaging_args):
    """Load mean slices."""
    use_file = averaging_args['use_file']

    if use_file:
        if 'mean' not in state:
            res_str  = ('Full_Resolution' if not averaging_args['reduced']
                        else 'Reduced_Resolution')
            mean_dir = (f"{averaging_args['averages_dir']}/{res_str}"
                        f"/Slices_Time_Averaged")
            state['mean'] = dict(np.load(f'{mean_dir}/Slices_Time_Averaged.npz'))
        return state['mean']

    time_average = averaging_args['time_average']
    fixed_files  = averaging_args['fixed_files']
    last_file    = averaging_args['last_file']

    file_info = {
        'averages_dir' : averaging_args['averages_dir'],
        'file_number'  : -1,
        'start_file'   : max(file_num - int(time_average/2), 0),
        'end_file'     : min(file_num + int(time_average/2), last_file),
        'reduced'      : averaging_args['reduced']
    }
    if (fixed_files):
        file_info['start_file'] = averaging_args['fstart']
        file_info['end_file']   = averaging_args['fend']

    if 'fnums' not in state:
        fnums = Slice.get_fnums(**file_info)
        vols  = dict.fromkeys(averaging_args['mean_keys'])
        for key in vols.keys():
            vols[key] = True
        vols['Vol'] = False
        slice_info = {
            'averages_dir' : averaging_args['averages_dir'],
            'fnums'        : fnums,
            'keys'         : averaging_args['mean_keys'],
            'vols'         : vols,
            'reduced'      : averaging_args['reduced'],
            'dedim'        : averaging_args['dedim']
        }
        state['mean']       = Slice.load_slices_time_averaged(**slice_info)
        state['fnums']      = fnums
        state['slice_info'] = slice_info
    else:
        slice_info               = state['slice_info']
        slice_info['slices']     = state['mean']
        new_fnums                = Slice.get_fnums(**file_info)
        slice_info['fnums']      = new_fnums
        slice_info['fnums_old']  = state['fnums']
        state['mean']  = Slice.update_slices(**slice_info)
        state['fnums'] = slice_info['fnums']

    return state['mean']

def run_parallel_by_block(stresses, nb, blk_stp, MPI_info, read_args,
                          keys, CONSTANTS, cgs, data_dir, file_num, save_dir,
                          nslices, restart, nzro, averaging_args, file_type,
                          **kwargs):
    """Iterate through data in batches based on calibrated block step."""
    rank, size = MPI_info['rank'], MPI_info['size']
    nr, nth    = stresses[keys[0]].shape
    Slice      = cv.Slices()
    state      = {}
    mean       = get_time_averaged_mean(state, file_num, Slice, averaging_args)

    start_at = cv.resume_offset(stresses, True) if restart else 0
    nbatch   = len(range(start_at, nb, blk_stp))

    if start_at >= nb:
        if (rank == 0):
            print(f'File {file_num} already complete through block '
                  f'{start_at:d} of {nb:d}; nothing to do.', flush=True)
        return

    for batch_num, batch_sel in cv.iter_block_batches(nb, blk_stp,
                                                      start=start_at):
        stresses['saved'] = False
        rank_sel = cv.split_batch(batch_sel, rank, size)

        # Staggered start to take pressure off of filesystem
        timeit.sleep(0.05 * rank + np.random.rand())

        local_buffer = np.zeros((len(keys), nr, nth), dtype=np.float64)
        meta = None
        if rank_sel['end_block'] > rank_sel['start_block']:
            batch_args = {
                'data_dir'   : data_dir,
                'file_num'   : file_num,
                'read_args'  : read_args,
                'CONSTANTS'  : CONSTANTS,
                'cgs'        : cgs,
                'keys'       : keys,
                'mean'       : mean,
                'mean_keys'  : averaging_args['mean_keys'],
                'Slice'      : Slice,
                'last_slice' : (batch_num == nbatch - 1),
                'nr_out'     : nr
            }
            print(f"[EXTRACTION] rank {rank:d} batch {batch_num + 1:d}/"
                  f"{nbatch:d} reading blocks "
                  f"[{rank_sel['start_block']:d}, {rank_sel['end_block']:d}) "
                  f"of {nb:d}", flush=True)
            stress_loc, meta = process_batch(**batch_args, **rank_sel, **kwargs)
            for i, key in enumerate(keys):
                if key in stress_loc:
                    local_buffer[i, :, :] = stress_loc[key][:nr, :nth]
            del stress_loc
        else:
            print(f"[EXTRACTION] rank {rank:d} batch {batch_num + 1:d}/"
                  f"{nbatch:d} has no blocks; contributing zeros.", flush=True)

        global_buffer = None
        if (rank == 0):
            stresses['r']     = meta['r']
            stresses['theta'] = meta['theta']
            stresses['phi']   = meta['phi']
            stresses['dx1']   = meta['dx1']
            stresses['dx2']   = meta['dx2']
            stresses['dx3']   = meta['dx3']
            stresses['Vol']   = meta['Vol']
            stresses['rho0']  = meta['rho0']
            global_buffer     = np.zeros_like(local_buffer)

        MPI_info['comm'].Reduce(local_buffer, global_buffer, op=MPI.SUM, root=0)
        del local_buffer

        if (rank == 0):
            for i, key in enumerate(keys):
                stresses[key] += global_buffer[i, :, :]

            cv.record_progress(stresses, batch_sel, True, nb, blk_stp=blk_stp)
            if stresses['done']:
                # Re-dimensionalize phi-averages only once, at completion.
                for key in keys:
                    stresses[key] *= (stresses['rho0'] * kwargs.get('c') ** 2)
            cv.save_checkpoint(save_dir, file_num, nzro, stresses)

            print(f'Finished with batch {batch_num + 1:d} of {nbatch:d} '
                  f'for file number {file_num}', flush=True)
        del global_buffer
        MPI_info['comm'].Barrier()
        gc.collect()

    print(f'Done with all stresses for file number {file_num}.', flush=True)

def run_parallel_in_phi(stresses, nphi, iphi_stp, MPI_info, read_args,
                        keys, CONSTANTS, cgs, data_dir, file_num, save_dir,
                        nslices, restart, nzro, averaging_args, **kwargs):
    """Parallel in phi: each rank receives phi-slice of full domain."""
    Slice = cv.Slices()
    state = {}
    mean  = get_time_averaged_mean(state, file_num, Slice, averaging_args)

    total_slices = nphi // iphi_stp
    start_at     = cv.resume_offset(stresses, False) if restart else 0

    if start_at >= nphi:
        if (MPI_info['rank'] == 0):
            print(f'File {file_num} already complete through k={start_at:d} '
                  f'of {nphi:d}; nothing to do.', flush=True)
        return

    for slice_num, iphi in enumerate(range(start_at, nphi, iphi_stp)):
        stresses['saved'] = False
        k_min = int(iphi)
        k_max = int(iphi + iphi_stp)

        delta_k   = (k_max - k_min) / MPI_info['size']
        k_min_loc = int(k_min + delta_k * MPI_info['rank'])
        k_max_loc = int(k_min + delta_k * (MPI_info['rank'] + 1))

        # Staggered start to take pressure off of filesystem
        timeit.sleep(0.05 * MPI_info['rank'] + np.random.rand())

        batch_args = {
            'data_dir'   : data_dir,
            'file_num'   : file_num,
            'k_min'      : k_min_loc,
            'k_max'      : k_max_loc,
            'read_args'  : read_args,
            'CONSTANTS'  : CONSTANTS,
            'cgs'        : cgs,
            'keys'       : keys,
            'mean'       : mean,
            'mean_keys'  : averaging_args['mean_keys'],
            'Slice'      : Slice,
            'last_slice' : (slice_num == total_slices - 1)
        }

        stress_loc, meta = process_batch(**batch_args, **kwargs)

        nr, nth = len(meta['r']), len(meta['theta'])
        local_buffer = np.zeros((len(keys), nr, nth), dtype=np.float64)
        for i, key in enumerate(keys):
            if key in stress_loc:
                local_buffer[i, :, :] = stress_loc[key]
        global_buffer = None

        print(f"Rank {MPI_info['rank']:d} done analyzing range of k: " +
            f"({k_min_loc:d}, {k_max_loc:d}).", flush=True)

        if (MPI_info['rank'] == 0):
            stresses['r']     = meta['r']
            stresses['theta'] = meta['theta']
            stresses['phi']   = meta['phi']
            stresses['dx1']   = meta['dx1']
            stresses['dx2']   = meta['dx2']
            stresses['dx3']   = meta['dx3']
            stresses['Vol']   = meta['Vol']
            stresses['rho0']  = meta['rho0']
            global_buffer = np.zeros_like(local_buffer)

        MPI_info['comm'].Reduce(local_buffer, global_buffer, op=MPI.SUM, root=0)
        del local_buffer

        if (MPI_info['rank'] == 0):
            for i, key in enumerate(keys):
                stresses[key] += global_buffer[i, :, :]

            cv.record_progress(stresses, {'k_max': k_max}, False, nphi)
            stresses['slice_num_restart'] = slice_num + 1
            if stresses['done']:
                # Re-dimensionalize phi-averages before saving
                for key in keys:
                    stresses[key] *= (stresses['rho0'] * kwargs.get('c') ** 2)
            cv.save_checkpoint(save_dir, file_num, nzro, stresses)

            snum = int(slice_num + 1)
            print(f'Finished with slice {snum:d} of {nslices:d}', flush=True)

        MPI_info['comm'].Barrier()
        gc.collect()

    print(f'Done with all stresses for file number {file_num}.', flush=True)

def run_parallel_in_time(file_list_local, init_args, read_args, MPI_info,
                         keys, CONSTANTS, cgs, averaging_args, **kwargs):
    """Parallelization across dump number. Each rank receives a single dump."""
    data_dir = init_args['data_dir']
    save_dir = init_args['save_dir']
    Slice    = cv.Slices()
    state    = {}

    for file_num in file_list_local:
        init_args['file_num'] = file_num
        stresses, nphi, iphi_stp = cv.init_slices(**init_args, **read_args)
        if not init_args['restart']:
            stresses['done'] = False

        mean = get_time_averaged_mean(state, file_num, Slice, averaging_args)

        by_block = init_args.get('by_block', False)
        nb       = int(stresses['nb'])
        blk_stp  = init_args.get('blk_stp') or nb    # None/-1 -> one batch
        nbatch   = (len(range(0, nb, blk_stp)) if by_block
                    else nphi // iphi_stp)

        restart  = init_args['restart']
        start_at = cv.resume_offset(stresses, by_block) if restart else 0
        if start_at >= (nb if by_block else nphi):
            print(f'File {file_num} already complete; skipping.', flush=True)
            continue

        batches = (cv.iter_block_batches(nb, blk_stp, start=start_at)
                   if by_block else
                   cv.iter_phi_batches(nphi, iphi_stp, start=start_at))
        for slice_num, sel in batches:
            stresses['saved'] = False

            # Staggered start to take pressure off of filesystem
            timeit.sleep(0.05 * MPI_info['rank'] + np.random.rand())

            batch_args = {
                'data_dir'   : data_dir,
                'file_num'   : file_num,
                'read_args'  : read_args,
                'CONSTANTS'  : CONSTANTS,
                'cgs'        : cgs,
                'keys'       : keys,
                'mean'       : mean,
                'mean_keys'  : averaging_args['mean_keys'],
                'Slice'      : Slice,
                'last_slice' : (slice_num == nbatch - 1),
                'nr_out'     : stresses[keys[0]].shape[0]
            }
            batch_args.update(sel)

            lo, hi = ((sel['start_block'], sel['end_block']) if by_block
                      else (sel['k_min'], sel['k_max']))
            print(f"[EXTRACTION] rank {MPI_info['rank']:d} file {file_num} "
                  f"batch {slice_num + 1:d}/{nbatch} reading "
                  f"{'blocks' if by_block else 'k'} [{lo:d}, {hi:d})",
                  flush=True)
            stress_loc, meta = process_batch(**batch_args, **kwargs)

            for key in keys:
                if key in stress_loc:
                    stresses[key] += stress_loc[key]

            stresses['r']     = meta['r']
            stresses['theta'] = meta['theta']
            stresses['phi']   = meta['phi']
            stresses['dx1']   = meta['dx1']
            stresses['dx2']   = meta['dx2']
            stresses['dx3']   = meta['dx3']
            stresses['Vol']   = meta['Vol']
            stresses['rho0']  = meta['rho0']
            cv.record_progress(stresses, sel, by_block,
                               nb if by_block else nphi,
                               blk_stp=blk_stp if by_block else None)

            print(f"Finished with batch {slice_num + 1:d} of {nbatch} "
                  f'for file {file_num}', flush=True)

        if stresses['done']:
            # Re-dimensionalize phi-averages before saving
            for key in keys:
                stresses[key] *= (stresses['rho0'] * kwargs.get('c') ** 2)

        stresses['slice_num_restart'] = slice_num + 1   # legacy, not read
        cv.save_checkpoint(save_dir, file_num, init_args['nzro'], stresses)
        gc.collect()

        print(f'Done with all slices for file number {file_num}.', flush=True)

def process_batch(data_dir, file_num, read_args,
                  CONSTANTS, cgs, keys, mean, mean_keys, Slice,
                  last_slice,
                  k_min=None, k_max=None, start_block=None, end_block=None,
                  **kwargs):
    MBH       = kwargs.get('MBH')
    Rg        = kwargs.get('Rg')
    c         = kwargs.get('c')
    radiation = kwargs.get('radiation')
    two_temp  = kwargs.get('two_temp')
    vargamma  = kwargs.get('vargamma')
    dedim     = kwargs.get('dedim')

    # A batch is either a phi slice (k_min..k_max) or a contiguous range of
    # Local Block IDs (start_block..end_block); exactly one pair is given.
    #
    # read_args is COPIED, not mutated. The caller builds one read_args and
    # reuses it for every batch, so writing into it leaves the previous
    # batch's selection keys behind. That was harmless while only k_min/k_max
    # were ever written, but with two modes a stale by_block/start_block
    # would silently change what the next batch reads.
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

    # Read grid and data
    H.read_grid(file_num, direct=True)
    H.read_prims(file_num, skip_read_params=True)

    if 'tr1' in H.data.keys(): H.data['tr1'] = None
    if 'tr2' in H.data.keys(): H.data['tr2'] = None
    if 'dissipation' in H.data.keys(): H.data['dissipation'] = None
    if 'rad_src'     in H.data.keys(): H.data['rad_src']     = None

    # Deallocate the AMR block array. It is (nmax, 200) int32 -- 334 MB on
    # THIN_RAD_MAD_HIRES_TRACER -- and sized by nmax, every block the octree
    # COULD create, not the nb that exist. read_block builds it and
    # read_params reads six columns out of it; nothing else in the toolkit
    # touches it. amr_info now holds copies of those columns, so this
    # actually frees the memory rather than leaving views pinning it.
    # Safe here because this Read is not used for another read_params.
    H.block = None

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

    # Calculate sigma (3D)
    b  = Field.calc_b(**H.data, **H.grid, cpp=True)
    b2 = Field.calc_b2(b=b, **H.data, **H.grid, cpp=True)

    # Calculate gas pressure
    if (two_temp and vargamma):
        pgas = Fluid.calc_pgas(**H.data)
    else:
        pgas      = hr.Tools.zeroNd(H.data['u_gas'].shape, 1)
        gamma_adi = np.float32(5. / 3.)
        np.multiply(H.data['u_gas'], (gamma_adi - ONE), out=pgas)

    # Radius and theta
    r  = H.grid['r']
    th = H.grid['theta']
    sinth = np.sin(th) * np.ones_like(r)

    rho0 = H.params['density_scale'] if cgs else ONE
    B0   = Field.B0                  if cgs else ONE

    # Downscale the mean array to the current dump's resolution if needed.
    # Independent of dedim state: idempotent via the shape check, so it's
    # safe to run every batch regardless of whether units were removed yet.
    if by_block:
        # r/th are one block's axes here; the mean lives on the dump's
        # (r, theta) grid, which is what unfold2D writes into.
        bs1_lr  = int(H.params['bs1']) // H.lowres1
        bs2_lr  = int(H.params['bs2']) // H.lowres2
        nr_cur  = int(H.params['nb1'] * bs1_lr * 2**H.amr['max_levelx1'])
        nth_cur = int(H.params['nb2'] * bs2_lr * 2**H.amr['max_levelx2'])
        if kwargs.get('nr_out') is not None:
            nr_cur = min(nr_cur, int(kwargs['nr_out']))
    else:
        nr_cur  = len(r[0, :, 0, 0])
        nth_cur = len(th[0, 0, :, 0])
    if mean['w'].shape != (nr_cur, nth_cur):
        ds_args = {'min_nr': nr_cur, 'min_nth': nth_cur}
        for key in mean_keys:
            mean[key] = Slice.downscale(mean[key], **ds_args)

    # De-dimensionalize (dedim) the mean array, unless load_slices_time_averaged/
    # update_slices already removed units per-frame (marked via mean['dedim'])
    # because --dedim was set.
    if 'dedim' not in mean:
        mean['w'] *= (ONE / (rho0 * c * c))
        for wu_key in ['wuR', 'wuz', 'wu_phi', 'wur', 'wuth']:
            mean[wu_key] *= (ONE / (rho0 * c * c * c))
        for b_key in ['bR', 'bz', 'b_phi', 'br', 'bth']:
            mean[b_key] *= (ONE / B0)
        mean['dedim'] = True

    ones = np.ones_like(H.data['rho']) # (nb, nr, nth, nphi)
    if (cgs):
        w = H.data['rho'] * c * c + pgas + H.data['u_gas'] + b2/FOURPI
    else:
        w = H.data['rho'] + pgas + H.data['u_gas'] + b2
    if (radiation):
        w = (H.data['rho'] * c * c + pgas + H.data['u_gas']
                + b2/FOURPI + FOUR_THIRDS * H.data['u_rad'])

    # De-dimensionalize for precision
    w   *= (ONE / (rho0 * c * c))
    b   *= (ONE / B0)
    H.data['uu'] *= (ONE / c)

    # Free memory
    H.data['rho']   = None
    H.data['u_gas'] = None
    H.data['B']     = None
    if (two_temp):
        H.data['Ti'] = None
        H.data['Te'] = None
    del b2, pgas

    # A global (r, theta) mean cannot broadcast against per-block 3D data,
    # so under by_block it is folded onto each block's own cells; in phi mode
    # the grid is already global and a reshape is enough. Written to a
    # separate dict because `mean` is cached across batches by the caller.
    if by_block:
        mean_b = {k: Acc.fold2D(mean[k]) for k in mean_keys}
    else:
        mean_b = {k: mean[k][None, :, :, None] for k in mean_keys}

    stress_loc = {}

    # Enthalpy
    stress_loc['w'] = Acc.ph_sum(w)[0]

    # Reynolds stresses
    uu_cov  = ana.Tools.contract_vector_cpp(H.data['uu'], H.grid['gcov'])
    # r-phi (u^r u_phi_sph / c^2)
    uusph   = Coords.int_to_ortho_sph_vec(H.data['uu'], cpp=True)
    uu_sph  = Coords.int_to_ortho_sph_one_form(uu_cov,  cpp=True)
    uru_phi = uusph[1] * uu_sph[3]
    ur      = uusph[1]
    u_phi   = uu_sph[3]
    del uusph, uu_sph
    # R-phi (u^R u_phi_cyl / c^2)
    uucyl   = Coords.int_to_cyl_vec(H.data['uu'], cpp=True)
    uu_cyl  = Coords.int_to_cyl_one_form(uu_cov,  cpp=True)
    uRu_phi = uucyl[1] * uu_cyl[2]
    # z-phi (u^z u_phi_cyl / c^2)
    uzu_phi = uucyl[3] * uu_cyl[2]
    del uucyl, uu_cyl, uu_cov
    H.data['uu'] = None

    # r-phi Reynolds stresses
    # wu_phi is in cylindrical coordinates, convert phi_cyl -> phi_sph
    mean_b['wu_phi_sph'] = sinth * mean_b['wu_phi']
    mean_wuru_phi = (mean_b['wur'] * mean_b['wu_phi_sph'] / mean_b['w']) * ones
    stress_loc['Trphi_Rey_mean'] = Acc.ph_sum(mean_wuru_phi)[0]
    # turb = total - mean
    stress_loc['Trphi_Rey_turb'] = Acc.ph_sum(w * uru_phi - mean_wuru_phi)[0]
    del mean_wuru_phi

    # Decomposition of r-phi Reynolds stress
    mean_wur    = mean_b['wur']    * ones
    mean_wu_phi = mean_b['wu_phi'] * ones
    mean_w      = mean_b['w']      * ones
    mean_ur     = mean_wur / mean_w
    mean_u_phi  = mean_wu_phi / mean_w
    del mean_wur, mean_wu_phi
    dur    = ur - mean_ur
    du_phi = u_phi - mean_u_phi
    dw     = w - mean_w

    stress_loc['wdurdu_phi']  = Acc.ph_sum(mean_w * dur * du_phi)[0]
    stress_loc['urdwdu_phi']  = Acc.ph_sum(mean_ur* dw  * du_phi)[0]
    stress_loc['dwuru_phi']   = Acc.ph_sum(dw * mean_ur  * mean_u_phi)[0]
    stress_loc['dwduru_phi']  = Acc.ph_sum(dw     * dur * mean_u_phi)[0]
    stress_loc['dwdurdu_phi'] = Acc.ph_sum(dw     * dur * du_phi)[0]
    del mean_ur, mean_u_phi, dur, du_phi, mean_w, dw, ur, u_phi

    # R-phi Reynolds stresses
    mean_wuRu_phi = (mean_b['wuR'] * mean_b['wu_phi'] / mean_b['w']) * ones
    stress_loc['TRphi_Rey_mean'] = Acc.ph_sum(mean_wuRu_phi)[0]
    stress_loc['TRphi_Rey_turb'] = Acc.ph_sum(w * uRu_phi - mean_wuRu_phi)[0]
    del mean_wuRu_phi

    # z-phi Reynolds stresses
    mean_wuzu_phi = (mean_b['wuz'] * mean_b['wu_phi'] / mean_b['w']) * ones
    stress_loc['Tzphi_Rey_mean'] = Acc.ph_sum(mean_wuzu_phi)[0]
    stress_loc['Tzphi_Rey_turb'] = Acc.ph_sum(w * uzu_phi - mean_wuzu_phi)[0]
    del mean_wuzu_phi, w

    # Maxwell stresses
    b_cov   = ana.Tools.contract_vector_cpp(b, H.grid['gcov'])
    # r-phi (b^r b_phi_sph)
    bsph    = Coords.int_to_ortho_sph_vec(b,          cpp=True)
    b_sph   = Coords.int_to_ortho_sph_one_form(b_cov, cpp=True)
    brb_phi = bsph[1] * b_sph[3]
    del bsph, b_sph
    # R-phi (b^R b_phi_cyl)
    bcyl    = Coords.int_to_cyl_vec(b,           cpp=True)
    b_cyl   = Coords.int_to_cyl_one_form(b_cov,  cpp=True)
    bRb_phi = bcyl[1] * b_cyl[2]
    # z-phi (b^z b_phi_cyl)
    bzb_phi = bcyl[3] * b_cyl[2]
    del bcyl, b_cyl, b_cov

    # r-phi Maxwell stresses
    # b_phi is in cylindrical coordinates, convert phi_cyl -> phi_sph
    mean_b['b_phi_sph'] = sinth * mean_b['b_phi']
    mean_brb_phi = (mean_b['br'] * mean_b['b_phi_sph']) * ones
    stress_loc['Trphi_Max_mean'] = Acc.ph_sum(-mean_brb_phi)[0]
    stress_loc['Trphi_Max_turb'] = Acc.ph_sum(-brb_phi - (-mean_brb_phi))[0]
    del mean_brb_phi, brb_phi

    # R-phi Maxwell stresses
    mean_bRb_phi = (mean_b['bR'] * mean_b['b_phi']) * ones
    stress_loc['TRphi_Max_mean'] = Acc.ph_sum(-mean_bRb_phi)[0]
    stress_loc['TRphi_Max_turb'] = Acc.ph_sum(-bRb_phi - (-mean_bRb_phi))[0]
    del mean_bRb_phi, bRb_phi

    # z-phi Maxwell stresses
    mean_bzb_phi = (mean_b['bz'] * mean_b['b_phi']) * ones
    stress_loc['Tzphi_Max_mean'] = Acc.ph_sum(-mean_bzb_phi)[0]
    stress_loc['Tzphi_Max_turb'] = Acc.ph_sum(-bzb_phi - (-mean_bzb_phi))[0]
    del mean_bzb_phi, bzb_phi

    # Radiation Diffusion stresses
    if (radiation):
        stress_energy_RAD_13 = Radi.calc_TRAD(**H.data, **H.grid,
            upper_lower=True, index=[1, 3], cpp=True)
        stress_energy_RAD_23 = Radi.calc_TRAD(**H.data, **H.grid,
            upper_lower=True, index=[2, 3], cpp=True)
        R_vec    = np.zeros_like(H.data['uu_rad'])
        R_vec[1] = stress_energy_RAD_13
        R_vec[2] = stress_energy_RAD_23
        R_cyl    = Coords.int_to_cyl_vec(      R_vec, cpp=True)
        R_sph    = Coords.int_to_ortho_sph_vec(R_vec, cpp=True)

        H.data['uu_rad'] = None
        # de-dimensionalize
        R_cyl           *= (ONE / (rho0 * c * c))
        R_sph           *= (ONE / (rho0 * c * c))
        H.data['u_rad'] *= (ONE / (rho0 * c * c))

        # r-phi Radiation Diffusion stress
        Rrphi = R_sph[1] - FOUR_THIRDS * (H.data['u_rad'] * uru_phi)
        stress_loc['Rrphi'] = Acc.ph_sum(Rrphi)[0]
        del stress_energy_RAD_13, stress_energy_RAD_23
        del R_vec, R_sph, Rrphi, uru_phi

        # R-phi Radiation Diffusion stress
        RRphi = R_cyl[1] - FOUR_THIRDS * (H.data['u_rad'] * uRu_phi)
        stress_loc['RRphi'] = Acc.ph_sum(RRphi)[0]
        del RRphi, uRu_phi

        # z-phi Radiation Diffusion stress
        Rzphi = R_cyl[3] - FOUR_THIRDS * (H.data['u_rad'] * uzu_phi)
        stress_loc['Rzphi'] = Acc.ph_sum(Rzphi)[0]
        del R_cyl, Rzphi, uzu_phi

    if by_block:
        # global_coords1D returns exp(x1) directly, already geometric, while
        # grid['r'] carries the cgs_grid scaling that the /Rg undoes.
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
        'Vol'   : mean['Vol'],
        'rho0'  : H.params['density_scale']
    }

    # Deallocate class instances
    del Acc, Coords, Fluid, Field, Radi

    if (last_slice and not dedim):
        # Re-dimensionalize mean arrays so the state carried into the
        # next file's sliding-window update (or, in single-file mode,
        # left in a physically-meaningful state) is properly scaled.
        # Skipped when --dedim is set: in that mode load_slices_time_averaged/
        # update_slices already keep `mean` dimensionless between files (each
        # applies remove_units to the raw per-frame data it merges in), so
        # re-dimensionalizing here would desync it from what update_slices
        # expects on the next call.
        mean['w'] *= (rho0 * c * c)
        for wu_key in ['wuR', 'wuz', 'wu_phi', 'wur', 'wuth']:
            mean[wu_key] *= (rho0 * c * c * c)
        for b_key in ['bR', 'bz', 'b_phi', 'br', 'bth']:
            mean[b_key] *= B0
        del mean['dedim']

    return stress_loc, coords_data

def get_last_file_num(data_dir):
    max_num = -1
    pattern = re.compile(r'^dumps(\d+)')

    with os.scandir(data_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                match = pattern.search(entry.name)
                if match:
                    # Convert the captured group (the digits) to int
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num

    return max_num if max_num != -1 else None

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Calculate 2D poloidal slices of
                        stress. If --end_file is supplied, parallelization
                        is done in time (each dump is processed by a
                        separate MPI rank). Otherwise, ranks parallelize
                        phi slices of a single dump.""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('file_num', type=int, help = """dump number""")
    parser.add_argument('--end_file', type=int, required=False,
                        default=-1, help = """dump number
                        of end file in stresses extraction.""")
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
                        should be included in profile extraction.""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="""boolean to determine if two-temperature
                        physics should be included in profile extraction.""")
    parser.add_argument('--vargamma', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="""boolean to determine if variable gamma
                        should be used for ion/electron energy/pressure.""")
    parser.add_argument('--waves', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="""Alfven wave analysis should be performed
                        in addition to standard stress analysis. Currently
                        unused -- preserved from the pre-merge scripts,
                        where it was already a no-op.""")
    parser.add_argument('--MBH', type=float, default=10.,
                        required=False,
                        help = """black hole mass in Msun (default: 10)""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if profile extraction
                        should be done on reduced data.""")
    parser.add_argument('--restart', action='store_true', required=False,
                        help="""boolean set to determine if profile
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
    parser.add_argument('--time_average', type=int, default=1,
                        required=False,
                        help = """number of slices (enumerated by dump number)
                        to average in time when computing mean and
                        turbulent/fluctuating stresses""")
    parser.add_argument('--fstart', type=int, default=-1,
                        required=False,
                        help = """starting dump number for
                        computing mean stresses""")
    parser.add_argument('--fend', type=int, default=-1,
                        required=False,
                        help = """ending dump number for
                        computing mean stresses""")
    parser.add_argument('--use_file', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="""boolean to determine if mean arrays should
                        be loaded in from Averages/../Slices_Time_Averaged,
                        in either single-file or time-parallel mode.""")
    parser.add_argument('--dedim', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="""boolean to determine if the time-averaging of
                        mean slices (load_slices_time_averaged/update_slices)
                        is itself done on de-dimensionalized (rho0/c/B0
                        divided-out) per-frame data, rather than on raw cgs
                        values. Improves numerical precision of the running
                        average; does not change the final (re-dimensionalized)
                        output. Has no effect combined with --use_file, since
                        that path loads a precomputed mean file directly.""")
    args = parser.parse_args()
    main(**vars(args))
