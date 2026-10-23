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

COORD_IDX = ['t', 'x', 'y', 'z']

def process_batch(data_dir, file_num, k_min, k_max, read_args,
                       CONSTANTS, cgs, MBH, radiation, two_temp, vargamma,
                       bulk_mom, c, Rg):
    """Read one phi-range [k_min, k_max) of one dump and compute this
    slice's contribution to every angular-momentum/accretion-rate
    quantity in `profile_keys`.

    Shared by both parallelization modes -- the only difference between
    them is which (file_num, k_min, k_max) triples get assigned to which
    rank, and whether partial results need to be MPI-reduced afterward.

    Returns (local, meta): `local` maps each profile key to its
    partial-sum array for this slice; `meta` holds grid-derived
    bookkeeping (`r`, `dx1`, `rho0`) that's identical across phi-slices
    of the same file.
    """
    read_args = dict(read_args)
    read_args['k_min'] = k_min
    read_args['k_max'] = k_max

    H = hr.Read(data_dir, **read_args)
    H.read_grid(file_num, direct=True)
    H.read_prims(file_num, skip_read_params=True)

    if 'tr1' in H.data.keys(): H.data['tr1'] = None
    if 'tr2' in H.data.keys(): H.data['tr2'] = None
    if 'dissipation' in H.data.keys(): H.data['dissipation'] = None
    if 'rad_src'     in H.data.keys(): H.data['rad_src']     = None

    # Deallocate coordinate transformation array and block object
    H.grid['dxdxp'] = None
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

    class_args = {
        'pars'      : H.params,
        'CONSTANTS' : CONSTANTS, # float32 constants
        'cgs'       : cgs
    }

    # Initialize Accretion class
    Acc = ana.Accretion(**acc_class_args)
    # Initialize Fluid class
    Fluid = ana.Fluid(**class_args)
    # Initialize Field class
    Field = ana.Field(**class_args)
    # Initialize Radiation class
    Radi  = rad.Radiation(**class_args)
    # Initialize Tilt class
    Tilt = coords.Tilt(**acc_class_args)

    # Calculate sigma (3D)
    b  = Field.calc_b(**H.data, **H.grid, cpp=True)
    b2 = Field.calc_b2(b=b, **H.data, **H.grid, cpp=True)
    if (cgs):
        sigma  = (b2 / FOURPI)
        sigma /= (H.data['rho'] * c * c)
    else:
        sigma = b2 / H.data['rho']

    # Calculate gas pressure
    if (two_temp and vargamma):
        pgas = Fluid.calc_pgas(**H.data)
    else:
        pgas      = hr.Tools.zeroNd(H.data['u_gas'].shape, 1)
        gamma_adi = np.float32(H.params['gam'])
        np.multiply(H.data['u_gas'], (gamma_adi - ONE), out=pgas)

    # Calculate u^mu u_nu tensor
    uu_tnsr = ana.Tools.construct_dyadic(H.data['uu'] / c,
        H.grid['gcov'], upper_lower=True, index=[1, 3], cpp=True)

    # Mass and angular momentum accretion/outflow rates
    stress_energy_HD  = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas,
        uu_tnsr=uu_tnsr, upper_lower=True, index=[1, 3], cpp=True)
    stress_energy_EM  = Field.calc_TEM(**H.data, **H.grid, b2=b2, b=b,
        uu_tnsr=uu_tnsr, upper_lower=True, index=[1, 3], cpp=True)
    if (radiation):
        stress_energy_RAD = Radi.calc_TRAD(**H.data, **H.grid,
            upper_lower=True, index=[1, 3], cpp=True)
    del uu_tnsr, b, b2

    # Deallocate magnetic field in the lab frame
    H.data['B'] = None

    stress_energy_MHD  = stress_energy_HD
    stress_energy_MHD += stress_energy_EM
    if (radiation):
        stress_energy_MHD += stress_energy_RAD

    acc_args = {
        'sigma_cut' : np.float32(3.),
        'sigma'     : sigma
    }

    jet_args = acc_args.copy()
    jet_args['jet'] = True
    # Avoid recalculating sigma selection each time
    acc_args['sigma_selection'] = Acc.sigma_cut(**acc_args)

    local = {}
    local['Mdot_in']        = Acc.calc_Mdot(inward=True)[0]
    local['Mdot_in_sigma']  = Acc.calc_Mdot(inward=True, **acc_args)[0]
    local['Mdot_out']       = Acc.calc_Mdot(outward=True)[0]
    local['Mdot_out_sigma'] = Acc.calc_Mdot(outward=True, **acc_args)[0]

    local['Ldot']       = Acc.calc_Ldot(TMHD=stress_energy_MHD)[0]
    local['Ldot_sigma'] = Acc.calc_Ldot(TMHD=stress_energy_MHD, **acc_args)[0]
    local['Ldot_EM']    = Acc.calc_Ldot(TMHD=stress_energy_EM,  **acc_args)[0]
    local['Ldot_jet']   = Acc.calc_Ldot(TMHD=stress_energy_EM,  **jet_args)[0]
    local['Ldot_Fluid'] = Acc.calc_Ldot(TMHD=stress_energy_HD,  **acc_args)[0]
    if (radiation):
        local['Ldot_Rad'] = Acc.calc_Ldot(TMHD=stress_energy_RAD)[0]

    del stress_energy_MHD
    del stress_energy_HD, stress_energy_EM
    if (radiation):
        del stress_energy_RAD

    # Deallocate radiation energy density and 4-velocity
    if (radiation):
        H.data['u_rad']  = None
        H.data['uu_rad'] = None

    # Calculate angular momentum tensor L^{mu nu}
    # Equation 17 of Fragile & Anninos (2005)
    Lmn = Tilt.calc_angular_momentum_tnsr(pgas=pgas)
    for i, mu in enumerate(COORD_IDX):
        for j, nu in enumerate(COORD_IDX):
            local[f'L{mu}{nu}'] = Acc.thph_sum(Lmn[i, j], **acc_args)[0]
    del Lmn

    if bulk_mom:
        THD_vec = np.zeros_like(H.data['uu'])
        THD_vec[0] = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas, index=[0, 0], cpp=True)
        THD_vec[1] = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas, index=[1, 0], cpp=True)
        THD_vec[2] = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas, index=[2, 0], cpp=True)
        THD_vec[3] = Fluid.calc_THD(**H.data, **H.grid, pgas=pgas, index=[3, 0], cpp=True)

        THD_cart_vec = Tilt.int_to_cart_vec(THD_vec, cpp=True)

        local['St'] = Acc.thph_sum(THD_cart_vec[0], **acc_args)[0]
        local['Sx'] = Acc.thph_sum(THD_cart_vec[1], **acc_args)[0]
        local['Sy'] = Acc.thph_sum(THD_cart_vec[2], **acc_args)[0]
        local['Sz'] = Acc.thph_sum(THD_cart_vec[3], **acc_args)[0]

        gcov_cart         = Tilt.int_to_cart_cotnsr(H.grid['gcov'])
        THD_cart_one_form = ana.Tools.contract_vector(THD_cart_vec, gcov_cart, cpp=True)

        local['S_t'] = Acc.thph_sum(THD_cart_one_form[0], **acc_args)[0]
        local['S_x'] = Acc.thph_sum(THD_cart_one_form[1], **acc_args)[0]
        local['S_y'] = Acc.thph_sum(THD_cart_one_form[2], **acc_args)[0]
        local['S_z'] = Acc.thph_sum(THD_cart_one_form[3], **acc_args)[0]

        del THD_vec, THD_cart_vec, THD_cart_one_form, gcov_cart

    meta = {
        'r'    : H.grid['r'][0, :, 0, 0] / Rg,
        'dx1'  : H.params['dx1'],
        'rho0' : H.params['density_scale']
    }

    del pgas, sigma

    H.data['rho']   = None
    H.data['uu']    = None
    H.data['u_gas'] = None
    H.grid['gcov']  = None
    H.grid['gcon']  = None
    H.grid['gdet']  = None

    # Deallocate temperatures and energy densities
    # in preparation for large transformation matrices
    if (two_temp):
        H.data['Ti'] = None
        H.data['Te'] = None

    del Acc, Field, Fluid, Tilt
    if (radiation):
        del Radi
    acc_args['sigma']           = None
    jet_args['sigma']           = None
    acc_args['sigma_selection'] = None

    return local, meta

def run_phi_slice_parallel(file_num, nslices, restart, profile_keys,
                            profiles, nphi, iphi_stp, data_dir, read_args,
                            save_dir, nzeros, comm, rank, size, **slice_kwargs):
    """Single huge file: ranks split *that file's* phi range. Partial
    results from all ranks working the same slice get MPI-reduced onto
    rank 0, which checkpoints to disk after every slice."""
    for slice_num, iphi in enumerate(range(0, nphi, iphi_stp)):
        if (restart):
            if slice_num < profiles['slice_num_restart']:
                continue

        profiles['saved'] = False
        k_min = int(iphi)
        k_max = int(iphi + iphi_stp)

        delta_k   = (k_max - k_min) / size
        k_min_loc = int(k_min + delta_k * rank)
        k_max_loc = int(k_min + delta_k * (rank + 1))

        # Staggered start to take pressure off of filesystem
        timeit.sleep(0.05 * rank + np.random.rand())

        local, meta = process_batch(data_dir, file_num, k_min_loc,
            k_max_loc, read_args, **slice_kwargs)

        nr = len(meta['r'])
        local_buffer = np.zeros((len(profile_keys), nr), dtype=np.float64)
        for i, key in enumerate(profile_keys):
            if key in local:
                local_buffer[i, :] = local[key]
        global_buffer = None

        print(f'Rank {rank:d} done analyzing range of k: ' +
            f'({k_min_loc:d}, {k_max_loc:d}).', flush=True)

        if (rank == 0):
            profiles['r']    = meta['r']
            profiles['dx1']  = meta['dx1']
            profiles['rho0'] = meta['rho0']
            global_buffer = np.zeros_like(local_buffer)

        comm.Reduce(local_buffer, global_buffer, op=MPI.SUM, root=0)

        if (rank == 0):
            for i, key in enumerate(profile_keys):
                profiles[key] += global_buffer[i, :]

            profiles['k_max'] = k_max
            profiles['slice_num_restart'] = slice_num + 1
            profiles['saved'] = True
            fnum_str = str(file_num).zfill(nzeros)
            if (profiles['k_max'] == nphi):
                profiles['done'] = True
            np.savez(save_dir + f'/Angular_Momentum_{fnum_str}', **profiles)

            snum = int(slice_num + 1)
            print(f'Finished with slice {snum:d} of {nslices:d} ' +
                  f'for file number {file_num}', flush=True)
        comm.Barrier()
        gc.collect()

    print(f'Done with all angular momentum profiles for file number {file_num}.')

def run_time_parallel(file_list_local, nslices, restart, profile_keys,
                       init_args, data_dir, read_args, save_dir, nzeros,
                       rank, **slice_kwargs):
    """Many dumps: ranks split *which files* they own via
    `cv.get_file_list_local`. Each rank owns its files outright, so
    results accumulate directly (no MPI reduction needed), and each file
    is checkpointed once, after all of its phi-slices are done."""
    for file_num in file_list_local:
        init_args['file_num'] = file_num
        profiles, nphi, iphi_stp = cv.init_profiles(**init_args, **read_args)
        profiles['done'] = False

        for slice_num, iphi in enumerate(range(0, nphi, iphi_stp)):
            if (restart):
                if slice_num < profiles['slice_num_restart']:
                    continue

            profiles['saved'] = False
            k_min = int(iphi)
            k_max = int(iphi + iphi_stp)

            # Staggered start to take pressure off of filesystem
            timeit.sleep(0.05 * rank + np.random.rand())

            local, meta = process_batch(data_dir, file_num, k_min,
                k_max, read_args, **slice_kwargs)

            for key in profile_keys:
                if key in local:
                    profiles[key] += local[key]

            profiles['rho0']   = meta['rho0']
            profiles['r']      = meta['r']
            profiles['dx1']    = meta['dx1']
            profiles['k_max']  = k_max

            print(f'Rank {rank:d} done analyzing range of k: ' +
                f'({k_min:d}, {k_max:d}).', flush=True)

            snum = int(slice_num + 1)
            print(f'Finished with slice {snum:d} of {nslices} ' +
                f'for file {file_num}', flush=True)

        profiles['slice_num_restart'] = slice_num + 1
        profiles['saved'] = True
        if (profiles['k_max'] == nphi):
            profiles['done'] = True
        fnum_str = str(file_num).zfill(nzeros)
        np.savez(save_dir + f'/Angular_Momentum_{fnum_str}', **profiles)
        gc.collect()
        print("Done with all angular momentum profiles " +
              f"for file number {file_num}.", flush=True)

def main(**kwargs):
    sim         = kwargs.get('sim')
    file_num    = int(kwargs.get('file_num'))
    start_file  = int(kwargs.get('start_file'))
    end_file    = int(kwargs.get('end_file'))
    nslices     = kwargs.get('nslices')
    lowres1     = kwargs.get('lowres1')
    lowres2     = kwargs.get('lowres2')
    lowres3     = kwargs.get('lowres3')
    stride      = kwargs.get('stride')
    reduced     = kwargs.get('reduced')
    restart     = kwargs.get('restart')
    overwrite   = kwargs.get('overwrite')
    shared      = kwargs.get('shared')
    title       = kwargs.get('title')
    radiation   = kwargs.get('radiation')
    two_temp    = kwargs.get('two_temp')
    vargamma    = kwargs.get('vargamma')
    bulk_mom    = kwargs.get('bulk_momentum')
    MBH         = np.float32(kwargs.get('MBH'))

    # A single dump number selects phi-slice-parallel mode (one huge
    # file, ranks split its phi range); a start/end range selects
    # time-parallel mode (ranks split up which files they own).
    phi_slice_mode = (file_num != -1)
    if phi_slice_mode and (start_file != -1 or end_file != -1):
        raise ValueError('Pass either --file_num (phi-slice-parallel '
            'mode) or --start_file/--end_file (time-parallel mode), not both.')
    if not phi_slice_mode and (start_file == -1 or end_file == -1):
        raise ValueError('Pass either --file_num (phi-slice-parallel '
            'mode) or --start_file/--end_file (time-parallel mode).')

    # Define constants
    cgs = True if (two_temp or radiation) else False
    if (two_temp or radiation):
        Msun = np.float32(hr.CONSTANTS['Msun'])
        c    = np.float32(hr.CONSTANTS['c'])
        G    = np.float32(hr.CONSTANTS['G'])
        Rg   = G * MBH * Msun / (c * c)
    else:
        c, G, Rg = ONE, ONE, ONE

    # Ensure CONSTANTS are 32-bit floats
    CONSTANTS = {key : np.float32(value)
                 for (key, value) in hr.CONSTANTS.items()}

    # Determine data extraction and save directories
    data_dir = cv.get_data_dir(sim, reduced, shared)
    save_dir = cv.get_save_dir(sim if title == None else title,
                               'Averages', 'Angular_Momentum', reduced=reduced)

    nzeros = len(str(len(glob.glob(os.path.join(data_dir, 'dumps*')))))

    # Quantities (quants) for profile averaging
    keys = []
    for mu in COORD_IDX:
        for nu in COORD_IDX:
            keys.append(f'L{mu}{nu}')

    if bulk_mom:
        keys += ['St', 'Sx', 'Sy', 'Sz', 'S_t', 'S_x', 'S_y', 'S_z']

    # Accretion rates (mass and angular momentum)
    acc_keys = [
        'Mdot_in', 'Mdot_out', 'Mdot_in_sigma', 'Mdot_out_sigma',
        'Ldot', 'Ldot_sigma', 'Ldot_Fluid', 'Ldot_EM', 'Ldot_jet'
    ]
    if radiation:
        acc_keys += ['Ldot_Rad']

    profile_keys = keys + acc_keys

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

    slice_kwargs = {
        'CONSTANTS' : CONSTANTS,
        'cgs'       : cgs,
        'MBH'       : MBH,
        'radiation' : radiation,
        'two_temp'  : two_temp,
        'vargamma'  : vargamma,
        'bulk_mom'  : bulk_mom,
        'c'         : c,
        'Rg'        : Rg
    }

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if phi_slice_mode:
        init_args = {
            'data_dir' : data_dir,
            'file_num' : file_num,
            'keys'     : profile_keys,
            'save_dir' : save_dir,
            'nslices'  : nslices,
            'restart'  : restart,
            'nzro'     : nzeros,
            'two_temp' : two_temp,
            'radiation': radiation
        }
        profiles, nphi, iphi_stp = cv.init_profiles(**init_args, **read_args)
        profiles['done'] = False

        run_phi_slice_parallel(file_num, nslices, restart, profile_keys,
            profiles, nphi, iphi_stp, data_dir, read_args, save_dir,
            nzeros, comm, rank, size, **slice_kwargs)
    else:
        file_list = list(range(start_file, end_file + 1))
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
                averages_dir, file_type='Angular_Momentum', reduced=reduced)
            existing_set = set(existing_profiles)
            file_list_local = [f for f in file_list_local if f not in existing_set]

        init_args = {
            'data_dir' : data_dir,
            'keys'     : profile_keys,
            'save_dir' : save_dir,
            'nslices'  : nslices,
            'restart'  : restart,
            'nzro'     : nzeros,
            'two_temp' : two_temp,
            'radiation': radiation
        }

        run_time_parallel(file_list_local, nslices, restart, profile_keys,
            init_args, data_dir, read_args, save_dir, nzeros, rank,
            **slice_kwargs)

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Calculate 1D profiles necessary
                        for angular momentum analysis. Runs in
                        phi-slice-parallel mode if --file_num is given
                        (one dump, ranks split its phi range), or
                        time-parallel mode if --start_file/--end_file
                        are given (ranks split up which dumps they
                        own).""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('--file_num', type=int, required=False, default=-1,
                        help = """dump number for phi-slice-parallel
                        mode""")
    parser.add_argument('--start_file', type=int, required=False,
                        default=-1, help = """dump number of start file
                        for time-parallel mode""")
    parser.add_argument('--end_file', type=int, required=False,
                        default=-1, help = """dump number of end file
                        for time-parallel mode""")
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
                        (time-parallel mode only)""")
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
    parser.add_argument('--bulk_momentum',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="""boolean to determine if the bulk momentum
                        vector S from Fragile & Anninos (2005) should be
                        extracted.""")
    parser.add_argument('--MBH', type=float, default=10.,
                        required=False,
                        help = """black hole mass in Msun (default: 10)""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if profile extraction
                        should be done on reduced data.""")
    parser.add_argument('--restart', action='store_true', required=False,
                        help="""boolean set to determine if profile
                        integration should continue from last slice_num.""")
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="""boolean to determine if existing profile file
                        should be overwritten (time-parallel mode only).""")
    parser.add_argument('--shared', action='store_true', required=False,
                        help="""boolean set to determine if proj-shared should
                        be used as the location of the data directory.""")
    parser.add_argument('--title', type=str, required=False, default=None,
                        help="""title of simulation used for Averages directory
                        if different from the title in proj-shared.""")
    args = parser.parse_args()
    main(**vars(args))
