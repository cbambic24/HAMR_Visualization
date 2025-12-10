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
import glob
from mpi4py import MPI
from mpi4py.util import pkl5
import time as timeit

def main(**kwargs):
    sim          = kwargs.get('sim')
    start_file   = int(kwargs.get('start_file'))
    end_file     = int(kwargs.get('end_file'))
    nslices      = kwargs.get('nslices')
    lowres1      = kwargs.get('lowres1')
    lowres2      = kwargs.get('lowres2')
    lowres3      = kwargs.get('lowres3')
    reduced      = kwargs.get('reduced')
    restart      = kwargs.get('restart')
    shared       = kwargs.get('shared')
    title        = kwargs.get('title')
    radiation    = kwargs.get('radiation')
    two_temp     = kwargs.get('two_temp')
    time_average = kwargs.get('time_average')

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

    # Determine data extraction and save directories
    data_dir = cv.get_data_dir(sim, reduced, shared)
    slice_dir = cv.get_save_dir(sim if title == None else title, 
                               'Averages', 'Slices', reduced=reduced)
    save_dir = cv.get_save_dir(sim if title == None else title, 
                               'Averages', 'Stresses', reduced=reduced)

    nzeros = len(str(int(os.popen(
        f'ls -d {data_dir}/dumps* | wc -l').read())))
        
    # Quantities (quants) for slice averaging
    keys = [
        'TRphi_Rey_mean', 'TRphi_Rey_turb', 'TRphi_Max_mean', 'TRphi_Max_turb', 
        'Tzphi_Rey_mean', 'Tzphi_Rey_turb', 'Tzphi_Max_mean', 'Tzphi_Max_turb', 
        'Trphi_Rey_mean', 'Trphi_Rey_turb', 'Trphi_Max_mean', 'Trphi_Max_turb', 'w' 
    ]
    # Time-averages needed for slice averaging
    mean_keys = [
        'wuR',  'wuz', 'wu_phi', 'wur', 'wuth',
        'bR',   'bz',  'b_phi',  'br',  'bth', 'Vol'
    ]
    if (radiation):
        keys += ['Rrphi', 'RRphi', 'Rzphi']
    
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

    last_file  = get_last_file_num(data_dir)
    first_call = True

    if (time_average):
        Tavg = Time_Averages(slice_dir, mean_keys, nzeros)

    # Outermost loop over file number (each file on single core)
    for file_num in file_list_local:
        stresses, nphi, iphi_stp = init_slices(data_dir, file_num, keys,
            save_dir, nslices, restart, nzeros, **read_args)
        
        if (time_average):
            start_fnum = max(file_num - int(time_average / 2), 0)
            end_fnum   = min(file_num + int(time_average / 2), last_file)
            time_average_args = {
                'fnum'         : file_num,
                'file_numbers' : range(start_fnum, end_fnum + 1),
                'first_call'   : first_call
            }

            Tavg.update_data(**time_average_args)
            Tavg.calc_time_averages()
            
            mean = Tavg.mean
            if (first_call): first_call = False

        # Loop over whole data slice that fits on single node.
        # Process slivers of data slice assigned to each core in parallel.
        for slice_num, iphi in enumerate(range(0, nphi, iphi_stp)):
            if (restart):
                if slice_num < stresses['slice_num_restart']:
                    continue

            stresses['saved'] = False
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
        
            ones = np.ones_like(H.data['rho']) # (nr, nth, nphi)
            if (radiation):
                w = (H.data['rho'] * c * c + pgas + H.data['u_gas'] 
                     + b2 / (4.*np.pi) + (4./3.) * H.data['u_rad']) 
            else:
                w = (H.data['rho'] * c * c + pgas + H.data['u_gas'] + b2) 

            # Enthalpy
            stresses['w'] += ph_sum(w, H.grid, H.data, H.params, 
                                    hr.CONSTANTS, **acc_args)[0]

            # Reynolds stresses
            uu_tnsr13 = ana.Tools.construct_dyadic(H.data['uu'], 
                H.grid['gcov'], upper_lower=True, index=[1, 3], cpp=True)
            uu_tnsr23 = ana.Tools.construct_dyadic(H.data['uu'], 
                H.grid['gcov'], upper_lower=True, index=[2, 3], cpp=True)
            uu_tnsr_vec = np.zeros_like(H.data['uu'])
            uu_tnsr_vec[1] = uu_tnsr13
            uu_tnsr_vec[2] = uu_tnsr23
            uu_tnsr_cyl    = Coords.int_to_cyl_vec(uu_tnsr_vec)
            uu_tnsr_sph    = Coords.int_to_ortho_sph_vec(uu_tnsr_vec)

            # r-phi Reynolds stresses
            wurwu_phi = (mean['wur'] * mean['wu_phi'])[None, :, :, None]
            stresses['Trphi_Rey_mean'] += ph_sum(wurwu_phi, H.grid, H.data, 
                H.params, hr.CONSTANTS, **acc_args)[0]
            stresses['Trphi_Rey_turb'] += ph_sum(
                (w * uu_tnsr_sph[1]) - (wurwu_phi * ones), 
                H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
            
            del uu_tnsr13, uu_tnsr23, uu_tnsr_sph, wurwu_phi

            # R-phi Reynolds stresses
            wuRwu_phi = (mean['wuR'] * mean['wu_phi'])[None, :, :, None]
            stresses['TRphi_Rey_mean'] += ph_sum(wuRwu_phi, H.grid, H.data, 
                H.params, hr.CONSTANTS, **acc_args)[0]
            stresses['TRphi_Rey_turb'] += ph_sum(
                (w * uu_tnsr_cyl[1]) - (wuRwu_phi * ones), 
                H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
            
            del uu_tnsr_vec, wuRwu_phi
            
            # z-phi Reynolds stresses
            wuzu_phi = (mean['wuz'] * mean['wu_phi'])[None, :, :, None]
            stresses['Tzphi_Rey_mean'] += ph_sum(wuzu_phi, H.grid, H.data, 
                H.params, hr.CONSTANTS, **acc_args)[0]
            stresses['Tzphi_Rey_turb'] += ph_sum(
                (w * uu_tnsr_cyl[3]) - (wuzu_phi * ones), 
                H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
        
            del uu_tnsr_cyl, wuzu_phi

            # Maxwell stresses
            bb_tnsr13 = ana.Tools.construct_dyadic(b, H.grid['gcov'], 
                upper_lower=True, index=[1, 3], cpp=True)
            bb_tnsr23 = ana.Tools.construct_dyadic(b, H.grid['gcov'], 
                upper_lower=True, index=[2, 3], cpp=True)
            bb_tnsr_vec = np.zeros_like(b)
            bb_tnsr_vec[1] = bb_tnsr13
            bb_tnsr_vec[2] = bb_tnsr23
            bb_tnsr_cyl    = Coords.int_to_cyl_vec(bb_tnsr_vec)
            bb_tnsr_sph    = Coords.int_to_ortho_sph_vec(bb_tnsr_vec)

            # r-phi Maxwell stresses
            brb_phi = (mean['br'] * mean['b_phi'])[None, :, :, None]
            if (two_temp or radiation):
                stresses['Trphi_Max_mean'] += ph_sum(-brb_phi / (4. * np.pi), 
                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
                stresses['Trphi_Max_turb'] += ph_sum(
                    (-bb_tnsr_sph[1] - (-brb_phi * ones)) / (4. * np.pi), 
                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
            else:
                stresses['Trphi_Max_mean'] += ph_sum(-brb_phi, 
                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
                stresses['Trphi_Max_turb'] += ph_sum(
                    (-bb_tnsr_sph[1] - (-brb_phi * ones)), 
                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]

            del bb_tnsr13, bb_tnsr23, brb_phi, bb_tnsr_sph

            # R-phi Maxwell stresses
            bRb_phi = (mean['bR'] * mean['b_phi'])[None, :, :, None]
            if (two_temp or radiation):
                stresses['TRphi_Max_mean'] += ph_sum(-bRb_phi / (4. * np.pi), 
                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
                stresses['TRphi_Max_turb'] += ph_sum(
                    (-bb_tnsr_cyl[1] - (-bRb_phi * ones)) / (4. * np.pi), 
                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
            else:
                stresses['TRphi_Max_mean'] += ph_sum(-bRb_phi, 
                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
                stresses['TRphi_Max_turb'] += ph_sum(
                    (-bb_tnsr_cyl[1] - (-bRb_phi * ones)), 
                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]

            del bb_tnsr_vec, bRb_phi

             # z-phi Maxwell stresses
            bzb_phi = (mean['bz'] * mean['b_phi'])[None, :, :, None]
            if (two_temp or radiation):
                stresses['Tzphi_Max_mean'] += ph_sum(-bzb_phi / (4. * np.pi), 
                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
                stresses['Tzphi_Max_turb'] += ph_sum(
                    (-bb_tnsr_cyl[3] - (-bzb_phi * ones)) / (4. * np.pi), 
                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
            else:
                stresses['Tzphi_Max_mean'] += ph_sum(-bzb_phi, 
                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
                stresses['Tzphi_Max_turb'] += ph_sum(
                    (-bb_tnsr_cyl[3] - (-bzb_phi * ones)), 
                    H.grid, H.data, H.params, hr.CONSTANTS, **acc_args)[0]
        
            del bb_tnsr_cyl, bzb_phi

            # Radiation diffusion stresses
            if (radiation):
                stress_energy_RAD_13 = Radi.calc_TRAD(**H.data, **H.grid, 
                    upper_lower=True, index=[1, 3], cpp=True)
                stress_energy_RAD_23 = Radi.calc_TRAD(**H.data, **H.grid, 
                    upper_lower=True, index=[2, 3], cpp=True)
                R_vec    = np.zeros_like(H.data['uu'])
                R_vec[1] = stress_energy_RAD_13
                R_vec[2] = stress_energy_RAD_23
                R_cyl    = Coords.int_to_cyl_vec(R_vec)
                R_sph    = Coords.int_to_ortho_sph_vec(R_vec)

                uu_cov = np.einsum('ij...,j...->i...', H.grid['gcov'], H.data['uu'])
                uucyl  = Coords.int_to_cyl_vec(H.data['uu'])
                uusph  = Coords.int_to_ortho_sph_vec(H.data['uu'])
                uu_cyl = Coords.int_to_cyl_one_form(uu_cov)
                uu_sph = Coords.int_to_ortho_sph_one_form(uu_cov)

                del uu_cov

                # r-phi radiation diffusion stress
                Rrphi = R_sph[1] - (H.data['u_rad'] * uusph[1] * uu_sph[3])
                stresses['Rrphi'] += ph_sum(Rrphi, H.grid, H.data, 
                    H.params, hr.CONSTANTS, **acc_args)[0]
                
                del stress_energy_RAD_13, stress_energy_RAD_23
                del R_vec, R_sph, Rrphi, uusph, uu_sph

                # R-phi radiation diffusion stress
                RRphi = R_cyl[1] - (H.data['u_rad'] * uucyl[1] * uu_cyl[2])
                stresses['RRphi'] += ph_sum(RRphi, H.grid, H.data, 
                    H.params, hr.CONSTANTS, **acc_args)[0]
                
                del RRphi
                
                # z-phi radiation diffusion stress
                Rzphi = R_cyl[3] - (H.data['u_rad'] * uucyl[3] * uu_cyl[2])
                stresses['Rzphi'] += ph_sum(Rzphi, H.grid, H.data, 
                    H.params, hr.CONSTANTS, **acc_args)[0]
                
                del R_cyl, Rzphi
    
            # Put spherical radius in dictionary 
            stresses['r']     = H.grid['r'][0, :, 0, 0] / Rg
            stresses['theta'] = H.grid['theta'][0, 0, :, 0]
            stresses['phi']   = H.grid['phi'][0, 0, 0, :]
            stresses['dx1']   = H.params['dx1']
            stresses['dx2']   = H.params['dx2']
            stresses['dx3']   = H.params['dx3']
        
            # Density scale
            stresses['rho0'] = H.params['density_scale']
            stresses['k_max'] = k_min + (k_max - k_min)

            print(f'Rank {rank:d} done analyzing range of k: ' + 
                f'({k_min:d}, {k_max:d}).', flush=True)
       
            snum = int(slice_num + 1)
            print(f'Finished with slice {snum:d} of {nslices} ' +
                f'for file {file_num}', flush=True) 

        if (time_average):
            Tavg.reset_mean()

        stresses['slice_num_restart'] = slice_num + 1
        fnum_str = str(file_num).zfill(nzeros)
        np.savez(save_dir + f'/Stresses_{fnum_str}', **stresses)
        stresses['saved'] = True

        print(f'Done with all slices for file number {file_num}.',flush=True) 

def thph_sum(q, grid, data, params, CONSTANTS, **kwargs):
    w    = kwargs.get('weight', 1.0)
    Acc2 = ana.Accretion(grid, data, params, CONSTANTS, 
            cgs=True, radiation=True, MBH = 10.0)
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

class Time_Averages():
    def __init__(self, slice_dir, mean_keys, nzeros):
        self.slice_dir    = slice_dir
        self.mean_keys    = mean_keys
        self.nzeros       = nzeros

    def update_data(self, fnum, file_numbers, first_call):
        if (first_call): 
            self.mean     = dict.fromkeys(self.mean_keys)
        self.fnum = fnum
        self.file_numbers = file_numbers
        self.first_call   = first_call
        print(f"Data has been updated for file {self.fnum}")
                 
    def calc_time_averages(self):
        nt = len(self.file_numbers)
        self.check_AMR()
        self.check_dims_changed()
        print("AMR active? ", self.AMR)
        if (self.first_call):
            nr0, nth0 = self.nr[f'{self.fnum}'], self.nth[f'{self.fnum}']
            for key in self.mean_keys:
                print("Initalizing mean array")
                self.mean[f'{key}'] = np.zeros([nr0, nth0])
                for file_num in self.file_numbers:
                    self.add_slice_AMR(file_num, key, add=True)
            for key in self.mean_keys[:-1]:
                self.mean[f'{key}']= self.mean[f'{key}']/(nt*self.mean['Vol'])
        else:
            for key in self.mean_keys:
                self.add_slice_AMR(self.file_numbers[-1], key, add=True)
            for key in self.mean_keys[:-1]:
                self.mean[f'{key}']= self.mean[f'{key}']/(nt*self.mean['Vol'])
    
    def add_slice_AMR(self, file_num, key, **kwargs):
        slice_file = f'{self.slice_dir}/Slices_%s.npz' %(
            str(file_num).zfill(self.nzeros))
        if not self.AMR:
            if (kwargs.get('add', True)):
                self.mean[f'{key}'] += np.load(slice_file)[f'{key}']
            else:
                self.mean[f'{key}'] -= np.load(slice_file)[f'{key}']
        elif self.AMR:
            level_x1 = np.log2(self.nr[f'{self.fnum}']/self.nr[f'{file_num}'])
            level_x2 = np.log2(self.nth[f'{self.fnum}']/self.nth[f'{file_num}'])
            if ((not level_x1.is_integer()) or (not level_x2.is_integer())):
                raise ValueError(f"Non-trivial AMR. Interpolation required. " +  
                                 f"level x1 = {level_x1}, " +  
                                 f"level x2 = {level_x2}")
            up_x1 = level_x1 > 0
            up_x2 = level_x2 > 0
            dn_x1 = level_x1 < 0
            dn_x2 = level_x2 < 0
            print(f"AMR is {self.AMR} and level_x1 is {level_x1} while level_x2 is {level_x2}.")
            if (level_x1 == 0 and level_x2 == 0):
                if (kwargs.get('add', True)):
                    self.mean[f'{key}'] += np.load(slice_file)[f'{key}']
                else:
                    print("And now I'm subtracting")
                    self.mean[f'{key}'] -= np.load(slice_file)[f'{key}']
            elif (up_x1 or up_x2):
                self.upscale_AMR(file_num, key, level_x1, level_x2, 
                                 add=kwargs.get('add', True))
            elif (dn_x1 or dn_x2):
                self.downscale_AMR(file_num, key, level_x1, level_x2, 
                                   add=kwargs.get('add', True))

    def upscale_AMR(self, file_num, key, level_x1, level_x2, add):
        print(f"I'm upscaling {key}...")
        slice_file = f'{self.slice_dir}/Slices_%s.npz' %(
            str(file_num).zfill(self.nzeros))
        nr  = self.nr[f'{file_num}']
        nth = self.nth[f'{file_num}']
        q = np.load(slice_file)[f'{key}']
        for i in range(nr):
            for j in range(nth):
                istrt = int(i       * (2 ** level_x1))
                iend  = int((i + 1) * (2 ** level_x1))
                jstrt = int(j       * (2 ** level_x2))
                jend  = int((j + 1) * (2 ** level_x2))

                self.mean[f'{key}'][istrt:iend, jstrt:jend] += (
                    add * q[i, j] - (not add) * q[i, j])
                
    def downscale_AMR(self, file_num, key, level_x1, level_x2, add):
        print(f"I'm downscaling {key}...")
        slice_file = f'{self.slice_dir}/Slices_%s.npz' %(
            str(file_num).zfill(self.nzeros))
        q = np.load(slice_file)[f'{key}']
        sx1 = int(2 ** -level_x1)
        sx2 = int(2 ** -level_x2)

        self.mean[f'{key}'] += (
            add * q[::sx1, ::sx2] - (not add) * q[::sx1, ::sx2])
        
    def reset_mean(self):
        for key in self.mean_keys[:-1]:
            nt = len(self.file_numbers)
            self.mean[f'{key}'] *= self.mean['Vol'] * nt

        # subtract off first file
        start_fnum = self.file_numbers[0]
        for key in self.mean_keys:            
            self.add_slice_AMR(start_fnum, key, add=False)

    def check_AMR(self):
        nr_array, nth_array = [], []
        self.nr  = dict.fromkeys(self.file_numbers)
        self.nth = dict.fromkeys(self.file_numbers)
        for file_num in self.file_numbers:
            slice_file = f'{self.slice_dir}/Slices_%s.npz' %(
                str(file_num).zfill(self.nzeros))
            self.nr[f'{file_num}']  = len(np.load(slice_file)['r'])
            self.nth[f'{file_num}'] = len(np.load(slice_file)['theta'])
            
            nr_array.append(self.nr[f'{file_num}'])
            nth_array.append(self.nth[f'{file_num}'])

        last = self.file_numbers[-1]

        self.AMR = ((nr_array.count(self.nr[f'{last}'])   != len(nr_array)) or 
                    (nth_array.count(self.nth[f'{last}']) != len(nth_array)))
    
    def check_dims_changed(self):
        if not self.first_call:
            nr0, nth0 = self.nr[f'{self.fnum}'], self.nth[f'{self.fnum}']
            if (self.mean[f'{self.mean_keys[0]}'].shape != (nr0, nth0)):
                self.first_call = True
                print("The dimensions have changed!")

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

def get_last_file_num(data_dir):
    files = list((os.popen(f'ls -d {data_dir}/dumps*').read()).split('\n'))
    file_list = [f.strip(f'{data_dir}').strip('dumps') for f in files]
    while '' in file_list: file_list.remove('')
    file_list = [int(f) for f in file_list]
    return max(file_list)

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Calculate 2D poloidal
                        slices of stress for all simulation outputs.""")
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
                        should be included in profile extraction.""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if two-temperature 
                        physics should be included in profile extraction.""")    
    parser.add_argument('--waves', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""Alfven wave analysis should be performed
                        in addition to standard stress analysis.""")    
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
    parser.add_argument('--time_average', type=int, default=1,
                        required=False,
                        help = """number of slices (enumerated by dump number)
                        to average in time when computing mean and 
                        turbulent/fluctuating stresses""")
    args = parser.parse_args()
    main(**vars(args))