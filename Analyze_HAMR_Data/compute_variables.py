import sys
import os
import glob
import numpy as np
asys_path   = os.path.dirname(os.path.abspath(__file__))
vis_path    = os.path.dirname(asys_path)
read_path   = os.path.join(vis_path, "Read_HAMR_Data")
user_path   = os.path.dirname(vis_path)
run_path    = os.path.join(user_path, "RUN")
shared_path = os.path.dirname(user_path)
sys.path.insert(0, read_path)
sys.path.insert(0, asys_path)
sys.path.insert(0, run_path)
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata
import analysis as ana
import coordinates as coords
import radiation as rad
import theory as theory
import hamr_read as hr

##############################################################################

def get_data_dir(sim, reduced, shared):
    if not reduced and not shared:
        data_dir = f'{user_path}/DATA/{sim}'
    elif reduced and not shared:
        data_dir = f'{user_path}/DATA/{sim}/reduced/'
    elif not reduced and shared:
        data_dir = f'{shared_path}/{sim}'
    elif reduced and shared:
        data_dir = f'{shared_path}/{sim}/reduced/'
    return data_dir 

def get_save_dir(sim, type, diagnostic, **kwargs):
    subtype = kwargs.get('subtype', False)
    reduced = kwargs.get('reduced', False)
    if not subtype and not reduced:
        save_dir = f'{user_path}/{type}/{sim}/Full_Resolution/{diagnostic}'
    elif subtype and not reduced:
        save_dir = f'{user_path}/{type}/{sim}/Full_Resolution/{subtype}/{diagnostic}'
    elif not subtype and reduced:
        save_dir = f'{user_path}/{type}/{sim}/Reduced_Resolution/{diagnostic}'
    elif subtype and reduced:
        save_dir = f'{user_path}/{type}/{sim}/Reduced_Resolution/{subtype}/{diagnostic}'
    return save_dir

def get_file_list_local(file_list, size, rank, data_dir, **kwargs):
    nout             = len(file_list)
    nfiles_nproc     = int(np.floor(nout / size))
    nfiles_remaining = nout % size
    files_per_proc   = ((size - nfiles_remaining) * [nfiles_nproc] +
        nfiles_remaining * [nfiles_nproc + 1])
    file_list_local  = ((sum(files_per_proc[:rank]) +
        np.arange(0, files_per_proc[rank])) + file_list[0]).tolist()

    if not kwargs.get('file_type', False):
        file_list_local = check_dumps_exist(data_dir, file_list_local)
    else:
        file_list_local = check_files_exist(data_dir, file_list_local, **kwargs)
    
    return file_list_local

def check_dumps_exist(path_to_dumps, file_list):
    removed_files = []
    for file_num in file_list:
        if not os.path.isdir(path_to_dumps + f'/dumps{file_num}'):
            print(f'Dumps {file_num} is missing.')
            removed_files.append(file_num)
            if os.path.isdir(path_to_dumps + f'/dumps{file_num}_'):
                found_file = path_to_dumps + f'/dumps{file_num}_'
                print(f'The file can be found as {found_file}', flush=True) 
    for removed_file in removed_files:
        file_list.remove(removed_file)
    return file_list

def check_files_exist(path_to_averages, file_list, **kwargs):
    file_type = kwargs.get('file_type', False)
    reduced   = kwargs.get('reduced',   False)
    file_dir  = (f'{path_to_averages}/Full_Resolution/{file_type}' if not reduced 
                 else f'{path_to_averages}/Reduced_Resolution/{file_type}')
    data_list = glob.glob(f'{file_type}*', root_dir=file_dir)
    nums = [int(name.strip(f'{file_type}_').strip('.npz')) for name in data_list]
    nums.sort()
    file_list = [f for f in file_list if f in set(nums)]
    return file_list

def check_images_exist(path_to_images, file_list, title, subtype, **kwargs):
    reduced   = kwargs.get('reduced',   False)
    file_dir  = (
        f'{path_to_images}/Full_Resolution/{subtype}/{title}' if not reduced 
        else f'{path_to_averages}/Reduced_Resolution/{subtype}/{title}'
    )
    data_list = glob.glob(f'*.png', root_dir=file_dir)
    nums = [int(name.strip(f'{title}_').strip('.png')) for name in data_list]
    nums.sort()
    file_list = [f for f in file_list if f in set(nums)]
    return file_list
    
def get_nice_time(time):
    raw_string = '%.5e' %time
    coeff = raw_string.split('e+')[0]
    expon = raw_string.split('e+')[1]
    return r'$%.5f \times 10^%d$' %(float(coeff), int(expon))

def get_nice_Mdot(Mdot):
    if (Mdot >= 1.e3):
        raw_string = '%.5e' %Mdot
        coeff = raw_string.split('e+')[0]
        expon = raw_string.split('e+')[1]
        return r'$%.1f \times 10^%d$' %(float(coeff), int(expon))
    elif (Mdot > 10 and Mdot < 1.e3):
        return r'$%.1f$' %Mdot
    elif (Mdot < 1.e-2):
        raw_string = '%.5e' %Mdot
        coeff = raw_string.split('e-')[0]
        expon = raw_string.split('e-')[1]
        if (round(float(coeff)) == 10.):
            expon, coeff = int(expon), float(coeff)
            expon -= 1
            coeff = 1.
        return r'$%.1f \times 10^{-%d}$' %(float(coeff), int(expon))
    else:
        return r'$%.2f$' %Mdot

def check_extend(q, lims):
    qmin, qmax = np.amin(q), np.amax(q)
    lower, upper = False, False 
    if (qmin < lims[0]): lower = True
    if (qmax > lims[1]): upper = True
    if (lower and upper):
        return 'both'
    elif (lower and not upper):
        return 'min'
    elif (not lower and upper):
        return 'max'
    else:
        return 'neither'

# Specific to X-Z (poloidal) plots
def get_contours(plt_args, PS_r, PS_l, contour_colors, sigma_contour, photospheres):
    quantities, levels, colors, linewidths, linestyles = (dict.fromkeys(['l', 'r']) for i in range(5))
    quantities['r'], levels['r'], colors['r'], linewidths['r'], linestyles['r'] = ([] for i in range(5))
    quantities['l'], levels['l'], colors['l'], linewidths['l'], linestyles['l'] = ([] for i in range(5))
    
    if (sigma_contour != 0):
        quantities['r'].append(PS_r.calc_sigma(0))
        quantities['l'].append(PS_l.calc_sigma(0))
        for key in ['r', 'l']:
            levels[    f'{key}'].append(sigma_contour)
            colors[    f'{key}'].append(contour_colors[0])
            linestyles[f'{key}'].append(['-'])
    if (photospheres['scattering']):
        opacities = dict.fromkeys(['electron_scattering'])
        quantities['r'].append(PS_r.calc_tau(opacities, 0, integral=photospheres['integral']))
        quantities['l'].append(PS_l.calc_tau(opacities, 0, integral=photospheres['integral']))
        for key in ['r', 'l']:
            levels[    f'{key}'].append(1.)
            colors[    f'{key}'].append(contour_colors[1])
            linestyles[f'{key}'].append(['-'])
    if (photospheres['effective']):
        opacities = dict.fromkeys(['electron_scattering', 'bound_free', 
            'CHIANTI', 'free_free', 'hydrogen', 'molecular', 'synchrotron']) 
        quantities['r'].append(PS_r.calc_tau(opacities, 0, integral=photospheres['integral']))
        quantities['l'].append(PS_l.calc_tau(opacities, 0, integral=photospheres['integral']))
        for key in ['r', 'l']:
            levels[    f'{key}'].append(1.)
            colors[    f'{key}'].append(contour_colors[1])
            linestyles[f'{key}'].append(['--'])

    for key in ['r', 'l']: linewidths[f'{key}'] += len(levels[f'{key}']) * [2.0]

    plt_args['contours_right'] = {
        'quantities' : quantities['r'], 
        'levels'     : levels['r'], 
        'colors'     : colors['r'], 
        'linewidths' : linewidths['r'],
        'linestyles' : linestyles['r']
    }
    plt_args['contours_left'] = {
        'quantities' : quantities['l'], 
        'levels'     : levels['l'], 
        'colors'     : colors['l'], 
        'linewidths' : linewidths['l'],
        'linestyles' : linestyles['l']
    }
    return plt_args

def grid(ax, **kwargs):
    ax.grid(which='major', color=kwargs.get('color', 'white'), linestyle='-',
            linewidth=0.5, alpha=kwargs.get('alpha', 0.3))
    ax.grid(which='minor', color=kwargs.get('color', 'white'), linestyle='-',
            linewidth=0.5, alpha=kwargs.get('alpha', 0.3))

def init_profiles(data_dir, file_num, keys, 
    save_dir, nslices, restart, nzro, **kwargs):
    profiles = {}
    
    # First, determine the limits of the slices.
    H = hr.Read(data_dir, **kwargs)
    H.read_params(file_num)
    profiles['t'] = H.params['t']
    profiles['a'] = H.params['a']

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
        profiles = dict(np.load(f'{save_dir}/Profiles_{fnum_str}.npz'))
        first_call = False

    # Initialize profiles dictionary
    if first_call:
        for key in keys:
            profiles[f'{key}'] = np.zeros(nr, dtype=np.float64)
        first_call = False
        profiles['slice_num_restart'] = 0

    return profiles, nphi, iphi_stp

def init_slices(data_dir, file_num, keys, 
    save_dir, nslices, restart, nzro, **kwargs):
    slices = {}
    
    file_type = kwargs.get('file_type', 'Slices')
    
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

    if 'r_max' in kwargs:
        H.gridsizex1 = int(
            H.params['nb1'] * H.params['bs1'] * H.upscalex1 / H.lowres1)
        i_min, nr = H.get_rbox_coords()

    # Logic to allow restarts if profile extraction is interrupted.
    if not restart:
        first_call = True
    else:
        fnum_str = str(file_num).zfill(nzro)
        slices = dict(np.load(f'{save_dir}/{file_type}_{fnum_str}.npz'))
        first_call = False

    # Initialize slices dictionary
    if first_call:
        for key in keys:
            slices[f'{key}'] = np.zeros([nr, nth])
        first_call = False
        slices['slice_num_restart'] = 0

    return slices, nphi, iphi_stp

# Compute quantities at constant phi
class Phi_Slice():
    def __init__(self, directory, fnum, **kwargs):
        self.H = hr.Read(directory, **kwargs)
        self.H.read_grid(fnum, direct=True)
        self.H.read_prims(fnum, skip_read_params=True)
        self.cgs  = kwargs.get('cgs_data', True)
        self.MBH  = kwargs.get('MBH', 10.0) if self.cgs else 1.
        self.Msun = hr.CONSTANTS['Msun']
        self.rho0 = self.H.params['density_scale'] if self.cgs else 1. # g/cm^3
        self.c    = hr.CONSTANTS['c'] if self.cgs else 1. # cm/s
        self.G    = hr.CONSTANTS['G'] if self.cgs else 1. # cm^3/g s^2
        self.Rg   = (self.G * self.MBH * self.Msun / (self.c * self.c)
                     if self.cgs else 1.)

        self.calc_gridsize()

        self.r    = self.H.grid['r'] / self.Rg  
        self.th   = self.H.grid['theta']
        self.phi  = self.H.grid['phi']

        phi0        = np.pi
        self.phi_1D = self.phi[0, 0, 0, :]
        self.iphi   = np.argmin(np.abs(self.phi_1D - phi0))

        self.two_temp  = True if 'Te' and 'Ti' in self.H.data else False
        self.radiation = True if 'u_rad' and 'uu_rad' in self.H.data else False

        acc_args = {
            'grid'      : self.H.grid,
            'data'      : self.H.data,
            'pars'      : self.H.params,
            'CONSTANTS' : hr.CONSTANTS,
            'cgs'       : self.cgs,
            'MBH'       : self.MBH,
            'radiation' : self.radiation
        }
        class_args = {
            'pars'      : self.H.params,
            'CONSTANTS' : hr.CONSTANTS,
            'cgs'       : self.cgs
        }

        # Initialize Accretion class
        self.Acc   = ana.Accretion(**acc_args)
        # Initialize Fluid class
        self.Fluid = ana.Fluid(**class_args)
        # Initialize Field class
        self.Field = ana.Field(**class_args)
        # Initialize Radiation class
        self.Rad   = rad.Radiation(**class_args)
        # Initialize Analytic class
        self.Analytic = theory.Analytic(hr.CONSTANTS, cgs=self.cgs)

    def calc_gridsize(self):
        self.nr   = int(self.H.params['nb1'] * self.H.params['bs1'] * self.H.upscalex1 / self.H.lowres1)
        self.nth  = int(self.H.params['nb2'] * self.H.params['bs2'] * self.H.upscalex2 / self.H.lowres2)
        self.nphi = int(self.H.params['nb3'] * self.H.params['bs3'] * self.H.upscalex3 / self.H.lowres3)
        return None

    def calc_XZ(self, iphi):
        r   = self.r
        th  = self.th
        phi = self.phi
        
        # Convert to Cartesian coordinates
        # Note reflection for left panel
        self.x = (r*np.sin(th)*np.cos(phi))[0, :, :, iphi]
        self.y = (r*np.sin(th)*np.sin(phi))[0, :, :, iphi]
        self.z = (r*np.cos(th)*np.ones(phi.shape))[0, :, :, iphi]
        
        x, z = {}, {}
        x['r'] = self.x
        z['r'] = self.z
        x['l'] = -self.x
        z['l'] = self.z
        return x, z

    def calc_RZ(self, iphi):
        r   = self.r
        th  = self.th
        phi = self.phi
        
        # Convert to cylindrical coordinates
        # Note reflection for left panel
        self.x = (r*np.sin(th)*np.cos(phi))[0, :, :, iphi]
        self.y = (r*np.sin(th)*np.sin(phi))[0, :, :, iphi]
        self.z = (r*np.cos(th)*np.ones(phi.shape))[0, :, :, iphi]
        
        R, z = {}, {}
        R['r'] = np.sqrt(np.square(self.x) + np.square(self.y))
        z['r'] = self.z
        R['l'] = -np.sqrt(np.square(self.x) + np.square(self.y))
        z['l'] = self.z
        return R, z

    def calc_XZ_uni(self, box_lim, npts):
        # Make uniform grid for streamline plots
        x1D = np.linspace(-box_lim, box_lim, npts)
        z1D = np.linspace(-box_lim, box_lim, npts)
        X_uni, Z_uni = np.meshgrid(x1D, z1D)
        return X_uni, Z_uni
    
    def calc_RZ_uni(self, box_lim, npts):
        # Make uniform grid for streamline plots
        R1D = np.linspace(0., box_lim, npts)
        z1D = np.linspace(-box_lim, box_lim, npts)
        R_uni, Z_uni = np.meshgrid(R1D, z1D)
        return R_uni, Z_uni
    
    def get_vars2D(self, keys, iphi):
        variables = []
        for key in keys:
            if key in self.H.data.keys():
                shape = list(self.H.data[f'{key}'].shape)
                if len(shape) == 4:
                    variables.append(self.H.data[f'{key}'][0, :, :, iphi])
                if len(shape) == 5:
                    variables.append(self.H.data[f'{key}'][:, 0, :, :, iphi])
            elif key in self.H.grid.keys():
                shape = list(self.H.grid[f'{key}'].shape)
                # Note that this should not be used to retrieve r,th,phi in 2D
                if len(shape) == 4:
                    variables.append(self.H.grid[f'{key}'][0, :, :, 0])
                if len(shape) == 6:
                    variables.append(self.H.grid[f'{key}'][:, :, 0, :, :, 0])
            else:
                return ValueError("Invalid variable key.")
        return variables[0] if len(variables) == 1 else variables
    
    def set_Coords2D(self, iphi):
        ones  = np.ones_like(self.H.data['rho'])
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]
        self.Coords2D = coords.Coordinates(r2D, th2D, phi2D, 
                                           self.H.params, hr.CONSTANTS)
        return None
    
    def set_Tetrads2D(self, iphi, **kwargs):
        ones  = np.ones_like(self.H.data['rho'])
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]

        self.Tetrads2D = coords.Tetrads(r2D, th2D, phi2D, 
            self.H.params, hr.CONSTANTS, **kwargs)
        return None
    
    # Thermodynamic quantities
    def calc_rho(self, iphi):
        # Density
        rho = self.get_vars2D(['rho'], iphi)
        return rho
    
    def calc_Te(self, iphi):
        # Electron temperature
        Te = self.get_vars2D(['Te'], iphi)
        return Te
    
    def calc_Ti(self, iphi):
        # Ion temperature
        Ti = self.get_vars2D(['Ti'], iphi)
        return Ti
    
    def calc_Te_Ti(self, iphi):
        # Electron temperature, ion temperature
        Te, Ti = self.get_vars2D(['Te', 'Ti'], iphi)
        return Te, Ti
    
    def calc_sigma(self, iphi):
        # sigma, BetaI = pmag/pgas
        rho0, c = self.rho0, self.c    
        B02     = self.Field.B0 * self.Field.B0  
        keys    = ['rho', 'B', 'uu', 'gcov']
        rho, B, uu, gcov = self.get_vars2D(keys, iphi)
    
        # Calculate sigma
        b     = self.Field.calc_b(B, uu, gcov)
        b2    = self.Field.calc_b2(b, B, gcov)
        sigma = (b2 / B02) / (rho / rho0)
        return sigma
    
    def calc_BetaInv(self, iphi, **kwargs):
        # BetaI = pmag/pgas
        rho0, c = self.rho0, self.c    
        B02     = self.Field.B0 * self.Field.B0  
        keys    = ['B', 'uu', 'u_gas', 'Te', 'Ti', 'gcov']
        B, uu, u_gas, Te, Ti, gcov = self.get_vars2D(keys, iphi)
        
        # Calculate gas pressure and Beta^-1
        if (self.cgs):
            if kwargs.get('vargamma', True):
                pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
            else:
                gamma_adi = 5./3.
                pgas  = (gamma_adi - 1.) * u_gas
        else:
            gamma_adi = 5./3.
            pgas  = (gamma_adi - 1.) * u_gas
        b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        
        BetaI = ((b2 / B02) / 2.) / (pgas / (rho0 * c * c))
        return BetaI

    def calc_radiation_energy_density(self, iphi):
        # radiation energy density
        return self.get_vars2D(['u_rad'], iphi)
    
    def calc_u_t(self, iphi):
        u, gcov = self.get_vars2D(['uu', 'gcov'], iphi)
        u_t = np.einsum('ij...,j...->i...', gcov, u)[0]
        return u_t

    def calc_ion_density(self, iphi):
        if 'tr1' not in self.H.data.keys():
            raise ValueError("tr1 is required to calculate ion density")
        rho, tr1 = self.get_vars2D(['rho', 'tr1'], iphi)
        return tr1 * rho
    
    def calc_lepton_density(self, iphi):
        if 'tr2' not in self.H.data.keys():
            raise ValueError("tr2 is required to calculate lepton density")
        rho, tr2 = self.get_vars2D(['rho', 'tr2'], iphi)
        return tr2 * rho

    # Pressures
    def calc_pgas(self, iphi, **kwargs):
        if (self.two_temp and self.radiation):
            u_gas, Te, Ti = self.get_vars2D(['u_gas', 'Te', 'Ti'], iphi)
            return self.Fluid.calc_pgas(u_gas, Te, Ti)
        elif (self.two_temp and self.radiation and 
              not kwargs.get('vargamma', True)):
            u_gas = self.get_vars2D(['u_gas'], iphi)
            gamma_adi = 5./3.
            return (gamma_adi - 1.) * u_gas
        else:
            u_gas = self.get_vars2D(['u_gas'], iphi)
            gamma_adi = 5./3.
            return (gamma_adi - 1.) * u_gas

    def calc_pmag(self, iphi):
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        b2    = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)  
        return (b2 / (8. * np.pi) if self.cgs else b2/2.)
    
    def calc_prad(self, iphi):
        u_rad = self.get_vars2D(['u_rad'], iphi)
        prad  = (1./3.) * u_rad
        return prad
    
    def calc_pmag_pradgas(self, iphi, **kwargs):
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        b2    = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu) 
        keys  = ['u_gas', 'u_rad', 'Te', 'Ti']
        if (self.two_temp and self.radiation and kwargs.get('vargamma', True)):
            u_gas, u_rad, Te, Ti = self.get_vars2D(keys, iphi) 
            pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        elif (not kwargs.get('vargamma', True) and self.radiation):
            u_gas, u_rad = self.get_vars2D(['u_gas', 'u_rad'], iphi) 
            gamma_adi = 5./3.
            pgas  = (gamma_adi - 1.) * u_gas
        
        pmag  = b2 / (8. * np.pi) if self.cgs else b2 / 2.
        prad  = (1./3.) * u_rad
        return pmag / (prad + pgas)
    
    def calc_pmag_prad(self, iphi):
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        b2    = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu) 
        u_rad = self.get_vars2D(['u_rad'], iphi) 
        
        pmag  = b2 / (8. * np.pi) if self.cgs else b2 / 2.
        prad  = (1./3.) * u_rad
        return pmag / prad
    
    def calc_prad_pgas(self, iphi):  
        keys  = ['u_gas', 'u_rad', 'Te', 'Ti']
        u_gas, u_rad, Te, Ti = self.get_vars2D(keys, iphi)
        
        pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        prad  = (1./3.) * u_rad
        return prad / pgas

    def calc_rhov2_2b2(self, iphi):
        rho, B, gcov, uu = self.get_vars2D(['rho', 'B', 'gcov', 'uu'], iphi)
        self.set_Coords2D(iphi)

        b2       = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        uusph    = self.Coords2D.int_to_sph_vec(uu)
        gcov_sph = self.Coords2D.int_to_sph_cotnsr(gcov)
        uu_sph   = np.einsum('ij...,j...->i...', gcov_sph, uusph)
        # v_r^2 = v^r v_r = (u^r/u^t) * (u_r/u_t)
        v2    = (uusph[1]/uusph[0]) * (uu_sph[1]/uu_sph[0])
        rhov2 = rho * v2
        return rhov2 / (b2/(4. * np.pi))
    
    def calc_J_cross_B(self, iphi): 
        Coords = coords.Coordinates(
            self.H.grid['r'] / self.Rg, self.H.grid['theta'], self.H.grid['phi'], 
            self.H.params, hr.CONSTANTS)
        J = self.Field.calc_J(self.H.data['uu'], self.H.data['B'], 
                              self.H.grid['gcov'], self.H.grid['gdet'], frame='fluid') / self.Field.B0
        b = self.Field.calc_b(self.H.data['B'], self.H.data['uu'], self.H.grid['gcov'])/ self.Field.B0
        J_cart = Coords.int_to_cart_vec(J)
        b_cart = Coords.int_to_cart_vec(b)
        J_cross_B_vec3 = np.zeros_like(self.H.data['uu'][1:])
        J_cross_B_vec3[0] = J_cart[1] * b_cart[2] - J_cart[2] * b_cart[1]
        J_cross_B_vec3[1] = J_cart[2] * b_cart[0] - J_cart[0] * b_cart[2]
        J_cross_B_vec3[2] = J_cart[0] * b_cart[1] - J_cart[1] * b_cart[0]
        return np.sqrt(J_cross_B_vec3[0] * J_cross_B_vec3[0] + J_cross_B_vec3[1] * J_cross_B_vec3[1] + J_cross_B_vec3[2] * J_cross_B_vec3[2])[0, :, :, 0]
        #J_cross_B_vec3 = np.einsum('ijk...,j...,k...->i...', 
        #    ana.Tools.calc_Levi_Civita(3, self.H.grid['gdet']), 
        #    Coords.int_to_cart_vec(J)[1:], 
        #    Coords.int_to_cart_vec(b)[1:])
        #J_cross_B_vec4 = np.zeros_like(self.H.data['uu'])
        #J_cross_B_vec4[1:] = J_cross_B_vec3
        #return ana.Tools.dot(J_cross_B_vec4, J_cross_B_vec4, self.H.grid['gcov'])[0, :, :, 0]
 
    # Magnetic Fields
    def calc_Bphi(self, iphi):
        B = self.get_vars2D(['B'], iphi)
        self.set_Coords2D(iphi)

        # Calculate B in cylindrical coordinates
        B_cyl = self.Coords2D.int_to_cyl_vec(B)
        return B_cyl[2]

    def calc_Bz(self, iphi):
        B = self.get_vars2D(['B'], iphi)
        self.set_Coords2D(iphi)
    
        # Calculate B in cylindrical coordinates
        B_cyl = self.Coords2D.int_to_cyl_vec(B)
        return B_cyl[3]

    def calc_B_cyl(self, iphi, **kwargs):
        # Only returns BR, Bz
        B = self.get_vars2D(['B'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in cylindrical coordinates
        B_cyl = self.Coords2D.int_to_cyl_vec(B) / self.Field.B0
        if not kwargs.get('lr', False):
            return B_cyl[1], B_cyl[3]
        else:
            BR, Bz = {}, {}
            BR['r'] = B_cyl[1]
            Bz['r'] = B_cyl[3]
            BR['l'] = -B_cyl[1]
            Bz['l'] = B_cyl[3]
            if not kwargs.get('in_out', False):
                return BR, Bz
            else:
                in_out = {}
                in_out['r'] = np.sign(B[1])
                in_out['l'] = np.sign(B[1])
                return BR, Bz, in_out
    
    def calc_B_cyl_vec(self, iphi, **kwargs):
        # Returns BR, Bphi, Bz
        B = self.get_vars2D(['B'], iphi)
        self.set_Coords2D(iphi)

        # Calculate B in cylindrical coordinates
        B_cyl = self.Coords2D.int_to_cyl_vec(B)
        if not kwargs.get('lr', False):
            return B_cyl[1], B_cyl[2], B_cyl[3]
        else:
            BR, Bphi, Bz = {}, {}, {}
            BR['r']   = B_cyl[1]
            Bphi['r'] = B_cyl[2]
            Bz['r']   = B_cyl[3]
            BR['l']   = -B_cyl[1]
            Bphi['l'] = B_cyl[2]
            Bz['l']   = B_cyl[3]
            if not kwargs.get('in_out', False):
                return BR, Bphi, Bz
            else:
                in_out = {}
                in_out['r'] = np.sign(B[1])
                in_out['l'] = np.sign(B[1])
                return BR, Bphi, Bz, in_out

    def calc_B_cart(self, iphi, **kwargs):
        # Only returns Bx, Bz
        B = self.get_vars2D(['B'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in cylindrical coordinates
        B_cart = self.Coords2D.int_to_cart_vec(B) / self.Field.B0
        if not kwargs.get('lr', False):
            return B_cart[1], B_cart[3]
        else:
            Bx, Bz = {}, {}
            Bx['r'] = B_cart[1]
            Bz['r'] = B_cart[3]
            Bx['l'] = -B_cart[1]
            Bz['l'] = B_cart[3]
            if not kwargs.get('in_out', False):
                return Bx, Bz
            else:
                in_out = {}
                in_out['r'] = np.sign(B[1])
                in_out['l'] = np.sign(B[1])
                return Bx, Bz, in_out

    def calc_bpol(self, iphi):
        # Returns signed poloidal field sqrt(BR^2 + Bz^2)
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        self.set_Coords2D(iphi)

        b    = self.Field.calc_b(B, uu, gcov)
        bcyl = self.Coords2D.int_to_cyl_vec(b, dtype=np.float64) 
        bpol = np.copy(bcyl)
        bpol[2] = 0.
        gcov_cyl = self.Coords2D.int_to_cyl_cotnsr(gcov)
        return np.sign(bcyl[3]) * np.sqrt(ana.Tools.dot(bpol, bpol, gcov_cyl))

    def calc_btor(self, iphi):
        # Returns toroidal field in orthonormal cylindrical coordinates
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        self.set_Coords2D(iphi)

        b    = self.Field.calc_b(B, uu, gcov)
        bcyl = self.Coords2D.int_to_cyl_vec(b, dtype=np.float64) 
        return bcyl[2]

    def calc_bpol_btor(self, iphi):
        # Returns signed ratio of poloidal to toroidal field strength
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        self.set_Coords2D(iphi)

        b    = self.Field.calc_b(B, uu, gcov)
        bcyl = self.Coords2D.int_to_cyl_vec(b, dtype=np.float64) 
        bpol = np.copy(b)
        btor = np.copy(b)
        bpol[0], bpol[3]  = 0., 0.
        btor[0], btor[1], btor[2] = 0., 0., 0.

        sgn     = np.sign(bcyl[3])
        b2pol   = self.Field.calc_b2(bpol, bpol, gcov)
        b2tor   = self.Field.calc_b2(btor, btor, gcov)
        return sgn * np.sqrt(b2pol / b2tor)

    # Velocities
    def calc_vr(self, iphi):
        uu = self.get_vars2D(['uu'], iphi)
        return uu[1] / uu[0]

    def calc_vr_sph(self, iphi):
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in spherical coordinates
        uu_sph = self.Coords2D.int_to_sph_vec(uu)
        return self.c * uu_sph[1] / uu_sph[0]   

    def calc_vphi_cyl(self, iphi):
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in Cartesian coordinates
        uu_cyl = self.Coords2D.int_to_cyl_vec(uu)
        return self.c * uu_cyl[2] / uu_cyl[0]
        
    def calc_vR_vKep(self, iphi):
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        # Calculate v in cylindridcal coordinates
        uu_cyl = self.Coords2D.int_to_cyl_vec(uu) 
        vR     = self.c * uu_cyl[1] / uu_cyl[0]
        
        vKep   = np.sqrt(self.G * self.MBH * self.Msun / (self.Rg * self.r))
        return vR / vKep[0, :, :, 0]

    def calc_uu_cyl(self, iphi, **kwargs):
        # Returns uuR, uuz
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in cylindrical coordinates
        uu_cyl = self.Coords2D.int_to_cyl_vec(uu) / self.c
        if not kwargs.get('lr', False):
            return uu_cyl[1], uu_cyl[3]
        else:
            uuR, uuz = {}, {}
            uuR['r'] = uu_cyl[1]
            uuz['r'] = uu_cyl[3]
            uuR['l'] = -uu_cyl[1]
            uuz['l'] = uu_cyl[3]
            if not kwargs.get('in_out', False):
                return uuR, uuz
            else:
                in_out = {}
                in_out['r'] = np.sign(uu[1])
                in_out['l'] = np.sign(uu[1])
                return uuR, uuz, in_out
    
    def calc_v_cyl_vec(self, iphi, **kwargs):
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in cylindrical coordinates
        uu_cyl = self.Coords2D.int_to_cyl_vec(uu) / self.c
        if not kwargs.get('lr', False):
            return (
                uu_cyl[1]/uu_cyl[0], 
                uu_cyl[2]/uu_cyl[0], 
                uu_cyl[3]/uu_cyl[0]
            )
        else:
            vR, vphi, vz = {}, {}, {}
            vR['r']   = uu_cyl[1] / uu_cyl[0]
            vphi['r'] = uu_cyl[2] / uu_cyl[0]
            vz['r']   = uu_cyl[3] / uu_cyl[0]
            vR['l']   = -uu_cyl[1] / uu_cyl[0]
            vphi['l'] = uu_cyl[2] / uu_cyl[0]
            vz['l']   = uu_cyl[3] / uu_cyl[0]
            if not kwargs.get('in_out', False):
                return vR, vphi, vz
            else:
                in_out = {}
                in_out['r'] = np.sign(uu[1])
                in_out['l'] = np.sign(uu[1])
                return vR, vphi, vz, in_out
    
    def calc_uu_cart(self, iphi, **kwargs):
        # Returns uux, uuz
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in Cartesian coordinates
        uu_cart = self.Coords2D.int_to_cart_vec(uu) / self.c
        if not kwargs.get('lr', False):
            return uu_cart[1], uu_cart[3]
        else:
            uux, uuz = {}, {}
            uux['r'] = uu_cart[1]
            uuz['r'] = uu_cart[3]
            uux['l'] = -uu_cart[1]
            uuz['l'] = uu_cart[3]
            if not kwargs.get('in_out', False):
                return uux, uuz
            else:
                in_out = {}
                in_out['r'] = np.sign(uu[1])
                in_out['l'] = np.sign(uu[1])
                return uux, uuz, in_out
    
    def calc_vpol_vtor(self, iphi):
        uu, gcov = self.get_vars2D(['uu', 'gcov'], iphi)
        self.set_Coords2D(iphi)

        uucyl = self.Coords2D.int_to_cyl_vec(uu, dtype=np.float64) 
        uupol = np.copy(uu)
        uutor = np.copy(uu)
        uupol[0], uupol[3]  = 0., 0.
        uutor[0], uutor[1], uutor[2] = 0., 0., 0.

        sgn     = np.sign(uucyl[3])
        v2pol   = ana.Tools.dot(uupol, uupol, gcov)
        v2tor   = ana.Tools.dot(uutor, uutor, gcov)
        return sgn * np.sqrt(v2pol / v2tor)
    
    # Radiation 4-velocity
    def calc_uu_rad_cyl(self, iphi, **kwargs):
        # Returns radiation uuR, uuz
        uu_rad = self.get_vars2D(['uu_rad'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu_rad in cylindrical coordinates
        uu_rad_cyl = self.Coords2D.int_to_cyl_vec(uu_rad) / self.c
        if not kwargs.get('lr', False):
            return uu_rad_cyl[1], uu_rad_cyl[3]
        else:
            uu_radR, uu_radz = {}, {}
            uu_radR['r'] = uu_rad_cyl[1]
            uu_radz['r'] = uu_rad_cyl[3]
            uu_radR['l'] = -uu_rad_cyl[1]
            uu_radz['l'] = uu_rad_cyl[3]
            if not kwargs.get('in_out', False):
                return uu_radR, uu_radz
            else:
                in_out = {}
                in_out['r'] = np.sign(uu_rad[1])
                in_out['l'] = np.sign(uu_rad[1])
                return uu_radR, uu_radz, in_out

    def calc_uu_rad_cart(self, iphi, **kwargs):
        # Returns radiation uux, uuz 
        uu_rad = self.get_vars2D(['uu_rad'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu_rad in Cartesian coordinates
        uu_rad_cart = self.Coords2D.int_to_cart_vec(uu_rad) / self.c
        if not kwargs.get('lr', False):
            return uu_rad_cart[1], uu_rad_cart[3]
        else:
            uu_radx, uu_radz = {}, {}
            uu_radx['r'] = uu_rad_cart[1]
            uu_radz['r'] = uu_rad_cart[3]
            uu_radx['l'] = -uu_rad_cart[1]
            uu_radz['l'] = uu_rad_cart[3]
            if not kwargs.get('in_out', False):
                return uu_radx, uu_radz
            else:
                in_out = {}
                in_out['r'] = np.sign(uu_rad[1])
                in_out['l'] = np.sign(uu_rad[1])
                return uu_radx, uu_radz, in_out
     
    # Calculate Edot from stress-energy tensors
    # Minus sign is included such that Edot is positive
    # Options: dictionary=True returns dictionary Edot
    # all=True returns Hydro + EM + Radiation Edot
    # otherwise, set hydro, em, and rad flags separately
    def calc_Edot(self, iphi, **kwargs):
        all = kwargs.get('all', False)
        if (all):
            hydro = True
            em    = True
            radi  = True if self.radiation else False
        else:
            hydro = kwargs.get('hydro', True) 
            em    = kwargs.get('em',    True) 
            radi  = kwargs.get('rad',   True) if self.radiation else False
        data2D = {}
        gdet, gcov, gcon = self.get_vars2D(['gdet', 'gcov', 'gcon'], iphi)
        data2D['gdet']   = gdet
        data2D['gcov']   = gcov
        data2D['gcon']   = gcon
        if (hydro or em):
            uu            = self.get_vars2D(['uu'], iphi)
            data2D['uu']  = uu
        if (hydro):
            rho, u_gas = self.get_vars2D(['rho', 'u_gas'], iphi)
            if self.two_temp:
                Te, Ti = self.get_vars2D(['Te', 'Ti'], iphi)
                pgas    = self.Fluid.calc_pgas(u_gas, Te, Ti)
            else:
                pgas    = (5./3. - 1.) * u_gas
            data2D['rho']   = rho
            data2D['u_gas'] = u_gas
            data2D['pgas']  = pgas
        if (em):
            B           = self.get_vars2D(['B'], iphi)
            data2D['B'] = B
        if (radi):
            u_rad, uu_rad    = self.get_vars2D(['u_rad', 'uu_rad'], iphi)
            data2D['u_rad']  = u_rad
            data2D['uu_rad'] = uu_rad 
        
        if (hydro):
            THD = self.Fluid.calc_THD(**data2D, upper_lower=True, index=[1,0])
        if (em):
            TEM = self.Field.calc_TEM(**data2D, upper_lower=True, index=[1,0])
        if (radi):
            TRAD = self.Rad.calc_TRAD(**data2D, upper_lower=True, index=[1,0])
        
        ones = np.ones_like(self.r * self.th * self.phi)
        Area = 4 * np.pi * np.square(self.Rg * self.r * ones)[0, :, :, iphi]
        if kwargs.get('dictionary', False):
            Edot = {}
            if (hydro):                Edot['HD']      = THD
            if (em):                   Edot['EM']      = TEM
            if (radi):                 Edot['RAD']     = TRAD
            if (hydro and em):         Edot['MHD']     = THD + TEM
            if (hydro and em and rad): Edot['MHD_RAD'] = THD + TEM + TRAD

            c = self.c
            for key in Edot.keys():
                Edot[f'{key}'] = -self.Acc.upscale64(Edot[f'{key}'], Area * c)
            return Edot
        else:
            T_MHD_RAD = np.zeros_like(ones)[0, :, :, iphi]
            if (hydro): T_MHD_RAD += THD
            if (em)   : T_MHD_RAD += TEM
            if (radi) : T_MHD_RAD += TRAD
            return -self.Acc.upscale64(T_MHD_RAD, Area * c)
    
    # Stresses
    def calc_Rphi_stress(self, iphi, **kwargs):
        rho0, c, B0  = self.rho0, self.c, self.Field.B0
        all = kwargs.get('all', False)
        if (all):
            Reynolds = True
            Maxwell  = True
            Radiation   = True if self.radiation else False
        else:
            Reynolds  = kwargs.get('Reynolds',  True) 
            Maxwell   = kwargs.get('Maxwell',   True) 
            Radiation = kwargs.get('Radiation', True) if self.radiation else False
        rho, uu, B, gcov = self.get_vars2D(['rho', 'uu', 'B', 'gcov'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu and b in cylindridcal coordinates
        b      = self.Field.calc_b(B, uu, gcov)
        b_cyl  = self.Coords2D.int_to_cyl_vec(b)
        uu_cyl = self.Coords2D.int_to_cyl_vec(uu)

        # Get mean 4-velocities
        uuR_mean   = kwargs.get('uuR_mean', 0.)
        uuphi_mean = kwargs.get('uuphi_mean', 0.)

        if (Reynolds):
            TRphi_Rey = rho * (uu_cyl[1] - uuR_mean) * (uu_cyl[2] - uuphi_mean)
        if (Maxwell):
            TRphi_Max = -(b_cyl[1] / B0) * (b_cyl[2] / B0) * (rho0 * c * c)
        if (Radiation):
            u_rad, uu_rad = self.get_vars2D(['u_rad', 'uu_rad'], iphi)
            uu_rad_cyl    = self.Coords2D.int_to_cyl_vec(uu_rad)
            duR           = (uu_rad_cyl[1] - uuR_mean) / c
            duphi         = (uu_rad_cyl[2] - uuphi_mean) / c
            TRphi_Rad     = u_rad * duR * duphi
        
        TRphi = np.zeros_like(rho)
        if (kwargs.get('dictionary', False)):
            stresses = {}
            if (Reynolds):  stresses['Reynolds']  = TRphi_Rey
            if (Maxwell):   stresses['Maxwell']   = TRphi_Max
            if (Radiation): stresses['Radiation'] = TRphi_Rad
            if (Reynolds and Maxwell):       
                stresses['MHD'] = TRphi_Rey + TRphi_Max
            if (Reynolds and Maxwell and Radiation): 
                stresses['MHD_RAD'] = TRphi_Rey + TRphi_Max + TRphi_Rad
            return stresses
        else:
            if (Reynolds):  TRphi += TRphi_Rey
            if (Maxwell):   TRphi += TRphi_Max
            if (Radiation): TRphi += TRphi_Rad
            return TRphi

    def calc_zphi_stress(self, iphi, **kwargs):
        rho0, c, B0  = self.rho0, self.c, self.Field.B0
        all = kwargs.get('all', False)
        if (all):
            Reynolds = True
            Maxwell  = True
            Radiation   = True if self.radiation else False
        else:
            Reynolds  = kwargs.get('Reynolds',  True) 
            Maxwell   = kwargs.get('Maxwell',   True) 
            Radiation = kwargs.get('Radiation', True) if self.radiation else False
        rho, uu, B, gcov = self.get_vars2D(['rho', 'uu', 'B', 'gcov'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu and b in cylindridcal coordinates
        b      = self.Field.calc_b(B, uu, gcov)
        b_cyl  = self.Coords2D.int_to_cyl_vec(b)
        uu_cyl = self.Coords2D.int_to_cyl_vec(uu)

        # Get mean 4-velocities
        uuz_mean   = kwargs.get('uuz_mean', 0.)
        uuphi_mean = kwargs.get('uuphi_mean', 0.)

        if (Reynolds):
            Tzphi_Rey = rho * (uu_cyl[3] - uuz_mean) * (uu_cyl[2] - uuphi_mean)
        if (Maxwell):
            Tzphi_Max = -(b_cyl[3] / B0) * (b_cyl[2] / B0) * (rho0 * c * c)
        if (Radiation):
            u_rad, uu_rad = self.get_vars2D(['u_rad', 'uu_rad'], iphi)
            uu_rad_cyl    = self.Coords2D.int_to_cyl_vec(uu_rad)
            duR           = (uu_rad_cyl[3] - uuz_mean) / c
            duphi         = (uu_rad_cyl[2] - uuphi_mean) / c
            Tzphi_Rad     = u_rad * duR * duphi
        
        Tzphi = np.zeros_like(rho)
        if (kwargs.get('dictionary', False)):
            stresses = {}
            if (Reynolds):  stresses['Reynolds']  = Tzphi_Rey
            if (Maxwell):   stresses['Maxwell']   = Tzphi_Max
            if (Radiation): stresses['Radiation'] = Tzphi_Rad
            if (Reynolds and Maxwell):       
                stresses['MHD'] = Tzphi_Rey + Tzphi_Max
            if (Reynolds and Maxwell and Radiation): 
                stresses['MHD_RAD'] = Tzphi_Rey + Tzphi_Max + Tzphi_Rad
            return stresses
        else:
            if (Reynolds):  Tzphi += Tzphi_Rey
            if (Maxwell):   Tzphi += Tzphi_Max
            if (Radiation): Tzphi += Tzphi_Rad
            return Tzphi

    # Cooling rates
    # Inverse-Compton cooling rate
    def calc_qIC(self, iphi, **kwargs):
        rho, Te, u_rad = self.get_vars2D(['rho', 'Te', 'u_rad'], iphi)
        uu, gcov       = self.get_vars2D(['uu', 'gcov'], iphi)
        u_t            = np.einsum('ij...,j...->i...', gcov, uu)[0] / self.c
        return -u_t * self.Rad.calc_qIC(rho, Te, u_rad, **kwargs) 

    # Coulomb cooling rate
    def calc_qie(self, iphi, **kwargs):
        mid         = kwargs.get('midplane', False)
        rho, Ti, Te = self.H.data['rho'], self.H.data['Ti'], self.H.data['Te']
        uu, gcov    = self.H.data['uu'], self.H.grid['gcov']
        u_t         = np.einsum('ij...,j...->i...', gcov, uu)[0] / self.c
        qie = self.Fluid.calc_qie(rho, Te, Ti, **kwargs)
        return -(u_t * qie)[0, :, :, iphi] if not mid else -(u_t * qie)[0]

    # Synchrotron cooling rate
    def calc_qsync(self, iphi, **kwargs):
        rho, Te, u_rad = self.get_vars2D(['rho', 'Te', 'u_rad'], iphi)
        B, uu, gcov    = self.get_vars2D(['B', 'uu', 'gcov'], iphi)
        u_t            = np.einsum('ij...,j...->i...', gcov, uu)[0] / self.c
        # b2 is required for synchrotron opacities
        if 'b2' not in kwargs:
            b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        else:
            b2 = kwargs.get('b2')
        opacities = dict.fromkeys(['synchrotron'])
        sync_args = {
            'opacities' : opacities, 
            'b2'        : b2, 
            'XRB'       : True,
            'RIAF'      : kwargs.get('RIAF', False)
        }
        return -u_t * self.Rad.calc_q_emi(rho, Te, u_rad, **sync_args)

    # Dissipation rate in the coordinate frame
    def calc_qdiss(self, iphi):
        qdiss, uu, gcov = self.get_vars2D(['dissipation', 'uu', 'gcov'], iphi)
        u_t             = np.einsum('ij...,j...->i...', gcov, uu)[0] / self.c
        return -u_t * qdiss

    # Radiation source term in the coordinate frame (G_t)
    # Note that the radiation is sourced by -G_t
    def calc_qrad(self, iphi):
        qrad = self.get_vars2D(['rad_src'], iphi)
        return -qrad
    
    # Ion heating fraction
    def calc_delta_i(self, iphi):
        rho, Te, Ti, u_gas = self.get_vars2D(['rho', 'Te', 'Ti', 'u_gas'], iphi)
        B, uu, gcov        = self.get_vars2D(['B', 'uu', 'gcov'], iphi)
        delta_e            = self.Fluid.calc_delta_e(rho, u_gas, Te, Ti, B, gcov, uu=uu)
        return 1. - delta_e

    # 1 - (n_e / n_i)
    def calc_ion_fraction(self, iphi):
        if 'tr1' not in self.H.data.keys() or 'tr2' not in self.H.data.keys():
            raise ValueError("tr1 and tr2 is required to calculate ion fraction")
        tr1, tr2 = self.get_vars2D(['tr1', 'tr2'], iphi)
        return (tr1 > tr2) * (tr1 > 0.) * (1. - (tr2 / tr1))

    # Volume
    def calc_Vol(self, iphi):
        gdet = self.get_vars2D(['gdet'], iphi)
        dx1  = self.H.params['dx1']
        dx2  = self.H.params['dx2']
        dx3  = self.H.params['dx3']
        return (self.Rg ** 3) * gdet * dx1 * dx2 * dx3
    
    # Current sheet analysis
    def calc_B_rec(self, iphi, R_uni, Z_uni):
        B = self.get_vars2D(['B'], iphi)
        self.set_Coords2D(iphi)

        B_sph = self.Coords2D.int_to_sph_vec(B)

        # Convert to locally flat (Minkowski) frame
        self.set_Tetrads2D(iphi, system='Takahashi_Kerr_Schild')
        B_sph_flat = np.einsum('ij...,j...->i...', self.Tetrads2D.e, B_sph)

        # Convert to cylindrical coordinates
        self.set_Coords2D(iphi)
        B_cyl_flat = self.Coords2D.ortho_sph_to_cyl_vec(B_sph_flat)

        # Interpolate all fields to a uniform grid
        ones = np.ones(self.H.data['rho'].shape)
        r2D  = (ones * self.r)[  0, :, :, iphi]
        th2D = (ones * self.th)[ 0, :, :, iphi]

        R2D  = r2D * np.sin(th2D)
        Z2D  = r2D * np.cos(th2D)

        BR   = griddata((R2D.ravel(), Z2D.ravel()), B_cyl_flat[1].ravel(), 
                        (R_uni, Z_uni), method='cubic')
        Bphi = griddata((R2D.ravel(), Z2D.ravel()), B_cyl_flat[2].ravel(), 
                        (R_uni, Z_uni), method='cubic')
        Bz   = griddata((R2D.ravel(), Z2D.ravel()), B_cyl_flat[3].ravel(), 
                        (R_uni, Z_uni), method='cubic')
        return BR, Bphi, Bz
 
    def calc_mag_energy_rec(self, iphi, R_uni, Z_uni):
        B, uu, gcov = self.get_vars2D(['B', 'uu', 'gcov'], iphi)
        rho0, c = self.rho0, self.c
        B02     = self.Field.B0 * self.Field.B0
        b2  = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)

        # Interpolate energy density to a uniform grid
        ones = np.ones(self.H.data['rho'].shape)
        r2D  = (ones * self.r)[  0, :, :, iphi]
        th2D = (ones * self.th)[ 0, :, :, iphi]

        R2D  = r2D * np.sin(th2D)
        Z2D  = r2D * np.cos(th2D)

        Emag_uni = griddata((R2D.ravel(), Z2D.ravel()), 
                            (0.5 * (b2 / B02) * rho0 * c * c).ravel(), 
                            (R_uni, Z_uni), method='cubic')
        return Emag_uni

    def calc_vrec_z(self, iphi, R_uni, Z_uni):
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        uu_sph = self.Coords2D.int_to_sph_vec(uu)

        # Convert to locally flat (Minkowski) frame
        self.set_Tetrads2D(iphi, system='Takahashi_Kerr_Schild')
        uu_sph_flat = np.einsum('ij...,j...->i...', self.Tetrads2D.e, uu_sph)

        # Convert to cylindrical coordinates
        uu_cyl_flat = self.Coords2D.ortho_sph_to_cyl_vec(uu_sph_flat)

        # Interpolate to the uniform grid
        ones = np.ones(self.H.data['rho'].shape)
        r2D  = (ones * self.r)[  0, :, :, iphi]
        th2D = (ones * self.th)[ 0, :, :, iphi]

        R2D = r2D * np.sin(th2D)
        Z2D = r2D * np.cos(th2D)

        vz_uni = griddata((R2D.ravel(), Z2D.ravel()), 
                          (uu_cyl_flat[3] / uu_cyl_flat[0]).ravel(), 
                          (R_uni, Z_uni), method='cubic')
        return self.c * vz_uni 
    
    def calc_Poynt_rec_z(self, iphi, R_uni, Z_uni):
        B, uu, gcov, gcon = self.get_vars2D(['B', 'uu', 'gcov', 'gcon'], iphi)

        # Calculate Poynting flux in spherical coordinates
        self.set_Coords2D(iphi)
        Tmu_t_sph = self.Coords2D.int_to_sph_vec(
            self.Field.calc_TEM(uu, B, gcov, gcon, upper_lower=True)[:, 0])
        
        # Convert to locally flat (Minkowski) frame
        self.set_Tetrads2D(iphi, system='Takahashi_Kerr_Schild')
        Tmu_t_sph_flat = np.einsum(
            'ij...,j...->i...', self.Tetrads2D.e, Tmu_t_sph)
        
        # Convert spherical to cylindrical coordinates
        Tmu_t_cyl_flat = self.Coords2D.ortho_sph_to_cyl_vec(Tmu_t_sph_flat)

        # Interpolate to the uniform grid
        ones = np.ones(self.H.data['rho'].shape)
        r2D  = (ones * self.r)[  0, :, :, iphi]
        th2D = (ones * self.th)[ 0, :, :, iphi]

        R2D = r2D * np.sin(th2D)
        Z2D = r2D * np.cos(th2D)
        Tz_t_uni = griddata((R2D.ravel(), Z2D.ravel()), 
                            (Tmu_t_cyl_flat[3]).ravel(), 
                            (R_uni, Z_uni), method='cubic')
        return -Tz_t_uni

    def calc_Rad_rec_z(self, iphi, R_uni, Z_uni):
        B, uu, gcov, gcon = self.get_vars2D(['B', 'uu', 'gcov', 'gcon'], iphi)

        # Calculate Poynting flux in spherical coordinates
        self.set_Coords2D(iphi)
        Tmu_t_sph = self.Coords2D.int_to_sph_vec(
            self.Field.calc_TEM(uu, B, gcov, gcon, upper_lower=True)[:, 0])
        
        # Convert to locally flat (Minkowski) frame
        self.set_Tetrads2D(iphi, system='Takahashi_Kerr_Schild')
        Tmu_t_sph_flat = np.einsum(
            'ij...,j...->i...', self.Tetrads2D.e, Tmu_t_sph)
        
        # Convert spherical to cylindrical coordinates
        Tmu_t_cyl_flat = self.Coords2D.ortho_sph_to_cyl_vec(Tmu_t_sph_flat)

        # Interpolate to the uniform grid
        ones = np.ones(self.H.data['rho'].shape)
        r2D  = (ones * self.r)[  0, :, :, iphi]
        th2D = (ones * self.th)[ 0, :, :, iphi]

        R2D = r2D * np.sin(th2D)
        Z2D = r2D * np.cos(th2D)
        Tz_t_uni = griddata((R2D.ravel(), Z2D.ravel()), 
                            (Tmu_t_cyl_flat[3]).ravel(), 
                            (R_uni, Z_uni), method='cubic')
        return -Tz_t_uni
    
    # Plasma timescales
    # Ion-electron temperature equilibration
    def calc_teq(self, iphi, **kwargs):
        mid         = kwargs.get('midplane', False)
        rho, Ti, Te = self.H.data['rho'], self.H.data['Ti'], self.H.data['Te']
        teq         = self.Fluid.calc_teq(rho, Te, Ti, **kwargs)
        return teq[0, :, :, iphi] if not mid else teq[0]
    
    # Ion-ion collisions
    def calc_tii(self, iphi, **kwargs):
        mid         = kwargs.get('midplane', False)
        rho, Ti, Te = self.H.data['rho'], self.H.data['Ti'], self.H.data['Te']
        tii         = self.Fluid.calc_tii(rho, Te, Ti, **kwargs)
        return tii[0, :, :, iphi] if not mid else tii[0]
    
    # Electron-electron collisions
    def calc_tee(self, iphi, **kwargs):
        mid         = kwargs.get('midplane', False)
        rho, Ti, Te = self.H.data['rho'], self.H.data['Ti'], self.H.data['Te']
        tee         = self.Fluid.calc_tee(rho, Te, Ti, **kwargs)
        return tee[0, :, :, iphi] if not mid else tee[0]
    
    # Inverse Compton
    def calc_tIC(self, iphi, **kwargs):
        mid            = kwargs.get('midplane', False)
        rho, Te, u_rad = self.H.data['rho'], self.H.data['Te'], self.H.data['u_rad']
        tIC            = self.Rad.calc_tIC(rho, Te, u_rad, **kwargs)
        return tIC[0, :, :, iphi] if not mid else tIC[0]
        
    # Optical depths
    def calc_tau(self, opacities, iphi, **kwargs):
        tau = self.Acc.calc_tau(opacities=opacities, sigma_cut=3., **kwargs)
        return tau[0, :, :, iphi]
    
    # Opacities
    def calc_kappa_es(self, iphi, **kwargs):
        rho, Te, u_rad = self.get_vars2D(['rho', 'Te', 'u_rad'], iphi)
        opacities = dict.fromkeys(['electron_scattering'])
        Opac = rad.Opacity(opacities, rho, Te, u_rad, self.H.params, 
                           hr.CONSTANTS, mechanism='emission')
        return Opac.opacities['electron_scattering']

    def calc_kappa_eff(self, iphi, **kwargs):
        rho, Te, u_rad = self.get_vars2D(['rho', 'Te', 'u_rad'], iphi)
        opacities = dict.fromkeys(['electron_scattering', 'bound_free', 
            'CHIANTI', 'free_free', 'hydrogen', 'molecular', 'synchrotron'])
        if 'b2' not in kwargs:
            B, uu, gcov = self.get_vars2D(['B', 'uu', 'gcov'], iphi)
            b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        else:
            b2 = kwargs.get('b2')
        Opac = rad.Opacity(opacities, rho, Te, u_rad, self.H.params, 
            hr.CONSTANTS, mechanism='absorption', b2=b2, XRB=True)
        kappa_abs = Opac.opacities['absorption_total']
        kappa_es  = Opac.opacities['electron_scattering']
        kappa_eff = np.sqrt(kappa_abs * (kappa_abs + kappa_es))
        return kappa_eff
    
    def calc_kappa_sync(self, iphi, **kwargs):
        rho, Te, u_rad = self.get_vars2D(['rho', 'Te', 'u_rad'], iphi)
         # b2 is required for synchrotron opacities
        if 'b2' not in kwargs:
            B, uu, gcov = self.get_vars2D(['B', 'uu', 'gcov'], iphi)
            b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        else:
            b2 = kwargs.get('b2')
        opacities = dict.fromkeys(['synchrotron'])
        if not kwargs.get('RIAF', False):
            Opac = rad.Opacity(opacities, rho, Te, u_rad, self.H.params, 
                               hr.CONSTANTS, mechanism='emission', b2=b2, XRB=True)
        else:
            Opac = rad.RIAF_Opacity(opacities, rho, Te, u_rad, self.H.params, 
                                    hr.CONSTANTS, mechanism='emission', b2=b2, XRB=True)
        return Opac.opacities['synchrotron_emi']
    
    def calc_kappa_ff(self, iphi, **kwargs):
        rho, Te, u_rad = self.get_vars2D(['rho', 'Te', 'u_rad'], iphi)
         # b2 is required for synchrotron opacities
        opacities = dict.fromkeys(['free_free'])
        Opac = rad.Opacity(opacities, rho, Te, u_rad, self.H.params, 
                           hr.CONSTANTS, mechanism='emission', XRB=True)
        return Opac.opacities['free_free_emi']

    # Orbital frequency
    def calc_OmegaK(self):
        return self.Analytic.calc_OmegaK(
            self.MBH, self.H.params['a'], self.r)[0, :, 0, :]
    
    # Scale heights ratios/ Disk thicknesses (H/r)
    def calc_HoR_thm(self, iphi):
        Te, Ti, uu = self.get_vars2D(['Te', 'Ti', 'uu'], iphi)
        c    = self.c
        kB   = hr.CONSTANTS['kB']
        mp   = hr.CONSTANTS['mp'] 

        x_AB = hr.CONSTANTS['X_AB']
        y_AB = hr.CONSTANTS['Y_AB']

        # Equation 11 in Sadowski+ (2017)
        mu_i = 4.0 / (4.0 * x_AB + y_AB)
        
        cs = np.sqrt(kB * (Ti + Te) / (mu_i * mp))
        self.set_Coords2D(iphi)

        # Calculate uu in orthonormal spherical coordinates
        uu_sph = self.Coords2D.int_to_ortho_sph_vec(uu)
        vphi   = c * (uu_sph[3] / uu_sph[0])
        return cs/vphi

    def calc_HoR_mag(self, iphi):
        rho, uu      = self.get_vars2D(['rho', 'uu'], iphi)
        u_gas, u_rad = self.get_vars2D(['u_gas', 'u_rad'], iphi)
        B, gcov      = self.get_vars2D(['B', 'gcov'], iphi)
        b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        c  = self.c
        gamma_adi = 5./3.
        pgas = (gamma_adi - 1.) * u_gas
        w  = rho * c * c + b2 / (4. * np.pi) + u_gas + pgas + (4./3.) * u_rad
        # Calculate uu in orthonormal spherical coordinates
        uu_sph = self.Coords2D.int_to_ortho_sph_vec(uu)
        vphi   = c * (uu_sph[3] / uu_sph[0])
        return (c / vphi) * np.sqrt(b2 / (4. * np.pi * w))

    def calc_HoR_rad(self, iphi):
        rho, uu      = self.get_vars2D(['rho', 'uu'], iphi)
        u_gas, u_rad = self.get_vars2D(['u_gas', 'u_rad'], iphi)
        c  = self.c
        gamma_adi = 5./3.
        pgas  = (gamma_adi - 1.) * u_gas
        prad  = (1./3.) * u_rad
        ptot  = pgas + prad
        rhoc2 = rho * c * c
        cs_rad = c * np.sqrt((4./3.) * ptot / (rhoc2 + u_rad + u_gas))
        # Calculate uu in orthonormal spherical coordinates
        uu_sph = self.Coords2D.int_to_ortho_sph_vec(uu)
        vphi   = c * (uu_sph[3] / uu_sph[0])
        return cs_rad / vphi
    
    def calc_HoR_tot(self, iphi):
        rho, Te, Ti, uu = self.get_vars2D(['rho', 'Te', 'Ti', 'uu'], iphi)
        u_gas, u_rad    = self.get_vars2D(['u_gas', 'u_rad'], iphi)
        B, gcov         = self.get_vars2D(['B', 'gcov'], iphi)

        # Thermal
        kB   = hr.CONSTANTS['kB']
        mp   = hr.CONSTANTS['mp'] 
        x_AB = hr.CONSTANTS['X_AB']
        y_AB = hr.CONSTANTS['Y_AB']
        # Equation 11 in Sadowski+ (2017)
        mu_i = 4.0 / (4.0 * x_AB + y_AB)
        cs     = np.sqrt(kB * (Ti + Te) / (mu_i * mp))

        # Radiation
        c      = self.c
        gamma_adi = 5./3.
        pgas   = (gamma_adi - 1.) * u_gas
        prad   = (1./3.) * u_rad
        ptot   = pgas + prad
        rhoc2  = rho * c * c
        cs_rad = c * np.sqrt((4./3.) * ptot / (rhoc2 + u_rad + u_gas))
        
        # Magnetic
        b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        w  = rhoc2 + (b2 / (4. * np.pi)) + u_gas + pgas + (4./3.) * u_rad
        vA = c * np.sqrt(b2 / (4. * np.pi * w))
        vtot = np.sqrt(vA * vA + cs_rad * cs_rad + cs * cs)
        # Calculate uu in orthonormal spherical coordinates
        uu_sph = self.Coords2D.int_to_ortho_sph_vec(uu)
        vphi   = c * (uu_sph[3] / uu_sph[0])
        return vtot/vphi

# Compute quantities at constant theta
class Theta_Slice(Phi_Slice):
    def __init__(self, directory, fnum, **kwargs):
        self.H = hr.Read(directory, **kwargs)
        self.H.read_grid(fnum, direct=True)
        self.H.read_prims(fnum, skip_read_params=True)

# Compute quantities in the miplane (averaging over two nearest cells)
class Midplane_Slice(Phi_Slice):
    def __init__(self, directory, fnum, **kwargs):
        H = hr.Read(directory, **kwargs)
        H.read_params(fnum)
        nth = int(H.params['nb2'] * H.params['bs2'] * H.upscalex2 / H.lowres2)
        if 'j_min' in kwargs: kwargs.pop('j_min')
        if 'j_max' in kwargs: kwargs.pop('j_max')

        self.H = hr.Read(directory, j_min = nth/2-1, j_max = nth/2+1, **kwargs)
        self.H.read_grid( fnum, direct=True)
        self.H.read_prims(fnum, skip_read_params=True)

        self.cgs  = kwargs.get('cgs_data', True)
        self.MBH  = kwargs.get('MBH', 10.0) if self.cgs else 1.
        self.Msun = hr.CONSTANTS['Msun']
        self.rho0 = self.H.params['density_scale'] if self.cgs else 1. # g/cm^3
        self.c    = hr.CONSTANTS['c'] if self.cgs else 1. # cm/s
        self.G    = hr.CONSTANTS['G'] if self.cgs else 1. # cm^3/g s^2
        self.Rg   = (self.G * self.MBH * self.Msun / (self.c * self.c)
                     if self.cgs else 1.)

        self.r    = self.H.grid['r'] / self.Rg  
        self.th   = self.H.grid['theta']
        self.phi  = self.H.grid['phi']
        
        self.two_temp  = True if 'Te' and 'Ti' in self.H.data else False
        self.radiation = True if 'u_rad' and 'uu_rad' in self.H.data else False

        acc_args = {
            'grid'      : self.H.grid,
            'data'      : self.H.data,
            'pars'      : self.H.params,
            'CONSTANTS' : hr.CONSTANTS,
            'cgs'       : self.cgs,
            'MBH'       : self.MBH,
            'radiation' : self.radiation
        }
        class_args = {
            'pars'      : self.H.params,
            'CONSTANTS' : hr.CONSTANTS,
            'cgs'       : self.cgs
        }

        # Initialize Accretion class
        self.Acc   = ana.Accretion(**acc_args)
        # Initialize Fluid class
        self.Fluid = ana.Fluid(**class_args)
        # Initialize Field class
        self.Field = ana.Field(**class_args)
        # Initialize Radiation class
        self.Rad   = rad.Radiation(**class_args)
        # Initialize Analytic class
        self.Analytic = theory.Analytic(hr.CONSTANTS, cgs=self.cgs)

    def mid_val(self, q):
        return (q[:, 0, :] + q[:, 1, :]) / 2.

    def calc_XY(self):
        r   = self.r
        th  = self.th
        phi = self.phi
        
        # Convert to Cartesian coordinates
        # Note reflection for left panel
        self.x =  (r*np.sin(th)*np.cos(phi))
        self.y =  (r*np.sin(th)*np.sin(phi))
        self.z =  (r*np.cos(th)*np.ones(phi.shape))
        
        x, y = {}, {}
        x['r'] = (self.x[0, :, 0, :] + self.x[0, :, 1, :]) / 2.
        y['r'] = (self.y[0, :, 0, :] + self.y[0, :, 1, :]) / 2.
        return x, y
    
    def calc_XY_uni(self, box_lim, npts):
        # Make uniform grid for streamline plots
        x1D = np.linspace(-box_lim, box_lim, npts)
        y1D = np.linspace(-box_lim, box_lim, npts)
        X_uni, Y_uni = np.meshgrid(x1D, y1D)
        return X_uni, Y_uni

    def get_vars3D(self, keys):
        variables = []
        for key in keys:
            if key in self.H.data.keys():
                shape = list(self.H.data[f'{key}'].shape)
                if len(shape) == 4:
                    variables.append(self.H.data[f'{key}'][0])
                if len(shape) == 5:
                    variables.append(self.H.data[f'{key}'][:, 0])
            elif key in self.H.grid.keys():
                shape = list(self.H.grid[f'{key}'].shape)
                # Note that this should not be used to retrieve r,th,phi in 2D
                if len(shape) == 4:
                    variables.append(self.H.grid[f'{key}'][0, :, :, :])
                if len(shape) == 6:
                    variables.append(self.H.grid[f'{key}'][:, :, 0, :, :, :])
            else:
                return ValueError("Invalid variable key.")
        return variables[0] if len(variables) == 1 else variables
    
    def get_vars2D(self, keys, iphi):
        return self.get_vars3D(keys)
    
    def set_Coords3D(self):
        ones  = np.ones(self.H.data['rho'].shape)
        r3D   = (ones * self.r)[  0]
        th3D  = (ones * self.th)[ 0]
        phi3D = (ones * self.phi)[0]
        self.Coords3D = coords.Coordinates(r3D, th3D, phi3D, 
                                           self.H.params, hr.CONSTANTS)
        return None
    
    def set_Coords2D(self, iphi):
        self.set_Coords3D()

    # Thermodynamic quantities
    def calc_rho(self):
        # Density
        return self.mid_val(super().calc_rho(0))
    
    def calc_Te(self):
        # Electron temperature
        return self.mid_val(super().calc_Te(0))
    
    def calc_Ti(self):
        # Ion temperature
        return self.mid_val(super().calc_Ti(0))
    
    def calc_Te_Ti(self):
        # Electron temperature, ion temperature
        Te = self.mid_val(super().calc_Te(0))
        Ti = self.mid_val(super().calc_Ti(0))
        return Te, Ti
    
    def calc_sigma(self):
        # sigma
        return self.mid_val(super().calc_sigma(0))

    def calc_BetaInv(self):
        # Beta^(-1)
        return self.mid_val(super().calc_BetaInv(0))

    def calc_radiation_energy_density(self):
        # radiation energy density
        return self.mid_val(super().calc_radiation_energy_density(0))
    
    def calc_u_t(self):
        return self.mid_val(super().calc_u_t(0))

    def calc_ion_density(self):
        return self.mid_val(super().calc_ion_density(0))
    
    def calc_lepton_density(self):
        return self.mid_val(super().calc_lepton_density(0))

    # Pressures
    def calc_pgas(self, **kwargs):
        return self.mid_val(super().calc_pgas(0, **kwargs))

    def calc_pmag(self):
        return self.mid_val(super().calc_pmag(0))

    def calc_prad(self):
        return self.mid_val(super().calc_prad(0))
    
    def calc_pmag_pradgas(self, **kwargs):
        pmag = self.mid_val(super().calc_pmag(0))
        prad = self.mid_val(super().calc_prad(0))
        pgas = self.mid_val(super().calc_pgas(0, **kwargs))
        return pmag / (prad + pgas)

    def calc_pmag_prad(self):
        pmag = self.mid_val(super().calc_pmag(0))
        prad = self.mid_val(super().calc_prad(0))
        return pmag / prad
    
    def calc_prad_pgas(self, **kwargs):
        prad = self.mid_val(super().calc_prad(0))
        pgas = self.mid_val(super().calc_pgas(0, **kwargs))
        return prad / pgas

    # Magnetic Fields
    def calc_Bphi(self):
        return self.mid_val(super().calc_Bphi(0))

    def calc_Bz(self):
        return self.mid_val(super().calc_Bz(0))

    def calc_B_cyl(self, **kwargs):
        B = self.get_vars3D(['B'])
        self.set_Coords3D()

        # Calculate B in cylindrical coordinates
        B_cyl = self.Coords3D.int_to_cyl_vec(B)
        if not kwargs.get('lr', False):
            BR   = self.mid_val(B_cyl[1])
            Bphi = self.mid_val(B_cyl[2])
            Bz   = self.mid_val(B_cyl[3])
            return BR, Bphi, Bz
        else:
            BR, Bphi, Bz = {}, {}, {}
            BR['r']   = self.mid_val(B_cyl[1])
            Bphi['r'] = self.mid_val(B_cyl[2])
            Bz['r']   = self.mid_val(B_cyl[3])
            if not kwargs.get('in_out', False):
                return BR, Bphi, Bz
            else:
                in_out = {'r' : np.sign(B[1, :, 0, :] + B[1, :, 1, :])}
                return BR, Bphi, Bz, in_out

    def calc_B_cart(self, **kwargs):
        B = self.get_vars3D(['B'])
        self.set_Coords3D()

        # Calculate B in Cartesian coordinates
        B_cart = self.Coords3D.int_to_cart_vec(B) / self.Field.B0
        if not kwargs.get('lr', False):
            Bx = self.mid_val(B_cart[1])
            By = self.mid_val(B_cart[2])
            return Bx, By
        else:
            Bx, By = {}, {}
            Bx['r'] = self.mid_val(B_cart[1])
            By['r'] = self.mid_val(B_cart[2])
            if not kwargs.get('in_out', False):
                return Bx, By
            else:
                in_out = {'r' : np.sign(B[1, :, 0, :] + B[1, :, 1, :])}
                return Bx, By, in_out

    def calc_bpol(self):
        return self.mid_val(super().calc_bpol(0))

    def calc_btor(self):
        return self.mid_val(super().calc_btor(0))

    def calc_bpol_btor(self):
        return self.mid_val(super().calc_bpol_btor(0))

    # Fluid velocities
    def calc_vr(self):
        return self.mid_val(super().calc_vr(0))

    def calc_vr_sph(self):
        return self.mid_val(super().calc_vr_sph(0))

    def calc_vphi_cyl(self):
        return self.mid_val(super().calc_vphi_sph(0))

    def calc_vR_vKep(self):
        return self.mid_val(super().calc_vR_vKep(0))

    def calc_v_cyl(self, **kwargs):
        uu = self.get_vars3D(['uu'])
        self.set_Coords3D()

        # Calculate uu in cylindrical coordinates
        uu_cyl = self.Coords3D.int_to_cyl_vec(uu) / self.c
        if not kwargs.get('lr', False):
            ut   = self.mid_val(uu_cyl[0])
            uR   = self.mid_val(uu_cyl[1])
            uphi = self.mid_val(uu_cyl[2])
            uz   = self.mid_val(uu_cyl[3])
            return uR/ut, uphi/ut, uz/ut
        else:
            vR, vphi, vz = {}, {}, {}
            ut   = self.mid_val(uu_cyl[0])
            uR   = self.mid_val(uu_cyl[1])
            uphi = self.mid_val(uu_cyl[2])
            uz   = self.mid_val(uu_cyl[3])
            vR['r']   = uR / ut
            vphi['r'] = uphi / ut
            vz['r']   = uz / ut
            if not kwargs.get('in_out', False):
                return vR, vphi, vz
            else:
                in_out = {'r' : np.sign(uu[1, :, 0, :] + uu[1, :, 1, :])}
                return vR, vphi, vz, in_out

    def calc_uu_cart(self, **kwargs):
        uu = self.get_vars3D(['uu'])
        self.set_Coords3D()

        # Calculate uu in Cartesian coordinates
        uu_cart = self.Coords3D.int_to_cart_vec(uu) / self.c
        if not kwargs.get('lr', False):
            uux = (uu_cart[1][:, 0, :] + uu_cart[1][:, 1, :]) / 2.
            uuy = (uu_cart[2][:, 0, :] + uu_cart[2][:, 1, :]) / 2.
            return uux, uuy
        else:
            uux, uuy = {}, {}
            uux['r'] = (uu_cart[1][:, 0, :] + uu_cart[1][:, 1, :]) / 2.
            uuy['r'] = (uu_cart[2][:, 0, :] + uu_cart[2][:, 1, :]) / 2.
            if not kwargs.get('in_out', False):
                return uux, uuy
            else:
                in_out = {}
                in_out['r'] = np.sign(uu[1, :, 0, :] + uu[1, :, 1, :])
                return uux, uuy, in_out

    def calc_vpol_vtor(self):
        return self.mid_val(super().calc_vpol_vtor(0)) 
    
    # Radiation 4-velocity
    def calc_uu_rad_cart(self, **kwargs):
        uu_rad = self.get_vars3D(['uu_rad'])
        self.set_Coords3D()

        # Calculate uu_rad in Cartesian coordinates
        uuR_cart = self.Coords3D.int_to_cart_vec(uu_rad) / self.c
        if not kwargs.get('lr', False):
            uuRx = self.mid_val(uuR_cart[1])
            uuRy = self.mid_val(uuR_cart[2])
            return uuRx, uuRy
        else:
            uuRx, uuRy = {}, {}
            uuRx['r'] = self.mid_val(uuR_cart[1])
            uuRy['r'] = self.mid_val(uuR_cart[2])
            if not kwargs.get('in_out', False):
                return uuRx, uuRy
            else:
                in_out = {'r' : np.sign(uu_rad[1,:,0,:] + uu_rad[1,:,1,:])}
                return uuRx, uuRy, in_out

    # Cooling rates    
    # Inverse-Compton cooling rate
    def calc_qIC(self, **kwargs):
        return self.mid_val(super().calc_qIC(0, **kwargs))

    # Coulomb cooling rate
    def calc_qie(self, **kwargs):
        return self.mid_val(super().calc_qie(0, midplane=True, **kwargs))

    # Synchrotron cooling rate
    def calc_qsync(self, **kwargs):
        return self.mid_val(super().calc_qsync(0, **kwargs)) 
    
    # Cooling timescales
    def calc_teq(self, **kwargs):  
        return self.mid_val(super().calc_teq(0, midplane=True, **kwargs))
    
    def calc_tii(self, **kwargs):
        return self.mid_val(super().calc_tii(0, midplane=True, **kwargs))
    
    def calc_tee(self, **kwargs):
        return self.mid_val(super().calc_tee(0, midplane=True, **kwargs))
    
    def calc_tIC(self, **kwargs):   
        return self.mid_val(super().calc_tIC(0, **kwargs))
    
    # Orbital frequency
    def calc_OmegaK(self):
        return self.Analytic.calc_OmegaK(
            self.MBH, self.H.params['a'], self.r)[0, :, 0, :]

    # Opacities    
    def calc_kappa_eff(self, **kwargs):
        return self.mid_val(super().calc_kappa_eff(0, **kwargs))

    def calc_kappa_sync(self, **kwargs):
        return self.mid_val(super().calc_kappa_sync(0, **kwargs))

    def calc_kappa_ff(self, **kwargs):
        return self.mid_val(super().calc_kappa_ff(0, **kwargs))

class Load_Data():
    # Retrieve file numbers based on file_number, start_file, and end file
    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        file_type  = kwargs.get('file_type', 'Profiles')
        reduced    = kwargs.get('reduced', False)
        data_dir   = (f'{averages_dir}/Full_Resolution/{file_type}' if not reduced 
                      else f'{averages_dir}/Reduced_Resolution/{file_type}')
        single_file = True if (file_number != -1) else False
        if (single_file):
            fnums = list([file_number])
        else:
            names = glob.glob(f"{file_type}*", root_dir=data_dir)
            nums = [int(name.strip(f'{file_type}_').strip('.npz')) for name in names]
            nums.sort()
            if ((start_file != -1) and (end_file != -1)):
                fnums = list(range(start_file, end_file + 1))
            elif ((start_file != -1) and (end_file == -1)):
                last_file = nums[-1]
                fnums = list(range(start_file, last_file + 1))
            elif ((start_file == -1) and (end_file != -1)):
                first_file = nums[0]
                fnums = list(range(first_file, last_file + 1))
            else:
                fnums = list(range(nums[0], nums[-1] + 1))
            fnums = self.check_files_exist(fnums, nums, file_type)
        return fnums

    def check_files_exist(self, fnums, data_nums, file_type):
        removed_files = []
        for fnum in fnums:
            if fnum not in data_nums:
                print(f"{file_type} {fnum} is missing.")
                removed_files.append(fnum)
        for removed_file in removed_files:
            fnums.remove(removed_file)
        return fnums

    # Create slices dictionary [file_number][key]
    def load_data(self, averages_dir, fnums, **kwargs):
        file_type  = kwargs.get('file_type', 'Profiles')
        reduced    = kwargs.get('reduced', False)
        data_dir   = (f'{averages_dir}/Full_Resolution/{file_type}' 
                    if not reduced 
                    else f'{averages_dir}/Reduced_Resolution/{file_type}')
        print(f"Loading from {data_dir}...")
        names = glob.glob(f"{file_type}*", root_dir=data_dir)
        nzeros = max([
            len(name.strip(f'{file_type}_').strip('.npz')) for name in names
        ])
        if 'profiles' not in kwargs and 'slices' not in kwargs:
            data = {}
        elif 'profiles' in kwargs and 'slices' not in kwargs:
            data = kwargs.get('profiles')
        elif 'profiles' not in kwargs and 'slices' in kwargs:
            data = kwargs.get('slices')
        for fnum in fnums:
            if (file_type == 'Slices' or file_type == 'Heating_Slices' or 
                file_type == 'Stresses'):
                print(f"File number {fnum}...", flush=True)
            data_file = (
                f'{data_dir}/{file_type}_%s.npz' %str(fnum).zfill(nzeros))
            if not kwargs.get('keys', False):
                data[f'{fnum}'] = np.load(data_file)
                files = data[f'{fnum}'].files
            else:
                if 'profiles' not in kwargs and 'slices' not in kwargs:
                    data[f'{fnum}'] = {}
                for key in kwargs.get('keys'):
                    data[f'{fnum}'][f'{key}']=np.load(data_file)[f'{key}']
        return data
    
class Slices(Load_Data):
    # Retrieve file numbers based on file_number, start_file, and end file
    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        return super().get_fnums(averages_dir, file_number, start_file, end_file, 
                                 file_type='Slices', **kwargs)

    # Create slices dictionary [file_number][key]
    def load_slices(self, averages_dir, fnums, **kwargs):
        return super().load_data(averages_dir, fnums, 
                                 file_type='Slices', **kwargs)

    # Create slices dictionary (time-averaged) slices[key]
    def load_slices_time_averaged(self, averages_dir, fnums, **kwargs):
        reduced = kwargs.get('reduced', False)
        coords = self.load_slices(averages_dir, fnums, 
            keys=['r' , 'theta', 'a', 'dx1', 'dx2'], reduced=reduced)
        self.check_AMR(fnums, coords)
        nt = len(fnums)
        if 'keys' in kwargs:
            slices = dict.fromkeys(kwargs.get('keys'))
            for key in kwargs.get('keys'):
                slices[f'{key}'] = np.zeros([max(self.nr), max(self.nth)])
        else:
            raise ValueError("Must supply keys for time-averaging.")
        if 'vols' in kwargs:
            vols = kwargs.get('vols')
        else:
            raise ValueError("Must supply vols for time-averaging.")
    
        keys = kwargs.pop('keys')
        for fnum in fnums:
            pkeys = (keys + ['r', 'theta']) if ('r' not in keys and 'theta' not in keys) else keys
            pkeys = (pkeys + ['Vol']) if 'Vol' not in keys else pkeys
            single_slice = self.load_slices(averages_dir, [fnum], keys=pkeys, **kwargs)
            for key in keys:
                # "Time-average" takes care of upscaling for single file number
                slices[f'{key}'] += (1. / nt) * (
                    self.time_avg(key, single_slice, vol=vols[f'{key}']))
        slices['r']     = coords[f'{self.fnums0}']['r']
        slices['theta'] = coords[f'{self.fnums0}']['theta']
        slices['a']     = coords[f'{self.fnums0}']['a']
        slices['dx1']   = coords[f'{self.fnums0}']['dx1']
        slices['dx2']   = coords[f'{self.fnums0}']['dx2']
        return slices

    def check_AMR(self, fnums, slices):
        self.nr, self.nth, self.nums = [], [], []
        for fnum in fnums:
            self.nums.append(fnum)
            self.nr.append(len(slices[f'{fnum}']['r']))
            self.nth.append(len(slices[f'{fnum}']['theta']))
        self.AMRx1 = (self.nr.count(self.nr[0]) != len(self.nr))
        self.AMRx2 = (self.nth.count(self.nth[0]) != len(self.nth))
        if (self.AMRx1 and self.AMRx2):
            max_arr     = max(self.nr) * np.ones_like(self.nr)
            nr_eq_max   = np.array(self.nums)[self.nr == max_arr]
            self.fnums0 = nr_eq_max[0] if len(nr_eq_max) != 0 else nr_eq_max
        elif (self.AMRx1 and not self.AMRx2):
            max_arr     = max(self.nr) * np.ones_like(self.nr)
            nr_eq_max   = np.array(self.nums)[self.nr == max_arr]
            self.fnums0 = nr_eq_max[0] if len(nr_eq_max) != 0 else nr_eq_max
        elif (not self.AMRx1 and self.AMRx2):
            max_arr     = max(self.nth) * np.ones_like(self.nth)
            nth_eq_max  = np.array(self.nums)[self.nth == max_arr]
            self.fnums0 = nth_eq_max[0] if len(nth_eq_max) != 0 else nth_eq_max
        else:
            self.fnums0 = self.nums[0]

    def time_avg(self, quant, slices, **kwargs):
        fnums = slices.keys() if 'fnums' not in kwargs else kwargs.get('fnums')
        if not hasattr(self, "AMRx1") or not hasattr(self, "AMRx2"):
            self.check_AMR(fnums, slices)
        q_avg = np.zeros([max(self.nr), max(self.nth)])
        nt    = len(fnums)
        if not self.AMRx1 or not self.AMRx2:
            for fnum in fnums:
                vol    = slices[f'{fnum}']['Vol'] if kwargs.get('vol', False) else 1.
                q_avg += slices[f'{fnum}'][f'{quant}'] / (nt * vol)
        else:
            for fnum in fnums:
                vol    = self.upscale(slices[f'{fnum}']['Vol']) if kwargs.get('vol', False) else 1.
                q_avg += self.upscale(slices[f'{fnum}'][f'{quant}']) / (nt * vol)
        return q_avg

    def upscale(self, q, **kwargs):
        max_nr  = kwargs.get('max_nr',  max(self.nr))
        max_nth = kwargs.get('max_nth', max(self.nth))
        level_x1 = np.log2(max_nr / len(q[:, 0])) 
        level_x2 = np.log2(max_nth/ len(q[0, :])) 
        if (level_x1 == 0 and level_x2 == 0): 
            return q
        else:
            q_up = np.zeros([max_nr, max_nth])
            for i in range(len(q[:, 0])):
                for j in range(len(q[0, :])):
                    istrt = int(i       * (2 ** level_x1))
                    iend  = int((i + 1) * (2 ** level_x1))
                    jstrt = int(j       * (2 ** level_x2))
                    jend  = int((j + 1) * (2 ** level_x2))
                    q_up[istrt:iend, jstrt:jend] = q[i, j]
            return q_up

    # Assumes slices has form slices[key] as if from load_slices_time_averaged
    def update_slices(self, averages_dir, fnums, slices, **kwargs):
        fnums_old   = kwargs.get('fnums_old', fnums)
        sub_fnum    = fnums_old[0]
        add_fnum    = fnums[-1]
        nt_old      = len(fnums_old)
        nt          = len(fnums)
        load_fnums  = [sub_fnum, add_fnum]
        keys        = kwargs.get('keys', list(slices.keys()))
        if 'vols' in kwargs:
            vols = kwargs.get('vols')
        else:
            raise ValueError("Cannot update slice without passing vols dict.")

        if (len(fnums_old) == len(fnums)):
            switch_slices = self.load_slices(averages_dir, load_fnums, keys=keys)
            update_keys   = keys.copy()
            if 'Vol' in update_keys: update_keys.remove('Vol')
            for key in update_keys:
                vol = vols[f'{key}']
                slices[f'{key}'] *= nt_old
                slices[f'{key}'] = self.sub_slice(
                    key, slices, switch_slices[f'{sub_fnum}'], vol=vol)
                slices[f'{key}'] = self.add_slice(
                    key, slices, switch_slices[f'{add_fnum}'], vol=vol)
                slices[f'{key}'] *= (1. / nt)
        return slices
    
    def sub_slice(self, key, slices, switch_slices, **kwargs):
        q       = slices[f'{key}']
        qdV_sub = switch_slices[f'{key}']
        if (q.shape == qdV_sub.shape):
            vol = switch_slices['Vol'] if kwargs.get('vol', False) else 1.
            return q - (qdV_sub / vol)
        else:
            return q

    def add_slice(self, key, slices, switch_slices, **kwargs):
        q       = slices[f'{key}']
        qdV_add = switch_slices[f'{key}']
        if (q.shape == qdV_add.shape):
            vol = switch_slices['Vol'] if kwargs.get('vol', False) else 1.
            return q + (qdV_add / vol)
        else:
            return q
        
class Heating_Slices(Slices):
    # Retrieve file numbers based on file_number, start_file, and end file
    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        return super(Slices, self).get_fnums(averages_dir, file_number, 
                                             start_file, end_file, 
                                             file_type='Heating_Slices', **kwargs)

    # Create slices dictionary [file_number][key]
    def load_slices(self, averages_dir, fnums, **kwargs):
        return super(Slices, self).load_data(averages_dir, fnums, 
                                             file_type='Heating_Slices', **kwargs)

class Custom_Slices(Slices):
    def __init__(self, file_type):
        self.file_type = file_type

    # Retrieve file numbers based on file_number, start_file, and end file
    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        return super(Slices, self).get_fnums(averages_dir, file_number, 
                                             start_file, end_file, 
                                             file_type=self.file_type, **kwargs)

    # Create slices dictionary [file_number][key]
    def load_slices(self, averages_dir, fnums, **kwargs):
        return super(Slices, self).load_data(averages_dir, fnums, 
                                             file_type=self.file_type, **kwargs)

class Stresses(Slices):
    # Retrieve file numbers based on file_number, start_file, and end file
    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        return super(Slices, self).get_fnums(averages_dir, file_number, 
                                             start_file, end_file, 
                                             file_type='Stresses', **kwargs)

    # Create slices dictionary [file_number][key]
    def load_slices(self, averages_dir, fnums, **kwargs):
        return super(Slices, self).load_data(averages_dir, fnums, 
                                             file_type='Stresses', **kwargs)

class Load_Vertical_Data():
    # Load theta profiles at given r0
    # If time_average = True: profiles[file_number][key]
    # otherwise: profiles[key]
    def load_theta_profiles(self, averages_dir, fnums, r0, **kwargs):
        if not kwargs.get('time_average', False): 
            slices   = self.load_slices(averages_dir, fnums, **kwargs)
            profiles = dict.fromkeys(slices.keys())
            for fnum in profiles.keys():
                keys = list(slices[f'{fnum}'].keys())
                if 'r' in keys: keys.remove('r')
                if 'theta' in keys: keys.remove('theta')
                profiles[f'{fnum}'] = dict.fromkeys(keys)
                for key in profiles[f'{fnum}'].keys():
                    r   = slices[f'{fnum}']['r']
                    idx = np.argmin(np.abs(r - r0))
                    q   = slices[f'{fnum}'][f'{key}']
                    if (len(list(q.shape)) == 2):
                        profiles[f'{fnum}'][f'{key}'] = q[idx, :]
                    else:
                        profiles[f'{fnum}'][f'{key}'] = q
                profiles[f'{fnum}']['theta'] = slices[f'{fnum}']['theta']
                profiles[f'{fnum}']['r']     = slices[f'{fnum}']['r']
        elif kwargs.get('time_average', False):
            slices = self.load_slices_time_averaged(averages_dir, fnums, **kwargs)
            keys   = list(slices.keys())
            if 'r' in keys: keys.remove('r')
            if 'theta' in keys: keys.remove('theta')
            profiles = dict.fromkeys(keys)
            for key in keys:
                r   = slices['r']
                idx = np.argmin(np.abs(r - r0))
                q   = slices[f'{key}']
                if (len(list(q.shape)) == 2):
                    profiles[f'{key}'] = q[idx, :]
                else:
                    profiles[f'{key}'] = q
            profiles['theta'] = slices['theta']
            profiles['r']     = slices['r']
        return profiles

    def load_z_profiles(self, averages_dir, fnums, r0, **kwargs):
        npts = int(kwargs.get('npts', 100.))
        dR   = kwargs.get('dR', 0.1 * r0)
        R1D  = np.linspace(r0 - dR, r0 + dR, int(npts / 10) + 1)
        z1D  = np.linspace(-1. * r0, 1. * r0, int(100 * npts))
        R_uni, Z_uni = np.meshgrid(R1D, z1D)
        idx  = np.argmin(np.abs(R_uni[0, :] - r0))

        if not kwargs.get('time_average', False): 
            slices   = self.load_slices(averages_dir, fnums, **kwargs)
            profiles = dict.fromkeys(slices.keys())
            for fnum in profiles.keys():
                keys = list(slices[f'{fnum}'].keys())
                if 'r' in keys: keys.remove('r')
                if 'theta' in keys: keys.remove('theta')
                profiles[f'{fnum}'] = dict.fromkeys(keys)
                r     = slices[f'{fnum}']['r']
                theta = slices[f'{fnum}']['theta']
                # Interpolate to the uniform grid
                ones = np.ones_like(r[:, None] * theta[None, :])
                r2D  =  r[:, None] * ones
                th2D =  theta[None, :] * ones
                R2D  = r2D * np.sin(th2D)
                Z2D  = r2D * np.cos(th2D)
                
                for key in profiles[f'{fnum}'].keys():
                    print(f"Loading {key}...", flush=True)
                    z_min, z_max = 1.1 * np.amin(z1D), 1.1 * np.amax(z1D)

                    q   = slices[f'{fnum}'][f'{key}']
                    if (len(list(q.shape)) == 2):
                        R_cut = (R2D < 2. * r0) & (R2D > r0 - dR)
                        Z_cut = (Z2D < z_max) & (Z2D > z_min)
                        q_dat   = q[R_cut & Z_cut]
                        R2D_dat = R2D[R_cut & Z_cut]
                        Z2D_dat = Z2D[R_cut & Z_cut]
                        q_uni = griddata((R2D_dat.ravel(), Z2D_dat.ravel()), 
                                         q_dat.ravel(), 
                                        (R_uni, Z_uni), method='cubic')
                        profiles[f'{fnum}'][f'{key}'] = q_uni[:, idx]
                    else:
                        profiles[f'{fnum}'][f'{key}'] = q

                profiles[f'{fnum}']['z_r']   = z1D / R1D[idx]
                profiles[f'{fnum}']['theta'] = slices[f'{fnum}']['theta']
                profiles[f'{fnum}']['r']     = slices[f'{fnum}']['r']

        elif kwargs.get('time_average', False):
            slices = self.load_slices_time_averaged(averages_dir, fnums, **kwargs)
            keys   = list(slices.keys())
            if 'r' in keys: keys.remove('r')
            if 'theta' in keys: keys.remove('theta')
            profiles = dict.fromkeys(keys)
            r     = slices['r']
            theta = slices['theta']
            # Interpolate to the uniform grid
            ones = np.ones_like(r[:, None] * theta[None, :])
            r2D  =  r[:, None] * ones
            th2D =  theta[None, :] * ones
            R2D  = r2D * np.sin(th2D)
            Z2D  = r2D * np.cos(th2D)
            
            for key in keys:
                print(f"Loading {key}...", flush=True)
                z_min, z_max = 1.1 * np.amin(z1D), 1.1 * np.amax(z1D)
                q   = slices[f'{key}']

                if (len(list(q.shape)) == 2):
                    R_cut   = (R2D < 2. * r0) & (R2D > r0 - dR)
                    Z_cut   = (Z2D < z_max) & (Z2D > z_min)
                    q_dat   = q[R_cut & Z_cut]
                    R2D_dat = R2D[R_cut & Z_cut]
                    Z2D_dat = Z2D[R_cut & Z_cut]
                    q_uni = griddata((R2D_dat.ravel(), Z2D_dat.ravel()), 
                                     q_dat.ravel(), 
                                    (R_uni, Z_uni), method='cubic')
                    profiles[f'{key}'] = q_uni[:, idx]
                else:
                    profiles[f'{key}'] = q
            
            profiles['z_r']   = z1D / R1D[idx]
            profiles['theta'] = slices['theta']
            profiles['r']     = slices['r']

        return profiles

class Vertical_Profiles(Load_Vertical_Data, Slices):
    pass

class Vertical_Stress_Profiles(Load_Vertical_Data, Stresses):
    pass

class Profiles(Load_Data):
    # Retrieve file numbers based on file_number, start_file, and end file
    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        return super().get_fnums(averages_dir, file_number, start_file, end_file, 
            file_type='Profiles', **kwargs)

    # Create slices dictionary [file_number][key]
    def load_profiles(self, averages_dir, fnums, **kwargs):
        return super().load_data(averages_dir, fnums, 
            file_type='Profiles', **kwargs)
 
    def time_avg(self, quant, profiles, **kwargs):
        fnums = profiles.keys() if 'fnums' not in kwargs else kwargs.get('fnums')
        if not hasattr(self, "AMR"):
            self.check_AMR(fnums, profiles)
        q_avg = np.zeros(len(profiles[f'{self.fnums0}']['r']))
        nt    = len(fnums)
        if not self.AMR:
            for fnum in fnums:
                vol    = profiles[f'{fnum}']['Vol'] if kwargs.get('vol', False) else 1.
                q_avg += profiles[f'{fnum}'][f'{quant}'] / (nt * vol)
        else:
            for fnum in fnums:
                vol    = self.upscale(profiles[f'{fnum}']['Vol']) if kwargs.get('vol', False) else 1.
                q_avg += self.upscale(profiles[f'{fnum}'][f'{quant}']) / (nt * vol)
        return q_avg

    def upscale(self, q):
        level = np.log2(max(self.nr) / len(q))
        if (level == 0): 
            return q
        else:
            q_up = np.zeros(max(self.nr))
            for i in range(len(q)):
                istrt = int(i       * (2 ** level))
                iend  = int((i + 1) * (2 ** level))
                q_up[istrt:iend] = q[i]
            return q_up

    def check_AMR(self, fnums, profiles):
        self.nr, self.nums = [], []
        for fnum in fnums:
            self.nums.append(fnum)
            self.nr.append(len(profiles[f'{fnum}']['r']))
        self.AMR = (self.nr.count(self.nr[0]) != len(self.nr))
        if (self.AMR):
            max_arr     = max(self.nr) * np.ones_like(self.nr) 
            nr_eq_max   = np.array(self.nums)[self.nr == max_arr]
            self.fnums0 = nr_eq_max[0] if len(nr_eq_max) != 0 else nr_eq_max
        else:
            self.fnums0 = self.nums[0] 

class Heating_Profiles(Profiles):
    # Retrieve file numbers based on file_number, start_file, and end file
    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        return super(Profiles, self).get_fnums(averages_dir, file_number, 
            start_file, end_file, file_type='Heating_Profiles', **kwargs)

    # Create slices dictionary [file_number][key]
    def load_profiles(self, averages_dir, fnums, **kwargs):
        return super(Profiles, self).load_data(averages_dir, fnums, 
            file_type='Heating_Profiles', **kwargs)

class Time_Series(Profiles):
    def calc_time_series(self, key, profiles, **kwargs):
        fnums = list([int(fnum) for fnum in profiles.keys()])
        fnums.sort()
        time  = np.zeros(len(fnums))
        q     = np.zeros(len(fnums))
        r0    = kwargs.get('r0', 0.)
        isarr = sum((profiles[f'{fnums[0]}'][f'{key}']).shape)
        if isarr:
            for i, fnum in enumerate(fnums):
                r       = profiles[f'{fnum}']['r']
                idx     = np.argmin(np.abs(r - r0))
                time[i] = profiles[f'{fnum}']['t']
                vol     = (profiles[f'{fnum}']['Vol'] 
                           if kwargs.get('vol', False) else 1.)
                q[i]    = (profiles[f'{fnum}'][f'{key}']/vol)[idx]
        else:
            for i, fnum in enumerate(fnums):
                r       = profiles[f'{fnum}']['r']
                idx     = np.argmin(np.abs(r - r0))
                time[i] = profiles[f'{fnum}']['t']
                vol     = (profiles[f'{fnum}']['Vol'] 
                           if kwargs.get('vol', False) else 1.)
                q[i]    = (profiles[f'{fnum}'][f'{key}']/vol)
        return time, q