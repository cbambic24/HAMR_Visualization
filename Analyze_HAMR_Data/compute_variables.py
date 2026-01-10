import sys
import os
import argparse
import glob
import numpy as np
read_path   = '/lustre/orion/ast178/proj-shared/cbambic/HAMR_Visualization/Read_HAMR_Data'
asys_path   = '/lustre/orion/ast178/proj-shared/cbambic/HAMR_Visualization/Analyze_HAMR_Data'
run_path    = '/lustre/orion/ast178/proj-shared/cbambic/RUN'
user_path   = '/lustre/orion/ast178/proj-shared/cbambic'
shared_path = '/lustre/orion/ast178/proj-shared'
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
        data_dir = user_path + '/DATA/' + sim
    elif reduced and not shared:
        data_dir = user_path + '/DATA/' + sim + '/reduced/'
    elif not reduced and shared:
        data_dir = f'{shared_path}/' + sim
    elif reduced and shared:
        data_dir = f'{shared_path}/' + sim + '/reduced/'
    return data_dir 

def get_save_dir(sim, type, diagnostic, **kwargs):
    subtype = kwargs.get('subtype', False)
    reduced = kwargs.get('reduced', False)
    if not subtype and not reduced:
        save_dir = user_path + f'/{type}/' + sim + f'/Full_Resolution/{diagnostic}'
    elif subtype and not reduced:
        save_dir = user_path + f'/{type}/' + sim + f'/Full_Resolution/{subtype}/{diagnostic}'
    elif not subtype and reduced:
        save_dir = user_path + f'/{type}/' + sim + f'/Reduced_Resolution/{diagnostic}'
    elif subtype and reduced:
        save_dir = user_path + f'/{type}/' + sim + f'/Reduced_Resolution/{subtype}/{diagnostic}'
    return save_dir

def get_file_list_local(file_list, size, rank, data_dir):
    nout             = len(file_list)
    nfiles_nproc     = int(np.floor(nout / size))
    nfiles_remaining = nout % size
    files_per_proc   = ((size - nfiles_remaining) * [nfiles_nproc] +
        nfiles_remaining * [nfiles_nproc + 1])
    file_list_local  = ((sum(files_per_proc[:rank]) +
        np.arange(0, files_per_proc[rank])) + file_list[0]).tolist()
    
    file_list_local = check_dumps_exist(data_dir, file_list_local)
    
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

# Create profiles dictionary [file_number][key]
def load_profiles(averages_dir, fnums, **kwargs):
    reduced      = kwargs.get('reduced', False)
    profiles_dir = (f'{averages_dir}/Full_Resolution/Profiles' if not reduced 
                    else f'{averages_dir}/Reduced_Resolution/Profiles')
    print(f"Loading from {profiles_dir}...")
    profs = glob.glob("Profiles*", root_dir=profiles_dir)
    prof_nums = [
        int(prof_name.strip('Profiles_').strip('.npz')) for prof_name in profs
    ]
    prof_nums.sort()
    nzeros    = len(str(prof_nums[-1]))
    if 'profiles' not in kwargs:
        profiles = {}
    else:
        profiles = kwargs.get('profiles')
    for fnum in fnums:
        profile_file = (
            f'{profiles_dir}/Profiles_%s.npz' %str(fnum).zfill(nzeros))
        if not kwargs.get('keys', False):
            profiles[f'{fnum}'] = np.load(profile_file)
            files = profiles[f'{fnum}'].files
        else:
            if 'profiles' not in kwargs:
                profiles[f'{fnum}'] = {}
            for key in kwargs.get('keys'):
                 profiles[f'{fnum}'][f'{key}']=np.load(profile_file)[f'{key}']
    return profiles

# Compute quantities at constant phi
class Phi_Slice():
    def __init__(self, directory, fnum, **kwargs):
        self.H = hr.Read(directory, **kwargs)
        self.H.read_grid(fnum, direct=True)
        self.H.read_prims(fnum, skip_read_params=True)
        self.cgs  = kwargs.get('cgs_data', True)
        self.MBH  = kwargs.get('MBH', 10.0)
        self.Msun = hr.CONSTANTS['Msun']
        if (self.cgs):
            self.rho0 = self.H.params['density_scale']
            self.G    = hr.CONSTANTS['G']
            self.c    = hr.CONSTANTS['c']
            self.Rg   = self.G * self.MBH * self.Msun / (self.c * self.c)
        else:
            self.rho0 = 1.
            self.G    = 1.
            self.c    = 1.
            self.Rg   = 1.

        self.calc_gridsize()

        self.r    = self.H.grid['r'] / self.Rg  
        self.th   = self.H.grid['theta']
        self.phi  = self.H.grid['phi']

        phi0        = np.pi
        self.phi_1D = self.phi[0, 0, 0, :]
        self.iphi   = np.argmin(np.abs(self.phi_1D - phi0))

        self.two_temp  = True if 'Te' and 'Ti' in self.H.data else False
        self.radiation = True if 'u_rad' and 'uu_rad' in self.H.data else False

        # Initialize Accretion class
        self.Acc   = ana.Accretion(self.H.grid, self.H.data, self.H.params, 
            hr.CONSTANTS, cgs=self.cgs, MBH=self.MBH, radiation=self.radiation)
        # Initialize Fluid class
        self.Fluid = ana.Fluid(self.H.params, hr.CONSTANTS, cgs=self.cgs)
        # Initialize Field class
        self.Field = ana.Field(self.H.params, hr.CONSTANTS, cgs=self.cgs)
        # Initialize Radiation class
        self.Rad   = rad.Radiation(self.H.params, hr.CONSTANTS, cgs=self.cgs)
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
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]
        self.Coords2D = coords.Coordinates(r2D, th2D, phi2D, 
                                           self.H.params, hr.CONSTANTS)
        return None
    
    def set_Tetrads2D(self, iphi, **kwargs):
        ones  = np.ones(self.H.data['rho'].shape)
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
    
    def calc_BetaInv(self, iphi):
        # BetaI = pmag/pgas
        rho0, c = self.rho0, self.c    
        B02     = self.Field.B0 * self.Field.B0  
        keys    = ['rho', 'B', 'uu', 'u_gas', 'Te', 'Ti', 'gcov']
        rho, B, uu, u_gas, Te, Ti, gcov = self.get_vars2D(keys, iphi)
        
        # Calculate gas pressure and Beta^-1
        if (self.cgs):
            pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        else:
            pgas  = ((5./3.) - 1.) * u_gas
        b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        
        BetaI = ((b2 / B02) / 2.) / (pgas / (rho0 * c * c))
        return BetaI
    
    def calc_sigma_BetaI(self, iphi):
        # sigma, BetaI = pmag/pgas
        rho0, c = self.rho0, self.c    
        B02     = self.Field.B0 * self.Field.B0  
        keys    = ['rho', 'B', 'uu', 'u_gas', 'Te', 'Ti', 'gcov']
        rho, B, uu, u_gas, Te, Ti, gcov = self.get_vars2D(keys, iphi)
    
        # Calculate sigma
        b     = self.Field.calc_b(B, uu, gcov)
        b2    = self.Field.calc_b2(b, B, gcov)
        sigma = (b2 / B02) / (rho / rho0)
        # Calculate gas pressure and Beta^-1
        if (self.cgs):
            pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        else:
            pgas  = ((5./3.) - 1.) * u_gas
        BetaI = ((b2 / B02) / 2.) / (pgas / (rho0 * c * c))
        return sigma, BetaI
    
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
    def calc_pgas(self, iphi):
        u_gas, Te, Ti = self.get_vars2D(['u_gas', 'Te', 'Ti'], iphi)
        return self.Fluid.calc_pgas(u_gas, Te, Ti)
    
    def calc_prad(self, iphi):
        u_rad = self.get_vars2D(['u_rad'], iphi)
        prad  = (1./3.) * u_rad
        return prad
    
    def calc_pmag(self, iphi):
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        b2    = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)     
        return b2 / (8. * np.pi)

    def calc_pmag_pradgas(self, iphi):
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        b2    = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu) 
        keys  = ['u_gas', 'u_rad', 'Te', 'Ti']
        u_gas, u_rad, Te, Ti = self.get_vars2D(keys, iphi) 
        
        pmag  = b2 / (8. * np.pi)
        pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        prad  = (1./3.) * u_rad
        return pmag / (prad + pgas)
    
    def calc_pmag_prad(self, iphi):
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        b2    = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu) 
        u_rad = self.get_vars2D(['u_rad'], iphi) 
        
        pmag  = b2 / (8. * np.pi)
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
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        self.set_Coords2D(iphi)

        b    = self.Field.calc_b(B, uu, gcov)
        bcyl = self.Coords2D.int_to_cyl_vec(b, dtype=np.float64) 
        bpol = np.copy(bcyl)
        bpol[2] = 0.
        gcov_cyl = self.Coords2D.int_to_cyl_cotnsr(gcov)
        return np.sign(bcyl[3]) * np.sqrt(ana.Tools.dot(bpol, bpol, gcov_cyl))

    def calc_btor(self, iphi):
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        self.set_Coords2D(iphi)

        b    = self.Field.calc_b(B, uu, gcov)
        bcyl = self.Coords2D.int_to_cyl_vec(b, dtype=np.float64) 
        return bcyl[2]

    def calc_bpol_btor(self, iphi):
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
        rho, Ti, Te = self.H.data['rho'], self.H.data['Ti'], self.H.data['Te']
        uu, gcov    = self.H.data['uu'], self.H.grid['gcov']
        u_t         = np.einsum('ij...,j...->i...', gcov, uu)[0] / self.c
        qie = self.Fluid.calc_qie(rho, Te, Ti, **kwargs)
        return (-u_t * qie)[0, :, :, iphi]

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
        rho, Ti, Te = self.H.data['rho'], self.H.data['Ti'], self.H.data['Te']
        return self.Fluid.calc_teq(rho, Te, Ti, **kwargs)[0, :, :, iphi]
    
    # Ion-ion collisions
    def calc_tii(self, iphi, **kwargs):
        rho, Ti, Te = self.H.data['rho'], self.H.data['Ti'], self.H.data['Te']
        return self.Fluid.calc_tii(rho, Te, Ti, **kwargs)[0, :, :, iphi]
    
    # Electron-electron collisions
    def calc_tee(self, iphi, **kwargs):
        rho, Ti, Te = self.H.data['rho'], self.H.data['Ti'], self.H.data['Te']
        return self.Fluid.calc_tee(rho, Te, Ti, **kwargs)[0, :, :, iphi]
    
    # Inverse Compton
    def calc_tIC(self, iphi, **kwargs):
        rho, Te, u_rad = self.H.data['rho'], self.H.data['Te'], self.H.data['u_rad']
        return self.Rad.calc_tIC(rho, Te, u_rad, **kwargs)[0, :, :, iphi]

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

# Compute quantities at constant theta
class Theta_Slice(Phi_Slice):
    def __init__(self, directory, fnum, **kwargs):
        self.H = hr.Read(directory, **kwargs)
        self.H.read_grid(fnum, direct=True)
        self.H.read_prims(fnum, skip_read_params=True)

# Compute quantities in the miplane (averaging over two nearest cells)
class Midplane_Slice():
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
        self.MBH  = kwargs.get('MBH', 10.0)
        self.Msun = hr.CONSTANTS['Msun']
        if (self.cgs):
            self.rho0 = self.H.params['density_scale']
            self.G    = hr.CONSTANTS['G']
            self.c    = hr.CONSTANTS['c']
            self.Rg   = self.G * self.MBH * self.Msun / (self.c * self.c)
        else:
            self.rho0 = 1.
            self.G    = 1.
            self.c    = 1.
            self.Rg   = 1.

        self.r    = self.H.grid['r'] / self.Rg  
        self.th   = self.H.grid['theta']
        self.phi  = self.H.grid['phi']
        
        self.two_temp  = True if 'Te' and 'Ti' in self.H.data else False
        self.radiation = True if 'u_rad' and 'uu_rad' in self.H.data else False

        # Initialize Accretion class
        self.Acc   = ana.Accretion(self.H.grid, self.H.data, self.H.params, 
            hr.CONSTANTS, cgs=self.cgs, MBH=self.MBH, radiation=self.radiation)
        # Initialize Coordinates class
        self.Coords = coords.Coordinates(self.H.grid['r'], self.H.grid['theta'], 
            self.H.grid['phi'], self.H.params, hr.CONSTANTS)
        # Initialize Fluid class
        self.Fluid = ana.Fluid(self.H.params, hr.CONSTANTS, cgs=self.cgs)
        # Initialize Field class
        self.Field = ana.Field(self.H.params, hr.CONSTANTS, cgs=self.cgs)
        # Initialize Radiation class
        self.Rad   = rad.Radiation(self.H.params, hr.CONSTANTS, cgs=self.cgs)
        # Initialize Analytic class
        self.Analytic = theory.Analytic(hr.CONSTANTS, cgs=self.cgs)

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
    
    def set_Coords3D(self):
        ones  = np.ones(self.H.data['rho'].shape)
        r3D   = (ones * self.r)[  0]
        th3D  = (ones * self.th)[ 0]
        phi3D = (ones * self.phi)[0]
        self.Coords3D = coords.Coordinates(r3D, th3D, phi3D, 
                                           self.H.params, hr.CONSTANTS)
        return None

    # Thermodynamic quantities
    def calc_rho(self):
        # Density
        rho = self.get_vars3D(['rho'])
        return (rho[:, 0, :] + rho[:, 1, :]) / 2.
    
    def calc_Te(self):
        # Electron temperature
        Te = self.get_vars3D(['Te'])
        return (Te[:, 0, :] + Te[:, 1, :]) / 2.
    
    def calc_Ti(self):
        # Ion temperature
        Ti = self.get_vars3D(['Ti'])
        return (Ti[:, 0, :] + Ti[:, 1, :]) / 2.
    
    def calc_Te_Ti(self):
        # Electron temperature, ion temperature
        Te3D, Ti3D = self.get_vars3D(['Te', 'Ti'])
        Te = (Te3D[:, 0, :] + Te3D[:, 1, :]) / 2.
        Ti = (Ti3D[:, 0, :] + Ti3D[:, 1, :]) / 2.
        return Te, Ti
    
    def calc_sigma(self):
        # sigma, BetaI = pmag/pgas
        rho0, c = self.rho0, self.c    
        B02     = self.Field.B0 * self.Field.B0  
        keys    = ['rho', 'B', 'uu', 'gcov']
        rho, B, uu, gcov = self.get_vars2D(keys)
    
        # Calculate sigma
        b     = self.Field.calc_b(B, uu, gcov)
        b2    = self.Field.calc_b2(b, B, gcov)
        sigma = (b2 / B02) / (rho / rho0)
        return (sigma[:, 0, :] + sigma[:, 1, :]) / 2.

    def calc_sigma_BetaI(self):
        # sigma, BetaI = pmag/pgas
        rho0, c, B02 = self.rho0, self.c, self.Field.B0 * self.Field.B0
        rho, u_gas   = self.get_vars3D(['rho', 'u_gas'])
        uu, B, gcov  = self.get_vars3D(['uu', 'B', 'gcov'])
        Te, Ti       = self.get_vars3D(['Te', 'Ti'])

        # Calculate sigma
        b       = self.Field.calc_b(B, uu, gcov)
        b2      = self.Field.calc_b2(b, B, gcov)
        sigma3D = (b2 / B02) / (rho / rho0)
        # Calculate gas pressure and Beta^-1
        if (self.cgs):
            pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        else:
            pgas  = ((5./3.) - 1.) * u_gas
        BetaI3D = ((b2 / B02) / 2.) / (pgas / (rho0 * c * c))
        sigma = (sigma3D[:, 0, :] + sigma3D[:, 1, :]) / 2.
        BetaI = (BetaI3D[:, 0, :] + BetaI3D[:, 1, :]) / 2. 
        return sigma, BetaI

    # Velocities
    def calc_vphi_cyl(self):
        uu    = self.get_vars3D(['uu'])
        self.set_Coords3D()
        
        uu_cyl   = self.Coords3D.int_to_cyl_vec(uu, cpp=True)
        vphi_cyl = self.c * uu_cyl[2] / uu_cyl[0] 
        return (vphi_cyl[:, 0, :] + vphi_cyl[:, 1, :]) / 2.

    # Pressures
    def calc_pmag_pradgas(self):
        rho0, c, B02 = self.rho0, self.c, self.Field.B0 * self.Field.B0
        rho, u_gas   = self.get_vars3D(['rho', 'u_gas'])
        uu, B, gcov  = self.get_vars3D(['uu', 'B', 'gcov'])
        Te, Ti       = self.get_vars3D(['Te', 'Ti'])
        u_rad        = self.get_vars3D(['u_rad'])

        b2    = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)     
        pmag  = ((b2 / B02) / 2.) * (rho0 * c * c)
        if self.cgs:
            pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti) 
        else:
            pgas = ((5./3.) - 1.) * u_gas
        prad  = (1./3.) * u_rad

        pmag_avg = (pmag[:, 0, :] + pmag[:, 1, :]) / 2.
        prad_avg = (prad[:, 0, :] + prad[:, 1, :]) / 2.
        pgas_avg = (pgas[:, 0, :] + pgas[:, 1, :]) / 2.

        return pmag_avg / (prad_avg + pgas_avg)
    
    def calc_pmag_prad(self):
        rho0, c, B02 = self.rho0, self.c, self.Field.B0 * self.Field.B0
        rho, u_gas   = self.get_vars3D(['rho', 'u_gas'])
        uu, B, gcov  = self.get_vars3D(['uu', 'B', 'gcov'])
        u_rad        = self.get_vars3D(['u_rad'])

        b2    = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)     
        pmag  = ((b2 / B02) / 2.) * (rho0 * c * c)
        prad  = (1./3.) * u_rad

        pmag_avg = (pmag[:, 0, :] + pmag[:, 1, :]) / 2.
        prad_avg = (prad[:, 0, :] + prad[:, 1, :]) / 2.

        return pmag_avg / prad_avg
    
    def calc_prad_pgas(self):
        u_gas, u_rad = self.get_vars3D(['u_gas', 'u_rad'])
        Te, Ti       = self.get_vars3D(['Te', 'Ti'])
        
        if self.cgs:
            pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti) 
        else:
            pgas = ((5./3.) - 1.) * u_gas
        prad  = (1./3.) * u_rad

        prad_avg = (prad[:, 0, :] + prad[:, 1, :]) / 2.
        pgas_avg = (pgas[:, 0, :] + pgas[:, 1, :]) / 2.

        return prad_avg / pgas_avg
    
    # Magnetic Fields
    def calc_B_cyl(self, **kwargs):
        B = self.get_vars3D(['B'])
        self.set_Coords3D()

        # Calculate B in cylindrical coordinates
        B_cyl = self.Coords3D.int_to_cyl_vec(B)
        if not kwargs.get('lr', False):
            BR   = (B_cyl[1][:, 0, :] + B_cyl[1][:, 1, :]) / 2.
            Bphi = (B_cyl[2][:, 0, :] + B_cyl[2][:, 1, :]) / 2.
            Bz   = (B_cyl[3][:, 0, :] + B_cyl[3][:, 1, :]) / 2.
            return BR, Bphi, Bz
        else:
            BR, Bphi, Bz = {}, {}, {}
            BR['r']   = (B_cyl[1][:, 0, :] + B_cyl[1][:, 1, :]) / 2.
            Bphi['r'] = (B_cyl[2][:, 0, :] + B_cyl[2][:, 1, :]) / 2.
            Bz['r']   = (B_cyl[3][:, 0, :] + B_cyl[3][:, 1, :]) / 2.
            if not kwargs.get('in_out', False):
                return BR, Bphi, Bz
            else:
                in_out = {}
                in_out['r'] = np.sign(B[1, :, 0, :] + B[1, :, 1, :])
                return BR, Bphi, Bz, in_out
    
    def calc_bpol_btor(self):
        uu, B, gcov  = self.get_vars3D(['uu', 'B', 'gcov'])
        self.set_Coords3D()
        b        = self.Field.calc_b(B, uu, gcov)
        bcyl     = self.Coords3D.int_to_cyl_vec(b, dtype=np.float64) 
        bpol     = np.copy(b)
        btor     = np.copy(b)
        bpol[0], bpol[3]          = 0., 0.
        btor[0], btor[1], btor[2] = 0., 0., 0.

        sgn     = np.sign(bcyl[3])
        b2pol   = self.Field.calc_b2(bpol, bpol, gcov)
        b2tor   = self.Field.calc_b2(btor, btor, gcov)
        pol_tor = sgn * np.sqrt(b2pol / b2tor)
        return (pol_tor[:, 0, :] + pol_tor[:, 1, :]) / 2.

    def calc_B_cart(self, **kwargs):
        B = self.get_vars3D(['B'])
        self.set_Coords3D()

        # Calculate B in Cartesian coordinates
        B_cart = self.Coords3D.int_to_cart_vec(B) / self.Field.B0
        if not kwargs.get('lr', False):
            Bx = (B_cart[1][:, 0, :] + B_cart[1][:, 1, :]) / 2.
            By = (B_cart[2][:, 0, :] + B_cart[2][:, 1, :]) / 2.
            return Bx, By
        else:
            Bx, By = {}, {}
            Bx['r'] = (B_cart[1][:, 0, :] + B_cart[1][:, 1, :]) / 2.
            By['r'] = (B_cart[2][:, 0, :] + B_cart[2][:, 1, :]) / 2.
            if not kwargs.get('in_out', False):
                return Bx, By
            else:
                in_out = {}
                in_out['r'] = np.sign(B[1, :, 0, :] + B[1, :, 1, :])
                return Bx, By, in_out
    
    # Fluid velocities
    def calc_v_cyl(self, **kwargs):
        uu = self.get_vars3D(['uu'])
        self.set_Coords3D()

        # Calculate uu in cylindrical coordinates
        uu_cyl = self.Coords3D.int_to_cyl_vec(uu) / self.c
        if not kwargs.get('lr', False):
            ut   = (uu_cyl[1][:, 0, :] + uu_cyl[1][:, 1, :]) / 2.
            uR   = (uu_cyl[1][:, 0, :] + uu_cyl[1][:, 1, :]) / 2.
            uphi = (uu_cyl[2][:, 0, :] + uu_cyl[2][:, 1, :]) / 2.
            uz   = (uu_cyl[3][:, 0, :] + uu_cyl[3][:, 1, :]) / 2.
            return uR/ut, uphi/ut, uz/ut
        else:
            vR, vphi, vz = {}, {}, {}
            ut        = (uu_cyl[0][:, 0, :] + uu_cyl[0][:, 1, :]) / 2.
            vR['r']   = (uu_cyl[1][:, 0, :] + uu_cyl[1][:, 1, :]) / 2.
            vphi['r'] = (uu_cyl[2][:, 0, :] + uu_cyl[2][:, 1, :]) / 2.
            vz['r']   = (uu_cyl[3][:, 0, :] + uu_cyl[3][:, 1, :]) / 2.
            vR['r']  *= 1./ut
            vphi['r']*= 1./ut
            vz['r']  *= 1./ut
            if not kwargs.get('in_out', False):
                return vR, vphi, vz
            else:
                in_out = {}
                in_out['r'] = np.sign(uu[1, :, 0, :] + uu[1, :, 1, :])
                return vR, vphi, vz, in_out
    
    def calc_vpol_vtor(self):
        uu, gcov  = self.get_vars3D(['uu', 'gcov'])
        self.set_Coords3D()
        uu_cyl    = self.Coords3D.int_to_cyl_vec(uu, dtype=np.float64) / self.c
        uu_pol    = np.copy(uu)
        uu_tor    = np.copy(uu)
        uu_pol[0], uu_pol[3]            = 0., 0.
        uu_tor[0], uu_tor[1], uu_tor[2] = 0., 0., 0.

        sgn     = np.sign(uu_cyl[3])
        u2pol   = ana.Tools.dot(uu_pol, uu_pol, gcov)
        u2tor   = ana.Tools.dot(uu_tor, uu_tor, gcov)
        pol_tor = sgn * np.sqrt(u2pol / u2tor)
        return (pol_tor[:, 0, :] + pol_tor[:, 1, :]) / 2.
    
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
    
    def calc_uu_rad_cart(self, **kwargs):
        uu_rad = self.get_vars3D(['uu_rad'])
        self.set_Coords3D()

        # Calculate uu_rad in Cartesian coordinates
        uuR_cart = self.Coords3D.int_to_cart_vec(uu_rad) / self.c
        if not kwargs.get('lr', False):
            uuRx = (uuR_cart[1][:, 0, :] + uuR_cart[1][:, 1, :]) / 2.
            uuRy = (uuR_cart[2][:, 0, :] + uuR_cart[2][:, 1, :]) / 2.
            return uuRx, uuRy
        else:
            uuRx, uuRy = {}, {}
            uuRx['r'] = (uuR_cart[1][:, 0, :] + uuR_cart[1][:, 1, :]) / 2.
            uuRy['r'] = (uuR_cart[2][:, 0, :] + uuR_cart[2][:, 1, :]) / 2.
            if not kwargs.get('in_out', False):
                return uuRx, uuRy
            else:
                in_out = {}
                in_out['r'] = np.sign(uu_rad[1, :, 0, :] + uu_rad[1, :, 1, :])
                return uuRx, uuRy, in_out

    # Cooling rates
    # Inverse-Compton cooling rate
    def calc_qIC(self, **kwargs):
        rho, Te, u_rad = self.get_vars3D(['rho', 'Te', 'u_rad'])
        uu, gcov       = self.get_vars3D(['uu', 'gcov'])
        u_t            = np.einsum('ij...,j...->i...', gcov, uu)[0] / self.c
        qIC            = self.Rad.calc_qIC(rho, Te, u_rad, **kwargs) 
        qICu_t         = ((qIC * u_t)[:, 0, :] + (qIC * u_t)[:, 1, :]) / 2.
        return -qICu_t

    # Coulomb cooling rate
    def calc_qie(self, **kwargs):
        rho, Ti, Te = self.get_vars3D(['rho', 'Ti', 'Te'])
        uu, gcov    = self.get_vars3D(['uu', 'gcov'])
        u_t         = np.einsum('ij...,j...->i...', gcov, uu)[0] / self.c
        qie         = self.Fluid.calc_qie(rho, Te, Ti, **kwargs)
        qieu_t      = ((qie * u_t)[:, 0, :] + (qie * u_t)[:, 1, :]) / 2.
        return -qieu_t

    # Synchrotron cooling rate
    def calc_qsync(self, **kwargs):
        rho, Te, u_rad = self.get_vars3D(['rho', 'Te', 'u_rad'])
        B, uu, gcov    = self.get_vars3D(['B', 'uu', 'gcov'])
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
            'XRB'       : True
        }
        qsync    = self.Rad.calc_q_emi(rho, Te, u_rad, **sync_args)
        qsyncu_t = ((qsync * u_t)[:, 0, :] + (qsync * u_t)[:, 1, :]) / 2.
        return -qsyncu_t

    # Cooling timescales
    def calc_teq(self, **kwargs):
        rho, Te, Ti = self.get_vars3D(['rho', 'Te', 'Ti'])
        teq = self.Fluid.calc_teq(rho, Te, Ti, **kwargs)     
        return (teq[:, 0, :] + teq[:, 1, :]) / 2.
    
    def calc_tii(self, **kwargs):
        rho, Te, Ti = self.get_vars3D(['rho', 'Te', 'Ti'])
        tii = self.Fluid.calc_tii(rho, Te, Ti, **kwargs)     
        return (tii[:, 0, :] + tii[:, 1, :]) / 2.
    
    def calc_tee(self, **kwargs):
        rho, Te, Ti = self.get_vars3D(['rho', 'Te', 'Ti'])
        tee = self.Fluid.calc_tee(rho, Te, Ti, **kwargs)     
        return (tee[:, 0, :] + tee[:, 1, :]) / 2.
    
    def calc_tIC(self, **kwargs):
        rho, Te, u_rad = self.get_vars3D(['rho', 'Te', 'u_rad'])
        tIC = self.Rad.calc_tIC(rho, Te, u_rad, **kwargs)     
        return (tIC[:, 0, :] + tIC[:, 1, :]) / 2.
    
    # Orbital frequency
    def calc_OmegaK(self):
        return self.Analytic.calc_OmegaK(
            self.MBH, self.H.params['a'], self.r)[0, :, 0, :]

    # Opacities
    def calc_kappa_es(self, **kwargs):
        rho, Te, u_rad = self.get_vars3D(['rho', 'Te', 'u_rad'])
        opacities = dict.fromkeys(['electron_scattering'])
        Opac = rad.Opacity(opacities, rho, Te, u_rad, self.H.params, 
                           hr.CONSTANTS, mechanism='emission')
        return Opac.opacities['electron_scattering']
    
    def calc_kappa_sync(self, **kwargs):
        rho, Te, u_rad = self.get_vars3D(['rho', 'Te', 'u_rad'])
         # b2 is required for synchrotron opacities
        if 'b2' not in kwargs:
            B, uu, gcov = self.get_vars3D(['B', 'uu', 'gcov'])
            b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        else:
            b2 = kwargs.get('b2')
        opacities = dict.fromkeys(['synchrotron'])
        Opac = rad.Opacity(opacities, rho, Te, u_rad, self.H.params, 
                           hr.CONSTANTS, mechanism='emission', b2=b2, XRB=True)
        kappa = Opac.opacities['synchrotron_emi']
        return (kappa[:, 0, :] + kappa[:, 1, :]) / 2.
    
    def calc_kappa_ff(self, **kwargs):
        rho, Te, u_rad = self.get_vars3D(['rho', 'Te', 'u_rad'])
         # b2 is required for synchrotron opacities
        opacities = dict.fromkeys(['free_free'])
        Opac = rad.Opacity(opacities, rho, Te, u_rad, self.H.params, 
                           hr.CONSTANTS, mechanism='emission', XRB=True)
        kappa = Opac.opacities['free_free_emi']
        return (kappa[:, 0, :] + kappa[:, 1, :]) / 2.

class Slices():
    # Retrieve file numbers based on file_number, start_file, and end file
    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        reduced    = kwargs.get('reduced', False)
        slices_dir = (f'{averages_dir}/Full_Resolution/Slices' if not reduced 
                      else f'{averages_dir}/Reduced_Resolution/Slices')
        single_file = True if (file_number != -1) else False
        if (single_file):
            fnums = list([file_number])
        else:
            slcs = glob.glob("Slices*", root_dir=slices_dir)
            slice_nums = [int(slice_name.strip('Slices_').strip('.npz')) for slice_name in slcs]
            slice_nums.sort()
            if ((start_file != -1) and (end_file != -1)):
                fnums = list(range(start_file, end_file + 1))
            elif ((start_file != -1) and (end_file == -1)):
                last_file = slice_nums[-1]
                fnums = list(range(start_file, last_file + 1))
            elif ((start_file == -1) and (end_file != -1)):
                first_file = slice_nums[0]
                fnums = list(range(first_file, last_file + 1))
            else:
                fnums = list(range(slice_nums[0], slice_nums[-1] + 1))
            fnums = self.check_slices_exist(fnums, slice_nums)
        return fnums

    def check_slices_exist(self, fnums, slice_nums):
        removed_files = []
        for fnum in fnums:
            if fnum not in slice_nums:
                print(f"Slice {fnum} is missing.")
                removed_files.append(fnum)
        for removed_file in removed_files:
            fnums.remove(removed_file)
        return fnums

    # Create slices dictionary [file_number][key]
    def load_slices(self, averages_dir, fnums, **kwargs):
        reduced      = kwargs.get('reduced', False)
        slices_dir = (f'{averages_dir}/Full_Resolution/Slices' if not reduced 
                      else f'{averages_dir}/Reduced_Resolution/Slices')
        print(f"Loading from {slices_dir}...")
        slcs = glob.glob("Slices*", root_dir=slices_dir)
        slice_nums = [
            int(slice_name.strip('Slices_').strip('.npz')) for slice_name in slcs
        ]
        slice_nums.sort()
        nzeros = len(str(int(len(slice_nums))))
        if 'slices' not in kwargs:
            slices = {}
        else:
            slices = kwargs.get('slices')
        for fnum in fnums:
            slice_file = (
                f'{slices_dir}/Slices_%s.npz' %str(fnum).zfill(nzeros))
            if not kwargs.get('keys', False):
                slices[f'{fnum}'] = np.load(slice_file)
                files = slices[f'{fnum}'].files
            else:
                if 'slices' not in kwargs:
                    slices[f'{fnum}'] = {}
                for key in kwargs.get('keys'):
                    slices[f'{fnum}'][f'{key}']=np.load(slice_file)[f'{key}']
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
            self.fnums0 = self.nums[self.nr == max(self.nr)][0]
        elif (self.AMRx1 and not self.AMRx2):
            self.fnums0 = self.nums[self.nr == max(self.nr)][0]
        elif (not self.AMRx1 and self.AMRx2):
            self.fnums0 = self.nums[self.nth == max(self.nth)][0]
        else:
            self.fnums0 = self.nums[0]

    def time_avg(self, quant, slices, **kwargs):
        fnums = slices.keys() if 'fnums' not in kwargs else kwargs.get('fnums')
        if not hasattr(self, "AMRx1") or not hasattr(self, "AMRx2"):
            self.check_AMR(fnums, slices)
        q_avg = np.zeros([len(slices[f'{self.fnums0}']['r']), 
                          len(slices[f'{self.fnums0}']['theta'])])
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

    def upscale(self, q):
        level_x1 = (max(self.nr) / len(q[:, 0])) - 1
        level_x2 = (max(self.nth)/ len(q[0, :])) - 1
        if (level_x1 == 0 and level_x2 == 0): 
            return q
        else:
            q_up = np.zeros([max(self.nr), max(self.nth)])
            for i in range(len(q[:, 0])):
                for j in range(len(q[0, :])):
                    istrt = int(i       * (2 ** level_x1))
                    iend  = int((i + 1) * (2 ** level_x1))
                    jstrt = int(j       * (2 ** level_x2))
                    jend  = int((j + 1) * (2 ** level_x2))
                q_up[istrt:iend, jstrt:jend] = q[i, j]
            return q_up

class Profiles():
    # Retrieve file numbers based on file_number, start_file, and end file
    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        reduced      = kwargs.get('reduced', False)
        profiles_dir = (f'{averages_dir}/Full_Resolution/Profiles' if not reduced 
                        else f'{averages_dir}/Reduced_Resolution/Profiles')
        single_file = True if (file_number != -1) else False
        if (single_file):
            fnums = list([file_number])
        else:
            profs = glob.glob("Profiles*", root_dir=profiles_dir)
            prof_nums = [int(prof_name.strip('Profiles_').strip('.npz')) for prof_name in profs]
            prof_nums.sort()
            if ((start_file != -1) and (end_file != -1)):
                fnums = list(range(start_file, end_file + 1))
            elif ((start_file != -1) and (end_file == -1)):
                last_file = prof_nums[-1]
                fnums = list(range(start_file, last_file + 1))
            elif ((start_file == -1) and (end_file != -1)):
                first_file = prof_nums[0]
                fnums = list(range(first_file, last_file + 1))
            else:
                fnums = list(range(prof_nums[0], prof_nums[-1] + 1))
            fnums = self.check_profiles_exist(fnums, prof_nums)
        return fnums

    def check_profiles_exist(self, fnums, prof_nums):
        removed_files = []
        for fnum in fnums:
            if fnum not in prof_nums:
                print(f"Profile {fnum} is missing.")
                removed_files.append(fnum)
        for removed_file in removed_files:
            fnums.remove(removed_file)
        return fnums
    
    # Create profiles dictionary [file_number][key]
    def load_profiles(self, averages_dir, fnums, **kwargs):
        reduced      = kwargs.get('reduced', False)
        profiles_dir = (f'{averages_dir}/Full_Resolution/Profiles' if not reduced 
                        else f'{averages_dir}/Reduced_Resolution/Profiles')
        print(f"Loading from {profiles_dir}...")
        profs = glob.glob("Profiles*", root_dir=profiles_dir)
        prof_nums = [
            int(prof_name.strip('Profiles_').strip('.npz')) for prof_name in profs
        ]
        prof_nums.sort()
        nzeros = len(str(int(len(prof_nums))))
        if 'profiles' not in kwargs:
            profiles = {}
        else:
            profiles = kwargs.get('profiles')
        for fnum in fnums:
            profile_file = (
                f'{profiles_dir}/Profiles_%s.npz' %str(fnum).zfill(nzeros))
            if not kwargs.get('keys', False):
                profiles[f'{fnum}'] = np.load(profile_file)
                files = profiles[f'{fnum}'].files
            else:
                if 'profiles' not in kwargs:
                    profiles[f'{fnum}'] = {}
                for key in kwargs.get('keys'):
                    profiles[f'{fnum}'][f'{key}']=np.load(profile_file)[f'{key}']
        return profiles
    
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
        level = (max(self.nr) / len(q)) - 1
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
            self.fnums0 = self.nums[self.nr == max(self.nr)][0]
        else:
            self.fnums0 = self.nums[0] 

class Time_Series(Profiles):
    def calc_time_series(self, key, profiles, **kwargs):
        fnums = profiles.keys()
        time  = np.zeros(len(fnums))
        q     = np.zeros(len(fnums))
        r0    = kwargs.get('r0', 0.)
        for i, fnum in enumerate(fnums):
            r       = profiles[f'{fnum}']['r']
            idx     = np.argmin(np.abs(r - r0))
            time[i] = profiles[f'{fnum}']['t']
            vol     = profiles[f'{fnum}']['Vol'] if kwargs.get('vol', False) else 1.
            q[i]    = (profiles[f'{fnum}'][f'{key}']/vol)[idx]
        return time, q