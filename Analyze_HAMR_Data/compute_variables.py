import sys
import os
import argparse
import numpy as np
sys.path.insert(0, 
'/home/matthew/Desktop/cbambic/HAMR_Visualization/Read_HAMR_Data')
from scipy.interpolate import CubicSpline
import analysis as ana
import radiation as rad
import hamr_read as hr

##############################################################################

# Compute quantities at constant phi
class Phi_Slice():
    def __init__(self, directory, fnum, **kwargs):
        self.MBH = kwargs.get('MBH', 10.0)
        self.H = hr.Read(directory, cgs_data=True, cgs_grid=True, 
            coords1D=True, MBH=self.MBH, 
            lowres1=int(kwargs.get('lowres1', 1)), 
            lowres2=int(kwargs.get('lowres2', 1)), 
            lowres3=int(kwargs.get('lowres3', 1)),
            r_max   = kwargs.get('r_max', None),
            phi_min = kwargs.get('phi_min', 0.0),
            phi_max = kwargs.get('phi_max', 2. * np.pi))
        self.H.read_grid( fnum, direct=True)
        self.H.read_prims(fnum, skip_read_params=True)

        self.Msun = hr.CONSTANTS['Msun']
        self.G    = hr.CONSTANTS['G']
        self.c    = hr.CONSTANTS['c']
        self.Rg   = self.G * self.MBH * self.Msun / (self.c * self.c)

        self.r    = self.H.grid['r'] / self.Rg  
        self.th   = self.H.grid['theta']
        self.phi  = self.H.grid['phi']

        phi0        = np.pi
        self.phi_1D = self.phi[0, 0, 0, :]
        self.iphi   = np.argmin(np.abs(self.phi_1D - phi0))

        # Initialize Accretion class
        self.Acc = ana.Accretion(self.H.grid, self.H.data, self.H.params, 
            hr.CONSTANTS, cgs=True, MBH=self.MBH, radiation=True)
        # Initialize Coordinates class
        self.Coords = ana.Coordinates(self.H.grid['r'], self.H.grid['theta'], 
            self.H.grid['phi'], self.H.params, hr.CONSTANTS)
        # Initialize Fluid class
        self.Fluid = ana.Fluid(self.H.params, hr.CONSTANTS, cgs=True)
        # Initialize Field class
        self.Field = ana.Field(self.H.params, hr.CONSTANTS, cgs=True)
        # Initialize Radiation class
        self.Rad = rad.Radiation(self.H.params, hr.CONSTANTS, cgs=True)
    
    def calc_XZ(self, iphi):
        r   = self.r
        th  = self.th
        phi = self.phi
        
        # Convert to Cartesian coordinates
        # Note reflection for left panel
        self.x =  (r*np.sin(th)*np.cos(phi))[0, :, :, iphi]
        self.y =  (r*np.sin(th)*np.sin(phi))[0, :, :, iphi]
        self.z =  (r*np.cos(th)*np.ones(phi.shape))[0, :, :, iphi]
        
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
        self.x =  (r*np.sin(th)*np.cos(phi))[0, :, :, iphi]
        self.y =  (r*np.sin(th)*np.sin(phi))[0, :, :, iphi]
        self.z =  (r*np.cos(th)*np.ones(phi.shape))[0, :, :, iphi]
        
        R, z = {}, {}
        R['r'] = np.sqrt(np.square(self.x) + np.square(self.y))
        z['r'] = self.z
        R['l'] = -np.sqrt(np.square(self.x) + np.square(self.y))
        z['l'] = self.z
        return R, z
    
    def calc_rho(self, iphi):
        # Density
        rho = self.H.data['rho'][0, :, :, iphi]
        return rho

    def calc_thermal_variables(self, iphi):
        # Density, electron temperature, ion temperature
        rho = self.H.data['rho'][0, :, :, iphi]
        Te  = self.H.data['Te'][ 0, :, :, iphi]
        Ti  = self.H.data['Ti'][ 0, :, :, iphi]
        return rho, Te, Ti

    def calc_sigma_BetaI(self, iphi):
        # sigma, BetaI = pmag/pgas
        c = self.c # required constants
        rho   = self.H.data['rho'][       0, :, :, iphi]
        B     = self.H.data['B'][:,       0, :, :, iphi]
        uu    = self.H.data['uu'][:,      0, :, :, iphi]
        u_gas = self.H.data['u_gas'][     0, :, :, iphi]
        Te    = self.H.data['Te'][        0, :, :, iphi]
        Ti    = self.H.data['Ti'][        0, :, :, iphi]
        gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]
    
        # Calculate sigma
        b     = self.Field.calc_b(B, uu, gcov)
        b2    = self.Field.calc_b2(b, B, gcov)
        sigma = (b2 / (4. * np.pi)) / (rho * c * c)
        # Calculate gas pressure and Beta^-1
        pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        BetaI = (b2 / (8. * np.pi)) / pgas
        return sigma, BetaI

    def calc_stress_energy(self, iphi):
        # MHD, electromagnetic, radiation, and fluid energy outflow rates 
        c, Rg = self.c, self.Rg # required constants
        rho   = self.H.data['rho'][       0, :, :, iphi]
        uu    = self.H.data['uu'][:,      0, :, :, iphi]
        u_gas = self.H.data['u_gas'][     0, :, :, iphi]
        B     = self.H.data['B'][:,       0, :, :, iphi]
        Te    = self.H.data['Te'][        0, :, :, iphi]
        Ti    = self.H.data['Ti'][        0, :, :, iphi]
        u_rad = self.H.data['u_rad'][     0, :, :, iphi]
        uu_rad= self.H.data['uu_rad'][:,  0, :, :, iphi]
        gdet  = self.H.grid['gdet'][      0, :, :, 0]
        gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]
        gcon  = self.H.grid['gcon'][:, :, 0, :, :, 0]

        pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)

        # Stress-energy tensor(s)
        T_Hydro = self.Fluid.calc_THD(rho, uu, u_gas, pgas, gcov, gcon, 
            upper_lower=True, index = [1, 0])
        T_EM    = self.Field.calc_TEM(uu, B, gcov, gcon, 
            upper_lower=True, index = [1, 0])
        T_Rad   = self.Rad.calc_TRAD(u_rad, uu_rad, gcov, gcon, 
            upper_lower=True, index = [1, 0])

        T_MHD  = T_Hydro + T_EM + T_Rad

        Area = 4 * np.pi * np.square(Rg * self.r)[0, :, :, iphi]
        Edot_MHD  = -Area * c * T_MHD 
        Edot_EM   = -Area * c * T_EM 
        Edot_Rad  = -Area * c * T_Rad
        return Edot_MHD, Edot_EM, Edot_Rad

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

    def calc_B_cart(self, iphi):
        Bx, Bz, in_out = {}, {}, {}
        B     = self.H.data['B'][:,       0, :, :, iphi]
        uu    = self.H.data['uu'][:,      0, :, :, iphi]
        gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]
    
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]

        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate B in Cartesian coordinates
        B_cart = Coords.int_to_cart_vec(B) / self.Field.B0
        in_out['r'] = np.sign(B[1])
        in_out['l'] = np.sign(B[1])

        Bx['r'] = B_cart[1]
        Bz['r'] = B_cart[3]
        Bx['l'] = -B_cart[1]
        Bz['l'] = B_cart[3]
        return Bx, Bz, in_out

    def calc_uu_cart(self, iphi):
        uux, uuz, in_out = {}, {}, {}
        uu = self.H.data['uu'][:, 0, :, :, iphi]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate uu in Cartesian coordinates
        uu_cart = Coords.int_to_cart_vec(uu) / self.c
        in_out['r'] = np.sign(uu[1])
        in_out['l'] = np.sign(uu[1])
        
        uux['r'] = uu_cart[1]
        uuz['r'] = uu_cart[3]
        uux['l'] = -uu_cart[1]
        uuz['l'] = uu_cart[3]
        return uux, uuz, in_out
    
    def calc_uu_rad_cart(self, iphi):
        uu_radx, uu_radz, in_out = {}, {}, {}
        uu_rad = self.H.data['uu_rad'][:, 0, :, :, iphi]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        rho0 = self.H.params['density_scale']
        c    = self.c
        # Calculate uu in Cartesian coordinates
        uu_rad_cart = Coords.int_to_cart_vec(uu_rad) / (rho0 * c * c * c)
        
        in_out['r'] = np.sign(uu_rad[1])
        in_out['l'] = np.sign(uu_rad[1])
        
        uu_radx['r'] = uu_rad_cart[1]
        uu_radz['r'] = uu_rad_cart[3]
        uu_radx['l'] = -uu_rad_cart[1]
        uu_radz['l'] = uu_rad_cart[3]
        return uu_radx, uu_radz, in_out
    
    def calc_B_cyl(self, iphi):
        BR, Bz, in_out = {}, {}, {}
        B     = self.H.data['B'][:,       0, :, :, iphi]
        uu    = self.H.data['uu'][:,      0, :, :, iphi]
        gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]
    
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]

        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate B in Cartesian coordinates
        B_cyl = Coords.int_to_cyl_vec(B) / self.Field.B0
        in_out['r'] = np.sign(B[1])
        in_out['l'] = np.sign(B[1])

        BR['r'] = B_cyl[1]
        Bz['r'] = B_cyl[3]
        BR['l'] = -B_cyl[1]
        Bz['l'] = B_cyl[3]
        return BR, Bz, in_out

    def calc_uu_cyl(self, iphi):
        uuR, uuz, in_out = {}, {}, {}
        uu = self.H.data['uu'][:, 0, :, :, iphi]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate uu in Cartesian coordinates
        uu_cyl = Coords.int_to_cyl_vec(uu) / self.c
        in_out['r'] = np.sign(uu[1])
        in_out['l'] = np.sign(uu[1])
        
        uuR['r'] = uu_cyl[1]
        uuz['r'] = uu_cyl[3]
        uuR['l'] = -uu_cyl[1]
        uuz['l'] = uu_cyl[3]
        return uuR, uuz, in_out
    
    def calc_uu_rad_cyl(self, iphi):
        uu_radR, uu_radz, in_out = {}, {}, {}
        uu_rad = self.H.data['uu_rad'][:, 0, :, :, iphi]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        rho0 = self.H.params['density_scale']
        c    = self.c
        # Calculate uu in Cartesian coordinates
        uu_rad_cyl = Coords.int_to_cyl_vec(uu_rad) / (rho0 * c * c * c)
        
        in_out['r'] = np.sign(uu_rad[1])
        in_out['l'] = np.sign(uu_rad[1])
        
        uu_radR['r'] = uu_rad_cyl[1]
        uu_radz['r'] = uu_rad_cyl[3]
        uu_radR['l'] = -uu_rad_cyl[1]
        uu_radz['l'] = uu_rad_cyl[3]
        return uu_radR, uu_radz, in_out

    def calc_TRphi_Tzphi(self, iphi):
        B     = self.H.data['B'][:,       0, :, :, iphi]
        uu    = self.H.data['uu'][:,      0, :, :, iphi]
        gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]
    
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]

        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate b in cylindridcal coordinates
        b     = self.Field.calc_b(B, uu, gcov)
        b_cyl = Coords.int_to_cyl_vec(b) 
        Trphi = -(b_cyl[1] * b_cyl[2]) / (4. * np.pi)
        Tzphi = -(b_cyl[3] * b_cyl[2]) / (4. * np.pi)
        return Trphi, Tzphi

    def calc_qcool(self, iphi, **kwargs):
        # Thermal/radiation variables
        rho   = self.H.data['rho'][  0, :, :, iphi]
        Te    = self.H.data['Te'][   0, :, :, iphi]
        Ti    = self.H.data['Ti'][   0, :, :, iphi]
        u_rad = self.H.data['u_rad'][0, :, :, iphi]

        # b2 is required for synchrotron opacities
        if 'b2' not in kwargs:
            uu    = self.H.data['uu'][:,      0, :, :, iphi]
            B     = self.H.data['B'][:,       0, :, :, iphi]
            gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]
            b     = self.Field.calc_b(B, uu, gcov)
            b2    = self.Field.calc_b2(b, B, gcov)
        else:
            b2    = kwargs.get('b2')

        # Calculate q_IC, q_sync, q_ie
        opacities  = {'synchrotron': None}
        qIC   = self.Rad.calc_qIC(rho, Te, u_rad)
        qsync = self.Rad.calc_q_emi(rho, Te, u_rad, 
            opacities = opacities, b2 = b2, XRB=True)
        qie   = self.Fluid.calc_qie(rho, Te, Ti)

        # Clean q_IC
        Trad  = (u_rad / hr.CONSTANTS['aRAD'])**(1./4.)
        qIC   = self.clean_minima(qIC, (Te < Trad))

        # Calculate q_sync / q_ie
        qie_clean = np.copy(qie)
        qie_clean[qie == 0.] = 1.
        qsync_qie = qsync / qie_clean
        qsync_qie[(Te > Ti)] = np.nan
    
        # Calculate q_IC / q_ie
        qie_clean = np.copy(qie)
        qie_clean[qie == 0.] = 1.
        qIC_qie = qIC / qie_clean
        # Te > Ti: not two-temperature
        qIC_qie[(Te > Ti)] = np.nan
        # Te < Tr: no Compton cooling
        qIC_qie[(Te < Trad)] = np.nan

        # Clean q_ie
        qie = self.clean_minima(qie, (Te > Ti))
    
        return qIC, qsync, qie, qIC_qie, qsync_qie
        
    def calc_teq_tIC(self, iphi, **kwargs):
        # Thermal/radiation variables
        rho   = self.H.data['rho'][  0, :, :, iphi]
        Te    = self.H.data['Te'][   0, :, :, iphi]
        Ti    = self.H.data['Ti'][   0, :, :, iphi]
        u_rad = self.H.data['u_rad'][0, :, :, iphi]

        # Calculate t_eq / t_IC
        tIC   = self.Rad.calc_tIC(  rho, Te, u_rad, cooling=True)
        teq   = self.Fluid.calc_teq(rho, Te, Ti,    cooling=True)
        
        teq[tIC == 0.] = 0.0
        tIC[teq == 0.] = self.H.params['dt']
        tIC[tIC == 0.] = self.H.params['dt']
        teq_tIC        = teq / tIC
        teq_tIC[teq_tIC < 1.e-6] = 1.e-6
        return teq_tIC

    def clean_minima(self, q, condition):
        # set all q where condition is met to 0, then
        # set all zeros to minimum non-zero q
        q[condition] = 0.0
        min_q        = np.amin(q[q > 0.])
        q[q < min_q] = min_q
        return q

    def calc_tau(self, opacities, iphi, **kwargs):
        c, Rg = self.c, self.Rg
        # Thermal/radiation variables
        rho   = self.H.data['rho'][  0, :, :, iphi]
        Te    = self.H.data['Te'][   0, :, :, iphi]
        Ti    = self.H.data['Ti'][   0, :, :, iphi]
        u_rad = self.H.data['u_rad'][0, :, :, iphi]
        gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]

        if 'synchrotron' in opacities or 'sigma' not in kwargs:
            # b2 is required for synchrotron opacities
            if 'b2' not in kwargs:
                uu    = self.H.data['uu'][:,      0, :, :, iphi]
                B     = self.H.data['B'][:,       0, :, :, iphi]
                b     = self.Field.calc_b(B, uu, gcov)
                b2    = self.Field.calc_b2(b, B, gcov)
            else:
                b2    = kwargs.get('b2')
            sigma = b2 / (4. * np.pi * rho * c * c)
            Opac  = rad.Opacity(opacities, rho, Te, u_rad, 
                self.H.params, hr.CONSTANTS, b2 = b2, XRB = True)
        else:
            sigma = kwargs.get('sigma')
            Opac  = rad.Opacity(opacities, rho, Te, u_rad, 
                self.H.params, hr.CONSTANTS, XRB = True)
        
        if (len(Opac.opacities) == 1 and 'electron_scattering' in opacities):
            kappa = Opac.opacities['electron_scattering']
        else:
            kappa = Opac.opacities['absorption_total']

        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]

        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        if kwargs.get('integral') == 'z':
            return Rg * Coords.cum_integrate_z(kappa * rho, gcov[2, 2],
                sigma_cut = 3.0, sigma = sigma, theta_cut = np.pi/6.)
        else:
            return Rg * Coords.cum_integrate_th(kappa * rho, gcov[2, 2],
                sigma_cut = 3.0, sigma = sigma, theta_cut = np.pi/6.)


# Compute quantities at constant theta
class Theta_Slice():
    def __init__(self, directory, fnum, **kwargs):
        self.MBH = kwargs.get('MBH', 10.0)
        self.H = hr.Read(directory, cgs_data=True, cgs_grid=True, 
            coords1D=True, MBH=self.MBH, 
            lowres1=int(kwargs.get('lowres1', 1)), 
            lowres2=int(kwargs.get('lowres2', 1)), 
            lowres3=int(kwargs.get('lowres3', 1)),
            r_max     = kwargs.get('r_max', None),
            theta_min = 0.499 * np.pi, theta_max = 0.501 * np.pi)
        self.H.read_grid( fnum, direct=True)
        self.H.read_prims(fnum, skip_read_params=True)

        self.Msun = hr.CONSTANTS['Msun']
        self.G    = hr.CONSTANTS['G']
        self.c    = hr.CONSTANTS['c']
        self.Rg   = self.G * self.MBH * self.Msun / (self.c * self.c)

        self.r    = self.H.grid['r'] / self.Rg  
        self.th   = self.H.grid['theta']
        self.phi  = self.H.grid['phi']

        th0       = np.pi / 2.
        self.th_1D= self.th[0, 0, :, 0]
        self.ith  = np.argmin(np.abs(self.th_1D - th0))

        # Initialize Accretion class
        self.Acc = ana.Accretion(self.H.grid, self.H.data, self.H.params, 
            hr.CONSTANTS, cgs=True, MBH=self.MBH, radiation=True)
        # Initialize Coordinates class
        self.Coords = ana.Coordinates(self.H.grid['r'], self.H.grid['theta'], 
            self.H.grid['phi'], self.H.params, hr.CONSTANTS)
        # Initialize Fluid class
        self.Fluid = ana.Fluid(self.H.params, hr.CONSTANTS, cgs=True)
        # Initialize Field class
        self.Field = ana.Field(self.H.params, hr.CONSTANTS, cgs=True)
        # Initialize Radiation class
        self.Rad = rad.Radiation(self.H.params, hr.CONSTANTS, cgs=True)
    
    def calc_XY(self, ith):
        r   = self.r
        th  = self.th
        phi = self.phi
        
        # Convert to Cartesian coordinates
        # Note reflection for left panel
        self.x =  (r*np.sin(th)*np.cos(phi))[0, :, ith, :]
        self.y =  (r*np.sin(th)*np.sin(phi))[0, :, ith, :]
        self.z =  (r*np.cos(th)*np.ones(phi.shape))[0, :, ith, :]
        
        x, y = {}, {}
        x['r'] = self.x
        y['r'] = self.y
        return x, y
    
    def calc_rho(self, ith):
        # Density, electron temperature, ion temperature
        rho = self.H.data['rho'][0, :, ith, :]
        return rho

    def calc_thermal_variables(self, ith):
        # Density, electron temperature, ion temperature
        rho = self.H.data['rho'][0, :, ith, :]
        Te  = self.H.data['Te'][ 0, :, ith, :]
        Ti  = self.H.data['Ti'][ 0, :, ith, :]
        return rho, Te, Ti

    def calc_sigma_BetaI(self, ith):
        # sigma, BetaI = pmag/pgas
        c = self.c # required constants
        rho   = self.H.data['rho'][       0, :, ith, :]
        B     = self.H.data['B'][:,       0, :, ith, :]
        uu    = self.H.data['uu'][:,      0, :, ith, :]
        u_gas = self.H.data['u_gas'][     0, :, ith, :]
        Te    = self.H.data['Te'][        0, :, ith, :]
        Ti    = self.H.data['Ti'][        0, :, ith, :]
        gcov  = self.H.grid['gcov'][:, :, 0, :, ith, :]
    
        # Calculate sigma
        b     = self.Field.calc_b(B, uu, gcov)
        b2    = self.Field.calc_b2(b, B, gcov)
        sigma = (b2 / (4. * np.pi)) / (rho * c * c)
        # Calculate gas pressure and Beta^-1
        pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        BetaI = (b2 / (8. * np.pi)) / pgas
        return sigma, BetaI
    
    def calc_TRphi_Tzphi(self, ith):
        B     = self.H.data['B'][:,       0, :, ith, :]
        uu    = self.H.data['uu'][:,      0, :, ith, :]
        gcov  = self.H.grid['gcov'][:, :, 0, :, ith, :]
    
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, ith, :]
        th2D  = (ones * self.th)[ 0, :, ith, :]
        phi2D = (ones * self.phi)[0, :, ith, :]

        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate b in cylindridcal coordinates
        b     = self.Field.calc_b(B, uu, gcov)
        b_cyl = Coords.int_to_cyl_vec(b) 
        Trphi = -(b_cyl[1] * b_cyl[2]) / (4. * np.pi)
        Tzphi = -(b_cyl[3] * b_cyl[2]) / (4. * np.pi)
        return Trphi, Tzphi
    
    def calc_XY_uni(self, box_lim, npts):
        # Make uniform grid for streamline plots
        x1D = np.linspace(-box_lim, box_lim, npts)
        y1D = np.linspace(-box_lim, box_lim, npts)
        X_uni, Y_uni = np.meshgrid(x1D, y1D)
        return X_uni, Y_uni

    def calc_B_cart(self, ith):
        Bx, By, in_out = {}, {}, {}
        B     = self.H.data['B'][:,       0, :, ith, :]
        uu    = self.H.data['uu'][:,      0, :, ith, :]
        gcov  = self.H.grid['gcov'][:, :, 0, :, ith, :]
    
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, ith, :]
        th2D  = (ones * self.th)[ 0, :, ith, :]
        phi2D = (ones * self.phi)[0, :, ith, :]

        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate B in Cartesian coordinates
        B_cart = Coords.int_to_cart_vec(B) / self.Field.B0
        in_out['r'] = np.sign(B[3])

        Bx['r'] = B_cart[1]
        By['r'] = B_cart[2]
        return Bx, By, in_out

    def calc_uu_cart(self, ith):
        uux, uuy, in_out = {}, {}, {}
        uu = self.H.data['uu'][:, 0, :, ith, :]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, ith, :]
        th2D  = (ones * self.th)[ 0, :, ith, :]
        phi2D = (ones * self.phi)[0, :, ith, :]
 
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate uu in Cartesian coordinates
        uu_cart = Coords.int_to_cart_vec(uu) / self.c
        in_out['r'] = np.sign(uu[3])
        
        uux['r'] = uu_cart[1]
        uuy['r'] = uu_cart[2]
        return uux, uuy, in_out
    
    def calc_uu_rad_cart(self, ith):
        uu_radx, uu_rady, in_out = {}, {}, {}
        uu_rad = self.H.data['uu_rad'][:, 0, :, ith, :]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, ith, :]
        th2D  = (ones * self.th)[ 0, :, ith, :]
        phi2D = (ones * self.phi)[0, :, ith, :]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        rho0 = self.H.params['density_scale']
        c    = self.c
        # Calculate uu in Cartesian coordinates
        uu_rad_cart = Coords.int_to_cart_vec(uu_rad) / (rho0 * c * c * c)
        in_out['r'] = np.sign(uu_rad[3])
        
        uu_radx['r'] = uu_rad_cart[1]
        uu_rady['r'] = uu_rad_cart[2]
        return uu_radx, uu_rady, in_out

    def calc_BetaIz(self, ith):
        B     = self.H.data['B'][:,       0, :, ith, :]
        uu    = self.H.data['uu'][:,      0, :, ith, :]
        u_gas = self.H.data['u_gas'][     0, :, ith, :]
        Te    = self.H.data['Te'][        0, :, ith, :]
        Ti    = self.H.data['Ti'][        0, :, ith, :]
        gcov  = self.H.grid['gcov'][:, :, 0, :, ith, :]

        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, ith, :]
        th2D  = (ones * self.th)[ 0, :, ith, :]
        phi2D = (ones * self.phi)[0, :, ith, :]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate b_z^2 
        b     = self.Field.calc_b(B, uu, gcov) 
        bz2   = Coords.dot(b, b, gcov, 'cylindrical', component=3)
        pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        BetaIz= (bz2 / (8. * np.pi)) / pgas 
        return BetaIz

    def calc_vR_vKep(self, ith):
        uu    = self.H.data['uu'][:,      0, :, ith, :]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, ith, :]
        th2D  = (ones * self.th)[ 0, :, ith, :]
        phi2D = (ones * self.phi)[0, :, ith, :]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate v in cylindridcal coordinates
        uu_cyl = Coords.int_to_cyl_vec(uu) 
        vR     = self.c * uu_cyl[1] / uu_cyl[0]
        
        vKep   = np.sqrt(self.G * self.MBH * self.Msun / (self.Rg * self.r))
        return vR / vKep[0, :, 0, :]

class Read_Profiles():
    def __init__(self, prof_dir, **kwargs):
        fnums    = list(range(2291, 2300))
        # quantities calclated via volume average
        keys   = [
            'rho',  'Ti',   'Te',
            'Pmag', 'Pthm', 'Prad',
            'Eion', 'Eelc', 'vR', 'vphi', 'vz'
        ]

        self.profiles = {}
        for fnum in fnums:
            profile_file = (prof_dir + 'Profiles_%d.npy' %int(fnum))
            self.profiles['%s' %str(fnum)] = np.load(
                profile_file, allow_pickle=True).ravel()[0]
            # dx1 = dr / r and r is dimensionless (r / Rg)
            self.dx1 = self.profiles['%s' %str(fnum)]['dx1']
            self.r   = self.profiles['%s' %str(fnum)]['r']
            for key in keys:
                self.profiles['%s' %str(fnum)]['%s' %key] = (
                    self.profiles['%s' %str(fnum)]['%s' %key] / (
                    self.profiles['%s' %str(fnum)]['Vol'] ))
        return None

    def get_pressure(self):
        return self.time_avg('Pthm', self.profiles)

    def time_avg(self, quant, profiles):
        keys  = profiles.keys()
        q_avg = np.zeros(len(profiles[next(iter(profiles))]['r']))
        for key in keys:
            q_avg += profiles['%s' %key]['%s' %quant] / len(keys)
        return q_avg

    # Interpolate q(r_old) to q_new(r_new)
    def interp(self, q, r_old, r_new):
        spl = CubicSpline(r_old, q)
        return spl(r_new)
