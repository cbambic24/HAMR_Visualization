import numpy as np
from numba import jit
from numba import prange
from scipy.special import kn
import tracemalloc

class Accretion():
    def __init__(self, grid, data, pars, CONSTANTS, **kwargs):
        self.dx1, self.dx2, self.dx3 = pars['dx1'], pars['dx2'], pars['dx3']
        self.gdet                    = grid['gdet']
        self.r, self.th, self.phi    = grid['r'], grid['theta'], grid['phi']
        self.grid                    = grid
        self.data                    = data
        self.cgs                     = kwargs.get('cgs', False)
        self.MBH                     = kwargs.get('MBH', 1.0)
        if (self.cgs):
            self.rho0 = pars['density_scale'] # g/cm^3 
            self.c    = CONSTANTS['c']        # cm/s
            self.G    = CONSTANTS['G']        # cm^3/g s^2
            self.Msun = CONSTANTS['Msun']     # g
            self.B0   = np.sqrt(4 * np.pi * self.rho0) * self.c # G
        else:
            self.rho0 = 1.0
            self.c    = 1.0
            self.G    = 1.0
            self.Msun = 1.0
            self.B0   = 1.0
            self.MBH  = 1.0

        self.Rg        = (self.G * self.MBH * self.Msun) / (self.c * self.c) 
        self.pars      = pars
        self.CONSTANTS = CONSTANTS
        self.kwargs    = kwargs

    # Calculate disk mass
    def calc_Mtot(self, **kwargs):
        rho, uu = self.data['rho'], self.data['uu']
        Mtot =  np.sum((rho * uu[0]) * self.sigma_cut(**kwargs) * (
                        self.gdet * self.dx1 * self.dx2 * self.dx3))
        return self.upscale64(Mtot, self.Rg * self.Rg * self.Rg)

    # Calculate inward mass accretion rate as function of radius
    def calc_Mdot(self, **kwargs):
        rho, uu = self.data['rho'], self.data['uu']
        Mdot = (-rho * uu[1] * self.sigma_cut(**kwargs) * (
            self.temp_cut(**kwargs) * (
            self.gdet * self.dx2 * self.dx3))).sum(-1).sum(-1)
        return self.upscale64(Mdot, self.Rg * self.Rg)
    
    # Calculate unsigned radial magnetic flux (Phi_BH) as a function of radius
    # Factor of sqrt(4 pi) accounts for conversion to Gaussian units in 
    # Tchekhovskoy et al. (2011)
    def calc_PhiBH(self, **kwargs):
        B = self.data['B']
        norm = np.sqrt(4. * np.pi) if not self.cgs else 1.
        return (0.5 * norm * np.abs(B[1]) * (self.Rg * self.Rg) * (
            self.gdet * self.dx2 * self.dx3)).sum(-1).sum(-1)
    
    # Calculate Phi_sph (Equation 1 in Begelman, Scepi & Dexter 2022)
    def calc_PhiSph(self, **kwargs):
        B    = self.data['B']
        norm = np.sqrt(4. * np.pi) if not self.cgs else 1.
        if (kwargs.get('hemisphere', 'upper') == 'upper'):
            norm = (self.th < np.pi/2.) * norm
            return (norm * B[1] * (self.Rg * self.Rg) * (
                self.gdet * self.dx2 * self.dx3)).sum(-1).sum(-1)
        else:
            norm = (self.th > np.pi/2.) * norm
            return -(norm * self.B[1] * (self.Rg * self.Rg) * (
                self.gdet * self.dx2 * self.dx3)).sum(-1).sum(-1)
        
    # Calculate Phi_mid (Equation 2 in Begelman, Scepi & Dexter 2022)
    def calc_PhiMid(self, **kwargs):
        B    = self.data['B']
        norm = np.sqrt(4. * np.pi) if not self.cgs else 1.
        ith0 = int(len(self.th[0, 0, :, 0])/2 - 1)
        return (0.5 * norm * (
            (B[2] * self.gdet)[:, :, ith0, :] + 
            (B[2] * self.gdet)[:, :, ith0+1, :]) * (
            (self.Rg * self.Rg) * self.dx3)).sum(-1)

    # Calculate Phi_max (Equation 8 in Liska et al. 2019)
    def calc_PhiMax(self, **kwargs):
        B    = self.data['B']
        norm = np.sqrt(4. * np.pi) if not self.cgs else 1.
        len_2 = len(self.th[0, 0, :, 0]) / 2.
        if (kwargs.get('hemisphere', 'upper') == 'upper'):
            norm = ((self.th < np.pi/2.) * norm * len_2)
            return (norm * np.amax(
                np.abs(B[1] * self.gdet), axis=2, keepdims=True) * (
                (self.Rg * self.Rg) * self.dx2 * self.dx3)).sum(-1).sum(-1)
        else:
            norm = ((self.th > np.pi/2.) * norm * len_2)
            return -(norm * np.amax(
                np.abs(B[1] * self.gdet), axis=2, keepdims=True) * (
                (self.Rg * self.Rg) * self.dx2 * self.dx3)).sum(-1).sum(-1)
    
    # Calculate energy accretion rate as a function of radius
    # Note that T^{mu}_{nu} > 0 implies inward accretion.
    def calc_Edot(self, **kwargs):
        rho0c2 = self.rho0 * self.c * self.c
        if 'TMHD' not in kwargs:
            stress_energy_MHD = self.calc_TMHD(upper_lower=True, 
                index = [1, 0], **kwargs)
        else:
            stress_energy_MHD = kwargs.get('TMHD')
        stress_energy_MHD *= (1. / rho0c2)
        Edot = (stress_energy_MHD * self.sigma_cut(**kwargs) * (
            self.gdet * self.dx2 * self.dx3)).sum(-1).sum(-1)
        stress_energy_MHD *= rho0c2
        return self.upscale64(Edot, self.Rg * self.Rg * self.c * rho0c2)

    # Calculate angular momentum accretion rate as a function of radius
    def calc_Ldot(self, **kwargs):
        rho0c2 = self.rho0 * self.c * self.c
        if 'TMHD' not in kwargs:
            stress_energy_MHD = self.calc_TMHD(upper_lower=True, 
                index = [1, 3], **kwargs)
        else:
            stress_energy_MHD = kwargs.get('TMHD')
        stress_energy_MHD *= (1. / rho0c2)
        Ldot = (stress_energy_MHD * self.sigma_cut(**kwargs) * (
            self.gdet * self.dx2 * self.dx3)).sum(-1).sum(-1)
        stress_energy_MHD *= rho0c2
        return self.upscale64(Ldot, self.Rg * self.Rg * rho0c2)

    # Calculate full MHD (+ Radiation) stress-energy tensor
    def calc_TMHD(self, **kwargs):
        Flu = Fluid(self.pars, self.CONSTANTS, cgs=self.kwargs.get('cgs'))
        Fld = Field(self.pars, self.CONSTANTS, cgs=self.kwargs.get('cgs'))
        if kwargs.get('radiation') and self.kwargs.get('radiation'):
            import radiation
            Rad = radiation.Radiation(self.pars, self.CONSTANTS, 
                                      cgs=self.kwargs.get('cgs'))
        elif kwargs.get('radiation') and not self.kwargs.get('radiation'):
            raise ValueError("""Cannot use radiation in calculation of 
                              stress-energy tensor without initializing
                              Accretion class with radiation=True.""")
        if 'pgas' in kwargs:
            pgas = kwargs.get('pgas')
        else:
            gamma_adi = 5./3.
            pgas = (gamma_adi - 1.) * self.u_gas
        stress_energy_HD = Flu.calc_THD(**self.data, **self.grid, pgas=pgas, 
                                        EOS=self.kwargs.get('EOS'), **kwargs)
        stress_energy_EM = Fld.calc_TEM(**self.data, **self.grid, **kwargs)
        if kwargs.get('radiation'):
            stress_energy_RAD = Rad.calc_TRAD(**self.data, **self.grid, **kwargs)
            return stress_energy_HD + stress_energy_EM + stress_energy_RAD
        else:
            return stress_energy_HD + stress_energy_EM
    
    # Return where sigma < sigma_cut if jet = False
    # Return where sigma > sigma_cut if jet = True
    def sigma_cut(self, **kwargs):
        if 'sigma_cut' in kwargs and 'sigma' not in kwargs:
            sigma = self.calc_sigma(**kwargs)
            return ((sigma < kwargs.get('sigma_cut')) 
                    if not kwargs.get('jet', False)
                    else (sigma > kwargs.get('sigma_cut')))
        elif 'sigma_cut' in kwargs and 'sigma' in kwargs:
            sigma = kwargs.get('sigma')
            return ((sigma < kwargs.get('sigma_cut')) 
                    if not kwargs.get('jet', False)
                    else (sigma > kwargs.get('sigma_cut')))
        else:
            return 1.0
    
    # Calculate sigma parameter: b^2 / 4 pi rho c^2 (b^2/rho if cgs=False)
    def calc_sigma(self, **kwargs):
        B02 = self.B0 * self.B0
        Fld = Field(self.pars, self.CONSTANTS, cgs=self.kwargs.get('cgs'))
        b2  = Fld.calc_b2(b=self.data['B'], **self.data, 
                          **self.grid, from_B=True)
        sigma = (b2 / B02) / (self.data['rho'] / self.rho0)
        return sigma
    
    # Return where T < Temp_cut if cold = True (default)
    # Return where T > Temp_cut if cold = False
    def temp_cut(self, **kwargs):
        cold=kwargs.get('cold', True) if not kwargs.get('hot', False) else True
        T = kwargs.get('Temp', 0.) if 'Temp' in kwargs else self.data['Ti']
        if 'Temp_cut' in kwargs:
            T_cut = kwargs.get('Temp_cut')
            return (T < T_cut) if cold else (T > T_cut)
        else:
            return 1.0

    # If optically_thin = True, return where tau < tau_cut
    def tau_cut(self, **kwargs):
        if ('sigma_cut' not in kwargs and 'tau' not in kwargs and 
            'tau_cut' in kwargs):
            import warnings
            warnings.warn("""Warning: integrating optical depth without
                          a cut on sigma...""")
        if 'tau_cut' in kwargs and 'tau' not in kwargs:
            tau = self.calc_tau(**kwargs)
            if (kwargs.get('optically_thin', True)):
                return (tau < kwargs.get('tau_cut'))
            else:
                return (tau >= kwargs.get('tau_cut'))
        elif 'tau_cut' in kwargs and 'tau' in kwargs:
            tau = kwargs.get('tau')
            if (kwargs.get('optically_thin', True)):
                return (tau < kwargs.get('tau_cut'))
            else:
                return (tau >= kwargs.get('tau_cut'))
        else:
            return 1.0

    # Calculate optical depth if required for tau_cut  
    # Returns optical depth to electron scattering 
    # if opacities = ['electron_scattering']
    # otherwise, returns effective absorption+scattering opacity
    def calc_tau(self, **kwargs):
        c     = self.c
        kappa = self.calc_kappa(**kwargs)
        
        if 'sigma_cut' in kwargs and 'sigma' not in kwargs:
            # Save memory and computation time by using b^2
            if 'synchrotron' in kwargs.get('opacities'):
                sigma = self.b2 / (4. * np.pi * self.data['rho'] * c * c)
            else:
                sigma = self.calc_sigma()
            kwargs['sigma'] = sigma
        elif 'sigma_cut' in kwargs and 'sigma' in kwargs:
            sigma = kwargs.get('sigma')

        return self.integrate_kappa(kappa, **kwargs)

    # Returns opacity used for optical depth integration 
    # if opacities = ['electron_scattering']
    # otherwise, returns sqrt(kappa_a * (kappa_a + kappa_es))
    def calc_kappa(self, **kwargs):
        if self.kwargs.get('radiation'):
            import radiation
            Rad = radiation.Radiation(self.pars, self.CONSTANTS, 
                                      cgs=self.kwargs.get('cgs'))     
        else:
            raise ValueError("""Cannot calculate optical depth tau 
                             without initializing Accretion class with 
                             radiation=True.""")
        if 'opacities' in kwargs:
            opacities = kwargs.get('opacities')
        else:
            raise ValueError("""Cannot calculate tau without opacities 
                             dictionary passed as kwarg.""")
        
        opacity_args = {
            'pars'      : self.pars,
            'CONSTANTS' : self.CONSTANTS,
            'XRB'       : True
        }
        
        if 'synchrotron' in opacities:
            # b2 is required for synchrotron opacities
            if 'b2' not in kwargs:
                Fld = Field(self.pars, self.CONSTANTS, 
                            cgs=self.kwargs.get('cgs'))
                self.b2 = Fld.calc_b2(b=self.data['B'], **self.data, 
                                      **self.grid, from_B=True)
            else:
                self.b2 = kwargs.get('b2')
            opacity_args['b2'] = self.b2

        if not kwargs.get('RIAF', False):
            Opac = radiation.Opacity(opacities, **self.data, **opacity_args)
        else: 
            Opac = radiation.RIAF_Opacity(opacities, **self.data, **opacity_args)

        if (len(Opac.opacities) == 1 and 'electron_scattering' in opacities):
            kappa = Opac.opacities['electron_scattering']
        else:
            kappa_abs = Opac.opacities['absorption_total']
            kappa_es  = Opac.opacities['electron_scattering']
            kappa     = np.sqrt(kappa_abs * (kappa_abs + kappa_es))
        return kappa

    # Determine how to integrate kappa ds (s = z, r, theta)
    def integrate_kappa(self, kappa, **kwargs):
        if kwargs.get('integral', 'theta') == 'z':
            return self.integrate_in_z(kappa, **kwargs)
        elif kwargs.get('integral', 'theta') == 'r':
            return self.integrate_in_r(kappa, **kwargs)
        else:
            return self.integrate_in_theta(kappa, **kwargs)

    def integrate_in_z(self, kappa, **kwargs):
        import coordinates
        Rg = self.Rg
        Coords = coordinates.Coordinates(self.r / Rg, self.th, self.phi, 
                                         self.pars, self.CONSTANTS)            
        gcov_cyl = Coords.int_to_cyl_cotnsr(self.grid['gcov'])

        # Initialize Integrals class
        r   = self.r   * np.ones_like(self.data['rho'])
        th  = self.th  * np.ones_like(self.data['rho'])
        phi = self.phi * np.ones_like(self.data['rho'])

        Integrals = coordinates.Integrals(r / self.Rg, th, phi, 
                                          self.pars, self.CONSTANTS)
        
        tau = self.Rg * Integrals.cum_integrate_z(
            (kappa * self.data['rho']), gcov_cyl[3, 3], **kwargs) 

        return tau
    
    def integrate_in_r(self, kappa, **kwargs):
        import coordinates
        Integrals = coordinates.Integrals(self.r / self.Rg, self.th, self.phi, 
                                          self.pars, self.CONSTANTS)
        return self.Rg * Integrals.cum_integrate_r(kappa * self.data['rho'], 
            self.grid['gcov'][1, 1], **kwargs)
    
    def integrate_in_theta(self, kappa, **kwargs):
        import coordinates
        Integrals = coordinates.Integrals(self.r / self.Rg, self.th, self.phi, 
                                          self.pars, self.CONSTANTS)
        return self.Rg * Integrals.cum_integrate_th(
            kappa * self.data['rho'], self.grid['gcov'][2, 2], **kwargs)
    
    # Upscale q to 64-bit precision and apply units
    def upscale64(self, q, units):
        q_dbl  = np.float64(q.copy())
        q_dbl *= units
        return q_dbl

class AccretionBK(Accretion):
    def __init__(self, amr, amr_info, grid, data, pars, CONSTANTS, **kwargs):
        self.n_ord       = amr_info['n_ord']
        self.levelx1     = amr_info['levelx1']
        self.levelx2     = amr_info['levelx2']
        self.levelx3     = amr_info['levelx3']
        self.max_levelx1 = amr['max_levelx1']
        self.max_levelx2 = amr['max_levelx2']
        self.max_levelx3 = amr['max_levelx3']
        self.amr         = amr
        self.amr_info    = amr_info
        self.pars        = pars

        self.calc_dx()

        self.params = self.pars.copy()
        self.params['dx1'] = self.dx1[:, None, None, None]
        self.params['dx2'] = self.dx2[:, None, None, None]
        self.params['dx3'] = self.dx3[:, None, None, None]

        super().__init__(grid, data, self.params, CONSTANTS, **kwargs)

    def calc_Mtot(self, **kwargs):
        Mtot = super().calc_Mtot(**kwargs)
        return self.fold(Mtot)

    def calc_Mdot(self, **kwargs):
        Mdot = super().calc_Mdot(**kwargs)
        return self.fold(Mdot)
    
    def calc_PhiBHBK(self, **kwargs):
        PhiBH = super().calc_PhiBH(**kwargs)
        return self.fold(PhiBH)
    
    def calc_Edot(self, **kwargs):
        Edot = super().calc_Edot(**kwargs)
        return self.fold(Edot)

    def calc_LdotBK(self, **kwargs):
        Ldot = super().calc_Ldot(**kwargs)
        return self.fold(Ldot)

    def calc_dx(self):
        n = np.arange(0, self.pars['n_active_total'])
        self.upscalex1 = 2**(self.max_levelx1 - self.levelx1[self.n_ord[n]])
        self.upscalex2 = 2**(self.max_levelx2 - self.levelx2[self.n_ord[n]])
        self.upscalex3 = 2**(self.max_levelx3 - self.levelx3[self.n_ord[n]])

        self.dx1 = self.pars['dx1'] * self.upscalex1
        self.dx2 = self.pars['dx2'] * self.upscalex2
        self.dx3 = self.pars['dx3'] * self.upscalex3
        return None
    
    def fold(self, q):
        return q

class Field():
    def __init__(self, pars, CONSTANTS, **kwargs):
        self.cgs = kwargs.get('cgs', False)
        if (self.cgs):
            self.rho0 = pars['density_scale'] # g/cm^3 
            self.c    = CONSTANTS['c']        # cm/s
            self.B0   = np.sqrt(4 * np.pi * self.rho0) * self.c
        else:
            self.rho0 = 1.0
            self.c    = 1.0
            self.B0   = 1.0

        self.dx1, self.dx2, self.dx3 = pars['dx1'], pars['dx2'], pars['dx3']

    # Compute b from B
    # Assumes B has units and returns b with units
    def calc_b(self, B, uu, gcov, **kwargs):
        c, B0  = self.c, self.B0
        b      = np.zeros(B.shape, dtype=(np.float32 
            if 'dtype' not in kwargs else kwargs.get('dtype')))
        # de-dimensionalize inputs
        uu    *=  1. / c
        B     *=  1. / B0
        # compute b^t
        b[0]   = Tools.dot(uu, B, gcov, **kwargs)
        # compute b^i
        for i in range(1, 4):
            b[i] = (B[i] + (uu[i] * b[0])) / uu[0]
        # re-dimensionalize inputs and b
        uu *= c
        B  *= B0
        return b * B0
    
    # Compute b^2 
    # Assumes B and uu have units
    def calc_b2(self, b, B, gcov, **kwargs):
        if (kwargs.get('from_B')):
            if 'uu' in kwargs:
                b = self.calc_b(B=B, gcov=gcov, **kwargs)
            else: 
                raise TypeError("Must specify uu to use from_B.")
        return Tools.dot(b, b, gcov, **kwargs)
        
    # Compute the electromagnetic stress-energy tensor
    # Default TEM^{mu, nu}
    # "upper_lower": TEM^{mu}_{nu} 
    # "lower_lower": TEM_{mu, nu}
    # If b and/or b2 arguments provided, assume they have units
    def calc_TEM(self, uu, B, gcov, gcon, **kwargs):
        rho0   = self.rho0
        c      = self.c
        rho0c2 = rho0 * c * c
        B0     = self.B0
        B02    = self.B0 * self.B0
        # de-dimensionalize uu
        uu    *= 1. / c
        if 'uu_tnsr' not in kwargs:
            uu_tnsr = Tools.construct_dyadic(uu, gcov, **kwargs)
        elif 'uu_tnsr' in kwargs:
            uu_tnsr = kwargs.get('uu_tnsr')
        if 'b' in kwargs and 'b2' in kwargs:
            b       = kwargs.get('b')
            b      *= 1. / B0
            bb_tnsr = Tools.construct_dyadic(b, gcov, **kwargs)
            b      *= B0
            b2      = kwargs.get('b2') 
        elif 'b' in kwargs and 'b2' not in kwargs:
            b       = kwargs.get('b')
            b      *= 1. / B0
            bb_tnsr = Tools.construct_dyadic(b, gcov, **kwargs)
            b2      = self.calc_b2(b, b, gcov) * B02
            b      *= B0
        elif 'b' not in kwargs and 'b2' in kwargs:
            uu     *= c   
            b       = self.calc_b(B, uu, gcov)
            uu     *= 1. / c   
            b      *= 1. / B0   
            bb_tnsr = Tools.construct_dyadic(b, gcov, **kwargs)
            b      *= B0
            b2      = kwargs.get('b2')
        else:
            uu     *= c   
            b       = self.calc_b(B, uu, gcov)
            uu     *= 1. / c
            b      *= 1. / B0
            bb_tnsr = Tools.construct_dyadic(b, gcov, **kwargs)
            b2      = self.calc_b2(b, b, gcov) * B02
            b      *= B0
        # re-dimensionalize uu
        uu *= c
        
        g_tnsr = Tools.g_metric(gcov, gcon, **kwargs)
        return ( (b2 / B02) * uu_tnsr + 
            0.5 *(b2 / B02) * g_tnsr - bb_tnsr) * rho0c2
    
    # Compute the electromagnetic field-strength tensor
    # Referred to as the Faraday tensor in MTW
    # Default FEM^{mu, nu}
    # "upper_lower": FEM^{mu}_{nu} 
    # "lower_lower": FEM_{mu, nu}
    # Assumes uu and B have units
    # If frame = generic, assumes B is measured in frame uu,
    # where uu = n would be frame of normal observer, etc.
    # If frame = generic, electric field must be supplied as kwarg.
    # Similarly, if non-ideal MHD, electric field must be supplied.
    def calc_FEM(self, uu, B, gcov, gdet, **kwargs):
        if (kwargs.get('frame', 'fluid') == 'fluid'):
            if 'b' not in kwargs:
                b = self.calc_b(B, uu, gcov, **kwargs)
            elif 'b' in kwargs:
                b = kwargs.get('b')
            b  *= 1. / self.B0
            uu *= 1. / self.c
            uu_cov = np.einsum('ij...,j...->i...', gcov, uu)
            b_cov  = np.einsum('ij...,j...->i...', gcov, b)
            levi_civita = Tools.calc_Levi_Civita(4, gdet)
            FEM = -np.einsum(
                'ijkl...,k...,l...->ij...', levi_civita, uu_cov, b_cov)
            uu *= self.c
        elif (kwargs.get('frame', 'fluid') == 'generic'):
            if 'E' not in kwargs:
                raise ValueError("Must supply E kwarg to use generic frame.")
            E  *= 1. / np.sqrt(self.rho0) * self.c
            B  *= 1. / self.B0
            uu *= 1. / self.c
            uu_cov = np.einsum('ij...,j...->i...', gcov, uu)
            b_cov  = np.einsum('ij...,j...->i...', gcov, b)
            levi_civita = Tools.calc_Levi_Civita(4, gdet)
            FEM = (np.einsum('i...,j...->ij...', uu, E) + 
                   np.einsum('j...,i...->ij...', uu, E) -
                   np.einsum('ijkl...,k...,l...->ij...', 
                    levi_civita, uu_cov, b_cov))
            uu *= self.c
            B  *= self.B0
            E  *= np.sqrt(self.rho0) * self.c
        if ('upper_lower' not in kwargs and 'lower_lower' not in kwargs):
            return FEM * (self.c * self.B0)
        elif ('upper_lower' in kwargs and 'lower_lower' not in kwargs):
            return (self.c * self.B0) * np.einsum(
                'ij...,jk...->ik...', gcov, FEM) 
        elif ('upper_lower' not in kwargs and 'lower_lower' in kwargs):
            return (self.c * self.B0) * np.einsum(
                'ij...,kl...,jl...->ik...', gcov, gcov, FEM)
        else:
            raise ValueError(
                "Please select a valid frame for the Faraday tensor. " + 
                "Options are frame=fluid or frame=generic."
            )
    
    # Compute the dual of the electromagnetic field-strength tensor
    # Referred to as the Maxwell tensor in MTW
    # Default *FEM^{mu, nu}
    # "upper_lower": *FEM^{mu}_{nu} 
    # "lower_lower": *FEM_{mu, nu}
    # Assumes uu and B have units
    # If frame = generic, assumes B is measured in frame uu,
    # where uu = n would be frame of normal observer, etc.
    # If frame = generic, electric field must be supplied as kwarg.
    # Similarly, if non-ideal MHD, electric field must be supplied.
    def calc_dual_FEM(self, uu, B, gcov, gdet, **kwargs):
        if (kwargs.get('frame', 'fluid') == 'fluid'):
            if 'b' not in kwargs:
                b = self.calc_b(B, uu, gcov, **kwargs)
            elif 'b' in kwargs:
                b = kwargs.get('b')
            b  *= 1. / self.B0
            uu *= 1. / self.c
            dual_FEM = (np.einsum('i...,j...->ij...', b, uu) - 
                        np.einsum('j...,i...->ij...', b, uu))
            uu *= self.c
            if ('upper_lower' not in kwargs and 'lower_lower' not in kwargs):
                return dual_FEM * (self.c * self.B0)
            elif ('upper_lower' in kwargs and 'lower_lower' not in kwargs):
                return (self.c * self.B0) * np.einsum(
                    'ij...,jk...->ik...', gcov, dual_FEM)
            elif ('upper_lower' not in kwargs and 'lower_lower' in kwargs):
                return (self.c * self.B0) * np.einsum(
                    'ij...,kl...,jl...->ik...', gcov, gcov, dual_FEM)
        elif (kwargs.get('frame', 'fluid') == 'generic'):
            pass
        else:
            raise ValueError(
                "Please select a valid frame for the Maxwell tensor. " + 
                "Options are frame=fluid or frame=generic."
            )
        
    # Calculate the 4-current by taking the divergence of the 
    # field-strength (Faraday) tensor
    def calc_J(self, uu, B, gcov, gdet, **kwargs):
        faraday = self.calc_FEM(uu, B, gcov, gdet, **kwargs)
        return Tools.div_tnsr(faraday, gdet, self.dx1, self.dx2, self.dx3)
    
    # Calculate J^2 = J_mu J^mu
    # If from_FEM = True, calculate J first
    def calc_J2(self, J, gcov, **kwargs):
        if kwargs.get('from_FEM', False):
            if ('uu' in kwargs and 'B' in kwargs and 'gdet' in kwargs):
                J = self.calc_J(gcov=gcov, **kwargs)
            else:
                raise ValueError("Must specify uu, B, and gdet " + 
                                 "to use from_FEM=True.")
        return Tools.dot(J, J, gcov, **kwargs)

    # Relativistic Alfvén velocity (vA)
    def calc_vA(self, rho, uu, B, u_gas, gcov, **kwargs):
        rho0   = self.rho0
        c      = self.c
        rho0c2 = rho0 * c * c
        B0     = self.B0
        B02    = self.B0 * self.B0
        if 'b' not in kwargs and 'b2' not in kwargs:
            b  = self.calc_b(B, uu, gcov, **kwargs)
            b2 = self.calc_b2(b, b, gcov)
        elif 'b' in kwargs and 'b2' not in kwargs:
            b  = kwargs.get('b')
            b2 = self.calc_b2(b, b, gcov)
        elif 'b' not in kwargs and 'b2' in kwargs:
            b  = self.calc_b(B, uu, gcov, **kwargs)
            b2 = kwargs.get('b2')
        elif 'b' in kwargs and 'b2' in kwargs:
            b  = kwargs.get('b')
            b2 = kwargs.get('b2')
        
        if 'pgas' not in kwargs:
            gamma_adi = 5./3.
            pgas = (gamma_adi - 1.) * u_gas
        else:
            pgas = kwargs.get('pgas')

        u_rad = kwargs.get('u_rad', 0.0) 

        return c * (b / B0) / np.sqrt(
            (rho / rho0)  +  (u_gas / rho0c2)  + (pgas / rho0c2) + 
            (b2 / B02) + (4./3.) * (u_rad / rho0c2)
        )

class Fluid():
    def __init__(self, pars, CONSTANTS, **kwargs):
        self.cgs = kwargs.get('cgs', False)
        if (self.cgs):
            self.rho0 = pars['density_scale'] # g/cm^3 
            self.c    = CONSTANTS['c']        # cm/s
            self.kB   = CONSTANTS['kB']       # erg/K
            # Equation 11 in Sadowski+ (2017)
            self.mu_i = 4.0 / (
                    4.0 * CONSTANTS['X_AB'] + CONSTANTS['Y_AB'])
            # Equation 10 in Sadowski+ (2017)
            self.mu_e = 2.0 / (1.0 + CONSTANTS['X_AB'])
            self.mp   = CONSTANTS['mp']     # g
            self.mi   = ((CONSTANTS['X_AB'] + 
                          CONSTANTS['Y_AB']) * CONSTANTS['mp']) # g 
            self.me   = CONSTANTS['me']     # g
            self.sigT = CONSTANTS['sigmaT'] # cm^2
        else:
            self.rho0 = 1.0
            self.c    = 1.0
            self.kB   = 1.0
            self.mu_i = 1.0
            self.mu_e = 1.0
            self.mp   = 1.0
            self.mi   = 1.0
            self.me   = 1.0
            self.sigT = 1.0

        self.Theta_min = 2.e-2
        self.kwargs    = kwargs
        self.CONSTANTS = CONSTANTS
        self.params    = pars

    # Compute the hydrodynamic stress-energy tensor
    # Default THD^{mu, nu} 
    # "upper_lower": THD^{mu}_{nu} 
    # "lower_lower": THD_{mu, nu}
    def calc_THD(self, rho, uu, u_gas, pgas, gcov, gcon, **kwargs):
        if (kwargs.get('EOS', False) == '2T'):
            if 'Ti' in kwargs and 'Te' in kwargs:
                Te   = kwargs.get('Te')
                Ti   = kwargs.get('Ti')
                pgas = self.calc_pgas(u_gas, Te, Ti)
            else:
                raise ValueError("Must supply Te and Ti to use EOS='2T'.")
        c      = self.c
        rho0   = self.rho0
        rho0c2 = rho0 * c * c

        if 'uu_tnsr' not in kwargs:
            uu     *= 1. / c
            uu_tnsr = Tools.construct_dyadic(uu, gcov, **kwargs)
            uu     *= c
        elif 'uu_tnsr' in kwargs:
            uu_tnsr = kwargs.get('uu_tnsr')
        g_tnsr  = Tools.g_metric(gcov, gcon, **kwargs)
        return (
            (((rho / rho0) + (u_gas / rho0c2) + (pgas / rho0c2)) * uu_tnsr + 
            (pgas / rho0c2) * g_tnsr) * rho0c2
        )

    # Compute pgas from u_gas using gamma law EOS
    # Equation 23 in Sadowski+ (2017)
    def calc_pgas(self, u_gas, Te, Ti, **kwargs):
        Theta_e = (self.kB * Te) / (self.me * self.c * self.c)
        Theta_i = (self.kB * Ti) / (self.mi * self.c * self.c)

        # Calculate gamma (ratio of specific heats)
        # Equation 28 in Sadowski+ (2017)
        gam_e = (10. + 20. * Theta_e) / (6. + 15. * Theta_e)
        gam_i = (10. + 20. * Theta_i) / (6. + 15. * Theta_i)
        
        gam_g = 1. + ((
                (gam_e - 1.) * (gam_i - 1.) * (
                (self.mu_i / self.mu_e) + (Ti / Te)) ) / (
                (Ti / Te) * (gam_e - 1.) + 
                (self.mu_i / self.mu_e) * (gam_i - 1.) ))
        return (gam_g - 1.) * u_gas # erg/cm^3

    # Compute ion-electron equilibration timescale teq
    # Assumes rho in g/cm^3, Te and Ti in K, u_gas in erg/cm^3
    def calc_teq(self, rho, Te, Ti, **kwargs):
        Theta_e = (self.kB * Te) / (self.me * self.c * self.c)
        Theta_i = (self.kB * Ti) / (self.mi * self.c * self.c)

        # Calculate gamma (ratio of specific heats)
        # Equation 28 in Sadowski+ (2017)
        gam_e = (10. + 20. * Theta_e) / (6. + 15. * Theta_e)
        gam_i = (10. + 20. * Theta_i) / (6. + 15. * Theta_i)

        u_i = (1./(gam_i-1.)) * (rho / (self.mu_i * self.mp)) * self.kB * Ti
        u_e = (1./(gam_e-1.)) * (rho / (self.mu_e * self.mp)) * self.kB * Te

        # u_i > u_e whenever Ti > Te for narrow range of 
        # Te and Ti around Te < Ti < 2 Te and Te \approx 1e9 K, i.e., when
        # electrons are relativistic and ions are non-relativistic.
        # In that case, use ion gamma and mu for energy difference.
        delta_energy = u_i - u_e
        delta_energy[(delta_energy < 0) & (Ti > Te)] = ((1./(gam_i-1.)) * (
            rho / (self.mu_i * self.mp)) * self.kB * (Ti- Te))[
            (delta_energy < 0) & (Ti > Te)]

        if 'qie' not in kwargs:
            qie = self.calc_qie(rho, Te, Ti, **kwargs)
        else:
            qie = kwargs.get('qie')
        
        teq = delta_energy / qie
        teq[qie == 0.] = 0.

        # If temperature difference < 2% of Te, then assume negligible teq
        teq[np.abs(Ti - Te) < (self.Theta_min * Te)] = 0.0
        if kwargs.get('cooling'):
            teq[Ti < Te] = 0.0
            return teq # s
        elif kwargs.get('heating'):
            teq[Ti > Te] = 0.0
            return teq # s
        else:
            return teq # s

    # Compute cooling/heating rate due to ion-electron collisions
    # Assumes rho in g/cm^3, Te and Ti in K
    # Equation 20 in Sadowski+ (2017)
    # see also Stepney & Guilbert (1983) Equation 15
    def calc_qie(self, rho, Te, Ti, **kwargs):
        if kwargs.get('cpp', False):
            return self.calc_qie_cpp(rho, Te, Ti, **kwargs)
        Theta_e = (self.kB * Te) / (self.me * self.c * self.c)
        Theta_i = (self.kB * Ti) / (self.mi * self.c * self.c)
        ne      = rho / (self.mu_e * self.mp)
        X_ab, Y_ab = self.CONSTANTS['X_AB'], self.CONSTANTS['Y_AB'] 
        n_avg   = ((X_ab + Y_ab) * rho) / self.mp
        cksigmaT= self.c * self.kB * self.sigT
        me_mi   = self.me / self.mi
        # Note that ne * n_avg will cause overflow errors in the disk
        qie = (
            (3./2.) * me_mi * cksigmaT * ne * (Ti - Te) * n_avg * ( 
            self.Coulomb_log(ne, Te) * (
            self.Bessel_terms(Theta_e, Theta_i)))
        )
        # If temperature difference < 2% of Te, then assume negligible qie
        qie[np.abs(Ti - Te) < (self.Theta_min * Te)] = 0.0
        if kwargs.get('cooling'):
            qie[Ti < Te] = 0.0
            return qie # erg/cm^3 s
        elif kwargs.get('heating'):
            qie[Ti > Te] = 0.0
            return qie # erg/cm^3 s
        else:    
            return qie # erg/cm^3 s
    
    # Compute ion-ion momentum-exchange (collision) timescale tii
    # Assumes rho in g/cm^3, Te and Ti in K, u_gas in erg/cm^3
    def calc_tii(self, rho, Te, Ti, **kwargs):
        ne = rho / (self.mu_e * self.mp)
        ni = rho / (self.mu_i * self.mp)
        return (Ti / Te) ** (3./2.) * np.sqrt(self.me / self.mi) * (
            (self.Coulomb_log(ne, Te) / self.Coulomb_log(ni, Ti)) * (
            self.calc_teq(rho, Te, Ti, **kwargs)))
    
    # Compute electron-electron momentum-exchange (collision) timescale tee
    # Assumes rho in g/cm^3, Te and Ti in K, u_gas in erg/cm^3
    def calc_tee(self, rho, Te, Ti, **kwargs):
        return (self.me / self.mi) * self.calc_teq(rho, Te, Ti, **kwargs)

    # H-AMR uses approximations for the Bessel functions when the
    # electrons (E) or ions (I) are cold (Theta < Theta_min)
    def Bessel_terms(self, Theta_e, Theta_i):
        coldE = (Theta_e <  self.Theta_min)
        coldI = (Theta_i <  self.Theta_min)
        hotE  = (Theta_e >= self.Theta_min)
        hotI  = (Theta_i >= self.Theta_min)
    
        coldE_coldI = coldE * coldI
        hotE_coldI  = hotE  * coldI
        coldE_hotI  = coldE * hotI
        hotE_hotI   = hotE  * hotI
            
        return (
            np.where(coldE_coldI, self.coldE_coldI(Theta_e, Theta_i), 0.) +
            np.where(hotE_coldI,  self.hotE_coldI( Theta_e, Theta_i), 0.) +
            np.where(coldE_hotI,  self.coldE_hotI( Theta_e, Theta_i), 0.) +
            np.where(hotE_hotI,   self.hotE_hotI(  Theta_e, Theta_i), 0.)
        )
            
    def coldE_coldI(self, Theta_e, Theta_i):
        Theta_sum = Theta_e + Theta_i
        return ((2. * (Theta_sum ** 2) + 2. * Theta_sum + 1.) / (
            np.sqrt(0.5 * np.pi * (Theta_sum ** 3))))

    def hotE_coldI(self, Theta_e, Theta_i):
        Theta_sum  = Theta_e + Theta_i 
        k2e        = kn(2, (1. / Theta_e))
        return (((2. * (Theta_sum ** 2) + 2. * Theta_sum + 1.) * (
            np.sqrt(Theta_e))) / ( k2e * np.exp(1. / Theta_e) * (
            np.sqrt(Theta_sum ** 3))))

    def coldE_hotI(self, Theta_e, Theta_i):
        Theta_sum = Theta_e + Theta_i 
        k2i       = kn(2, (1. / Theta_i))
        return (((2. * (Theta_sum ** 2) + 2. * Theta_sum + 1.) * (
            np.sqrt(Theta_i))) / ( k2i * np.exp(1. / Theta_i) * (
            np.sqrt(Theta_sum ** 3))))

    def hotE_hotI(self, Theta_e, Theta_i):
        return (self.k2k2_qie(Theta_e, Theta_i) * (
            self.k1_qie(Theta_e, Theta_i) + self.k0_qie(Theta_e, Theta_i)))

    def k2k2_qie(self, Theta_e, Theta_i):
        return 1. / (kn(2, (1. / Theta_i)) * kn(2, (1. / Theta_e)))

    def k1_qie(self, Theta_e, Theta_i):
        Theta_m = 1. / ((1. / Theta_e) + (1. / Theta_i))
        return (
                ((2. * np.square(Theta_e + Theta_i) + 1.) / 
                (Theta_e + Theta_i)) * kn(1, (1. / Theta_m))
        )

    def k0_qie(self, Theta_e, Theta_i):
        Theta_m = 1. / ((1. / Theta_e) + (1. / Theta_i))
        return 2. * kn(0, (1. / Theta_m))

    def Coulomb_log(self, ne, Te):
        return 35.4 + np.log((Te / 1.e7) * np.sqrt(1.0e-3 / ne))
    
    def calc_qie_cpp(self, rho, Te, Ti, **kwargs):
        if (not self.kwargs.get('cgs')):
            import warnings
            warnings.warn("""Warning: calculating Coulomb energy exchange
                          rate without CGS units applied...""")
        import hamr_readc
        Fluidc = hamr_readc.PyFluid(self.params, self.CONSTANTS)
        dims   = list(rho.shape)
        dtype  = np.float64 # C++ layer expects double
        if (len(dims) != 4):
            raise ValueError("This array with shape " + 
                             str(dims) + " cannot be passed to C++.")
        [nb, nx1, nx2, nx3] = dims
        qie = np.zeros(rho.shape, dtype=dtype)
        Fluidc.calc_qie(qie, rho, Te, Ti, nb, nx1, nx2, nx3)
        # If temperature difference < 2% of Te, then assume negligible qie
        qie[np.abs(Ti - Te) < (self.Theta_min * Te)] = 0.0
        if kwargs.get('cooling'):
            qie[Ti < Te] = 0.0
            return qie # erg/cm^3 s
        elif kwargs.get('heating'):
            qie[Ti > Te] = 0.0
            return qie # erg/cm^3 s
        else:    
            return qie # erg/cm^3 s

    def heating_Qi_Qe(self, rho, u_gas, **kwargs):
        print("Implement the electron-to-ion heating ratio from Rowan+2017.")

class Tools():
    def dot(a, b, gcov, **kwargs):
        # contract two 4-vectors (g_ij a^j b^i)
        if (kwargs.get('cpp', False)):
            return Tools.dot_cpp(a, b, gcov, **kwargs)
        a_cov = np.einsum('ij...,j...->i...', gcov, a)
        return np.einsum('i...,i...->...', a_cov, b)

    def construct_dyadic(u, gcov, **kwargs):
        if kwargs.get('cpp', False):
            return Tools.construct_dyadic_cpp(u, gcov, **kwargs)
        if 'index' in kwargs:
            [mu, nu] = kwargs.get('index')
            if ('upper_lower' not in kwargs and 'lower_lower' not in kwargs):
                uu_tnsr = u[mu] * u[nu]
            elif ('upper_lower' in kwargs):
                uu_tnsr = u[mu] * np.einsum('ij...,j...->i...', gcov, u)[nu]
            elif ('lower_lower' in kwargs):
                uu_tnsr = ( 
                    np.einsum('ij...,j...->i...', gcov, u)[mu] * (
                    np.einsum('ij...,j...->i...', gcov, u)[nu]))
        else:
            if ('upper_lower' not in kwargs and 'lower_lower' not in kwargs):
                uu_tnsr = np.einsum('i...,j...->ij...', u, u)
            elif ('upper_lower' in kwargs):
                uu_tnsr = np.einsum('i...,j...->ij...', u, 
                            np.einsum('ij...,j...->i...', gcov, u))
            elif ('lower_lower' in kwargs):
                uu_tnsr = np.einsum('i...,j...->ij...', 
                        np.einsum('ij...,j...->i...', gcov, u), 
                        np.einsum('ij...,j...->i...', gcov, u))
        return uu_tnsr
            
    def g_metric(gcov, gcon, **kwargs):
        if 'index' in kwargs:
            [mu, nu] = kwargs.get('index')
            if ('upper_lower' not in kwargs and 'lower_lower' not in kwargs):
                g_tnsr = gcon[mu, nu]
            elif ('upper_lower' in kwargs):
                g_tnsr = 1.0 * (mu == nu)
            elif ('lower_lower' in kwargs):
                g_tnsr = gcov[mu, nu]
        else:
            if ('upper_lower' not in kwargs and 'lower_lower' not in kwargs):
                g_tnsr = gcon
            elif ('upper_lower' in kwargs):
                g_tnsr = Tools.get_identity(gcov, **kwargs)
            elif ('lower_lower' in kwargs):
                g_tnsr = gcov
        return g_tnsr

    def get_identity(gcov, **kwargs):
        shape     = list(gcov.shape)
        id_matrix = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        for mu in range(4):
            for nu in range(4):
                id_matrix[mu, nu] = (mu == nu) 
        return id_matrix
    
    @staticmethod
    def antisymmetric_symbol3(i, j, k):
        if not all(n in (i, j, k) for n in range(3)):
            return 0
        if ((i, j, k) == (0, 1, 2) or
            (i, j, k) == (1, 2, 0) or 
            (i, j, k) == (2, 0, 1)):
            return 1
        return -1
    
    @staticmethod
    def antisymmetric_symbol4(i, j, k, l):
        if not all(n in (i, j, k, l) for n in range(4)):
            return 0
        if ((i, j, k, l) == (0, 1, 2, 3) or (i, j, k, l) == (0, 2, 3, 1) or
            (i, j, k, l) == (0, 3, 1, 2) or (i, j, k, l) == (1, 0, 3, 2) or
            (i, j, k, l) == (1, 3, 2, 0) or (i, j, k, l) == (1, 2, 0, 3) or
            (i, j, k, l) == (2, 0, 1, 3) or (i, j, k, l) == (2, 3, 0, 1) or
            (i, j, k, l) == (2, 1, 3, 0) or (i, j, k, l) == (3, 1, 0, 2) or
            (i, j, k, l) == (3, 0, 2, 1) or (i, j, k, l) == (3, 2, 1, 0)):
            return 1
        return -1
    
    def calc_Levi_Civita(n, gdet, **kwargs):
        if (n == 3):
            shape   = [3, 3, 3] + list(gdet.shape)
            epsilon = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        epsilon[i, j, k] = Tools.antisymmetric_symbol3(i, j,k)
        elif (n == 4):
            shape   = [4, 4, 4, 4] + list(gdet.shape)
            epsilon = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        for l in range(4):
                            epsilon[i, j, k, l] = (
                                Tools.antisymmetric_symbol4(i, j, k, l))
        return (1. / gdet) * epsilon
        
    def div_tnsr(tnsr, gdet, dx1, dx2, dx3, **kwargs):
        return (1. / gdet) * (
            np.gradient((gdet * tnsr[:, 1]), axis=2) / (2. * dx1) + 
            np.gradient((gdet * tnsr[:, 2]), axis=3) / (2. * dx2) + 
            np.gradient((gdet * tnsr[:, 3]), axis=4) / (2. * dx3)
        )
    
    def dot_cpp(a, b, gcov, **kwargs):
        import hamr_readc
        Toolsc = hamr_readc.PyTools()
        dims   = list(a.shape)
        dtype  = np.float32 # C++ layer expects float 
        if (len(dims) != 5 or dims[0] != 4):
            raise ValueError("This 4-vector with shape " +
                             str(dims) + " cannot be passed to C++.")
        [ndim, nb, nx1, nx2, nx3] = dims
        axisym = 1 if (gcov.shape[-1] == 1) else 0
        a_cov = np.zeros(a.shape, dtype=dtype)
        Toolsc.lower_index(np.float32(a), a_cov, gcov, 
            nb, nx1, nx2, nx3, axisym)
        return np.einsum('i...,i...->...', a_cov, b)

    def contract_vector_cpp(a, gcov, **kwargs):
        import hamr_readc
        Toolsc = hamr_readc.PyTools()
        dims   = list(a.shape)
        dtype  = np.float32 # C++ layer expects float 
        if (len(dims) != 5 or dims[0] != 4):
            raise ValueError("This 4-vector with shape " +
                             str(dims) + " cannot be passed to C++.")
        [ndim, nb, nx1, nx2, nx3] = dims
        axisym = 1 if (gcov.shape[-1] == 1) else 0 
        a_cov = np.zeros(a.shape, dtype=dtype)
        Toolsc.lower_index(np.float32(a), a_cov, gcov, 
            nb, nx1, nx2, nx3, axisym)
        return a_cov

    def construct_dyadic_cpp(u, gcov, **kwargs):
        import hamr_readc
        Toolsc = hamr_readc.PyTools()
        dims   = list(u.shape)
        dtype  = np.float32 # C++ layer expects float 
        if (len(dims) != 5 or dims[0] != 4):
            raise ValueError("This 4-vector with shape " + 
                             str(dims) + " cannot be passed to C++.")
        [ndim, nb, nx1, nx2, nx3] = dims
        axisym = 1 if (gcov.shape[-1] == 1) else 0
        if 'index' in kwargs:
            [mu, nu] = kwargs.get('index')
            if ('upper_lower' not in kwargs and 'lower_lower' not in kwargs):
                uu_tnsr = u[mu] * u[nu]
            elif ('upper_lower' in kwargs):
                u_nu    = np.zeros(u[nu].shape, dtype=dtype)
                Toolsc.lower_index_mu(np.float32(u), u_nu, gcov, nu, 
                                      nb, nx1, nx2, nx3, axisym)
                uu_tnsr = u[mu] * u_nu
            elif ('lower_lower' in kwargs):
                u_mu = np.zeros(u[mu].shape, dtype=dtype)
                u_nu = np.zeros(u[nu].shape, dtype=dtype)
                Toolsc.lower_index_mu(np.float32(u), u_mu, gcov, mu, 
                                      nb, nx1, nx2, nx3, axisym)
                Toolsc.lower_index_mu(np.float32(u), u_nu, gcov, nu, 
                                      nb, nx1, nx2, nx3, axisym)
                uu_tnsr = u_mu * u_nu
        else:
            if ('upper_lower' not in kwargs and 'lower_lower' not in kwargs):
                uu_tnsr = np.einsum('i...,j...->ij...', u, u, optimize=True)
            elif ('upper_lower' in kwargs):
                u_nu   = np.zeros(u.shape, dtype=dtype)
                Toolsc.lower_index(np.float32(u), u_nu, gcov, 
                                   nb, nx1, nx2, nx3, axisym)
                uu_tnsr = np.einsum(
                    'i...,j...->ij...', u, u_nu, optimize=True)
            elif ('lower_lower' in kwargs):
                u_mu = np.zeros(u.shape, dtype=dtype)
                u_nu = np.zeros(u.shape, dtype=dtype)
                Toolsc.lower_index(np.float32(u), u_mu, gcov, 
                                   nb, nx1, nx2, nx3, axisym)
                Toolsc.lower_index(np.float32(u), u_nu, gcov, 
                                   nb, nx1, nx2, nx3, axisym)
                uu_tnsr = np.einsum(
                    'i...,j...->ij...', u_mu, u_nu, optimize=True)
        return uu_tnsr

    # Smooth out function using convolution
    def smooth(y, Lsmth, Lx, **kwargs): 
        Nsmth = Lx/Lsmth
        window = int(len(y)/Nsmth)
        avg_mask = np.ones(window)/window
        if (kwargs.get('edge') == 'periodic'):
            # periodic edges
            y = np.concatenate((y[-window:], y, y[:window]))
        elif (kwargs.get('edge') == 'constant'):
            # constant edges
            y = np.concatenate((y[0]*np.ones(window), y, y[-1]*np.ones(window)))
        else:
            # reflective edges
            y = np.concatenate((y[:window][::-1], y, y[-window:][::-1]))
        yout = np.convolve(y, avg_mask, 'same')[window:-window]
        return yout

if __name__ == "__main__":
    print("Executing analysis.")


'''
    def integrate_th(self, q, gcov, **kwargs):
        shape = list(q.shape)
        if (len(shape) == 2): axis = -1
        else: axis = 2
        qdth = np.zeros(q.shape)
        qdth[self.theta < np.pi/2.] 
        np.cumsum((q * gcov[2, 2] * self.dx2 * (
            self.sigma_cut(**kwargs))[self.theta < np.pi/2.]), axis=axis)
        qdth[self.theta > np.pi/2.] np.cumsum((q * gcov[2, 2] * self.dx2 * (
            self.sigma_cut(**kwargs))[self.theta > np.pi/2.][:, ::-1]), axis=axis)
        return qdth


                theta_max = np.asin(Rcyl_max[ir, ith] / r[-1, 0])
                r_min     = Rcyl_min[ir, ith]
                if (np.any(r[:, 0] < r_min)):
                    ir_min  = np.where(r[:, 0] < r_min)[0][-1]
                else:
                    ir_min = 0
                ir_max    = len(r[:, 0])
                if (z[ir, ith] > 0):
                    it_min    = 0
                    it_max    = np.where(th[0, :] > theta_max)[0][0]
                else:
                    it_min = len(th[0, :]) / 2
                    it_max = np.where(th[0, :] > theta_max)[0][0] + it_min
                temp_sum  = 0.0
                for jr in range(ir_min, ir_max):
                    for jth in range(it_min, it_max):
                        temp_sum += (q * dz)[jr, jth]
                int_qdz[ir, ith] = temp_sum
                
                #in_R_range = (
                #    (R_cyl < Rcyl_max[ir, ith]) &
                #    (R_cyl > Rcyl_min[ir, ith]))
                #if (z[ir, ith] >= 0.):
                #    in_z_range = (z > z[ir, ith])
                #elif (z[ir, ith] < 0.):
                #    in_z_range = (z < z[ir, ith])
                #int_qdz[ir, ith] = np.sum(
                #    q * dz * in_R_range * in_z_range)


# Calculate precession period
def calc_PrecPeriod(angle_tilt):
    gam,a,precperiod
    uu_proj = project_vector(uu)
    L = rho * r * uu_proj[3]
    tilt=np.zeros((nb,bs1new,1,1),dtype=np.float32)
    tilt[0,:,0,0]=(np.nan_to_num(angle_tilt)+0.1)/360.0*2.0*np.pi
    Z1 = 1.0 + (1.0 - a ** 2.0) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    Z2 = np.sqrt(3.0 * a ** 2.0 + Z1 ** 2.0)
    r_isco = (3.0 + Z2 - np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)))
    L_tot = np.nan_to_num(L * gdet * _dx1 * _dx2 * _dx3 * np.sin(tilt)* (r > r_isco) * (r < 150)).sum(-1).sum(-1).sum(-1)
    vnod=1.0/(r**1.5+a)*(1.0-np.sqrt(1.0-4.0*a/r**1.5+3.0*a*a/r**2))
    tau_tot = np.nan_to_num(L * vnod * gdet * _dx1 * _dx2 * _dx3 * np.sin(tilt)* (r > r_isco) * (r < 150)).sum(-1).sum(-1).sum(-1)
    precperiod = 2 * np.pi * L_tot / tau_tot

# Calculate mass accretion rate as function of radius
def calc_Mdot():
    global Mdot
    Mdot = (-gdet * rho * uu[1] * _dx2 * _dx3).sum(-1).sum(-1)

def calc_profiles(cutoff):
    global pgas_avg, rho_avg, pb_avg, Q_avg1_1,Q_avg1_2,Q_avg1_3, Q_avg2_1,Q_avg2_2,Q_avg2_3
    calc_Q()
    norm1 = (gdet * rho * (rho > cutoff)).sum(-1).sum(-1)
    norm2 = (gdet * rho * (rho > cutoff)).sum(-1).sum(-1)
    norm3 = (gdet * np.sqrt(np.abs(rho * bsq)) * (rho > cutoff)).sum(-1).sum(-1)
    fact1 = gdet * rho * (rho > cutoff)
    fact2 = gdet * np.sqrt(np.abs(rho * bsq)) * (rho > cutoff)
    pgas_avg = (fact1 * (gam - 1.0) * ug).sum(-1).sum(-1) / norm1
    pb_avg = (fact1 * bsq / 2.0).sum(-1).sum(-1) / norm1
    rho_avg = (gdet * (rho > cutoff) * rho * rho)[:, :, :, :].sum(-1).sum(-1) / (gdet * rho* (rho > cutoff)).sum(-1).sum(-1)
    Q_avg1_1 = (fact1 * Q[1]).sum(-1).sum(-1) / norm2
    Q_avg1_2 = (fact1 * Q[2]).sum(-1).sum(-1) / norm2
    Q_avg1_3 = (fact1 * Q[3]).sum(-1).sum(-1) / norm2
    Q_avg2_1 = (fact2 * Q[1]).sum(-1).sum(-1) / norm3
    Q_avg2_2 = (fact2 * Q[2]).sum(-1).sum(-1) / norm3
    Q_avg2_3 = (fact2 * Q[3]).sum(-1).sum(-1) / norm3

def calc_profiles_M1(scaleheight, cutoff=1.0e-5):
    global rho, bsq, Te, Ti, TE, TI, kappa_es, kappa_abs, kappa_emmit, kappa_sy2, density_scale, gcov, _dx1, _dx2, _dx3
    global Mdot, M_EDD_RAT, bs1new, bs2new, bs3new, gcov, gcon, gdet, R_G_CGS
    global tau_es_tot, tau_em_tot, tau_abs_tot, cool_frac, rho_filtered, emmission_tot, emmission_hard, Mdot_rad, Rdot_rad
    global prad_tot, pe_tot, pi_tot, rho_tot, pb_tot
    global t_diff, t_emmit_disk, t_abs_disk, t_compton_disk
    global t_emmit_corona, t_abs_corona, t_compton_corona
    global tau_es

    # Calculate vertically and azimuthally integrated optical depths; exclude poles
    tau_es_int = ((kappa_es) * rho * density_scale * R_G_CGS * np.sqrt(gcov[2, 2]) * _dx2)[0, :, 10:bs2new - 10, :].sum(1)
    tau_abs_int = ((kappa_abs) * rho * density_scale * R_G_CGS * np.sqrt(gcov[2, 2]) * _dx2)[0, :, 10:bs2new - 10, :].sum(1)
    tau_em_int = ((kappa_emmit) * rho * density_scale * R_G_CGS * np.sqrt(gcov[2, 2]) * _dx2)[0, :, 10:bs2new - 10, :].sum(1)

    # Set the filter for the cool thin disk
    rho_filtered = rho * (Ti > 10 ** 8.0) * (Te > 10 ** 8.0)
    cold_filtered = rho * (Ti < 10 ** 8.0) * (Te < 10 ** 8.0)

    # Calculate Mdot normalize to eddington ratio
    Mdot_rad = (Mdot * (M_EDD_RAT / Mdot[0, 5]))[0]

    #Calculate Rdot in code units
    ud_rad = gcov[:, 0] * uu_rad[0] + gcov[:, 1] * uu_rad[1] + gcov[:, 2] * uu_rad[2] + gcov[:, 3] * uu_rad[3]
    Rdot_rad =((4.0/3.0)*E_rad*uu_rad[1]*ud_rad[0]*gdet*_dx2*_dx3).sum(-1).sum(-1)[0]

    # Calculate filtered rho
    rho_max = np.zeros((bs1new, bs3new), dtype=np.float64)
    for i in range(0, bs1new):
        for z in range(0, bs3new):
            rho_max[i, z] = np.max(cold_filtered[0, i, :, z]) * (tau_es_int[i, z] > 1.0)

    # Allocate arrays
    rho_frac = np.zeros((bs1new), dtype=np.float64)
    tau_es_tot = np.zeros((bs1new), dtype=np.float64)
    tau_abs_tot = np.zeros((bs1new), dtype=np.float64)
    tau_em_tot = np.zeros((bs1new), dtype=np.float64)
    cool_frac = np.zeros((bs1new), dtype=np.float64)

    # Compute hard fraction of emmission
    for i in range(0, bs1new):
        for z in range(0, bs3new):
            if (rho_max[i, z] > 0.0):
                cool_frac[i] = cool_frac[i] + 1.0 / bs3new
            tau_es_tot[i] = tau_es_tot[i] + tau_es_int[i, z]/bs3new
            tau_abs_tot[i] = tau_abs_tot[i] + tau_abs_int[i, z]/bs3new
            tau_em_tot[i] = tau_em_tot[i] + tau_em_int[i, z]/bs3new
    # emmission_tot=((4.0/3.0)*E_rad*uu_rad[1]*ud_rad[0]*_dx2*_dx3*gdet).sum(-1).sum(-1)
    emmission_tot = ((kappa_emmit) * rho * R_G_CGS * density_scale * 2.2714380625235033e+21 * (TE ** 4) * _dx1 * _dx2 * _dx3 * gdet).sum(-1).sum(-1).cumsum(axis=1)[0]
    emmission_hard = ((kappa_emmit) * rho_filtered * R_G_CGS * density_scale * 2.2714380625235033e+21 * (TE ** 4) * _dx1 * _dx2 * _dx3 * gdet).sum(-1).sum(-1).cumsum(axis=1)[0]

    # Set normalizations
    norm1 = (gdet * rho * (rho > cutoff) * _dx1 * _dx2 * _dx3).sum(-1).sum(-1)[0]
    fact1 = rho * (rho > cutoff)[0]

    # Calculate total radiation pressure as function of radius
    prad_tot = (E_rad * (4.0 / 3.0 - 1.0) * _dx2 * np.sqrt(gcov[2,2])).sum(-1).sum(-1)[0]

    # Calculate electron and ion pressures
    pe_tot = (TE * rho * _dx2 * np.sqrt(gcov[2,2])).sum(-1).sum(-1)[0]
    pi_tot = (TI * rho * _dx2 * np.sqrt(gcov[2,2])).sum(-1).sum(-1)[0]

    # Calculate gas mass in radial shell
    rho_tot = (rho * _dx2 * np.sqrt(gcov[2,2])).sum(-1).sum(-1)[0]

    # Calculate total magnetic pressure
    pb_tot = (bsq / 2.0 * _dx2 * np.sqrt(gcov[2,2])).sum(-1).sum(-1)[0]

    # Calculate optical depth integrated from pole
    tau_es = (kappa_es * rho * density_scale * R_G_CGS * _dx2 * np.sqrt(gcov[2, 2]))[:, :, ::-1, :].cumsum(axis=2)[:, :, ::-1, :]
    tau_es2 = (kappa_es * rho * density_scale * R_G_CGS * _dx2 * np.sqrt(gcov[2, 2])).cumsum(axis=2)
    tau_es[tau_es > tau_es2] = tau_es2[tau_es > tau_es2]


def calc_lum():
    global lum
    p = (gam - 1.0) * ug
    lum = np.sum((rho ** 3.0 * p ** (-2.0) * np.exp(-0.2 * (rho ** 2.0 / (np.sqrt(bsq) * p ** 2.0)) ** (1.0 / 3.0)) * (h > np.pi / 3.0) * (h < 2.0 / 3.0 * np.pi) * (r < 50.) * gdet * _dx1 * _dx2 * _dx3))

# Calculate energy accretion rate as function of radius
def calc_Edot():
    global Edot, Edotj, beta_gamma_sq, beta_gamma_cut, Mdotj
    temp=Tcalcud_new(1, 0)* gdet * _dx2 * _dx3
    beta_gamma_sq = np.power(Tcalcud_new(1, 0)/(rho*uu[1]),2)-1 # Equation (11) paper V EHT 2017
    Edot = (temp).sum(-1).sum(-1)
    Edotj = (temp*(bsq/rho>3)).sum(-1).sum(-1)
    Mdotj = (-gdet * rho * uu[1] *(beta_gamma_sq>beta_gamma_cut) * _dx2 * _dx3).sum(-1).sum(-1)
    
def calc_Ldot():
    global Ldot
    Ldot = (Tcalcud_new(1, 3)* gdet * _dx2 * _dx3).sum(-1).sum(-1)

# Calculate the Q resolution paramters and their average Q_avg in the disk
def calc_Q():
    global Q, Q_avg, lowres1, lowres2, lowres3
    Q = np.zeros((4, 1, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    Q_avg = np.zeros((4), dtype=np.float32, order='C')
    dx = np.zeros((4, nb, bs1new, bs2new, 1), dtype=np.float32, order='C')

    dx[1] = _dx1 / lowres1 * np.sqrt(gcov[1, 1, :, :, :, :])
    dx[2] = _dx2 / lowres2 * np.sqrt(gcov[2, 2, :, :, :, :])
    dx[3] = _dx3 / lowres3 * np.sqrt(gcov[3, 3, :, :, :, :])
    bu_proj = project_vector(bu)

    for dir in range(1, 4):
        alf_speed = np.sqrt(np.abs(bu_proj[dir] * bu_proj[dir]) / (rho + bsq + (gam) * ug))
        vrot = np.sqrt((uu[3] * uu[3] * gcov[3][3] + uu[2] * uu[2] * gcov[2][2] + uu[1] * uu[1] * gcov[1][1])) / uu[0]
        wavelength = 2 * 3.14 * alf_speed * r / vrot
        if (dir == 1):
            Q[dir] = wavelength / dx[dir]
        if (dir == 2):
            Q[dir] = wavelength / (dx[2]*theta_to_theta+dx[3]*np.abs(phi_to_theta))
        if (dir == 3):
            Q[dir] = wavelength / (dx[2]*np.abs(theta_to_phi) + dx[3] * phi_to_phi)
        Q[dir] = np.nan_to_num(Q[dir])
'''
