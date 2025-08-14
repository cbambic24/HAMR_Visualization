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
            self.gdet * self.dx2 * self.dx3)).sum(-1).sum(-1)
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
        b2  = Fld.calc_b2(b=self.data['B'], **data, **grid, from_B=True)
        sigma = (b2 / B02) / (self.rho / self.rho0)
        return sigma
    
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
            sigma = self.b2 / (4. * np.pi * self.rho * c * c)
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
        if 'opacities' not in kwargs:
            raise ValueError("""Cannot calculate tau without opacities 
                             dictionary passed as kwarg.""")
        else:
            opacities = kwargs.get('opacities')
        if 'synchrotron' in opacities:
            # b2 is required for synchrotron opacities
            if 'b2' not in kwargs:
                Fld = Field(self.pars, self.CONSTANTS, 
                            cgs=self.kwargs.get('cgs'))
                self.b2 = Fld.calc_b2(b=self.data['B'], **self.data, 
                                      **self.grid, from_B=True)
            else:
                self.b2 = kwargs.get('b2')
            Opac = radiation.Opacity(opacities, **self.data, pars=self.pars, 
                CONSTANTS=self.CONSTANTS, b2 = self.b2, XRB = True)
        else:
            Opac = radiation.Opacity(opacities, **self.data, pars=self.pars, 
                                     CONSTANTS=self.CONSTANTS, XRB = True)

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
        if not kwargs.get('interpolate', False):
            return self.integrate_in_z_direct(kappa, **kwargs)
        elif kwargs.get('interpolate', False):
            return self.integrate_in_z_interpolate(kappa, **kwargs)

    def integrate_in_z_direct(self, kappa, **kwargs):
        import coordinates
        Rg = self.Rg
        Coords = coordinates.Coordinates(self.r / Rg, self.th, self.phi, 
                                         self.pars, self.CONSTANTS)            
        gcov_cart = Coords.int_to_cart_cotnsr(self.grid['gcov'])
        if (len(list(self.data['rho'].shape)) > 2 and 
            self.data['rho'].shape[0] == 1):
            print("z integration without interpolation may be " + 
                  "very slow on 3D arrays. Consider using interpolate=True.")
            tau   = np.zeros(self.data['rho'].shape)
            if 'sigma' in kwargs:
                sigma_in_args = True
                sigma = kwargs.pop('sigma')
            else:
                if 'sigma_cut' in kwargs:
                    sigma = self.calc_sigma(**kwargs)
                    sigma_in_args = True
                else:
                    print("Integrating in z without a sigma cut " +
                          "can be dangerous for optical depths.")
                    sigma_in_args = False
            r   = self.r   * np.ones_like(self.data['rho'])
            th  = self.th  * np.ones_like(self.data['rho'])
            phi = self.phi * np.ones_like(self.data['rho'])
            for k in range(tau.shape[-1]):
                Integrals = coordinates.Integrals(r[0, :, :, k] / self.Rg, 
                    th[0, :, :, k], phi[0, :, :, k], self.pars, self.CONSTANTS)
                if (sigma_in_args):
                    kwargs['sigma'] = sigma[0, :, :, k]
                tau[0, :, :, k] = self.Rg * Integrals.cum_integrate_z(
                    (kappa * self.data['rho'])[0, :, :, k], 
                    gcov_cart[3, 3, 0, :, :, k], **kwargs)
        else:
            Integrals = coordinates.Integrals(self.r / self.Rg, 
                self.th, self.phi, self.pars, self.CONSTANTS)
            tau = self.Rg * Integrals.cum_integrate_z(
                kappa * self.data['rho'], gcov_cart[3, 3], **kwargs)
        return tau

    def integrate_in_z_interpolate(self, kappa, **kwargs):
        import coordinates
        c, Rg = self.c, self.Rg
        Coords = coordinates.Coordinates(self.r / Rg, self.th, self.phi, 
                                         self.pars, self.CONSTANTS)            
        gcov_cart = Coords.int_to_cart_cotnsr(self.gcov)
        Integrals = coordinates.Integrals(self.r / self.Rg, self.th, self.phi, 
                                          self.pars, self.CONSTANTS)
        pass

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
