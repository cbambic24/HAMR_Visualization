import numpy as np
from numba import jit
from numba import prange
from scipy.special import kn
import hamr_read as hr

# Numerical constants for performance
ONE       = np.float32(1.)
HALF      = np.float32(0.5)
ONE_THIRD = np.float32(1. / 3.)
FOURPI    = np.float32(4. * np.pi)
EIGHTPI   = np.float32(8. * np.pi)
PI_2      = np.float32(np.pi / 2.)

class Accretion():
    """
    Accretion class manages all mass, energy, and angular momentum 
    accretion and outflow rates. Designed to be aware of the global grid and 
    perform integrals in phi (ph_sum) and theta and phi (thph_sum). 
    Implements cuts on sigma, tau (optical depth), temperature, scaleheight,
    inflow/outflow (ur), and Bernoulli parameter. Various calculations of the
    magnetic flux are implemented here.

    Attributes
    ----------
    grid : dict from H-AMR read object
        Dictionary containing grid variables (r, theta, phi, gcov, gdet, etc.).
    data : dict from H-AMR read object
        Dictionary containing primitive variables (rho, uu, u_gas, B, etc.).
    pars : dict from H-AMR read object
        Parameters dictionary (dx1, dx2, density_scale, etc.).
    CONSTANTS : dict
        Constants dictionary (hr.CONSTANTS, usually).
    **kwargs : dict
        Optional configurations (e.g. cgs).
    """
    def __init__(self, grid, data, pars, CONSTANTS, **kwargs):
        self.dx1, self.dx2, self.dx3 = pars['dx1'], pars['dx2'], pars['dx3']
        self.gdet                    = grid['gdet']
        self.r, self.th, self.phi    = grid['r'], grid['theta'], grid['phi']
        self.grid                    = grid
        self.data                    = data
        self.cgs                     = kwargs.get('cgs', False)
        
        self.MBH  = kwargs.get('MBH', ONE) if self.cgs else ONE # g
        self.rho0 = pars['density_scale']  if self.cgs else ONE # g/cm^3
        self.c    = CONSTANTS['c']         if self.cgs else ONE # cm/s
        self.G    = CONSTANTS['G']         if self.cgs else ONE # cm^3/g s^2
        self.Msun = CONSTANTS['Msun']      if self.cgs else ONE # g
        B0        = np.sqrt(4 * np.pi * self.rho0) * self.c
        self.B0   = np.float32(B0)         if self.cgs else ONE # G

        self.Rg        = (self.G * self.MBH * self.Msun) / (self.c * self.c) 
        self.pars      = pars
        self.CONSTANTS = CONSTANTS
        self.kwargs    = kwargs

    def calc_Mtot(self, **kwargs):
        """Calculate disk mass."""
        rho, uu = self.data['rho'], self.data['uu']
        Mtot =  np.sum((rho * uu[0]) * self.sigma_cut(**kwargs) * (
                        self.gdet * self.dx1 * self.dx2 * self.dx3))
        return self.upscale64(Mtot, self.Rg * self.Rg * self.Rg)

    def calc_Mdot(self, **kwargs):
        """
        Calculate inward mass-accretion rate as function of radius.
        Inward: positive Mdot. Outward: negative Mdot.

        Parameters
        ----------
        **kwargs : dict
            Optional configurations. Accepted keys include:

            sigma_cut : float
                Mask all values with sigma > sigma_cut if jet=False (default).
                If jet=True, mask all values with sigma < sigma_cut.
                If sigma_mask is in kwargs, that mask will be used.
                If sigma is in kwargs, that sigma will be used to make mask.
                Otherwise, sigma will be recalculated.
            Temp_cut : float
                Mask all values with Temperature (Ti) < Temp_cut 
                if cold=True (default). If hot=True, then masks values
                with Temperature > Temp_cut. If 'Temp' is in kwargs,
                then Temperature=Temp. Otherwise, Temperature=data['Ti'].
            HoR_cut : float
                Mask all values where local disk thickness H/R > HoR_cut
                unless the "above" keyword is passed. Then, only keep
                data above HoR = 1.
            inward : bool
                Mask all values where u1 > 0.
            outward : bool
                Mask all values where u1 < 0.
            unbound : bool
                Mask all values where Bernoulli parameter (Be) < 0.
            bound : bool
                Mask all values where Bernoulli parameter (Be) > 0.
                If Be is provided, then the provided Be will be used
                for the Bernoulli paramter. Otherwise, Be will be 
                calculated.
            tau_cut : float
                Mask all values with optical depth tau > tau_cut if
                optically_thin=True (default). If optically_thin=False,
                then mask out values where tau < tau_cut. 
                If tau is in kwargs, that optical depth will be used.
                Otherwise, tau will be recalculated. If tau is not provided,
                then an opacities dict must be passed.
        
        Returns
        -------
        array of double (1D)
            Mdot (r).
        """
        rho, uu = self.data['rho'], self.data['uu']
        Mdot = (-rho * uu[1] * self.sigma_cut(**kwargs) * (
            self.temp_cut(**kwargs) * self.height_cut(**kwargs) * (
            self.in_or_out(**kwargs) * self.Bernoulli_cut(**kwargs) * (
            self.gdet * self.dx2 * self.dx3)))).sum(-1).sum(-1)
        return self.upscale64(Mdot, self.Rg * self.Rg)
    
    def calc_PhiBH(self, **kwargs):
        """
        Calculate unsigned radial magnetic flux (Phi_BH) as function of radius.
        Factor of sqrt(4 pi) accounts for conversion to Gaussian units in 
        Tchekhovskoy et al. (2011).
        """
        B = self.data['B']
        norm = np.sqrt(FOURPI) if not self.cgs else ONE
        return (0.5 * norm * np.abs(B[1]) * (self.Rg * self.Rg) * (
            self.gdet * self.dx2 * self.dx3)).sum(-1).sum(-1)
    
    def calc_PhiSph(self, **kwargs):
        """Calculate Phi_sph (Equation 1 in Begelman, Scepi & Dexter 2022)."""
        B    = self.data['B']
        norm = np.sqrt(FOURPI) if not self.cgs else ONE
        if (kwargs.get('hemisphere', 'upper') == 'upper'):
            norm = (self.th < PI_2) * norm
            return (norm * B[1] * (self.Rg * self.Rg) * (
                self.gdet * self.dx2 * self.dx3)).sum(-1).sum(-1)
        else:
            norm = (self.th > PI_2) * norm
            return -(norm * B[1] * (self.Rg * self.Rg) * (
                self.gdet * self.dx2 * self.dx3)).sum(-1).sum(-1)
        
    def calc_PhiMid(self, **kwargs):
        """Calculate Phi_mid (Equation 2 in Begelman, Scepi & Dexter 2022)."""
        B    = self.data['B']
        norm = np.sqrt(FOURPI) if not self.cgs else ONE
        ith0 = int(len(self.th[0, 0, :, 0])/2 - 1)
        return (0.5 * norm * (
            (B[2] * self.gdet)[:, :, ith0, :] + 
            (B[2] * self.gdet)[:, :, ith0+1, :]) * (
            (self.Rg * self.Rg) * self.dx3)).sum(-1)

    def calc_PhiMax(self, **kwargs):
        """Calculate Phi_max (Equation 8 in Liska et al. 2019)."""
        B    = self.data['B']
        norm = np.sqrt(FOURPI) if not self.cgs else ONE
        len_2 = len(self.th[0, 0, :, 0]) / 2.
        if (kwargs.get('hemisphere', 'upper') == 'upper'):
            norm = ((self.th < PI_2) * norm * len_2)
            return (norm * np.amax(
                np.abs(B[1] * self.gdet), axis=2, keepdims=True) * (
                (self.Rg * self.Rg) * self.dx2 * self.dx3)).sum(-1).sum(-1)
        else:
            norm = ((self.th > PI_2) * norm * len_2)
            return -(norm * np.amax(
                np.abs(B[1] * self.gdet), axis=2, keepdims=True) * (
                (self.Rg * self.Rg) * self.dx2 * self.dx3)).sum(-1).sum(-1)
    
    def calc_Edot(self, **kwargs):
        """
        Calculate energy-accretion rate as a function of radius.
        Note that T^{mu}_{nu} > 0 implies inward accretion. To calculate
        Edot from individual components of the stress-energy tensor (i.e.
        Fluid, Electromagnetic, or Radiation), pass the desired stress-energy
        tensor with kwarg TMHD=stress_energy_tensor.
        
        Parameters
        ----------
        **kwargs : dict
            Optional configurations. See 'calc_Mdot' for a full description 
            of accepted keys (e.g., sigma_cut, Temp_cut, in, out, bound, etc.).
        
        Returns
        -------
        array of double (1D)
            Edot (r).
        """
        rho0c2 = self.rho0 * self.c * self.c
        if 'TMHD' not in kwargs:
            stress_energy_MHD = self.calc_TMHD(upper_lower=True, index=[1,0], 
                                               **kwargs)
        else:
            stress_energy_MHD = kwargs.get('TMHD')
        stress_energy_MHD *= (ONE / rho0c2)
        Edot = (stress_energy_MHD * self.sigma_cut(**kwargs) * (
            self.temp_cut(**kwargs) * self.height_cut(**kwargs) * (
            self.in_or_out(**kwargs) * self.Bernoulli_cut(**kwargs) * (
            self.gdet * self.dx2 * self.dx3)))).sum(-1).sum(-1)
        stress_energy_MHD *= rho0c2
        return self.upscale64(Edot, self.Rg * self.Rg * self.c * rho0c2)

    def calc_Ldot(self, **kwargs):
        """
        Calculate angular momentum-accretion rate as a function of radius.
        
        Parameters
        ----------
        **kwargs : dict
            Optional configurations. See 'calc_Mdot' for a full description 
            of accepted keys (e.g., sigma_cut, Temp_cut, inward, outward, bound, etc.).
        
        Returns
        -------
        array of double (1D)
            Ldot (r).
        """
        rho0c2 = self.rho0 * self.c * self.c
        if 'TMHD' not in kwargs:
            stress_energy_MHD = self.calc_TMHD(upper_lower=True, index=[1,3], 
                                               **kwargs)
        else:
            stress_energy_MHD = kwargs.get('TMHD')
        stress_energy_MHD *= (ONE / rho0c2)
        Ldot = (stress_energy_MHD * self.sigma_cut(**kwargs) * (
            self.temp_cut(**kwargs) * self.height_cut(**kwargs) * (
            self.in_or_out(**kwargs) * self.Bernoulli_cut(**kwargs) * (
            self.gdet * self.dx2 * self.dx3)))).sum(-1).sum(-1)
        stress_energy_MHD *= rho0c2
        return self.upscale64(Ldot, self.Rg * self.Rg * rho0c2)

    def calc_TMHD(self, **kwargs):
        """
        Calculate full MHD (+ Radiation) stress-energy tensor.
        If pgas is supplied, that thermal pressure is used. Otherwise,
        pgas = (5/3 - 1) u_gas. If EOS='2T' is used, then pgas is calculated
        from Te and Ti.
        """
        cgs     = self.kwargs.get('cgs')
        use_rad = kwargs.get('radiation')
        if use_rad and not self.kwargs.get('radiation'):
            raise ValueError(
                "Cannot use radiation in calculation of stress-energy tensor "
                "without initializing Accretion class with radiation=True."
            )

        Flu = Fluid(self.pars, self.CONSTANTS, cgs=cgs)
        Fld = Field(self.pars, self.CONSTANTS, cgs=cgs)
        
        pgas = kwargs.get('pgas')
        if pgas is None:
            gamma_adi = kwargs.get('gamma_adi', np.float32(self.pars['gam']))
            pgas      = (gamma_adi - 1) * self.data['u_gas']
            
        stress_energy = (
            Flu.calc_THD(**self.data, **self.grid, 
                         pgas=pgas, EOS=self.kwargs.get('EOS'), **kwargs)
            + Fld.calc_TEM(**self.data, **self.grid, **kwargs)
        )
        
        if use_rad:
            import radiation
            Rad = radiation.Radiation(self.pars, self.CONSTANTS, cgs=cgs)
            stress_energy += Rad.calc_TRAD(**self.data, **self.grid, **kwargs)
            
        return stress_energy
    
    def sigma_cut(self, **kwargs):
        """
        Returns where sigma < sigma_cut if jet = False.
        Returns where sigma > sigma_cut if jet = True.
        """
        if 'sigma_mask' in kwargs:
            return kwargs.get('sigma_mask', ONE)
            
        if 'sigma_cut' not in kwargs:
            return ONE

        sigma = kwargs.get('sigma')
        if sigma is None:
            sigma = self.calc_sigma(**kwargs)

        sigma_cut_val = kwargs.get('sigma_cut')
        if kwargs.get('jet', False):
            return sigma > sigma_cut_val
        else:
            return sigma < sigma_cut_val
    
    def calc_sigma(self, **kwargs):
        """
        Calculate sigma parameter: 
        b^2 / 4 pi rho c^2 (b^2/rho if cgs=False).
        """
        B02 = self.B0 * self.B0
        Fld = Field(self.pars, self.CONSTANTS, cgs=self.kwargs.get('cgs'))
        b2  = Fld.calc_b2(b=self.data['B'], **self.data, 
                          **self.grid, from_B=True)
        sigma = (b2 / B02) / (self.data['rho'] / self.rho0)
        return sigma

    def temp_cut(self, **kwargs):
        """
        Return where T < Temp_cut if cold = True (default).
        Return where T > Temp_cut if cold = False.
        """
        if 'Temp_mask' in kwargs:
            return kwargs.get('Temp_mask', ONE)
        
        if 'Temp_cut' not in kwargs:
            return ONE

        cold = False if kwargs.get('hot', False) else kwargs.get('cold', True)

        T = kwargs.get('Temp')
        if T is None:
            T = self.data['Ti']

        T_cut = kwargs.get('Temp_cut')
        return (T < T_cut) if cold else (T > T_cut)

    def height_cut(self, **kwargs):
        """
        Return where |z| < H if above = False (default).
        Return where |z| > H if above = True.
        """
        if 'HoR_cut' not in kwargs:
            return ONE

        hor_cut = kwargs.get('HoR_cut')
        above   = kwargs.get('above', False)

        cos_th_abs = np.abs(np.cos(self.th))

        return (cos_th_abs > hor_cut) if above else (cos_th_abs < hor_cut)
        
    def in_or_out(self, **kwargs):
        """Select material either moving "in" or "out" based on u1."""
        inward  = kwargs.get('inward',  False)
        outward = kwargs.get('outward', False)
        
        if inward and outward:
            raise ValueError("Cannot specify both inward and outward.")
            
        if not inward and not outward:
            return ONE
            
        return (self.data['uu'][1] > 0) if outward else (self.data['uu'][1] < 0)

    def Bernoulli_cut(self, **kwargs):
        """
        Select material based on Bernoulli parameter Be.
        If Be is not provided as a kwarg, Be is recalculated.
        Note that this Be does not include radiation contributions.
        """
        unbound = kwargs.get('unbound', False)
        bound   = kwargs.get('bound',   False)
        
        # Enforce that exactly one flag must be True
        if bound == unbound:
            return ONE
            
        # Grab Be if provided, otherwise calculate it
        Be = kwargs.get('Be', self.calc_Bernoulli(**kwargs))
        
        return (Be < ONE) if bound else (Be > ONE)

    def calc_Bernoulli(self, **kwargs):
        """Calculate Bernoulli parameter without radiation contribution."""
        c      = self.c
        rho0c2 = self.rho0 * c * c 
        B02    = self.B0 * self.B0

        # Retrieve pgas or recalculate it if not in kwargs.
        pgas = kwargs.get('pgas')
        if pgas is None:
            gamma_adi = kwargs.get('gamma_adi', self.pars['gam'])
            pgas      = (gamma_adi - 1) * self.data['u_gas']
            
        # Retrieve b2 or recalculate it if not in kwargs.
        b2 = kwargs.get('b2')
        if b2 is None:
            Fld = Field(self.pars, self.CONSTANTS, cgs=self.kwargs.get('cgs'))
            b2  = Fld.calc_b2(b=self.data['B'], **self.data, 
                              **self.grid, from_B=True)

        # Calculate enthalpy
        w = self.data['rho']*c*c + pgas + self.data['u_gas'] + (b2/B02)*rho0c2
        
        # Calculate u_t = g_{t,mu} u^{mu}
        u_t = np.einsum('ij...,j...->i...', self.grid['gcov'], self.data['uu'])[0]

        # Return Bernoulli parameter
        return -w * (u_t / c) / (self.data['rho'] * c * c) - ONE

    def tau_cut(self, **kwargs):
        """
        If optically_thin = True, return where tau < tau_cut.
        Otherwise, return where tau > tau_cut. If tau is not provided,
        then tau is recalculated from opacities dictionary.
        """
        if 'tau_mask' in kwargs:
            return kwargs.get('tau_mask', ONE)

        if 'tau_cut' not in kwargs:
            return ONE
        
        if 'sigma_cut' not in kwargs and 'tau' not in kwargs:
            import warnings
            warnings.warn(
                "Warning: integrating optical depth without a cut on sigma..."
            )

        tau = kwargs.get('tau')
        if tau is None:
            tau = self.calc_tau(**kwargs)

        tau_cut_val = kwargs.get('tau_cut')
        if kwargs.get('optically_thin', True):
            return tau < tau_cut_val
        else:
            return tau >= tau_cut_val

    def calc_tau(self, **kwargs):
        """
        Calculate optical depth if required for tau_cut. 
        Returns optical depth to electron scattering 
        if opacities = ['electron_scattering'].
        Otherwise, returns effective (absorption + scattering) optical depth.
        """
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

    def calc_kappa(self, **kwargs):
        """
        Returns opacity used for optical depth integration 
        if opacities = ['electron_scattering']. 
        Otherwise, returns sqrt(kappa_a * (kappa_a + kappa_es)).
        """
        if not self.kwargs.get('radiation'):
            raise ValueError(
                "Cannot calculate optical depth tau without initializing "
                "Accretion class with radiation=True."
            )
            
        opacities = kwargs.get('opacities')
        if opacities is None:
            raise ValueError(
                "Cannot calculate tau without opacities dictionary passed "
                "as kwarg."
            )
        
        import radiation

        opacity_args = {
            'pars'      : self.pars,
            'CONSTANTS' : self.CONSTANTS,
            'XRB'       : True
        }
        
        if 'synchrotron' in opacities:
            # b2 is required for synchrotron opacities
            self.b2 = kwargs.get('b2')
            if self.b2 is None:
                Fld = Field(self.pars, self.CONSTANTS, 
                            cgs=self.kwargs.get('cgs'))
                self.b2 = Fld.calc_b2(b=self.data['B'], **self.data, 
                                      **self.grid, from_B=True)
            opacity_args['b2'] = self.b2

        RIAF         = kwargs.get('RIAF', False)
        OpacityClass = radiation.RIAF_Opacity if RIAF else radiation.Opacity
        Opac         = OpacityClass(opacities, **self.data, **opacity_args)

        if len(Opac.opacities) == 1 and 'electron_scattering' in opacities:
            return Opac.opacities['electron_scattering']

        kappa_abs = Opac.opacities['absorption_total']
        kappa_es  = Opac.opacities['electron_scattering']
        return np.sqrt(kappa_abs * (kappa_abs + kappa_es))

    def integrate_kappa(self, kappa, **kwargs):
        """Determine how to integrate kappa ds (s = z, r, theta)."""
        import coordinates
        Rg = self.Rg
        
        # Initialize Integrals class
        integral_args = {
            'r'         : (self.r / Rg) * np.ones_like(self.data['rho']),
            'theta'     : self.th       * np.ones_like(self.data['rho']),
            'phi'       : self.phi      * np.ones_like(self.data['rho']),
            'pars'      : self.pars,
            'CONSTANTS' : self.CONSTANTS 
        }

        self.Integrals = coordinates.Integrals(**integral_args)

        if kwargs.get('integral', 'theta') == 'z':
            return self.integrate_in_z(kappa, **kwargs)
        elif kwargs.get('integral', 'theta') == 'r':
            return self.integrate_in_r(kappa, **kwargs)
        else:
            return self.integrate_in_theta(kappa, **kwargs)

    def integrate_in_z(self, kappa, **kwargs):
        """Perform integral in z (requires gcov in cylindrical coordinates)."""
        import coordinates
        Rg = self.Rg
        coord_args = {
            'r'         : self.r / Rg,
            'theta'     : self.th,
            'phi'       : self.phi,
            'pars'      : self.pars,
            'CONSTANTS' : self.CONSTANTS
        }
        Coords = coordinates.Coordinates(**coord_args)
        gcov_cyl = Coords.int_to_cyl_cotnsr(self.grid['gcov'])
 
        tau = self.Rg * self.Integrals.cum_integrate_z(
            (kappa * self.data['rho']), gcov_cyl[3, 3], **kwargs) 

        return tau
    
    def integrate_in_r(self, kappa, **kwargs):
        """Perform integral in r (really, in x1 = log(r))."""
        return self.Rg * self.Integrals.cum_integrate_r(
            kappa * self.data['rho'], self.grid['gcov'][1, 1], **kwargs)
    
    def integrate_in_theta(self, kappa, **kwargs):
        """Perform integral in theta (really, in x2)."""
        return self.Rg * self.Integrals.cum_integrate_th(
            kappa * self.data['rho'], self.grid['gcov'][2, 2], **kwargs)

    def thph_sum(self, q, **kwargs):
        """Integrate 3D data in theta (th) and phi (ph)."""
        w          = kwargs.get('weight', ONE)
        sigma_mask = self.sigma_cut(**kwargs)
        tau_mask   = self.tau_cut(**kwargs)
        Temp_mask  = self.temp_cut(**kwargs)
        dx2        = np.float32(self.dx2)
        dx3        = np.float32(self.dx3)
        dtype      = kwargs.get('dtype', np.float32)
        return (q * w * sigma_mask * tau_mask * Temp_mask * (
            self.gdet * dx2 * dx3)).sum(-1, dtype=dtype).sum(-1, dtype=dtype)

    def ph_sum(self, q, **kwargs):
        """Integrate 3D data in phi (ph)."""
        w          = kwargs.get('weight', ONE)
        sigma_mask = self.sigma_cut(**kwargs)
        tau_mask   = self.tau_cut(**kwargs)
        Temp_mask  = self.temp_cut(**kwargs)
        dx3        = np.float32(self.dx3)
        dtype      = kwargs.get('dtype', np.float32)
        return (q * w * sigma_mask * tau_mask * Temp_mask * (
            self.gdet * dx3)).sum(-1, dtype=dtype)

    def upscale64(self, q, units):
        """Upscale q to 64-bit precision and apply units."""
        q_dbl = hr.Tools.zeroNd(q.shape, 1, dtype=np.float64)
        np.copyto(q_dbl, q, casting='safe')
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
        if kwargs.get('unfold', False):
            return self.unfold(super().calc_Mtot(**kwargs))
        else:
            return super().calc_Mtot(**kwargs)

    def calc_Mdot(self, **kwargs):
        if kwargs.get('unfold', False):
            return self.unfold(super().calc_Mdot(**kwargs))
        else:
            return super().calc_Mdot(**kwargs)
    
    def calc_PhiBHBK(self, **kwargs):
        PhiBH = super().calc_PhiBH(**kwargs)
        return self.fold(PhiBH)
    
    def calc_Edot(self, **kwargs):
        if kwargs.get('unfold', False):
            return self.unfold(super().calc_Edot(**kwargs))
        else:
            return super().calc_Edot(**kwargs)

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
    """
    Field class manages quantities derived from the magnetic field 3-vector B.
    The 4-magnetic field b^{mu}, the magnetic energy density b^2, Alfven 
    velocity vA, the electromagnetic (EM) field-strength tensor FEM^{mu nu}, 
    and the EM stress-energy tensor TEM^{mu nu} are handled by this class.

    Functions can be called with **H.data, **H.grid if H is an object of the
    Read class from hamr_read.py.
    
    Attributes
    ----------
    pars : dict from H-AMR read object
        Parameters dictionary (dx1, dx2, density_scale, etc.).
    CONSTANTS : dict
        Constants dictionary (hr.CONSTANTS, usually).
    **kwargs : dict
        Optional configurations (e.g. cgs).
    """
    def __init__(self, pars, CONSTANTS, **kwargs):
        self.cgs  = kwargs.get('cgs', False)
        self.rho0 = pars['density_scale']  if self.cgs else ONE # g/cm^3
        self.c    = CONSTANTS['c']         if self.cgs else ONE # cm/s
        B0        = np.sqrt(FOURPI * self.rho0) * self.c
        self.B0   = np.float32(B0)         if self.cgs else ONE # G

        self.dx1, self.dx2, self.dx3 = pars['dx1'], pars['dx2'], pars['dx3']
        self.pars = pars

    def calc_b(self, B, uu, gcov, **kwargs):
        """Compute b from B. Assumes B has units and returns b with units."""
        c, B0  = self.c, self.B0
        b      = hr.Tools.zeroNd(B.shape, 1, **kwargs) 
        # de-dimensionalize inputs
        uu    *=  ONE / c
        B     *=  ONE / B0
        # compute b^t
        b[0]   = Tools.dot(uu, B, gcov, **kwargs)
        # compute b^i
        for i in range(1, 4):
            b[i] = (B[i] + (uu[i] * b[0])) / uu[0]
        # re-dimensionalize inputs and b
        uu *= c
        B  *= B0
        b  *= B0
        return b
    
    def calc_b2(self, b, B, gcov, **kwargs):
        """Compute b^2. Assumes B and uu have units."""
        if (kwargs.get('from_B')):
            if 'uu' in kwargs:
                b = self.calc_b(B=B, gcov=gcov, **kwargs)
            else: 
                raise TypeError("Must specify uu to use from_B.")
        return Tools.dot(b, b, gcov, **kwargs)
        
    def calc_TEM(self, uu, B, gcov, gcon, **kwargs):
        """
        Compute the electromagnetic stress-energy tensor TEM.
        Default TEM^{mu nu}.
        "upper_lower": TEM^{mu}_{nu}.
        "lower_lower": TEM_{mu nu}.
        """
        # If b and/or b2 arguments provided, assume they have units.
        rho0   = self.rho0
        c      = self.c
        rho0c2 = rho0 * c * c
        B0     = self.B0
        B02    = self.B0 * self.B0

        b = kwargs.get('b')
        if b is None:
            # uu must be in cgs units for calc_b
            b = self.calc_b(B, uu, gcov)
            b *= ONE / B0
        else:
            b *= ONE / B0

        b2 = kwargs.get('b2')
        if b2 is None:
            b2 = self.calc_b2(b, b, gcov) * B02

        # Construct bb dyadic tensor.
        bb_tnsr = Tools.construct_dyadic(b, gcov, **kwargs)

        # Construct uu dyadic tensor.
        uu_tnsr = kwargs.get('uu_tnsr')
        if uu_tnsr is None:
            uu     *= ONE / c
            uu_tnsr = Tools.construct_dyadic(uu, gcov, **kwargs)
            uu     *= c

        # Calculate metric tensor (g^{mu nu}, delta^{mu}_{nu}, g_{mu nu})
        g_tnsr = Tools.g_metric(gcov, gcon, **kwargs)

        # Restore original units to b in place.
        b *= B0
        
        return ( (b2 / B02) * uu_tnsr + 
            0.5 *(b2 / B02) * g_tnsr - bb_tnsr) * rho0c2
    
    def calc_FEM(self, uu, B, gcov, gdet, **kwargs):
        """
        Compute the electromagnetic field-strength tensor.
        Referred to as the Faraday tensor in MTW.
        Default FEM^{mu nu}.
        "upper_lower": FEM^{mu}_{nu}.
        "lower_lower": FEM_{mu, nu}.
        Assumes uu and B have units.
        If frame = generic, assumes B is measured in frame uu,
        where uu = n would be frame of normal observer, etc.
        If frame = generic, electric field must be supplied as kwarg.
        Similarly, if non-ideal MHD, electric field must be supplied.
        """
        frame = kwargs.get('frame', 'fluid')
        if frame not in ('fluid', 'generic'):
            raise ValueError(
                "Please select a valid frame for the Faraday tensor. " + 
                "Options are frame=fluid or frame=generic."
            )
        if 'E' not in kwargs and frame == 'generic':
            raise ValueError("Must supply E kwarg to use generic frame.")

        # Fetch or calculate b.
        b = kwargs.get('b')
        if b is None:
            b = self.calc_b(B, uu, gcov, **kwargs)
        
        b  *= ONE / self.B0
        uu *= ONE / self.c

        # Require u_mu and b_mu for both frames.    
        uu_cov = np.einsum('ij...,j...->i...', gcov, uu)
        b_cov  = np.einsum('ij...,j...->i...', gcov, b)
        levi_civita = Tools.calc_Levi_Civita(4, gdet)
        
        if (frame == 'fluid'):
            # Equation 6 of Porth et al. (2017).
            FEM = -np.einsum(
                'ijkl...,k...,l...->ij...', levi_civita, uu_cov, b_cov)
        elif (frame == 'generic'):
            E   = kwargs.get('E')
            E  *= ONE / (np.sqrt(self.rho0) * self.c)

            # Equation 2 of Porth et al. (2017).
            FEM = (np.einsum('i...,j...->ij...', uu, E) - 
                   np.einsum('j...,i...->ij...', uu, E) -
                   np.einsum('ijkl...,k...,l...->ij...', 
                    levi_civita, uu_cov, b_cov))

            E  *= np.sqrt(self.rho0) * self.c

        # Restore units in place.        
        uu *= self.c
        if 'b' in kwargs:
            b  *= self.B0

        # Units of electromangetic tensor are B0.
        B0 = self.B0

        if 'upper_lower' in kwargs:
            return B0 * np.einsum('ij...,jk...->ik...', gcov, FEM)
        elif 'lower_lower' in kwargs:
            return B0 * np.einsum('ij...,kl...,jl...->ik...', gcov, gcov, FEM)
        
        return FEM * B0
    
    def calc_dual_FEM(self, uu, B, gcov, gdet, **kwargs):
        """
        Compute the dual of the electromagnetic field-strength tensor.
        Referred to as the Maxwell tensor in MTW.
        Default *FEM^{mu nu}.
        "upper_lower": *FEM^{mu}_{nu}.
        "lower_lower": *FEM_{mu nu}.
        Assumes uu and B have units.
        If frame = generic, assumes B is measured in frame uu,
        where uu = n would be frame of normal observer, etc.
        If frame = generic, electric field must be supplied as kwarg.
        Similarly, if non-ideal MHD, electric field must be supplied.
        """
        frame = kwargs.get('frame', 'fluid')
        if frame not in ('fluid', 'generic'):
            raise ValueError(
                "Please select a valid frame for the Maxwell tensor. " + 
                "Options are frame=fluid or frame=generic."
            )
        if 'E' not in kwargs and frame == 'generic':
            raise ValueError("Must supply E kwarg to use generic frame.")

        # Fetch or calculate b.
        b = kwargs.get('b')
        if b is None:
            b = self.calc_b(B, uu, gcov, **kwargs)
        
        b  *= ONE / self.B0
        uu *= ONE / self.c

        if frame == 'fluid':
            # Equation 6 of Porth et al. (2017).
            dual_FEM = (np.einsum('i...,j...->ij...', b, uu) - 
                        np.einsum('j...,i...->ij...', b, uu))
        elif frame == 'generic':
            B *= ONE / self.B0

            E  = kwargs.get('E')
            E *= ONE / (np.sqrt(self.rho0) * self.c)

            uu_cov = np.einsum('ij...,j...->i...', gcov, uu)
            E_cov  = np.einsum('ij...,j...->i...', gcov, E)
            levi_civita = Tools.calc_Levi_Civita(4, gdet)

            # Equation 3 of Porth et al. (2017).
            dual_FEM = (np.einsum('i...,j...->ij...', uu, B) - 
                        np.einsum('j...,i...->ij...', uu, B) -
                   np.einsum('ijkl...,k...,l...->ij...', 
                    levi_civita, uu_cov, E_cov))

            B *= self.B0
            E *= np.sqrt(self.rho0) * self.c
        
        # Restore units in place.        
        uu *= self.c
        if 'b' in kwargs:
            b  *= self.B0

        # Units of Hodge dual of electromangetic tensor are B0.
        B0 = self.B0

        if 'upper_lower' in kwargs:
            return B0 * np.einsum('ij...,jk...->ik...', gcov, dual_FEM)
        elif 'lower_lower' in kwargs:
            return B0 * np.einsum(
                'ij...,kl...,jl...->ik...', gcov, gcov, dual_FEM)
        
        return dual_FEM * B0
        
    def calc_J(self, uu, B, gcov, gdet, **kwargs):
        """
        Calculate the 4-current by taking the divergence of the 
        field-strength (Faraday) tensor.
        """
        faraday = self.calc_FEM(uu, B, gcov, gdet, **kwargs)
        return Tools.div_tnsr(faraday, gdet, self.dx1, self.dx2, self.dx3)
    
    def calc_J2(self, J, gcov, **kwargs):
        """Calculate J^2. If from_FEM = True, calculate J first."""
        if kwargs.get('from_FEM', False):
            if all(k in kwargs for k in ('uu', 'B', 'gdet')):
                J = self.calc_J(gcov=gcov, **kwargs)
            else:
                raise ValueError("Must specify uu, B, and gdet " + 
                                 "to use from_FEM=True.")
        return Tools.dot(J, J, gcov, **kwargs)

    def calc_vA(self, rho, uu, B, u_gas, gcov, **kwargs):
        """Relativistic Alfvén velocity (vA)."""
        rho0   = self.rho0
        c      = self.c
        rho0c2 = rho0 * c * c
        B0     = self.B0
        B02    = self.B0 * self.B0
        
        b = kwargs.get('b')
        if b is None:
            b = self.calc_b(B, uu, gcov)

        b2 = kwargs.get('b2')
        if b2 is None:
            b2 = self.calc_b2(b, b, gcov)

        pgas = kwargs.get('pgas') 
        if pgas is None:
            gamma_adi = np.float32(self.pars['gam'])
            pgas      = (gamma_adi - 1) * u_gas

        u_rad = kwargs.get('u_rad', 0) 

        return c * (b / B0) / np.sqrt(
            (rho / rho0)  +  (u_gas / rho0c2)  + (pgas / rho0c2) + 
            (b2 / B02) + (4./3.) * (u_rad / rho0c2)
        )

class Fluid():
    """
    Fluid class manages hydrodynamic quantities. All methods related to
    two-temperature thermodynamics are contained in this class.
    Calculations of the gas thermal pressure pgas, the hydrodynamic (HD)
    stress-energy tensor THD^{mu nu}, and Coulomb-collisional timescales
    for ion-ion momentum exchange (ii), ion-electron energy exchange (ie),
    and electron-electron momentum exchange (ee) are handled by this class.

    Functions can be called with **H.data, **H.grid if H is an object of the
    Read class from hamr_read.py.
    
    Attributes
    ----------
    pars : dict from H-AMR read object
        Parameters dictionary (dx1, dx2, density_scale, etc.).
    CONSTANTS : dict
        Constants dictionary (hr.CONSTANTS, usually).
    **kwargs : dict
        Optional configurations (e.g. cgs).
    """
    def __init__(self, pars, CONSTANTS, **kwargs):
        self.cgs  = kwargs.get('cgs', False)
        self.rho0 = pars['density_scale']  if self.cgs else ONE # g/cm^3
        self.c    = CONSTANTS['c']         if self.cgs else ONE # cm/s
        self.kB   = CONSTANTS['kB']        if self.cgs else ONE # erg/K
        # Equation 11 in Sadowski+ (2017)
        X_AB, Y_AB = CONSTANTS['X_AB'], CONSTANTS['Y_AB']
        self.mu_i = np.float32(4./(4. * X_AB + Y_AB)) if self.cgs else ONE
        # Equation 10 in Sadowski+ (2017)
        self.mu_e = np.float32(2. / (1. + X_AB)) if self.cgs else ONE
        self.mp   = CONSTANTS['mp']        if self.cgs else ONE # g
        self.mi   = ((X_AB + Y_AB) * CONSTANTS['mp']) if self.cgs else ONE # g 
        self.me   = CONSTANTS['me']        if self.cgs else ONE # g
        self.sigT = CONSTANTS['sigmaT']    if self.cgs else ONE # cm^2

        self.Theta_min = 2.e-2
        self.kwargs    = kwargs
        self.CONSTANTS = CONSTANTS
        self.params    = pars

    def calc_THD(self, rho, uu, u_gas, pgas, gcov, gcon, **kwargs):
        """
        Compute the hydrodynamic stress-energy tensor THD.
        Default THD^{mu nu}.
        "upper_lower": THD^{mu}_{nu}.
        "lower_lower": THD_{mu nu}.
        """
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
        
        # Construct uu dyadic tensor.
        uu_tnsr = kwargs.get('uu_tnsr')
        if uu_tnsr is None:
            uu     *= ONE / c
            uu_tnsr = Tools.construct_dyadic(uu, gcov, **kwargs)
            uu     *= c
        
        g_tnsr  = Tools.g_metric(gcov, gcon, **kwargs)
        return (
            (((rho / rho0) + (u_gas / rho0c2) + (pgas / rho0c2)) * uu_tnsr + 
            (pgas / rho0c2) * g_tnsr) * rho0c2
        )

    def calc_pgas(self, u_gas, Te, Ti, **kwargs):
        """
        Compute pgas from u_gas using gamma law EOS.
        Equation 23 in Sadowski+ (2017).
        """
        Theta_e = (self.kB * Te) / (self.me * self.c * self.c)
        Theta_i = (self.kB * Ti) / (self.mi * self.c * self.c)

        # Calculate gamma (ratio of specific heats).
        # Equation 28 in Sadowski+ (2017).
        gam_e = (10. + 20. * Theta_e) / (6. + 15. * Theta_e)
        gam_i = (10. + 20. * Theta_i) / (6. + 15. * Theta_i)
        
        gam_g = 1. + ((
                (gam_e - 1.) * (gam_i - 1.) * (
                (self.mu_i / self.mu_e) + (Ti / Te)) ) / (
                (Ti / Te) * (gam_e - 1.) + 
                (self.mu_i / self.mu_e) * (gam_i - 1.) ))
        return (gam_g - 1.) * u_gas # erg/cm^3

    def calc_teq(self, rho, Te, Ti, **kwargs):
        """
        Compute ion-electron equilibration timescale teq.
        Assumes rho in g/cm^3, Te and Ti in K, u_gas in erg/cm^3.
        """
        Theta_e = (self.kB * Te) / (self.me * self.c * self.c)
        Theta_i = (self.kB * Ti) / (self.mi * self.c * self.c)

        # Calculate gamma (ratio of specific heats).
        # Equation 28 in Sadowski+ (2017).
        gam_e = (10. + 20. * Theta_e) / (6. + 15. * Theta_e)
        gam_i = (10. + 20. * Theta_i) / (6. + 15. * Theta_i)

        u_i = (1./(gam_i-1.)) * (rho / (self.mu_i * self.mp)) * self.kB * Ti
        u_e = (1./(gam_e-1.)) * (rho / (self.mu_e * self.mp)) * self.kB * Te

        # u_i > u_e whenever Ti > Te except for narrow range of 
        # Te and Ti around Te < Ti < 2 Te and Te \approx 1e9 K, i.e., when
        # electrons are relativistic and ions are non-relativistic.
        # In that case, use ion gamma and mu for energy difference.
        delta_energy = u_i - u_e
        delta_energy[(delta_energy < 0) & (Ti > Te)] = ((1./(gam_i-1.)) * (
            rho / (self.mu_i * self.mp)) * self.kB * (Ti- Te))[
            (delta_energy < 0) & (Ti > Te)]

        qie = kwargs.get('qie')
        if qie is None:
            qie = self.calc_qie(rho, Te, Ti, **kwargs)
        
        teq = delta_energy / qie
        teq[qie == 0.] = 0.

        # If temperature difference < 2% of Te, then assume negligible teq
        teq[np.abs(Ti - Te) < (self.Theta_min * Te)] = 0.0

        # Cooling: Ti > Te. Heating: Te > Ti.
        if kwargs.get('cooling'):
            teq[Ti < Te] = 0.0
            return teq # s
        elif kwargs.get('heating'):
            teq[Ti > Te] = 0.0
            return teq # s
        
        return teq # s
    
    def calc_tii(self, rho, Te, Ti, **kwargs):
        """
        Compute ion-ion momentum-exchange (collision) timescale tii.
        Assumes rho in g/cm^3, Te and Ti in K, u_gas in erg/cm^3.
        """
        ne = rho / (self.mu_e * self.mp)
        ni = rho / (self.mu_i * self.mp)
        return (Ti / Te) ** (3./2.) * np.sqrt(self.me / self.mi) * (
            (self.Coulomb_log(ne, Te) / self.Coulomb_log(ni, Ti)) * (
            self.calc_teq(rho, Te, Ti, **kwargs)))
    
    def calc_tee(self, rho, Te, Ti, **kwargs):
        """
        Compute electron-electron momentum-exchange (collision) timescale tee.
        Assumes rho in g/cm^3, Te and Ti in K, u_gas in erg/cm^3.
        """
        return (self.me / self.mi) * self.calc_teq(rho, Te, Ti, **kwargs)

    def calc_qie(self, rho, Te, Ti, **kwargs):
        """
        Compute cooling/heating rate due to ion-electron collisions.
        Assumes rho in g/cm^3, Te and Ti in K.
        Equation 20 in Sadowski+ (2017).
        See also Stepney & Guilbert (1983) Equation 15.
        """
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
        
        # Cooling: Ti > Te. Heating: Te > Ti.
        if kwargs.get('cooling'):
            qie[Ti < Te] = 0.0
            return qie # erg/cm^3 s
        elif kwargs.get('heating'):
            qie[Ti > Te] = 0.0
            return qie # erg/cm^3 s
        
        return qie # erg/cm^3 s
    
    def Bessel_terms(self, Theta_e, Theta_i):
        """
        H-AMR uses approximations for the Bessel functions when the
        electrons (E) or ions (I) are cold (Theta < Theta_min).
        """
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
        """Cold electrons and cold ions."""
        Theta_sum = Theta_e + Theta_i
        return ((2. * (Theta_sum ** 2) + 2. * Theta_sum + 1.) / (
            np.sqrt(0.5 * np.pi * (Theta_sum ** 3))))

    def hotE_coldI(self, Theta_e, Theta_i):
        """Hot electrons and cold ions."""
        Theta_sum  = Theta_e + Theta_i 
        k2e        = kn(2, (1. / Theta_e))
        return (((2. * (Theta_sum ** 2) + 2. * Theta_sum + 1.) * (
            np.sqrt(Theta_e))) / ( k2e * np.exp(1. / Theta_e) * (
            np.sqrt(Theta_sum ** 3))))

    def coldE_hotI(self, Theta_e, Theta_i):
        """Cold electrons and hot ions."""
        Theta_sum = Theta_e + Theta_i 
        k2i       = kn(2, (1. / Theta_i))
        return (((2. * (Theta_sum ** 2) + 2. * Theta_sum + 1.) * (
            np.sqrt(Theta_i))) / ( k2i * np.exp(1. / Theta_i) * (
            np.sqrt(Theta_sum ** 3))))

    def hotE_hotI(self, Theta_e, Theta_i):
        """Hot electrons and hot ions."""
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
        return 35.4 + np.log((Te / 1.e7) * np.sqrt(1.e-3 / ne))
    
    def calc_qie_cpp(self, rho, Te, Ti, **kwargs):
        """Cooling/heating rate due to ion-electron collisions (C++)."""
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
        qie = hr.Tools.zeroNd(rho.shape, 1, dtype=dtype)
        Fluidc.calc_qie(qie, rho, Te, Ti, nb, nx1, nx2, nx3)
        # If temperature difference < 2% of Te, then assume negligible qie
        qie[np.abs(Ti - Te) < (self.Theta_min * Te)] = 0.0
        
        # Cooling: Ti > Te. Heating: Te > Ti.
        if kwargs.get('cooling'):
            qie[Ti < Te] = 0.0
            return qie # erg/cm^3 s
        elif kwargs.get('heating'):
            qie[Ti > Te] = 0.0
            return qie # erg/cm^3 s
        
        return qie # erg/cm^3 s

    def calc_delta_e(self, rho, u_gas, Te, Ti, B, gcov, **kwargs):
        """
        Electron heating fraction.
        Equation 13 of Chael et al. (2018).
        """
        c = self.c
        if 'pgas' not in kwargs:
            pgas = self.calc_pgas(u_gas, Te, Ti)
        else:
            pgas = kwargs.get('pgas')
        if 'b2' not in kwargs:
            if 'uu' not in kwargs:
                raise ValueError("Must supply uu to construct b2.")
            Fld = Field(self.params, self.CONSTANTS, cgs=self.cgs)
            b2  = Fld.calc_b2(b=B, B=B, gcov=gcov, 
                              uu=kwargs.get('uu'), from_B=True)
        else:
            b2  = kwargs.get('b2')
        sigma_w = b2 / (4. * np.pi * (rho * c * c + u_gas + pgas))
        n_i     = rho / (self.mu_i * self.mp)
        Beta_i  = n_i * self.kB * Ti / (b2 / (8. * np.pi))
        Beta_max= 1. / (4. * sigma_w)
        delta_e = 0.5 * np.exp(-(1. - Beta_i/Beta_max)/(0.8 + sigma_w ** 0.5))
        return delta_e

class MRI():
    """
    MRI (magnetorotational instability) class handles quality factors and
    growth rates for modes of the MRI.
    
    Attributes
    ----------
    r : arr of float
        Coordinate radius array from grid object, grid['r'].
    theta : arr of float
        Coordinate theta array from grid object, grid['theta'].
    phi : arr of float
        Coordinate phi array from grid object, grid['phi'].
    pars : dict from H-AMR read object
        Parameters dictionary (dx1, dx2, density_scale, etc.).
    CONSTANTS : dict
        Constants dictionary (hr.CONSTANTS, usually).
    **kwargs : dict
        Optional configurations (e.g. cgs).

    """
    def __init__(self, r, theta, phi, pars, CONSTANTS, **kwargs):
        self.dx1, self.dx2, self.dx3 = pars['dx1'], pars['dx2'], pars['dx3']

        self.cgs  = kwargs.get('cgs', False)
        self.MBH  = kwargs.get('MBH', ONE) if self.cgs else ONE # g
        self.c    = CONSTANTS['c']         if self.cgs else ONE # cm/s
        self.G    = CONSTANTS['G']         if self.cgs else ONE # cm^3/g s^2
        self.Msun = CONSTANTS['Msun']      if self.cgs else ONE # g
        Rg        = (self.G * self.MBH * self.Msun) / (self.c * self.c)

        self.r  = r / Rg
        self.th = theta
        self.ph = phi

    def calc_Q(self, rho, uu, B, u_gas, gcov, **kwargs):
        """MRI quality (Q) factors."""
        Fld = Field(self.pars, self.CONSTANTS, cgs=self.cgs)
        vA  = Fld.calc_vA(rho, uu, B, u_gas, gcov, **kwargs)
        TWOPI = np.float32(2. * np.pi)
        if 'amr' in kwargs:
            self.calc_upscale(**kwargs)
        else:
            raise ValueError("Must supply amr dictionary and " + 
                             "level_x1, x2, x3 to calculate quality factors")
        dx1   = self.dx1 * np.sqrt(gcov[1, 1]) * self.upscalex1
        dx2   = self.dx2 * np.sqrt(gcov[2, 2]) * self.upscalex2
        dx3   = self.dx3 * np.sqrt(gcov[3, 3]) * self.upscalex3
        vrot       = uu[3] / uu[0]
        lambda_MRI = TWOPI * (vA / vrot) * self.r
        Q       = hr.Tools.zeroNd(uu.shape, 1, dtype=np.float32)
        Q[1]    = lambda_MRI / dx1  
        Q[2]    = lambda_MRI / dx2
        Q[3]    = lambda_MRI / dx3
        return Q

    def calc_upscale(self, amr, level_x1, level_x2, level_x3, **kwargs):
        self.max_levelx1 = amr['max_levelx1']
        self.max_levelx2 = amr['max_levelx2']
        self.max_levelx3 = amr['max_levelx3']
        lr1              = kwargs.get('lowres1', 1)
        lr2              = kwargs.get('lowres2', 1)
        lr3              = kwargs.get('lowres3', 1)
        self.upscalex1   = 2 ** (self.max_levelx1 - level_x1) / lr1
        self.upscalex2   = 2 ** (self.max_levelx2 - level_x2) / lr2
        self.upscalex3   = 2 ** (self.max_levelx3 - level_x3) / lr3

class Tools():
    _Toolsc_obj = None
    
    @classmethod
    def _get_cpp_tools(cls):
        if cls._Toolsc_obj is None:
            import hamr_readc
            cls._Toolsc_obj = hamr_readc.PyTools()
        return cls._Toolsc_obj

    @staticmethod
    def dot(a, b, gcov, **kwargs):
        """Contract two 4-vectors: g_{mu nu} a^nu b^mu."""
        if (kwargs.get('cpp', False)):
            return Tools.dot_cpp(a, b, gcov, **kwargs)
        a_cov = np.einsum('ij...,j...->i...', gcov, a)
        return np.einsum('i...,i...->...', a_cov, b)
    
    @staticmethod
    def contract_vector(a, gcov, **kwargs):
        """Contract 4-vector with tensor gcov: g_{mu nu} a^nu."""
        if (kwargs.get('cpp', False)):
            return Tools.contract_vector_cpp(a, gcov, **kwargs)
        return np.einsum('ij...,j...->i...', gcov, a)

    @staticmethod
    def construct_dyadic(u, gcov, **kwargs):
        """Construct dyadic tensor uu from 4-vector u."""
        if kwargs.get('cpp', False):
            return Tools.construct_dyadic_cpp(u, gcov, **kwargs)
            
        upper_lower = 'upper_lower' in kwargs
        lower_lower = 'lower_lower' in kwargs
        
        if 'index' in kwargs:
            mu, nu = kwargs.get('index')
            if upper_lower:
                return u[mu] * np.einsum('j...,j...->...', gcov[nu], u)
            elif lower_lower:
                return (np.einsum('j...,j...->...', gcov[mu], u) * 
                        np.einsum('j...,j...->...', gcov[nu], u))
            else:
                return u[mu] * u[nu]
                
        if upper_lower:
            # u^mu u_nu
            return np.einsum('i...,jk...,k...->ij...', u, gcov, u, 
                             optimize='greedy')
        elif lower_lower:
            # u_mu u_nu 
            return np.einsum('ik...,jl...,k...,l...->ij...', 
                             gcov, gcov, u, u, optimize='greedy')
        else:
            # u^mu u^nu 
            return np.einsum('i...,j...->ij...', u, u)

    @staticmethod
    def g_metric(gcov, gcon, **kwargs):
        """Calculate metric tensor that appears in stress-energy tensors."""
        if 'upper_lower' in kwargs:
            # \delta^{\mu}_{\nu}
            if 'index' in kwargs:
                mu, nu = kwargs['index']
                return 1.0 * (mu == nu)
            return Tools.get_identity(gcov, **kwargs)
        
        elif 'lower_lower' in kwargs:
            # Return metric.
            g_tnsr = gcov
        
        else:
            # Default to inverse metric g^{mu nu}.
            g_tnsr = gcon

        if 'index' in kwargs:
            mu, nu = kwargs['index']
            return g_tnsr[mu, nu]
            
        return g_tnsr

    @staticmethod
    def get_identity(gcov, **kwargs):
        """Generate a memory-efficient identity tensor using broadcasting."""
        id_tnsr = np.eye(4, dtype=gcov.dtype)

        grid_ndim = gcov.ndim - 2
        broadcast_shape = (4, 4) + (1,) * grid_ndim

        return id_tnsr.reshape(broadcast_shape)

    @staticmethod
    def antisymmetric_symbol3(i, j, k):
        """Levi-Civita permutation symbol in 3D using inversion counting."""
        indices = (i, j, k)
        if len(set(indices)) < 3 or any(n < 0 or n > 2 for n in indices):
            return 0
        inversions = (i > j) + (i > k) + (j > k)
        return 1 if inversions % 2 == 0 else -1

    @staticmethod
    def antisymmetric_symbol4(i, j, k, l):
        """Levi-Civita permutation symbol in 4D using inversion counting."""
        indices = (i, j, k, l)
        if len(set(indices)) < 4 or any(n < 0 or n > 3 for n in indices):
            return 0
        inversions = ((i > j) + (i > k) + (i > l) +
                      (j > k) + (j > l) +
                      (k > l))
        return 1 if inversions % 2 == 0 else -1

    @staticmethod
    def calc_Levi_Civita(n, gdet, **kwargs):
        """
        Compute the Levi-Civita tensor.
        Multiply by 1/gdet if cov=False (default) or gdet if cov=True.
        """
        if n == 3:
            # 1. Populate a tiny, flat 3x3x3 tensor
            epsilon_core = np.fromiter(
                (Tools.antisymmetric_symbol3(i, j, k)
                 for i in range(3) for j in range(3) for k in range(3)),
                dtype=gdet.dtype
            ).reshape(3, 3, 3)

        elif n == 4:
            # 1. Populate a tiny, flat 4x4x4x4 tensor
            epsilon_core = np.fromiter(
                (Tools.antisymmetric_symbol4(i, j, k, l)
                 for i in range(4) for j in range(4) for k in range(4) for l in range(4)),
                dtype=gdet.dtype
            ).reshape(4, 4, 4, 4)
        else:
            raise ValueError("Levi-Civita tensor calculation " + 
                             "only supports n=3 or n=4 dimensions.")

        # Match grid shape
        grid_ndim = gdet.ndim
        broadcast_shape = epsilon_core.shape + (1,) * grid_ndim
        epsilon_view = epsilon_core.reshape(broadcast_shape)

        cov = kwargs.get('cov', False)
        return gdet * epsilon_view if cov else (ONE / gdet) * epsilon_view
 
    @staticmethod
    def div_tnsr(tnsr, gdet, dx1, dx2, dx3, **kwargs):
        """Calculate divergence of a tensor."""
        return (1. / gdet) * (
            np.gradient((gdet * tnsr[:, 1]), dx1, axis=2) + 
            np.gradient((gdet * tnsr[:, 2]), dx2, axis=3) + 
            np.gradient((gdet * tnsr[:, 3]), dx3, axis=4)
        )
    
    @staticmethod
    def dot_cpp(a, b, gcov, **kwargs):
        """Compute dot product g_{mu nu} a^nu b^mu with C++ acceleration.""" 
        dims = a.shape
        if len(dims) != 5 or dims[0] != 4:
            raise ValueError(f"This 4-vector with shape {dims} " + 
                             "cannot be passed to C++.")
            
        Toolsc = Tools._get_cpp_tools()
        
        # Zero-allocation view/cast check
        a_c = np.asarray(a, dtype=np.float32)
        b_c = np.asarray(b, dtype=np.float32)
        
        _, nb, nx1, nx2, nx3 = dims
        axisym = 1 if gcov.shape[-1] == 1 else 0
        
        # Allocate raw uninitialized buffer
        a_cov = np.empty(dims, dtype=np.float32)
        
        # Lower index in C++
        Toolsc.lower_index(a_c, a_cov, gcov, nb, nx1, nx2, nx3, axisym)
        
        # Dot product contraction
        return np.einsum('i...,i...->...', a_cov, b_c, optimize='greedy')

    @staticmethod
    def contract_vector_cpp(a, gcov, **kwargs):
        """Contract 4-vector with gcov using C++ acceleration."""
        dims   = a.shape
        if len(dims) != 5 or dims[0] != 4:
            raise ValueError(f"This 4-vector with shape {dims} " + 
                             "cannot be passed to C++.")

        Toolsc = Tools._get_cpp_tools()
        
        # Zero-allocation view/cast check
        a_c = np.asarray(a, dtype=np.float32)
        
        _, nb, nx1, nx2, nx3 = dims
        axisym = 1 if gcov.shape[-1] == 1 else 0
        
        # Allocate raw uninitialized buffer
        a_cov = np.empty(dims, dtype=np.float32)

        # Lower index in C++
        Toolsc.lower_index(a_c, a_cov, gcov, nb, nx1, nx2, nx3, axisym)

        return a_cov
        
    @staticmethod
    def construct_dyadic_cpp(u, gcov, **kwargs):
        """Construct dyadic tensor uu with C++ acceleration."""
        dims = u.shape
        if len(dims) != 5 or dims[0] != 4:
            raise ValueError(f"This 4-vector with shape {dims} " + 
                             "cannot be passed to C++.")
            
        # Get our cached C++ interface safely
        Toolsc = Tools._get_cpp_tools()
        
        # Zero-allocation type-check
        u_c = np.asarray(u, dtype=np.float32)
        
        _, nb, nx1, nx2, nx3 = dims
        axisym = 1 if gcov.shape[-1] == 1 else 0
        
        upper_lower = 'upper_lower' in kwargs
        lower_lower = 'lower_lower' in kwargs
        
        if 'index' in kwargs:
            mu, nu = kwargs.get('index')
            if upper_lower:
                u_nu = np.empty(dims[1:], dtype=np.float32)
                Toolsc.lower_index_mu(u_c, u_nu, gcov, nu, 
                                      nb, nx1, nx2, nx3, axisym)
                return u_c[mu] * u_nu
            elif lower_lower:
                u_mu = np.empty(dims[1:], dtype=np.float32)
                u_nu = np.empty(dims[1:], dtype=np.float32)
                Toolsc.lower_index_mu(u_c, u_mu, gcov, mu, 
                                      nb, nx1, nx2, nx3, axisym)
                Toolsc.lower_index_mu(u_c, u_nu, gcov, nu, 
                                      nb, nx1, nx2, nx3, axisym)
                return u_mu * u_nu
            else:
                return u_c[mu] * u_c[nu]
                
        if upper_lower or lower_lower:
            u_cov = np.empty(dims, dtype=np.float32)
            Toolsc.lower_index(u_c, u_cov, gcov, nb, nx1, nx2, nx3, axisym)
            
            if upper_lower:
                return np.einsum('i...,j...->ij...', u_c, u_cov, 
                                 optimize='greedy')
            else: # lower_lower
                return np.einsum('i...,j...->ij...', u_cov, u_cov, 
                                 optimize='greedy')
        else:
            return np.einsum('i...,j...->ij...', u_c, u_c, optimize='greedy')

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