import numpy as np

ONE = np.float32(1.)

class Analytic():
    def __init__(self, CONSTANTS, **kwargs):
        self.CONSTANTS = CONSTANTS
        self.cgs = kwargs.get('cgs', True)
        self.c   = CONSTANTS['c'] if self.cgs else ONE # cm/s

    def calc_LEdd(self, MBH):
        """Calculate Eddington luminosity. MBH in solar masses."""
        return 1.26e38 * MBH # erg/s

    def calc_Mdot_Edd(self, MBH, **kwargs):
        """Calculate Eddington mass-accretion rate. MBH in solar masses."""
        eta = kwargs.get('eta', 0.1)
        return self.calc_LEdd(MBH) / (eta * self.c * self.c)

    def calc_R_Horizon(self, a):
        """Calculate radius of (outer) event horizon for Kerr black hole."""
        # a is spin parameter
        return 1. + np.sqrt(1 - a*a)
    
    def calc_R_ISCO(self, a):
        """
        Radius of the innermost stable circular orbit (R_ISCO / R_g).
        Eq. 5.4.8a in Novikov & Thorne (1973).
        Sign correction shown in Eq. 15k of Page & Thorne (1974).
        """
        z1 = self.calc_Z1(a)
        z2 = self.calc_Z2(a)
        return (3. + z2 - 
            np.sign(a) * ((3. - z1) * (3. + z1 + 2. * z2))**(1./2.))
    
    def calc_R_Ergosphere(self, theta, a):
        """Ergosphere radius in Rg."""
        return 1. + np.sqrt(1. - np.square(a * np.cos(theta)))
    
    def calc_OmegaK(self, MBH, a, r_Rg):
        """Calculate Keplerian orbital frequency."""
        MBH  = MBH                                if self.cgs else ONE
        G    = self.CONSTANTS['G']                if self.cgs else ONE
        Msun = self.CONSTANTS['Msun']             if self.cgs else ONE
        Rg   = G * MBH * Msun / (self.c * self.c) if self.cgs else ONE
        # Eq. 36 of Abramowicz & Fragile (2013).
        return np.sqrt(G * MBH * Msun / Rg ** 3) * (
            1. / (r_Rg ** (3./2.) + a))

    def calc_Z1(self, a):
        """Eq. 5.4.8b in Novikov & Thorne (1973)."""
        return (1. + (1. - a * a)**(1./3.) * (
            (1. + a)**(1./3.) + (1. - a)**(1./3.)))
    
    def calc_Z2(self, a):
        """Eq. 5.4.8b in Novikov & Thorne (1973)."""
        return (3. * a * a + (self.calc_Z1(a))**2.)**(1./2.)
    
    def calc_efficiency_NT73(self, r, a):
        """
        Calculate Novikov & Thorne (1973) (with corrections from
        Page & Thorne 1974) efficiency of thin-disk accretion eta.
        Expects dimensionless radius (units of R_g).
        """
        deta_dr = self.calc_dissipation_NT73(r, a)
        dr      = np.append(np.diff(r), np.diff(r)[-1])
        eta     = np.zeros(len(r))
        eta     = np.cumsum(deta_dr * dr)
        return eta

    def calc_dissipation_NT73(self, r, a):
        """Calculate Novikov & Thorne (1973) local dissipation rate."""
        NT73    = Novikov_Thorne(self.CONSTANTS)
        r_ISCO  = self.calc_R_ISCO(a)
        r_NT    = r[r > r_ISCO]
        deta_dr = np.zeros(len(r))
        deta_dr[r > r_ISCO] = NT73.calc_dEdot_Mdot_dr(r_NT, a)
        return deta_dr
    
class Novikov_Thorne(Analytic):
    # Inherited class for Novikov-Thorne thin disk functions
    # with corrections due to Page & Thorne (1974)
    def __init__(self, CONSTANTS, **kwargs):
        super().__init__(CONSTANTS, **kwargs)
    
    # Calculate d/dr (Edot/Mdot) given in Page & Thorne (1974)
    # Eq. 15n (function called f)
    # Expects r as dimensionless radius (units of R_g)
    def calc_dEdot_Mdot_dr(self, r, a):
        x    = np.sqrt(r)
        x0   = np.sqrt(super().calc_R_ISCO(a))
        x1   = 2. * np.cos( (1./3.) * np.acos(a) - np.pi / 3. )
        x2   = 2. * np.cos( (1./3.) * np.acos(a) + np.pi / 3. )
        x3   = -2. * np.cos( (1. / 3.) * np.acos(a) )
        xarr = [x, x1, x2, x3, x0]
        return ((3./2.) * (1. / (x ** 2. * (x ** 3. - 3. * x + 2. * a))) * (
            x - x0 - (3./2.) * a * np.log(x / x0) 
            - self.get_log_term(xarr, a, 1)
            - self.get_log_term(xarr, a, 2)
            - self.get_log_term(xarr, a, 3)

        ))
    
    def get_log_term(self, xarr, a, n):
        x  = xarr[0]
        x0 = xarr[-1]
        xc = xarr[:-1] + xarr[1:-1] # "x cyclic" simplifies cyclic pattern
        return (3. * ((xarr[n] - a) ** 2.) * (
            np.log((x - xarr[n])/(x0 - xarr[n])) * (
            1. / (xarr[n] * (xarr[n] - xc[n + 1]) * (xarr[n] - xc[n + 2])))
        ))

class Shakura_Sunyaev(Analytic):
    # Calculate gas or radiation dominated Shakura-Sunyaev (1973)
    # solutions for thin-disk accretion flows
    # Mdot in units of Mdot_Edd
    def __init__(self, Mdot_MdotEdd, CONSTANTS, **kwargs):
        super().__init__(CONSTANTS, **kwargs)
        self.Mdot  = Mdot_MdotEdd
        self.MBH   = kwargs.get('MBH', ONE)
        self.R0    = kwargs.get('R0',  ONE) # measurement radius in Rg
        self.alpha = kwargs.get('alpha', 0.1)
        self.eta   = kwargs.get('eta', 0.179) # Novikov-Thorne efficiency
        gas_dom    = kwargs.get('gas_dominated',       True)
        rad_dom    = kwargs.get('radiation_dominated', False)
        # Calculate gravitational radius
        G       = self.CONSTANTS['G']
        Msun    = self.CONSTANTS['Msun']
        c       = self.CONSTANTS['c']
        self.Rg =  G * self.MBH * Msun / (c * c)
        # Calculate gas or radiation-dominated solutions
        self.Sigma = {}
        self.HoR   = {}
        if (rad_dom):
            gas_dom = False
        elif (rad_dom and gas_dom):
            raise ValueError("Invalid choice, must select gas_dominated ",
                "(default) or radiation_dominated.")
        if (gas_dom):
            self.Sigma['ff'] = self.calc_Sigma_gas_ff()
            self.Sigma['es'] = self.calc_Sigma_gas_es()
            self.HoR['ff'] = self.calc_HoR_gas_ff()
            self.HoR['es'] = self.calc_HoR_gas_es()
        else:
            self.Sigma['es'] = self.calc_Sigma_rad_es()
            self.HoR['es']   = self.calc_HoR_rad_es()

    # Equations 97 of Abramowicz & Fragile (2013)
    def calc_Sigma_gas_ff(self):
        # surface density for gas pressure dominated disk
        # with free-free opacity
        a = np.power(self.alpha, -4./5.)
        m = np.power(self.MBH,    1./5.)
        l = np.power(self.Mdot / self.eta, 7./10.)
        r = np.power(self.R0, -3./4.)
        return 4.e5 * a * m * l * r

    def calc_HoR_gas_ff(self):
        # scale height for gas dominated disk with free-free opacities
        a = np.power(self.alpha, -1./10.)
        m = np.power(self.MBH,    9./10.)
        l = np.power(self.Mdot / self.eta, 3./20.)
        r = np.power(self.R0, 9./8.)
        return 4.e2 * a * m * l * r / (self.R0 * self.Rg)

    # Equations 98 of Abramowicz & Fragile (2013)
    def calc_Sigma_gas_es(self):
        # surface density for gas pressure dominated disk
        # with electron-scattering opacity
        a = np.power(self.alpha, -4./5.)
        m = np.power(self.MBH,    1./5.)
        l = np.power(self.Mdot / self.eta, 3./5.)
        r = np.power(self.R0, -3./5.)
        return 9.e4 * a * m * l * r

    def calc_HoR_gas_es(self):
        # scale height for gas dominated disk 
        # with electron-scattering opacity
        a = np.power(self.alpha, -1./10.)
        m = np.power(self.MBH,    9./10.)
        l = np.power(self.Mdot / self.eta, 1./5.)
        r = np.power(self.R0, 21./20.)
        return 1.e3 * a * m * l * r / (self.R0 * self.Rg)

    # Equations 99 of Abramowicz & Fragile (2013)
    def calc_Sigma_rad_es(self):
        # surface density for radiation dominated disk
        # with electron-scattering opacity
        a = np.power(self.alpha, -1.)
        l = np.power(self.Mdot / self.eta, -1.)
        r = np.power(self.R0, 3./2.)
        return 5. * a * l * r
    
    def calc_HoR_rad_es(self):
        # scale height for radiation dominated disk
        # with electron-scattering opacity
        # Equation 99 of Abramowicz & Fragile (2013) omits the factor of m;
        # H = 3 kappa Mdot / (8 pi c) scales as m * mdot, so H/R is independent of m.
        m = np.power(self.MBH, 1.)
        l = np.power(self.Mdot / self.eta, 1.)
        return 1.e5 * m * l / (self.R0 * self.Rg)

    # Radiation-dominated, electron-scattering zone WITH the inner-boundary
    # factor. calc_Sigma_rad_es and calc_HoR_rad_es above are the A&F (2013)
    # Eq. 99 power laws, which drop f; they are kept as they were. Here the
    # zero-torque factor f = 1 - sqrt(r_in / r) of Shakura & Sunyaev (1973)
    # is kept, with r_in = R_ISCO(a):
    #     H     = 3 kappa_es Mdot f / (8 pi c)
    #     nu    = alpha c_s H = alpha Omega_K H^2,  nu Sigma = Mdot f / (3 pi)
    #     Sigma = Mdot f / (3 pi alpha Omega_K H^2)
    # H is constant at large r and goes to zero at the ISCO, so H/R rises
    # inward, peaks, then falls to zero at r_in. Sigma diverges as 1/f there.
    # Newtonian Omega_K, and none of the Novikov-Thorne relativistic
    # correction factors. Requires cgs=True. Returns NaN at r <= r_in, where
    # the solution does not exist.
    def calc_f_inner(self, a):
        """Inner-boundary factor f = 1 - sqrt(R_ISCO / R0)."""
        return 1. - np.sqrt(self.calc_R_ISCO(a) / self.R0)

    def calc_H_rad_es_inner(self, a):
        """Scale height H in cm."""
        f      = self.calc_f_inner(a)
        kappa  = 0.2 * (1. + self.CONSTANTS['X_AB'])    # cm^2/g
        Mdot   = self.Mdot * self.calc_Mdot_Edd(self.MBH, eta=self.eta)
        H      = 3. * kappa * Mdot * f / (8. * np.pi * self.c)
        return np.where(f > 0., H, np.nan)

    def calc_HoR_rad_es_inner(self, a):
        return self.calc_H_rad_es_inner(a) / (self.R0 * self.Rg)

    def calc_Sigma_rad_es_inner(self, a):
        """Surface density in g/cm^2."""
        f      = self.calc_f_inner(a)
        Mdot   = self.Mdot * self.calc_Mdot_Edd(self.MBH, eta=self.eta)
        H      = self.calc_H_rad_es_inner(a)
        OmegaK = np.sqrt(self.CONSTANTS['G'] * self.MBH * self.CONSTANTS['Msun'] /
                         (self.R0 * self.Rg) ** 3)
        return np.where(f > 0., Mdot * f / (3. * np.pi * self.alpha * OmegaK * H * H), np.nan)

class ADAF(Analytic):
    # Narayan and Yi (1995b) ADAF solution
    def __init__(self, Mdot_MdotEdd, CONSTANTS, **kwargs):
        super().__init__(CONSTANTS, **kwargs)
        self.Mdot  = Mdot_MdotEdd
        self.MBH   = kwargs.get('MBH', ONE)
        self.R0    = kwargs.get('R0',  ONE) # measurement radius in Rg
        self.alpha = kwargs.get('alpha', 0.1)
        self.eta   = kwargs.get('eta', 0.179) # Novikov-Thorne efficiency
        # fraction of energy advected (dissipated in ions)
        self.f_ion = kwargs.get('f_adv',     0.8)
        self.gamma = kwargs.get('gamma_adi', 5./3.)
        # Calculate gravitational radius
        G       = self.CONSTANTS['G']
        Msun    = self.CONSTANTS['Msun']
        c       = self.CONSTANTS['c']
        self.Rg =  G * self.MBH * Msun / (c * c)
        # Calculate surface density and thickness
        self.Sigma = self.calc_Sigma()
        self.HoR   = self.calc_HoR()
 
    def epsilon(self):
        # Equation 2.1 epsilon prime (epsilon/ f)
        gamma_ratio = ((5./3.) - self.gamma)/ (self.gamma - 1.)
        return (1. / self.f_ion) * gamma_ratio
    
    def g_fun(self):
        eps    = self.epsilon()
        gplus1 = np.sqrt(
            1. + 18. * np.square(self.alpha) / np.square(5. + 2.* eps)
        )
        return gplus1 - 1.

    def c1(self):
        eps    = self.epsilon()
        g_fun  = self.g_fun()
        mult_g = (5. + 2. * eps) / (3. * np.square(self.alpha))
        return mult_g * g_fun
    
    def c2(self):
        eps    = self.epsilon()
        g_fun  = self.g_fun()
        mult_g = (5. + 2. * eps) / (9. * np.square(self.alpha))
        return np.sqrt(2. * eps * mult_g * g_fun)

    def c3(self):
        eps    = self.epsilon()
        g_fun  = self.g_fun()
        mult_g = (5. + 2. * eps) / (9. * np.square(self.alpha))
        return 2. * mult_g * g_fun

    def calc_HoR(self):
        c2    = self.c2()
        c3    = self.c3()
        cs2   = 9.e20  * c3 * np.power(self.R0, -1.)
        Omega = 2.03e5 * c2 * np.power(self.R0, -3./2.) * np.power(self.MBH, -1.)
        l     = np.power(self.Mdot / self.eta, 0.)
        r     = np.power(self.R0, 0.)
        return np.sqrt(2.5 * c3) * l * r

    def calc_rho(self):
        c1 = self.c1()
        c3 = self.c3()
        a  = np.power(self.alpha, -1.)
        m  = np.power(self.MBH, -1.)
        l  = np.power(self.Mdot / self.eta, 1.)
        r  = np.power(self.R0, -3./2.)
        c1_pow = np.power(c1, -1.)
        c3_pow = np.power(c3, -1./2.)
        return 1.07e-5 * a * m * l * r * c1_pow * c3_pow
    
    def calc_Sigma(self):
        rho = self.calc_rho()
        HoR = self.calc_HoR()
        R   = self.R0 * self.Rg
        H   = HoR * R
        return 2. * rho * H