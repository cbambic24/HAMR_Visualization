import numpy as np

ONE = np.float32(1.)

class Analytic():
    def __init__(self, CONSTANTS, **kwargs):
        self.CONSTANTS = CONSTANTS
        self.cgs = kwargs.get('cgs', True)
        self.c   = CONSTANTS['c'] if self.cgs else ONE # cm/s

    # Calculate Eddington luminosity for black hole of mass MBH (solar masses)
    def calc_LEdd(self, MBH):
        return 1.26e38 * MBH # erg/s

    # Calculate Eddington accretion rate for black hole of mass MBH
    def calc_Mdot_Edd(self, MBH, **kwargs):
        eta = kwargs.get('eta', 0.1)
        return self.calc_LEdd(MBH) / (eta * self.c * self.c)

    # Calculate radius of (outer) event horizon for Kerr black hole 
    # with spin parameter a
    def calc_R_Horizon(self, a):
        return 1. + np.sqrt(1 - a*a)
    
    # Radius of the innermost stable circular orbit (R_ISCO / R_g)
    # Eq. 5.4.8a in Novikov & Thorne (1973)
    # Sign correction shown in Eq. 15k of Page & Thorne (1974)
    def calc_R_ISCO(self, a):
        z1 = self.calc_Z1(a)
        z2 = self.calc_Z2(a)
        return (3. + z2 - 
            np.sign(a) * ((3. - z1) * (3. + z1 + 2. * z2))**(1./2.))
    
    # Keplerian orbital frequency
    # Eq. 36 of Abramowicz & Fragile (2013)
    def calc_OmegaK(self, MBH, a, r_Rg):
        MBH  = MBH                                if self.cgs else ONE
        G    = self.CONSTANTS['G']                if self.cgs else ONE
        Msun = self.CONSTANTS['Msun']             if self.cgs else ONE
        Rg   = G * MBH * Msun / (self.c * self.c) if self.cgs else ONE
        return np.sqrt(G * MBH * Msun / Rg ** 3) * (
            1. / (r_Rg ** (3./2.) + a))

    # Eq. 5.4.8b in Novikov & Thorne (1973)
    def calc_Z1(self, a):
        return (1. + (1. - a * a)**(1./3.) * (
            (1. + a)**(1./3.) + (1. - a)**(1./3.)))
    
    # Eq. 5.4.8b in Novikov & Thorne (1973)
    def calc_Z2(self, a):
        return (3. * a * a + (self.calc_Z1(a))**2.)**(1./2.)
    
    # Calculate Novikov & Thorne (1973) (with corrections from
    # Page & Thorne 1974) efficiency of thin-disk accretion eta
    # Expects dimensionless radius (units of R_g)
    def calc_efficiency_NT73(self, r, a):
        deta_dr = self.calc_dissipation_NT73(r, a)
        dr      = np.append(np.diff(r), np.diff(r)[-1])
        eta     = np.zeros(len(r))
        eta     = np.cumsum(deta_dr * dr)
        return eta

    # Calculate Novikov & Thorne (1973) local dissipation rate
    def calc_dissipation_NT73(self, r, a):
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

