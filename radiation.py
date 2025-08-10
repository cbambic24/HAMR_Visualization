import numpy as np
from analysis import Tools
import time as get_time

class Radiation():
    def __init__(self, pars, CONSTANTS, **kwargs):
        self.cgs = kwargs.get('cgs', False)
        if (self.cgs):
            self.rho0 = pars['density_scale'] # g/cm^3 
            self.c    = CONSTANTS['c']        # cm/s
            self.kB   = CONSTANTS['kB']       # erg/K
            self.me   = CONSTANTS['me']       # g
            self.sigT = CONSTANTS['sigmaT']   # cm^2
            # Equation 10 in Sadowski+ (2017)
            self.mu_e = 2.0 / (1.0 + CONSTANTS['X_AB'])
            self.aRAD = CONSTANTS['aRAD'] # erg/cm^3/K^4
        else:
            self.rho0 = 1.0
            self.c    = 1.0
            self.kB   = 1.0
            self.me   = 1.0
            self.sigT = 1.0
            self.mu_e = 1.0
            self.aRAD = 1.0

        self.kwargs    = kwargs
        self.CONSTANTS = CONSTANTS
        self.params    = pars

    # Compute the radiation stress-energy tensor
    # Default TRAD^{mu, nu}
    # "upper_lower": TRAD^{mu}_{nu} 
    # "lower_lower": TRAD_{mu, nu}
    def calc_TRAD(self, urad, uu_rad, gcov, gcon, **kwargs):
        c       = self.c
        rho0c2  = self.rho0 * self.c * self.c
        uu_tnsr = Tools.construct_dyadic(uu_rad / c, gcov, **kwargs) 
        g_tnsr  = Tools.g_metric(gcov, gcon, **kwargs)
        return ((4./3.) * (urad / rho0c2) * uu_tnsr + 
                (1./3.) * (urad / rho0c2) * g_tnsr) * rho0c2

    # Compton cooling/heating time
    def calc_tIC(self, rho, Te, urad, **kwargs):
        # Calculate gamma (ratio of specific heats)
        # Equation 28 in Sadowski+ (2017)
        Theta_e = (self.kB * Te) / (self.me * self.c * self.c)
        gam_e   = (10. + 20. * Theta_e) / (6. + 15. * Theta_e)
        # Electron internal energy
        u_e   = (1./(gam_e-1.)) * (rho / (self.mu_e * self.me)) * self.kB * Te
        # In instances where qIC = 0, set tIC to 0.
        qIC = self.calc_qIC(rho, Te, urad, **kwargs)
        tIC = np.zeros(qIC.shape)
        tIC[qIC != 0.] = u_e[qIC != 0.] / qIC[qIC != 0.]
        if 'cooling' in kwargs:
            tIC[tIC < 0.0] = 0.0
            return tIC
        elif 'heating' in kwargs:
            tIC[tIC > 0.0] = 0.0
            return -1.0 * tIC
        else:
            return tIC

    # Compton cooling rate
    # Equation 35 in Sadowski+ (2017)
    def calc_qIC(self, rho, Te, urad, **kwargs):
        rho0, rho0c2 = self.rho0, self.rho0 * self.c * self.c 
        opacities = dict.fromkeys(['electron_scattering'])
        Opac      = Opacity(opacities, rho, Te, urad, 
                            self.params, self.CONSTANTS)
        kappa_es  = Opac.opacities['electron_scattering']
        Theta_e   = (self.kB * Te) / (self.me * self.c * self.c)
        Trad      = (urad / self.aRAD) ** (1./4.)
        mc2       = self.me * self.c * self.c
        qIC_float = (
            self.c * kappa_es * (rho / rho0) * (urad / rho0c2) * (
            4. * self.kB * (Te - Trad) / mc2) * (
            1. + 3.683 * Theta_e + 4. * np.square(Theta_e)) * (
            1. / (1. + Theta_e))) 
        qIC = rho0 * rho0c2 * np.float64(qIC_float) # erg/cm^3 s
        if kwargs.get('cooling'):
            qIC[Te < Trad] = 0.0
            return qIC # erg/cm^3 s
        elif kwargs.get('heating'):
            qIC[Te > Trad] = 0.0
            return qIC # erg/cm^3 s
        else:
            return qIC # erg/cm^3 s
    
    # Electron cooling time due to emission of radiation, i.e.
    # electron-radiation temperature equilibration timescale
    # t_rad^eq = (1 / kappa rho) * u_e / (a Te^4 c + q_IC) if Compton = True
    def calc_trad_emi(self, rho, Te, urad, **kwargs):
        # Calculate gamma (ratio of specific heats)
        # Equation 28 in Sadowski+ (2017)
        Theta_e = (self.kB * Te) / (self.me * self.c * self.c)
        gam_e   = (10. + 20. * Theta_e) / (6. + 15. * Theta_e)
        # Electron internal energy
        u_e   = (1./(gam_e-1.)) * (rho / (self.mu_e * self.me)) * self.kB * Te
        if kwargs.get('Compton'):
            return (u_e / (self.calc_q_emi(rho, Te, urad, **kwargs) + 
                self.q_IC(rho, Te, urad)))
        else:
            return u_e / self.calc_q_emi(rho, Te, urad, **kwargs)

    # Emission rate assuming that radiation is blackbody spectrum
    # Sink term in Sadowski+ (2017) Equation 31 (term with Bhat)
    def calc_q_emi(self, rho, Te, urad, **kwargs):
        if (kwargs.get('cpp', False)):
            return self.calc_q_emi_cpp(rho, Te, urad, **kwargs)
        # Calculate total emission opacity
        if 'opacities' in kwargs:
            opacities = kwargs.get('opacities')
            kwargs.pop('opacities') # do not pass opacities as kwarg
        else:
            opacities = dict.fromkeys(['electron_scattering',
                'bound_free', 'CHIANTI', 'free_free', 'hydrogen',
                'molecular', 'synchrotron']) # 'iron_line' not used
        Opac = Opacity(opacities, rho, Te, urad, 
                       self.params, self.CONSTANTS, 
                       mechansim='emission', **kwargs)
        kappa_em = Opac.opacities['emission_total']
        Te_dbl   = np.float64(Te.copy())
        return kappa_em * rho * self.aRAD * (Te_dbl ** 4) * self.c

    # Electron heating time due to absorption of radiation, i.e.
    # electron-radiation temperature equilibration timescale
    # t_rad^eq = (1 / kappa rho) * u_e / (a T_rad^4 c)
    def calc_trad_abs(self, rho, Te, urad, **kwargs):
        # Calculate gamma (ratio of specific heats)
        # Equation 28 in Sadowski+ (2017)
        Theta_e = (self.kB * Te) / (self.me * self.c * self.c)
        gam_e   = (10. + 20. * Theta_e) / (6. + 15. * Theta_e)
        # Electron internal energy
        u_e   = (1./(gam_e-1.)) * (rho / (self.mu_e * self.me)) * self.kB * Te
        return u_e / self.calc_q_abs(rho, Te, urad, **kwargs)

    # Absorption rate assuming that radiation is blackbody spectrum
    # Source term in Sadowski+ (2017) Equation 31 (term with Bhat)
    def calc_q_abs(self, rho, Te, urad, **kwargs):
        if (kwargs.get('cpp', False)):
            return self.calc_q_abs_cpp(rho, Te, urad, **kwargs)
        # Calculate total absorption opacity
        if 'opacities' in kwargs:
            opacities = kwargs.get('opacities')
            kwargs.pop('opacities') # do not pass opacities as kwarg
        else:
            opacities = dict.fromkeys(['electron_scattering',
                'bound_free', 'CHIANTI', 'free_free', 'hydrogen',
                'molecular', 'synchrotron']) # 'iron_line' not used
        Opac = Opacity(opacities, rho, Te, urad, 
                       self.params, self.CONSTANTS, 
                       mechansim='absorption', **kwargs)
        kappa_abs = Opac.opacities['absorption_total']
        return kappa_abs * rho * urad * self.c
    
    # C++ accelerated
    # Emission rate assuming that radiation is blackbody spectrum
    # Sink term in Sadowski+ (2017) Equation 31 (term with Bhat)
    def calc_q_emi_cpp(self, rho, Te, urad, **kwargs):
        if 'b2' in kwargs:
            b2 = np.float32(kwargs.get('b2'))
        else:
            raise ValueError("""Missing keyword argument b2
                              for calculating synchrotron opacity.""") 
        Tr = (urad / self.aRAD) ** (1./4.)
        import hamr_readc
        Radiationc = hamr_readc.PyRadiation(self.params, self.CONSTANTS)
        dtype    = np.float64 # C++ layer expects double
        [nb, nx1, nx2, nx3] = self.check_dims(rho, Te, Tr, b2)
        
        XRB  = 1 if kwargs.get('XRB') else 0
        qemi = np.zeros(rho.shape, dtype=dtype)
        Radiationc.calc_q_emi_tot(qemi, rho, Te, Tr, b2, 
            XRB, nb, nx1, nx2, nx3)
        return qemi
            
    # C++ accelerated 
    # Absorption rate assuming that radiation is blackbody spectrum
    # Source term in Sadowski+ (2017) Equation 31 (term with Bhat)
    def calc_q_abs_cpp(self, rho, Te, urad, **kwargs):
        if 'b2' in kwargs:
            b2 = np.float32(kwargs.get('b2'))
        else:
            raise ValueError("""Missing keyword argument b2
                              for calculating synchrotron opacity.""") 
        Tr = (urad / self.aRAD) ** (1./4.)
        import hamr_readc
        Radiationc = hamr_readc.PyRadiation(self.params, self.CONSTANTS)
        dtype    = np.float64 # C++ layer expects double
        [nb, nx1, nx2, nx3] = self.check_dims(rho, Te, Tr, b2)
        XRB  = 1 if kwargs.get('XRB') else 0
        qabs = np.zeros(rho.shape, dtype=dtype)
        Radiationc.calc_q_abs_tot(qabs, rho, Te, Tr, b2, 
            XRB, nb, nx1, nx2, nx3)
        return qabs
    
    # Necessary for C++ acceleration
    def check_dims(self, rho, Te, Tr, b2):
        rho_dims = list(rho.shape)
        Te_dims  = list(Te.shape)
        Tr_dims  = list(Tr.shape)
        b2_dims  = list(b2.shape)

        if (len(rho_dims) != len(Te_dims) != len(Tr_dims) != len(b2_dims)):
            raise ValueError("""Your arrays must have the same dimensions 
                                to be passed to C++.""")
        if (len(rho_dims) != 4):
            raise ValueError("""Your arrays must have dimension 4
                                to be passed to C++""")
        return rho_dims

class Opacity():
    def __init__(self, opacities, rho, Te, urad, pars, CONSTANTS, **kwargs):
        self.X_ab = CONSTANTS['X_AB']
        self.Y_ab = CONSTANTS['Y_AB']
        self.Z_ab = CONSTANTS['Z_AB']

        self.c    = CONSTANTS['c']   # cm/s
        self.kB   = CONSTANTS['kB']  # erg/K
        self.e    = CONSTANTS['e']   # statC
        self.h    = CONSTANTS['h']   # erg s
        # Sadowski+ (2017) Equation 10
        self.mu_e = 2.0 / (1.0 + self.X_ab)
        self.mp   = CONSTANTS['mp']  # g
        self.me   = CONSTANTS['me']  # g
        self.aRAD = CONSTANTS['aRAD'] # erg/cm^3/K^4

        self.rho  = rho
        self.Te   = Te
        self.Tr   = (urad / self.aRAD) ** (1./4.)

        self.params    = pars
        self.CONSTANTS = CONSTANTS
        self.opacities = opacities.copy()

        # Electron-scattering (Thomson) opacity
        kappa_es = 0.2 * (1 + self.X_ab) # cm^2/g

        # All opacities are "kappa" (units of cm^2/g)
        if 'electron_scattering' in opacities:
            self.opacities['electron_scattering'] = kappa_es
        
        if kwargs.get('mechanism') == 'absorption':
            self.calc_opacities('absorption', **kwargs)
        elif kwargs.get('mechanism') == 'emission':
            self.calc_opacities('emission', **kwargs)
        else:
            self.calc_opacities('both', **kwargs)

    def calc_opacities(self, process, **kwargs):
        if (kwargs.get('cpp', False) and 'bound_free' in self.opacities and
            'CHIANTI' in self.opacities and 'free_free' in self.opacities and
            'hydrogen' in self.opacities and 'molecular' in self.opacities and
            'synchrotron' in self.opacities):
            return self.calc_kappa_total_cpp(process, **kwargs)
        if 'bound_free' in self.opacities:
            self.bound_free_opacity(process)
        if 'CHIANTI' in self.opacities:
            self.CHIANTI_opacity(process)
        if 'free_free' in self.opacities:
            self.free_free_opacity(process)
        if 'hydrogen' in self.opacities:
            self.hydrogen_opacity(process)
        if 'iron_line' in self.opacities:
            self.iron_line_opacity(process)
        if 'molecular' in self.opacities:
            self.molecular_opacity(process)
        if 'synchrotron' in self.opacities:
            if 'b2' in kwargs:
                self.b2 = kwargs.get('b2')
            else:
                raise ValueError("""Missing keyword argument b2
                                  for calculating synchrotron opacity.""")
            self.synchrotron_opacity(process, **kwargs)
        if ('bound_free' in self.opacities or 'CHIANTI'  in self.opacities or 
            'hydrogen'  in self.opacities or 'free_free' in self.opacities or
            'molecular' in self.opacities or 'synchrotron' in self.opacities):
            if process == 'absorption' or process == 'both':
                self.calc_kappa('absorption')
            if process == 'emission' or process == 'both':
                self.calc_kappa('emission')

    # Bound-free opacity
    # Replace (1+X)(1-Z) in the free-free opacity with
    # 750 Z (1 + X + 0.75Y) as per Rybicki & Lightman (1986) and
    # the discussion in section C4 of McKinney+ (2017)
    def bound_free_opacity(self, process):
        if process == 'absorption' or process == 'both':
            self.opacities['bound_free_abs'] = (9.0e26 * (
                    self.Z_ab * (1. + self.X_ab + 0.75 * self.Y_ab) * (
                    self.rho * (self.Te ** -0.5) * (self.Tr ** -3) * (
                    np.log(1. + 1.6 * (self.Tr / self.Te))))))
        if process == 'emission' or process == 'both':
            self.opacities['bound_free_emi'] = (9.0e26 * (
                    self.Z_ab * (1. + self.X_ab + 0.75 * self.Y_ab) * (
                    self.rho * (self.Te ** -3.5) * np.log(1. + 1.6))))
        return None

    # Opacities based on calculations from the CHIANTI atomic database
    # McKinney+ (2017) Equation C11
    def CHIANTI_opacity(self, process):
        if process == 'absorption' or process == 'both':
            self.opacities['CHIANTI_abs'] = (3.0e34 * self.rho * (
                    (0.1 + (self.Z_ab / 0.02)) * (
                    (self.Te ** -1.7) * (self.Tr ** -3.0))))
        if process == 'emission' or process == 'both':
            self.opacities['CHIANTI_emi'] = (3.0e34 * self.rho * (
                    (0.1 + (self.Z_ab / 0.02)) * (self.Te ** -4.7)))
        return None

    # Free-free opacity
    # Close to the same form as Sadowski+ (2017) Equations 36-38
    # Sadowski prefactor is 1.33e24 instead of 1.2e24 and Sadowski includes
    # a factor of 1.047 in the absorption opacity
    def free_free_opacity(self, process):
        if process == 'absorption' or process == 'both':
            self.opacities['free_free_abs'] = (1.2e24 * (
                    (1. + self.X_ab) * (1. - self.Z_ab) * self.rho * (
                    (self.Te ** -0.5) * (self.Tr ** -3) * (
                    np.log(1.0 + 1.6 * (self.Tr / self.Te)) * (
                    (1. + 4.4e-10 * self.Te))))))
        if process == 'emission' or process == 'both':
            self.opacities['free_free_emi'] = (1.2e24 * (
                    (1. + self.X_ab) * (1. - self.Z_ab) * self.rho * (
                    (self.Te ** -3.5) * np.log(1.0 + 1.6) * (
                    (1. + 4.4e-10 * self.Te)))))
        return None
    
    # H- opacity 
    # McKinney+ (2017) Equation C12
    def hydrogen_opacity(self, process):
        # Avoid overflow warnings
        Te      = np.float64(self.Te.copy())
        kappa_H = np.float64(33. * 1.e-25 * (
            np.sqrt(self.Z_ab) * np.sqrt(self.rho) * (
            Te ** 7.7)))
        del Te
        if process == 'absorption' or process == 'both':
            self.opacities['hydrogen_abs'] = kappa_H
        if process == 'emission' or process == 'both':
            self.opacities['hydrogen_emi'] = kappa_H
        return None

    # Iron-line opacity
    # McKinney+ (2017) Equation C15
    def iron_line_opacity(self, process):
        if process == 'absorption' or process == 'both':
            self.opacities['iron_line_abs'] = (0.3 * (self.Z_ab / 0.02) * (
                    np.exp(-6.0 * ((-12.0 + np.log(self.Te)) ** 2)))) 
        if process == 'emission' or process == 'both':
            self.opacities['iron_line_emi'] = (0.3 * (self.Z_ab / 0.02) * (
                    np.exp(-6.0 * ((-12.0 + np.log(self.Te)) ** 2))))
        return None

    # Molecular opacity
    # McKinney+ (2017) sentence after Equation C12
    def molecular_opacity(self, process):
        if process == 'absorption' or process == 'both':
            self.opacities['molecular_abs'] = 3. * self.Z_ab
        if process == 'emission' or process == 'both':
            self.opacities['molecular_emi'] = 3. * self.Z_ab
        return None
    
    # Synchrotron opacity
    # McKinney+ (2017) Equations D2, D5, D6, and D7
    def synchrotron_opacity(self, process, **kwargs):
        if kwargs.get('cpp', False):
            return self.synchrotron_opacity_cpp(process, **kwargs)
        ne       = self.rho / (self.mu_e * self.mp)
        Theta_e  = (self.kB * self.Te) / (self.me * self.c * self.c)
        e_2pimc  = self.e / (2. * np.pi * self.me * self.c)
        # McKinney+ (2017) Equation D2
        nu_mu    = (3./2.) * e_2pimc * np.sqrt(self.b2) * (Theta_e ** 2)
        if process == 'absorption' or process == 'both':
            # McKinney+ (2017) Equation D5
            phi = (self.kB * self.Tr / (self.h * nu_mu)).astype(np.float64)
            # McKinney+ (2017) Equation D6
            g   = 5.85374e-14 * ne * phi * (Theta_e ** -3) * (1. / self.Tr)
            self.opacities['synchrotron_abs'] = self.calc_kappa_sync(
                                                            g, phi, **kwargs)
        if process == 'emission' or process == 'both':
            # McKinney+ (2017) Equation D5
            phi = (self.kB * self.Te / (self.h * nu_mu)).astype(np.float64)
            # McKinney+ (2017) Equation D6
            g   = 5.85374e-14 * ne * phi * (Theta_e ** -3) * (1. / self.Te)
            self.opacities['synchrotron_emi'] = self.calc_kappa_sync(
                                                            g, phi, **kwargs)
        return None

    # Calculate synchrotron kappa from McKinney+ (2017) Equation D7
    def calc_kappa_sync(self, g, phi, **kwargs):
        a, b, c, d , e = self.get_abcde_synchrotron(**kwargs)
        kap_g_inv = (1. / (a * (phi ** -b) * np.log(1. + c * phi)) + (
                     1. / (d * (phi ** -e))))
        # remember that kappa = kappa_McKinney / rho
        return (g * (1.0 / kap_g_inv)) / self.rho 

    # Calculate a, b, c, d, e coefficients used in 
    # McKinney+ (2017) Equation D7
    # with the assumption that exp(-Trad/Te) = 1
    # (see pages 2262-2263)
    def get_abcde_synchrotron(self, **kwargs):
        if kwargs.get('XRB'):
            a, b, c, d, e = 1.27, 1.03, 0.000763, 0.616, 2.91
        elif kwargs.get('AGN'):
            a, b, c, d, e = 0.206, 2.59, 3.44, 9.33, 3.09
        else:
            raise ValueError("""Must specify XRB or AGN to use 
                              synchrotron opacities.""")
        return a, b, c, d, e

    # Perform synchrotron opacity calculations in C++
    def synchrotron_opacity_cpp(self, process, **kwargs):
        import hamr_readc
        Opacityc = hamr_readc.PyOpacity(self.params, self.CONSTANTS)
        dtype    = np.float64 # C++ layer expects double
        [nb, nx1, nx2, nx3] = self.check_dims()
        
        XRB = 1 if kwargs.get('XRB') else 0
        if process == 'absorption' or process == 'both':
            kappa_sync = np.zeros(self.rho.shape, dtype=dtype)
            Opacityc.calc_kappa_sync(kappa_sync, self.rho, self.Te, self.Tr, 
                                     self.b2, 0, XRB, nb, nx1, nx2, nx3)
            self.opacities['synchrotron_abs'] = kappa_sync
        if process == 'emission' or process == 'both':
            kappa_sync = np.zeros(self.rho.shape, dtype=dtype)
            Opacityc.calc_kappa_sync(kappa_sync, self.rho, self.Te, self.Tr, 
                                     self.b2, 1, XRB, nb, nx1, nx2, nx3)
            self.opacities['synchrotron_emi'] = kappa_sync

    # Calculate total opacity
    # This is a generalized way to write
    # kappa_tot = 1 / (
    #     (1 / (hydrogen + molecular)) + 
    #     (1 / (bound_free + ff + CHIANT)) ) + synchrotron
    def calc_kappa(self, process):
        if process == 'absorption':
            proc = 'abs'
        if process == 'emission':
            proc = 'emi'

        synchrotron           = 0
        hydrogen_or_molecular = 0
        bf_or_CHI_or_ff       = 0
        if 'synchrotron' in self.opacities:
            synchrotron = 1.0
            kappa_synchrotron = self.opacities['synchrotron_%s' %proc]
        if 'hydrogen' in self.opacities or 'molecular' in self.opacities:
            hydrogen_or_molecular = 1.0
            sum_keys = []
            if 'hydrogen' in self.opacities:
                sum_keys.append('hydrogen_%s' %proc)
            if 'molecular' in self.opacities:
                sum_keys.append('molecular_%s' %proc)
            kappa_H_Mol = self.sum_opacities(sum_keys)
        if ('bound_free' in self.opacities or 
            'CHIANTI'    in self.opacities or
            'free_free'  in self.opacities):
            bf_or_CHI_or_ff = 1.0
            sum_keys = []
            if 'bound_free' in self.opacities:
                sum_keys.append('bound_free_%s' %proc)
            if 'CHIANTI' in self.opacities:
                sum_keys.append('CHIANTI_%s' %proc)
            if 'free_free' in self.opacities:
                sum_keys.append('free_free_%s' %proc)
            kappa_bf_CHI_ff = self.sum_opacities(sum_keys)

        if ('bound_free' in self.opacities or 'CHIANTI'   in self.opacities or 
            'hydrogen'   in self.opacities or 'free_free' in self.opacities or
            'molecular'  in self.opacities):
            self.opacities['%s_total' %process] = (
                    (1. / (hydrogen_or_molecular * (1. / kappa_H_Mol) + (
                    bf_or_CHI_or_ff * (1. / kappa_bf_CHI_ff)))) + (
                    synchrotron * kappa_synchrotron))
        else:
            self.opacities['%s_total' %process] = kappa_synchrotron
        return None
            
    def calc_kappa_total_cpp(self, process, **kwargs):
        if 'b2' in kwargs:
            self.b2 = kwargs.get('b2')
        else:
            raise ValueError("""Missing keyword argument b2
                              for calculating synchrotron opacity.""") 
        import hamr_readc
        Opacityc = hamr_readc.PyOpacity(self.params, self.CONSTANTS)
        dtype    = np.float64 # C++ layer expects double
        [nb, nx1, nx2, nx3] = self.check_dims()
        
        XRB = 1 if kwargs.get('XRB') else 0
        if process == 'absorption' or process == 'both':
            proc = 'absorption'
            kappa_abs = np.zeros(self.rho.shape, dtype=dtype)
            Opacityc.calc_kappa_abs_tot(kappa_abs, self.rho, self.Te, self.Tr, 
                                     self.b2, XRB, nb, nx1, nx2, nx3)
            self.opacities['%s_total' %proc] = kappa_abs
        if process == 'emission' or process == 'both':
            proc = 'emission'
            kappa_emi = np.zeros(self.rho.shape, dtype=dtype)
            Opacityc.calc_kappa_emi_tot(kappa_emi, self.rho, self.Te, self.Tr, 
                                     self.b2, XRB, nb, nx1, nx2, nx3)
            self.opacities['%s_total' %proc] = kappa_emi

    def sum_opacities(self, keys):
        summed_opacity = self.opacities['%s' %keys[0]]
        if len(keys) > 1:
            for key in keys[1:]:
                summed_opacity += self.opacities['%s' %key]
        return summed_opacity

    def check_dims(self):
        rho_dims = list(self.rho.shape)
        Te_dims  = list(self.Te.shape)
        Tr_dims  = list(self.Tr.shape)
        b2_dims  = list(self.b2.shape)

        if (len(rho_dims) != len(Te_dims) != len(Tr_dims) != len(b2_dims)):
            raise ValueError("""Your arrays must have the same dimensions 
                                to be passed to C++.""")
        if (len(rho_dims) != 4):
            raise ValueError("""Your arrays must have dimension 4
                                to be passed to C++""")
        return rho_dims
            
