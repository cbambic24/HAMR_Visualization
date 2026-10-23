import numpy as np
from analysis import Tools

# Numerical constants for performance
ONE = np.float32(1.)

class Radiation():
    def __init__(self, pars, CONSTANTS, **kwargs):
        self.cgs  = kwargs.get('cgs', False)
        self.rho0 = pars['density_scale']  if self.cgs else ONE # g/cm^3
        self.c    = CONSTANTS['c']         if self.cgs else ONE # cm/s
        self.kB   = CONSTANTS['kB']        if self.cgs else ONE # erg/K
        self.mp   = CONSTANTS['mp']        if self.cgs else ONE # g
        self.me   = CONSTANTS['me']        if self.cgs else ONE # g
        self.sigT = CONSTANTS['sigmaT']    if self.cgs else ONE # cm^2
        # Equation 10 in Sadowski+ (2017)
        self.mu_e = 2.0 / (1.0 + CONSTANTS['X_AB']) if self.cgs else ONE
        self.aRAD = CONSTANTS['aRAD']      if self.cgs else ONE # erg/cm^3/K^4

        self.CONSTANTS = CONSTANTS
        self.params    = pars

    # Compute the radiation stress-energy tensor
    # Default TRAD^{mu, nu}
    # "upper_lower": TRAD^{mu}_{nu} 
    # "lower_lower": TRAD_{mu, nu}
    def calc_TRAD(self, u_rad, uu_rad, gcov, gcon, **kwargs):
        c       = self.c
        rho0c2  = self.rho0 * self.c * self.c
        uu_tnsr = Tools.construct_dyadic(uu_rad / c, gcov, **kwargs) 
        g_tnsr  = Tools.g_metric(gcov, gcon, **kwargs)
        return ((4./3.) * (u_rad / rho0c2) * uu_tnsr + 
                (1./3.) * (u_rad / rho0c2) * g_tnsr) * rho0c2

    # Electron internal energy density, erg/cm^3.
    # gam_e is Equation 28 in Sadowski+ (2017); the electron number density
    # is n_e = rho / (mu_e m_p), with mu_e = 2/(1+X) the mean molecular
    # weight per electron in units of the *proton* mass.
    def calc_u_e(self, rho, Te):
        Theta_e = (self.kB * Te) / (self.me * self.c * self.c)
        gam_e   = (10. + 20. * Theta_e) / (6. + 15. * Theta_e)
        return (1./(gam_e-1.)) * (rho / (self.mu_e * self.mp)) * self.kB * Te

    # Compton cooling/heating time
    def calc_tIC(self, rho, Te, u_rad, **kwargs):
        u_e = self.calc_u_e(rho, Te)
        # In instances where qIC = 0, set tIC to 0.
        qIC = self.calc_qIC(rho, Te, u_rad, **kwargs)
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
    def calc_qIC(self, rho, Te, u_rad, **kwargs):
        rho0, rho0c2 = self.rho0, self.rho0 * self.c * self.c 
        kappa_es  = Opacity.calc_kappa_es(self.CONSTANTS['X_AB'])
        Theta_e   = (self.kB * Te) / (self.me * self.c * self.c)
        Trad      = (u_rad / self.aRAD) ** (1./4.)
        mc2       = self.me * self.c * self.c
        qIC_float = (
            self.c * kappa_es * (rho / rho0) * (u_rad / rho0c2) * (
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
    def calc_trad_emi(self, rho, Te, u_rad, **kwargs):
        u_e = self.calc_u_e(rho, Te)
        if kwargs.get('Compton'):
            return (u_e / (self.calc_q_emi(rho, Te, u_rad, **kwargs) + 
                self.calc_qIC(rho, Te, u_rad)))
        else:
            return u_e / self.calc_q_emi(rho, Te, u_rad, **kwargs)

    # Build the opacity model and return the total for one mechanism.
    # Shared by calc_q_emi / calc_q_abs.
    def calc_kappa_total(self, process, rho, Te, u_rad, **kwargs):
        if 'opacities' in kwargs:
            opacities = kwargs.get('opacities')
            kwargs.pop('opacities') # do not pass opacities as kwarg
        else:
            opacities = dict.fromkeys(['electron_scattering',
                'bound_free', 'CHIANTI', 'free_free', 'hydrogen',
                'molecular', 'synchrotron']) # 'iron_line' is opt-in
        if not kwargs.get('RIAF', False):
            Opac = Opacity(opacities, rho, Te, u_rad,
                           self.params, self.CONSTANTS,
                           mechanism=process, **kwargs)
        else:
            Opac = RIAF_Opacity(opacities, rho, Te, u_rad,
                                self.params, self.CONSTANTS,
                                mechanism=process, **kwargs)
        return Opac.opacities['%s_total' %process]

    # Emission rate assuming that radiation is blackbody spectrum
    # Sink term in Sadowski+ (2017) Equation 31 (term with Bhat)
    def calc_q_emi(self, rho, Te, u_rad, **kwargs):
        if (kwargs.get('cpp', False)):
            return self.calc_q_emi_cpp(rho, Te, u_rad, **kwargs)
        kappa_em = self.calc_kappa_total('emission', rho, Te, u_rad, **kwargs)
        Te_dbl   = np.float64(Te.copy())
        return kappa_em * rho * self.aRAD * (Te_dbl ** 4) * self.c

    # Electron heating time due to absorption of radiation, i.e.
    # electron-radiation temperature equilibration timescale
    # t_rad^eq = (1 / kappa rho) * u_e / (a T_rad^4 c)
    def calc_trad_abs(self, rho, Te, u_rad, **kwargs):
        u_e = self.calc_u_e(rho, Te)
        return u_e / self.calc_q_abs(rho, Te, u_rad, **kwargs)

    # Absorption rate assuming that radiation is blackbody spectrum
    # Source term in Sadowski+ (2017) Equation 31 (term with Bhat)
    def calc_q_abs(self, rho, Te, u_rad, **kwargs):
        if (kwargs.get('cpp', False)):
            return self.calc_q_abs_cpp(rho, Te, u_rad, **kwargs)
        kappa_abs = self.calc_kappa_total('absorption', rho, Te, u_rad,
                                          **kwargs)
        return kappa_abs * rho * u_rad * self.c
    
    # C++ accelerated emission/absorption rate, assuming the radiation is
    # a blackbody spectrum. Source and sink terms in Sadowski+ (2017)
    # Equation 31 (the terms with Bhat).
    def calc_q_cpp(self, process, rho, Te, u_rad, **kwargs):
        if 'b2' in kwargs:
            b2 = np.float32(kwargs.get('b2'))
        else:
            raise ValueError("""Missing keyword argument b2
                              for calculating synchrotron opacity.""")
        Tr = (u_rad / self.aRAD) ** (1./4.)
        import hamr_readc
        Radiationc = hamr_readc.PyRadiation(self.params, self.CONSTANTS)
        dtype    = np.float64 # C++ layer expects double
        [nb, nx1, nx2, nx3] = self.check_dims(rho, Te, Tr, b2)

        XRB  = 1 if kwargs.get('XRB') else 0
        RIAF = 1 if kwargs.get('RIAF', False) else 0
        q    = np.zeros(rho.shape, dtype=dtype)
        calc = (Radiationc.calc_q_emi_tot if process == 'emission'
                else Radiationc.calc_q_abs_tot)
        calc(q, rho, Te, Tr, b2, XRB, RIAF, nb, nx1, nx2, nx3)
        return q

    def calc_q_emi_cpp(self, rho, Te, u_rad, **kwargs):
        return self.calc_q_cpp('emission', rho, Te, u_rad, **kwargs)

    def calc_q_abs_cpp(self, rho, Te, u_rad, **kwargs):
        return self.calc_q_cpp('absorption', rho, Te, u_rad, **kwargs)
    
    # Necessary for C++ acceleration
    def check_dims(self, rho, Te, Tr, b2):
        return check_dims_4d(rho=rho, Te=Te, Tr=Tr, b2=b2)

class Opacity():
    # Cython wrapper holding this class's C++ implementation. Subclasses
    # override it so calc_kappa_total_cpp dispatches to their own C++ class
    # instead of always using the base one. Must be a class attribute:
    # __init__ calls calc_opacities() before a subclass __init__ body runs.
    cpp_class = 'PyOpacity'

    def __init__(self, opacities, rho, Te, u_rad, pars, CONSTANTS, **kwargs):
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
        self.Tr   = (u_rad / self.aRAD) ** (1./4.)

        self.params    = pars
        self.CONSTANTS = CONSTANTS
        self.opacities = opacities.copy()

        # All opacities are "kappa" (units of cm^2/g)
        if 'electron_scattering' in opacities:
            self.opacities['electron_scattering'] = self.calc_kappa_es(self.X_ab)
        
        if kwargs.get('mechanism') == 'absorption':
            self.calc_opacities('absorption', **kwargs)
        elif kwargs.get('mechanism') == 'emission':
            self.calc_opacities('emission', **kwargs)
        else:
            self.calc_opacities('both', **kwargs)

    # Electron-scattering (Thomson) opacity, cm^2/g.
    # A constant given composition: no density or temperature dependence,
    # and neither absorption nor emission, so it never enters calc_kappa's
    # totals -- callers combine it themselves (e.g. the effective opacity
    # sqrt(kappa_abs (kappa_abs + kappa_es)) in poloidal_slices.py).
    @staticmethod
    def calc_kappa_es(X_ab):
        return 0.2 * (1.0 + X_ab)

    def calc_opacities(self, process, **kwargs):
        # The C++ total implements bound-free + free-free + CHIANTI +
        # hydrogen + molecular + synchrotron only. Iron-line is opt-in and has
        # no C++ counterpart, so requesting it falls through to the Python
        # path below rather than being silently dropped.
        if (kwargs.get('cpp', False) and 'bound_free' in self.opacities and
            'CHIANTI' in self.opacities and 'free_free' in self.opacities and
            'hydrogen' in self.opacities and 'molecular' in self.opacities and
            'synchrotron' in self.opacities and
            'iron_line' not in self.opacities):
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
            'molecular' in self.opacities or 'synchrotron' in self.opacities or
            'iron_line' in self.opacities):
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
            self.opacities['synchrotron_abs'] = self.calc_kappa_sync(g, phi, **kwargs)
        if process == 'emission' or process == 'both':
            # McKinney+ (2017) Equation D5
            phi = (self.kB * self.Te / (self.h * nu_mu)).astype(np.float64)
            # McKinney+ (2017) Equation D6
            g   = 5.85374e-14 * ne * phi * (Theta_e ** -3) * (1. / self.Te)
            self.opacities['synchrotron_emi'] = self.calc_kappa_sync(g, phi, **kwargs)
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
        proc = 'abs' if process == 'absorption' else 'emi'

        # Low-temperature channel: rises steeply with T (kappa_H ~ T^7.7)
        # and carries no radiation-temperature dependence.
        lowT_keys  = [k for k in ('hydrogen', 'molecular')
                      if k in self.opacities]
        # High-temperature channel: falls with T and scales as Tr^-3.
        # 'iron_line' is opt-in (a bound-bound line opacity, like CHIANTI).
        highT_keys = [k for k in ('bound_free', 'CHIANTI', 'free_free',
                                  'iron_line') if k in self.opacities]

        kappa_sync = (self.opacities['synchrotron_%s' %proc]
                      if 'synchrotron' in self.opacities else 0.)

        if not lowT_keys and not highT_keys:
            self.opacities['%s_total' %process] = kappa_sync
            return None

        # Harmonic sum: 1/(1/a + 1/b) ~ min(a, b), so whichever channel is
        # smaller dominates. That is what bridges the low-T and high-T
        # opacity regimes. A channel with no requested contributors is
        # omitted rather than treated as zero opacity. Mirrors
        # Opacity::kappaAbsTotal in hamr_analysis.cpp.
        inv_kappa = 0.
        if lowT_keys:
            kappa_lowT  = self.sum_opacities(['%s_%s' %(k, proc)
                                              for k in lowT_keys])
            inv_kappa   = inv_kappa + 1. / kappa_lowT
        if highT_keys:
            kappa_highT = self.sum_opacities(['%s_%s' %(k, proc)
                                              for k in highT_keys])
            inv_kappa   = inv_kappa + 1. / kappa_highT

        self.opacities['%s_total' %process] = 1. / inv_kappa + kappa_sync
        return None
            
    def calc_kappa_total_cpp(self, process, **kwargs):
        if 'b2' in kwargs:
            self.b2 = kwargs.get('b2')
        else:
            raise ValueError("""Missing keyword argument b2
                              for calculating synchrotron opacity.""") 
        import hamr_readc
        Opacityc = getattr(hamr_readc, self.cpp_class)(self.params,
                                                      self.CONSTANTS)
        dtype    = np.float64 # C++ layer expects double
        [nb, nx1, nx2, nx3] = self.check_dims()

        XRB = 1 if kwargs.get('XRB') else 0
        if process == 'absorption' or process == 'both':
            proc = 'absorption'
            kappa_abs = np.zeros(self.rho.shape, dtype=dtype)
            Opacityc.calc_kappa_abs_tot(kappa_abs, self.rho, self.Te, self.Tr, 
                                     np.float32(self.b2), XRB, nb, nx1, nx2, nx3)
            self.opacities['%s_total' %proc] = kappa_abs
        if process == 'emission' or process == 'both':
            proc = 'emission'
            kappa_emi = np.zeros(self.rho.shape, dtype=dtype)
            Opacityc.calc_kappa_emi_tot(kappa_emi, self.rho, self.Te, self.Tr, 
                                     np.float32(self.b2), XRB, nb, nx1, nx2, nx3)
            self.opacities['%s_total' %proc] = kappa_emi

    def sum_opacities(self, keys):
        total = self.opacities['%s' %keys[0]]
        if len(keys) == 1:
            return total
        # Copy before accumulating -- never mutate the stored per-channel
        # entry. hydrogen_abs and hydrogen_emi are the *same* array object,
        # so an in-place sum corrupted both and made the emission pass
        # double-count kappa_mol. One allocation; accumulation stays
        # in-place from here.
        total = np.array(total, copy=True)
        for key in keys[1:]:
            total += self.opacities['%s' %key]
        return total

    def check_dims(self):
        return check_dims_4d(rho=self.rho, Te=self.Te,
                             Tr=self.Tr,   b2=self.b2)

class RIAF_Opacity(Opacity):
    cpp_class = 'PyRIAF_Opacity'

    def __init__(self, opacities, rho, Te, u_rad, pars, CONSTANTS, **kwargs):
        super().__init__(opacities, rho, Te, u_rad, pars, CONSTANTS, **kwargs)
    
    # Synchrotron opacity
    # Sadowski+ (2017) Equations 41-43
    # Appropriate or low-Mdot radiatively inefficient accretion flows
    def synchrotron_opacity(self, process, **kwargs):
        if kwargs.get('cpp', False):
            return self.synchrotron_opacity_cpp(process, **kwargs)
        ne        = self.rho / (self.mu_e * self.mp)
        xi        = self.Tr / self.Te
        kappa_emi = 1.59e-30 * ne * self.b2 * self.Te ** (-2.) / self.rho
        if process == 'absorption' or process == 'both':
            # Sadowski+ (2017) Equation 42
            pime3c5_ekh = (np.pi * self.me ** 3. * self.c ** 5. / 
                (self.e * self.kB * self.h))
            ksi = (4./3.) * pime3c5_ekh * (
                self.Tr / (np.sqrt(self.b2) * self.Te ** 2.))
            # Sadowski+ (2017) Equation 43
            self.opacities['synchrotron_abs'] = kappa_emi * xi ** (-3.) * (
                1. / (1. + 5.444*(ksi**(-2./3.)) + 7.218*(ksi**(-4./3.)))
            )
        if process == 'emission' or process == 'both':
            self.opacities['synchrotron_emi'] = kappa_emi 
        return None
    
    # Perform synchrotron opacity calculations in C++
    def synchrotron_opacity_cpp(self, process, **kwargs):
        import hamr_readc
        Opacityc = hamr_readc.PyRIAF_Opacity(self.params, self.CONSTANTS)
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


# The C++ layer walks every array with a single linear index, so the shapes
# must match exactly -- not merely share a number of axes.
def check_dims_4d(**arrays):
    shapes = {name: np.shape(a) for name, a in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError("Arrays passed to C++ must all have the same "
                         f"shape; got {shapes}.")
    dims = list(next(iter(shapes.values())))
    if len(dims) != 4:
        raise ValueError("Arrays passed to C++ must be 4-dimensional; "
                         f"got {len(dims)} ({tuple(dims)}).")
    return dims
