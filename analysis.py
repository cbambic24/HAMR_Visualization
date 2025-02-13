import numpy as np
import tracemalloc

class Accretion():
    def __init__(self, grid, data, pars, CONSTANTS, **kwargs):
        self.dx1, self.dx2, self.dx3 = pars['dx1'], pars['dx2'], pars['dx3']
        self.gdet                    = grid['gdet']
        self.r                       = grid['r']
        self.gcov, self.gcon         = grid['gcov'], grid['gcon']
        self.rho, self.uu, self.ugas = data['rho'], data['uu'], data['u_gas']
        self.B                       = data['B']
        self.cgs                     = kwargs.get('cgs', False)
        if (self.cgs):
            self.rho0 = pars['density_scale'] # g/cm^3 
            self.c    = CONSTANTS['c']        # cm/s
            self.B0   = np.sqrt(4 * np.pi * self.rho0) * self.c
        else:
            self.rho0 = 1.0
            self.c    = 1.0
            self.B0   = 1.0
        self.pars      = pars
        self.CONSTANTS = CONSTANTS
        self.kwargs    = kwargs

    # Calculate disk mass
    def calc_Mtot(self, **kwargs):
        return np.sum((self.rho * self.uu[0]) * (
                        self.sigma_cut(**kwargs) * (
                        self.gdet * self.dx1 * self.dx2 * self.dx3)))
    
    # Calculate inward mass accretion rate as function of radius
    def calc_Mdot(self, **kwargs):
        return (-self.rho * self.uu[1] * (
            self.sigma_cut(**kwargs) * (
            self.gdet * self.dx2 * self.dx3))).sum(-1).sum(-1)
    
    # Calculate energy accretion rate as a function of radius
    def calc_Edot(self, **kwargs):
        # WHY isn't this negative? It's an energy flux, right???
        stress_energy_MHD = self.calc_TMHD(index = [1, 0], **kwargs)
        return (stress_energy_MHD * (
            self.sigma_cut(**kwargs) * (
            self.gdet * self.dx2 * self.dx3))).sum(-1).sum(-1)

    # Calculate unsigned radial magnetic flux (Phi_BH) as a function of radius
    def calc_PhiBH(self, **kwargs):
        return (0.5 * np.abs(self.B[1]) * (
            self.gdet * self.dx2 * self.dx3)).sum(-1).sum(-1)

    # Calculate full MHD stress-energy tensor
    def calc_TMHD(self, **kwargs):
        Flu = Fluid(self.pars, self.CONSTANTS, cgs=self.kwargs.get('cgs'))
        Fld = Field(self.pars, self.CONSTANTS, cgs=self.kwargs.get('cgs'))
        if 'pgas' in kwargs:
            pgas = kwargs.get('pgas')
        else:
            gamma_adi = 5./3.
            pgas = (gamma_adi - 1.) * self.ugas
        stress_energy_HD = Flu.calc_THD(self.rho, self.uu, 
                                        self.ugas, pgas, self.gcon,
                                        EOS=self.kwargs.get('EOS'), **kwargs)
        stress_energy_EM = Fld.calc_TEM(self.uu, self.B, self.gcov, self.gcon,
                                        **kwargs)
        return stress_energy_HD + stress_energy_EM
    
    def sigma_cut(self, **kwargs):
        if 'sigma_cut' in kwargs and 'sigma' not in kwargs:
            B02 = self.B0 * self.B0
            Fld = Field(self.pars, self.CONSTANTS, cgs=self.kwargs.get('cgs'))
            b2  = Fld.calc_b2(self.B, self.B, self.gcov, 
                    from_B=True, uu=self.uu) / B02
            sigma = (b2 * B02 / (8. * np.pi)) / (self.rho * self.c * self.c)
        elif 'sigma_cut' in kwargs and 'sigma' in kwargs:
            sigma = kwargs.get('sigma')
            return (sigma < kwargs.get('sigma_cut'))
        else:
            return 1.0

class Analytic():
    def __init__(self, CONSTANTS, **kwargs):
        self.cgs = kwargs.get('cgs', False)
        if (self.cgs):
            self.c    = CONSTANTS['c']        # cm/s
        else:
            self.c    = 1.0
        
    # Calculate Eddington luminosity for black hole of mass MBH (solar masses)
    def calc_LEdd(self, MBH):
        return 1.26e38*MBH # erg/s
    
    # Calculate Eddington accretion rate for black hole of mass MBH
    def calc_Mdot_Edd(self, MBH, **kwargs):
        eta = kwargs.get('eta', 0.1)
        return self.calc_LEdd(MBH) / (eta * self.c * self.c)

class Coordinates():
    def __init__(self, r, theta, phi):
        self.r  = r
        self.th = theta
        self.ph = phi

    def sph_to_cart_vec(self, vec):
        shape = list(vec.shape)
        shape.insert(0, 4)
        drdx = np.zeros(shape, dtype=np.float32)

        # Transformation to Cartesian Kerr Schild from Spherical Kerr Schild
        drdx[0, 0] = 1.0
        drdx[1, 1] = (np.sin(self.th) * np.cos(self.ph))
        drdx[1, 2] = (self.r * np.cos(self.th) * np.cos(self.ph))
        drdx[1, 3] = (-self.r * np.sin(self.th) * np.sin(self.ph))
        drdx[2, 1] = (np.sin(self.th) * np.sin(self.ph))
        drdx[2, 2] = (self.r * np.cos(self.th) * np.sin(self.ph))
        drdx[2, 3] = (self.r * np.sin(self.th) * np.cos(self.ph))
        drdx[3, 1] = (np.cos(self.th))
        drdx[3, 2] = (-self.r * np.sin(self.th))

        return np.einsum('ij...,j...->i...', drdx, vec)
    
    def int_to_sph_vec(self, vec):
        shape = list(vec.shape)
        shape.insert(0, 4)
        dxdxp = np.zeros(shape, dtype=np.float32)

        # Transformation to Spherical Kerr Schild from internal coordinates
        dxdxp[0, 0] = 1.0
        dxdxp[1, 1] = self.r
        dxdxp[2, 2] = np.pi / 2.0
        dxdxp[3, 3] = 1.0

        return np.einsum('ij...,j...->i...', dxdxp, vec)
    
    def int_to_cart_vec(self, vec):
        return self.sph_to_cart_vec(self.int_to_sph_vec(vec))

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

    # Compute b from B
    def calc_b(self, B_in, uu_in, gcov):
        B, uu = B_in.copy(), uu_in.copy()
        # de-dimensionalized 4-vector uu and 3-vector B^i
        B     *= (1.0 / self.B0)
        uu    *= (1.0 / self.c)
        uu_cov = np.einsum('ij...,j...->i...', gcov, uu)
        b      = np.zeros(B.shape)
        # compute b^t
        b[0]   = B[1]*uu_cov[1] + B[2]*uu_cov[2] + B[3]*uu_cov[3]
        # compute b^i
        for i in range(1, 4):
            b[i] = (B[i] + uu[i]*b[0]) / uu[0]
        # re-dimensionalize 4-vector
        return b * self.B0
    
    # Compute b^2 (assumes B and uu have units)
    def calc_b2(self, b, B, gcov, **kwargs):
        if (kwargs.get('from_B')):
            if (kwargs.get('uu') is None):
                raise TypeError("Must specify uu to use from_B.")
            else:
                uu = kwargs.get('uu')
                b  = self.calc_b(B, uu, gcov)
        return Tools.dot(b, b, gcov)
    
    # Compute the electromagnetic stress-energy tensor
    def calc_TEM(self, uu_in, B_in, gcov, gcon, **kwargs):
        uu = uu_in.copy() 
        if 'index' in kwargs:
            [i, j] = kwargs.get('index')
            uu_tnsr = (uu[i] / self.c) * (uu[j] / self.c)
        else:
            uu_tnsr = np.einsum('i...,j...->ij...', uu / self.c, uu / self.c)

        if 'b' in kwargs:
            b = kwargs.get('b')
        else:
            B   = B_in.copy()
            b   = self.calc_b(B, uu, gcov) / self.B0
        
        if 'index' in kwargs:
            bb_tnsr = b[i] * b[j]
        else:
            bb_tnsr = np.einsum('i...,j...->ij...', b, b)
        
        if 'b2' in kwargs:
            b2 = kwargs.get('b2')
        else:
            b2 = self.calc_b2(b, b, gcov)
        
        B02 = self.B0 * self.B0
        if 'index' not in kwargs:
            return (b2*uu_tnsr + 0.5*b2*gcon - bb_tnsr) * B02
        else:
            return (b2*uu_tnsr + 0.5*b2*gcon[i, j] - bb_tnsr) * B02

    
class Fluid():
    def __init__(self, pars, CONSTANTS, **kwargs):
        self.cgs = kwargs.get('cgs', False)
        if (self.cgs):
            self.rho0 = pars['density_scale'] # g/cm^3 
            self.c    = CONSTANTS['c']        # cm/s
        else:
            self.rho0 = 1.0
            self.c    = 1.0

        self.CONSTANTS = CONSTANTS

    # Compute the hydrodynamic stress-energy tensor
    def calc_THD(self, rho_in, uu_in, ugas_in, pgas_in, gcon, **kwargs):
        rho, uu, ugas = rho_in.copy(), uu_in.copy(), ugas_in.copy()
        if (kwargs.get('EOS')):
            Te_in  = kwargs.get('Te')
            Ti_in  = kwargs.get('Ti')
            Te, Ti = Te_in.copy(), Ti_in.copy()
            pgas   = self.calc_pgas(ugas, Te, Ti)
        else:
            pgas = pgas_in.copy()
        rho0c2 = self.rho0 * self.c * self.c
        rho  *= 1. / self.rho0
        ugas *= 1. / rho0c2
        pgas *= 1. / rho0c2

        if 'index' not in kwargs:
            uu_tnsr = np.einsum('i...,j...->ij...', uu / self.c, uu / self.c)
            return ((rho + ugas + pgas) * uu_tnsr + pgas * gcon) * rho0c2
        else:
            [i, j] = kwargs.get('index')
            uu_tnsr = (uu[i] / self.c) * (uu[j] / self.c)
            return ((rho + ugas + pgas) * uu_tnsr + pgas * gcon[i, j]) * rho0c2

    # Compute pgas from ugas using gamma law EOS
    def calc_pgas(self, ugas_in, Te_in, Ti_in):
        ugas, Te, Ti = ugas_in.copy(), Te_in.copy(), Ti_in.copy()
        MU_I  = 4.0 / (4.0 * self.CONSTANTS['X_AB'] + self.CONSTANTS['Y_AB'])
        MU_E  = 2.0 / (1.0 + self.CONSTANTS['X_AB'])
        MI_ME = self.CONSTANTS['mi'] / self.CONSTANTS['me']

        # Calculate gamma (ratio of specific heats)
        Theta_e = Te * (MU_E * MI_ME)
        Theta_i = Ti * MU_I

        gam_e = (10.0 + 20.0 * Theta_e) / (6.0 + 15.0 * Theta_e)
        gam_i = (10.0 + 20.0 * Theta_i) / (6.0 + 15.0 * Theta_i)
        
        gam_g = 1. + ((
                (gam_e - 1.) * (gam_i - 1.) * (
                (MU_I / (MU_E * MI_ME)) + (Theta_i / Theta_e)) ) / (
                (Theta_i / Theta_e) * (gam_e - 1.) + 
                (MU_I / (MU_E * MI_ME)) * (gam_i - 1.) ))
        return (gam_g - 1.) * ugas

class Tools():
    def dot(a, b, gcov):
        # contract two 4-vectors (g_ij a^j b^i)
        a_cov = np.einsum('ij...,j...->i...', gcov, a)
        return np.einsum('i...,i...->...', a_cov, b)

if __name__ == "__main__":
    print("Executing analysis.")

'''
# calculate disk mass
def calc_Mtot(grid, data, params):
    gdet          = grid['gdet']
    rho, uu       = data['rho'], data['uu']
    dx1, dx2, dx3 = params['dx1'], params['dx2'], params
    ['dx3'] 
    return np.sum((rho * uu[0]) * gdet * dx1 * dx2 * dx3)

# Calculate inward mass accretion rate as function of radius
def calc_Mdot(grid, data, params):
    gdet     = grid['gdet']
    rho, uu  = data['rho'], data['uu']
    dx2, dx3 = params['dx2'], params['dx3'] 
    return (-rho * uu[1] * gdet * dx2 * dx3).sum(-1).sum(-1)
'''

'''
# Calculate precession period
def calc_PrecPeriod(angle_tilt):
    global gam,a,precperiod
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

    # Calculate density averaged emmission rate
    # cool_emmit=(fact1*kappa_emmit*rho*R_G_CGS*density_scale*2.2714380625235033e+21*(TE**4)*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)[0]/norm1
    # cool_emmit=(fact1*(kappa_emmit*rho*R_G_CGS*density_scale*2.2714380625235033e+21*(TE**4)-kappa_abs*rho*R_G_CGS*density_scale*Ehat/ENERGY_DENSITY_SCALE)*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)[0]/norm1
    # cool_abs=(fact1*kappa_abs*rho*R_G_CGS*density_scale*Ehat/ENERGY_DENSITY_SCALE*gdet*_dx1*_dx2*_dx3).sum(-1).sum(-1)[0]/norm1

    # Calculate relevant timescales for emmission, diffusion and absorption in disk
    # Theta_r = Tr * 1.6863687454173171e-10
    # tau_scaleheight=tau_es_avg/(2.0*scaleheight*r[0,:,0,0])
    # vdiff=np.sqrt(16.0/(9.0*tau_scaleheight*tau_scaleheight))
    # vdiff[vdiff>1.0]=1.0
    # t_diff=scaleheight*r[0,:,0,0]/vdiff
    # t_emmit_disk=(pe_tot/(gam-1.0)/(cool_emmit))
    # t_abs_disk=(pe_tot/(gam-1.0)/(cool_abs))
    # t_compton_disk=((pe_tot/(gam-1.0)*ENERGY_DENSITY_SCALE)/((kappa_es *rho*R_G_CGS*density_scale* Ehat * 4.0 * (Theta_e + Theta_r)*gdet*_dx1*_dx2*_dx3)*(Theta_e>Theta_r)).sum(-1).sum(-1)[0])

    # Calculate relevant timescales for emmission, diffusion and absorption in disk
    # filter1=(bsq/rho<1.0)
    # norm2=(gdet * filter1*rho/rho *_dx1*_dx2*_dx3).sum(-1).sum(-1)[0]
    # cool_emmit_corona=kappa_emmit*rho*R_G_CGS*density_scale*2.2714380625235033e+21*(TE**4)
    # t_emmit_corona=(TE*rho*filter1/(gam-1.0)/cool_emmit_corona).sum(-1).sum(-1)[0]/bs2new/bs3new

def calc_lum():
    global lum
    p = (gam - 1.0) * ug
    lum = np.sum((rho ** 3.0 * p ** (-2.0) * np.exp(-0.2 * (rho ** 2.0 / (np.sqrt(bsq) * p ** 2.0)) ** (1.0 / 3.0)) * (h > np.pi / 3.0) * (h < 2.0 / 3.0 * np.pi) * (r < 50.) * gdet * _dx1 * _dx2 * _dx3))

def calc_rad_avg():
    global rad_avg
    rad_avg = (r * rho * gdet * _dx1 * _dx2 * _dx3).sum(-1).sum(-1).sum(-1) / ((rho * gdet * _dx1 * _dx2 * _dx3).sum(-1).sum(-1).sum(-1))

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

# Calculate magnetic flux phibh as function of radius
def calc_phibh():
    global phibh
    phibh = 0.5 * (np.abs(gdet * B[1]) * _dx2 * _dx3).sum(-1).sum(-1)

def calc_rad_dump():
    global cool_time_emmit, cool_time_abs, cool_time_compt, cool_time_coul, tau_vertical_emmit, tau_vertical_abs, tau_vertical_es
    global Tr, Te, TE, TI, rho, MU_E, MH_CGS, kappa_emmit, kappa_abs, kappa_es

    # Some variables
    Theta_r = Tr * 1.6863687454173171e-10
    Theta_e = Te * 1.6863687454173171e-10
    ne_cgs = rho * density_scale / (MU_E * MH_CGS)
    n_cgs = rho * density_scale / (MH_CGS)

    # Cooling rates
    lambda_emmit = np.abs((kappa_emmit) * rho * R_G_CGS * density_scale * np.float64(ARAD * (np.float64(Te) ** 4)) / ENERGY_DENSITY_SCALE)
    lambda_abs = (kappa_abs) * rho * R_G_CGS * density_scale * Ehat / ENERGY_DENSITY_SCALE
    lambda_scatter = np.abs(kappa_es * rho * R_G_CGS * density_scale * Ehat / ENERGY_DENSITY_SCALE * 4.0 * (np.float64(Theta_e)))
    lambda_coul = np.float64(1.5 * ME_CGS / MH_CGS * coulog * C_CGS * BOLTZ_CGS * THOMSON_CGS)
    lambda_coul *= np.float64(ne_cgs) * np.float64(n_cgs) * (np.float64(Ti) - np.float64(Te))
    lambda_coul *= R_G_CGS / (C_CGS * ENERGY_DENSITY_SCALE)

    # Cooling timescales
    ue = TE*rho/(game-1.0)
    ui = TI * rho / (gami - 1.0)
    cool_time_emmit = ((gdet * ue*((bsq/rho)<1.0)).sum(-1).sum(-1) / (gdet * lambda_emmit*((bsq/rho)<1.0)).sum(-1).sum(-1))[0]
    cool_time_abs = ((gdet * ue*((bsq/rho)<1.0)).sum(-1).sum(-1) / (gdet * lambda_abs*((bsq/rho)<1.0)).sum(-1).sum(-1))[0]
    cool_time_compt = ((gdet * ue*((bsq/rho)<1.0)).sum(-1).sum(-1) / (gdet * lambda_scatter*((bsq/rho)<1.0)).sum(-1).sum(-1))[0]
    cool_time_coul = ((gdet * (ui + ue)*((bsq/rho)<1.0)).sum(-1).sum(-1) / (gdet * lambda_coul*((bsq/rho)<1.0)).sum(-1).sum(-1))[0]

    # Vertical optical depth
    tau_vertical_emmit = 0.5*((np.sqrt(gcov[2, 2]) * _dx2 * kappa_emmit * R_G_CGS * rho * density_scale*((bsq/rho)<1.0)).sum(-1).sum(-1) / bs3new)[0]
    tau_vertical_abs = 0.5*((np.sqrt(gcov[2, 2]) * _dx2 * kappa_abs * R_G_CGS * rho * density_scale*((bsq/rho)<1.0)).sum(-1).sum(-1) / bs3new)[0]
    tau_vertical_es = 0.5*((np.sqrt(gcov[2, 2]) * _dx2 * kappa_es * R_G_CGS * rho * density_scale*((bsq/rho)<1.0)).sum(-1).sum(-1) / bs3new)[0]

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
