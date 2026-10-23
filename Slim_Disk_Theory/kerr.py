"""Kerr equatorial geometry and circular-geodesic quantities.

Geometric units G = c = M_BH = 1 throughout this module; radii in R_g.
Self-contained on purpose - Slim_Disk_Theory does not import the rest of
HAMR_Visualization, so it cannot be broken by changes there.

CONVENTION TRAP: `l_K` here is u_phi, the conserved specific angular momentum
that appears in the angular-momentum conservation law. Much of the slim-disk
literature writes l = -u_phi/u_t, which is a DIFFERENT function. Use
`l_from_ell()` to convert if you need to compare with a paper.
"""
import numpy as np


class Kerr:
    def __init__(self, a):
        if not -1. < a < 1.:
            raise ValueError(f'spin must satisfy |a| < 1, got {a}')
        self.a = float(a)

    # --- metric functions, equatorial plane -----------------------------
    def Delta(self, r):
        return r*r - 2.*r + self.a**2

    def Amet(self, r):
        """A = (r^2+a^2)^2 - a^2 Delta  at theta = pi/2."""
        return r**4 + r*r*self.a**2 + 2.*r*self.a**2

    def omega(self, r):
        """Frame-dragging angular velocity of the ZAMO."""
        return 2.*self.a*r / self.Amet(r)

    def lapse(self, r):
        """alpha_L = r sqrt(Delta) / sqrt(A)."""
        return r*np.sqrt(self.Delta(r) / self.Amet(r))

    def Rtilde(self, r):
        """Proper circumferential radius sqrt(g_phiphi) = sqrt(A)/r."""
        return np.sqrt(self.Amet(r)) / r

    # --- circular geodesics (Bardeen, Press & Teukolsky 1972) -----------
    def OmegaK(self, r, retrograde=False):
        s = -1. if retrograde else 1.
        return s / (r**1.5 + s*self.a)

    def l_K(self, r):
        """u_phi of a circular equatorial geodesic."""
        a, sr = self.a, np.sqrt(r)
        return (r*r - 2.*a*sr + a*a) / (r**0.75 * np.sqrt(r**1.5 - 3.*sr + 2.*a))

    def E_K(self, r):
        """-u_t of a circular equatorial geodesic."""
        a, sr = self.a, np.sqrt(r)
        return (r*r - 2.*r + a*sr) / (r * np.sqrt(r*r - 3.*r + 2.*a*sr))

    def r_horizon(self):
        return 1. + np.sqrt(1. - self.a**2)

    def r_isco(self):
        a = self.a
        z1 = 1. + (1. - a*a)**(1./3.) * ((1. + a)**(1./3.) + (1. - a)**(1./3.))
        z2 = np.sqrt(3.*a*a + z1*z1)
        return 3. + z2 - np.sign(a) * np.sqrt((3. - z1)*(3. + z1 + 2.*z2))

    def r_mb(self):
        """Marginally bound orbit; the inner edge cannot go inside this."""
        a = self.a
        return 2. - a + 2.*np.sqrt(1. - a)

    # --- slim-disk kinematics ------------------------------------------
    def kinematics(self, r, ell, V):
        """Given u_phi = ell and corotating radial velocity V, return the rest.

        Inverts  ell = Rtilde * gamma_V * vphi/sqrt(1-vphi^2)  for the ZAMO
        azimuthal velocity vphi, then builds Omega and E = -u_t.
        """
        aL, Rt, om = self.lapse(r), self.Rtilde(r), self.omega(r)
        gV = 1. / np.sqrt(1. - V*V)
        X = ell / (Rt * gV)                  # = vphi / sqrt(1-vphi^2)
        vphi = X / np.sqrt(1. + X*X)
        gphi = np.sqrt(1. + X*X)             # = 1/sqrt(1-vphi^2)
        Omega = om + aL * vphi / Rt
        Ener = aL * gV * gphi + om * ell     # -u_t
        return Omega, Ener, vphi, gV, gphi

    def Omega_perp2(self, r, ell, Ener):
        """Vertical epicyclic frequency squared, in the fluid's PROPER time.

        Abramowicz, Lanza & Percival (1997).  Reduces to 1/r^3 far out, and
        for Schwarzschild reproduces Omega_K^2/(1-3/r) = 1/(r^2 (r-3)).
        """
        return (ell*ell - self.a**2 * (Ener*Ener - 1.)) / r**4

    def l_from_ell(self, r, ell, Ener):
        """Convert u_phi -> l = -u_phi/u_t, the OTHER convention."""
        return ell / Ener

    # --- the radial-momentum gravity term -------------------------------
    def grav_A(self, r, Omega, vphi):
        """Script-A in  V/(1-V^2) dV/dr = A/r - (1/(Sigma c^2)) dPi/dr.

        Vanishes when Omega = Omega_K. Newtonian limit: A -> r^2 Omega^2 - 1/r,
        so A/r -> r Omega^2 - 1/r^2, i.e. centrifugal minus gravity.
        """
        Op, Om_ = self.OmegaK(r), self.OmegaK(r, retrograde=True)
        pref = self.Amet(r) / (r**3 * self.Delta(r) * Op * Om_)
        return -pref * (Omega - Op) * (Omega - Om_) / (1. - vphi*vphi)
