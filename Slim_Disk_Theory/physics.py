"""Equation of state and the pluggable cooling function.

CGS. One-zone vertical structure: midplane rho = Sigma/(2H), p = Pi/(2H).
"""
import numpy as np

CONST = dict(G=6.67430e-8, c=2.99792458e10, Msun=1.98892e33,
             kB=1.380649e-16, mp=1.67262192e-24, aRAD=7.5657e-15,
             sigSB=5.670374e-5, X_AB=0.7)

KES = 0.2 * (1. + CONST['X_AB'])      # 0.34 cm^2/g, electron scattering
MU = 0.615                            # fully ionised solar


def kappa_ff(rho, T):
    """Kramers free-free (Rosseland mean), cm^2/g."""
    return 6.4e22 * rho * T**-3.5


# --------------------------------------------------------------------------
# Cooling: Q-(Sigma, T, H, ...) -> erg/cm^2/s, TOTAL from both faces.
# Swap by passing cooling=<callable> to SlimDisk.
# --------------------------------------------------------------------------
def cooling_lte(Sigma, T, H, **kw):
    """Diffusion-limited blackbody cooling, electron scattering only.

    Q- = 4 a c T^4 / (3 kappa_es Sigma).  This is the convention already used
    in Bambic_Analysis/plot_surface_density_slim_disk.py, kept identical so
    the theory curves are comparable with what is already plotted.
    """
    C = CONST
    return 4. * C['aRAD'] * C['c'] * T**4 / (3. * KES * Sigma)


def cooling_taueff(Sigma, T, H, **kw):
    """Bridge to the effectively-optically-thin regime.

    tau_eff = sqrt(tau_abs (tau_abs + tau_es)).  The plan file records
    tau_eff = 0.3-25 while tau_es = 700-43000 in the simulation, so the pure
    blackbody form above can overestimate the cooling. This is the hook for
    the deferred Q- work, NOT a validated result - it is not the default.
    """
    C = CONST
    rho = Sigma / (2. * H)
    t_es = 0.5 * KES * Sigma
    t_ab = 0.5 * kappa_ff(rho, T) * Sigma
    t_eff = np.sqrt(np.maximum(t_ab * (t_ab + t_es), 1e-30))
    return 8. * C['sigSB'] * T**4 / (3. * t_es + np.sqrt(3.) + 1. / np.maximum(t_ab, 1e-30)) \
        * (t_eff / np.maximum(t_eff, 1e-30))


COOLING = {'lte': cooling_lte, 'taueff': cooling_taueff}
