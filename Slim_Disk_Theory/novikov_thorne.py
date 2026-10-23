"""Novikov-Thorne / Page-Thorne thin disk - the reference the slim disk must
reduce to as mdot -> 0.  Geometric units; F returned dimensionless.
"""
import numpy as np
from scipy.integrate import quad
from kerr import Kerr


class NovikovThorne:
    def __init__(self, a):
        self.k = Kerr(a)
        self.risco = self.k.r_isco()

    def efficiency(self):
        """eta = 1 - E(r_isco)."""
        return 1. - self.k.E_K(self.risco)

    def l_in(self):
        return self.k.l_K(self.risco)

    def flux_dimensionless(self, r):
        """F(r) = (Mdot c^2 / (4 pi Rg^2)) * script-F(r), one side of the disk.

        KNOWN ISSUE - READ BEFORE USING. Integrating this over the disk gives
        an efficiency 1.9% high at a=0 and 6.6% high at a=0.9375, i.e. the
        normalisation carries a spin-dependent error (most likely the area
        element / sqrt(-g) convention, which differs between Page & Thorne's
        radial variable and Boyer-Lindquist r). It is NOT a constant factor.

        The SOLVER NEVER CALLS THIS. It is a reference curve only. The
        efficiency and l_in used as acceptance targets come from
        efficiency() = 1 - E(r_isco) and l_in() = l_K(r_isco), which are exact
        closed forms and are verified to 1e-12 in TEST 1. Do not use this flux
        profile for quantitative work until the normalisation is fixed.

        script-F = (1/r) (-dOmega/dr)/(E - Omega l)^2 * int_isco^r (E-Omega l) dl/dr' dr'
        Normalised so that 2 * int F 2 pi R dR = eta Mdot c^2 is checked in the tests.
        """
        k = self.k
        r = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.zeros_like(r)
        eps = 1e-7

        def dOm(x):
            return (k.OmegaK(x + eps*x) - k.OmegaK(x - eps*x)) / (2.*eps*x)

        def dl(x):
            return (k.l_K(x + eps*x) - k.l_K(x - eps*x)) / (2.*eps*x)

        def integrand(x):
            return (k.E_K(x) - k.OmegaK(x)*k.l_K(x)) * dl(x)

        for i, rr in enumerate(r):
            if rr <= self.risco*(1. + 1e-9):
                continue
            I, _ = quad(integrand, self.risco, rr, limit=200)
            den = (k.E_K(rr) - k.OmegaK(rr)*k.l_K(rr))**2
            out[i] = -dOm(rr) * I / (rr * den)
        return out if out.size > 1 else out[0]
