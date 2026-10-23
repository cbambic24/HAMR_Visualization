"""Relativistic slim-disk solver: ODEs + shooting for the eigenvalue l_in.

Inputs: MBH [Msun], a, mdot [Eddington], alpha.  Nothing else is fitted.

EQUATION SET  (Sadowski 2009; Abramowicz, Chen, Granath & Lasota 1996;
Kato, Fukue & Mineshige for the advection term)

  E1 mass       Mdot   = -2 pi Sigma sqrt(Delta) Rg  (V c) gamma_V
  E2 ang.mom.   Mdot (L - L_in) Rg c = 2 pi alpha Pi sqrt(Delta A)/r Rg^2
  E3 radial     V/(1-V^2) dV/dr = A_grav/r - (1/(Sigma c^2)) dPi/dr
  E4 entropy    Q_adv = Q+ - Q-
                Q_adv = -(Mdot/(2 pi r^2 Rg^2))(Pi/Sigma)
                        [(12-10.5 b) dlnT/dlnr - (4-3 b) dlnSigma/dlnr]
                Q+    = -alpha Pi (A/r^3) dOmega/dr
  E5 vertical   H = sqrt(Pi/Sigma) / Omega_perp,
                Omega_perp^2 = (L^2 - a^2(E^2-1))/r^4   [ALP97, proper time]

L is u_phi, NOT -u_phi/u_t.

REDUCTION.  E1/E2/E5 + EOS are solved algebraically at each (r, V, T), giving
Sigma(r,V) and Pi(r,V,T), Omega(r,V,T).  E3 and E4 then become a LINEAR 2x2
system for (dln(-V)/dlnr, dlnT/dlnr):

        [m11 m12] [P]   [b1]
        [m21 m22] [Q] = [b2]

so  D = det(M)  is the critical-point denominator and the Cramer numerators
are N_V, N_T.  D -> 0 is the sonic point; regularity (N_V -> 0 there too) is
ONE condition, which is what fixes L_in.

SHOOTING.  Integrate INWARD from a thin-disk outer boundary, which is stable
because the thin-disk branch is the inward attractor, and bisect on L_in using
the sign of N_V at the D = 0 crossing.  Then relaunch from the sonic point to
continue supersonically to the horizon.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

from kerr import Kerr
from novikov_thorne import NovikovThorne
from physics import CONST, KES, MU, COOLING, cooling_lte

C = CONST


class SlimDiskError(RuntimeError):
    pass


class SlimDisk:
    def __init__(self, MBH, a, mdot, alpha, cooling='lte', eta_edd=None,
                 mu=MU, kappa_es=KES, alpha_index=0., alpha_r0=10.):
        """alpha may be radius-dependent:  alpha(r) = alpha * (r/alpha_r0)**alpha_index

        alpha_index = 0   -> constant alpha (the default; previous behaviour)
        alpha_index = -1  -> alpha ~ 1/r, normalised so alpha(alpha_r0) = alpha

        The measured low-res slope over r = 6-20 is d ln alpha/d ln r = -1.00
        (all epochs, sigma<3 column); other windows give -0.65 to -0.86. The
        index is left free so it can be swept.
        """
        self.MBH, self.a, self.mdot, self.alpha = float(MBH), float(a), float(mdot), float(alpha)
        self.alpha_index = float(alpha_index)
        self.alpha_r0 = float(alpha_r0)
        self.k = Kerr(a)
        self.nt = NovikovThorne(a)
        self.mu, self.kes = mu, kappa_es
        self.cool = COOLING[cooling] if isinstance(cooling, str) else cooling
        self.Rg = C['G'] * self.MBH * C['Msun'] / C['c']**2
        # Eddington rate referenced to the NT efficiency of THIS spin, so that
        # mdot = 1 means L = L_Edd for a radiatively efficient disk.
        self.eta_edd = self.nt.efficiency() if eta_edd is None else float(eta_edd)
        self.LEdd = 1.26e38 * self.MBH
        self.MdotEdd = self.LEdd / (self.eta_edd * C['c']**2)
        self.Mdot = self.mdot * self.MdotEdd
        self.risco, self.rhor = self.k.r_isco(), self.k.r_horizon()
        self._pi_cache = None

    # ------------------------------------------------------------------
    # algebraic inner solve
    # ------------------------------------------------------------------
    def alpha_of(self, r):
        """Local alpha. Radial derivatives of alpha are picked up automatically:
        matrix() forms dPi/dr by finite-differencing state() at r +/- dr, and
        state() calls alpha_of(r +/- dr), so d(alpha)/dr enters without any
        extra analytics."""
        if self.alpha_index == 0.:
            return self.alpha
        return self.alpha * (r / self.alpha_r0)**self.alpha_index

    def _torque_coef(self, r):
        """kappa2 such that  L = L_in + kappa2 * Pi   (from E2)."""
        Del, A = self.k.Delta(r), self.k.Amet(r)
        G2 = np.sqrt(Del * A) / r
        return 2. * np.pi * self.alpha_of(r) * G2 * self.Rg / (self.Mdot * C['c'])

    def _sigma(self, r, V):
        gV = 1. / np.sqrt(1. - V*V)
        return -self.Mdot / (2.*np.pi*np.sqrt(self.k.Delta(r))*self.Rg*V*C['c']*gV)

    def state(self, r, V, T, l_in, pi_hint=None):
        """Solve E2 + E5 + EOS at fixed (r, V, T). Returns a dict of cgs state.

        With Omega_perp held fixed the closure
            Pi = Sigma cg2 + (2/3)(sqrt(Pi/Sigma)/Omega_perp) a T^4
        is a quadratic in sqrt(Pi), so it inverts analytically; only the weak
        Omega_perp(Pi) dependence (via E2 -> L -> the ALP97 tidal term) is
        iterated. brentq is kept as the fallback.
        """
        k = self.k
        Sigma = self._sigma(r, V)
        cg2 = C['kB'] * T / (self.mu * C['mp'])
        kap2 = self._torque_coef(r)
        aT4 = C['aRAD'] * T**4

        def aux_of(Pi):
            ell = l_in + kap2 * Pi
            Om, En, vphi, gV, gphi = k.kinematics(r, ell, V)
            op2 = k.Omega_perp2(r, ell, En)
            if not np.isfinite(op2) or op2 <= 0.:
                return None
            return ell, Om, En, vphi, np.sqrt(op2) * C['c'] / self.Rg

        def pi_of(Operp):
            B = (2./3.) * aT4 / (Operp * np.sqrt(Sigma))
            return (0.5*(B + np.sqrt(B*B + 4.*Sigma*cg2)))**2

        # Solve  g(Pi) = pi_of(Omega_perp(Pi)) - Pi = 0.
        #
        # Do NOT iterate this as a fixed point. In the radiation- and
        # torque-dominated regime L ~ kap2*Pi, so Omega_perp ~ Pi while
        # pi_of ~ 1/Omega_perp^2: the map has log-gain -2 and DIVERGES. That is
        # what broke every mdot >~ 40.
        #
        # Instead bracket it exactly. Omega_perp increases with Pi and pi_of
        # decreases with Omega_perp, so g is monotone decreasing and the root is
        # unique. With lo = Sigma*cg2 (the H -> 0 floor, where g > 0), the single
        # point hi = pi_of(Omega_perp(lo)) satisfies g(hi) <= 0, because
        # hi >= lo implies pi_of(Omega_perp(hi)) <= pi_of(Omega_perp(lo)) = hi.
        # So one evaluation gives a guaranteed bracket.
        def g(Pi):
            aa = aux_of(Pi)
            if aa is None:
                return np.nan, None
            return pi_of(aa[4]) - Pi, aa

        lo = Sigma * cg2 * (1. + 1e-12)
        glo, alo = g(lo)
        if alo is None:
            raise SlimDiskError('no vertical equilibrium at r=%.4g' % r)
        hi = pi_of(alo[4])
        if not np.isfinite(hi) or hi <= lo:
            Pi, aux = lo, alo                      # gas-pressure floor
        else:
            ghi, ahi = g(hi)
            if not np.isfinite(ghi):
                raise SlimDiskError('vertical equilibrium lost at r=%.4g' % r)
            if ghi >= 0.:
                Pi, aux = hi, ahi
            else:
                Pi = brentq(lambda p: g(p)[0], lo, hi, xtol=1e-300, rtol=1e-13)
                aux = aux_of(Pi)
                if aux is None:
                    raise SlimDiskError('vertical equilibrium lost at r=%.4g' % r)

        ell, Om, En, vphi, Operp = aux
        H = np.sqrt(Pi / Sigma) / Operp
        beta = Sigma * cg2 / Pi
        return dict(r=r, V=V, T=T, Sigma=Sigma, Pi=Pi, H=H, rho=Sigma/(2.*H),
                    ell=ell, Omega=Om * C['c'] / self.Rg, Omega_geo=Om,
                    E=En, vphi=vphi, beta=beta, Operp=Operp, cg2=cg2,
                    cs2=Pi/Sigma, Prad_Pgas=(1. - beta)/max(beta, 1e-300))

    # ------------------------------------------------------------------
    # the 2x2 linear system
    # ------------------------------------------------------------------
    def matrix(self, r, V, T, l_in, h=1e-5, pi_hint=None):
        st = self.state(r, V, T, l_in, pi_hint=pi_hint)
        Sigma, Pi, H, beta = st['Sigma'], st['Pi'], st['H'], st['beta']
        Om, ell, vphi = st['Omega'], st['ell'], st['vphi']

        hint = st['Pi']

        def pack(rr, VV, TT):
            s = self.state(rr, VV, TT, l_in, pi_hint=hint)
            return s['Pi'], s['Omega'], s['Sigma']

        dr, dV, dT = r*h, abs(V)*h, T*h
        Pp, Op, Sp = pack(r+dr, V, T); Pm, Om_, Sm = pack(r-dr, V, T)
        Pi_r, Om_r, Sg_r = (Pp-Pm)/(2*dr), (Op-Om_)/(2*dr), (Sp-Sm)/(2*dr)
        Pp, Op, _ = pack(r, V+dV, T);    Pm, Om_, _ = pack(r, V-dV, T)
        Pi_V, Om_V = (Pp-Pm)/(2*dV), (Op-Om_)/(2*dV)
        Sg_V = self._sigma(r, V)*(-1./V - V/(1.-V*V))      # analytic
        Pp, Op, _ = pack(r, V, T+dT);    Pm, Om_, _ = pack(r, V, T-dT)
        Pi_T, Om_T = (Pp-Pm)/(2*dT), (Op-Om_)/(2*dT)

        c2 = C['c']**2
        Agrav = self.k.grav_A(r, st['Omega_geo'], vphi)

        # E3, multiplied through by r:  m11 P + m12 Q = b1
        m11 = V*V/(1. - V*V) + Pi_V*V/(Sigma*c2)
        m12 = Pi_T*T/(Sigma*c2)
        b1 = Agrav - r*Pi_r/(Sigma*c2)

        # E4
        Cadv = -(self.Mdot/(2.*np.pi*r*r*self.Rg**2)) * (Pi/Sigma)
        aTc, aSc = 12. - 10.5*beta, 4. - 3.*beta
        Kq = self.alpha_of(r) * Pi * (self.k.Amet(r)/r**3)
        Qm = self.cool(Sigma, T, H)
        m21 = -Cadv*aSc*Sg_V*V/Sigma + Kq*Om_V*V/r
        m22 = Cadv*aTc + Kq*Om_T*T/r
        b2 = Cadv*aSc*Sg_r*r/Sigma - Kq*Om_r - Qm

        # row-normalise (leaves N/D invariant, helps conditioning)
        s1 = max(abs(m11), abs(m12), abs(b1), 1e-300)
        s2 = max(abs(m21), abs(m22), abs(b2), 1e-300)
        m11, m12, b1 = m11/s1, m12/s1, b1/s1
        m21, m22, b2 = m21/s2, m22/s2, b2/s2

        D = m11*m22 - m12*m21
        NV = b1*m22 - b2*m12
        NT = m11*b2 - m21*b1
        st.update(D=D, NV=NV, NT=NT, m=(m11, m12, m21, m22), b=(b1, b2),
                  Qplus=-Kq*(Om_r + Om_V*0. + Om_T*0.), Qminus=Qm)
        return st

    def rhs(self, x, y, l_in):
        r, V, T = np.exp(x), -np.exp(y[0]), np.exp(y[1])
        st = self.matrix(r, V, T, l_in, pi_hint=self._pi_cache)
        self._pi_cache = st['Pi']
        D = st['D']
        if abs(D) < 1e-300:
            return np.array([0., 0.])
        return np.array([st['NV']/D, st['NT']/D])

    # ------------------------------------------------------------------
    # outer thin-disk boundary condition
    # ------------------------------------------------------------------
    def outer_state(self, r_out, l_in):
        """Thin-disk start: Omega = Omega_K, Q+ = Q- (no advection)."""
        k = self.k
        ell = k.l_K(r_out)
        if ell <= l_in:
            raise SlimDiskError('l_in exceeds l_K at the outer boundary')
        Pi = (ell - l_in) / self._torque_coef(r_out)
        En = k.E_K(r_out)
        op2 = k.Omega_perp2(r_out, ell, En)
        Operp = np.sqrt(op2) * C['c'] / self.Rg
        eps = 1e-6
        dOmK = (k.OmegaK(r_out*(1+eps)) - k.OmegaK(r_out*(1-eps)))/(2*eps*r_out)
        Qp = -self.alpha_of(r_out) * Pi * (k.Amet(r_out)/r_out**3) * dOmK * C['c']/self.Rg

        def f(T):
            Sigma = 4.*C['aRAD']*C['c']*T**4/(3.*self.kes*Qp)   # from Q+ = Q-
            cg2 = C['kB']*T/(self.mu*C['mp'])
            H = np.sqrt(Pi/Sigma)/Operp
            return Sigma*cg2 + (2./3.)*H*C['aRAD']*T**4 - Pi

        T = brentq(f, 1e3, 1e10, xtol=1e-300, rtol=1e-13)
        Sigma = 4.*C['aRAD']*C['c']*T**4/(3.*self.kes*Qp)
        # invert E1 for V:  Sigma = -Mdot/(2 pi sqrt(Delta) Rg V c gamma_V)
        VG = -self.Mdot/(2.*np.pi*Sigma*np.sqrt(k.Delta(r_out))*self.Rg*C['c'])
        V = VG/np.sqrt(1. + VG*VG)
        return V, T

    # ------------------------------------------------------------------
    # inward integration + eigenvalue shoot
    # ------------------------------------------------------------------
    def integrate_inward(self, l_in, r_out=300., r_min=None, rtol=1e-5,
                         r_arm=30.):
        """Integrate from the thin-disk outer boundary inward, in two stages.

        STAGE 1 (r_out -> r_arm), no events: the solution relaxes onto the
        thin-disk attractor from the approximate outer boundary condition. That
        startup transient makes N_V cross zero near r_out for every l_in, which
        is not the turnaround we are shooting on, so events stay off here.

        STAGE 2 (r_arm -> r_min), events armed. Two terminal events race:
          D  -> 0 first : reached the sonic point with N_V != 0, dV/dr blows up
          NV -> 0 first : dV/dr = 0, the inflow turns around, never goes sonic
        They coincide only for the regular transonic solution, so the ORDER of
        the two events is the shooting flag.

        Returns (outcome, (sol1, sol2), info), outcome in
        {'D_first', 'NV_first', 'ran_out', 'failed'}.
        """
        r_min = r_min if r_min is not None else 1.05 * self.rhor
        # The thin-disk outer BC needs a Q+ = Q- root to exist at r_out. In the
        # radiation-pressure-dominated zone it does not, and that zone grows
        # with mdot, so push r_out out until the boundary condition exists.
        V0 = T0 = None
        for _ in range(9):
            try:
                V0, T0 = self.outer_state(r_out, l_in)
                break
            except Exception:
                r_out *= 2.
        if V0 is None:
            return 'failed', (None, None), dict(err='no thin-disk outer BC found')
        self._pi_cache = None
        y0 = np.array([np.log(-V0), np.log(T0)])
        f = lambda x, y: self.rhs(x, y, l_in)

        try:
            sol1 = solve_ivp(f, (np.log(r_out), np.log(r_arm)), y0,
                             method='LSODA', rtol=rtol, atol=1e-9,
                             dense_output=True, max_step=0.15)
        except Exception as e:
            return 'failed', (None, None), dict(err=f'stage1: {e}')
        if sol1.status != 0:
            return 'failed', (sol1, None), dict(msg=f'stage1: {sol1.message}')

        def _mat(x, y):
            return self.matrix(np.exp(x), -np.exp(y[0]), np.exp(y[1]), l_in,
                               pi_hint=self._pi_cache)

        def ev_D(x, y):
            try:
                return _mat(x, y)['D']
            except Exception:
                return np.nan

        def ev_N(x, y):
            try:
                return _mat(x, y)['NV']
            except Exception:
                return np.nan

        def ev_lum(x, y):
            """Guard: ZAMO azimuthal velocity approaching c wrecks the algebra."""
            try:
                return 0.999 - abs(_mat(x, y)['vphi'])
            except Exception:
                return np.nan

        for e in (ev_D, ev_N, ev_lum):
            e.terminal, e.direction = True, 0

        try:
            sol2 = solve_ivp(f, (np.log(r_arm), np.log(r_min)), sol1.y[:, -1],
                             method='LSODA', events=[ev_D, ev_N, ev_lum],
                             rtol=rtol, atol=1e-9, dense_output=True,
                             max_step=0.02)
        except Exception as e:
            return 'failed', (sol1, None), dict(err=f'stage2: {e}')

        hits = [(te[0], i) for i, te in enumerate(sol2.t_events) if te.size]
        if hits:
            xs, idx = max(hits)          # integrating toward decreasing x
            ys = sol2.y_events[idx][0]
            st = self.matrix(np.exp(xs), -np.exp(ys[0]), np.exp(ys[1]), l_in)
            name = ('D_first', 'NV_first', 'failed')[idx]
            return name, (sol1, sol2), dict(r_s=np.exp(xs), V_s=-np.exp(ys[0]),
                                            T_s=np.exp(ys[1]), NV=st['NV'],
                                            D=st['D'], st=st)
        if sol2.status == 0:
            return 'ran_out', (sol1, sol2), dict(r_end=np.exp(sol2.t[-1]))
        return 'failed', (sol1, sol2), dict(msg=f'stage2: {sol2.message}')

    def residual(self, l_in, r_out=300., r_arm=30.):
        """+1 if the flow turns around (N_V first), -1 if dV/dr blows up (D first)."""
        outcome, sols, info = self.integrate_inward(l_in, r_out=r_out, r_arm=r_arm)
        val = {'NV_first': +1., 'D_first': -1.}.get(outcome, np.nan)
        return val, outcome, info

    # ------------------------------------------------------------------
    # l'Hopital at the sonic point + supersonic continuation
    # ------------------------------------------------------------------
    def lhopital(self, r_s, V_s, T_s, l_in, h=1e-5):
        """Slopes of the two solution curves crossing the X-point at r_s.

        Returns (roots, classification). Roots are P = dln(-V)/dlnr; the
        accretion branch is P < 0 (|V| grows inward), the wind branch P > 0.
        Opposite signs => saddle, same sign => NODE, in which case regularity
        alone no longer picks out a unique solution and l_in is not determined
        by this condition.  (Michel shows the same transition at
        Gamma = 3/2 + 3 c_s^2; see Muchotrzeb & Paczynski 1982.)
        """
        x0, p0, q0 = np.log(r_s), np.log(-V_s), np.log(T_s)

        def DNV(x, p, q):
            st = self.matrix(np.exp(x), -np.exp(p), np.exp(q), l_in)
            return st['D'], st['NV'], st['m'], st['b']

        D0, N0, m0, b0 = DNV(x0, p0, q0)
        g = {}
        for nm, (i, v0) in zip('xpq', enumerate((x0, p0, q0))):
            d = h * max(abs(v0), 1.)
            args_p = [x0, p0, q0]; args_p[i] += d
            args_m = [x0, p0, q0]; args_m[i] -= d
            Dp, Np, _, _ = DNV(*args_p)
            Dm, Nm, _, _ = DNV(*args_m)
            g['D' + nm] = (Dp - Dm) / (2*d)
            g['N' + nm] = (Np - Nm) / (2*d)
        m11, m12, m21, m22 = m0
        b1, _ = b0
        # exact constraint on the singular row: m11 P + m12 Q = b1
        al, be = b1/m12, -m11/m12          # Q = al + be P
        # P (Dx + Dp P + Dq Q) = Nx + Np P + Nq Q
        A2 = g['Dp'] + g['Dq']*be
        A1 = g['Dx'] + g['Dq']*al - g['Np'] - g['Nq']*be
        A0 = -(g['Nx'] + g['Nq']*al)
        disc = A1*A1 - 4.*A2*A0
        if disc < 0.:
            return None, 'spiral'
        roots = np.sort([(-A1 - np.sqrt(disc))/(2.*A2), (-A1 + np.sqrt(disc))/(2.*A2)])
        kind = 'saddle' if roots[0] < 0. < roots[1] else 'node'
        return (roots, al + be*roots), kind

    def continue_supersonic(self, r_s, V_s, T_s, l_in, r_min=None, eps=1e-4,
                            max_evals=20000):
        """Launch on the accretion root just inside r_s and run to the horizon.

        CAPPED ON PURPOSE. The supersonic branch between the sonic point and the
        horizon is stiff and LSODA can crawl there essentially forever (a 25 min
        non-finish was observed at mdot=100). `max_evals` aborts and returns
        whatever was integrated. This branch is NOT needed for the r = 10-30
        comparison, which is entirely subsonic, so it is off by default.
        """
        r_min = r_min if r_min is not None else 1.02*self.rhor
        out, kind = self.lhopital(r_s, V_s, T_s, l_in)
        if out is None:
            return None, kind
        (roots, qs) = out
        P, Q = roots[0], qs[0]                 # accretion branch
        dx = -eps
        x0 = np.log(r_s) + dx
        y0 = np.array([np.log(-V_s) + P*dx, np.log(T_s) + Q*dx])
        n = [0]

        def capped(x, y):
            n[0] += 1
            if n[0] > max_evals:
                raise RuntimeError('supersonic continuation exceeded %d evals'
                                   % max_evals)
            return self.rhs(x, y, l_in)

        try:
            sol = solve_ivp(capped, (x0, np.log(r_min)), y0, method='LSODA',
                            rtol=1e-6, atol=1e-12, max_step=0.02)
        except RuntimeError as e:
            return None, f'{kind} (aborted: {e})'
        return sol, kind

    # ------------------------------------------------------------------
    def solve(self, r_out=300., r_arm=30., span=(0.80, 1.45), nscan=8,
              tol=1e-5, verbose=True, supersonic=False):
        """Find l_in by bisecting the event-order flag. Returns a dict.

        residual = +1 (NV_first, flow turns around)  -> l_in too LOW
        residual = -1 (D_first,  dV/dr blows up)     -> l_in too HIGH
        The transonic solution is the discontinuity between them, so this is a
        plain bisection on a discrete flag, not a root find on a smooth
        residual - which is exactly why it is robust.
        """
        lis = self.nt.l_in()

        cache = {}

        def flag(f):
            outcome, sols, info = self.integrate_inward(lis*f, r_out=r_out,
                                                        r_arm=r_arm)
            v = {'NV_first': +1., 'D_first': -1.}.get(outcome, np.nan)
            if outcome == 'D_first':
                cache['sols'] = sols
            return v, outcome, info

        lo = hi = None
        prev = None
        best = None
        for f in np.linspace(span[0], span[1], nscan):
            v, out, info = flag(f)
            if verbose:
                rs = info.get('r_s', np.nan) if info else np.nan
                print(f'   scan l_in/l_isco={f:7.4f}  {out:9s}  r={rs:8.4f}', flush=True)
            if prev is not None and np.isfinite(v) and np.isfinite(prev[1]) and prev[1]*v < 0.:
                lo, hi = (prev[0], f) if prev[1] > 0 else (f, prev[0])
                if v < 0:
                    best = (f, info, cache.get('sols'))
                break
            if np.isfinite(v):
                prev = (f, v)
        if lo is None:
            return dict(converged=False, reason='no NV_first/D_first transition '
                        f'over l_in/l_isco in {span}')

        # lo -> NV_first (+1), hi -> D_first (-1)
        while abs(hi - lo) > tol * 0.5 * (hi + lo):
            mid = 0.5 * (lo + hi)
            v, out, info = flag(mid)
            if verbose:
                rs = info.get('r_s', np.nan) if info else np.nan
                print(f'   bisect l_in/l_isco={mid:9.6f}  {out:9s}  r={rs:8.4f}', flush=True)
            if not np.isfinite(v):
                break
            if v > 0:
                lo = mid
            else:
                hi = mid
                best = (mid, info, cache.get('sols'))
        if best is None:
            v, out, info = flag(hi)
            best = (hi, info, cache.get('sols'))
        f, info, sols = best
        l_in = lis * f
        sup, kind = None, 'not requested'
        if supersonic:
            try:
                sup, kind = self.continue_supersonic(info['r_s'], info['V_s'],
                                                     info['T_s'], l_in)
            except Exception as e:
                kind = f'lhopital failed: {e}'
        return dict(converged=True, l_in=l_in, l_isco=lis,
                    l_ratio=0.5*(lo + hi), l_ratio_hi=f,
                    r_s=info['r_s'], V_s=info['V_s'], T_s=info['T_s'],
                    critical_point=kind, sols=sols, sup=sup,
                    bracket=(lo, hi), r_out=r_out)

    def profile_from_sols(self, sols, l_in, npts=400, trust_frac=0.25):
        """Expand the two-stage solution into physical profiles on a log grid.

        TRIMS THE RELAXATION ZONE. The outer boundary condition (Omega = Omega_K,
        Q+ = Q-) is approximate, and the solution needs roughly two e-folds in r
        to settle onto the thin-disk attractor. Inside that zone Omega/Omega_K
        dives and snaps back to 1 at r_out, which is boundary artefact, not
        physics. Everything outside trust_frac * r_out is therefore dropped.
        Set trust_frac=None to see the raw integration.
        """
        xs = []
        for sol in sols:
            if sol is not None and sol.t.size > 1:
                xs.append((min(sol.t), max(sol.t), sol))
        if not xs:
            return None
        x_hi = max(h for _, h, _ in xs)
        x_lo = min(l for l, _, _ in xs)
        grid = np.linspace(x_hi, x_lo, npts)
        y = np.empty((2, grid.size))
        for i, x in enumerate(grid):
            for lo_, hi_, sol in xs:
                if lo_ - 1e-12 <= x <= hi_ + 1e-12:
                    y[:, i] = sol.sol(x)
                    break
            else:
                y[:, i] = np.nan
        ok = np.isfinite(y).all(axis=0)
        r_all = np.exp(grid)
        if trust_frac is not None:
            ok &= r_all <= trust_frac * np.exp(x_hi)
        if not ok.any():
            return None
        return self.profile(r_all[ok], y[:, ok], l_in)

    def profile(self, r, y, l_in):
        """Expand an integration into physical profiles."""
        out = {k: [] for k in ('r', 'Sigma', 'H', 'HoR', 'T', 'Omega', 'OmOmK',
                               'V', 'vR', 'beta', 'PrPg', 'cs2_vK2', 'ell',
                               'Qplus', 'Qminus', 'rho')}
        for rr, p, q in zip(r, y[0], y[1]):
            st = self.state(rr, -np.exp(p), np.exp(q), l_in)
            R = rr*self.Rg
            vK = C['c']/np.sqrt(rr)
            out['r'].append(rr);              out['Sigma'].append(st['Sigma'])
            out['H'].append(st['H']);         out['HoR'].append(st['H']/R)
            out['T'].append(st['T']);         out['Omega'].append(st['Omega'])
            out['OmOmK'].append(st['Omega_geo']/self.k.OmegaK(rr))
            out['V'].append(st['V']);         out['vR'].append(abs(st['V']))
            out['beta'].append(st['beta']);   out['PrPg'].append(st['Prad_Pgas'])
            out['cs2_vK2'].append(st['cs2']/vK**2)
            out['ell'].append(st['ell']);     out['rho'].append(st['rho'])
            out['Qminus'].append(self.cool(st['Sigma'], st['T'], st['H']))
            out['Qplus'].append(np.nan)
        return {k: np.array(v) for k, v in out.items()}
