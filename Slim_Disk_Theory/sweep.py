"""Sweep mdot with continuation and cache the slim-disk predictions.

Continuation is what makes this converge: each mdot is seeded with the
previous solution's l_in/l_isco, so the bracket search is narrow. A cold start
at mdot ~ 300 does not converge.

    python sweep.py --MBH 10 --a 0.9375 --alpha 0.1 --radii 10 20 30
"""
import os, sys, time, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from solver import SlimDisk

TS = '/lustre/orion/ast178/proj-shared/cbambic/Time_Series'


def solve_one(sd, seed, r_arm=12., tol=3e-3, verbose=False):
    """Try progressively wider brackets around the seed."""
    spans = ([(seed*0.94, seed*1.10), (seed*0.85, seed*1.25),
              (seed*0.70, seed*1.60)] if seed else []) + [(0.60, 1.80)]
    for sp in spans:
        res = sd.solve(r_out=max(300., 6.*sd.mdot), r_arm=r_arm, span=sp,
                       nscan=5, tol=tol, verbose=verbose)
        if res['converged']:
            return res
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--MBH', type=float, default=10.)
    p.add_argument('--a', type=float, default=0.9375)
    p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--radii', type=float, nargs='+', default=[10., 20., 30.])
    p.add_argument('--mdot-min', type=float, default=1.)
    p.add_argument('--mdot-max', type=float, default=500.)
    p.add_argument('--n', type=int, default=11)
    p.add_argument('--r-arm', type=float, default=12.)
    p.add_argument('--tag', default='Transition_XRB_HighMdot_SANE')
    a = p.parse_args()

    grid = np.geomspace(a.mdot_min, a.mdot_max, a.n)
    keys = ('Sigma', 'HoR', 'cs2_vK2', 'OmOmK', 'vR', 'PrPg', 'T', 'beta')
    out = {k: np.full((grid.size, len(a.radii)), np.nan) for k in keys}
    l_in = np.full(grid.size, np.nan)
    l_rat = np.full(grid.size, np.nan)
    r_son = np.full(grid.size, np.nan)
    seed = None
    l_isco = None

    for i, md in enumerate(grid):
        t0 = time.time()
        sd = SlimDisk(a.MBH, a.a, md, a.alpha)
        l_isco = sd.nt.l_in()
        res = solve_one(sd, seed, r_arm=a.r_arm)
        if not res['converged']:
            print(f'[{i+1}/{grid.size}] mdot={md:8.3f}  FAILED: {res["reason"]}',
                  flush=True)
            continue
        seed = res['l_ratio']
        l_in[i], l_rat[i], r_son[i] = res['l_in'], res['l_ratio'], res['r_s']
        prof = sd.profile_from_sols(res['sols'], res['l_in'])
        if prof is not None:
            for j, rt in enumerate(a.radii):
                if prof['r'].min() <= rt <= prof['r'].max():
                    k = np.argmin(abs(prof['r'] - rt))
                    for key in keys:
                        out[key][i, j] = prof[key][k]
        print(f'[{i+1}/{grid.size}] mdot={md:8.3f}  l_in/l_isco={res["l_ratio"]:.5f}'
              f'  r_s={res["r_s"]:7.4f}  Sigma(10)={out["Sigma"][i,0]:.4e}'
              f'  H/R(10)={out["HoR"][i,0]:.4f}  [{time.time()-t0:5.1f}s]', flush=True)

    fn = os.path.join(TS, f'slimdisk_ode_sweep_{a.tag}_alpha{a.alpha:g}.npz')
    np.savez(fn, mdot=grid, radii=np.array(a.radii), l_in=l_in, l_ratio=l_rat,
             l_isco=l_isco, r_sonic=r_son, MBH=a.MBH, a=a.a, alpha=a.alpha, **out)
    print(f'\nwrote {fn}')


if __name__ == '__main__':
    main()
