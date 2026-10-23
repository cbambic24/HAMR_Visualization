"""Solve the relativistic slim disk and write profiles + a figure.

    python run_slim_disk.py --MBH 10 --a 0.9375 --mdot 100 --alpha 0.043

The four inputs are MBH [Msun], a, mdot [Eddington], alpha. Nothing is fitted;
l_in is PREDICTED by the regularity condition at the sonic point.
"""
import os
import sys
import argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import SlimDisk
from physics import CONST

PDF = '/lustre/orion/ast178/proj-shared/cbambic/PDF'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--MBH', type=float, required=True, help='black hole mass [Msun]')
    p.add_argument('--a', type=float, required=True, help='spin parameter')
    p.add_argument('--mdot', type=float, required=True, help='Mdot / Mdot_Edd')
    p.add_argument('--alpha', type=float, required=True, help='viscosity parameter')
    p.add_argument('--cooling', default='lte', choices=['lte', 'taueff'])
    p.add_argument('--supersonic', action='store_true',
                   help='also continue inside the sonic point (slow, stiff)')
    p.add_argument('--r-out', type=float, default=300.)
    p.add_argument('--out', default=None, help='npz output path')
    p.add_argument('--fig', default=None, help='figure output path')
    a = p.parse_args()

    sd = SlimDisk(a.MBH, a.a, a.mdot, a.alpha, cooling=a.cooling)
    print(f'MBH={a.MBH} Msun   a={a.a}   mdot={a.mdot}   alpha={a.alpha}')
    print(f'Rg        = {sd.Rg:.5e} cm')
    print(f'Mdot_Edd  = {sd.MdotEdd:.5e} g/s   (eta_Edd = {sd.eta_edd:.5f})')
    print(f'Mdot      = {sd.Mdot:.5e} g/s')
    print(f'r_isco    = {sd.risco:.5f}   r_hor = {sd.rhor:.5f}')
    print(f'l_isco    = {sd.nt.l_in():.6f}   eta_NT = {sd.nt.efficiency():.6f}')
    print('\nshooting for l_in:')
    res = sd.solve(r_out=max(a.r_out, 6.*a.mdot), supersonic=a.supersonic)
    if not res['converged']:
        print(f"\nNOT CONVERGED: {res['reason']}")
        return 1
    print(f"\nl_in        = {res['l_in']:.6f}  ({res['l_ratio']:.5f} l_isco)")
    print(f"sonic point = {res['r_s']:.5f} Rg   ({res['critical_point']})")
    print(f"V_s         = {res['V_s']:.5e} c")
    print(f"T_s         = {res['T_s']:.5e} K")

    prof = sd.profile_from_sols(res['sols'], res['l_in'])
    if prof is None:
        print('no profile could be reconstructed')
        return 1
    tag = f"M{a.MBH:g}_a{a.a:g}_mdot{a.mdot:g}_alpha{a.alpha:g}"
    out = a.out or f'slimdisk_{tag}.npz'
    np.savez(out, l_in=res['l_in'], l_isco=res['l_isco'], r_s=res['r_s'],
             critical_point=res['critical_point'], MBH=a.MBH, a=a.a,
             mdot=a.mdot, alpha=a.alpha, Rg=sd.Rg, **prof)
    print(f'\nprofiles -> {out}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['figure.dpi'] = 150
    matplotlib.rcParams['text.usetex'] = False
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    panels = [('Sigma', r'$\Sigma$ [g cm$^{-2}$]', True),
              ('cs2_vK2', r'$c_s^2/v_K^2$', True),
              ('OmOmK', r'$\Omega/\Omega_K$', False),
              ('vR', r'$|v_r|/c$', True),
              ('PrPg', r'$P_{\rm rad}/P_{\rm gas}$', True),
              ('T', r'$T$ [K]', True)]
    for axi, (key, lab, logy) in zip(ax.ravel(), panels):
        axi.plot(prof['r'], prof[key], 'k-', lw=1.6)
        axi.set_xscale('log')
        if logy:
            axi.set_yscale('log')
        axi.set_xlabel(r'$r\ [R_g]$')
        axi.set_ylabel(lab)
        axi.axvline(sd.risco, color='0.6', ls=':', lw=1)
        axi.axvline(res['r_s'], color='crimson', ls='--', lw=1)
    fig.suptitle(f"slim disk  M={a.MBH:g} $M_\\odot$, a={a.a:g}, "
                 f"$\\dot m$={a.mdot:g}, $\\alpha$={a.alpha:g}   |   "
                 f"$l_{{\\rm in}}$={res['l_in']:.4f} = "
                 f"{res['l_ratio']:.4f} $l_{{\\rm ISCO}}$, "
                 f"$r_s$={res['r_s']:.3f} ({res['critical_point']})")
    fig.tight_layout()
    figp = a.fig or os.path.join(PDF, f'SlimDiskODE_Profiles_{tag}.png')
    fig.savefig(figp, bbox_inches='tight')
    print(f'figure   -> {figp}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
