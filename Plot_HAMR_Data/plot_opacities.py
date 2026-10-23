import sys
import os
import argparse
import numpy as np
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{file_dir}')
asys_path = os.path.join(os.path.dirname(file_dir), "Analyze_HAMR_Data")
sys.path.insert(0, f'{asys_path}')
import compute_variables as cv
import hamr_read as hr
import radiation as rad

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.style as style
style.use('seaborn-v0_8-bright')
matplotlib.rcParams['figure.dpi'] = 300

FS = 22

# Emission opacities depend only on (rho, Te, b2), so u_rad is inert for
# every channel below. Compton is the exception: it scatters off an existing
# radiation field, so q_IC carries u_rad and the emission panels have to name
# a radiation temperature (--Tr0). Absorption is the opposite case -- every
# channel except molecular carries Te, and bound-free, free-free, CHIANTI and
# synchrotron carry Tr too, so those panels sweep Tr and set u_rad = a Tr^4.

# Opacity.__init__ recovers Tr as (u_rad / aRAD)**0.25 in float32, so the
# intermediate u_rad / aRAD = Tr**4 overflows once Tr exceeds this. The same
# ceiling applies to production runs, not just to this figure.
TR_MAX_F32 = np.float64(np.finfo(np.float32).max) ** 0.25

# PROCESS_KEYS sets the plotting and legend order. LOWT_KEYS and HIGHT_KEYS
# are summed linearly, so their order carries no meaning.
PROCESS_KEYS = ['free_free', 'bound_free', 'CHIANTI',
                'hydrogen', 'molecular', 'synchrotron']
LOWT_KEYS    = ['hydrogen', 'molecular']
HIGHT_KEYS   = ['free_free', 'bound_free', 'CHIANTI']

PROCESS_LABEL = {
    'free_free'   : r'$\rm{free-free}$',
    'bound_free'  : r'$\rm{bound-free}$',
    'CHIANTI'     : r'$\rm{CHIANTI}$',
    'hydrogen'    : r'$\rm{H}^{-}$',
    'molecular'   : r'$\rm{molecular}$',
    'synchrotron' : r'$\rm{synchrotron}$',
}
PROCESS_COLOR = {
    'free_free'   : 'teal',
    'bound_free'  : 'dodgerblue',
    'CHIANTI'     : 'crimson',
    'hydrogen'    : 'forestgreen',
    'molecular'   : 'saddlebrown',
    'synchrotron' : 'darkorange',
}
# Compton is not an opacity channel and so takes no part in the harmonic
# sum. It is kept out of PROCESS_KEYS and drawn as a separate rate.
COMPTON_COLOR = 'darkviolet'
COMPTON_LABEL = r'$\rm{Compton}$'

def main(**kwargs):
    Te_min    = kwargs.get('Te_min')
    Te_max    = kwargs.get('Te_max')
    n_Te      = kwargs.get('n_Te')
    rho_min   = kwargs.get('rho_min')
    rho_max   = kwargs.get('rho_max')
    n_rho     = kwargs.get('n_rho')
    B_min     = kwargs.get('B_min')
    B_max     = kwargs.get('B_max')
    Tr_min    = kwargs.get('Tr_min')
    Tr_max    = kwargs.get('Tr_max')
    n_Tr      = kwargs.get('n_Tr')
    Te0       = kwargs.get('Te0')
    Tr0       = kwargs.get('Tr0')
    rho0      = kwargs.get('rho0')
    B0        = kwargs.get('B0')
    AGN       = kwargs.get('AGN')
    cpp       = kwargs.get('cpp')
    breakdown = kwargs.get('breakdown')
    plot_title  = kwargs.get('plot_title')
    disable_TeX = kwargs.get('disable_TeX')

    matplotlib.rcParams['text.usetex'] = False if disable_TeX else True
    save_dir = cv.user_path + '/PDF'
    os.makedirs(save_dir, exist_ok=True)

    Te = np.logspace(np.log10(Te_min), np.log10(Te_max), n_Te)

    if breakdown:
        if cpp:
            # The C++ path only writes '*_total'; the per-channel opacities
            # exist solely on the Python path.
            print('--breakdown requires the Python opacities; '
                  'ignoring --cpp.', flush=True)
        if Tr_max > TR_MAX_F32:
            print(f'--Tr_max = {Tr_max:.2e} K exceeds the float32 ceiling '
                  f'{TR_MAX_F32:.2e} K on (u_rad / aRAD)**0.25; the '
                  'absorption panels will overflow above it.', flush=True)
        Tr = np.logspace(np.log10(Tr_min), np.log10(Tr_max), n_Tr)
        if not Tr[0] <= Te0 <= Tr[-1]:
            print(f'--Te0 = {Te0:.2e} K lies outside the Tr sweep '
                  f'[{Tr[0]:.2e}, {Tr[-1]:.2e}] K, so the LTE crossing will '
                  'not appear in the absorption panels.', flush=True)
        fname = plot_breakdown(Te, Tr, Te0, Tr0, rho0, B0, AGN,
                               save_dir, plot_title)
    else:
        rho    = np.logspace(np.log10(rho_min), np.log10(rho_max), n_rho)
        B_list = np.logspace(np.log10(B_min), np.log10(B_max), 9)
        fname  = plot_qemi_grid(Te, rho, B_list, AGN, cpp,
                                calc_u_rad(Tr0), save_dir, plot_title)
    print(f'Saved {fname}', flush=True)

def build_fields(rho, Te, B, u_rad, n):
    """Shape the inputs into the (1, n, 1, 1) 4D fields the toolkit requires.
    Any argument may be a scalar or a length-n sweep: the emission panels
    sweep Te, the absorption panels sweep u_rad. b2 is taken as B^2 with B
    in Gauss."""
    def field(value):
        return np.array(np.broadcast_to(value, n),
                        dtype=np.float32).reshape(1, n, 1, 1)
    return dict(rho=field(rho), Te=field(Te),
                u_rad=field(u_rad), b2=field(B * B))

def calc_u_rad(Tr):
    """Radiation energy density in erg/cm^3 for a blackbody at Tr."""
    return hr.CONSTANTS['aRAD'] * (np.float64(Tr) ** 4)

def calc_qIC(rho, Te, Tr0):
    """Compton cooling rate in erg/cm^3/s, swept over Te at fixed radiation
    temperature. Sadowski+ (2017) Equation 35 is signed and proportional to
    (Te - Tr): positive is cooling. Only the cooling branch belongs on an
    emission panel, so the heating branch (Te < Tr0) is zeroed, and zeros
    simply do not appear on a log axis."""
    n   = Te.size
    f   = build_fields(rho, Te, 0., calc_u_rad(Tr0), n)
    Rad = rad.Radiation(pars={'density_scale': 1.0},
                        CONSTANTS=hr.CONSTANTS, cgs=True)
    return np.ravel(Rad.calc_qIC(f['rho'], f['Te'], f['u_rad'], cooling=True))

def calc_qemi(rho, Te, B, AGN, cpp, u_rad):
    """Total emission rate in erg/cm^3/s for scalar rho and B."""
    f   = build_fields(rho, Te, B, u_rad, Te.size)
    Rad = rad.Radiation(pars={'density_scale': 1.0},
                        CONSTANTS=hr.CONSTANTS, cgs=True)
    return np.ravel(Rad.calc_q_emi(f['rho'], f['Te'], f['u_rad'],
                                   b2=f['b2'], XRB=(not AGN), AGN=AGN,
                                   cpp=cpp))

def calc_q_channels(f, n, mechanism, factor, AGN):
    """Rate in erg/cm^3/s for each individual channel, for the two groups
    they combine into, and for the total.

    Within lowT and within highT the opacities add linearly. The two groups
    then combine harmonically, and synchrotron adds on top:
        kappa_total = 1/(1/kappa_lowT + 1/kappa_highT) + kappa_sync
    so the individual channels are NOT additive across that boundary.
    """
    Opac = rad.Opacity(dict.fromkeys(PROCESS_KEYS + ['electron_scattering']),
                       rho=f['rho'], Te=f['Te'], u_rad=f['u_rad'],
                       pars={'density_scale': 1.0}, CONSTANTS=hr.CONSTANTS,
                       b2=f['b2'], XRB=(not AGN), AGN=AGN,
                       mechanism=mechanism, cpp=False)
    suffix = 'emi' if mechanism == 'emission' else 'abs'

    def kappa(keys):
        total = 0.
        for key in keys:
            total = total + np.ravel(np.broadcast_to(
                Opac.opacities['%s_%s' %(key, suffix)], (1, n, 1, 1)))
        return total

    q = {key: kappa([key]) * factor for key in PROCESS_KEYS}
    q['lowT']  = kappa(LOWT_KEYS)  * factor
    q['highT'] = kappa(HIGHT_KEYS) * factor
    q['total'] = np.ravel(Opac.opacities['%s_total' %mechanism]) * factor
    return q

def calc_qemi_channels(rho, Te, B, AGN, u_rad):
    """Emission channels at scalar rho and B, swept over Te. The common
    factor is q_emi = kappa_emi rho a Te^4 c. u_rad is passed through only
    because the constructor requires it; no emission channel uses it."""
    n      = Te.size
    f      = build_fields(rho, Te, B, u_rad, n)
    factor = (rho * hr.CONSTANTS['aRAD']
              * (Te.astype(np.float64) ** 4) * hr.CONSTANTS['c'])
    return calc_q_channels(f, n, 'emission', factor, AGN)

def calc_qabs_channels(rho, Tr, Te0, B, AGN):
    """Absorption channels at scalar rho, B and electron temperature, swept
    over the radiation temperature. The common factor is
    q_abs = kappa_abs rho u_rad c with u_rad = a Tr^4, so Tr enters both
    through the opacity and through the radiation energy density."""
    n      = Tr.size
    u_rad  = calc_u_rad(Tr)
    f      = build_fields(rho, Te0, B, u_rad, n)
    factor = rho * u_rad * hr.CONSTANTS['c']
    return calc_q_channels(f, n, 'absorption', factor, AGN)

def set_ylim_from(ax, ref, below=1.e2, above=1.e2):
    """Scale the axes to the reference curve. Channels far above it have been
    switched off by the harmonic sum, and simply leave the frame."""
    good = ref[np.isfinite(ref) & (ref > 0.)]
    if good.size:
        ax.set_ylim(good.min() / below, good.max() * above)

def plot_qemi_grid(Te, rho, B_list, AGN, cpp, u_rad, save_dir, plot_title):
    """3x3 grid: one panel per field strength, one curve per density.
    Every panel shares one y-range so the panels can be compared by eye.
    Opacity channels only -- Compton would need a second curve per density
    and appears on the --breakdown panels instead."""
    fig  = plt.figure(figsize=[21., 18.])
    gs   = gridspec.GridSpec(3, 3, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]
    cmap = plt.get_cmap('viridis')

    curves = [[calc_qemi(rho_i, Te, B, AGN, cpp, u_rad) for rho_i in rho]
              for B in B_list]
    stacked = np.concatenate([q for row in curves for q in row])
    good    = stacked[np.isfinite(stacked) & (stacked > 0.)]
    ylim    = (good.min() / 2., good.max() * 2.)

    for ax, B, row in zip(axes, B_list, curves):
        for i, q in enumerate(row):
            ax.plot(Te, q, lw=2, color=cmap(i / max(len(rho) - 1, 1)),
                    label=r'$10^{%.0f}$' %np.log10(rho[i]))
        ax.set_ylim(ylim)
        ax.text(0.05, 0.90, r'$B = 10^{%.0f}\ \rm{G}$' %np.log10(B),
                transform=ax.transAxes, fontsize=FS)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(Te[0], Te[-1])
        cv.grid(ax, alpha=0.2, color='black')
        ax.tick_params(labelsize=FS-6)

    for ax in axes[6:]:
        ax.set_xlabel(r'$T_e\ [\rm{K}]$', fontsize=FS)
    for ax in axes[::3]:
        ax.set_ylabel(r'$q_{\rm emi}\ [\rm{erg\ cm^{-3}\ s^{-1}}]$',
                      fontsize=FS)
    axes[0].legend(loc='lower right', fontsize=FS-8, frameon=False, ncol=2,
                   title=r'$\rho\ [\rm{g\ cm^{-3}}]$', title_fontsize=FS-8)

    fig.tight_layout()
    fname = f'{save_dir}/{plot_title}.png'
    plt.savefig(fname, format='png'); plt.close()
    return fname

def plot_combination(ax, x, q, ylabel, ref=None):
    """Top-row panel: how the model combines. lowT and highT are joined
    harmonically and synchrotron is then added linearly, so the total tracks
    whichever of the two groups is smaller.

    The y-range follows the total unless ref is given. Passing a wider ref is
    how Compton stays on the frame: a channel above the total is one the
    harmonic sum switched off, but Compton is above the total because it
    wins, and the two cases must not look alike."""
    ax.plot(x, q['total'], lw=4.0, color='black', label=r'$\rm{total}$')
    ax.plot(x, q['highT'], lw=2.5, color='slategray',
            label=r'$\rm{high}\ T\ (ff + bf + CHIANTI)$')
    ax.plot(x, q['lowT'],  lw=2.5, color='seagreen',
            label=r'$\rm{low}\ T\ (H^{-} + mol)$')
    ax.plot(x, q['synchrotron'], lw=2.5, color=PROCESS_COLOR['synchrotron'],
            label=PROCESS_LABEL['synchrotron'])
    set_ylim_from(ax, q['total'] if ref is None else ref)
    ax.set_ylabel(ylabel, fontsize=FS)

def plot_channels(ax, x, q, ylabel, ylim):
    """Bottom-row panel: the individual channels, on the y-range of the panel
    above. That shared range is the point -- a channel sitting far above the
    total has been switched off by the harmonic sum."""
    for key in PROCESS_KEYS:
        ax.plot(x, q[key], lw=1.8, ls='--', alpha=0.9,
                color=PROCESS_COLOR[key], label=PROCESS_LABEL[key])
    ax.plot(x, q['total'], lw=3.0, color='black', label=r'$\rm{total}$')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=FS)

def plot_breakdown(Te, Tr, Te0, Tr0, rho0, B0, AGN, save_dir, plot_title):
    """Four panels mirroring the structure of the opacity model. The left
    column sweeps Te at fixed Tr0; no emission opacity depends on it, but the
    Compton rate does. The right column holds Te fixed and sweeps Tr, which
    is the only way to see absorption: at Tr = Te every channel reduces
    exactly to its emission counterpart, so q_abs crosses q_emi there and the
    net cooling vanishes.
    """
    u_rad = calc_u_rad(Tr0)
    q_emi = calc_qemi_channels(rho0, Te, B0, AGN, u_rad)
    q_abs = calc_qabs_channels(rho0, Tr, Te0, B0, AGN)
    q_IC  = calc_qIC(rho0, Te, Tr0)
    # q_emi is a single number on the right column: no emission channel
    # depends on Tr, so it is flat across the whole sweep. Compton is left
    # out of it deliberately -- the exact q_abs = q_emi crossing at Tr = Te
    # is a statement about the opacities alone.
    q_emi_ref = calc_qemi_channels(rho0, np.array([Te0]), B0, AGN,
                                   u_rad)['total'][0]

    fig = plt.figure(figsize=[22., 13.])
    gs  = gridspec.GridSpec(2, 2, figure=fig)
    ax_emi_grp, ax_abs_grp, ax_emi_ch, ax_abs_ch = [
        fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    label_emi = r'$q_{\rm emi}\ [\rm{erg\ cm^{-3}\ s^{-1}}]$'
    label_abs = r'$q_{\rm abs}\ [\rm{erg\ cm^{-3}\ s^{-1}}]$'
    # Compton runs far above the opacity total once Te exceeds Tr0, so the
    # emission panels are scaled to both.
    plot_combination(ax_emi_grp, Te, q_emi, label_emi,
                     ref=np.concatenate([q_emi['total'], q_IC]))
    plot_combination(ax_abs_grp, Tr, q_abs, label_abs)
    plot_channels(ax_emi_ch, Te, q_emi, label_emi, ax_emi_grp.get_ylim())
    plot_channels(ax_abs_ch, Tr, q_abs, label_abs, ax_abs_grp.get_ylim())

    # Compton sits outside the opacity model, so it is drawn on the emission
    # panels without entering 'total'. It only cools for Te > Tr0; the
    # heating branch below that is zeroed and drops off the log axis.
    for ax in (ax_emi_grp, ax_emi_ch):
        ax.plot(Te, q_IC, lw=2.5, ls='-.', color=COMPTON_COLOR,
                label=COMPTON_LABEL)
        ax.axvline(Tr0, lw=1.8, ls=':', color='black',
                   label=r'$T_e = T_{\rm rad,0}$')

    for ax in (ax_abs_grp, ax_abs_ch):
        ax.axhline(q_emi_ref, lw=2.0, ls='-.', color='dimgray',
                   label=r'$q_{\rm emi}(T_e)$')
        ax.axvline(Te0, lw=1.8, ls=':', color='black',
                   label=r'$T_r = T_e\ \rm{(LTE)}$')

    for ax, x in ((ax_emi_grp, Te), (ax_emi_ch, Te),
                  (ax_abs_grp, Tr), (ax_abs_ch, Tr)):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(x[0], x[-1])
        cv.grid(ax, alpha=0.2, color='black')
        ax.tick_params(labelsize=FS-4)
    for ax in (ax_emi_grp, ax_abs_grp):
        ax.legend(loc='upper left', fontsize=FS-8, frameon=False)
    for ax in (ax_emi_ch, ax_abs_ch):
        ax.legend(loc='upper left', fontsize=FS-8, frameon=False, ncol=2)

    ax_emi_ch.set_xlabel(r'$T_e\ [\rm{K}]$', fontsize=FS)
    ax_abs_ch.set_xlabel(r'$T_{\rm rad}\ [\rm{K}]$', fontsize=FS)
    ax_emi_grp.set_title(r'$\rm{emission}: \ \rho = 10^{%.1f}\ '
                         r'\rm{g\ cm^{-3}}, \ B = 10^{%.1f}\ \rm{G}, \ '
                         r'T_{\rm rad} = 10^{%.1f}\ \rm{K}$'
                         %(np.log10(rho0), np.log10(B0), np.log10(Tr0)),
                         fontsize=FS)
    ax_abs_grp.set_title(r'$\rm{absorption}: \ T_e = 10^{%.1f}\ \rm{K}$'
                         %np.log10(Te0), fontsize=FS)

    fig.tight_layout()
    fname = f'{save_dir}/{plot_title}.png'
    plt.savefig(fname, format='png'); plt.close()
    return fname

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description="""Plot the radiative emission and
                        absorption rates predicted by the opacity model.
                        Requires no simulation data.""")
    parser.add_argument('--breakdown', action='store_true', required=False,
                        help="""boolean to plot the individual channels at
                        one density and field strength (--rho0, --B0)
                        instead of the panel grid: emission against Te in
                        the left column, absorption against Tr at fixed
                        --Te0 in the right column.""")
    parser.add_argument('--Te_min', type=float, default=1.e4,
                        required=False,
                        help="""minimum electron temperature in K""")
    parser.add_argument('--Te_max', type=float, default=1.e11,
                        required=False,
                        help="""maximum electron temperature in K""")
    parser.add_argument('--n_Te', type=int, default=256,
                        required=False,
                        help="""number of electron temperature samples""")
    parser.add_argument('--rho_min', type=float, default=1.e-10,
                        required=False,
                        help="""minimum density in g/cm^3""")
    parser.add_argument('--rho_max', type=float, default=1.e-5,
                        required=False,
                        help="""maximum density in g/cm^3""")
    parser.add_argument('--n_rho', type=int, default=6,
                        required=False,
                        help="""number of density curves per panel""")
    parser.add_argument('--B_min', type=float, default=1.e1,
                        required=False,
                        help="""minimum magnetic field strength in Gauss""")
    parser.add_argument('--B_max', type=float, default=1.e9,
                        required=False,
                        help="""maximum magnetic field strength in Gauss""")
    parser.add_argument('--Tr_min', type=float, default=1.e4,
                        required=False,
                        help="""minimum radiation temperature in K, swept by
                        the --breakdown absorption panels""")
    parser.add_argument('--Tr_max', type=float, default=1.e9,
                        required=False,
                        help="""maximum radiation temperature in K, swept by
                        the --breakdown absorption panels. Capped below the
                        Te ceiling because the toolkit stores u_rad rather
                        than Tr, and Tr**4 overflows float32 near 4.3e9 K.""")
    parser.add_argument('--n_Tr', type=int, default=256,
                        required=False,
                        help="""number of radiation temperature samples""")
    parser.add_argument('--Tr0', type=float, default=1.e7,
                        required=False,
                        help="""radiation temperature in K held fixed while
                        the emission panels sweep Te. No emission opacity
                        depends on it, but the Compton rate does, and it
                        changes sign at Te = Tr0.""")
    parser.add_argument('--Te0', type=float, default=1.e7,
                        required=False,
                        help="""electron temperature in K held fixed while
                        the --breakdown absorption panels sweep Tr. Should
                        fall inside [--Tr_min, --Tr_max] so that the LTE
                        crossing q_abs = q_emi is visible.""")
    parser.add_argument('--rho0', type=float, default=1.e-7,
                        required=False,
                        help="""density in g/cm^3 used for --breakdown""")
    parser.add_argument('--B0', type=float, default=1.e5,
                        required=False,
                        help="""magnetic field strength in Gauss used
                        for --breakdown""")
    parser.add_argument('--AGN', action='store_true', required=False,
                        help="""boolean to use the AGN synchrotron
                        coefficients instead of the XRB coefficients""")
    parser.add_argument('--cpp', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="""boolean to determine if the C++ accelerated
                        opacities should be used.""")
    parser.add_argument('--disable_TeX', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="""boolean to disable TeX in plotting""")
    parser.add_argument('--plot_title', type=str, required=False,
                        default='Opacities',
                        help="""title of PDF output""")
    args = parser.parse_args()
    main(**vars(args))
