import sys
import os
import argparse
import numpy as np
from mpi4py import MPI
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{file_dir}/../../Analyze_HAMR_Data')
import compute_variables as cv
import hamr_read as hr
import time as timeit

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.rcParams['text.usetex'] = True
import matplotlib.patches as patches
import cmasher as cmr # colormaps I like
import matplotlib.style as style
style.use('dark_background')
matplotlib.rcParams['figure.dpi'] = 300
import figs

figsize = [32.0, 18.0]
FS      = 60
plt.rcParams['xtick.labelsize'] = FS
plt.rcParams['ytick.labelsize'] = FS

def main(**kwargs):
    sim       = kwargs.get('sim')
    start_file= int(kwargs.get('start_file'))
    end_file  = int(kwargs.get('end_file'))
    lowres1   = kwargs.get('lowres1')
    lowres2   = kwargs.get('lowres2')
    lowres3   = kwargs.get('lowres3')
    reduced   = kwargs.get('reduced')
    shared    = kwargs.get('shared')
    title     = kwargs.get('title')
    movie_title = kwargs.get('movie_title')
    radiation   = kwargs.get('radiation')
    two_temp    = kwargs.get('two_temp')
    fieldlines  = kwargs.get('fieldlines')
    streamlines = kwargs.get('streamlines')
    rad_streamlines = kwargs.get('rad_streamlines')
    sigma_contour   = kwargs.get('sigma_contour')
    scat_photo      = kwargs.get('scattering_photosphere')
    eff_photo       = kwargs.get('effective_photosphere')
    z_integral      = kwargs.get('integrate_in_z')

    vec = True if (fieldlines or streamlines or rad_streamlines) else False
    draw_contour = True if (sigma_contour != 0 or scat_photo or eff_photo) else False
    photospheres = {}
    photospheres['scattering'] = True if scat_photo else False
    photospheres['effective']  = True if eff_photo else False
    photospheres['integral']   = 'z' if z_integral else 'theta'

    # Define constants
    MBH = 10.0
    cgs = True if (two_temp or radiation) else False

    # Determine data extraction and save directories
    data_dir = cv.get_data_dir(sim, reduced, shared)
    save_dir = cv.get_save_dir(sim if title == None else title, 'Images', 
        f'{movie_title}', subtype='Poloidal', reduced=reduced)

    # Ensure that nodes don't all attempt to make directory at same time
    timeit.sleep(np.random.rand())
    os.makedirs(save_dir, exist_ok=True)

    file_list = list(range(start_file, end_file + 1))
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    file_list_local = cv.get_file_list_local(file_list, size, rank, data_dir)

    read_args_r = {
        'cgs_grid' : cgs,
        'cgs_data' : cgs,
        'coords1D' : True,
        'MBH'      : MBH,
        'lowres1'  : lowres1,
        'lowres2'  : lowres2,
        'lowres3'  : lowres3,
        'r_max'    : 80.,
        'k_min'    : 0,
        'k_max'    : 1
    }

    read_args_l = read_args_r.copy()

    plt_args = {
        'cmap'     : cmr.iceburn,
        'FS'       : FS,
        'lim'      : [1.e4, 1.e12],
        'log'      : True,
        'fix_pole' : 'constant'
    }

    if (vec or draw_contour): colors = ['white', 'limegreen', 'magenta']
    if vec:
        vec_args = {
           'dens'               : 10.0,
           'linewidth'          : 0.6,
           'color'              : colors[0],
           'broken_streamlines' : True,
           'arrowsize'          : 1.0
        }

    for file_num in file_list_local:
        PS_r = cv.Phi_Slice(data_dir, file_num, **read_args_r)
        X, Z = PS_r.calc_XZ(0)
        ipi  = int(PS_r.nphi / 2 - 1)
        read_args_l['k_min'], read_args_l['k_max'] = ipi, ipi + 1
        PS_l = cv.Phi_Slice(data_dir, file_num, **read_args_l)
        Ti = {
            'r' : PS_r.calc_Ti(0),
            'l' : PS_l.calc_Ti(0)
        }

        if (draw_contour): 
            plt_args = cv.get_contours(plt_args, PS_r, PS_l, 
                colors, sigma_contour, photospheres)
            
        Cart = figs.Cartesian(PS_r.H.params)
        fig  = plt.figure(figsize = figsize)
        gs   = gridspec.GridSpec(1, 1, figure = fig)
        ax   = fig.add_subplot(gs[0, 0])

        Cart.plot_xz(fig, ax, X, Z, Ti, r=PS_r.r, theta=PS_r.th, phi=PS_r.phi, 
                     extend=cv.check_extend(Ti['r'], plt_args['lim']), **plt_args)
        
        if vec: 
            X_uni, Z_uni = PS_r.calc_XZ_uni(22., 100)
            mask = {
                'r' : np.sqrt(X['r'] * X['r'] + Z['r'] * Z['r']) > 22.,
                'l' : np.sqrt(X['l'] * X['l'] + Z['l'] * Z['l']) > 22.
            }
            vec_args['mask'] = mask
        if fieldlines:
            Bx, Bz           = PS_r.calc_B_cyl(0, lr=True)
            Bx['l'], Bz['l'] = PS_l.calc_B_cyl(0, lr=False)
            Bx['l']         *= -1.
            Cart.plot_xz_vec(fig, ax, X, Z, X_uni, Z_uni, Bx, Bz, **vec_args)
        if streamlines:
            if fieldlines: vec_args['color'] = colors[1]
            uux, uuz           = PS_r.calc_uu_cyl(0, lr=True)
            uux['l'], uuz['l'] = PS_l.calc_uu_cyl(0, lr=False)
            uux['l']          *= -1.
            Cart.plot_xz_vec(fig, ax, X, Z, X_uni, Z_uni, uux, uuz, **vec_args)
        if rad_streamlines:
            if fieldlines or streamlines:  vec_args['color'] = colors[1]
            if fieldlines and streamlines: vec_args['color'] = colors[2]
            uuRx, uuRz           = PS_r.calc_uu_rad_cyl(0, lr=True)
            uuRx['l'], uuRz['l'] = PS_l.calc_uu_rad_cyl(0, lr=False)
            uuRx['l']           *= -1.
            Cart.plot_xz_vec(fig, ax, X, Z, X_uni, Z_uni, uuRx, uuRz, **vec_args)

        ax.set_xlim([-20, 20])
        ax.set_ylim([-11, 11])
        ax.set_aspect('equal')

        ax.set_xlabel(r'$x \: (R_g)$', fontsize=FS)
        ax.set_ylabel(r'$z \: (R_g)$', fontsize=FS)
        
        opt = dict(ha='left', va='center', color='white',
            bbox=dict(facecolor='none', edgecolor='none'))

        ax.text(0.05, 0.90, r'$T_i \: (\rm{K})$', 
                fontsize=FS+12, transform=ax.transAxes, **opt)
        opt['bbox'] = dict(facecolor='black', alpha=0.4, edgecolor='none')
        ax.text(0.05, 0.10, r'$t = $' + cv.get_nice_time(PS_r.H.params['t']) + r' $R_g/c$', 
                fontsize=FS+6, transform=ax.transAxes, **opt)    
        ax.add_patch(patches.Circle((0,0), PS_r.H.params['Rin'], color='k', zorder=3))

        print("Saving figure ...")
        print("Saving file %d on rank %d" %(file_num, rank), flush=True)
        fig.tight_layout()
        plt.savefig(f'{save_dir}/{movie_title}_%06d.png' %(file_num), format='png')
        plt.close()

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(description="Poloidal slice of ion temperature.")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('start_file', type=int, help = """dump number
                        of start file in movie extraction""")
    parser.add_argument('end_file', type=int, help = """dump number
                        of end file in movie extraction""")
    parser.add_argument('--lowres1', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in x1""")
    parser.add_argument('--lowres2', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in x2""")
    parser.add_argument('--lowres3', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in x3""")
    parser.add_argument('--radiation', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if radiation quantities
                        should be included in movie extraction 
                        (default: True).""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if two-temperature 
                        physics should be included in movie extraction 
                        (default: True).""")    
    parser.add_argument('--fieldlines', action='store_true', required=False,
                        default=False,
                        help="""boolean to determine if magnetic-field lines 
                        should be drawn (default: False).""")
    parser.add_argument('--streamlines', action='store_true', required=False,
                        default=False,
                        help="""boolean to determine if 4-velocity streamlines 
                        should be drawn (default: False).""")
    parser.add_argument('--rad_streamlines', action='store_true', required=False,
                        default=False,
                        help="""boolean to determine if 
                        radiation 4-velocity streamlines 
                        should be drawn (default: False).""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if movie extraction
                        should be done on reduced data.""")
    parser.add_argument('--shared', action='store_true', required=False, 
                        help="""boolean set to determine if proj-shared 
                        should be used as the location of the data directory""")
    parser.add_argument('--sigma_contour', type=float, default=0,
                        required=False,
                        help = """draw sigma contour for specified value of sigma
                        (default: 0)""")
    parser.add_argument('--scattering_photosphere', action='store_true', 
                        required=False, 
                        help="""boolean to determine if scattering photosphere 
                        should be drawn""")
    parser.add_argument('--effective_photosphere', action='store_true', 
                        required=False, 
                        help="""boolean to determine if effective absorption 
                        photosphere should be drawn (based on all opacities)""")
    parser.add_argument('--integrate_in_z', action='store_true', 
                        required=False, 
                        help="""boolean to determine if photospheres should be 
                        computed from theta integral (default) vs z-integral""")
    parser.add_argument('--title', type=str, required=False, default=None, 
                        help="""title of simulation used for Movie directory
                        if different from the simulation title in proj-shared""")
    parser.add_argument('--movie_title', type=str, required=False, 
                        default='Ion_Temperature', 
                        help="""title of movie (default: Ion_Temperature)""")
    args = parser.parse_args()
    main(**vars(args))
