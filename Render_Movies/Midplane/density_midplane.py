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

    vec = True if (fieldlines or streamlines or rad_streamlines) else False

    # Define constants
    MBH = 10.0
    cgs = True if (two_temp or radiation) else False

    # Determine data extraction and save directories
    data_dir = cv.get_data_dir(sim, reduced, shared)
    save_dir = cv.get_save_dir(sim if title == None else title, 'Images', 
        f'{movie_title}', subtype='Midplane', reduced=reduced)

    # Ensure that nodes don't all attempt to make directory at same time
    timeit.sleep(np.random.rand())
    os.makedirs(save_dir, exist_ok=True)

    file_list = list(range(start_file, end_file + 1))
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    file_list_local = cv.get_file_list_local(file_list, size, rank, data_dir)

    read_args = {
        'cgs_grid' : cgs,
        'cgs_data' : cgs,
        'coords1D' : True,
        'MBH'      : MBH,
        'lowres1'  : lowres1,
        'lowres2'  : lowres2,
        'lowres3'  : lowres3,
        'r_max'    : 80.
    }

    plt_args = {
        'cmap'     : cmr.torch,
        'FS'       : FS,
        'lim'      : [1.e-8, 0.1],
        'log'      : True,
        'fix_pole' : 'average'
    }

    if vec:
        colors = ['white', 'limegreen', 'magenta']
        vec_args = {
           'dens'               : 10.0,
           'linewidth'          : 0.6,
           'color'              : colors[0],
           'broken_streamlines' : True,
           'arrowsize'          : 1.0
        }
    
    opt = dict(ha='left', va='center', color='black',
           bbox=dict(facecolor='none', edgecolor='none'))

    for file_num in file_list_local:
        MS = cv.Midplane_Slice(data_dir, file_num, **read_args)
        rho = {}
        rho['r'] = MS.calc_rho()

        X, Y = MS.calc_XY()

        Cart = figs.Cartesian(MS.H.params)
        fig  = plt.figure(figsize = figsize)
        gs   = gridspec.GridSpec(1, 1, figure = fig)
        ax   = fig.add_subplot(gs[0, 0])

        Cart.plot_xy(fig, ax, X, Y, rho, r=MS.r, theta=MS.th, phi=MS.phi, 
                     extend=cv.check_extend(rho['r'], plt_args['lim']), **plt_args)
        
        if vec: 
            X_uni, Y_uni = MS.calc_XY_uni(22., 100)
            mask = {'r' : np.sqrt(X['r'] * X['r'] + Y['r'] * Y['r']) > 22.}
            vec_args['mask'] = mask
        if fieldlines:
            Bx, By = MS.calc_B_cart(lr=True)
            Cart.plot_xy_vec(fig, ax, X, Y, X_uni, Y_uni, Bx, By, **vec_args)
        if streamlines:
            if fieldlines: vec_args['color'] = colors[1]
            uux, uuy = MS.calc_uu_cart(lr=True)
            Cart.plot_xy_vec(fig, ax, X, Y, X_uni, Y_uni, uux, uuy, **vec_args)
        if rad_streamlines:
            if fieldlines or streamlines:  vec_args['color'] = colors[1]
            if fieldlines and streamlines: vec_args['color'] = colors[2]
            uuRx, uuRy = MS.calc_uu_rad_cart(lr=True)
            Cart.plot_xy_vec(fig, ax, X, Y, X_uni, Y_uni, uuRx, uuRy, **vec_args)

        ax.set_xlim([-40, 40])
        ax.set_ylim([-22, 22])
        ax.set_aspect('equal')

        ax.set_xlabel(r'$x \: (R_g)$', fontsize=FS)
        ax.set_ylabel(r'$y \: (R_g)$', fontsize=FS)

        ax.text(0.05, 0.90, r'$\rho \: (\rm{g/cm}^3)$', 
                fontsize=FS+12, transform=ax.transAxes, **opt)
        ax.text(0.05, 0.10, r'$t = $' + cv.get_nice_time(MS.H.params['t']) + r' $R_g/c$', 
                fontsize=FS+6, transform=ax.transAxes, **opt)    
        ax.add_patch(patches.Circle((0,0), MS.H.params['Rin'], color='k', zorder=3))

        print("Saving figure ...")
        print("Saving file %d on rank %d" %(file_num, rank), flush=True)
        fig.tight_layout()
        plt.savefig(f'{save_dir}/{movie_title}_%06d.png' %(file_num), format='png')
        plt.close()

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="Midplane slice of fluid-frame density.")
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
    parser.add_argument('--title', type=str, required=False, default=None, 
                        help="""title of simulation used for Movie directory
                        if different from the simulation title in proj-shared""")
    parser.add_argument('--movie_title', type=str, required=False, 
                        default='Density', 
                        help="""title of movie (default: Density)""")
    args = parser.parse_args()
    main(**vars(args))
