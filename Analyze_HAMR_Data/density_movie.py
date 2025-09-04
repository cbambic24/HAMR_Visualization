import sys
import os
import argparse
import numpy as np
sys.path.insert(0, '/lustre/orion/ast178/proj-shared/' +
    'cbambic/HAMR_Visualization/Read_HAMR_Data')
from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#matplotlib.rcParams['text.usetex'] = True
from mpi4py import MPI
import matplotlib.patches as patches
import cmasher as cmr # colormaps I like
import matplotlib.style as style
style.use('dark_background')
matplotlib.rcParams['figure.dpi'] = 300
import tracemalloc
import compute_variables as cv
import figs

##############################################################################
figsize = [24.0, 22.0]
FS      = 30
plt.rcParams['xtick.labelsize'] = FS
plt.rcParams['ytick.labelsize'] = FS

lopt = dict(ha='left', va='center', color='white',
           bbox=dict(facecolor='black', alpha=0.3, edgecolor='none'))

ropt = dict(ha='right', va='center', color='white',
           bbox=dict(facecolor='black', alpha=0.3, edgecolor='none'))

MBH  = 10.0

def main(**kwargs):
    sim      = kwargs.get('sim')
    main_dir = '/lustre/orion/ast178/proj-shared/cbambic/'
    data_dir = main_dir + 'DATA/' + sim
    prof_dir = main_dir + 'Averages/' + sim + '/'
    movie_dir= main_dir + 'Movies/' + sim + '/'
    title    = kwargs.get('title')
    png_dir  = movie_dir + 'PNG/' + title + '/'
 
    start_file = int(kwargs.get('start_file'))
    end_file   = int(kwargs.get('end_file'))
    
    box_lim  = kwargs.get('box_lim')
    lowres1  = kwargs.get('lowres1')
    lowres2  = kwargs.get('lowres2')
    lowres3  = kwargs.get('lowres3')
    inset    = kwargs.get('inset')
    restart  = kwargs.get('restart')

    file_list = range(start_file, end_file)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    nout             = len(file_list)
    nfiles_nproc     = int(np.floor((nout + 1)/ size))
    nfiles_remaining = (nout + 1) % size
    files_per_proc   = ((size - nfiles_remaining) * [nfiles_nproc] +
        nfiles_remaining * [nfiles_nproc + 1])
    file_list_local  = ((sum(files_per_proc[:rank]) +
        np.arange(0, files_per_proc[rank])) + start_file).tolist()

    if 2061 in file_list_local:
        file_list_local.remove(2061)
    if 2062 in file_list_local:
        file_list_local.remove(2062)

    for file_num in file_list_local:
        rho_xy, rho_xz = {}, {}
        epsilon = 1.e-3 if (file_num != 1989) else 2.e-3
        tracemalloc.start()

        M = cv.Theta_Slice(data_dir, file_num, MBH = MBH, 
                lowres1 = lowres1, lowres2 = lowres2, lowres3 = lowres3)
    
        X_xy, Y_xy  = M.calc_XY(M.ith)
        rho_xy['r'] = M.calc_rho(M.ith)
        params      = M.H.params
        time_string = get_nice_time(params['t'])

        del M

        if (rank == 0):
            print(tracemalloc.get_traced_memory(), 'M', flush=True)

        R = cv.Phi_Slice(data_dir, file_num, MBH = MBH, 
                lowres1 = lowres1, lowres2 = lowres2, lowres3 = lowres3,
                phi_max = np.pi * epsilon)
        
        X_xz, Z_xz  = R.calc_RZ(0)
        rho_xz['r'] = R.calc_rho(0)

        del R
        
        if (rank == 0):
            print(tracemalloc.get_traced_memory(), 'R', flush=True)

        L = cv.Phi_Slice(data_dir, file_num, MBH = MBH, 
                lowres1 = lowres1, lowres2 = lowres2, lowres3 = lowres3,
                phi_min = np.pi * (1. - 0.5 * epsilon),
                phi_max = np.pi * (1. + 0.5 * epsilon))
        
        rho_xz['l'] = L.calc_rho(L.iphi)

        del L
        
        if (rank == 0):
            print(tracemalloc.get_traced_memory(), 'L', flush=True)

        Cart = figs.Cartesian(params)
        fig  = plt.figure(figsize = figsize)
        gs   = gridspec.GridSpec(2, 2, figure = fig)
        oaxes, iaxes, axins = [], [], []

        for i in range(2):
            oaxes.append(fig.add_subplot(gs[0, i]))
            iaxes.append(fig.add_subplot(gs[1, i]))
        axes = oaxes + iaxes

        if (inset != None):
            axins.append(axes[1].inset_axes([0.6, 0.08, 0.2, 0.2]))

        Cart.plot_xy(fig, oaxes[0], X_xy, Y_xy, rho_xy, 
            cmr.sepia, FS, lim = [1.e-11, 0.1], log=True)

        Cart.plot_xy(fig, iaxes[0], X_xy, Y_xy, rho_xy, 
            cmr.sepia, FS, lim = [1.e-11, 0.1], log=True)

        Cart.plot_xz(fig, oaxes[1], X_xz, Z_xz, rho_xz, 
            cmr.sepia, FS, lim = [1.e-11, 0.1], log=True)

        Cart.plot_xz(fig, iaxes[1], X_xz, Z_xz, rho_xz, 
            cmr.sepia, FS, lim = [1.e-11, 0.1], log=True)

        xy_axes = [oaxes[0], iaxes[0]]
        xz_axes = [oaxes[1], iaxes[1]]

        axes[0].text(0.05, 0.06, r'$t = $' + time_string + r' $R_g/c$',
            fontsize=FS+8, transform=axes[0].transAxes, **lopt)

        for ax in xy_axes:
            ax.set_xlabel(r'$x \: (R_g)$', fontsize=FS)
            ax.set_ylabel(r'$y \: (R_g)$', fontsize=FS)
        
        for ax in xz_axes:
            ax.set_xlabel(r'$x \: (R_g)$', fontsize=FS)
            ax.set_ylabel(r'$z \: (R_g)$', fontsize=FS)

        for ax in oaxes:
            ax.set_xlim([-300, 300])
            ax.set_ylim([-300, 300])
        
        for ax in iaxes:
            ax.set_xlim([-22, 22])
            ax.set_ylim([-22, 22])

        for ax in axes:
            ax.set_aspect('equal')
            
        if (inset != None):
            for ax in axins:
                height = inset[1] - inset[0]
                ax.set_xlim([inset[0], inset[1]])
                ax.set_ylim([-height/2., height/2.])
                ax.set_aspect('equal')
                    
        if (rank == 0):
            print("Saving figure ...", flush=True)
            if not os.path.exists(png_dir):
                os.makedirs(png_dir)
        
        print("Saving file %d on rank %d" %(file_num, rank), flush=True)
        fig.tight_layout()
        plt.savefig(png_dir + '%s_%06d.png' %(title, file_num), format='png')
        plt.close()
        
        del X_xy, Y_xy, X_xz, Z_xz, rho_xy, rho_xz
        
        if (rank == 0):
            print(tracemalloc.get_traced_memory(), 'After plot', flush=True)

def get_nice_time(time):
    raw_string = '%.5e' %time
    coeff = raw_string.split('e+')[0]
    expon = raw_string.split('e+')[1]
    return r'$%.5f \times 10^%d$' %(float(coeff), int(expon))

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Simple movie of density evolution.""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('start_file', type=int, 
                        help = """dump number of first file in movie list""")
    parser.add_argument('end_file', type=int, 
                        help = """dump number of last file in movie list""")
    parser.add_argument('--box_lim', type=float, default=10.,
                        required=False,
                        help = """box limit""")
    parser.add_argument('--lowres1', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in x1""")
    parser.add_argument('--lowres2', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in x2""")
    parser.add_argument('--lowres3', type=int, default=1,
                        required=False,
                        help = """downsampling parameter in x3""")
    parser.add_argument('--title', type=str, default='Density_Movie',
                        required=False,
                        help = """title of PNGs, default: Density_Movie""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if movie
                        should be made from reduced data.""")
    args = parser.parse_args()
    main(**vars(args))

