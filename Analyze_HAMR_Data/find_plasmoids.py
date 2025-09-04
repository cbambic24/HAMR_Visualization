import sys
import os
import argparse
import numpy as np
sys.path.insert(0, 
'/home/matthew/Desktop/cbambic/HAMR_Visualization/Read_HAMR_Data')
from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.rcParams['text.usetex'] = True
from mpi4py import MPI
import matplotlib.patches as patches
import cmasher as cmr # colormaps I like
import matplotlib.style as style
style.use('dark_background')
matplotlib.rcParams['figure.dpi'] = 300
import analysis as ana
import radiation as rad
import figs
import hamr_read as hr

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
    main_dir = '/home/matthew/Desktop/cbambic/'
    data_dir = main_dir + 'DATA/' + sim
    prof_dir = main_dir + 'Averages/' + sim + '/'
    movie_dir= main_dir + 'Movies/' + sim + '/'
    title    = kwargs.get('title')
    png_dir  = movie_dir + 'PNG/' + title + '/' 

    fnum     = int(kwargs.get('fnum'))
    nslices  = int(kwargs.get('nslices'))
    box_lim  = kwargs.get('box_lim')
    lowres1  = kwargs.get('lowres1')
    lowres2  = kwargs.get('lowres2')
    lowres3  = kwargs.get('lowres3')
    inset    = kwargs.get('inset')
    restart  = kwargs.get('restart')

    # First, determine the limits of the slices.
    H = hr.Read(main_dir + 'DATA/' + sim,
        cgs_data = True, cgs_grid = True, MBH = MBH)
    H.read_params(fnum)

    nx = int(hr.params['nb1'] * hr.params['bs1'] * H.upscalex1)
    nz = int(hr.params['nb3'] * hr.params['bs3'] * H.upscalex3)
    phi_1D = np.zeros(nz)
    for k in range(0, nz):
        phi_1D[k] = (hr.params['startx3'] + (k + 0.5) * hr.params['dx3'])

    dphi = phi_1D[1] - phi_1D[0]

    if (len(phi_1D) % nslices != 0):
        raise ValueError(
                "nslices must evenly divide len(phi_1D) = %d" %len(phi_1D))
    iphi_stp = int(len(phi_1D)/nslices)

    # Logic to allow restarts if figure production is interrupted.
    if not restart:
        iphi_start = 0
        restart_info = {}
    else:
        restart_info = read_restart(png_dir + 'restart_info.npy')
        if (restart_info['saved']):
            iphi_start = restart_info['k_max']
        else:
            iphi_start = restart_info['k_max'] - iphi_stp + 1

    TS = Theta_Slice(data_dir, fnum, MBH = MBH, 
        lowres1 = lowres1, lowres2 = lowres2, lowres3 = lowres3,
        r_max   = 1.5 * box_lim)

    time_string = get_nice_time(TS.H.params['t'])

    Msun = hr.CONSTANTS['Msun']
    G    = hr.CONSTANTS['G']
    c    = hr.CONSTANTS['c']
    Rg   = G * MBH * Msun / (c * c)
    
    ith  = TS.ith
    Ti_mid, vR_vK_mid = {}, {}
    
    xmid_2D, ymid_2D = TS.calc_XY(ith)
    rho_xy, Te_xy, Ti_mid['r']            = TS.calc_thermal_variables(ith)
    Trphi_mid, Tzphi_mid                  = TS.calc_Trphi_Tzphi(ith)
    vR_vK_mid['r']                        = TS.calc_vR_vKep(ith)
    sigma_mid, BetaI_mid                  = TS.calc_sigma_BetaI(ith)
    Xmid_uni, Ymid_uni                    = TS.calc_XY_uni(box_lim, 40)
    uux_mid, uuy_mid, uumid_in_out        = TS.calc_uu_cart(ith)
    Bx_mid, By_mid, Bmid_in_out           = TS.calc_B_cart(ith)
    uuradx_mid, uurady_mid, radmid_in_out = TS.calc_uu_rad_cart(ith)

    # Make figure
    first_call = True
    Cart = figs.Cartesian(hr.params)
    fig  = plt.figure(figsize = figsize)
    gs   = gridspec.GridSpec(2, 2, figure = fig)
    axes, caxes, axins = [], [], []
    for j in range(2):
        axes.append(fig.add_subplot(gs[0, j]))
        caxes.append(fig.add_subplot(gs[1, j]))
    if (inset != None):
        axins.append(axes[1].inset_axes([0.6, 0.08, 0.2, 0.2]))
    
    contours = {
        'quantities' : [sigma_mid],
        'levels'     : [3.0],
        'linewidths' : [0.5],
        'colors'     : ['white']
    }
    
    Cart.plot_xy(fig, caxes[0], xmid_2D, ymid_2D, vR_vK_mid,
                 cmr.iceburn, FS, lim = [-1, 1], 
                 sym=True,
                 contours=contours)
    
    color_streamlines = {
        'values' : Bmid_in_out,
        'cmap'   : cmr.redshift
    }
    
    Cart.plot_xy_vec(fig, caxes[0], xmid_2D, ymid_2D, 
                     Xmid_uni, Ymid_uni, Bx_mid, By_mid, 
                     dens = 2.0, linewidth = 0.3, 
                     color_streamlines = color_streamlines, 
                     broken_streamlines = False,
                     arrowsize = 0.0)
    
    Cart.plot_xy(fig, caxes[1], xmid_2D, ymid_2D, Ti_mid,
                 cmr.iceburn, FS, lim = [1.e4, 1.e12], log=True,
                 contours=contours)
    
    color_streamlines = {
        'values' : Bmid_in_out,
        'cmap'   : cmr.redshift
    }
    
    Cart.plot_xy_vec(fig, caxes[1], xmid_2D, ymid_2D, 
                     Xmid_uni, Ymid_uni, Bx_mid, By_mid, 
                     dens = 2.0, linewidth = 0.3, 
                     color_streamlines = color_streamlines, 
                     broken_streamlines = False,
                     arrowsize = 0.0)
        
    center_labels= [r'$v^R / (R \Omega_{\rm K})$',  r'$T_i \: (\rm{K})$']
        
    for (ax, center_label) in zip(caxes, center_labels):
        ax.set_xlabel(r'$x \: (R_g)$', fontsize=FS)
        ax.set_ylabel(r'$y \: (R_g)$', fontsize=FS)
        ax.set_xlim([-box_lim, box_lim])
        ax.set_ylim([-box_lim, box_lim])
        ax.set_aspect('equal')
        
        ax.text(0.05, 0.90, r'%s' %center_label, fontsize=FS+12, 
                transform=ax.transAxes, **lopt)
    
    r_in = TS.H.params['Rin']
    caxes[1].add_patch(patches.Circle((0,0), r_in, color='k', zorder=3))

    # Outer loop over whole data slice that fits on single node.
    # Inner loop over slivers of data slice assigned to each core.
    for slice_num, i in enumerate(range(iphi_start, len(phi_1D), iphi_stp)):
        restart_info['saved'] = False
        phi_min = phi_1D[i]
        phi_max = phi_1D[i + iphi_stp - 1]
    
        # Begin parallel section
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        delta_phi   = (phi_max - phi_min) / size
        phi_min_loc = phi_min + delta_phi * rank
        phi_max_loc = phi_min + delta_phi * (rank + 1)

        if (rank == 0):
            phi_min_loc -= dphi

        PS = Phi_Slice(data_dir, fnum, MBH = MBH, 
            lowres1 = lowres1, lowres2 = lowres2, lowres3 = lowres3,
            r_max   = 1.5 * box_lim,
            phi_min = phi_min_loc,
            phi_max = phi_max_loc)
        
        radius = PS.H.grid['r'][0, :, 0, 0] / Rg
        k_min, k_max = PS.H.block_info['k_min'], PS.H.block_info['k_max']
 
        iphi_max  = PS.H.data['rho'].shape[-1]
        for iphi in range(iphi_max):
            sigma, rho, Te, Ti = {}, {}, {}, {}

            R2D, z2D = PS.calc_RZ(iphi)
            rho['r'], Te['l'], Ti['r'] = PS.calc_thermal_variables(iphi)
            sigma['l'], BetaI          = PS.calc_sigma_BetaI(iphi) 
            b2 = sigma['l'] * 4. * np.pi * rho['r'] * c * c
            
            opacities = dict.fromkeys(['electron_scattering'])
            tau_es  = PS.calc_tau(opacities, iphi, 
                integral = 'theta', sigma = sigma['l'])
            
            R_uni, Z_uni               = PS.calc_RZ_uni(box_lim, 40)
            uuR, uuz, uu_in_out        = PS.calc_uu_cyl(iphi)
            BR, Bz, B_in_out           = PS.calc_B_cyl(iphi)
            uuradR, uuradz, rad_in_out = PS.calc_uu_rad_cyl(iphi)

            out_num = iphi_max * slice_num + iphi_max * rank + iphi

            phi0 = PS.H.grid['phi'][0, 0, 0, iphi]
            r_in = TS.H.params['Rin']
            r_out= TS.H.params['Rout']
            if (phi0 < np.pi/2. or phi0 > 3. * np.pi/2.):
                xpos = np.linspace(r_in, r_out, 100)
                ypos = np.tan(phi0) * xpos
                line, = caxes[0].plot(xpos, ypos, 
                    color='white', lw=0.7, linestyle='--', dashes=(10,20))
            else:
                xneg = np.linspace(r_in, r_out, 100)
                yneg = np.tan(phi0) * xneg
                line, = caxes[0].plot(-xneg, yneg, 
                    color='white', lw=0.5, linestyle='--', dashes=(10,20))

            caxes0_patch = caxes[0].add_patch(
                patches.Circle((0,0), r_in, color='k', zorder=3))

            phi_text = axes[0].text(0.95, 0.06, r'$\varphi = %.2f$' %phi0, 
                fontsize=FS+8, transform=axes[0].transAxes, **ropt)

            contours = {
                'quantities' : [sigma['l']],
                'levels'     : [3.0],
                'linewidths' : [1.0],
                'colors'     : ['white']
            }

            Cart.plot_xz_mirror(fig, axes[0], R2D, z2D, sigma, rho,
                                [cmr.iceburn, cmr.sepia], FS, 
                                lim_left  = [1.e-6, 1.e6],
                                lim_right = [1.e-11, 1.e-1],
                                log_left=True, log_right=True,
                                contours_left = contours,
                                contours_right = contours)
            
            color_streamlines = {
                'values' : uu_in_out,
                'cmap'   : cmr.redshift
            }
            
            Cart.plot_xy_vec(fig, axes[0], R2D, z2D, 
                             R_uni, Z_uni, uuR, uuz, 
                             dens = 2.0, linewidth = 0.3, 
                             color_streamlines = color_streamlines,
                             broken_streamlines = False,
                             arrowsize = 0.0)
            
            Cart.plot_xz_mirror(fig, axes[1], R2D, z2D, Te, Ti, 
                                [cmr.ember, cmr.iceburn], FS, 
                                lim_left  = [1.e8, 1.e10],
                                lim_right = [1.e4, 1.e12],
                                log_left=True, log_right=True,
                                contours_left = contours,
                                contours_right = contours)

            contours['linewidths'] = [1.0]

            color_streamlines = {
                'values' : B_in_out,
                'cmap'   : cmr.redshift
            }

            Cart.plot_xy_vec(fig, axes[1], R2D, z2D, 
                             R_uni, Z_uni, BR, Bz, 
                             dens = 2.0, linewidth = 0.3, 
                             color_streamlines = color_streamlines, 
                             broken_streamlines = False,
                             arrowsize = 0.0)
            
            if (inset != None):
                contours['linewidths'] = [0.3]
            
                color_streamlines = {
                    'values' : B_in_out,
                    'cmap'   : cmr.redshift
                }

                R_inset, z_inset = {}, {}
                R_inset['r'], z_inset['r'] = R2D['r'], z2D['r']
                Cart.plot_xy(fig, axins[0], R_inset, z_inset, Ti,
                             cmr.iceburn, FS, lim=[1.e4, 1.e12],
                             log=True, contours = contours, 
                             nocbar=True)

                Cart.plot_xy_vec(fig, axins[0], R_inset, z_inset, 
                                 R_uni, Z_uni, BR, Bz, 
                                 dens = 4.0, linewidth = 0.3, 
                                 color_streamlines = color_streamlines, 
                                 broken_streamlines = False,
                                 arrowsize = 0.0)
                 
            left_labels = [r'$\sigma$', r'$T_e \: (\rm{K})$'
            ]

            right_labels = [r'$\rho \: (\rm{g/cm}^3)$', r'$T_i \: (\rm{K})$'
            ]

            axes[0].text(0.05, 0.06, r'$t = $' + time_string + r'$ R_g/c$',
                         fontsize=FS+8, transform=axes[0].transAxes, **lopt)

            for (ax, left_label, right_label) in zip(
                axes, left_labels, right_labels):
                ax.set_xlabel(r'$R \: (R_g)$', fontsize=FS)
                ax.set_ylabel(r'$z \: (R_g)$', fontsize=FS)
                ax.set_xlim([-box_lim, box_lim])
                ax.set_ylim([-box_lim, box_lim])
                ax.set_aspect('equal')

                ax.text(0.05, 0.90, r'%s' %left_label, fontsize=FS+12, 
                        transform=ax.transAxes, **lopt)

                ax.text(0.95, 0.90, r'%s' %right_label, fontsize=FS+12, 
                        transform=ax.transAxes, **ropt)

                r_in = PS.H.params['Rin']
                ax.add_patch(patches.Circle((0,0), r_in, color='k', zorder=3))
 
            if (inset != None):
                for ax in axins:
                    height = inset[1] - inset[0]
                    ax.set_xlim([inset[0], inset[1]])
                    ax.set_ylim([-height/2., height/2.])
                    ax.set_aspect('equal')
                    
            comm.Barrier()
            if (rank == 0):
                print("Saving figure ...", flush=True)
                if not os.path.exists(png_dir):
                    os.makedirs(png_dir)
            if (first_call):
                fig.tight_layout()
                plt.savefig(png_dir + '%s_%04d.png' %(title, out_num), 
                    format='png')
                first_call = False
                fig.tight_layout()
                plt.savefig(png_dir + '%s_%04d.png' %(title, out_num), 
                    format='png')
            else:
                fig.tight_layout()
                plt.savefig(png_dir + '%s_%04d.png' %(title, out_num), 
                    format='png')

            phi_text.remove()
            line.remove()
            caxes0_patch.remove()
            for ax in axes:
                ax.clear()

            if (rank == 0):
                restart_info['saved'] = True
                restart_info['k_max'] = k_min + (k_max - k_min) * size
                np.save(png_dir + 'restart_info', restart_info)

def get_nice_time(time):
    raw_string = '%.2e' %time
    coeff = raw_string.split('e+')[0]
    expon = raw_string.split('e+')[1]
    return r'$%.2f \times 10^%d$' %(float(coeff), int(expon))

def read_restart(restart_file):
    return np.load(restart_file, allow_pickle=True).ravel()[0]

class Phi_Slice():
    def __init__(self, directory, fnum, **kwargs):
        self.MBH = kwargs.get('MBH', 10.0)
        self.H = hr.Read(directory, cgs_data=True, cgs_grid=True, 
            coords1D=True, MBH=self.MBH, 
            lowres1=int(kwargs.get('lowres1', 1)), 
            lowres2=int(kwargs.get('lowres2', 1)), 
            lowres3=int(kwargs.get('lowres3', 1)),
            r_max   = kwargs.get('r_max', None),
            phi_min = kwargs.get('phi_min', 0.0),
            phi_max = kwargs.get('phi_max', 2. * np.pi))
        self.H.read_grid( fnum, direct=True)
        self.H.read_prims(fnum, skip_read_params=True)

        self.Msun = hr.CONSTANTS['Msun']
        self.G    = hr.CONSTANTS['G']
        self.c    = hr.CONSTANTS['c']
        self.Rg   = self.G * MBH * self.Msun / (self.c * self.c)

        self.r    = self.H.grid['r'] / self.Rg  
        self.th   = self.H.grid['theta']
        self.phi  = self.H.grid['phi']

        self.phi_1D = self.phi[0, 0, 0, :]
        #self.iphi = np.argmin(np.abs(phi_1D - phi0))
        #self.jphi = np.argmin(np.abs(phi_1D - (phi0 + np.pi)))
    
        # Initialize Accretion class
        self.Acc = ana.Accretion(self.H.grid, self.H.data, self.H.params, 
            hr.CONSTANTS, cgs=True, MBH=self.MBH, radiation=True)
        # Initialize Coordinates class
        self.Coords = ana.Coordinates(self.H.grid['r'], self.H.grid['theta'], 
            self.H.grid['phi'], self.H.params, hr.CONSTANTS)
        # Initialize Fluid class
        self.Fluid = ana.Fluid(self.H.params, hr.CONSTANTS, cgs=True)
        # Initialize Field class
        self.Field = ana.Field(self.H.params, hr.CONSTANTS, cgs=True)
        # Initialize Radiation class
        self.Rad = rad.Radiation(self.H.params, hr.CONSTANTS, cgs=True)
    
    def calc_XZ(self, iphi):
        r   = self.r
        th  = self.th
        phi = self.phi
        
        # Convert to Cartesian coordinates
        # Note reflection for left panel
        self.x =  (r*np.sin(th)*np.cos(phi))[0, :, :, iphi]
        self.y =  (r*np.sin(th)*np.sin(phi))[0, :, :, iphi]
        self.z =  (r*np.cos(th)*np.ones(phi.shape))[0, :, :, iphi]
        
        x, z = {}, {}
        x['r'] = self.x
        z['r'] = self.z
        x['l'] = -self.x
        z['l'] = self.z
        return x, z

    def calc_RZ(self, iphi):
        r   = self.r
        th  = self.th
        phi = self.phi
        
        # Convert to cylindrical coordinates
        # Note reflection for left panel
        self.x =  (r*np.sin(th)*np.cos(phi))[0, :, :, iphi]
        self.y =  (r*np.sin(th)*np.sin(phi))[0, :, :, iphi]
        self.z =  (r*np.cos(th)*np.ones(phi.shape))[0, :, :, iphi]
        
        R, z = {}, {}
        R['r'] = np.sqrt(np.square(self.x) + np.square(self.y))
        z['r'] = self.z
        R['l'] = -np.sqrt(np.square(self.x) + np.square(self.y))
        z['l'] = self.z
        return R, z

    def calc_thermal_variables(self, iphi):
        # Density, electron temperature, ion temperature
        rho = self.H.data['rho'][0, :, :, iphi]
        Te  = self.H.data['Te'][ 0, :, :, iphi]
        Ti  = self.H.data['Ti'][ 0, :, :, iphi]
        return rho, Te, Ti

    def calc_sigma_BetaI(self, iphi):
        # sigma, BetaI = pmag/pgas
        c = self.c # required constants
        rho   = self.H.data['rho'][       0, :, :, iphi]
        B     = self.H.data['B'][:,       0, :, :, iphi]
        uu    = self.H.data['uu'][:,      0, :, :, iphi]
        u_gas = self.H.data['u_gas'][     0, :, :, iphi]
        Te    = self.H.data['Te'][        0, :, :, iphi]
        Ti    = self.H.data['Ti'][        0, :, :, iphi]
        gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]
    
        # Calculate sigma
        b     = self.Field.calc_b(B, uu, gcov)
        b2    = self.Field.calc_b2(b, B, gcov)
        sigma = (b2 / (4. * np.pi)) / (rho * c * c)
        # Calculate gas pressure and Beta^-1
        pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        BetaI = (b2 / (8. * np.pi)) / pgas
        return sigma, BetaI

    def calc_stress_energy(self, iphi):
        # MHD, electromagnetic, radiation, and fluid energy outflow rates 
        c, Rg = self.c, self.Rg # required constants
        rho   = self.H.data['rho'][       0, :, :, iphi]
        uu    = self.H.data['uu'][:,      0, :, :, iphi]
        u_gas = self.H.data['u_gas'][     0, :, :, iphi]
        B     = self.H.data['B'][:,       0, :, :, iphi]
        Te    = self.H.data['Te'][        0, :, :, iphi]
        Ti    = self.H.data['Ti'][        0, :, :, iphi]
        u_rad = self.H.data['u_rad'][     0, :, :, iphi]
        uu_rad= self.H.data['uu_rad'][:,  0, :, :, iphi]
        gdet  = self.H.grid['gdet'][      0, :, :, 0]
        gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]
        gcon  = self.H.grid['gcon'][:, :, 0, :, :, 0]

        pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)

        # Stress-energy tensor(s)
        T_Hydro = self.Fluid.calc_THD(rho, uu, u_gas, pgas, gcov, gcon, 
            upper_lower=True, index = [1, 0])
        T_EM    = self.Field.calc_TEM(uu, B, gcov, gcon, 
            upper_lower=True, index = [1, 0])
        T_Rad   = self.Rad.calc_TRAD(u_rad, uu_rad, gcov, gcon, 
            upper_lower=True, index = [1, 0])

        T_MHD  = T_Hydro + T_EM + T_Rad

        Area = 4 * np.pi * np.square(Rg * self.r)[0, :, :, iphi]
        Edot_MHD  = -Area * c * T_MHD 
        Edot_EM   = -Area * c * T_EM 
        Edot_Rad  = -Area * c * T_Rad
        return Edot_MHD, Edot_EM, Edot_Rad

    def calc_XZ_uni(self, box_lim, npts):
        # Make uniform grid for streamline plots
        x1D = np.linspace(-box_lim, box_lim, npts)
        z1D = np.linspace(-box_lim, box_lim, npts)
        X_uni, Z_uni = np.meshgrid(x1D, z1D)
        return X_uni, Z_uni
    
    def calc_RZ_uni(self, box_lim, npts):
        # Make uniform grid for streamline plots
        R1D = np.linspace(0., box_lim, npts)
        z1D = np.linspace(-box_lim, box_lim, npts)
        R_uni, Z_uni = np.meshgrid(R1D, z1D)
        return R_uni, Z_uni

    def calc_B_cart(self, iphi):
        Bx, Bz, in_out = {}, {}, {}
        B     = self.H.data['B'][:,       0, :, :, iphi]
        uu    = self.H.data['uu'][:,      0, :, :, iphi]
        gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]
    
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]

        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate B in Cartesian coordinates
        B_cart = Coords.int_to_cart_vec(B) / self.Field.B0
        in_out['r'] = np.sign(B[1])
        in_out['l'] = np.sign(B[1])

        Bx['r'] = B_cart[1]
        Bz['r'] = B_cart[3]
        Bx['l'] = -B_cart[1]
        Bz['l'] = B_cart[3]
        return Bx, Bz, in_out

    def calc_uu_cart(self, iphi):
        uux, uuz, in_out = {}, {}, {}
        uu = self.H.data['uu'][:, 0, :, :, iphi]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate uu in Cartesian coordinates
        uu_cart = Coords.int_to_cart_vec(uu) / self.c
        in_out['r'] = np.sign(uu[1])
        in_out['l'] = np.sign(uu[1])
        
        uux['r'] = uu_cart[1]
        uuz['r'] = uu_cart[3]
        uux['l'] = -uu_cart[1]
        uuz['l'] = uu_cart[3]
        return uux, uuz, in_out
    
    def calc_uu_rad_cart(self, iphi):
        uu_radx, uu_radz, in_out = {}, {}, {}
        uu_rad = self.H.data['uu_rad'][:, 0, :, :, iphi]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        rho0 = self.H.params['density_scale']
        c    = self.c
        # Calculate uu in Cartesian coordinates
        uu_rad_cart = Coords.int_to_cart_vec(uu_rad) / (rho0 * c * c * c)
        
        in_out['r'] = np.sign(uu_rad[1])
        in_out['l'] = np.sign(uu_rad[1])
        
        uu_radx['r'] = uu_rad_cart[1]
        uu_radz['r'] = uu_rad_cart[3]
        uu_radx['l'] = -uu_rad_cart[1]
        uu_radz['l'] = uu_rad_cart[3]
        return uu_radx, uu_radz, in_out
    
    def calc_B_cyl(self, iphi):
        BR, Bz, in_out = {}, {}, {}
        B     = self.H.data['B'][:,       0, :, :, iphi]
        uu    = self.H.data['uu'][:,      0, :, :, iphi]
        gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]
    
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]

        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate B in Cartesian coordinates
        B_cyl = Coords.int_to_cyl_vec(B) / self.Field.B0
        in_out['r'] = np.sign(B[1])
        in_out['l'] = np.sign(B[1])

        BR['r'] = B_cyl[1]
        Bz['r'] = B_cyl[3]
        BR['l'] = -B_cyl[1]
        Bz['l'] = B_cyl[3]
        return BR, Bz, in_out

    def calc_uu_cyl(self, iphi):
        uuR, uuz, in_out = {}, {}, {}
        uu = self.H.data['uu'][:, 0, :, :, iphi]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate uu in Cartesian coordinates
        uu_cyl = Coords.int_to_cyl_vec(uu) / self.c
        in_out['r'] = np.sign(uu[1])
        in_out['l'] = np.sign(uu[1])
        
        uuR['r'] = uu_cyl[1]
        uuz['r'] = uu_cyl[3]
        uuR['l'] = -uu_cyl[1]
        uuz['l'] = uu_cyl[3]
        return uuR, uuz, in_out
    
    def calc_uu_rad_cyl(self, iphi):
        uu_radR, uu_radz, in_out = {}, {}, {}
        uu_rad = self.H.data['uu_rad'][:, 0, :, :, iphi]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        rho0 = self.H.params['density_scale']
        c    = self.c
        # Calculate uu in Cartesian coordinates
        uu_rad_cyl = Coords.int_to_cyl_vec(uu_rad) / (rho0 * c * c * c)
        
        in_out['r'] = np.sign(uu_rad[1])
        in_out['l'] = np.sign(uu_rad[1])
        
        uu_radR['r'] = uu_rad_cyl[1]
        uu_radz['r'] = uu_rad_cyl[3]
        uu_radR['l'] = -uu_rad_cyl[1]
        uu_radz['l'] = uu_rad_cyl[3]
        return uu_radR, uu_radz, in_out

    def calc_Trphi_Tzphi(self, iphi):
        B     = self.H.data['B'][:,       0, :, :, iphi]
        uu    = self.H.data['uu'][:,      0, :, :, iphi]
        gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]
    
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]

        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate b in cylindridcal coordinates
        b     = self.Field.calc_b(B, uu, gcov)
        b_cyl = Coords.int_to_cyl_vec(b) 
        Trphi = -(b[1] * b[2]) / (4. * np.pi)
        Tzphi = -(b[3] * b[2]) / (4. * np.pi)
        return Trphi, Tzphi

    def calc_qcool(self, iphi, **kwargs):
        # Thermal/radiation variables
        rho   = self.H.data['rho'][  0, :, :, iphi]
        Te    = self.H.data['Te'][   0, :, :, iphi]
        Ti    = self.H.data['Ti'][   0, :, :, iphi]
        u_rad = self.H.data['u_rad'][0, :, :, iphi]

        # b2 is required for synchrotron opacities
        if 'b2' not in kwargs:
            uu    = self.H.data['uu'][:,      0, :, :, iphi]
            B     = self.H.data['B'][:,       0, :, :, iphi]
            gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]
            b     = self.Field.calc_b(B, uu, gcov)
            b2    = self.Field.calc_b2(b, B, gcov)
        else:
            b2    = kwargs.get('b2')

        # Calculate q_IC, q_sync, q_ie
        opacities  = {'synchrotron': None}
        qIC   = self.Rad.calc_qIC(rho, Te, u_rad)
        qsync = self.Rad.calc_q_emi(rho, Te, u_rad, 
            opacities = opacities, b2 = b2, XRB=True)
        qie   = self.Fluid.calc_qie(rho, Te, Ti)

        # Clean q_IC
        Trad  = (u_rad / hr.CONSTANTS['aRAD'])**(1./4.)
        qIC   = self.clean_minima(qIC, (Te < Trad))

        # Calculate q_sync / q_ie
        qie_clean = np.copy(qie)
        qie_clean[qie == 0.] = 1.
        qsync_qie = qsync / qie_clean
        qsync_qie[(Te > Ti)] = np.nan
    
        # Calculate q_IC / q_ie
        qie_clean = np.copy(qie)
        qie_clean[qie == 0.] = 1.
        qIC_qie = qIC / qie_clean
        # Te > Ti: not two-temperature
        qIC_qie[(Te > Ti)] = np.nan
        # Te < Tr: no Compton cooling
        qIC_qie[(Te < Trad)] = np.nan

        # Clean q_ie
        qie = self.clean_minima(qie, (Te > Ti))
    
        return qIC, qsync, qie, qIC_qie, qsync_qie
        
    def calc_teq_tIC(self, iphi, **kwargs):
        # Thermal/radiation variables
        rho   = self.H.data['rho'][  0, :, :, iphi]
        Te    = self.H.data['Te'][   0, :, :, iphi]
        Ti    = self.H.data['Ti'][   0, :, :, iphi]
        u_rad = self.H.data['u_rad'][0, :, :, iphi]

        # Calculate t_eq / t_IC
        tIC   = self.Rad.calc_tIC(  rho, Te, u_rad, cooling=True)
        teq   = self.Fluid.calc_teq(rho, Te, Ti,    cooling=True)
        
        teq[tIC == 0.] = 0.0
        tIC[teq == 0.] = self.H.params['dt']
        tIC[tIC == 0.] = self.H.params['dt']
        teq_tIC        = teq / tIC
        teq_tIC[teq_tIC < 1.e-6] = 1.e-6
        return teq_tIC

    def clean_minima(self, q, condition):
        # set all q where condition is met to 0, then
        # set all zeros to minimum non-zero q
        q[condition] = 0.0
        min_q        = np.amin(q[q > 0.])
        q[q < min_q] = min_q
        return q

    def calc_tau(self, opacities, iphi, **kwargs):
        c, Rg = self.c, self.Rg
        # Thermal/radiation variables
        rho   = self.H.data['rho'][  0, :, :, iphi]
        Te    = self.H.data['Te'][   0, :, :, iphi]
        Ti    = self.H.data['Ti'][   0, :, :, iphi]
        u_rad = self.H.data['u_rad'][0, :, :, iphi]
        gcov  = self.H.grid['gcov'][:, :, 0, :, :, 0]

        if 'synchrotron' in opacities or 'sigma' not in kwargs:
            # b2 is required for synchrotron opacities
            if 'b2' not in kwargs:
                uu    = self.H.data['uu'][:,      0, :, :, iphi]
                B     = self.H.data['B'][:,       0, :, :, iphi]
                b     = self.Field.calc_b(B, uu, gcov)
                b2    = self.Field.calc_b2(b, B, gcov)
            else:
                b2    = kwargs.get('b2')
            sigma = b2 / (4. * np.pi * rho * c * c)
            Opac  = rad.Opacity(opacities, rho, Te, u_rad, 
                self.H.params, hr.CONSTANTS, b2 = b2, XRB = True)
        else:
            sigma = kwargs.get('sigma')
            Opac  = rad.Opacity(opacities, rho, Te, u_rad, 
                self.H.params, hr.CONSTANTS, XRB = True)
        
        if (len(Opac.opacities) == 1 and 'electron_scattering' in opacities):
            kappa = Opac.opacities['electron_scattering']
        else:
            kappa = Opac.opacities['absorption_total']

        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, :, iphi]
        th2D  = (ones * self.th)[ 0, :, :, iphi]
        phi2D = (ones * self.phi)[0, :, :, iphi]

        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        if kwargs.get('integral') == 'z':
            return Rg * Coords.cum_integrate_z(kappa * rho, gcov[2, 2],
                sigma_cut = 3.0, sigma = sigma, theta_cut = np.pi/6.)
        else:
            return Rg * Coords.cum_integrate_th(kappa * rho, gcov[2, 2],
                sigma_cut = 3.0, sigma = sigma, theta_cut = np.pi/6.)

class Theta_Slice():
    def __init__(self, directory, fnum, **kwargs):
        self.MBH = kwargs.get('MBH', 10.0)
        self.H = hr.Read(directory, cgs_data=True, cgs_grid=True, 
            coords1D=True, MBH=self.MBH, 
            lowres1=int(kwargs.get('lowres1', 1)), 
            lowres2=int(kwargs.get('lowres2', 1)), 
            lowres3=int(kwargs.get('lowres3', 1)),
            r_max     = kwargs.get('r_max', None),
            theta_min = 0.495 * np.pi, theta_max = 0.505 * np.pi)
        self.H.read_grid( fnum, direct=True)
        self.H.read_prims(fnum, skip_read_params=True)

        self.Msun = hr.CONSTANTS['Msun']
        self.G    = hr.CONSTANTS['G']
        self.c    = hr.CONSTANTS['c']
        self.Rg   = self.G * self.MBH * self.Msun / (self.c * self.c)

        self.r    = self.H.grid['r'] / self.Rg  
        self.th   = self.H.grid['theta']
        self.phi  = self.H.grid['phi']

        th0       = np.pi / 2.
        self.th_1D= self.th[0, 0, :, 0]
        self.ith  = np.argmin(np.abs(self.th_1D - th0))
    
        # Initialize Accretion class
        self.Acc = ana.Accretion(self.H.grid, self.H.data, self.H.params, 
            hr.CONSTANTS, cgs=True, MBH=self.MBH, radiation=True)
        # Initialize Coordinates class
        self.Coords = ana.Coordinates(self.H.grid['r'], self.H.grid['theta'], 
            self.H.grid['phi'], self.H.params, hr.CONSTANTS)
        # Initialize Fluid class
        self.Fluid = ana.Fluid(self.H.params, hr.CONSTANTS, cgs=True)
        # Initialize Field class
        self.Field = ana.Field(self.H.params, hr.CONSTANTS, cgs=True)
        # Initialize Radiation class
        self.Rad = rad.Radiation(self.H.params, hr.CONSTANTS, cgs=True)
    
    def calc_XY(self, ith):
        r   = self.r
        th  = self.th
        phi = self.phi
        
        # Convert to Cartesian coordinates
        # Note reflection for left panel
        self.x =  (r*np.sin(th)*np.cos(phi))[0, :, ith, :]
        self.y =  (r*np.sin(th)*np.sin(phi))[0, :, ith, :]
        self.z =  (r*np.cos(th)*np.ones(phi.shape))[0, :, ith, :]
        
        x, y = {}, {}
        x['r'] = self.x
        y['r'] = self.y
        return x, y

    def calc_thermal_variables(self, ith):
        # Density, electron temperature, ion temperature
        rho = self.H.data['rho'][0, :, ith, :]
        Te  = self.H.data['Te'][ 0, :, ith, :]
        Ti  = self.H.data['Ti'][ 0, :, ith, :]
        return rho, Te, Ti

    def calc_sigma_BetaI(self, ith):
        # sigma, BetaI = pmag/pgas
        c = self.c # required constants
        rho   = self.H.data['rho'][       0, :, ith, :]
        B     = self.H.data['B'][:,       0, :, ith, :]
        uu    = self.H.data['uu'][:,      0, :, ith, :]
        u_gas = self.H.data['u_gas'][     0, :, ith, :]
        Te    = self.H.data['Te'][        0, :, ith, :]
        Ti    = self.H.data['Ti'][        0, :, ith, :]
        gcov  = self.H.grid['gcov'][:, :, 0, :, ith, :]
    
        # Calculate sigma
        b     = self.Field.calc_b(B, uu, gcov)
        b2    = self.Field.calc_b2(b, B, gcov)
        sigma = (b2 / (4. * np.pi)) / (rho * c * c)
        # Calculate gas pressure and Beta^-1
        pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        BetaI = (b2 / (8. * np.pi)) / pgas
        return sigma, BetaI
    
    def calc_Trphi_Tzphi(self, ith):
        B     = self.H.data['B'][:,       0, :, ith, :]
        uu    = self.H.data['uu'][:,      0, :, ith, :]
        gcov  = self.H.grid['gcov'][:, :, 0, :, ith, :]
    
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, ith, :]
        th2D  = (ones * self.th)[ 0, :, ith, :]
        phi2D = (ones * self.phi)[0, :, ith, :]

        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate b in cylindridcal coordinates
        b     = self.Field.calc_b(B, uu, gcov)
        b_cyl = Coords.int_to_cyl_vec(b) 
        Trphi = -(b[1] * b[2]) / (4. * np.pi)
        Tzphi = -(b[3] * b[2]) / (4. * np.pi)
        return Trphi, Tzphi
    
    def calc_XY_uni(self, box_lim, npts):
        # Make uniform grid for streamline plots
        x1D = np.linspace(-box_lim, box_lim, npts)
        y1D = np.linspace(-box_lim, box_lim, npts)
        X_uni, Y_uni = np.meshgrid(x1D, y1D)
        return X_uni, Y_uni

    def calc_B_cart(self, ith):
        Bx, By, in_out = {}, {}, {}
        B     = self.H.data['B'][:,       0, :, ith, :]
        uu    = self.H.data['uu'][:,      0, :, ith, :]
        gcov  = self.H.grid['gcov'][:, :, 0, :, ith, :]
    
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, ith, :]
        th2D  = (ones * self.th)[ 0, :, ith, :]
        phi2D = (ones * self.phi)[0, :, ith, :]

        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate B in Cartesian coordinates
        B_cart = Coords.int_to_cart_vec(B) / self.Field.B0
        in_out['r'] = np.sign(B[3])

        Bx['r'] = B_cart[1]
        By['r'] = B_cart[2]
        return Bx, By, in_out

    def calc_uu_cart(self, ith):
        uux, uuy, in_out = {}, {}, {}
        uu = self.H.data['uu'][:, 0, :, ith, :]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, ith, :]
        th2D  = (ones * self.th)[ 0, :, ith, :]
        phi2D = (ones * self.phi)[0, :, ith, :]
 
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate uu in Cartesian coordinates
        uu_cart = Coords.int_to_cart_vec(uu) / self.c
        in_out['r'] = np.sign(uu[3])
        
        uux['r'] = uu_cart[1]
        uuy['r'] = uu_cart[2]
        return uux, uuy, in_out
    
    def calc_uu_rad_cart(self, ith):
        uu_radx, uu_rady, in_out = {}, {}, {}
        uu_rad = self.H.data['uu_rad'][:, 0, :, ith, :]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, ith, :]
        th2D  = (ones * self.th)[ 0, :, ith, :]
        phi2D = (ones * self.phi)[0, :, ith, :]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        rho0 = self.H.params['density_scale']
        c    = self.c
        # Calculate uu in Cartesian coordinates
        uu_rad_cart = Coords.int_to_cart_vec(uu_rad) / (rho0 * c * c * c)
        in_out['r'] = np.sign(uu_rad[3])
        
        uu_radx['r'] = uu_rad_cart[1]
        uu_rady['r'] = uu_rad_cart[2]
        return uu_radx, uu_rady, in_out

    def calc_BetaIz(self, ith):
        B     = self.H.data['B'][:,       0, :, ith, :]
        uu    = self.H.data['uu'][:,      0, :, ith, :]
        u_gas = self.H.data['u_gas'][     0, :, ith, :]
        Te    = self.H.data['Te'][        0, :, ith, :]
        Ti    = self.H.data['Ti'][        0, :, ith, :]
        gcov  = self.H.grid['gcov'][:, :, 0, :, ith, :]

        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, ith, :]
        th2D  = (ones * self.th)[ 0, :, ith, :]
        phi2D = (ones * self.phi)[0, :, ith, :]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate b_z^2 
        b     = self.Field.calc_b(B, uu, gcov) 
        bz2   = Coords.dot(b, b, gcov, 'cylindrical', component=3)
        pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        BetaIz= (bz2 / (8. * np.pi)) / pgas 
        return BetaIz

    def calc_vR_vKep(self, ith):
        uu    = self.H.data['uu'][:,      0, :, ith, :]
        
        ones  = np.ones(self.H.data['rho'].shape)
        r2D   = (ones * self.r)[  0, :, ith, :]
        th2D  = (ones * self.th)[ 0, :, ith, :]
        phi2D = (ones * self.phi)[0, :, ith, :]
        
        Coords = ana.Coordinates(r2D, th2D, phi2D, self.H.params, hr.CONSTANTS)

        # Calculate v in cylindridcal coordinates
        uu_cyl = Coords.int_to_cyl_vec(uu) 
        vR     = self.c * uu_cyl[1] / uu_cyl[0]
        
        vKep   = np.sqrt(self.G * self.MBH * self.Msun / (self.Rg * self.r))
        return vR / vKep[0, :, 0, :]

class Read_Profiles():
    def __init__(self, prof_dir, **kwargs):
        fnums    = list(range(2291, 2300))
        # quantities calclated via volume average
        keys   = [
            'rho',  'Ti',   'Te',
            'Pmag', 'Pthm', 'Prad',
            'Eion', 'Eelc', 'vR', 'vphi', 'vz'
        ]

        self.profiles = {}
        for fnum in fnums:
            profile_file = (prof_dir + 'Profiles_%d.npy' %int(fnum))
            self.profiles['%s' %str(fnum)] = np.load(
                profile_file, allow_pickle=True).ravel()[0]
            # dx1 = dr / r and r is dimensionless (r / Rg)
            self.dx1 = self.profiles['%s' %str(fnum)]['dx1']
            self.r   = self.profiles['%s' %str(fnum)]['r']
            for key in keys:
                self.profiles['%s' %str(fnum)]['%s' %key] = (
                    self.profiles['%s' %str(fnum)]['%s' %key] / (
                    self.profiles['%s' %str(fnum)]['Vol'] ))
        return None

    def get_pressure(self):
        return self.time_avg('Pthm', self.profiles)

    def time_avg(self, quant, profiles):
        keys  = profiles.keys()
        q_avg = np.zeros(len(profiles[next(iter(profiles))]['r']))
        for key in keys:
            q_avg += profiles['%s' %key]['%s' %quant] / len(keys)
        return q_avg

    # Interpolate q(r_old) to q_new(r_new)
    def interp(self, q, r_old, r_new):
        spl = CubicSpline(r_old, q)
        return spl(r_new)

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
            description="""Plot 4 panels: 2 XZ mirror panels of
            sigma | rho and Te | Ti, and 2 XY panels of vR and Ti
            with velocity streamlines and magnetic field lines.""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('fnum', type=int, help = """dump number""")
    parser.add_argument('--nslices', type=int, required=False,
                        default=1,
                        help="""number of slices in phi 
                        used to load the data""")
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
    parser.add_argument('--inset', nargs='+', type=float,
                        required=False,
                        help = """starting and ending radius (in Rg)
                        for inset plot""")
    parser.add_argument('--title', type=str, default='Accretion_Movie',
                        required=False,
                        help = """title of PNGs, default: Accretion_Movie""")
    parser.add_argument('--restart', action='store_true', required=False,
                        help="""boolean set to determine if figure  
                        production should continue after interruption""")
    args = parser.parse_args()
    main(**vars(args))

