""""
Read in H-AMR primitive variable outputs and grid data.
Functions accelerated through Cython and C++
"""

import os
import numpy as np
import hamr_readc

CONSTANTS = {
    'aRAD'    :  7.5657e-15,     # Radiation constant (erg/cm^3/K^4)
    'mp'      :  1.673534e-24,   # Proton (ion) mass (g)
    'me'      :  9.1094e-28,     # Electron mass (g)
    'MMW'     :  1.69,           # Mean molecular weight (unitless)
    'Z_AB'    :  0.02,           # Solar abundance of "metals" (unitless)
    'Y_AB'    :  0.28,           # Solar abundance of He (unitless)
    'X_AB'    :  0.70,           # Solar abundance of H (unitless)
    'kB'      :  1.3806504e-16,  # Boltzmann constant (erg/K)
    'sigmaT'  :  6.652e-25,      # Thomson cross section (cm^2)
    'e'       :  4.80320427e-10, # Elementary charge (statcoulomb)
    'h'       :  6.6260755e-27,  # Planck's constant (erg s)
    'sigSB'   :  5.67051e-5,     # Stefan-Boltzmann constant (erg/cm^2/s/K^4)
    'c'       :  2.99792458e10,  # Speed of light (cm/s)
    'G'       :  6.67259e-8,     # Newton's gravitational constant (cm^3/g/s^2)
    'Msun'    :  1.998e33,       # Mass of the Sun (g) 
}

amr, physics, params = {}, {}, {}
amr_keys = [
    'ACTIVE', 'LEVEL', 'REFINED',
    'COORD1', 'COORD2', 'COORD3',
    'PARENT',
    'CHILD1', 'CHILD2', 'CHILD3', 'CHILD4',
    'CHILD5', 'CHILD6', 'CHILD7', 'CHILD8',
    'NBR1',   'NBR2',   'NBR3',   'NBR4',   'NBR5', 'NBR6',
    'NODE',   'POLE',   'GROUP',
    'CORN1',  'CORN2',  'CORN3', 'CORN4', 'CORN5', 'CORN6',
    'CORN7',  'CORN8',  'CORN9', 'CORN10','CORN11','CORN12',
    'TIMELEVEL',
    'LEVEL1', 'LEVEL2', 'LEVEL3',
    'NBR1_3', 'NBR1_4', 'NBR1_7', 'NBR1_8',
    'NBR2_1', 'NBR2_2', 'NBR2_3', 'NBR2_4',
    'NBR3_1', 'NBR3_2', 'NBR3_5', 'NBR3_6',
    'NBR4_5', 'NBR4_6', 'NBR3_7', 'NBR4_8',
    'NBR5_1', 'NBR5_3', 'NBR5_5', 'NBR5_7',
    'NBR6_2', 'NBR6_4', 'NBR6_6', 'NBR6_8',
    'NBR1P',  'NBR2P',  'NBR3P',  'NBR4P',  'NBR5P', 'NBR6P'
]

phys_keys = [
    'RAD_M1', 'RESISTIVE', 'TWO_T',
    'P_NUM', 'DO_YE', 'NEUTRINOS_M1',
    'DO_NUCLEAR', 'DOTRACER'
]

param_keys = [
    't', 'n_active', 'n_active_total', 'nstep',
    'Dtd', 'Dtl', 'Dtr', 'dump_cnt', 'rdump_cnt',
    'dt', 'failed',
    'bs1', 'bs2', 'bs3', 'nmax', 'nb1', 'nb2', 'nb3',
    'startx1', 'startx2', 'startx3',
    'dx1', 'dx2', 'dx3', 'tf', 'a', 'gam', 'cour',
    'Rin', 'Rout', 'R0', 'density_scale'
]

param_types = (
        [np.float64] + 3*[np.int32] + 3*[np.float64] + 2*[np.int32] +
        [np.float64] + 8*[np.int32] + 14*[np.float64]
)

for i, key in enumerate(amr_keys):
    if (i < 37):
        amr['%s' %key] = int(i)
    elif (i < 64) and (i > 36):
        amr['%s' %key] = int(73+i)
    else:
        amr['%s' %key] = int(97+i)

class Tools:
    def get_fromfile(infile, dtype):
        return np.fromfile(infile, dtype=dtype, count=1, sep='')[0]

    def zeroNd(shape, Narr, **kwargs): # returns Narr 2d arrays of zeros
        dtype = kwargs.get('dtype', np.float32)
        arrs = Narr*[None]
        for i in range(0, Narr):
            arrs[i] = np.zeros(shape, dtype=dtype)
        return arrs

class Read():
    def __init__(self, dir, **kwargs):
        self.dir         = dir
        self.by_block    = kwargs.get('by_block', 0)
        self.check_files = kwargs.get('check_files', 0)
        self.cgs_data    = kwargs.get('cgs_data', 0)
        self.cgs_grid    = kwargs.get('cgs_grid', 0)
        self.MBH         = kwargs.get('MBH', 0.0)
        self.lowres1     = kwargs.get('lowres1', 1)
        self.lowres2     = kwargs.get('lowres2', 1)
        self.lowres3     = kwargs.get('lowres3', 1)
        self.export_raytracing = kwargs.get('export_raytracing', 0)
        self.interpolate_var   = kwargs.get('interpolate_var', 0)
        self.axisym      = kwargs.get('axisym', 1)
        self.coords1D    = kwargs.get('coords1D', 0)

        if ('i_min' in kwargs or 'i_max' in kwargs or 
            'j_min' in kwargs or 'j_max' in kwargs or 
            'k_min' in kwargs or 'k_max' in kwargs):
            self.do_box_by_index = 1
            self.i_min  = kwargs.get('i_min', None)
            self.i_max  = kwargs.get('i_max', None)
            self.j_min  = kwargs.get('j_min', None)
            self.j_max  = kwargs.get('j_max', None)
            self.k_min  = kwargs.get('k_min', None)
            self.k_max  = kwargs.get('k_max', None)
        else:
            self.do_box_by_index = 0

        if ('r_min'     in kwargs or 'r_max'     in kwargs or 
            'theta_min' in kwargs or 'theta_max' in kwargs or
            'phi_min'   in kwargs or 'phi_max'   in kwargs or 
            self.do_box_by_index):
            self.do_box    = 1
            self.r_min     = kwargs.get('r_min', None)
            self.r_max     = kwargs.get('r_max', None)
            self.theta_min = kwargs.get('theta_min', 0.0)
            self.theta_max = kwargs.get('theta_max', np.pi)
            self.phi_min   = kwargs.get('phi_min', 0.0)
            self.phi_max   = kwargs.get('phi_max', 2.0*np.pi)
        else:
            self.do_box    = 0

        self.BYTES_PER_VAR = 4

        # this gets called once during initialization
        self.read_block()

        self.options = {
            'low_resx1' : self.lowres1,
            'low_resx2' : self.lowres2,
            'low_resx3' : self.lowres3
        }

    def read_block(self):
        # Read in AMR organizational data for every block
        # First, compute the number of variables stored in grid file
        if (os.path.isfile(self.dir + "/gdumps/grid")):
            fin = open(self.dir + "/gdumps/grid", "rb")
            size = os.path.getsize(self.dir + "/gdumps/grid")
            i = 1
            while(os.path.isfile(self.dir + "/gdumps/grid%d" % i)):
                if (i == 1): fin.close()
                fin = open(self.dir + "/gdumps/grid%d" % i, "rb")
                size += os.path.getsize(self.dir + "gdumps/grid%d" % i)
                fin.close()
                i += 1
            fin = open(self.dir + "/gdumps/grid", "rb")
            # nmax is the total number of possible blocks
            nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            # subtract off BYTES_PER_VAR since nmax is in the first slot
            # nv is the number of variables required to encode grid structure
            nv = (size - self.BYTES_PER_VAR) // nmax // self.BYTES_PER_VAR
        else:
            raise FileNotFoundError("Cannot find grid file in %s" 
                %(self.dir + "/gdumps/"))

        # Next, read in the information for the "block" array
        # which contains all information to encode grid structure
        # 'C' refers to C order (last index changing fastest)
        # self.block array stores info for grid structure
        self.block = np.zeros((nmax, 200), dtype=np.int32, order='C')
        gd = np.fromfile(fin, dtype=np.int32, count=nv * nmax, sep='')
        i = 1
        while (os.path.isfile(self.dir + "/gdumps/grid%d" % i)):
            if (i == 1): fin.close()
            fin = open(self.dir + "/gdumps/grid%d" % i, "rb")
            gd  = np.concatenate((gd,
                np.fromfile(fin,
                dtype=np.int32, count=nv * nmax, sep='')), axis=0)
            i += 1
            fin.close()
        # 'F' refers to Fortran order (first index changing fastest)
        gd = gd.reshape((nv, nmax), order='F').T
        self.block[:, 0:nv] = gd
        # indices from 0 to nv defined in "amr" dictionary.
        fin.close()

    def read_params(self, fnum):
        print("Loading parameters file")
        # Read in parameters from parameters file.
        if (os.path.isfile(self.dir + "/dumps%d/parameters" % fnum)):
            fin = open(self.dir + "/dumps%d/parameters" % fnum, "rb")
        else:
            raise FileNotFoundError("Cannot find parameters file in %s." 
                %(self.dir + "/dumps%d/parameters" % fnum))

        if (os.path.isfile(self.dir + "/dumps%d/grid" % fnum)):
            import warnings
            warnings.warn("Warning: this may be a legacy data set. " + 
                          "Please use Read_Legacy instead of Read.") 

        for par, par_type in zip(param_keys, param_types):
            params['%s' %par] = Tools.get_fromfile(fin, par_type)

        del params['Dtl']  
        del params['R0']

        # There are 13 integers before the physics integer flag
        for n in range(0, 13):
            trash = Tools.get_fromfile(fin, np.int32)
        del trash
        phys_int = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        # Using phys_int, determine which physics modules are active
        for key in phys_keys:
            physics['%s' % key] = 0
        phys_int = '{:08}'.format(phys_int)

        if (phys_int[-1] == '1'): physics['RAD_M1']       = 1
        if (phys_int[-2] == '1'): physics['RESISTIVE']    = 1
        if (phys_int[-3] == '1'): physics['TWO_T']        = 1
        if (phys_int[-4] == '1'): physics['P_NUM']        = 1
        if (phys_int[-5] == '1'): physics['DO_YE']        = 1
        if (phys_int[-6] == '1'): physics['NEUTRINOS_M1'] = 1
        if (phys_int[-7] == '1'): physics['DO_NUCLEAR']   = 1
        if (phys_int[-8] == '1'): physics['DOTRACER']     = 1

        trash = Tools.get_fromfile(fin, np.int32)
        del trash

        # Set grid spacing
        params['dx1'] = (np.log(params['Rout']) - np.log(params['Rin'])) / (
            (params['bs1'] * params['nb1']))
        fractheta     = -1.0 * params['startx2']
        params['dx2'] = 2.0 * fractheta / (params['bs2'] * params['nb2'])
        params['dx3'] = 2.0 * np.pi / (params['bs3'] * params['nb3'])

        nb         = params['n_active_total']
        # n_ord translates order of blocks in Z curve
        self.n_ord = np.zeros((params['nmax']), dtype=np.int32, order='C')
        node       = np.copy(self.n_ord)
        timelevel  = np.copy(self.n_ord)

        size = os.path.getsize(self.dir + "/dumps%d/parameters" % fnum)
        # This logic accounts for old and new file formats
        # 66 is the number of variables in output
        if (size >= 66 * self.BYTES_PER_VAR +
                3 * params['n_active_total'] * self.BYTES_PER_VAR):
            for n in range(params['n_active_total']):
                self.n_ord[n] = Tools.get_fromfile(fin, np.int32)
                timelevel[n]  = Tools.get_fromfile(fin, np.int32)
                node[n]       = Tools.get_fromfile(fin, np.int32)
        elif (size >= 66 * self.BYTES_PER_VAR +
              2 * params['n_active_total'] * self.BYTES_PER_VAR):
            for n in range(params['n_active_total']):
                self.n_ord[n] = Tools.get_fromfile(fin, np.int32)
                timelevel[n]  = Tools.get_fromfile(fin, np.int32)

        fin.close()

        if (self.do_box):
            if (self.r_min is None): self.r_min = params['Rin']
            if (self.r_max is None): self.r_max = params['Rout']
            self.r_min     = max(self.r_min, params['Rin'])
            self.r_max     = min(self.r_max, params['Rout'])
            self.theta_min = max(self.theta_min, 
                                 (np.pi/2.0) * (params['startx2'] + 1.0))
            self.phi_min   = max(self.phi_min, params['startx3'])

        self.check_raytracing_settings()

        amr['max_levelx1'] = np.max(self.block[self.n_ord, amr['LEVEL1']])
        amr['max_levelx2'] = np.max(self.block[self.n_ord, amr['LEVEL2']])
        amr['max_levelx3'] = np.max(self.block[self.n_ord, amr['LEVEL3']])

        self.amr_info = {
            'n_ord'   : self.n_ord,
            'levelx1' : self.block[:, amr['LEVEL1']],
            'levelx2' : self.block[:, amr['LEVEL2']],
            'levelx3' : self.block[:, amr['LEVEL3']],
            'coordx1' : self.block[:, amr['COORD1']],
            'coordx2' : self.block[:, amr['COORD2']],
            'coordx3' : self.block[:, amr['COORD3']]
        }

        self.upscalex1 = 2 ** amr['max_levelx1']
        self.upscalex2 = 2 ** amr['max_levelx2']
        self.upscalex3 = 2 ** amr['max_levelx3']

        # downsample differentials with lowres and 
        # upscale to highest AMR level
        params['dx1'] *= self.lowres1 * (1.0 / self.upscalex1)
        params['dx2'] *= self.lowres2 * (1.0 / self.upscalex2)
        params['dx3'] *= self.lowres3 * (1.0 / self.upscalex3)

        self.amr    = amr
        self.params = params

    def read_grid(self, fnum, **kwargs):
        skip_read_params = kwargs.get('skip_read_params', False)
        direct           = kwargs.get('direct', False)
        if not skip_read_params:
            self.read_params(fnum)

        # Read in grid from gdump files or directly compute grid variables
        self.check_lowres_settings()

        if (self.by_block):
            bs1_new = int(params['bs1'] / self.lowres1)
            bs2_new = int(params['bs2'] / self.lowres2)
            bs3_new = int(params['bs3'] / self.lowres3)

            nx  = bs1_new
            ny  = bs2_new
            nz  = bs3_new
            nb  = params['n_active_total']
            dim = nb

            i_min, i_max = 0, nx
            j_min, j_max = 0, ny
            k_min, k_max = 0, nz

        else:
            self.gridsizex1 = int(
                params['nb1']*params['bs1']*self.upscalex1 / self.lowres1)
            self.gridsizex2 = int(
                params['nb2']*params['bs2']*self.upscalex2 / self.lowres2)
            self.gridsizex3 = int(
                params['nb3']*params['bs3']*self.upscalex3 / self.lowres3)

            # Calculate inner and outer boundaries of selection box
            # after upscaling and downscaling; 
            # Assumes uniform grid x1=log(r) etc
            if (self.do_box or self.do_box_by_index):
                i_min, i_max = self.get_rbox_coords()
                j_min, j_max = self.get_tbox_coords()
                k_min, k_max = self.get_pbox_coords()

                self.gridsizex1 = i_max - i_min
                self.gridsizex2 = j_max - j_min
                self.gridsizex3 = k_max - k_min

                if (i_max < i_min or j_max < j_min or k_max < k_min):
                    raise ValueError("Bad box selection for file %d in read_grid." %fnum)
            else:
                i_min, i_max = 0, self.gridsizex1
                j_min, j_max = 0, self.gridsizex2
                k_min, k_max = 0, self.gridsizex3

            nx  = self.gridsizex1
            ny  = self.gridsizex2
            nz  = self.gridsizex3
            dim = 1
        
        if (self.check_files):
            self.check_griddump_exists()

        # Allocate memory
        if direct and not self.by_block and self.coords1D:
            x1, r     = Tools.zeroNd([nx], 2)
            x2, theta = Tools.zeroNd([ny], 2)
            x3, phi   = Tools.zeroNd([nz], 2)
        else:
            x1, x2, x3, r, theta, phi = Tools.zeroNd([dim, nx, ny, nz], 6)

        if (self.axisym):
            gcov, gcon, dxdxp = Tools.zeroNd([4, 4, dim, nx, ny, 1], 3)
            gdet = np.zeros([dim, nx, ny, 1], dtype=np.float32)
        else:
            gcov, gcon, dxdxp = Tools.zeroNd([4, 4, dim, nx, ny, nz], 3)
            gdet = np.zeros([dim, nx, ny, nz], dtype=np.float32)

        self.block_info = {
            'i_min' : i_min,
            'i_max' : i_max,
            'j_min' : j_min,
            'j_max' : j_max,
            'k_min' : k_min,
            'k_max' : k_max
        }
            
        self.Toolsc = hamr_readc.PyTools()

        if (direct):
            if not self.by_block:
                x1, x2, x3, r, theta, phi = self.coords_direct(x1, x2, x3, 
                    r, theta, phi, [dim, nx, ny, nz])
            elif self.by_block:
                x1, x2, x3, r, theta, phi = self.coords_directBK(x1, x2, x3,
                    r, theta, phi, [dim, nx, ny, nz])
            
            # Calculate gcov in internal coordinates and the transformation matrix
            # dxdxp to go from spherical Kerr-Schild to internal coordinates
            gcov, dxdxp = self.gcov_dxdxp_direct(dxdxp, r, theta, phi, nx, ny)

            # Invert gcov to find gcon in internal coordinates
            self.Toolsc.invert_4x4(gcov, gcon, dim, nx, ny, 1)

            # Calculate determinant of metric
            self.Toolsc.det_4x4(gcov, gdet, dim, nx, ny, 1)
            gdet = np.sqrt(-1.0*gdet)
        
        else:
            Readc = hamr_readc.PyRead(self.dir, self.axisym, self.by_block,
                self.export_raytracing, amr, self.amr_info, self.block_info, 
                self.options, params, physics)
            
            Readc.read_grid(x1, x2, x3, r, theta, phi, 
                gcov, gcon, gdet, dxdxp)

            if self.coords1D:
                x1, r     = x1[:, :, 0, 0], r[    :, :, 0, 0]
                x2, theta = x2[:, 0, :, 0], theta[:, 0, :, 0]
                x3, phi   = x3[:, 0, 0, :], phi[  :, 0, 0, :]
                    
        if not self.coords1D:
            self.grid = {
                'x1'    : x1,
                'x2'    : x2, 
                'x3'    : x3,
                'r'     : r,
                'theta' : theta,
                'phi'   : phi,
                'gcov'  : gcov,
                'gcon'  : gcon,
                'gdet'  : gdet,
                'dxdxp' : dxdxp
            }
        elif self.coords1D:
            if not self.by_block:
                self.grid = {
                    'x1'    : x1[None, :, None, None],
                    'x2'    : x2[None, None, :, None], 
                    'x3'    : x3[None, None, None, :],
                    'r'     : r[None, :, None, None],
                    'theta' : theta[None, None, :, None],
                    'phi'   : phi[None, None, None, :],
                    'gcov'  : gcov,
                    'gcon'  : gcon,
                    'gdet'  : gdet,
                    'dxdxp' : dxdxp
                }
            elif self.by_block:
                self.grid = {
                    'x1'    : x1[:, :, None, None],
                    'x2'    : x2[:, None, :, None], 
                    'x3'    : x3[:, None, None, :],
                    'r'     : r[:, :, None, None],
                    'theta' : theta[:, None, :, None],
                    'phi'   : phi[:, None, None, :],
                    'gcov'  : gcov,
                    'gcon'  : gcon,
                    'gdet'  : gdet,
                    'dxdxp' : dxdxp
                }

        if (self.cgs_grid):
            self.restore_units_grid()

    def read_prims(self, fnum, **kwargs):
        skip_read_params = kwargs.get('skip_read_params', False)
        if not skip_read_params:
            self.read_params(fnum)

        if (self.by_block):
            bs1_new = int(params['bs1'] / self.lowres1)
            bs2_new = int(params['bs2'] / self.lowres2)
            bs3_new = int(params['bs3'] / self.lowres3)

            nx  = bs1_new
            ny  = bs2_new
            nz  = bs3_new
            nb  = params['n_active_total']
            dim = nb
                
            i_min, i_max = 0, nx
            j_min, j_max = 0, ny
            k_min, k_max = 0, nz
        else:
            self.gridsizex1 = int(
                params['nb1']*params['bs1']*self.upscalex1 / self.lowres1)
            self.gridsizex2 = int(
                params['nb2']*params['bs2']*self.upscalex2 / self.lowres2)
            self.gridsizex3 = int(
                params['nb3']*params['bs3']*self.upscalex3 / self.lowres3)

            # Calculate inner and outer boundaries of selection box
            # after upscaling and downscaling; 
            # Assumes uniform grid x1=log(r) etc
            if (self.do_box or self.do_box_by_index):
                i_min, i_max = self.get_rbox_coords()
                j_min, j_max = self.get_tbox_coords()
                k_min, k_max = self.get_pbox_coords()

                self.gridsizex1 = i_max - i_min
                self.gridsizex2 = j_max - j_min
                self.gridsizex3 = k_max - k_min

                if (i_max < i_min or j_max < j_min or k_max < k_min):
                    raise ValueError("Bad box selection in read_prims.")
            else:
                i_min, i_max = 0, self.gridsizex1
                j_min, j_max = 0, self.gridsizex2
                k_min, k_max = 0, self.gridsizex3

            nx  = self.gridsizex1
            ny  = self.gridsizex2
            nz  = self.gridsizex3
            dim = 1

        # Allocate memory
        rho, u_gas = Tools.zeroNd([dim, nx, ny, nz], 2)
        uu         = np.zeros([4, dim, nx, ny, nz], dtype=np.float32)
        B          = np.zeros([4, dim, nx, ny, nz], dtype=np.float32)
        
        if (physics['RAD_M1']):
            u_rad  = np.zeros([dim, nx, ny, nz], dtype=np.float32)
            uu_rad = np.zeros([4, dim, nx, ny, nz], dtype=np.float32)
        else: 
            u_rad, uu_rad = rho, uu

        if (physics['RESISTIVE']):
            E = np.zeros([4, dim, nx, ny, nz], dtype=np.float32)
        else:
            E = B
        
        if (physics['TWO_T']):
            Te, Ti = Tools.zeroNd([dim, nx, ny, nz], 2)
        else:
            Te, Ti = rho, rho

        if (physics['P_NUM']):
            p_num = np.zeros([dim, nx, ny, nz], dtype=np.float32)
        else:
            p_num = rho

        if (physics['DO_YE']):
            Ye = np.zeros([dim, nx, ny, nz], dtype=np.float32)
        else:
            Ye = rho

        if (physics['DO_NUCLEAR']):
            X_alpha, X_atm = Tools.zeroNd([dim, nx, ny, nz], 2)
        else:
            X_alpha, X_atm = rho, rho
        
        if (physics['NEUTRINOS_M1']):
            u_nu1,  u_nu2,  u_nu3  = Tools.zeroNd([dim, nx, ny, nz], 3)
            uu_nu1, uu_nu2, uu_nu3 = Tools.zeroNd([4, dim, nx, ny, nz], 3)
            n_nu1,  n_nu2,  n_nu3  = Tools.zeroNd([dim, nx, ny, nz], 3)
        else:
            u_nu1,  u_nu2,  u_nu3  = rho, rho, rho
            uu_nu1, uu_nu2, uu_nu3 = uu, uu, uu
            n_nu1,  n_nu2,  n_nu3  = rho, rho, rho

        if (physics['DOTRACER']):
            tr1, tr2 = Tools.zeroNd([dim, nx, ny, nz], 2)
        else: 
            tr1, tr2 = rho, rho

        self.block_info = {
            'i_min' : i_min,
            'i_max' : i_max,
            'j_min' : j_min,
            'j_max' : j_max,
            'k_min' : k_min,
            'k_max' : k_max
        }

        Readc = hamr_readc.PyRead(self.dir, self.axisym, self.by_block,
            self.export_raytracing, amr, self.amr_info, self.block_info, 
            self.options, params, physics)
        
        Readc.read_prims(fnum, rho, u_gas, uu, B, 
            E, u_rad, uu_rad, Te, Ti, p_num, Ye, X_alpha, X_atm, 
            u_nu1, u_nu2, u_nu3, uu_nu1, uu_nu2, uu_nu3, 
            n_nu1, n_nu2, n_nu3, tr1, tr2)
        
        self.data = {
            'rho'   : rho,
            'u_gas' : u_gas,
            'uu'    : uu,
            'B'     : B
        }

        if (physics['RAD_M1']):
            self.data['u_rad']  = u_rad
            self.data['uu_rad'] = uu_rad

        if (physics['RESISTIVE']):
            self.data['E'] = E

        if (physics['TWO_T']):
            self.data['Te'] = Te
            self.data['Ti'] = Ti

        if (physics['P_NUM']):
            self.data['p_num'] = p_num

        if (physics['DO_YE']):
            self.data['Ye'] = Ye
        
        if (physics['DO_NUCLEAR']):
            self.data['X_alpha'] = X_alpha
            self.data['X_atm']   = X_atm

        if (physics['NEUTRINOS_M1']):
            self.data['u_nu1']  = u_nu1
            self.data['u_nu2']  = u_nu2
            self.data['u_nu3']  = u_nu3
            self.data['uu_nu1'] = uu_nu1
            self.data['uu_nu2'] = uu_nu2
            self.data['uu_nu3'] = uu_nu3
            self.data['n_nu1']  = n_nu1
            self.data['n_nu2']  = n_nu2
            self.data['n_nu3']  = n_nu3

        if (physics['DOTRACER']):
            self.data['tr1'] = tr1
            self.data['tr2'] = tr2

        if (self.cgs_data):
            self.restore_units_prims()

    def restore_units_prims(self):
        # Output primitives in CGS units
        rho0 = params['density_scale'] # g/cm^3
        c    = CONSTANTS['c']          # cm/s
        self.data['rho']   *= rho0         # g/cm^3
        self.data['u_gas'] *= rho0 * c * c # erg/cm^3
        self.data['uu']    *= c            # cm/s
        self.data['B']     *= np.sqrt(4 * np.pi) * np.sqrt(rho0) * c # Gauss

        if (physics['RAD_M1']):
            self.data['u_rad']  *= rho0 * c * c     # erg/cm^3
            self.data['uu_rad'] *= rho0 * c * c * c # erg/cm^2 s

        if (physics['RESISTIVE']):
            self.data['E'] *= np.sqrt(rho0) * c # statVolt/cm (same as Gauss)

        if (physics['TWO_T']):
            kB   = CONSTANTS['kB']
            mp   = CONSTANTS['mp']
            X_ab = CONSTANTS['X_AB']
            Y_ab = CONSTANTS['Y_AB']
            mu_i = 4.0 / (4.0 * X_ab + Y_ab)
            mu_e = 2.0 / (1.0 + X_ab)

            self.data['Te'] *= mu_e * mp * c * c / kB 
            self.data['Ti'] *= mu_i * mp * c * c / kB 

    def restore_units_grid(self):
        # Output grid variables in CGS units
        if (self.MBH == 0.0):
            raise TypeError("Must define black hole mass in Read class.")
        G    = CONSTANTS['G']
        Msun = CONSTANTS['Msun']
        c    = CONSTANTS['c']
        Rg = G*self.MBH*Msun/(c*c)
        self.grid['r'] *= Rg

    def get_gcov_KS(self, r, theta, phi, nx, ny):
        # Calculate covariant metric in spherical Kerr-Schild coordinates
        gcov = np.zeros((4, 4, 1, nx, ny, 1), dtype=np.float32)
        
        cth  = np.cos(theta)
        sth  = np.sin(theta)
        s2   = sth * sth
        rho2 = r * r + params['a'] * params['a'] * cth * cth
        
        gcov[0, 0, 0, :, :, 0] = (-1. + 2. * r / rho2)[0, :, :, 0]
        gcov[0, 1, 0, :, :, 0] = (2. * r / rho2)[0, :, :, 0]
        gcov[0, 3, 0, :, :, 0] = (-2. * params['a'] * r * s2 / rho2
                                  )[0, :, :, 0]
        gcov[1, 0, 0, :, :, 0] = gcov[0, 1, 0, :, :, 0]
        gcov[1, 1, 0, :, :, 0] =  (1. + 2. * r / rho2)[0, :, :, 0]
        gcov[1, 3, 0, :, :, 0] = (
            -params['a'] * s2 * (1. + 2. * r / rho2))[0, :, :, 0]
        gcov[2, 2, 0, :, :, 0] = rho2[0, :, :, 0]
        gcov[3, 0, 0, :, :, 0] = gcov[0, 3, 0, :, :, 0]
        gcov[3, 1, 0, :, :, 0] = gcov[1, 3, 0, :, :, 0]
        gcov[3, 3, 0, :, :, 0] = (s2 * (
            rho2 + params['a'] * params['a'] * s2 * (1. + 2. * r / rho2))
            )[0, :, :, 0]
        
        return gcov
    
    def get_gcon_KS(self, gcov, nx, ny):
        # Calculate contravariant metric in spherical Kerr-Schild coordinates
        gcon = np.zeros((4, 4, 1, nx, ny, 1), dtype=np.float32)
        self.Toolsc.invert_4x4(gcov, gcon, params)
        return gcon

    def gcov_dxdxp_direct(self, dxdxp, r, theta, phi, nx, ny):
        if not self.coords1D:
            gcov_kerr = self.get_gcov_KS(r, theta, phi, nx, ny)
        else:
            if not self.by_block:
                gcov_kerr = self.get_gcov_KS(r[None, :, None, None], 
                    theta[None, None, :, None], 
                    phi[None, None, None, :], nx, ny)
            else:
                gcov_kerr = self.get_gcov_KS(r[:, :, None, None], 
                    theta[:, None, :, None], 
                    phi[:, None, None, :], nx, ny)
        
        dxdxp = self.dxdxp_direct(dxdxp, r)
        # Set covariant metric in internal coordinates
        gcov = np.einsum('kl...,ki...,lj...->ij...', gcov_kerr, dxdxp, dxdxp)
        return gcov, dxdxp

    def dxdxp_direct(self, dxdxp, r):
        if not self.coords1D:
            if not self.by_block:
                dxdxp[0, 0] = 1.0
                dxdxp[1, 1, 0, :, :, 0] = r[0, :, :, 0]
                dxdxp[2, 2] = np.pi / 2.0
                dxdxp[3, 3] = 1.0
            else:
                dxdxp[0, 0] = 1.0
                dxdxp[1, 1, :, :, :, 0] = r[:, :, :, 0]
                dxdxp[2, 2] = np.pi / 2.0
                dxdxp[3, 3] = 1.0
        else:
            if not self.by_block:
                dxdxp[0, 0] = 1.0
                dxdxp[1, 1, 0, :, :, 0] = r[:, None]
                dxdxp[2, 2] = np.pi / 2.0
                dxdxp[3, 3] = 1.0
            else:
                dxdxp[0, 0] = 1.0
                dxdxp[1, 1, :, :, :, 0] = r[:, :, None]
                dxdxp[2, 2] = np.pi / 2.0
                dxdxp[3, 3] = 1.0
        return dxdxp

    def coords_direct(self, x1, x2, x3, r, theta, phi, indices):
        [dim, nx, ny, nz] = indices
        if not self.do_box:
            startx1 = params['startx1']
            startx2 = params['startx2']
            startx3 = params['startx3']
        else:
            startx1 = np.log(self.r_min)
            startx2 = ((2.0/np.pi) * self.theta_min) - 1.0
            startx3 = self.phi_min
        
        i_min, i_max = self.block_info['i_min'], self.block_info['i_max']
        j_min, j_max = self.block_info['j_min'], self.block_info['j_max']
        k_min, k_max = self.block_info['k_min'], self.block_info['k_max']

        for i, ii in enumerate(range(i_min, i_max)):
            if not self.coords1D:
                x1[:, i, :, :] = startx1 + (ii + 0.5) * params['dx1']
            else:
                x1[i]          = startx1 + (ii + 0.5) * params['dx1']
        for j, jj in enumerate(range(j_min, j_max)):
            if not self.coords1D:
                x2[:, :, j, :] = startx2 + (jj + 0.5) * params['dx2']
            else:
                x2[j]          = startx2 + (jj + 0.5) * params['dx2']
        for k, kk in enumerate(range(k_min, k_max)):
            if not self.coords1D:
                x3[:, :, :, k] = startx3 + (kk + 0.5) * params['dx3']
            else:
                x3[k]          = startx3 + (kk + 0.5) * params['dx3']
 
        r     = np.exp(x1)
        theta = (x2 + 1.0) * (np.pi/2.0)
        phi   = x3
        return x1, x2, x3, r, theta, phi

    def coords_directBK(self, x1, x2, x3, r, theta, phi, indices):
        Readc = hamr_readc.PyRead(self.dir, self.axisym, 
            self.by_block, self.export_raytracing, amr, self.amr_info, 
            self.block_info, self.options, params, physics)

        Readc.read_grid_coords(x1, x2, x3, r, theta, phi)
        if self.coords1D:
            x1_1D, r_1D     = x1[:, :, 0, 0], r[    :, :, 0, 0]
            x2_1D, theta_1D = x2[:, 0, :, 0], theta[:, 0, :, 0]
            x3_1D, phi_1D   = x3[:, 0, 0, :], phi[  :, 0, 0, :]

            x1, x2, x3      = x1_1D, x2_1D, x3_1D
            r, theta, phi   = r_1D, theta_1D, phi_1D
        return x1, x2, x3, r, theta, phi

    ### Helper functions for box selection ###
    def get_min_index(self, box_min, startx, dx):
        return max(np.int32((box_min - (startx + 0.5*dx))/dx + 1), 0)

    def get_max_index(self, box_max, startx, dx, max_val):
        return min(np.int32((box_max - (startx + 0.5*dx))/dx + 1), max_val)

    def get_rbox_coords(self):
        if (not self.do_box_by_index or 
            (self.do_box_by_index and self.i_min is None)):
            i_min = self.get_min_index(np.log(self.r_min),
                                   params['startx1'], params['dx1'])
        else:
            i_min = int(self.i_min)
        if (not self.do_box_by_index or
            (self.do_box_by_index and self.i_max is None)):
            i_max = self.get_max_index(np.log(self.r_max),
                                   params['startx1'], params['dx1'],
                                   self.gridsizex1)
        else:
            i_max = int(self.i_max)
        return i_min, i_max

    def get_tbox_coords(self):
        if (not self.do_box_by_index or 
            (self.do_box_by_index and self.j_min is None)):
            j_min = self.get_min_index((2.0 / np.pi) * self.theta_min - 1.0,
                                   params['startx2'], params['dx2'])
        else:
            j_min = int(self.j_min)
        if (not self.do_box_by_index or
            (self.do_box_by_index and self.j_max is None)):
            j_max = self.get_max_index((2.0 / np.pi) * self.theta_max - 1.0,
                                   params['startx2'], params['dx2'],
                                   self.gridsizex2)
        else:
            j_max = int(self.j_max)
        return j_min, j_max

    def get_pbox_coords(self):
        if (not self.do_box_by_index or 
            (self.do_box_by_index and self.k_min is None)):
            k_min = self.get_min_index(self.phi_min,
                                   params['startx3'], params['dx3'])
        else:
            k_min = int(self.k_min)
        if (not self.do_box_by_index or 
            (self.do_box_by_index and self.k_max is None)):
            k_max = self.get_max_index(self.phi_max,
                                   params['startx3'], params['dx3'],
                                   self.gridsizex3)
        else:
            k_max = int(self.k_max)
        return k_min, k_max

    ### Check settings ###
    def check_raytracing_settings(self):
        raytrace = (self.export_raytracing == 1)
        bad_lwrs = (params['bs1'] % self.lowres1 != 0 or
                    params['bs2'] % self.lowres2 != 0 or
                    params['bs3'] % self.lowres3 != 0)
        zro_lwrs = (self.lowres1 == 0 or
                    self.lowres2 == 0 or
                    self.lowres3 == 0)
        if (raytrace and bad_lwrs):
            raise ValueError("For raytracing, "
                             "block size needs to be divisible by lowres.")
        if (raytrace and zro_lwrs):
            raise ZeroDivisionError("For raytracing, "
                                    "lowres cannot be 0.")
        if (raytrace and self.interpolate_var == 0):
            raise RuntimeWarning("Variable interpolation is highly recommended "
                                 "for raytracing.")
                
    def check_lowres_settings(self):
        if (self.by_block):
            ndivisible_x1 = params['bs1'] % self.lowres1 != 0
            ndivisible_x2 = params['bs2'] % self.lowres2 != 0
            ndivisible_x3 = params['bs3'] % self.lowres3 != 0
        else:
            ndivisible_x1 = int(params['nb1'] * params['bs1'] *
                self.upscalex1) % self.lowres1 != 0
            ndivisible_x2 = int(params['nb2'] * params['bs2'] *
                self.upscalex2) % self.lowres2 != 0
            ndivisible_x3 = int(params['nb3'] * params['bs3'] *
                self.upscalex3) % self.lowres3 != 0
            
        if (ndivisible_x1 or ndivisible_x2 or ndivisible_x3):
            raise ValueError("Incompatible lowres settings in read_grid.")
        
    def check_griddump_exists(self):
        for n in range(params['n_active_total']):
            file_exists = os.path.isfile(
                self.dir + '/gdumps/gdump%d' % self.n_ord[n])
            if (file_exists == 0):
                raise FileExistsError(
                    "gdump file %d doesn't exist." % self.n_ord[n])

class Read_Legacy(Read):
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)

    def read_params(self, fnum):
        # Read in AMR organizational data for every block
        # In the legacy data format, grid file information is 
        # spread across dumps folders. Thus, the block array
        # that was produced is now invalid. 
        # Each individual grid file contains nmax
        if (os.path.isfile(self.dir + "/dumps%d/grid" % fnum)):
            fin = open(self.dir + "/dumps%d/grid" % fnum, "rb")
        else:
            raise FileNotFoundError("Cannot find grid file in %s." 
                %(self.dir + "/dumps%d/grid" % fnum))
        
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        nv = 36
        self.block = np.zeros((nmax, 200), dtype=np.int32, order='C')
        self.n_ord = np.zeros(nmax, dtype=np.int32, order='C')
        gd = np.fromfile(fin, dtype=np.int32, count=nv * nmax, sep='')
        i = 1
        while (os.path.isfile(self.dir + "/gdumps/grid%d" % i)):
            if (i == 1): fin.close()
            fin = open(self.dir + "/gdumps/grid%d" % i, "rb")
            gd  = np.concatenate((gd,
                np.fromfile(fin,
                dtype=np.int32, count=nv * nmax, sep='')), axis=0)
            i += 1
            fin.close()
        # 'F' refers to Fortran order (first index changing fastest)
        gd = gd.reshape((nv, nmax), order='F').T
        self.block[:, 0:nv] = gd

        self.block[:, amr['LEVEL1']] = gd[:, amr['LEVEL']]
        self.block[:, amr['LEVEL2']] = gd[:, amr['LEVEL']]
        self.block[:, amr['LEVEL3']] = gd[:, amr['LEVEL']]

        fin.close()

        # Read in parameters from parameters file.
        print("Loading parameters file")
        if (os.path.isfile(self.dir + "/dumps%d/parameters" % fnum)):
            fin = open(self.dir + "/dumps%d/parameters" % fnum, "rb")
        else:
            raise FileNotFoundError("Cannot find parameters file in %s." 
                %(self.dir + "/dumps%d/parameters" % fnum))
        
        for par, par_type in zip(param_keys, param_types):
            params['%s' %par] = Tools.get_fromfile(fin, par_type)

        del params['Dtl']  
        del params['R0']

        # There are 15 integers before the AMR information
        for n in range(15):
            trash = Tools.get_fromfile(fin, np.int32)
        del trash

        # In legacy data format, grid information is contained in the
        # parameters file as well
        grid_arr1 = np.fromfile(fin, dtype=np.int32, count=nmax, sep='')
        grid_arr2 = np.fromfile(fin, dtype=np.int32, count=nmax, sep='')
        for n in range(nmax):
            self.block[n, amr['REFINED']] = grid_arr1[n]
            self.block[n, amr['ACTIVE']]  = grid_arr2[n]

        i = 0
        for n in range(nmax):
            if self.block[n, amr['ACTIVE']] == 1:
                self.n_ord[i] = n
                i += 1
        
        fin.close()

        for key in phys_keys:
            physics['%s' %key] = 0

        # Set grid spacing
        params['dx1'] = (np.log(params['Rout']) - np.log(params['Rin'])) / (
            (params['bs1'] * params['nb1']))
        fractheta     = -1.0 * params['startx2']
        params['dx2'] = 2.0 * fractheta / (params['bs2'] * params['nb2'])
        params['dx3'] = 2.0 * np.pi / (params['bs3'] * params['nb3'])
        if (self.do_box):
            if (self.r_min is None): self.r_min = params['Rin']
            if (self.r_max is None): self.r_max = params['Rout']
            self.r_min     = max(self.r_min, params['Rin'])
            self.r_max     = min(self.r_max, params['Rout'])
            self.theta_min = max(self.theta_min, 
                                 (np.pi/2.0) * (params['startx2'] + 1.0))
            self.phi_min   = max(self.phi_min, params['startx3'])

        self.check_raytracing_settings()

        ref = self.get_ref_dir(params['nb2'], params['nb3'], params['bs3'])
        amr['max_levelx1'] = ref[1] * np.max(
            self.block[self.n_ord, amr['LEVEL1']])
        amr['max_levelx2'] = ref[2] * np.max(
            self.block[self.n_ord, amr['LEVEL2']])
        amr['max_levelx3'] = ref[3] * np.max(
            self.block[self.n_ord, amr['LEVEL3']])

        self.block[self.n_ord, amr['LEVEL1']] = ref[1]
        self.block[self.n_ord, amr['LEVEL2']] = ref[2]

        self.amr_info = {
            'n_ord'   : self.n_ord,
            'levelx1' : self.block[:, amr['LEVEL1']],
            'levelx2' : self.block[:, amr['LEVEL2']],
            'levelx3' : self.block[:, amr['LEVEL3']],
            'coordx1' : self.block[:, amr['COORD1']],
            'coordx2' : self.block[:, amr['COORD2']],
            'coordx3' : self.block[:, amr['COORD3']]
        }

        self.upscalex1 = 2 ** amr['max_levelx1']
        self.upscalex2 = 2 ** amr['max_levelx2']
        self.upscalex3 = 2 ** amr['max_levelx3']

        # downsample differentials with lowres and 
        # upscale to highest AMR level
        params['dx1'] *= self.lowres1 * (1.0 / self.upscalex1)
        params['dx2'] *= self.lowres2 * (1.0 / self.upscalex2)
        params['dx3'] *= self.lowres3 * (1.0 / self.upscalex3)

        self.amr    = amr
        self.params = params

    def get_ref_dir(self, nb2, nb3, bs3):
        ref_dir = 4 * [None]
        nb2_arr = [3 * 2**i for i in range(1, 7)]
        if nb2 in nb2_arr:
            if (nb3 <= 2):
                ref_dir[1], ref_dir[2] = 0, 0
                if (bs3 > 1): ref_dir[3] = 1
                else:         ref_dir[3] = 0
        else:
            ref_dir[1], ref_dir[2] = 1, 1
            if (bs3 > 1): ref_dir[1] = 1
            else:         ref_dir[1] = 0
        if (nb3 > 2):
            ref_dir[1], ref_dir[2], ref_dir[3] = 1, 1, 1
        return ref_dir

if __name__ == "__main__":
    print("Executing H-AMR Read")

"""
Parameters glossary
--------------------
t: current time in Rg/c
n_active: number of blocks on node 0
n_active_total: total number of active blocks
"active" means currently operating for that dump
nstep: current step of output (dump) file
Dtd: time between outputs (dumps)
Dtr: time between restart dumps
Dtl: meaningless (removed from parameter dictionary)
dump_cnt: current number of output/dump file (fnum)
rdump_cnt: number of most recent restart file for output fnum 
dt: timestep in Rg/c
failed: 0 or 1 (1 means output file did not write properly)
bs1, bs2, bs3: block sizes in each dimension
nb1, nb2, nb3:  number of blocks at the least refined AMR level
nmax: total number of possible blocks
startx1, startx2, startx3:  start coordinates in each dimension
dx1, dx2, dx3: grid spacing in internal coordinates at 0th refinement level
tf: final (stop) time
a: dimensionless black hole spin
gam: adiabatic index (ratio of specific heats)
cour: CFL number
Rin: inner radius in Rg
Rout: outer radius in Rg
R0: meaningless (removed from parameter dictionary)
density scale: rho0 in CGS units
"""
