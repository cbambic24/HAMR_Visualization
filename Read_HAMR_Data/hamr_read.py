""""
Read in H-AMR primitive variable outputs and grid data.
Functions accelerated through Cython and C++
"""

import os
import numpy as np
import sys
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
    'DO_NUCLEAR', 'DOTRACER', 'DUMP_DISS'
]

# Set parameter types
param_dtype = np.dtype([
    ('t',              np.float64),
    ('n_active',       np.int32),
    ('n_active_total', np.int32),
    ('nstep',          np.int32),
    ('Dtd',            np.float64),
    ('Dtl',            np.float64),
    ('Dtr',            np.float64),
    ('dump_cnt',       np.int32),
    ('rdump_cnt',      np.int32),
    ('dt',             np.float64),
    ('failed',         np.int32),
    ('bs1',            np.int32),
    ('bs2',            np.int32),
    ('bs3',            np.int32),
    ('nmax',           np.int32),
    ('nb1',            np.int32),
    ('nb2',            np.int32),
    ('nb3',            np.int32),
    ('startx1',        np.float64),
    ('startx2',        np.float64),
    ('startx3',        np.float64),
    ('dx1',            np.float64),
    ('dx2',            np.float64),
    ('dx3',            np.float64),
    ('tf',             np.float64),
    ('a',              np.float64),
    ('gam',            np.float64),
    ('cour',           np.float64),
    ('Rin',            np.float64),
    ('Rout',           np.float64),
    ('R0',             np.float64),
    ('density_scale',  np.float64)
])

param_keys = list(param_dtype.names)

class Tools:
    def get_fromfile(infile, dtype):
        return np.fromfile(infile, dtype=dtype, count=1, sep='')[0]

    def zeroNd(shape, Narr, **kwargs): # returns Narr 2d arrays of zeros
        dtype = kwargs.get('dtype', np.float32)
        arrs = []
        for _ in range(Narr):
            arrs.append(np.zeros(shape, dtype=dtype))
        return arrs if Narr !=1 else arrs[0]

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
        self.resolution  = kwargs.get('resolution', 0)

        self.amr     = {}
        self.params  = {}
        self.physics = {}

        #   ACTIVE .. TIMELEVEL -> columns 0-36
        #   LEVEL1 .. NBR6_8    -> columns 110-136
        #   NBR1P  .. NBR6P     -> columns 161-166
        for i, key in enumerate(amr_keys):
            if (i < 37):   self.amr[key] = i
            elif (i < 64): self.amr[key] = i + 73
            else:          self.amr[key] = i + 97

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
            
        self.start_block   = kwargs.get('start_block', None)
        self.end_block     = kwargs.get('end_block',   None)
        self.select_block  = 0
        if ('start_block' in kwargs or 'end_block' in kwargs):
            self.select_block = 1 if self.by_block else 0

        self.BYTES_PER_VAR = 4

        # This gets called once during initialization.
        self.read_block()

        self.options = {
            'low_resx1' : self.lowres1,
            'low_resx2' : self.lowres2,
            'low_resx3' : self.lowres3
        }

    def read_block(self):
        # Read in AMR organizational data for every block.
        # First, compute the number of variables stored in grid file.
        gdumps = f'{self.dir}/gdumps'
        if (os.path.isfile(f'{gdumps}/grid')):
            fin = open(f'{gdumps}/grid', "rb")
            size = os.path.getsize(f'{gdumps}/grid')
            i = 1
            while (os.path.isfile(f'{gdumps}/grid{i}')):
                if (i == 1): fin.close()
                fin = open(f'{gdumps}/grid{i}', "rb")
                size += os.path.getsize(f'{gdumps}/grid{i}')
                fin.close()
                i += 1
            fin = open(f'{gdumps}/grid', "rb")
            # nmax is the total number of possible blocks
            nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            # subtract off BYTES_PER_VAR since nmax is in the first slot
            # nv is the number of variables required to encode grid structure
            nv = (size - self.BYTES_PER_VAR) // nmax // self.BYTES_PER_VAR
        else:
            raise FileNotFoundError(f"Cannot find grid file in {gdumps}/")

        # Next, read in the information for the "block" array
        # which contains all information to encode grid structure
        # 'C' refers to C order (last index changing fastest)
        # self.block array stores info for grid structure
        self.block = np.zeros((nmax, 200), dtype=np.int32, order='C')
        gd = np.fromfile(fin, dtype=np.int32, count=nv * nmax, sep='')
        i = 1
        while (os.path.isfile(f'{gdumps}/grid{i}')):
            if (i == 1): fin.close()
            fin = open(f'{gdumps}/grid{i}', "rb")
            gd  = np.concatenate((gd, np.fromfile(fin,
                dtype=np.int32, count=nv * nmax, sep='')), axis=0)
            i += 1
            fin.close()
        # 'F' refers to Fortran order (first index changing fastest)
        gd = gd.reshape((nv, nmax), order='F').T
        self.block[:, 0:nv] = gd
        # indices from 0 to nv defined in "amr" dictionary.
        fin.close()

    def read_params(self, fnum, **kwargs):
        if not kwargs.get('suppress_output', False):
            print("Loading parameters file")
        # Read in parameters from parameters file.
        if (os.path.isfile(f"{self.dir}/dumps{fnum}/parameters")):
            fin = open(f"{self.dir}/dumps{fnum}/parameters", "rb")
        else:
            raise FileNotFoundError("Cannot find parameters file in %s." 
                %(f"{self.dir}/dumps{fnum}/parameters"))

        if (os.path.isfile(f"{self.dir}/dumps{fnum}/grid")):
            import warnings
            warnings.warn("Warning: this may be a legacy data set. " + 
                          "Please use Read_Legacy instead of Read.") 

        # One read of the whole 204-byte record; leaves the file positioned
        # exactly where the field-by-field loop did.
        record = np.fromfile(fin, dtype=param_dtype, count=1)[0]
        for par in param_dtype.names:
            self.params[par] = record[par]

        del self.params['Dtl']  
        del self.params['R0']

        # There are 13 integers before the physics integer flag
        for n in range(0, 13):
            trash = Tools.get_fromfile(fin, np.int32)
        del trash
        phys_int = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        # Using phys_int, determine which physics modules are active
        for key in phys_keys:
            self.physics[key] = 0
        phys_int = '{:09}'.format(phys_int)

        if (phys_int[-1] == '1'): self.physics['RAD_M1']       = 1
        if (phys_int[-2] == '1'): self.physics['RESISTIVE']    = 1
        if (phys_int[-3] == '1'): self.physics['TWO_T']        = 1
        if (phys_int[-4] == '1'): self.physics['P_NUM']        = 1
        if (phys_int[-5] == '1'): self.physics['DO_YE']        = 1
        if (phys_int[-6] == '1'): self.physics['NEUTRINOS_M1'] = 1
        if (phys_int[-7] == '1'): self.physics['DO_NUCLEAR']   = 1
        if (phys_int[-8] == '1'): self.physics['DOTRACER']     = 1
        if (phys_int[-9] == '1'): self.physics['DUMP_DISS']    = 1
        if (phys_int[-9] == '2'): self.physics['DUMP_DISS']    = 2

        trash = Tools.get_fromfile(fin, np.int32)
        del trash

        # Set grid spacing
        pars = self.params
        Rin  = pars['Rin']
        Rout = pars['Rout']
        pars['dx1'] = (np.log(Rout) - np.log(Rin)) / (pars['bs1'] * pars['nb1'])
        fractheta   = -1.0 * pars['startx2']
        pars['dx2'] = 2.0 * fractheta / (pars['bs2'] * pars['nb2'])
        pars['dx3'] = 2.0 * np.pi / (pars['bs3'] * pars['nb3'])

        nb = pars['n_active_total']
        # n_ord translates order of blocks in Z curve
        self.n_ord = np.zeros((pars['nmax']), dtype=np.int32, order='C')
        node       = np.copy(self.n_ord)
        timelevel  = np.copy(self.n_ord)

        size = os.path.getsize(f"{self.dir}/dumps{fnum}/parameters")
        # This logic accounts for old and new file formats.
        # 66 is the number of variables in output.
        ntot = pars['n_active_total']
        old_size3 = 66 * self.BYTES_PER_VAR + 3 * ntot * self.BYTES_PER_VAR
        old_size2 = 66 * self.BYTES_PER_VAR + 2 * ntot * self.BYTES_PER_VAR
        if (size >= old_size3):
            for n in range(ntot):
                self.n_ord[n] = Tools.get_fromfile(fin, np.int32)
                timelevel[n]  = Tools.get_fromfile(fin, np.int32)
                node[n]       = Tools.get_fromfile(fin, np.int32)
        elif (size >= old_size2):
            for n in range(ntot):
                self.n_ord[n] = Tools.get_fromfile(fin, np.int32)
                timelevel[n]  = Tools.get_fromfile(fin, np.int32)

        fin.close()

        if (self.do_box):
            if (self.r_min is None): self.r_min = pars['Rin']
            if (self.r_max is None): self.r_max = pars['Rout']
            self.r_min     = max(self.r_min, pars['Rin'])
            self.r_max     = min(self.r_max, pars['Rout'])
            self.theta_min = max(self.theta_min, (np.pi / 2.0) * (pars['startx2'] + 1.0))
            self.phi_min   = max(self.phi_min, pars['startx3'])

        # read individual blocks
        if (self.start_block is None): self.start_block = 0
        if (self.end_block is None):   self.end_block   = nb
        self.start_block = max(self.start_block, 0)
        self.end_block   = min(self.end_block,  nb)

        self.check_raytracing_settings()

        self.amr['max_levelx1'] = np.max(self.block[self.n_ord, self.amr['LEVEL1']])
        self.amr['max_levelx2'] = np.max(self.block[self.n_ord, self.amr['LEVEL2']])
        self.amr['max_levelx3'] = np.max(self.block[self.n_ord, self.amr['LEVEL3']])

        self.amr_info = {
            'n_ord'   : self.n_ord,
            'levelx1' : self.block[:, self.amr['LEVEL1']].copy(),
            'levelx2' : self.block[:, self.amr['LEVEL2']].copy(),
            'levelx3' : self.block[:, self.amr['LEVEL3']].copy(),
            'coordx1' : self.block[:, self.amr['COORD1']].copy(),
            'coordx2' : self.block[:, self.amr['COORD2']].copy(),
            'coordx3' : self.block[:, self.amr['COORD3']].copy()
        }

        self.upscalex1 = 2 ** self.amr['max_levelx1']
        self.upscalex2 = 2 ** self.amr['max_levelx2']
        self.upscalex3 = 2 ** self.amr['max_levelx3']

        # Downsample differentials with lowres and
        # upscale to highest AMR level.
        pars['dx1'] *= self.lowres1 * (1.0 / self.upscalex1)
        pars['dx2'] *= self.lowres2 * (1.0 / self.upscalex2)
        pars['dx3'] *= self.lowres3 * (1.0 / self.upscalex3)

        self.block_list = self.get_block_list()
        self.amr_info['block_list'] = self.block_list

    def get_block_list(self):
        """Local Block IDs to read, as an int32 array.

        by_block=False keeps every block: that path upscales onto one
        global unigrid, and the box is applied per cell inside the C++
        (checkIndexInRange) with a whole-block early exit
        (checkNotInBlock). by_block=True keeps blocks whole, so the box
        has to be resolved to a set of blocks here -- see blocks_in_box.
        """
        nb = self.params['n_active_total']
        if not self.by_block:
            return np.arange(nb, dtype=np.int32)
        if (self.do_box or self.do_box_by_index):
            # The batch window composes with the box rather than being
            # replaced by it: returning the whole box for every batch would
            # re-read and re-sum the same blocks once per batch. Defaults are
            # 0..nb, so this is a no-op when no batch window is set.
            blocks = self.blocks_in_box().astype(np.int32)
            return blocks[(blocks >= self.start_block) &
                          (blocks <  self.end_block)]
        return np.arange(self.start_block, self.end_block, dtype=np.int32)

    def read_grid(self, fnum, **kwargs):
        skip_read_params = kwargs.get('skip_read_params', False)
        direct           = kwargs.get('direct', False)
        if not skip_read_params:
            self.read_params(fnum, suppress_output=True)

        # Read in grid from gdump files or directly compute grid variables
        self.check_lowres_settings()

        nb = self.params['n_active_total']

        if (self.by_block):
            bs1_new = int(self.params['bs1'] / self.lowres1)
            bs2_new = int(self.params['bs2'] / self.lowres2)
            bs3_new = int(self.params['bs3'] / self.lowres3)

            nx = bs1_new
            ny = bs2_new
            nz = bs3_new

            # The direct path is sized to the selection, because
            # coords_directBK builds exactly the blocks in block_list and
            # nothing else. One expression covers all three cases:
            # block_list is arange(nb) with no selection, arange(start, end)
            # for start_block/end_block, and the scattered set of Local
            # Block IDs overlapping a box -- so len() is always the right
            # leading dimension.
            #
            # The non-direct path keeps dim = nb: Read::readGrid writes each
            # block at its Local Block ID offset, so the buffers must span
            # the whole dump and the selection is gathered out afterwards.
            if direct:
                dim = len(self.block_list)
            else:
                dim = nb

            i_min, i_max = 0, nx
            j_min, j_max = 0, ny
            k_min, k_max = 0, nz

        else:
            pars = self.params
            self.gridsizex1 = int(pars['nb1']*pars['bs1']*self.upscalex1 / self.lowres1)
            self.gridsizex2 = int(pars['nb2']*pars['bs2']*self.upscalex2 / self.lowres2)
            self.gridsizex3 = int(pars['nb3']*pars['bs3']*self.upscalex3 / self.lowres3)

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
                    raise ValueError(f"Bad box selection for file {fnum} in read_grid.")
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

        # Extent of the metric along x3. An axisymmetric metric is stored once
        # per (x1, x2) and broadcast over phi; otherwise it carries the full
        # x3 extent. Everything downstream that hands these arrays to the C++
        # layer must use nz_g, not a literal, or it will walk the wrong
        # number of cells.
        nz_g = 1 if self.axisym else nz

        gcov, gcon, dxdxp = Tools.zeroNd([4, 4, dim, nx, ny, nz_g], 3)
        gdet              = Tools.zeroNd([dim, nx, ny, nz_g], 1)

        if self.resolution:
            res_level_x1, res_level_x2, res_level_x3 = Tools.zeroNd(
                [dim, nx, ny, nz], 3, dtype=np.intc)

        self.block_info = {
            'i_min' : i_min,
            'i_max' : i_max,
            'j_min' : j_min,
            'j_max' : j_max,
            'k_min' : k_min,
            'k_max' : k_max,
        }

        self.block_info['start_block'] = self.start_block
        self.block_info['end_block']   = self.end_block

        self.Toolsc = hamr_readc.PyTools()

        if direct:
            if not self.by_block:
                x1, x2, x3, r, theta, phi = self.coords_direct(x1, x2, x3, 
                    r, theta, phi, [dim, nx, ny, nz])
            elif self.by_block:
                x1, x2, x3, r, theta, phi = self.coords_directBK(x1, x2, x3,
                    r, theta, phi, [dim, nx, ny, nz])
            
            # Logic for binary black holes would have to go here.

            # Metric, inverse metric and the transformation matrix dxdxp
            # taking spherical Kerr-Schild to internal coordinates. gcon is
            # built analytically rather than by inverting gcov numerically:
            # dxdxp is diagonal, so the inverse congruence is an elementwise
            # scale and invert_4x4 is not needed on this path.
            gcov, gcon, dxdxp = self.metric_direct(gcov, gcon, dxdxp, r, theta)

            # Calculate determinant of metric
            self.Toolsc.det_4x4(gcov, gdet, dim, nx, ny, nz_g)
            gdet = np.sqrt(-1.0*gdet)
        
        else:
            Readc = hamr_readc.PyRead(self.dir, self.axisym, self.by_block,
                self.export_raytracing, self.amr, self.amr_info,
                self.block_info, self.options, self.params, self.physics)
            
            Readc.read_grid(x1, x2, x3, r, theta, phi, gcov, gcon, gdet, dxdxp)

            if self.coords1D and not self.by_block:
                x1, r     = x1[0, :, 0, 0].copy(), r[    0, :, 0, 0].copy()
                x2, theta = x2[0, 0, :, 0].copy(), theta[0, 0, :, 0].copy()
                x3, phi   = x3[0, 0, 0, :].copy(), phi[  0, 0, 0, :].copy()
            elif self.coords1D and self.by_block:
                x1, r     = x1[:, :, 0, 0].copy(), r[    :, :, 0, 0].copy()
                x2, theta = x2[:, 0, :, 0].copy(), theta[:, 0, :, 0].copy()
                x3, phi   = x3[:, 0, 0, :].copy(), phi[  :, 0, 0, :].copy()

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
        # The direct path already built only the selected blocks, so it has
        # nothing left to cut -- doing it anyway would cut a second time
        # and return empty arrays for any start_block > 0. The non-direct
        # path wrote each block at its Local Block ID offset into full-size
        # buffers, so the selection is extracted here.
        #
        # block_list indexes rather than slices, so the selection no longer
        # has to be contiguous: a box picks out scattered Local Block IDs
        # (they are ordered node-major, so even a compact box fragments
        # into ~one run per GPU) and one fancy index gathers them all.
        # Fancy indexing already returns a copy; the .copy(order='C') is
        # kept so the result is contiguous for the C++ layer downstream.
        #
        # A plain view would keep the full [n_active_total, ...] buffer
        # alive for as long as the caller holds the grid -- ~10 GB of
        # mapping pinned to expose a few hundred MB, and an array whose
        # .nbytes bears no relation to what it costs. That is what made
        # read_grid look like it leaked.
        if (len(self.block_list) < nb and not direct):
            blocks = self.block_list
            for key in self.grid.keys():
                tnsr = self.grid[key].ndim > 4
                if tnsr:
                    self.grid[key] = (
                        self.grid[key][:, :, blocks]).copy(order='C')
                else:
                    self.grid[key] = (
                        self.grid[key][blocks]).copy(order='C')

        if (self.cgs_grid):
            self.restore_units_grid()

        if (self.resolution and not direct):
            Readc.read_grid_resolution(res_level_x1, res_level_x2, res_level_x3)
        elif (self.resolution and direct):
            Readc = hamr_readc.PyRead(self.dir, self.axisym, self.by_block,
                self.export_raytracing, self.amr, self.amr_info,
                self.block_info, self.options, self.params, self.physics)   
            Readc.read_grid_resolution(res_level_x1, res_level_x2, res_level_x3)
            
        if (self.resolution):
            self.grid['level_x1'] = res_level_x1
            self.grid['level_x2'] = res_level_x2
            self.grid['level_x3'] = res_level_x3

    def read_prims(self, fnum, **kwargs):
        skip_read_params = kwargs.get('skip_read_params', False)
        if not skip_read_params:
            self.read_params(fnum, suppress_output=True)

        pars = self.params

        if (self.by_block):
            bs1_new = int(pars['bs1'] / self.lowres1)
            bs2_new = int(pars['bs2'] / self.lowres2)
            bs3_new = int(pars['bs3'] / self.lowres3)

            nx  = bs1_new
            ny  = bs2_new
            nz  = bs3_new
            nb  = pars['n_active_total']
            # readPrimsData writes block i of blockList to slot i, so only
            # the selected blocks need to exist and nothing dump-sized is
            # allocated. Same expression read_grid uses on its direct path,
            # which is what keeps H.data and H.grid the same length and in
            # the same order for every kind of selection -- whole dump,
            # start_block/end_block, or a box.
            dim = len(self.block_list)
                
            i_min, i_max = 0, nx
            j_min, j_max = 0, ny
            k_min, k_max = 0, nz
        else:
            self.gridsizex1 = int(pars['nb1']*pars['bs1']*self.upscalex1 / self.lowres1)
            self.gridsizex2 = int(pars['nb2']*pars['bs2']*self.upscalex2 / self.lowres2)
            self.gridsizex3 = int(pars['nb3']*pars['bs3']*self.upscalex3 / self.lowres3)

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

        # Allocate memory. Only the arrays this dump's physics actually
        # produce are created, and each goes straight into `prims` under
        # its H.data name. readPrims takes the dict, so a disabled field is
        # simply an absent key -- replacing the old convention of passing a
        # dummy alias of rho/uu for every field that was switched off. That
        # convention also meant ~15 pointers aliased rho on every call.
        prims = {}
        prims['rho'], prims['u_gas'] = Tools.zeroNd([dim, nx, ny, nz], 2)
        prims['uu'],  prims['B']     = Tools.zeroNd([4, dim, nx, ny, nz], 2)

        if (self.physics['RAD_M1']):
            prims['u_rad']  = Tools.zeroNd([dim, nx, ny, nz], 1)
            prims['uu_rad'] = Tools.zeroNd([4, dim, nx, ny, nz], 1)

        if (self.physics['RESISTIVE']):
            prims['E'] = Tools.zeroNd([4, dim, nx, ny, nz], 1)

        if (self.physics['TWO_T']):
            prims['Te'], prims['Ti'] = Tools.zeroNd([dim, nx, ny, nz], 2)

        if (self.physics['P_NUM']):
            prims['p_num'] = Tools.zeroNd([dim, nx, ny, nz], 1)

        if (self.physics['DO_YE']):
            prims['Ye'] = Tools.zeroNd([dim, nx, ny, nz], 1)

        if (self.physics['DO_NUCLEAR']):
            prims['X_alpha'], prims['X_atm'] = Tools.zeroNd([dim, nx, ny, nz], 2)

        if (self.physics['NEUTRINOS_M1']):
            (prims['u_nu1'],  prims['u_nu2'],  prims['u_nu3'])  = Tools.zeroNd([dim, nx, ny, nz], 3)
            (prims['uu_nu1'], prims['uu_nu2'], prims['uu_nu3']) = Tools.zeroNd([4, dim, nx, ny, nz], 3)
            (prims['n_nu1'],  prims['n_nu2'],  prims['n_nu3'])  = Tools.zeroNd([dim, nx, ny, nz], 3)

        if (self.physics['DOTRACER']):
            prims['tr1'], prims['tr2'] = Tools.zeroNd([dim, nx, ny, nz], 2)

        if (self.physics['DUMP_DISS']):
            if (self.physics['DUMP_DISS'] == 1):
                prims['dissipation'] = Tools.zeroNd([dim, nx, ny, nz], 1)
                if (self.cgs_data):
                    self.diss_dbl = Tools.zeroNd([dim, nx, ny, nz], 1, dtype=np.float64)
            elif (self.physics['DUMP_DISS'] == 2):
                prims['dissipation'], prims['rad_src'] = Tools.zeroNd([dim, nx, ny, nz], 2)
                if (self.cgs_data):
                    self.diss_dbl, self.rad_src_dbl = Tools.zeroNd([dim, nx, ny, nz], 2, dtype=np.float64)

        self.block_info = {
            'i_min' : i_min,
            'i_max' : i_max,
            'j_min' : j_min,
            'j_max' : j_max,
            'k_min' : k_min,
            'k_max' : k_max
        }

        self.block_info['start_block'] = self.start_block
        self.block_info['end_block']   = self.end_block

        Readc = hamr_readc.PyRead(self.dir, self.axisym, self.by_block,
            self.export_raytracing, self.amr, self.amr_info,
            self.block_info, self.options, self.params, self.physics)

        Readc.read_prims(fnum, prims)

        # The arrays readPrims filled ARE the data dictionary; there is no
        # second set of conditionals rebuilding it key by key.
        self.data = prims

        # No post-hoc slice here any more: the arrays were allocated at the
        # size of the selection and readPrimsData wrote into them at
        # (n - blkStrt), so they already hold exactly the requested blocks.
        # Slicing again would apply [strt:end] a second time and return
        # empty arrays for any start_block > 0.

        if (self.cgs_data):
            self.restore_units_prims()

    def restore_units_prims(self):
        # Output primitives in CGS units
        rho0 = self.params['density_scale'] # g/cm^3
        c    = CONSTANTS['c']          # cm/s
        self.data['rho']   *= rho0         # g/cm^3
        self.data['u_gas'] *= rho0 * c * c # erg/cm^3
        self.data['uu']    *= c            # cm/s
        self.data['B']     *= np.sqrt(4 * np.pi) * np.sqrt(rho0) * c # Gauss

        if (self.physics['RAD_M1']):
            self.data['u_rad']  *= rho0 * c * c     # erg/cm^3
            self.data['uu_rad'] *= c # cm/s

        if (self.physics['RESISTIVE']):
            self.data['E'] *= np.sqrt(rho0) * c # statVolt/cm (same as Gauss)

        if (self.physics['TWO_T']):
            kB   = CONSTANTS['kB']
            mp   = CONSTANTS['mp']
            X_ab = CONSTANTS['X_AB']
            Y_ab = CONSTANTS['Y_AB']
            mu_i = 4.0 / (4.0 * X_ab + Y_ab)
            mu_e = 2.0 / (1.0 + X_ab)

            self.data['Te'] *= mu_e * mp * c * c / kB 
            self.data['Ti'] *= mu_i * mp * c * c / kB 
        
        if (self.physics['DUMP_DISS']):
            if (self.MBH == 0.0):
                raise TypeError("Must define black hole mass in Read class.")
            G    = CONSTANTS['G']
            Msun = CONSTANTS['Msun']
            c    = CONSTANTS['c']
            Rg = G*self.MBH*Msun/(c*c)

            # Dissipation/radiation production rates in erg/cm^3 s
            diss = self.data.pop('dissipation')
            diss = np.nan_to_num(diss, copy=False)
            np.copyto(self.diss_dbl, diss, casting='safe')
            self.data['dissipation']  = self.diss_dbl
            self.data['dissipation'] *= rho0 * c * c * (c / Rg)
            del diss

            if (self.physics['DUMP_DISS'] == 2):
                rad_src = self.data.pop('rad_src')
                rad_src = np.nan_to_num(rad_src, copy=False)
                np.copyto(self.rad_src_dbl, rad_src, casting='safe')
                self.data['rad_src']  = self.rad_src_dbl
                self.data['rad_src'] *= rho0 * c * c * (c / Rg)
                del rad_src
    
    def restore_units_grid(self):
        # Output grid variables in CGS units
        if (self.MBH == 0.0):
            raise TypeError("Must define black hole mass in Read class.")
        G    = CONSTANTS['G']
        Msun = CONSTANTS['Msun']
        c    = CONSTANTS['c']
        Rg = G*self.MBH*Msun/(c*c)
        self.grid['r'] *= Rg

    def broadcast_coords(self, r, theta):
        """Reshape r and theta to broadcast against (dim, nx, ny, nz)."""
        if not self.coords1D:
            # Keeps the phi index for an axisymmetric metric.
            return r[..., :1], theta[..., :1]
        if self.by_block:
            return r[:, :, None, None], theta[:, None, :, None]
        return r[None, :, None, None], theta[None, None, :, None]

    def get_gcov_KS(self, gcov, r, theta):
        """Covariant metric in spherical Kerr-Schild coordinates.

        Fills the buffer read_grid allocated rather than returning a new
        array. Components not assigned here are identically zero and keep
        their zeroNd value.
        """
        a    = self.params['a']
        cth  = np.cos(theta)
        s2   = np.sin(theta) ** 2
        rho2 = r * r + a * a * cth * cth
        tr   = 2. * r / rho2

        gcov[0, 0] = -1. + tr
        gcov[0, 1] = tr
        gcov[0, 3] = -a * s2 * tr
        gcov[1, 0] = gcov[0, 1]
        gcov[1, 1] = 1. + tr
        gcov[1, 3] = -a * s2 * (1. + tr)
        gcov[2, 2] = rho2
        gcov[3, 0] = gcov[0, 3]
        gcov[3, 1] = gcov[1, 3]
        gcov[3, 3] = s2 * (rho2 + a * a * s2 * (1. + tr))
        return gcov

    def get_gcon_KS(self, gcon, r, theta):
        """Inverse metric in spherical Kerr-Schild coordinates."""
        a     = self.params['a']
        cth   = np.cos(theta)
        s2    = np.sin(theta) ** 2
        rho2  = r * r + a * a * cth * cth
        Delta = r * r - 2. * r + a * a

        gcon[0, 0] = -1. - 2. * r / rho2
        gcon[0, 1] = 2. * r / rho2
        gcon[1, 0] = gcon[0, 1]
        gcon[1, 1] = Delta / rho2
        gcon[1, 3] = a / rho2
        gcon[2, 2] = 1. / rho2
        gcon[3, 1] = gcon[1, 3]
        gcon[3, 3] = 1. / (rho2 * s2)
        return gcon

    def dxdxp_direct(self, dxdxp, r):
        """Jacobian from spherical Kerr-Schild to internal coordinates."""
        dxdxp[0, 0] = 1.0
        dxdxp[1, 1] = r
        dxdxp[2, 2] = np.pi / 2.0
        dxdxp[3, 3] = 1.0
        return dxdxp

    def dxpdx_diag(self, r):
        """Diagonal of the inverse Jacobian dx'/dx."""
        return (1.0, 1.0 / r, 2.0 / np.pi, 1.0)

    def metric_direct(self, gcov, gcon, dxdxp, r, theta):
        """Metric, inverse metric and Jacobian in internal coordinates."""
        r_b, th_b = self.broadcast_coords(r, theta)

        self.get_gcov_KS(gcov, r_b, th_b)
        self.get_gcon_KS(gcon, r_b, th_b)
        dxdxp = self.dxdxp_direct(dxdxp, r_b)

        # Allocates to gcov directly.
        np.einsum('kl...,ki...,lj...->ij...', gcov, dxdxp, dxdxp, out=gcov)

        # Uses analytic gcon.
        d = self.dxpdx_diag(r_b)
        for i, j in ((0, 1), (1, 0), (1, 1), (1, 3), (2, 2), (3, 1)):
            gcon[i, j] *= d[i] * d[j]
        return gcov, gcon, dxdxp

    def coords_direct(self, x1, x2, x3, r, theta, phi, indices):
        [dim, nx, ny, nz] = indices

        # Always the GLOBAL origin, including under do_box. The box offset
        # is already carried by i_min/j_min/k_min, which get_?box_coords
        # derived from r_min/theta_min/phi_min against this same origin --
        # so taking a box origin here as well (startx1 = log(r_min)) added
        # it twice. Measured on a r_min=4 selection: i_min=657 and
        # log(4) + (657+0.5)*dx1 put the inner edge at r=13.198 instead of
        # r=4.003, silently, with correct shapes throughout.
        #
        # The global form is also the only one that agrees with the C++
        # reader: it returns the cell centres of the actual simulation
        # cells, whereas a box origin with local indices would lay down a
        # fresh grid starting exactly at r_min (r=4.0036, off by a fraction
        # of a cell). Keeping them identical is what lets direct=True be
        # checked against direct=False.
        startx1 = self.params['startx1']
        startx2 = self.params['startx2']
        startx3 = self.params['startx3']

        i_min, i_max = self.block_info['i_min'], self.block_info['i_max']
        j_min, j_max = self.block_info['j_min'], self.block_info['j_max']
        k_min, k_max = self.block_info['k_min'], self.block_info['k_max']

        x1_1D = startx1 + (np.arange(i_min, i_max) + 0.5) * self.params['dx1']
        x2_1D = startx2 + (np.arange(j_min, j_max) + 0.5) * self.params['dx2']
        x3_1D = startx3 + (np.arange(k_min, k_max) + 0.5) * self.params['dx3']

        if self.coords1D:
            x1[:] = x1_1D
            x2[:] = x2_1D
            x3[:] = x3_1D
        else:
            x1[...] = x1_1D[None, :, None, None]
            x2[...] = x2_1D[None, None, :, None]
            x3[...] = x3_1D[None, None, None, :]

        r     = np.exp(x1)
        theta = (x2 + 1.0) * (np.pi/2.0)
        phi   = x3
        return x1, x2, x3, r, theta, phi

    def global_coords1D(self, nr_out=None):
        """Global 1D r and theta axes of the finest uniform grid.

        A by_block read carries each block's own axes, but unfold/unfold2D
        place results on the full dump grid, so a caller saving axes
        alongside unfolded data needs these rather than grid['r'][0].
        Same origin and transform as coords_direct. nr_out matches the
        truncation unfold applies under --r_max, so the axis and the data it
        labels are the same length.
        """
        pars = self.params
        nr   = int(pars['nb1'] * pars['bs1'] * self.upscalex1 / self.lowres1)
        nth  = int(pars['nb2'] * pars['bs2'] * self.upscalex2 / self.lowres2)
        if nr_out is not None:
            nr = min(nr, int(nr_out))
        x1   = pars['startx1'] + (np.arange(nr)  + 0.5) * pars['dx1']
        x2   = pars['startx2'] + (np.arange(nth) + 0.5) * pars['dx2']
        return np.exp(x1), (x2 + 1.0) * (np.pi / 2.0)

    def block_coords(self, coord_key, level_key, max_level, blocks, bs,
                     startx, dx, dtype):
        """Cell-centre coordinate of every cell of the given blocks.

        coords_direct puts global cell ii at startx + (ii + 0.5)*dx on one
        uniform grid. Under AMR each block carries its own refinement level:
        a block at level l has cells 2**(max_level - l) finest-cells wide and
        starts at coord * upscale * bs on the finest grid. Same arithmetic as
        AccretionBK.calc_dx and unfold, and params already holds dx rescaled
        to the finest level, so it drops straight in.
        """
        n   = self.amr_info['n_ord'][blocks]
        up  = (2 ** (max_level - self.amr_info[level_key][n])).astype(np.int64)
        i_0 = self.amr_info[coord_key][n].astype(np.int64) * up * bs
        cells = np.arange(bs, dtype=np.float64)
        x = startx + (i_0[:,None] + (cells[None,:] + 0.5) * up[:,None]) * dx
        return x.astype(dtype)

    def coords_directBK(self, x1, x2, x3, r, theta, phi, indices):
        """Coordinates of the selected blocks, built from the AMR metadata.

        Deliberately does not call the C++ reader. read_grid_coords fills
        every block of the dump at its global offset, which forces the caller
        to allocate n_active_total blocks' worth of buffers and slice the
        selection out afterwards -- and since that slice is a view, the
        full-size buffers are never released. Computing the coordinates here
        touches only the requested blocks, so memory scales with the batch
        rather than with the dump.
        """
        [dim, nx, ny, nz] = indices
        startx1 = self.params['startx1']
        startx2 = self.params['startx2']
        startx3 = self.params['startx3']

        # Local Block IDs to build, straight from block_list. Deriving a
        # contiguous arange(start_block, start_block + dim) here instead
        # silently ignored a box selection: with only a box set,
        # select_block is 0 and dim is nb, so it rebuilt every block and
        # the box had no effect on this path at all.
        blocks = self.block_list

        x1_1D = self.block_coords('coordx1', 'levelx1', self.amr['max_levelx1'],
                                  blocks, nx, startx1, self.params['dx1'],
                                  x1.dtype)
        x2_1D = self.block_coords('coordx2', 'levelx2', self.amr['max_levelx2'],
                                  blocks, ny, startx2, self.params['dx2'],
                                  x2.dtype)
        x3_1D = self.block_coords('coordx3', 'levelx3', self.amr['max_levelx3'],
                                  blocks, nz, startx3, self.params['dx3'],
                                  x3.dtype)

        if self.coords1D:
            # Return the reduced arrays. The 4D buffers the caller allocated
            # are never written and are dropped on return.
            x1, x2, x3 = x1_1D, x2_1D, x3_1D
        else:
            x1[...] = x1_1D[:, :, None, None]
            x2[...] = x2_1D[:, None, :, None]
            x3[...] = x3_1D[:, None, None, :]

        r     = np.exp(x1)
        theta = (x2 + 1.0) * (np.pi/2.0)
        phi   = x3
        return x1, x2, x3, r, theta, phi

    ### Helper functions for box selection ###
    def get_min_index(self, box_min, startx, dx):
        return max(np.int32((box_min - (startx + 0.5*dx))/dx + 1), 0)

    def get_max_index(self, box_max, startx, dx, max_val):
        return min(np.int32((box_max - (startx + 0.5*dx))/dx + 1), max_val)

    def get_rbox_coords(self):
        pars = self.params
        if (not self.do_box_by_index or
            (self.do_box_by_index and self.i_min is None)):
            i_min = self.get_min_index(np.log(self.r_min),
                                   pars['startx1'], pars['dx1'])
        else:
            i_min = int(self.i_min)
        if (not self.do_box_by_index or
            (self.do_box_by_index and self.i_max is None)):
            i_max = self.get_max_index(np.log(self.r_max),
                                   pars['startx1'], pars['dx1'],
                                   self.gridsizex1)
        else:
            i_max = int(self.i_max)
        return i_min, i_max

    def get_tbox_coords(self):
        pars = self.params
        if (not self.do_box_by_index or
            (self.do_box_by_index and self.j_min is None)):
            j_min = self.get_min_index((2.0 / np.pi) * self.theta_min - 1.0,
                                   pars['startx2'], pars['dx2'])
        else:
            j_min = int(self.j_min)
        if (not self.do_box_by_index or
            (self.do_box_by_index and self.j_max is None)):
            j_max = self.get_max_index((2.0 / np.pi) * self.theta_max - 1.0,
                                   pars['startx2'], pars['dx2'],
                                   self.gridsizex2)
        else:
            j_max = int(self.j_max)
        return j_min, j_max

    def get_pbox_coords(self):
        pars = self.params
        if (not self.do_box_by_index or
            (self.do_box_by_index and self.k_min is None)):
            k_min = self.get_min_index(self.phi_min,
                                   pars['startx3'], pars['dx3'])
        else:
            k_min = int(self.k_min)
        if (not self.do_box_by_index or
            (self.do_box_by_index and self.k_max is None)):
            k_max = self.get_max_index(self.phi_max,
                                   pars['startx3'], pars['dx3'],
                                   self.gridsizex3)
        else:
            k_max = int(self.k_max)
        return k_min, k_max

    def blocks_in_box(self):
        """Local Block IDs of every block overlapping the box set by
        do_box/do_box_by_index.

        The by_block=True counterpart to i_min..k_max, which under
        by_block=False filter individual cells inside the C++ but are
        ignored entirely under by_block=True (checkIndexInRange returns 1
        unconditionally there). Blocks are read whole, so the box has to
        be resolved to a set of blocks instead.

        Two index spaces meet here, and mixing them is the easiest
        mistake to make in this file:

          local  -- 0..nb-1, one per ACTIVE block. What start_block/
                    end_block select, what the C++ loop variable n is,
                    and the leading axis of every grid/data array.
          global -- the Global Block ID, 0..nmax-1, over every block the
                    octree could create. Names the gdump files and
                    indexes amr_info's levelx*/coordx* arrays.

        n_ord maps local -> global. So the level/coord lookups below go
        through n_ord, while the returned array is local, because that
        is what block_list/start_block/end_block are.

        A block is kept if it overlaps the box at all, not only if fully
        contained, so a block clipped by the boundary still loads whole.
        Bounds are half-open [min, max), matching checkIndexInRange's
        `>= i_min && < i_max`.
        """
        pars = self.params
        self.gridsizex1 = int(pars['nb1'] * pars['bs1'] * self.upscalex1 / self.lowres1)
        self.gridsizex2 = int(pars['nb2'] * pars['bs2'] * self.upscalex2 / self.lowres2)
        self.gridsizex3 = int(pars['nb3'] * pars['bs3'] * self.upscalex3 / self.lowres3)

        i_min, i_max = self.get_rbox_coords()
        j_min, j_max = self.get_tbox_coords()
        k_min, k_max = self.get_pbox_coords()

        nb     = pars['n_active_total']
        local  = np.arange(nb)
        glob   = self.amr_info['n_ord'][local]

        def extent(level_key, coord_key, max_level, bs):
            """Half-open [lo, hi) extent of each block on the finest grid."""
            up = (2 ** (max_level - self.amr_info[level_key][glob])).astype(np.int64)
            lo = self.amr_info[coord_key][glob].astype(np.int64) * up * bs
            return lo, lo + up * bs

        i_lo, i_hi = extent('levelx1', 'coordx1', self.amr['max_levelx1'], pars['bs1'])
        j_lo, j_hi = extent('levelx2', 'coordx2', self.amr['max_levelx2'], pars['bs2'])
        k_lo, k_hi = extent('levelx3', 'coordx3', self.amr['max_levelx3'], pars['bs3'])

        overlap = ((i_hi > i_min) & (i_lo < i_max) &
                   (j_hi > j_min) & (j_lo < j_max) &
                   (k_hi > k_min) & (k_lo < k_max))

        return local[overlap]

    ### Check settings ###
    def check_raytracing_settings(self):
        raytrace = (self.export_raytracing == 1)
        bad_lwrs = (self.params['bs1'] % self.lowres1 != 0 or
                    self.params['bs2'] % self.lowres2 != 0 or
                    self.params['bs3'] % self.lowres3 != 0)
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
        pars = self.params
        if (self.by_block):
            ndivisible_x1 = pars['bs1'] % self.lowres1 != 0
            ndivisible_x2 = pars['bs2'] % self.lowres2 != 0
            ndivisible_x3 = pars['bs3'] % self.lowres3 != 0
        else:
            ndivisible_x1 = int(pars['nb1'] * pars['bs1'] *
                self.upscalex1) % self.lowres1 != 0
            ndivisible_x2 = int(pars['nb2'] * pars['bs2'] *
                self.upscalex2) % self.lowres2 != 0
            ndivisible_x3 = int(pars['nb3'] * pars['bs3'] *
                self.upscalex3) % self.lowres3 != 0
            
        if (ndivisible_x1 or ndivisible_x2 or ndivisible_x3):
            raise ValueError("Incompatible lowres settings in read_grid.")
        
    def check_griddump_exists(self):
        for n in range(self.params['n_active_total']):
            file_exists = os.path.isfile(f"{self.dir}/gdumps/gdump{self.n_ord[n]}")
            if (file_exists == 0):
                raise FileExistsError(f"gdump file {self.n_ord[n]} doesn't exist.")

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

        self.block[:, self.amr['LEVEL1']] = gd[:, self.amr['LEVEL']]
        self.block[:, self.amr['LEVEL2']] = gd[:, self.amr['LEVEL']]
        self.block[:, self.amr['LEVEL3']] = gd[:, self.amr['LEVEL']]

        fin.close()

        # Read in parameters from parameters file.
        print("Loading parameters file")
        if (os.path.isfile(self.dir + "/dumps%d/parameters" % fnum)):
            fin = open(self.dir + "/dumps%d/parameters" % fnum, "rb")
        else:
            raise FileNotFoundError("Cannot find parameters file in %s." 
                %(self.dir + "/dumps%d/parameters" % fnum))
        
        # One read of the whole 204-byte record; leaves the file positioned
        # exactly where the field-by-field loop did.
        record = np.fromfile(fin, dtype=param_dtype, count=1)[0]
        for par in param_dtype.names:
            self.params[par] = record[par]

        del self.params['Dtl']  
        del self.params['R0']

        # There are 15 integers before the AMR information
        for n in range(15):
            trash = Tools.get_fromfile(fin, np.int32)
        del trash

        # In legacy data format, grid information is contained in the
        # parameters file as well
        grid_arr1 = np.fromfile(fin, dtype=np.int32, count=nmax, sep='')
        grid_arr2 = np.fromfile(fin, dtype=np.int32, count=nmax, sep='')
        for n in range(nmax):
            self.block[n, self.amr['REFINED']] = grid_arr1[n]
            self.block[n, self.amr['ACTIVE']]  = grid_arr2[n]

        i = 0
        for n in range(nmax):
            if self.block[n, self.amr['ACTIVE']] == 1:
                self.n_ord[i] = n
                i += 1
        
        fin.close()

        for key in phys_keys:
            self.physics['%s' %key] = 0

        # Set grid spacing
        pars = self.params
        Rin  = pars['Rin']
        Rout = pars['Rout']
        pars['dx1'] = (np.log(Rout) - np.log(Rin)) / (pars['bs1'] * pars['nb1'])
        fractheta   = -1.0 * pars['startx2']
        pars['dx2'] = 2.0 * fractheta / (pars['bs2'] * pars['nb2'])
        pars['dx3'] = 2.0 * np.pi / (pars['bs3'] * pars['nb3'])
        if (self.do_box):
            if (self.r_min is None): self.r_min = pars['Rin']
            if (self.r_max is None): self.r_max = pars['Rout']
            self.r_min     = max(self.r_min, pars['Rin'])
            self.r_max     = min(self.r_max, pars['Rout'])
            self.theta_min = max(self.theta_min, (np.pi/2.0) * (pars['startx2'] + 1.0))
            self.phi_min   = max(self.phi_min, pars['startx3'])

        self.check_raytracing_settings()

        ref = self.get_ref_dir(pars['nb2'], pars['nb3'], pars['bs3'])
        self.amr['max_levelx1'] = ref[1] * np.max(self.block[self.n_ord, self.amr['LEVEL1']])
        self.amr['max_levelx2'] = ref[2] * np.max(self.block[self.n_ord, self.amr['LEVEL2']])
        self.amr['max_levelx3'] = ref[3] * np.max(self.block[self.n_ord, self.amr['LEVEL3']])

        self.block[self.n_ord, self.amr['LEVEL1']] = ref[1]
        self.block[self.n_ord, self.amr['LEVEL2']] = ref[2]

        # .copy() is load-bearing. self.block[:, COL] is a strided VIEW, so
        # without it these six entries pin the whole (nmax, 200) block array
        # for as long as anything holds amr_info -- 333.8 MB on
        # THIN_RAD_MAD_HIRES_TRACER (nmax=437520) -- while each entry
        # reports only 1.7 MB. Copying costs 6 x 1.7 MB and is what lets a
        # caller free the block array with `H.block = None` once the read is
        # done. Same failure mode as the coords1D reduction in read_grid.
        self.amr_info = {
            'n_ord'   : self.n_ord,
            'levelx1' : self.block[:, self.amr['LEVEL1']].copy(),
            'levelx2' : self.block[:, self.amr['LEVEL2']].copy(),
            'levelx3' : self.block[:, self.amr['LEVEL3']].copy(),
            'coordx1' : self.block[:, self.amr['COORD1']].copy(),
            'coordx2' : self.block[:, self.amr['COORD2']].copy(),
            'coordx3' : self.block[:, self.amr['COORD3']].copy()
        }

        self.upscalex1 = 2 ** self.amr['max_levelx1']
        self.upscalex2 = 2 ** self.amr['max_levelx2']
        self.upscalex3 = 2 ** self.amr['max_levelx3']

        # downsample differentials with lowres and 
        # upscale to highest AMR level
        pars['dx1'] *= self.lowres1 * (1.0 / self.upscalex1)
        pars['dx2'] *= self.lowres2 * (1.0 / self.upscalex2)
        pars['dx3'] *= self.lowres3 * (1.0 / self.upscalex3)
        
        self.block_list = self.get_block_list()
        self.amr_info['block_list'] = self.block_list

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

class Write_VTK(Read):
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)
        import vtk 
        from vtk.util import numpy_support
        self.vtk = vtk
        self.numpy_support = numpy_support
        if (kwargs.get('cgs_grid', False)):
            G       = CONSTANTS['G']
            c       = CONSTANTS['c']
            Msun    = CONSTANTS['Msun']
            self.Rg = G * self.MBH * Msun / (c * c)
        else:
            self.Rg = 1. 

    def write_prims(self, fnum, **kwargs):
        direct   = kwargs.get('grid_direct', False)
        if not hasattr(self, 'grid'):
            self.read_grid(fnum, direct=direct)
        if not hasattr(self, 'data') and not kwargs.get('user_only', False):
            self.read_prims(fnum)
        r   = self.grid['r'] / self.Rg
        th  = self.grid['theta']
        phi = self.grid['phi']
        
        shape = list((r * th * phi).shape)
        shape.append(3)
        pts   = np.zeros(shape, dtype=kwargs.get('dtype'))

        pts[..., 0] = r * np.sin(th) * np.cos(phi)
        pts[..., 1] = r * np.sin(th) * np.sin(phi)
        pts[..., 2] = r * np.cos(th) * np.ones_like(phi)
        if not self.by_block:
            pts = pts[0].transpose(2, 1, 0, 3).copy()
        else:
            pts = pts.transpose(0, 3, 2, 1, 4).copy()

        # Create the VTK points object
        self.points = self.vtk.vtkPoints()

        if not kwargs.get('user_only', False):
            if 'keys' in kwargs: keys = kwargs.get('keys')
            else:                keys = self.data.keys()

            if not self.by_block:
                # Store data points in VTK points object
                pts.shape = pts.size // 3, 3
                self.points.SetData(self.numpy_support.numpy_to_vtk(pts, deep=True))
                # Store the data associated with keys into StructuredGrid
                self.get_structured_grid(keys, 0)
                if not kwargs.get('vtk_dir', False): vtk_dir = f'{self.dir}/VTK_Files'
                else:                                vtk_dir = kwargs.get('vtk_dir')
                writer = self.vtk.vtkXMLStructuredGridWriter()
                os.makedirs(vtk_dir, exist_ok=True)
                writer.SetFileName(f"{vtk_dir}/Data_{fnum:06}.vts")
                writer.SetInputData(self.sgrid)
                writer.Write() 
                print(f"Finished creating {vtk_dir}/Data_{fnum:06}.vts") 
            else:
                pass

    def get_structured_grid(self, keys, block_ID):
        self.sgrid = self.vtk.vtkStructuredGrid()
        blk        = int(block_ID)
        for key in keys:                
            if len(list(self.data[f'{key}'].shape)) == 4:
                scalar, vector = True, False
            elif len(list(self.data[f'{key}'].shape)) == 5:
                scalar, vector = False, True

            if (scalar):
                dims      = self.data[f'{key}'][blk].shape
                self.sgrid.SetDimensions(dims[0], dims[1], dims[2])
                self.sgrid.SetPoints(self.points)
                vtk_scalar = self.numpy_support.numpy_to_vtk(
                    (self.data[f'{key}'][blk].T).ravel(), deep=True, 
                    array_type=self.vtk.VTK_FLOAT)
                vtk_scalar.SetName(f"{key}")
                self.sgrid.GetPointData().AddArray(vtk_scalar)
                self.sgrid.GetPointData().SetActiveScalars(f"{key}")
            elif (vector):
                # assumes 3-vectors
                dims      = self.data[f'{key}'][0, blk].shape
                self.sgrid.SetDimensions(dims[0], dims[1], dims[2])
                self.sgrid.SetPoints(self.points)
                # Create vector array with axes
                # (blk, x1, x2, x3, component)
                vector_array = (np.moveaxis(self.data[f'{key}'], 0, -1))[blk,:,:,:,1:]
                vtk_vector   = self.numpy_support.numpy_to_vtk(
                    (vector_array.T).ravel(), deep=True, 
                    array_type=self.vtk.VTK_FLOAT)
                vtk_vector.SetNumberOfComponents(3)
                vtk_vector.SetName(f"{key}")
                self.sgrid.GetPointData().AddArray(vtk_vector)
                self.sgrid.GetPointData().SetActiveVectors(f"{key}")

class Write_VTK_Claude(Read):
    """VTK writer with a uniform-grid mode and a by-block mode.

    Refactor of Write_VTK. The mode follows the read: a Read built with
    by_block=0 holds one global uniform grid and is written as a single
    .vts; by_block=1 holds each block at its own refinement level and is
    written as a vtkMultiBlockDataSet of structured grids, one per block,
    which is what ParaView needs to show AMR levels without upscaling.

    Usage mirrors the reader:

        W = Write_VTK_Claude(data_dir, by_block=True, coords1D=True,
                             start_block=0, end_block=512)
        W.write(fnum)

    write() reads the grid and primitives itself if they are not present.
    """

    # Cell-centred values are attached as POINT data at the cell centres:
    # the grid is curvilinear, so there is no cheap cell-corner geometry to
    # attach cell data to, and ParaView interpolates point data directly.
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)
        import vtk
        from vtk.util import numpy_support
        self.vtk           = vtk
        self.numpy_support = numpy_support
        if kwargs.get('cgs_grid', False):
            G, c, Msun = CONSTANTS['G'], CONSTANTS['c'], CONSTANTS['Msun']
            self.Rg = G * self.MBH * Msun / (c * c)
        else:
            self.Rg = 1.

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def write(self, fnum, **kwargs):
        """Write one dump. Dispatches on the reader's by_block setting."""
        if not hasattr(self, 'grid'):
            self.read_grid(fnum, direct=kwargs.get('grid_direct', True))
        if not hasattr(self, 'data'):
            self.read_prims(fnum, skip_read_params=True)

        keys    = list(kwargs.get('keys', self.data.keys()))
        extra   = kwargs.get('extra', {})
        vtk_dir = kwargs.get('vtk_dir') or f'{self.dir}/VTK_Files'
        os.makedirs(vtk_dir, exist_ok=True)

        arrays = self.collect_arrays(keys, extra)
        if not arrays:
            raise ValueError(
                "nothing to write: no key resolved to a (nb, nx1, nx2, nx3) "
                "scalar or a (4, nb, nx1, nx2, nx3) vector. Keys asked for: "
                f"{keys}")

        pts = self.cartesian_points(dtype=kwargs.get('dtype', np.float32))
        if self.by_block:
            return self.write_by_block(fnum, pts, arrays, vtk_dir, **kwargs)
        return self.write_uniform(fnum, pts, arrays, vtk_dir, **kwargs)

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def cartesian_points(self, dtype=np.float32):
        """Cell-centre Cartesian coordinates, shaped (nb, nx1, nx2, nx3, 3).

        r/theta/phi broadcast to the full block shape whether or not
        coords1D reduced them, because each product below touches all three.
        """
        r   = self.grid['r'] / self.Rg
        th  = self.grid['theta']
        ph  = self.grid['phi']

        x = r * np.sin(th) * np.cos(ph)
        y = r * np.sin(th) * np.sin(ph)
        z = r * np.cos(th) * np.ones_like(ph)
        return np.stack((x, y, z), axis=-1).astype(dtype, copy=False)

    # VTK structured grids run the FIRST index fastest, so both the points
    # and every array are laid out with x1 fastest -- transpose(2, 1, 0),
    # never .T on a 4D array, which would also reverse the component axis.
    def points_for_block(self, pts, blk):
        flat = pts[blk].transpose(2, 1, 0, 3).reshape(-1, 3)
        vtk_pts = self.vtk.vtkPoints()
        vtk_pts.SetData(self.numpy_support.numpy_to_vtk(
            np.ascontiguousarray(flat), deep=True))
        return vtk_pts

    # ------------------------------------------------------------------
    # arrays
    # ------------------------------------------------------------------
    def collect_arrays(self, keys, extra=None):
        """Classify requested keys as scalar or vector, skipping the rest.

        Deallocated entries are None by the time a production script gets
        here, and the physics flags mean a key may legitimately be absent,
        so neither is an error.
        """
        out = []
        source = dict(self.data)
        if extra:
            source.update(extra)
        for key in keys:
            arr = source.get(key)
            if arr is None:
                continue
            if arr.ndim == 4:
                out.append((key, 'scalar', arr))
            elif arr.ndim == 5 and arr.shape[0] == 4:
                out.append((key, 'vector', arr))
        return out

    def add_arrays(self, sgrid, arrays, blk):
        """Attach one block's arrays to a structured grid as point data."""
        first_scalar, first_vector = None, None
        for key, kind, arr in arrays:
            if kind == 'scalar':
                flat = arr[blk].transpose(2, 1, 0).ravel()
            else:
                # Drop the time component and put the spatial 3-vector last,
                # then order component-FASTEST, which is what VTK expects.
                # (arr.T would give component-slowest and silently scramble
                # the vector field.)
                vec  = np.moveaxis(arr[1:, blk], 0, -1)
                flat = vec.transpose(2, 1, 0, 3).reshape(-1, 3)

            vtk_arr = self.numpy_support.numpy_to_vtk(
                np.ascontiguousarray(flat), deep=True,
                array_type=self.vtk.VTK_FLOAT)
            vtk_arr.SetName(str(key))
            sgrid.GetPointData().AddArray(vtk_arr)

            if kind == 'scalar' and first_scalar is None:
                first_scalar = str(key)
            if kind == 'vector' and first_vector is None:
                first_vector = str(key)

        if first_scalar is not None:
            sgrid.GetPointData().SetActiveScalars(first_scalar)
        if first_vector is not None:
            sgrid.GetPointData().SetActiveVectors(first_vector)

    def structured_grid(self, pts, arrays, blk):
        """One vtkStructuredGrid for one block (or for the uniform grid)."""
        nx1, nx2, nx3 = pts.shape[1], pts.shape[2], pts.shape[3]
        sgrid = self.vtk.vtkStructuredGrid()
        # Dimensions are set ONCE, not per key: they describe the grid, not
        # the array, and re-setting them inside the key loop is wasted work.
        sgrid.SetDimensions(nx1, nx2, nx3)
        sgrid.SetPoints(self.points_for_block(pts, blk))
        self.add_arrays(sgrid, arrays, blk)
        return sgrid

    # ------------------------------------------------------------------
    # writers
    # ------------------------------------------------------------------
    def write_uniform(self, fnum, pts, arrays, vtk_dir, **kwargs):
        """One .vts holding the whole upscaled domain."""
        sgrid = self.structured_grid(pts, arrays, 0)
        path  = f'{vtk_dir}/Data_{fnum:06}.vts'
        writer = self.vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(path)
        writer.SetInputData(sgrid)
        writer.Write()
        print(f'Finished creating {path}', flush=True)
        return path

    def write_by_block(self, fnum, pts, arrays, vtk_dir, **kwargs):
        """One structured grid per block, collected into a .vtm.

        The blocks written are the ones the reader currently holds, i.e.
        amr_info['block_list']. A batched read therefore produces one .vtm
        per batch, named by its Local Block ID range, and the name records
        that so batches cannot overwrite one another.

        per_block_files=True writes a separate .vts per block instead, which
        avoids holding every grid in memory at once.
        """
        blocks = np.asarray(self.amr_info['block_list'])
        glob   = self.n_ord[blocks]
        tag    = f'{fnum:06}_blk{int(blocks[0])}-{int(blocks[-1]) + 1}'

        if kwargs.get('per_block_files', False):
            sub = f'{vtk_dir}/Data_{fnum:06}'
            os.makedirs(sub, exist_ok=True)
            paths = []
            for i, local in enumerate(blocks):
                sgrid = self.structured_grid(pts, arrays, i)
                self.tag_block(sgrid, int(local), int(glob[i]))
                path   = f'{sub}/block_{int(local):06}.vts'
                writer = self.vtk.vtkXMLStructuredGridWriter()
                writer.SetFileName(path)
                writer.SetInputData(sgrid)
                writer.Write()
                paths.append(path)
            print(f'Finished creating {len(paths)} .vts files in {sub}',
                  flush=True)
            return paths

        mblock = self.vtk.vtkMultiBlockDataSet()
        mblock.SetNumberOfBlocks(len(blocks))
        for i, local in enumerate(blocks):
            sgrid = self.structured_grid(pts, arrays, i)
            self.tag_block(sgrid, int(local), int(glob[i]))
            mblock.SetBlock(i, sgrid)
            mblock.GetMetaData(i).Set(self.vtk.vtkCompositeDataSet.NAME(),
                                      f'block_{int(local)}')

        path   = f'{vtk_dir}/Data_{tag}.vtm'
        writer = self.vtk.vtkXMLMultiBlockDataWriter()
        writer.SetFileName(path)
        writer.SetInputData(mblock)
        writer.Write()
        print(f'Finished creating {path} ({len(blocks)} blocks)', flush=True)
        return path

    def tag_block(self, sgrid, local, glob):
        """Attach block identity and refinement level as field data.

        Lets ParaView colour or threshold by AMR level, and keeps the Local
        and Global Block IDs together in the output -- they are easy to
        confuse and the mapping is not recoverable from the geometry.
        """
        meta = {
            'local_block_id' : local,
            'global_block_id': glob,
            'levelx1'        : int(self.amr_info['levelx1'][glob]),
            'levelx2'        : int(self.amr_info['levelx2'][glob]),
            'levelx3'        : int(self.amr_info['levelx3'][glob]),
        }
        for name, value in meta.items():
            arr = self.numpy_support.numpy_to_vtk(
                np.array([value], dtype=np.int32), deep=True,
                array_type=self.vtk.VTK_INT)
            arr.SetName(name)
            sgrid.GetFieldData().AddArray(arr)

class Write_RayTracing(Read):
    """
    Write H-AMR dumps in the formats expected by ray-tracing post-processors.

    Requires the full 3D grid, so initialize the class with coords1D=False;
    GRMonty needs r, theta and phi at every zone. 
    """

    def dump_grmonty(self, fnum, **kwargs):
        """Write dump 'fnum' in GRMonty's grid format.

        Options:
        - 'dtype'    (default: 'hdf5'): 'hdf5' or 'ascii'.
        - 'cut2D'    (default: False): write a single phi slice taken at
          the midplane index nx3 // 2 instead of the full 3D volume.
        - 'radcut'   (default: 100.): truncate the grid at this radius in
          Rg, at the first radial index exceeding radcut.
        - 'dscale'   (default: params['density_scale']): density scale in
          cgs, written to the header for GRMonty's unit conversion.
        - 'model'    (default: ''): tag inserted into the hdf5 file name.
        - 'save_dir' (default: ``{self.dir}/GRMonty_Files``).
        - 'hslope'   (default: 1.): theta-coordinate parameter. 1 is the
          correct value for a uniform grid.
        - 'R0'       (default: 0.): radial-coordinate offset.

        GRMonty uses the lab-frame three-velocity scaled by the Lorentz
        factor (v3 below), the primitive magnetic field B, and u_t / u^t, so
        the four-velocity is converted here rather than by the caller.
        """
        dtype    = kwargs.get('dtype', 'hdf5')
        cut2D    = kwargs.get('cut2D', False)
        radcut   = kwargs.get('radcut', 100.)
        dscale   = kwargs.get('dscale', None)
        model    = kwargs.get('model', '')
        hslope   = kwargs.get('hslope', 1.)
        R0       = kwargs.get('R0', 0.)
        # Not sure about this. - CB
        #save_dir = kwargs.get('save_dir', f'{self.dir}/GRMonty_Files')

        if dtype not in ('ascii', 'hdf5'):
            raise ValueError("dtype should be one of 'ascii', 'hdf5'")

        # Read grid and primitives on demand (same pattern as Write_VTK)
        if not hasattr(self, 'grid'):
            self.read_grid(fnum, direct=kwargs.get('grid_direct', False))
        if not hasattr(self, 'data'):
            self.read_prims(fnum)

        if (self.coords1D):
            raise ValueError("dump_grmonty needs the full 3D grid; "
                             "initialize the class with coords1D=False.")

        grid, data, pars = self.grid, self.data, self.params

        if (dscale is None):
            dscale = pars['density_scale']
        if (dscale == 0):
            raise ValueError("check the density scale value")

        r, th, phi        = grid['r'], grid['theta'], grid['phi']
        nb, nx1, nx2, nx3 = r.shape

        # Truncate the radial range at radcut. r varies only with x1, so any
        # (theta, phi) column gives the same cut index.
        if (r[0, nx1 - 1, nx2 // 2, 0] > radcut):
            N1 = np.where(r[0, :, nx2 // 2, 0] > radcut)[0][0]
            print('Cut the radius up to %f and index of %d'
                  % (r[0, N1, nx2 // 2, 0], N1), flush=True)
        else:
            N1 = nx1
            print('Entire range of radius', flush=True)

        # Lapse, shift, and the lab-frame three-velocity GRMonty expects:
        # alpha = 1/sqrt(-g^tt), beta^i = g^ti alpha^2, and
        # v^i = gamma * (beta^i + u^i/u^t)/alpha with gamma = alpha u^t.
        uu    = data['uu']
        alpha = 1. / (-grid['gcon'][0, 0]) ** 0.5
        beta  = Tools.zeroNd([4, nb, nx1, nx2, 1], 1)
        v3    = Tools.zeroNd([4, nb, nx1, nx2, nx3], 1)

        beta[1:4] = grid['gcon'][0, 1:4] * alpha * alpha
        v3[1:4]   = (beta[1:4] + uu[1:4] / uu[0]) / alpha
        v3[1:4]  *= uu[0] * alpha
        del beta

        # u_mu = g_{mu nu} u^nu; only u_t is written out
        gcov = grid['gcov']
        ud_t = (gcov[0, 0] * uu[0] + gcov[0, 1] * uu[1]
                + gcov[0, 2] * uu[2] + gcov[0, 3] * uu[3])

        # Midplane index, used for the 2D cut and the header coordinates
        iz = nx3 // 2

        meta = {
            't'      : pars['t'],
            'N1'     : N1,
            'N2'     : nx2,
            'N3'     : nx3,
            'dx1'    : pars['dx1'],
            'dx2'    : pars['dx2'],
            'dx3'    : pars['dx3'],
            'a'      : pars['a'],
            'gam'    : pars['gam'],
            'startx1': grid['x1'][0, 0, 0, iz],
            'startx2': grid['x2'][0, 0, 0, iz],
            'startx3': grid['x3'][0, 0, 0, iz],
            'Rin'    : r[0, 0, 0, iz],
            'Rout'   : r[0, N1 - 1, 0, iz],
            'hslope' : hslope,
            'R0'     : R0,
            'dscale' : dscale,
        }

        # Zone-centred fields. gdet is handled separately: it is stored with
        # the phi axis collapsed, so it cannot be flattened like the rest.
        fields = {
            'x1'   : grid['x1'],
            'x2'   : grid['x2'],
            'x3'   : grid['x3'],
            'r'    : r,
            'h'    : th,
            'ph'   : phi,
            'RHO'  : data['rho'],
            'UU'   : data['u_gas'],
            'TE'   : data['Te'],
            'TI'   : data['Ti'],
            'U1'   : v3[1],
            'U2'   : v3[2],
            'U3'   : v3[3],
            'B1'   : data['B'][1],
            'B2'   : data['B'][2],
            'B3'   : data['B'][3],
            'Ucov0': ud_t,
            'Ucon0': uu[0],
        }

        os.makedirs(save_dir, exist_ok=True)
        tag = '2Dcut_' if cut2D else ''
        if (dtype == 'ascii'):
            fname = f'{save_dir}/grmonty_{tag}{fnum}.dat'
            self.write_grmonty_ascii(fname, fields, grid['gdet'], meta,
                                     N1, iz, cut2D)
        else:
            name  = f'{tag}{fnum}' if cut2D else f'{model}_{fnum}'
            fname = f'{save_dir}/grmonty_{name}.hdf5'
            self.write_grmonty_hdf5(fname, fields, grid['gdet'], meta,
                                    N1, iz, cut2D)
        print(f'Finished writing {fname}', flush=True)

    def flatten_grmonty(self, quant, axisym, N1, iz, cut2D):
        """
        Flatten a zone-centred array into the 1D ordering GRMonty reads.

        Ordering is C-order over (x1, x2) for a 2D cut and over
        (x1, x2, x3) for the full volume, matching the nested loops in
        GRMonty's reader. 'axisym' marks quantities stored with the phi
        axis collapsed to one element (gdet, and the metric it comes from),
        which are always taken at phi index 0.
        """
        if (axisym):
            return quant[0, :N1, :, 0].ravel(order='C')
        if (cut2D):
            return quant[0, :N1, :, iz].ravel(order='C')
        return quant[0, :N1, :, :].ravel(order='C')

    def write_grmonty_ascii(self, fname, fields, gdet, meta, N1, iz, cut2D):
        """
        Write the GRMonty ASCII dump: one header line, then one line per
        zone. Columns are assembled and written with np.savetxt instead of a
        Python loop over zones, which is orders of magnitude faster on large
        grids and produces the same bytes.
        """
        if (cut2D):
            keys   = ['x1', 'x2', 'r', 'h', 'RHO', 'UU', 'TE', 'TI',
                      'U1', 'U2', 'U3', 'B1', 'B2', 'B3']
            header = ('%.17g %d %d %.17g %.17g %.17g %.17g %.17g %.17g '
                      '%.17g %.17g %.17g %.17g %.17g \n'
                      % (meta['t'], N1, meta['N2'],
                         meta['startx1'], meta['startx2'],
                         meta['dx1'], meta['dx2'], meta['a'], meta['gam'],
                         meta['Rin'], meta['Rout'], meta['hslope'],
                         meta['R0'], meta['dscale']))
        else:
            keys   = ['x1', 'x2', 'x3', 'r', 'h', 'ph', 'RHO', 'UU',
                      'TE', 'TI', 'U1', 'U2', 'U3', 'B1', 'B2', 'B3']
            header = ('%.17g %d %d %d %.17g %.17g %.17g %.17g %.17g %.17g '
                      '%.17g %.17g %.17g %.17g %.17g %.17g %.17g\n'
                      % (meta['t'], N1, meta['N2'], meta['N3'],
                         meta['startx1'], meta['startx2'], meta['startx3'],
                         meta['dx1'], meta['dx2'], meta['dx3'],
                         meta['a'], meta['gam'], meta['Rin'], meta['Rout'],
                         meta['hslope'], meta['R0'], meta['dscale']))

        columns = [self.flatten_grmonty(fields[key], False, N1, iz, cut2D)
                   for key in keys]

        # gdet carries no phi index, so repeat it across phi for the full
        # volume; every zone line then holds the value at its own (x1, x2).
        gdet_1d = self.flatten_grmonty(gdet, True, N1, iz, cut2D)
        if not (cut2D):
            gdet_1d = np.repeat(gdet_1d, meta['N3'])
        columns.append(gdet_1d)

        for key in ('Ucov0', 'Ucon0'):
            columns.append(self.flatten_grmonty(fields[key], False,
                                                N1, iz, cut2D))

        with open(fname, 'w') as fout:
            fout.write(header)
            np.savetxt(fout, np.column_stack(columns), fmt='%.9g',
                       delimiter=' ', newline=' \n')

    def write_grmonty_hdf5(self, fname, fields, gdet, meta, N1, iz, cut2D):
        """
        Write the GRMonty HDF5 dump: grid metadata as file attributes,
        each field as a flat 1D dataset.
        """
        import h5py

        keys = list(fields.keys())
        if (cut2D):
            # A 2D cut has no phi axis, so x3 and ph carry no information
            for key in ('x3', 'ph'):
                keys.remove(key)

        with h5py.File(fname, 'w') as fout:
            fout.attrs['t']  = np.float64(meta['t'])
            fout.attrs['N1'] = np.int64(N1)
            fout.attrs['N2'] = np.int64(meta['N2'])
            if (cut2D):
                fout.attrs['startx'] = np.asarray(
                    [meta['startx1'], meta['startx2']], dtype=np.float64)
                fout.attrs['dx'] = np.asarray(
                    [meta['dx1'], meta['dx2']], dtype=np.float64)
            else:
                fout.attrs['N3'] = np.int64(meta['N3'])
                # Full-volume startx is shifted to the zone corner; the 2D
                # cut is left at the zone centre (preserved as written).
                fout.attrs['startx'] = np.asarray(
                    [meta['startx1'] - 0.5 * meta['dx1'],
                     meta['startx2'] - 0.5 * meta['dx2'],
                     meta['startx3'] - 0.5 * meta['dx3']], dtype=np.float64)
                fout.attrs['dx'] = np.asarray(
                    [meta['dx1'], meta['dx2'], meta['dx3']], dtype=np.float64)
            for key in ('a', 'gam', 'Rin', 'Rout', 'hslope', 'R0', 'dscale'):
                fout.attrs[key] = np.float64(meta[key])

            for key in keys:
                fout.create_dataset(
                    key, data=self.flatten_grmonty(fields[key], False,
                                                   N1, iz, cut2D))
            # gdet is axisymmetric and is stored at its native (x1, x2) size
            # rather than repeated across phi.
            fout.create_dataset(
                'gdet', data=self.flatten_grmonty(gdet, True, N1, iz, cut2D))

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
