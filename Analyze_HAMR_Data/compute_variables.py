import sys
import os
import glob
import math
import numpy as np
from pathlib import Path
this_file   = Path(__file__).resolve()
asys_path   = str(this_file.parent)
vis_path    = str(this_file.parent.parent)
user_path   = str(this_file.parent.parent.parent)
shared_path = str(this_file.parent.parent.parent.parent)  # Modify shared path
read_path   = os.path.join(vis_path, "Read_HAMR_Data")
run_path    = os.path.join(user_path, "RUN")
sys.path.insert(0, read_path)
sys.path.insert(0, asys_path)
sys.path.insert(0, run_path)
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata
import analysis as ana
import coordinates as coords
import radiation as rad
import theory as theory
import hamr_read as hr

##############################################################################

# Directory handling
def get_data_dir(sim, reduced, shared):
    """
    Determine directory where H-AMR data is stored
    """
    base_dir = Path(shared_path) if shared else Path(user_path) / "DATA"
    data_dir = base_dir / sim
    if reduced:
        data_dir /= "reduced"
    return str(data_dir)

def get_save_dir(sim, file_type, diagnostic, **kwargs):
    """
    Determine directory where analysis output should be saved
    """
    subtype  = kwargs.get('subtype', False)
    reduced  = kwargs.get('reduced', False)
    res_str  = 'Reduced_Resolution' if reduced else 'Full_Resolution'
    save_dir = Path(user_path) / file_type / sim / res_str
    if subtype:
        save_dir /= subtype
    save_dir /= diagnostic 
    return str(save_dir)

def get_file_list_local(file_list, size, rank, data_dir, **kwargs):
    """
    Calculate list of files to analyze on the local processor (rank).

    Parameters
    ----------
    file_list : list of int
        List of all file numbers to analyze.
    size : int
        How many processors to use for analysis.
    rank : int
        Processor rank.
    data_dir : str
        Directory where data is stored.
    **kwargs : dict
        Optional configurations (e.g., 'stride').
        Use stride keyword to only analyze every nth file, where n=stride

    Returns
    -------
    list of int
        A list of integers containing files found on disk assigned to rank.
    """
    stride           = int(kwargs.get('stride', 1))
    nout             = len(file_list)
    nfiles_nproc     = nout // size
    nfiles_remaining = nout % size
    files_per_proc   = ([nfiles_nproc] * (size - nfiles_remaining) + 
                        [nfiles_nproc + 1] * nfiles_remaining)
    start_idx        = sum(files_per_proc[:rank])
    num_local_files  = files_per_proc[rank]
    file_list_local  = file_list[start_idx:(start_idx + num_local_files):stride]

    if kwargs.get('file_type', None) is None:
        file_list_local = check_dumps_exist(data_dir, file_list_local)
    else:
        file_list_local = check_files_exist(data_dir, file_list_local, **kwargs)
    
    return file_list_local

def check_dumps_exist(path_to_dumps, file_list):
    """
    Check if dumps contained in file_list exist.
    Some corrupt/old files are stored with _ as suffix, so this function
    checks if files with _ suffix exist if the dump number is not found.

    Parameters
    ----------
    path_to_dumps : str
        Path to H-AMR data dumps
    file_list : list of int
        List of all file numbers to analyze.

    Returns
    -------
    list of int
        A list of integers containing only the files found on disk.
    """
    valid_files = []
    base_path = Path(path_to_dumps)
    
    for file_num in file_list:
        dump_dir = base_path / f"dumps{file_num}"
        
        if dump_dir.is_dir():
            valid_files.append(file_num)
        else:
            print(f"Dumps {file_num} is missing.")
            # Check if directory exists with _ convention
            backup_dir = base_path / f"dumps{file_num}_"
            if backup_dir.is_dir():
                print(f"The file can be found as {backup_dir}", flush=True)
    
    return valid_files

def check_files_exist(path_to_averages, file_list, **kwargs):
    """
    Check if files of type file_type.npz in file_list exist.
    If images=True, checks for files of type subtype/title/title.png exist.

    Parameters
    ----------
    path_to_averages : str
        Path to Averages (or if images=True, Images) directory
    file_list : list of int
        List of all file numbers to analyze.
    **kwargs : dict
        Optional configurations (e.g., 'reduced').

    Returns
    -------
    list of int
        A list of integers containing only the files found on disk.
    """
    if not all(isinstance(f, int) for f in file_list):
        raise TypeError("file_list must contain only integers!")
    file_type = kwargs.get('file_type', None)
    if file_type is None and not kwargs.get('images', False):
        raise ValueError("Must supply file_type to use check_files_exist().")
    reduced   = kwargs.get('reduced',   False)
    res_str   = 'Reduced_Resolution' if reduced else 'Full_Resolution'
    # Build list of file numbers
    nums      = []
    if kwargs.get('images', False):
        subtype   = kwargs.get('subtype', None)
        title     = kwargs.get('title',   None)
        file_dir  = Path(path_to_averages) / res_str / subtype / title
        data_list = glob.glob(f'*.png', root_dir=file_dir)
        prefix    = f'{title}_'
    else:
        file_dir  = Path(path_to_averages) / res_str / file_type
        data_list = glob.glob(f'{file_type}*', root_dir=file_dir)
        prefix    = f'{file_type}_'
    
    for name in data_list:
        # Get the filename without the prefix and '.npz' or '.png'
        nums.append(int((Path(name).stem).removeprefix(prefix)))
    nums.sort()
    file_list = [f for f in file_list if f in set(nums)]
    return file_list

def check_images_exist(path_to_images, file_list, title, subtype, **kwargs):
    """
    Check if images of type subtype/title/title.png exist.

    Parameters
    ----------
    path_to_averages : str 
        Path to Images directory
    file_list : list of int
        List of all file numbers to analyze.
    title : str
        Title of png images.
    subtype : str
        Subtype of images (e.g. Poloidal, Midplane, QuickView).
    **kwargs : dict
        Optional configurations (e.g., 'reduced').

    Returns
    -------
    list of int
        A list of integers containing only the images found on disk.
    """
    image_args = {
        'images'  : True,
        'title'   : title,
        'subtype' : subtype
    }
    return check_files_exist(path_to_images, file_list, **image_args, **kwargs)

# LaTeX formatting for figures 
def get_time_coeff_expon(time):
    """
    Split a simulation time into (coefficient, exponent) for 10^exponent
    notation, handling the rounding edge case (e.g., 9.999999 -> 10.0).
    """
    expon = math.floor(math.log10(time))
    coeff = time / (10 ** expon)

    # Handle rounding up edge cases (e.g., 9.999999 -> 10.0)
    if round(coeff, 5) == 10.0:
        coeff = 1.0
        expon += 1
    return coeff, expon

def get_nice_time(time, **kwargs):
    """
    Format simulation time into a LaTeX string.

    Parameters
    ----------
    **kwargs : dict
        Optional configurations (e.g. prec (default = '%.5f')).
    """
    if time == 0:
        return r"$0.00000$"

    coeff, expon = get_time_coeff_expon(time)
    prec = kwargs.get('prec', 5)
    return rf"${coeff:.{prec}f} \times 10^{{{expon}}}$"

def get_nice_Mdot(Mdot):
    """
    Format the accretion rate (Mdot) into a clean LaTeX string.
    """
    if Mdot <= 0:
        return r'$0.0$'
        
    # 1. Standard decimal ranges
    if 10 < Mdot < 1000:
        return rf"${Mdot:.1f}$"
    elif 0.01 <= Mdot <= 10:
        return rf"${Mdot:.2f}$"
        
    # 2. Scientific notation ranges (Mdot >= 1000 or Mdot < 0.01)
    # Calculate exponent and coefficient
    expon = math.floor(math.log10(Mdot))
    coeff = Mdot / (10 ** expon)
    
    # Handle rounding edge-cases natively (e.g., if coeff rounds up to 10.0)
    if round(coeff, 1) == 10.0:
        coeff = 1.0
        expon += 1

    return rf"${coeff:.1f} \times 10^{{{expon}}}$"

def check_extend(q, lims):
    """
    Determine the Matplotlib colorbar 'extend' setting based on data limits.
    """
    qmin, qmax = np.amin(q), np.amax(q)
    
    lower = qmin < lims[0]
    upper = qmax > lims[1]
    
    if lower and upper:
        return 'both'
    if lower:
        return 'min'
    if upper:
        return 'max'    
    return 'neither'

def grid(ax, **kwargs):
    """
    Draw gridlines on figures.
    """
    ax.grid(which='major', color=kwargs.get('color', 'white'), linestyle='-',
            linewidth=0.5, alpha=kwargs.get('alpha', 0.3))
    ax.grid(which='minor', color=kwargs.get('color', 'white'), linestyle='-',
            linewidth=0.5, alpha=kwargs.get('alpha', 0.3))

def get_sym_log_lims(pos_data, neg_data):
    """
    Symmetric log-scale y-limits spanning both signs of a key's data.
    """
    log_pos = np.log10(np.amax(pos_data)) if len(pos_data) > 0 else -6
    log_neg = np.log10(np.abs(np.amin(neg_data))) if len(neg_data) > 0 else -6
    sym_log = np.floor(max(log_pos, log_neg)) + 1
    return [-10 ** sym_log, 10 ** sym_log]

def get_limits(key, user_limits, quants, plot_info):
    """
    Determine y-axis limits for a plotted key, unless the user supplied
    explicit limits. `quants` is the list of arrays already selected for
    `key` (as accumulated by the caller over curves/files/keys).
    """
    if user_limits[key] is not None:
        return user_limits[key]

    all_data = np.concatenate(quants)
    if not plot_info[key]['log']:
        return [np.amin(all_data), np.amax(all_data)]

    pos_data = all_data[all_data > 0]
    if plot_info[key].get('linthresh'):
        neg_data = all_data[all_data < 0]
        return get_sym_log_lims(pos_data, neg_data)

    if pos_data.size == 0:
        raise ValueError(f"No positive data for log-scaled key '{key}'; "
                          "check the input or set an explicit user limit.")

    log_min = np.floor(np.amin(np.log10(pos_data)))
    log_max = np.floor(np.amax(np.log10(pos_data)))
    return [10 ** log_min, 10 ** (log_max + 1)]

# Specific to X-Z (poloidal) plots
def get_contours(plt_args, PS_r, PS_l, contour_colors, sigma_contour, photospheres, **kwargs):
    """
    Add dictionaries needed to plot contours to plt_args dictionary.
    
    Parameters
    ----------
    plt_args : dict
        Dictionary of plot arguments.
    PS_r : class object
        Object of Phi_Slice class (right-hand side of plot).
    PS_l : class object
        Object of Phi_Slice class (left-hand side of plot).
    contour_colors : list
        List of contour colors.
    sigma_contour :  int
        Integer value of sigma contour.
    photospheres : dict
        Dictionary containing information needed for photosphere contours.
    **kwargs : dict
        Optional configurations.

    Returns
    -------
    dict
        Dictionary of plot arguments needed for drawing 
        sigma and/or photosphere contours.
    """
    quantities, levels, colors, linewidths, linestyles = (dict.fromkeys(['l', 'r']) for i in range(5))
    quantities['r'], levels['r'], colors['r'], linewidths['r'], linestyles['r'] = ([] for i in range(5))
    quantities['l'], levels['l'], colors['l'], linewidths['l'], linestyles['l'] = ([] for i in range(5))
    
    if (sigma_contour != 0):
        quantities['r'].append(PS_r.calc_sigma(0))
        quantities['l'].append(PS_l.calc_sigma(0))
        for key in ['r', 'l']:
            levels[    f'{key}'].append(sigma_contour)
            colors[    f'{key}'].append(contour_colors[0])
            linestyles[f'{key}'].append(['-'])
    if (photospheres['scattering']):
        opacities = dict.fromkeys(['electron_scattering'])
        quantities['r'].append(PS_r.calc_tau(opacities, 0, integral=photospheres['integral']))
        quantities['l'].append(PS_l.calc_tau(opacities, 0, integral=photospheres['integral']))
        for key in ['r', 'l']:
            levels[    f'{key}'].append([1.])
            colors[    f'{key}'].append([contour_colors[1]])
            linestyles[f'{key}'].append(['-'])
    if (photospheres['effective']):
        opacities = dict.fromkeys(['electron_scattering', 'bound_free', 
            'CHIANTI', 'free_free', 'hydrogen', 'molecular', 'synchrotron']) 
        quantities['r'].append(PS_r.calc_tau(opacities, 0, integral=photospheres['integral']))
        quantities['l'].append(PS_l.calc_tau(opacities, 0, integral=photospheres['integral']))
        for key in ['r', 'l']:
            levels[    f'{key}'].append(1.)
            colors[    f'{key}'].append(contour_colors[1])
            linestyles[f'{key}'].append(['--'])

    for key in ['r', 'l']: linewidths[f'{key}'] += len(levels[f'{key}']) * [2.0]

    plt_args['contours_right'] = {
        'quantities' : quantities['r'], 
        'levels'     : levels['r'], 
        'colors'     : colors['r'], 
        'linewidths' : linewidths['r'],
        'linestyles' : linestyles['r']
    }
    plt_args['contours_left'] = {
        'quantities' : quantities['l'], 
        'levels'     : levels['l'], 
        'colors'     : colors['l'], 
        'linewidths' : linewidths['l'],
        'linestyles' : linestyles['l']
    }
    return plt_args

# Specific to X-Z (poloidal) plots
def get_Aphi_contours(plt_args, PS_r, PS_l, contour_colors, **kwargs):
    """
    Add dictionaries needed to plot Aphi contours to plt_args dictionary.
    
    Parameters
    ----------
    plt_args : dict
        Dictionary of plot arguments.
    PS_r : class object
        Object of Phi_Slice class (right-hand side of plot).
    PS_l : class object
        Object of Phi_Slice class (left-hand side of plot).
    contour_colors : list
        List of contour colors.
    **kwargs : dict
        Optional configurations.

    Returns
    -------
    dict
        Dictionary of plot arguments needed for drawing 
        Aphi contours.
    """
    quantities, levels, colors, linewidths, linestyles = (dict.fromkeys(['l', 'r']) for i in range(5))
    quantities['r'], levels['r'], colors['r'], linewidths['r'], linestyles['r'] = ([] for i in range(5))
    quantities['l'], levels['l'], colors['l'], linewidths['l'], linestyles['l'] = ([] for i in range(5))
    
    Aphi = PS_r.calc_Aphi()
    normalize = kwargs.get('normalize', False)
    if normalize:
        # Normalize relative to flux on the horizon separately for each
        # hemisphere. Magenta curve shows separatrix between horizon-threading
        # and disk-threading field lines.
        ds1 = 2 ** PS_r.H.amr['max_levelx1']
        r   = PS_r.r[0, ::ds1, 0, 0][:Aphi.shape[0]]
        a   = PS_r.H.params['a']
        iH  = int(np.argmin(np.abs(r - (1. + np.sqrt(1. - a * a)))))
        nth = Aphi.shape[1]
        Aphi[:, :nth//2] /= Aphi[iH, nth//2 - 1]
        Aphi[:, nth//2:] /= Aphi[iH, nth//2]
    for key in ['r', 'l']:
        quantities[f'{key}'].append(Aphi)
        if normalize:
            levels[    f'{key}'].append(np.linspace(0.1, 0.9, 9))
            colors[    f'{key}'].append(contour_colors[0])
            linestyles[f'{key}'].append(9 * ['-'])
        else:
            aphi_levels = np.delete(np.linspace(-100, 100, 21), 15)
            levels[    f'{key}'].append(aphi_levels)
            colors[    f'{key}'].append(contour_colors[0])
            # Dashed for negative, solid for positive
            linestyles[f'{key}'].append(10 * ['--'] + 10 * ['-'])
        # Use special color for Phi = 50 (Phi = Phi_BH if normalized)
        quantities[f'{key}'].append(Aphi)
        levels[    f'{key}'].append(1. if normalize else 50.)
        colors[    f'{key}'].append(contour_colors[2])
        linestyles[f'{key}'].append(['-'])

    for key in ['r', 'l']: linewidths[f'{key}'] += len(levels[f'{key}']) * [2.0]

    plt_args['contours_right'] = {
        'quantities' : quantities['r'], 
        'levels'     : levels['r'], 
        'colors'     : colors['r'], 
        'linewidths' : linewidths['r'],
        'linestyles' : linestyles['r']
    }
    plt_args['contours_left'] = {
        'quantities' : quantities['l'], 
        'levels'     : levels['l'], 
        'colors'     : colors['l'], 
        'linewidths' : linewidths['l'],
        'linestyles' : linestyles['l']
    }
    return plt_args

def _init_averages(data_dir, file_num, keys, 
    save_dir, nslices, restart, nzro, **kwargs):
    """
    Internal function for creating profiles or slices dictionary
    
    Parameters
    ----------
    data_dir : str
        Directory where H-AMR data is stored.
    file_num : int
        Dump number.
    keys : list of str
        Keys used in profiles or slices dictionary.
    save_dir : str or path
        Directory where profiles or slices dictionary should be stored.
    nslices : int
        Number of slices to divide data into (must evenly divide nphi)
    restart : bool
        Whether or not extraction should be restarted using data in
        e.g. save_dir/Profiles_{file_num}.npz.
    nzro : int
        Number of zeros to pad file_num with before .npz in output file.
    **kwargs : dict
        Optional configurations.

    Returns
    -------
    dict
        Dictionary of profiles (1D) or slice (2D) arrays initialized to 0.
    """
    file_type = kwargs.get('file_type', None)
    if file_type is None:
        raise ValueError("Must supply file_type to use _init_averages.")
    averages  = {}
    
    # First, determine the limits of the slices.
    H = hr.Read(data_dir, **kwargs)
    H.read_params(file_num)
    averages['t'] = H.params['t']
    averages['a'] = H.params['a']

    two_temp  = kwargs.get('two_temp')
    radiation = kwargs.get('radiation')
    if two_temp is not None:
        if (two_temp and not H.physics['TWO_T']):
            raise ValueError("two_temp is True and was not used in this simulation.", flush=True)
    if radiation is not None:
        if (radiation and not H.physics['RAD_M1']):
            raise ValueError("radiation is True and was not used in this simulation.", flush=True)

    # Number of ACTIVE blocks in this dump. Carried in the averages dict
    # rather than returned separately, so the signature is unchanged and it
    # lands in the restart .npz automatically. run_parallel_in_time needs it
    # to plan block batches, and H already exists here -- deriving it later
    # would mean a second Read() and another nmax-sized block array.
    averages['nb'] = int(H.params['n_active_total'])

    nr   = int(H.params['nb1'] * H.params['bs1'] * H.upscalex1 / H.lowres1)
    nth  = int(H.params['nb2'] * H.params['bs2'] * H.upscalex2 / H.lowres2)
    nphi = int(H.params['nb3'] * H.params['bs3'] * H.upscalex3 / H.lowres3)

    if (nphi % nslices != 0):
        raise ValueError(f'nslices must evenly divide nphi = {nphi}')
    iphi_stp = int(nphi / nslices)
    
    if 'r_max' in kwargs:
        H.gridsizex1 = int(
            H.params['nb1'] * H.params['bs1'] * H.upscalex1 / H.lowres1)
        i_min, nr = H.get_rbox_coords()

    # Logic to allow restarts if extraction is interrupted.
    if not restart:
        first_call = True
    else:
        fnum_str = str(file_num).zfill(nzro)
        averages = dict(np.load(f'{save_dir}/{file_type}_{fnum_str}.npz'))
        first_call = False

    averages['file_type'] = file_type

    # Initialize profiles dictionary
    if first_call:
        if file_type == 'Profiles':
            for key in keys:
                averages[f'{key}'] = np.zeros(nr, dtype=np.float64)
        else:
            for key in keys:
                averages[f'{key}'] = np.zeros([nr, nth])

        first_call = False
        averages['slice_num_restart'] = 0

    return averages, nphi, iphi_stp

def init_profiles(data_dir, file_num, keys, 
    save_dir, nslices, restart, nzro, **kwargs):
    kwargs['file_type'] = 'Profiles'
    return _init_averages(data_dir, file_num, keys, save_dir, nslices, 
        restart, nzro, **kwargs)

def init_slices(data_dir, file_num, keys, 
    save_dir, nslices, restart, nzro, **kwargs):
    kwargs.setdefault('file_type', 'Slices')
    return _init_averages(data_dir, file_num, keys, save_dir, nslices,
        restart, nzro, **kwargs)

def iter_phi_batches(nphi, iphi_stp, start=0):
    """Yield (batch_num, selection) for each phi slice within one file."""
    for i, k in enumerate(range(int(start), int(nphi), int(iphi_stp))):
        yield i, {'k_min': int(k), 'k_max': int(min(k + iphi_stp, nphi))}

def iter_block_batches(nb, blk_stp, start=0):
    """Yield (batch_num, selection) for each block batch within one file."""
    for i, b in enumerate(range(int(start), int(nb), int(blk_stp))):
        yield i, {'start_block': int(b),
                  'end_block'  : int(min(b + blk_stp, nb))}

def split_batch(selection, rank, size):
    """This rank's share of one batch. Same keys in as out."""
    lo_key, hi_key = (('start_block', 'end_block') if 'start_block' in selection
                      else ('k_min', 'k_max'))
    lo, hi     = int(selection[lo_key]), int(selection[hi_key])
    base, rem  = divmod(hi - lo, size) # quotient and remainder
    start      = lo + rank * base + min(rank, rem)
    end        = start + base + (1 if rank < rem else 0)
    return {lo_key: int(start), hi_key: int(end)}

def resume_offset(averages, by_block=False):
    """Where a restart should pick up, as a position in the data."""
    was_by_block = bool(averages.get('by_block', False))
    if was_by_block != bool(by_block):
        raise ValueError(
            f"checkpoint was written by "
            f"{'--by_block' if was_by_block else 'phi-slice'} mode but this "
            f"run is {'--by_block' if by_block else 'phi-slice'} mode.")
    return int(averages.get('end_block' if by_block else 'k_max', 0))

def record_progress(averages, selection, by_block, limit, blk_stp=None):
    """Move the checkpoint's resume position to the end of this batch."""
    if by_block:
        averages['end_block'] = int(selection['end_block'])
        averages['blk_stp']   = blk_stp
        averages['done']      = (averages['end_block'] == limit)
    else:
        averages['k_max'] = int(selection['k_max'])
        averages['done']  = (averages['k_max'] == limit)
    averages['by_block'] = bool(by_block)
    averages['saved']    = True

def save_checkpoint(save_dir, file_num, nzro, averages, file_type=None):
    """Write the checkpoint atomically, so a kill cannot truncate it."""
    file_type = file_type or str(averages.get('file_type', 'Profiles'))
    fnum_str  = str(file_num).zfill(nzro)
    final     = os.path.join(save_dir, f'{file_type}_{fnum_str}.npz')
    tmp       = os.path.join(save_dir, f'.{file_type}_{fnum_str}.tmp.npz')
    np.savez(tmp, **averages)
    os.replace(tmp, final)

# Compute quantities at constant phi
class Phi_Slice():
    """
    Manages all derived quantities for a single wedge of data in phi. 
    If k_min and k_max (or phi_min and phi_max) are passed as kwargs,
    the __init__ function will call H-AMR read functions to only read 
    in these specific limits. Otherwise, the whole data set will be read in.
    This class is used to create poloidal plots of the data.

    Attributes
    ----------
    directory : str
        Directory where H-AMR data is stored.
    fnum : int
        File/dump number to be read in.
    **kwargs : dict
        Optional configurations (e.g. k_min, k_max, phi_min, phi_max, reduced).
    """
    def __init__(self, directory, fnum, **kwargs):
        self.H = hr.Read(directory, **kwargs)
        self.H.read_grid(fnum, direct=True)
        self.H.read_prims(fnum, skip_read_params=True)
        self.fnum = fnum
        self.cgs  = kwargs.get('cgs_data', True)
        self.MBH  = kwargs.get('MBH', 10.0) if self.cgs else 1.
        self.Msun = hr.CONSTANTS['Msun']
        self.rho0 = self.H.params['density_scale'] if self.cgs else 1. # g/cm^3
        self.c    = hr.CONSTANTS['c'] if self.cgs else 1. # cm/s
        self.G    = hr.CONSTANTS['G'] if self.cgs else 1. # cm^3/g s^2
        self.Rg   = (self.G * self.MBH * self.Msun / (self.c * self.c)
                     if self.cgs else 1.)

        self.calc_gridsize()

        self.r    = self.H.grid['r'] / self.Rg  
        self.th   = self.H.grid['theta']
        self.phi  = self.H.grid['phi']

        phi0        = np.pi
        self.phi_1D = self.phi[0, 0, 0, :]
        self.iphi   = np.argmin(np.abs(self.phi_1D - phi0))

        self.two_temp  = True if 'Te' and 'Ti' in self.H.data else False
        self.radiation = True if 'u_rad' and 'uu_rad' in self.H.data else False

        acc_args = {
            'grid'      : self.H.grid,
            'data'      : self.H.data,
            'pars'      : self.H.params,
            'CONSTANTS' : hr.CONSTANTS,
            'cgs'       : self.cgs,
            'MBH'       : self.MBH,
            'radiation' : self.radiation
        }
        class_args = {
            'pars'      : self.H.params,
            'CONSTANTS' : hr.CONSTANTS,
            'cgs'       : self.cgs
        }
        mri_args = {
            'r'         : self.H.grid['r'],
            'theta'     : self.H.grid['theta'],
            'phi'       : self.H.grid['phi'],
            'pars'      : self.H.params,
            'CONSTANTS' : hr.CONSTANTS,
            'cgs'       : self.cgs,
            'MBH'       : self.MBH
        }

        # Initialize Accretion class
        self.Acc   = ana.Accretion(**acc_args)
        # Initialize Fluid class
        self.Fluid = ana.Fluid(**class_args)
        # Initialize Field class
        self.Field = ana.Field(**class_args)
        # Initialize Radiation class
        self.Rad   = rad.Radiation(**class_args)
        # Initialize Analytic class
        self.Analytic = theory.Analytic(hr.CONSTANTS, cgs=self.cgs)
        # Initialize MRI class
        self.MRI   = ana.MRI(**mri_args)

    def calc_gridsize(self):
        """
        Calculate grid dimensions (nr, nth, nphi).
        """
        self.nr   = int(self.H.params['nb1'] * self.H.params['bs1'] * self.H.upscalex1 / self.H.lowres1)
        self.nth  = int(self.H.params['nb2'] * self.H.params['bs2'] * self.H.upscalex2 / self.H.lowres2)
        self.nphi = int(self.H.params['nb3'] * self.H.params['bs3'] * self.H.upscalex3 / self.H.lowres3)
        return None

    def calc_XZ(self, iphi):
        """
        Calculate X, Z coordinate array dictionaries 
        (keys 'l' and 'r' for left and right) at given iphi index.
        """
        r   = self.r
        th  = self.th
        phi = self.phi
        
        # Convert to Cartesian coordinates
        # Note reflection for left panel
        self.x = (r*np.sin(th)*np.cos(phi))[0, :, :, iphi]
        self.y = (r*np.sin(th)*np.sin(phi))[0, :, :, iphi]
        self.z = (r*np.cos(th)*np.ones(phi.shape))[0, :, :, iphi]
        
        x = {
            'r' :  self.x,
            'l' : -self.x
        }
        z = {
            'r' : self.z,
            'l' : self.z
        }
        return x, z

    def calc_RZ(self, iphi):
        """
        Calculate R, Z coordinate array dictionaries 
        (keys 'l' and 'r' for left and right) at given iphi index.
        """
        r   = self.r
        th  = self.th
        phi = self.phi
        
        # Convert to cylindrical coordinates
        # Note reflection for left panel
        self.x = (r*np.sin(th)*np.cos(phi))[0, :, :, iphi]
        self.y = (r*np.sin(th)*np.sin(phi))[0, :, :, iphi]
        self.z = (r*np.cos(th)*np.ones(phi.shape))[0, :, :, iphi]

        R = {
            'r' :  np.sqrt(np.square(self.x) + np.square(self.y)),
            'l' : -np.sqrt(np.square(self.x) + np.square(self.y))
        }
        z = {
            'r' : self.z,
            'l' : self.z
        }
        return R, z

    def calc_XZ_uni(self, box_lim, npts):
        """
        Calculate uniform X, Z grid for streamline plots.
        
        Parameters
        ----------
        box_lim : float
            Limit of square box of uniform points in Rg.
        npts : int
            Number of points in X and Z of uniform box.
        
        Returns
        -------
        array, array
            Two arrays of size (nr, nth) of uniform X and Z points.
        """
        x1D = np.linspace(-box_lim, box_lim, npts)
        z1D = np.linspace(-box_lim, box_lim, npts)
        X_uni, Z_uni = np.meshgrid(x1D, z1D)
        return X_uni, Z_uni
    
    def calc_RZ_uni(self, box_lim, npts):
        """
        Calculate uniform R, Z grid for streamline plots.
        
        Parameters
        ----------
        box_lim : float
            Limit of square box of uniform points in Rg.
        npts : int
            Number of points in R and Z of uniform box 
            (0, R_max) x (-z_max, +z_max), where R_max = z_max = box_lim.
        
        Returns
        -------
        array, array
            Two arrays of size (nr, nth) of uniform R and Z points.
        """
        R1D = np.linspace(0., box_lim, npts)
        z1D = np.linspace(-box_lim, box_lim, npts)
        R_uni, Z_uni = np.meshgrid(R1D, z1D)
        return R_uni, Z_uni
    
    def calc_XZ_base(self, iphi):
        """
        Calculate X, Z coordinate array dictionaries for the base resolution 
        (keys 'l' and 'r' for left and right) at given iphi index.
        This function is for use with Aphi contours. 
        """
        # Calculate X Z arrays at the base resolution
        r   = self.r
        th  = self.th
        phi = self.phi
        
        # Convert to Cartesian coordinates
        # Note reflection for left panel
        x2D = (r*np.sin(th)*np.cos(phi))[0, :, :, iphi]
        y2D = (r*np.sin(th)*np.sin(phi))[0, :, :, iphi]
        z2D = (r*np.cos(th)*np.ones(phi.shape))[0, :, :, iphi]

        # Determine how much to downsample to find base grid
        ds1 = 2 ** self.H.amr['max_levelx1']
        ds2 = 2 ** self.H.amr['max_levelx2']

        nrb, nthb = x2D.shape
        if (nrb % ds1 !=0):
            # I don't like this solution
            eps = 1.e-2
            if (nrb / ds1 > round(nrb / ds1 + eps)):
                x2D = x2D[:(nrb - (nrb % ds1)), :(nthb - (nthb % ds2))]
                y2D = y2D[:(nrb - (nrb % ds1)), :(nthb - (nthb % ds2))]
                z2D = z2D[:(nrb - (nrb % ds1)), :(nthb - (nthb % ds2))]
            else:
                x2D = x2D[:(nrb + (nrb % ds1)), :(nthb + (nthb % ds2))]
                y2D = y2D[:(nrb + (nrb % ds1)), :(nthb + (nthb % ds2))]
                z2D = z2D[:(nrb + (nrb % ds1)), :(nthb + (nthb % ds2))]

        x, z = {}, {}
        x['r'] = x2D[ ::ds1, ::ds2]
        z['r'] = z2D[ ::ds1, ::ds2]
        x['l'] = -x2D[::ds1, ::ds2]
        z['l'] = z2D[ ::ds1, ::ds2]
        return x, z
    
    def get_vars2D(self, keys, iphi):
        """
        Retrieve 2D variable arrays for listed keys at phi index iphi.
        """
        variables = []
        for key in keys:
            if key in self.H.data.keys():
                if self.H.data[key].ndim == 4:
                    variables.append(self.H.data[key][0, :, :, iphi])
                elif self.H.data[key].ndim == 5:
                    variables.append(self.H.data[key][:, 0, :, :, iphi])
            elif key in self.H.grid.keys():
                # Note that this should not be used to retrieve r,th,phi in 2D
                if self.H.grid[key].ndim == 4:
                    variables.append(self.H.grid[key][0, :, :, 0])
                elif self.H.grid[key].ndim == 6:
                    variables.append(self.H.grid[key][:, :, 0, :, :, 0])
            else:
                raise ValueError("Invalid variable key.")
        return variables[0] if len(variables) == 1 else variables
    
    def set_Coords2D(self, iphi):
        """
        Set the Coords2D object based on the 2D coordinate arrays at 
        phi index iphi.
        """
        ones  = np.ones_like(self.H.data['rho'])
        coord_args = {
            'r'         : (ones * self.r)[  0, :, :, iphi],
            'theta'     : (ones * self.th)[ 0, :, :, iphi],
            'phi'       : (ones * self.phi)[0, :, :, iphi],
            'pars'      : self.H.params,
            'CONSTANTS' : hr.CONSTANTS
        }
        self.Coords2D = coords.Coordinates(**coord_args)
        return None
    
    def set_Tetrads2D(self, iphi, **kwargs):
        """
        Set the Tetrads2D object based on the 2D coordinate arrays at 
        phi index iphi.
        """
        ones  = np.ones_like(self.H.data['rho'])
        coord_args = {
            'r'         : (ones * self.r)[  0, :, :, iphi],
            'theta'     : (ones * self.th)[ 0, :, :, iphi],
            'phi'       : (ones * self.phi)[0, :, :, iphi],
            'pars'      : self.H.params,
            'CONSTANTS' : hr.CONSTANTS
        }
        self.Tetrads2D = coords.Tetrads(**coord_args, **kwargs)
        return None
    
    # Thermodynamic quantities
    def calc_rho(self, iphi):
        # Density
        rho = self.get_vars2D(['rho'], iphi)
        return rho
    
    def calc_Te(self, iphi):
        # Electron temperature
        Te = self.get_vars2D(['Te'], iphi)
        return Te
    
    def calc_Ti(self, iphi):
        # Ion temperature
        Ti = self.get_vars2D(['Ti'], iphi)
        return Ti
    
    def calc_Te_Ti(self, iphi):
        # Electron temperature, ion temperature
        Te, Ti = self.get_vars2D(['Te', 'Ti'], iphi)
        return Te, Ti
    
    def calc_sigma(self, iphi):
        # sigma, BetaI = pmag/pgas
        rho0, c = self.rho0, self.c    
        B02     = self.Field.B0 * self.Field.B0  
        keys    = ['rho', 'B', 'uu', 'gcov']
        rho, B, uu, gcov = self.get_vars2D(keys, iphi)
    
        # Calculate sigma
        b     = self.Field.calc_b(B, uu, gcov)
        b2    = self.Field.calc_b2(b, B, gcov)
        sigma = (b2 / B02) / (rho / rho0)
        return sigma
    
    def calc_BetaInv(self, iphi, **kwargs):
        """
        Calculate pmag / pgas. 
        If vargamma = True, calculate gamma_adi based on Ti and Te.
        Otherwise, use gamma_adi = 5/3.
        """
        # BetaI = pmag/pgas
        rho0, c = self.rho0, self.c    
        B02     = self.Field.B0 * self.Field.B0  
        keys    = ['B', 'uu', 'u_gas', 'Te', 'Ti', 'gcov']
        B, uu, u_gas, Te, Ti, gcov = self.get_vars2D(keys, iphi)
        
        # Calculate gas pressure and Beta^-1
        if (self.cgs):
            if kwargs.get('vargamma', False):
                pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
            else:
                gamma_adi = 5./3.
                pgas  = (gamma_adi - 1.) * u_gas
        else:
            gamma_adi = 5./3.
            pgas  = (gamma_adi - 1.) * u_gas
        b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        
        BetaI = ((b2 / B02) / 2.) / (pgas / (rho0 * c * c))
        return BetaI

    def calc_radiation_energy_density(self, iphi):
        # radiation energy density
        return self.get_vars2D(['u_rad'], iphi)
    
    def calc_u_t(self, iphi):
        u, gcov = self.get_vars2D(['uu', 'gcov'], iphi)
        u_t = np.einsum('ij...,j...->i...', gcov, u)[0]
        return u_t

    def calc_ion_density(self, iphi):
        if 'tr1' not in self.H.data.keys():
            raise ValueError("tr1 is required to calculate ion density")
        rho, tr1 = self.get_vars2D(['rho', 'tr1'], iphi)
        return tr1 * rho
    
    def calc_lepton_density(self, iphi):
        if 'tr2' not in self.H.data.keys():
            raise ValueError("tr2 is required to calculate lepton density")
        rho, tr2 = self.get_vars2D(['rho', 'tr2'], iphi)
        return tr2 * rho

    # Pressures
    def calc_pgas(self, iphi, **kwargs):
        """
        Calculate gas thermal pressure. 
        If vargamma = True, calculate gamma_adi based on Ti and Te.
        Otherwise, use gamma_adi = 5/3.
        """
        if (self.two_temp and self.radiation):
            u_gas, Te, Ti = self.get_vars2D(['u_gas', 'Te', 'Ti'], iphi)
            return self.Fluid.calc_pgas(u_gas, Te, Ti)
        elif (self.two_temp and self.radiation and 
              not kwargs.get('vargamma', False)):
            u_gas = self.get_vars2D(['u_gas'], iphi)
            gamma_adi = 5./3.
            return (gamma_adi - 1.) * u_gas
        else:
            u_gas = self.get_vars2D(['u_gas'], iphi)
            gamma_adi = 5./3.
            return (gamma_adi - 1.) * u_gas

    def calc_pmag(self, iphi):
        # Magnetic pressure
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        b2    = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)  
        return (b2 / (8. * np.pi) if self.cgs else b2/2.)
    
    def calc_prad(self, iphi):
        # Radiation pressure
        u_rad = self.get_vars2D(['u_rad'], iphi)
        prad  = (1./3.) * u_rad
        return prad
    
    def calc_pmag_pradgas(self, iphi, **kwargs):
        # Magnetic / (Gas Thermal + Radiation)
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        b2    = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu) 
        keys  = ['u_gas', 'u_rad', 'Te', 'Ti']
        if (self.two_temp and self.radiation and kwargs.get('vargamma', True)):
            u_gas, u_rad, Te, Ti = self.get_vars2D(keys, iphi) 
            pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        elif (not kwargs.get('vargamma', True) and self.radiation):
            u_gas, u_rad = self.get_vars2D(['u_gas', 'u_rad'], iphi) 
            gamma_adi = 5./3.
            pgas  = (gamma_adi - 1.) * u_gas
        
        pmag  = b2 / (8. * np.pi) if self.cgs else b2 / 2.
        prad  = (1./3.) * u_rad
        return pmag / (prad + pgas)
    
    def calc_pmag_prad(self, iphi):
        # Magnetic pressure / Radiation pressure
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        b2    = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu) 
        u_rad = self.get_vars2D(['u_rad'], iphi) 
        
        pmag  = b2 / (8. * np.pi) if self.cgs else b2 / 2.
        prad  = (1./3.) * u_rad
        return pmag / prad
    
    def calc_prad_pgas(self, iphi):  
        # Radiation pressure / Gas thermal pressure
        keys  = ['u_gas', 'u_rad', 'Te', 'Ti']
        u_gas, u_rad, Te, Ti = self.get_vars2D(keys, iphi)
        
        pgas  = self.Fluid.calc_pgas(u_gas, Te, Ti)
        prad  = (1./3.) * u_rad
        return prad / pgas

    def calc_rhov2_2b2(self, iphi):
        # Ratio of ram pressure to magnetic pressure
        rho, B, gcov, uu = self.get_vars2D(['rho', 'B', 'gcov', 'uu'], iphi)
        self.set_Coords2D(iphi)

        b2       = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        uusph    = self.Coords2D.int_to_sph_vec(uu)
        gcov_sph = self.Coords2D.int_to_sph_cotnsr(gcov)
        uu_sph   = np.einsum('ij...,j...->i...', gcov_sph, uusph)
        # v_r^2 = v^r v_r = (u^r/u^t) * (u_r/u_t)
        v2    = (uusph[1]/uusph[0]) * (uu_sph[1]/uu_sph[0])
        rhov2 = rho * v2
        return rhov2 / (b2/(4. * np.pi))

    def calc_J2(self, iphi):
        c    = self.c
        uu   = self.H.data['uu']
        B    = self.H.data['B']
        gcov = self.H.grid['gcov']
        gdet = self.H.grid['gdet']
        J    = self.Field.calc_J(uu, B, gcov, gdet, Rg=self.Rg)
        uu_cov = ana.Tools.contract_vector(uu, gcov, cpp=True)
        delta  = np.eye(4, dtype=uu.dtype).reshape(4, 4, *([1] * (uu.ndim - 1)))
        h      = delta + np.einsum('i...,j...->ij...', uu/c, uu_cov/c)
        J_perp = ana.Tools.contract_vector(J, h, cpp=True)
        return ana.Tools.dot(J_perp, J_perp, gcov)[0, :, :, iphi]

    def calc_J_cross_B(self, iphi): 
        Coords = coords.Coordinates(
            self.H.grid['r'] / self.Rg, self.H.grid['theta'], self.H.grid['phi'], 
            self.H.params, hr.CONSTANTS)
        J = self.Field.calc_J(self.H.data['uu'], self.H.data['B'],
                              self.H.grid['gcov'], self.H.grid['gdet'],
                              frame='fluid', Rg=self.Rg) / self.Field.B0
        b = self.Field.calc_b(self.H.data['B'], self.H.data['uu'], self.H.grid['gcov'])/ self.Field.B0
        J_cart = Coords.int_to_cart_vec(J)
        b_cart = Coords.int_to_cart_vec(b)
        J_cross_B_vec3 = np.zeros_like(self.H.data['uu'][1:])
        J_cross_B_vec3[0] = J_cart[1] * b_cart[2] - J_cart[2] * b_cart[1]
        J_cross_B_vec3[1] = J_cart[2] * b_cart[0] - J_cart[0] * b_cart[2]
        J_cross_B_vec3[2] = J_cart[0] * b_cart[1] - J_cart[1] * b_cart[0]
        return np.sqrt(J_cross_B_vec3[0] * J_cross_B_vec3[0] + J_cross_B_vec3[1] * J_cross_B_vec3[1] + J_cross_B_vec3[2] * J_cross_B_vec3[2])[0, :, :, 0]
        #J_cross_B_vec3 = np.einsum('ijk...,j...,k...->i...', 
        #    ana.Tools.calc_Levi_Civita(3, self.H.grid['gdet']), 
        #    Coords.int_to_cart_vec(J)[1:], 
        #    Coords.int_to_cart_vec(b)[1:])
        #J_cross_B_vec4 = np.zeros_like(self.H.data['uu'])
        #J_cross_B_vec4[1:] = J_cross_B_vec3
        #return ana.Tools.dot(J_cross_B_vec4, J_cross_B_vec4, self.H.grid['gcov'])[0, :, :, 0]
 
    # Magnetic Fields
    # "vec" refers to all 3 vector components
    def calc_Bphi(self, iphi):
        B = self.get_vars2D(['B'], iphi)
        self.set_Coords2D(iphi)

        # Calculate B in cylindrical coordinates
        B_cyl = self.Coords2D.int_to_cyl_vec(B)
        return B_cyl[2]

    def calc_Bz(self, iphi):
        B = self.get_vars2D(['B'], iphi)
        self.set_Coords2D(iphi)
    
        # Calculate B in cylindrical coordinates
        B_cyl = self.Coords2D.int_to_cyl_vec(B)
        return B_cyl[3]

    def calc_B_cyl(self, iphi, **kwargs):
        """
        Calculate BR and Bz (cylindrical) dictionaries for streamline plots. 
        If lr = True, return two dictionaries BR and Bz with keys "l" and "r"
        for "left" and "right" side of plot.
        If in_out = True, return BR, Bz, in_out, where in_out contains the 
        sign of B1 (good for showing field/streamline reversals).
        Otherwise, returns BR, Bz arrays.
        """
        B = self.get_vars2D(['B'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in cylindrical coordinates
        B_cyl = self.Coords2D.int_to_cyl_vec(B) / self.Field.B0
        if not kwargs.get('lr', False):
            return B_cyl[1], B_cyl[3]
        else:
            BR = {
                'r' :  B_cyl[1],
                'l' : -B_cyl[1]
            }
            Bz = {
                'r' : B_cyl[3],
                'l' : B_cyl[3]
            }
            if not kwargs.get('in_out', False):
                return BR, Bz
            else:
                in_out = {
                    'r' : np.sign(B[1]),
                    'l' : np.sign(B[1])
                }
                return BR, Bz, in_out
    
    def calc_B_cyl_vec(self, iphi, **kwargs):
        """
        Calculate BR, Bphi, Bz (cylindrical) dictionaries. 
        If lr = True, return dictionaries BR, Bphi, Bz with keys "l" and "r"
        for "left" and "right" side of plot.
        If in_out = True, return BR, Bphi, Bz, in_out, where in_out contains 
        the sign of B1 (good for showing field/streamline reversals).
        Otherwise, returns BR, Bphi, Bz arrays.
        """
        B = self.get_vars2D(['B'], iphi)
        self.set_Coords2D(iphi)

        # Calculate B in cylindrical coordinates
        B_cyl = self.Coords2D.int_to_cyl_vec(B)
        if not kwargs.get('lr', False):
            return B_cyl[1], B_cyl[2], B_cyl[3]
        else:
            BR, Bphi, Bz = {}, {}, {}
            BR['r']   = B_cyl[1]
            Bphi['r'] = B_cyl[2]
            Bz['r']   = B_cyl[3]
            BR['l']   = -B_cyl[1]
            Bphi['l'] = B_cyl[2]
            Bz['l']   = B_cyl[3]
            if not kwargs.get('in_out', False):
                return BR, Bphi, Bz
            else:
                in_out = {}
                in_out['r'] = np.sign(B[1])
                in_out['l'] = np.sign(B[1])
                return BR, Bphi, Bz, in_out

    def calc_B_cart(self, iphi, **kwargs):
        """
        Calculate Bx and Bz (Cartesian) dictionaries for streamline plots. 
        If lr = True, return two dictionaries Bx and Bz with keys "l" and "r"
        for "left" and "right" side of plot.
        If in_out = True, return Bx, Bz, in_out, where in_out contains the 
        sign of B1 (good for showing field/streamline reversals).
        Otherwise, returns Bx, Bz arrays.
        """
        B = self.get_vars2D(['B'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in cylindrical coordinates
        B_cart = self.Coords2D.int_to_cart_vec(B) / self.Field.B0
        if not kwargs.get('lr', False):
            return B_cart[1], B_cart[3]
        else:
            Bx, Bz = {}, {}
            Bx['r'] = B_cart[1]
            Bz['r'] = B_cart[3]
            Bx['l'] = -B_cart[1]
            Bz['l'] = B_cart[3]
            if not kwargs.get('in_out', False):
                return Bx, Bz
            else:
                in_out = {}
                in_out['r'] = np.sign(B[1])
                in_out['l'] = np.sign(B[1])
                return Bx, Bz, in_out

    def calc_bpol(self, iphi):
        """Returns signed poloidal field sqrt(b^2_pol)."""
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        self.set_Coords2D(iphi)

        b    = self.Field.calc_b(B, uu, gcov)
        bcyl = self.Coords2D.int_to_cyl_vec(b, dtype=np.float64) 
        bpol = np.copy(bcyl)
        bpol[2] = 0.
        gcov_cyl = self.Coords2D.int_to_cyl_cotnsr(gcov)
        return np.sign(bcyl[3]) * np.sqrt(ana.Tools.dot(bpol, bpol, gcov_cyl))

    def calc_btor(self, iphi):
        """Returns toroidal field in orthonormal cylindrical coordinates."""
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        self.set_Coords2D(iphi)

        b    = self.Field.calc_b(B, uu, gcov)
        bcyl = self.Coords2D.int_to_cyl_vec(b, dtype=np.float64) 
        return bcyl[2]

    def calc_bpol_btor(self, iphi):
        """Returns signed ratio of poloidal to toroidal field strength."""
        B, gcov, uu = self.get_vars2D(['B', 'gcov', 'uu'], iphi)
        self.set_Coords2D(iphi)

        b    = self.Field.calc_b(B, uu, gcov)
        bcyl = self.Coords2D.int_to_cyl_vec(b, dtype=np.float64) 
        bpol = np.copy(b)
        btor = np.copy(b)
        bpol[0], bpol[3]  = 0., 0.
        btor[0], btor[1], btor[2] = 0., 0., 0.

        sgn     = np.sign(bcyl[3])
        b2pol   = self.Field.calc_b2(bpol, bpol, gcov)
        b2tor   = self.Field.calc_b2(btor, btor, gcov)
        return sgn * np.sqrt(b2pol / b2tor)
    
    def load_phi_averaged_field(self, averages_dir, **kwargs):
        Slice  = Slices()
        slice_args = {
            'averages_dir' : averages_dir,
            'fnums'        : [self.fnum],
            'keys'         : ['bR', 'bz', 'rho0'],
            'reduced'      : kwargs.get('reduced', False),
            'vols'         : [True, True, False]
        }
        slices = Slice.load_slices(**slice_args)
        if slices[f'{self.fnum}']['rho0'] != self.rho0:
            rho0 = slices[f'{self.fnum}']['rho0'] 
            if (hr.physics['TWO_T'] or hr.physics['RAD_M1']):
                c = hr.CONSTANTS['c']
            else:
                c = 1.
            B0 = np.sqrt(4 * np.pi * rho0) * c
        else:
            B0 = self.Field.B0
        nr, nth  = len(self.r[0, :, 0, 0]), len(self.th[0, 0, :, 0])
        lr1, lr2 = self.H.lowres1, self.H.lowres2
        bx = {
            'r' :  slices[f'{self.fnum}']['bR'][::lr1, ::lr2][:nr,:nth] / B0,
            'l' : -slices[f'{self.fnum}']['bR'][::lr1, ::lr2][:nr,:nth] / B0
        }
        bz = {
            'r' :  slices[f'{self.fnum}']['bz'][::lr1, ::lr2][:nr,:nth] / B0,
            'l' :  slices[f'{self.fnum}']['bz'][::lr1, ::lr2][:nr,:nth] / B0
        }
        return bx, bz
    
    def calc_Aphi(self):
        """Calculation of vector potential Aphi for contours."""
        gB1dx2dx3 = self.calc_B1dx2dx3()
        # Northern hemisphere integral starts from North pole
        Aphi_N =  (gB1dx2dx3).cumsum(-1)
        # Southern hemisphere integral starts from South pole
        Aphi_S = -(gB1dx2dx3)[:, ::-1].cumsum(-1)[:, ::-1]
        # Correct Southern integral to get Phi_BH at theta = pi/2
        Aphi_S += 0.5 * gB1dx2dx3
        Aphi   = np.zeros_like(Aphi_N)
        nth    = Aphi.shape[1]
        Aphi[:, :nth//2] = Aphi_N[:, :nth//2]
        Aphi[:, nth//2:] = Aphi_S[:, nth//2:]
        nr, nth = len(self.r[0, :, 0, 0]), len(self.th[0, 0, :, 0])
        # Normalization as in Tchekhovskoy et al. (2011)
        return np.sqrt(4. * np.pi) * Aphi[:nr, :nth]
    
    def calc_B1dx2dx3(self):
        """Loads B1 field at base resolution for Aphi contour plots."""
        read_args = {
            'cgs_grid' : False,
            'cgs_data' : False,
            'lowres1'  : 2 ** self.H.amr['max_levelx1'],
            'lowres2'  : 2 ** self.H.amr['max_levelx2'],
            'lowres3'  : 2 ** self.H.amr['max_levelx3'],
            'r_max'    : self.H.r_max
        }
        D = hr.Read( self.H.dir, **read_args)
        D.read_grid( self.fnum, direct=True)
        D.read_prims(self.fnum, skip_read_params=True)
        dx2, dx3 = D.params['dx2'], D.params['dx3']
        return ((D.grid['gdet'] * D.data['B'][1] * dx2 * dx3).sum(-1))[0]
    
    # Velocities
    def calc_vr(self, iphi):
        """ radial 3-velocity (internal coordinates)"""
        uu = self.get_vars2D(['uu'], iphi)
        return uu[1] / uu[0]

    def calc_vr_sph(self, iphi):
        """ radial 3-velocity (spherical coordinates)"""
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in spherical coordinates
        uu_sph = self.Coords2D.int_to_sph_vec(uu)
        return self.c * uu_sph[1] / uu_sph[0]   

    def calc_vphi_cyl(self, iphi):
        """phi 3-velocity (cylindrical coordinates)"""
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in Cartesian coordinates
        uu_cyl = self.Coords2D.int_to_cyl_vec(uu)
        return self.c * uu_cyl[2] / uu_cyl[0]
        
    def calc_vR_vKep(self, iphi):
        """radial 3-velocity (cylindrical coordinates) / Keplerian velocity"""
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        # Calculate v in cylindridcal coordinates
        uu_cyl = self.Coords2D.int_to_cyl_vec(uu) 
        vR     = self.c * uu_cyl[1] / uu_cyl[0]
        
        vKep   = np.sqrt(self.G * self.MBH * self.Msun / (self.Rg * self.r))
        return vR / vKep[0, :, :, 0]

    def calc_uu_cyl(self, iphi, **kwargs):
        # Returns uuR, uuz
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in cylindrical coordinates
        uu_cyl = self.Coords2D.int_to_cyl_vec(uu) / self.c
        if not kwargs.get('lr', False):
            return uu_cyl[1], uu_cyl[3]
        else:
            uuR, uuz = {}, {}
            uuR['r'] = uu_cyl[1]
            uuz['r'] = uu_cyl[3]
            uuR['l'] = -uu_cyl[1]
            uuz['l'] = uu_cyl[3]
            if not kwargs.get('in_out', False):
                return uuR, uuz
            else:
                in_out = {}
                in_out['r'] = np.sign(uu[1])
                in_out['l'] = np.sign(uu[1])
                return uuR, uuz, in_out
    
    def calc_v_cyl_vec(self, iphi, **kwargs):
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in cylindrical coordinates
        uu_cyl = self.Coords2D.int_to_cyl_vec(uu) / self.c
        if not kwargs.get('lr', False):
            return (
                uu_cyl[1]/uu_cyl[0], 
                uu_cyl[2]/uu_cyl[0], 
                uu_cyl[3]/uu_cyl[0]
            )
        else:
            vR, vphi, vz = {}, {}, {}
            vR['r']   = uu_cyl[1] / uu_cyl[0]
            vphi['r'] = uu_cyl[2] / uu_cyl[0]
            vz['r']   = uu_cyl[3] / uu_cyl[0]
            vR['l']   = -uu_cyl[1] / uu_cyl[0]
            vphi['l'] = uu_cyl[2] / uu_cyl[0]
            vz['l']   = uu_cyl[3] / uu_cyl[0]
            if not kwargs.get('in_out', False):
                return vR, vphi, vz
            else:
                in_out = {}
                in_out['r'] = np.sign(uu[1])
                in_out['l'] = np.sign(uu[1])
                return vR, vphi, vz, in_out
    
    def calc_uu_cart(self, iphi, **kwargs):
        # Returns uux, uuz
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu in Cartesian coordinates
        uu_cart = self.Coords2D.int_to_cart_vec(uu) / self.c
        if not kwargs.get('lr', False):
            return uu_cart[1], uu_cart[3]
        else:
            uux, uuz = {}, {}
            uux['r'] = uu_cart[1]
            uuz['r'] = uu_cart[3]
            uux['l'] = -uu_cart[1]
            uuz['l'] = uu_cart[3]
            if not kwargs.get('in_out', False):
                return uux, uuz
            else:
                in_out = {}
                in_out['r'] = np.sign(uu[1])
                in_out['l'] = np.sign(uu[1])
                return uux, uuz, in_out
    
    def calc_vpol_vtor(self, iphi):
        uu, gcov = self.get_vars2D(['uu', 'gcov'], iphi)
        self.set_Coords2D(iphi)

        uucyl = self.Coords2D.int_to_cyl_vec(uu, dtype=np.float64) 
        uupol = np.copy(uu)
        uutor = np.copy(uu)
        uupol[0], uupol[3]  = 0., 0.
        uutor[0], uutor[1], uutor[2] = 0., 0., 0.

        sgn     = np.sign(uucyl[3])
        v2pol   = ana.Tools.dot(uupol, uupol, gcov)
        v2tor   = ana.Tools.dot(uutor, uutor, gcov)
        return sgn * np.sqrt(v2pol / v2tor)
    
    # Radiation 4-velocity
    def calc_uu_rad_cyl(self, iphi, **kwargs):
        # Returns radiation uuR, uuz
        uu_rad = self.get_vars2D(['uu_rad'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu_rad in cylindrical coordinates
        uu_rad_cyl = self.Coords2D.int_to_cyl_vec(uu_rad) / self.c
        if not kwargs.get('lr', False):
            return uu_rad_cyl[1], uu_rad_cyl[3]
        else:
            uu_radR, uu_radz = {}, {}
            uu_radR['r'] = uu_rad_cyl[1]
            uu_radz['r'] = uu_rad_cyl[3]
            uu_radR['l'] = -uu_rad_cyl[1]
            uu_radz['l'] = uu_rad_cyl[3]
            if not kwargs.get('in_out', False):
                return uu_radR, uu_radz
            else:
                in_out = {}
                in_out['r'] = np.sign(uu_rad[1])
                in_out['l'] = np.sign(uu_rad[1])
                return uu_radR, uu_radz, in_out

    def calc_uu_rad_cart(self, iphi, **kwargs):
        # Returns radiation uux, uuz 
        uu_rad = self.get_vars2D(['uu_rad'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu_rad in Cartesian coordinates
        uu_rad_cart = self.Coords2D.int_to_cart_vec(uu_rad) / self.c
        if not kwargs.get('lr', False):
            return uu_rad_cart[1], uu_rad_cart[3]
        else:
            uu_radx, uu_radz = {}, {}
            uu_radx['r'] = uu_rad_cart[1]
            uu_radz['r'] = uu_rad_cart[3]
            uu_radx['l'] = -uu_rad_cart[1]
            uu_radz['l'] = uu_rad_cart[3]
            if not kwargs.get('in_out', False):
                return uu_radx, uu_radz
            else:
                in_out = {}
                in_out['r'] = np.sign(uu_rad[1])
                in_out['l'] = np.sign(uu_rad[1])
                return uu_radx, uu_radz, in_out

    def calc_Bernoulli(self, iphi, **kwargs):
        rho0, c, B0 = self.rho0, self.c, self.Field.B0
        rho0c2, B02 = rho0 * c * c, B0 * B0
        rho, u_gas, uu = self.get_vars2D(['rho', 'u_gas', 'uu'], iphi)
        B, gcov        = self.get_vars2D(['B', 'gcov'], iphi)
        b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)

        pgas = (self.H.params['gam'] - 1.) * u_gas
        w    = rho * c * c + pgas + u_gas + (b2 / B02) * rho0c2
        u_t  = np.einsum('ij...,j...->i...', gcov, uu)[0] / c

        return (-(w * u_t) / (rho * c * c)) - 1.
 
    # Calculate Edot from stress-energy tensors
    # Minus sign is included such that Edot is positive
    # Options: dictionary=True returns dictionary Edot
    # all=True returns Hydro + EM + Radiation Edot
    # otherwise, set hydro, em, and rad flags separately
    def calc_Edot(self, iphi, **kwargs):
        all = kwargs.get('all', False)
        if (all):
            hydro = True
            em    = True
            radi  = True if self.radiation else False
        else:
            hydro = kwargs.get('hydro', True) 
            em    = kwargs.get('em',    True) 
            radi  = kwargs.get('rad',   True) if self.radiation else False
        data2D = {}
        gdet, gcov, gcon = self.get_vars2D(['gdet', 'gcov', 'gcon'], iphi)
        data2D['gdet']   = gdet
        data2D['gcov']   = gcov
        data2D['gcon']   = gcon
        if (hydro or em):
            uu            = self.get_vars2D(['uu'], iphi)
            data2D['uu']  = uu
        if (hydro):
            rho, u_gas = self.get_vars2D(['rho', 'u_gas'], iphi)
            if self.two_temp:
                Te, Ti = self.get_vars2D(['Te', 'Ti'], iphi)
                pgas    = self.Fluid.calc_pgas(u_gas, Te, Ti)
            else:
                pgas    = (5./3. - 1.) * u_gas
            data2D['rho']   = rho
            data2D['u_gas'] = u_gas
            data2D['pgas']  = pgas
        if (em):
            B           = self.get_vars2D(['B'], iphi)
            data2D['B'] = B
        if (radi):
            u_rad, uu_rad    = self.get_vars2D(['u_rad', 'uu_rad'], iphi)
            data2D['u_rad']  = u_rad
            data2D['uu_rad'] = uu_rad 
        
        if (hydro):
            THD = self.Fluid.calc_THD(**data2D, upper_lower=True, index=[1,0])
        if (em):
            TEM = self.Field.calc_TEM(**data2D, upper_lower=True, index=[1,0])
        if (radi):
            TRAD = self.Rad.calc_TRAD(**data2D, upper_lower=True, index=[1,0])
        
        ones = np.ones_like(self.r * self.th * self.phi)
        Area = 4 * np.pi * np.square(self.Rg * self.r * ones)[0, :, :, iphi]
        if kwargs.get('dictionary', False):
            Edot = {}
            if (hydro):                Edot['HD']      = THD
            if (em):                   Edot['EM']      = TEM
            if (radi):                 Edot['RAD']     = TRAD
            if (hydro and em):         Edot['MHD']     = THD + TEM
            if (hydro and em and rad): Edot['MHD_RAD'] = THD + TEM + TRAD

            c = self.c
            for key in Edot.keys():
                Edot[f'{key}'] = -self.Acc.upscale64(Edot[f'{key}'], Area * c)
            return Edot
        else:
            T_MHD_RAD = np.zeros_like(ones)[0, :, :, iphi]
            if (hydro): T_MHD_RAD += THD
            if (em)   : T_MHD_RAD += TEM
            if (radi) : T_MHD_RAD += TRAD
            return -self.Acc.upscale64(T_MHD_RAD, Area * c)
    
    # Stresses
    def calc_Rphi_stress(self, iphi, **kwargs):
        rho0, c, B0  = self.rho0, self.c, self.Field.B0
        all = kwargs.get('all', False)
        if (all):
            Reynolds = True
            Maxwell  = True
            Radiation   = True if self.radiation else False
        else:
            Reynolds  = kwargs.get('Reynolds',  True) 
            Maxwell   = kwargs.get('Maxwell',   True) 
            Radiation = kwargs.get('Radiation', True) if self.radiation else False
        rho, uu, B, gcov = self.get_vars2D(['rho', 'uu', 'B', 'gcov'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu and b in cylindridcal coordinates
        b      = self.Field.calc_b(B, uu, gcov)
        b_cyl  = self.Coords2D.int_to_cyl_vec(b)
        uu_cyl = self.Coords2D.int_to_cyl_vec(uu)

        # Get mean 4-velocities
        uuR_mean   = kwargs.get('uuR_mean', 0.)
        uuphi_mean = kwargs.get('uuphi_mean', 0.)

        if (Reynolds):
            TRphi_Rey = rho * (uu_cyl[1] - uuR_mean) * (uu_cyl[2] - uuphi_mean)
        if (Maxwell):
            TRphi_Max = -(b_cyl[1] / B0) * (b_cyl[2] / B0) * (rho0 * c * c)
        if (Radiation):
            u_rad, uu_rad = self.get_vars2D(['u_rad', 'uu_rad'], iphi)
            uu_rad_cyl    = self.Coords2D.int_to_cyl_vec(uu_rad)
            duR           = (uu_rad_cyl[1] - uuR_mean) / c
            duphi         = (uu_rad_cyl[2] - uuphi_mean) / c
            TRphi_Rad     = u_rad * duR * duphi
        
        TRphi = np.zeros_like(rho)
        if (kwargs.get('dictionary', False)):
            stresses = {}
            if (Reynolds):  stresses['Reynolds']  = TRphi_Rey
            if (Maxwell):   stresses['Maxwell']   = TRphi_Max
            if (Radiation): stresses['Radiation'] = TRphi_Rad
            if (Reynolds and Maxwell):       
                stresses['MHD'] = TRphi_Rey + TRphi_Max
            if (Reynolds and Maxwell and Radiation): 
                stresses['MHD_RAD'] = TRphi_Rey + TRphi_Max + TRphi_Rad
            return stresses
        else:
            if (Reynolds):  TRphi += TRphi_Rey
            if (Maxwell):   TRphi += TRphi_Max
            if (Radiation): TRphi += TRphi_Rad
            return TRphi

    def calc_zphi_stress(self, iphi, **kwargs):
        rho0, c, B0  = self.rho0, self.c, self.Field.B0
        all = kwargs.get('all', False)
        if (all):
            Reynolds = True
            Maxwell  = True
            Radiation   = True if self.radiation else False
        else:
            Reynolds  = kwargs.get('Reynolds',  True) 
            Maxwell   = kwargs.get('Maxwell',   True) 
            Radiation = kwargs.get('Radiation', True) if self.radiation else False
        rho, uu, B, gcov = self.get_vars2D(['rho', 'uu', 'B', 'gcov'], iphi)
        self.set_Coords2D(iphi)

        # Calculate uu and b in cylindridcal coordinates
        b      = self.Field.calc_b(B, uu, gcov)
        b_cyl  = self.Coords2D.int_to_cyl_vec(b)
        uu_cyl = self.Coords2D.int_to_cyl_vec(uu)

        # Get mean 4-velocities
        uuz_mean   = kwargs.get('uuz_mean', 0.)
        uuphi_mean = kwargs.get('uuphi_mean', 0.)

        if (Reynolds):
            Tzphi_Rey = rho * (uu_cyl[3] - uuz_mean) * (uu_cyl[2] - uuphi_mean)
        if (Maxwell):
            Tzphi_Max = -(b_cyl[3] / B0) * (b_cyl[2] / B0) * (rho0 * c * c)
        if (Radiation):
            u_rad, uu_rad = self.get_vars2D(['u_rad', 'uu_rad'], iphi)
            uu_rad_cyl    = self.Coords2D.int_to_cyl_vec(uu_rad)
            duR           = (uu_rad_cyl[3] - uuz_mean) / c
            duphi         = (uu_rad_cyl[2] - uuphi_mean) / c
            Tzphi_Rad     = u_rad * duR * duphi
        
        Tzphi = np.zeros_like(rho)
        if (kwargs.get('dictionary', False)):
            stresses = {}
            if (Reynolds):  stresses['Reynolds']  = Tzphi_Rey
            if (Maxwell):   stresses['Maxwell']   = Tzphi_Max
            if (Radiation): stresses['Radiation'] = Tzphi_Rad
            if (Reynolds and Maxwell):       
                stresses['MHD'] = Tzphi_Rey + Tzphi_Max
            if (Reynolds and Maxwell and Radiation): 
                stresses['MHD_RAD'] = Tzphi_Rey + Tzphi_Max + Tzphi_Rad
            return stresses
        else:
            if (Reynolds):  Tzphi += Tzphi_Rey
            if (Maxwell):   Tzphi += Tzphi_Max
            if (Radiation): Tzphi += Tzphi_Rad
            return Tzphi

    # Cooling rates
    # Inverse-Compton cooling rate
    def calc_qIC(self, iphi, **kwargs):
        rho, Te, u_rad = self.get_vars2D(['rho', 'Te', 'u_rad'], iphi)
        uu, gcov       = self.get_vars2D(['uu', 'gcov'], iphi)
        u_t            = np.einsum('ij...,j...->i...', gcov, uu)[0] / self.c
        return -u_t * self.Rad.calc_qIC(rho, Te, u_rad, **kwargs) 

    # Coulomb cooling rate
    def calc_qie(self, iphi, **kwargs):
        mid         = kwargs.get('midplane', False)
        rho, Ti, Te = self.H.data['rho'], self.H.data['Ti'], self.H.data['Te']
        uu, gcov    = self.H.data['uu'], self.H.grid['gcov']
        u_t         = np.einsum('ij...,j...->i...', gcov, uu)[0] / self.c
        qie = self.Fluid.calc_qie(rho, Te, Ti, **kwargs)
        return -(u_t * qie)[0, :, :, iphi] if not mid else -(u_t * qie)[0]

    # Synchrotron cooling rate
    def calc_qsync(self, iphi, **kwargs):
        rho, Te, u_rad = self.get_vars2D(['rho', 'Te', 'u_rad'], iphi)
        B, uu, gcov    = self.get_vars2D(['B', 'uu', 'gcov'], iphi)
        u_t            = np.einsum('ij...,j...->i...', gcov, uu)[0] / self.c
        # b2 is required for synchrotron opacities
        if 'b2' not in kwargs:
            b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        else:
            b2 = kwargs.get('b2')
        opacities = dict.fromkeys(['synchrotron'])
        sync_args = {
            'opacities' : opacities, 
            'b2'        : b2, 
            'XRB'       : True,
            'RIAF'      : kwargs.get('RIAF', False)
        }
        return -u_t * self.Rad.calc_q_emi(rho, Te, u_rad, **sync_args)

    # Dissipation rate in the coordinate frame
    def calc_qdiss(self, iphi):
        qdiss, uu, gcov = self.get_vars2D(['dissipation', 'uu', 'gcov'], iphi)
        u_t             = np.einsum('ij...,j...->i...', gcov, uu)[0] / self.c
        return -u_t * qdiss

    # Radiation source term in the coordinate frame (G_t)
    # Note that the radiation is sourced by -G_t
    def calc_qrad(self, iphi):
        qrad = self.get_vars2D(['rad_src'], iphi)
        return -qrad
    
    # Ion heating fraction
    def calc_delta_i(self, iphi):
        rho, Te, Ti, u_gas = self.get_vars2D(['rho', 'Te', 'Ti', 'u_gas'], iphi)
        B, uu, gcov        = self.get_vars2D(['B', 'uu', 'gcov'], iphi)
        delta_e            = self.Fluid.calc_delta_e(rho, u_gas, Te, Ti, B, gcov, uu=uu)
        return 1. - delta_e

    # 1 - (n_e / n_i)
    def calc_ion_fraction(self, iphi):
        if 'tr1' not in self.H.data.keys() or 'tr2' not in self.H.data.keys():
            raise ValueError("tr1 and tr2 is required to calculate ion fraction")
        tr1, tr2 = self.get_vars2D(['tr1', 'tr2'], iphi)
        return (tr1 > tr2) * (tr1 > 0.) * (1. - np.nan_to_num(tr2 / tr1))

    # Volume
    def calc_Vol(self, iphi):
        gdet = self.get_vars2D(['gdet'], iphi)
        dx1  = self.H.params['dx1']
        dx2  = self.H.params['dx2']
        dx3  = self.H.params['dx3']
        return (self.Rg ** 3) * gdet * dx1 * dx2 * dx3
    
    # Current sheet analysis
    def calc_B_rec(self, iphi, R_uni, Z_uni):
        B = self.get_vars2D(['B'], iphi)
        self.set_Coords2D(iphi)

        B_sph = self.Coords2D.int_to_sph_vec(B)

        # Convert to locally flat (Minkowski) frame
        self.set_Tetrads2D(iphi, system='Takahashi_Kerr_Schild')
        B_sph_flat = np.einsum('ij...,j...->i...', self.Tetrads2D.e, B_sph)

        # Convert to cylindrical coordinates
        self.set_Coords2D(iphi)
        B_cyl_flat = self.Coords2D.ortho_sph_to_cyl_vec(B_sph_flat)

        # Interpolate all fields to a uniform grid
        ones = np.ones(self.H.data['rho'].shape)
        r2D  = (ones * self.r)[  0, :, :, iphi]
        th2D = (ones * self.th)[ 0, :, :, iphi]

        R2D  = r2D * np.sin(th2D)
        Z2D  = r2D * np.cos(th2D)

        BR   = griddata((R2D.ravel(), Z2D.ravel()), B_cyl_flat[1].ravel(), 
                        (R_uni, Z_uni), method='cubic')
        Bphi = griddata((R2D.ravel(), Z2D.ravel()), B_cyl_flat[2].ravel(), 
                        (R_uni, Z_uni), method='cubic')
        Bz   = griddata((R2D.ravel(), Z2D.ravel()), B_cyl_flat[3].ravel(), 
                        (R_uni, Z_uni), method='cubic')
        return BR, Bphi, Bz
 
    def calc_mag_energy_rec(self, iphi, R_uni, Z_uni):
        B, uu, gcov = self.get_vars2D(['B', 'uu', 'gcov'], iphi)
        rho0, c = self.rho0, self.c
        B02     = self.Field.B0 * self.Field.B0
        b2  = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)

        # Interpolate energy density to a uniform grid
        ones = np.ones(self.H.data['rho'].shape)
        r2D  = (ones * self.r)[  0, :, :, iphi]
        th2D = (ones * self.th)[ 0, :, :, iphi]

        R2D  = r2D * np.sin(th2D)
        Z2D  = r2D * np.cos(th2D)

        Emag_uni = griddata((R2D.ravel(), Z2D.ravel()), 
                            (0.5 * (b2 / B02) * rho0 * c * c).ravel(), 
                            (R_uni, Z_uni), method='cubic')
        return Emag_uni

    def calc_vrec_z(self, iphi, R_uni, Z_uni):
        uu = self.get_vars2D(['uu'], iphi)
        self.set_Coords2D(iphi)

        uu_sph = self.Coords2D.int_to_sph_vec(uu)

        # Convert to locally flat (Minkowski) frame
        self.set_Tetrads2D(iphi, system='Takahashi_Kerr_Schild')
        uu_sph_flat = np.einsum('ij...,j...->i...', self.Tetrads2D.e, uu_sph)

        # Convert to cylindrical coordinates
        uu_cyl_flat = self.Coords2D.ortho_sph_to_cyl_vec(uu_sph_flat)

        # Interpolate to the uniform grid
        ones = np.ones(self.H.data['rho'].shape)
        r2D  = (ones * self.r)[  0, :, :, iphi]
        th2D = (ones * self.th)[ 0, :, :, iphi]

        R2D = r2D * np.sin(th2D)
        Z2D = r2D * np.cos(th2D)

        vz_uni = griddata((R2D.ravel(), Z2D.ravel()), 
                          (uu_cyl_flat[3] / uu_cyl_flat[0]).ravel(), 
                          (R_uni, Z_uni), method='cubic')
        return self.c * vz_uni 
    
    def calc_Poynt_rec_z(self, iphi, R_uni, Z_uni):
        B, uu, gcov, gcon = self.get_vars2D(['B', 'uu', 'gcov', 'gcon'], iphi)

        # Calculate Poynting flux in spherical coordinates
        self.set_Coords2D(iphi)
        Tmu_t_sph = self.Coords2D.int_to_sph_vec(
            self.Field.calc_TEM(uu, B, gcov, gcon, upper_lower=True)[:, 0])
        
        # Convert to locally flat (Minkowski) frame
        self.set_Tetrads2D(iphi, system='Takahashi_Kerr_Schild')
        Tmu_t_sph_flat = np.einsum(
            'ij...,j...->i...', self.Tetrads2D.e, Tmu_t_sph)
        
        # Convert spherical to cylindrical coordinates
        Tmu_t_cyl_flat = self.Coords2D.ortho_sph_to_cyl_vec(Tmu_t_sph_flat)

        # Interpolate to the uniform grid
        ones = np.ones(self.H.data['rho'].shape)
        r2D  = (ones * self.r)[  0, :, :, iphi]
        th2D = (ones * self.th)[ 0, :, :, iphi]

        R2D = r2D * np.sin(th2D)
        Z2D = r2D * np.cos(th2D)
        Tz_t_uni = griddata((R2D.ravel(), Z2D.ravel()), 
                            (Tmu_t_cyl_flat[3]).ravel(), 
                            (R_uni, Z_uni), method='cubic')
        return -Tz_t_uni

    def calc_Rad_rec_z(self, iphi, R_uni, Z_uni):
        B, uu, gcov, gcon = self.get_vars2D(['B', 'uu', 'gcov', 'gcon'], iphi)

        # Calculate Poynting flux in spherical coordinates
        self.set_Coords2D(iphi)
        Tmu_t_sph = self.Coords2D.int_to_sph_vec(
            self.Field.calc_TEM(uu, B, gcov, gcon, upper_lower=True)[:, 0])
        
        # Convert to locally flat (Minkowski) frame
        self.set_Tetrads2D(iphi, system='Takahashi_Kerr_Schild')
        Tmu_t_sph_flat = np.einsum(
            'ij...,j...->i...', self.Tetrads2D.e, Tmu_t_sph)
        
        # Convert spherical to cylindrical coordinates
        Tmu_t_cyl_flat = self.Coords2D.ortho_sph_to_cyl_vec(Tmu_t_sph_flat)

        # Interpolate to the uniform grid
        ones = np.ones(self.H.data['rho'].shape)
        r2D  = (ones * self.r)[  0, :, :, iphi]
        th2D = (ones * self.th)[ 0, :, :, iphi]

        R2D = r2D * np.sin(th2D)
        Z2D = r2D * np.cos(th2D)
        Tz_t_uni = griddata((R2D.ravel(), Z2D.ravel()), 
                            (Tmu_t_cyl_flat[3]).ravel(), 
                            (R_uni, Z_uni), method='cubic')
        return -Tz_t_uni
    
    # Plasma timescales
    # Ion-electron temperature equilibration
    def calc_teq(self, iphi, **kwargs):
        mid         = kwargs.get('midplane', False)
        rho, Ti, Te = self.H.data['rho'], self.H.data['Ti'], self.H.data['Te']
        teq         = self.Fluid.calc_teq(rho, Te, Ti, **kwargs)
        return teq[0, :, :, iphi] if not mid else teq[0]
    
    # Ion-ion collisions
    def calc_tii(self, iphi, **kwargs):
        mid         = kwargs.get('midplane', False)
        rho, Ti, Te = self.H.data['rho'], self.H.data['Ti'], self.H.data['Te']
        tii         = self.Fluid.calc_tii(rho, Te, Ti, **kwargs)
        return tii[0, :, :, iphi] if not mid else tii[0]
    
    # Electron-electron collisions
    def calc_tee(self, iphi, **kwargs):
        mid         = kwargs.get('midplane', False)
        rho, Ti, Te = self.H.data['rho'], self.H.data['Ti'], self.H.data['Te']
        tee         = self.Fluid.calc_tee(rho, Te, Ti, **kwargs)
        return tee[0, :, :, iphi] if not mid else tee[0]
    
    # Inverse Compton
    def calc_tIC(self, iphi, **kwargs):
        mid            = kwargs.get('midplane', False)
        rho, Te, u_rad = self.H.data['rho'], self.H.data['Te'], self.H.data['u_rad']
        tIC            = self.Rad.calc_tIC(rho, Te, u_rad, **kwargs)
        return tIC[0, :, :, iphi] if not mid else tIC[0]
        
    # Optical depths
    def calc_tau(self, opacities, iphi, **kwargs):
        tau = self.Acc.calc_tau(opacities=opacities, sigma_cut=3., **kwargs)
        return tau[0, :, :, iphi]

    # Opacities
    def calc_kappa_es(self, iphi, **kwargs):
        rho, Te, u_rad = self.get_vars2D(['rho', 'Te', 'u_rad'], iphi)
        opacities = dict.fromkeys(['electron_scattering'])
        Opac = rad.Opacity(opacities, rho, Te, u_rad, self.H.params, 
                           hr.CONSTANTS, mechanism='emission')
        return Opac.opacities['electron_scattering']

    def calc_kappa_eff(self, iphi, **kwargs):
        rho, Te, u_rad = self.get_vars2D(['rho', 'Te', 'u_rad'], iphi)
        opacities = dict.fromkeys(['electron_scattering', 'bound_free', 
            'CHIANTI', 'free_free', 'hydrogen', 'molecular', 'synchrotron'])
        if 'b2' not in kwargs:
            B, uu, gcov = self.get_vars2D(['B', 'uu', 'gcov'], iphi)
            b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        else:
            b2 = kwargs.get('b2')
        Opac = rad.Opacity(opacities, rho, Te, u_rad, self.H.params, 
            hr.CONSTANTS, mechanism='absorption', b2=b2, XRB=True)
        kappa_abs = Opac.opacities['absorption_total']
        kappa_es  = Opac.opacities['electron_scattering']
        kappa_eff = np.sqrt(kappa_abs * (kappa_abs + kappa_es))
        return kappa_eff
    
    def calc_kappa_sync(self, iphi, **kwargs):
        rho, Te, u_rad = self.get_vars2D(['rho', 'Te', 'u_rad'], iphi)
         # b2 is required for synchrotron opacities
        if 'b2' not in kwargs:
            B, uu, gcov = self.get_vars2D(['B', 'uu', 'gcov'], iphi)
            b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        else:
            b2 = kwargs.get('b2')
        opacities = dict.fromkeys(['synchrotron'])
        if not kwargs.get('RIAF', False):
            Opac = rad.Opacity(opacities, rho, Te, u_rad, self.H.params, 
                               hr.CONSTANTS, mechanism='emission', b2=b2, XRB=True)
        else:
            Opac = rad.RIAF_Opacity(opacities, rho, Te, u_rad, self.H.params, 
                                    hr.CONSTANTS, mechanism='emission', b2=b2, XRB=True)
        return Opac.opacities['synchrotron_emi']
    
    def calc_kappa_ff(self, iphi, **kwargs):
        rho, Te, u_rad = self.get_vars2D(['rho', 'Te', 'u_rad'], iphi)
         # b2 is required for synchrotron opacities
        opacities = dict.fromkeys(['free_free'])
        Opac = rad.Opacity(opacities, rho, Te, u_rad, self.H.params, 
                           hr.CONSTANTS, mechanism='emission', XRB=True)
        return Opac.opacities['free_free_emi']

    # Orbital frequency
    def calc_OmegaK(self):
        return self.Analytic.calc_OmegaK(
            self.MBH, self.H.params['a'], self.r)[0, :, 0, :]
    
    # Scale heights ratios/ Disk thicknesses (H/r)
    def calc_HoR_thm(self, iphi):
        Te, Ti, uu = self.get_vars2D(['Te', 'Ti', 'uu'], iphi)
        c    = self.c
        kB   = hr.CONSTANTS['kB']
        mp   = hr.CONSTANTS['mp'] 

        x_AB = hr.CONSTANTS['X_AB']
        y_AB = hr.CONSTANTS['Y_AB']

        # Equation 11 in Sadowski+ (2017)
        mu_i = 4.0 / (4.0 * x_AB + y_AB)
        
        cs = np.sqrt(kB * (Ti + Te) / (mu_i * mp))
        self.set_Coords2D(iphi)

        # Calculate uu in orthonormal spherical coordinates
        uu_sph = self.Coords2D.int_to_ortho_sph_vec(uu)
        vphi   = c * (uu_sph[3] / uu_sph[0])
        return cs/vphi

    def calc_HoR_mag(self, iphi):
        rho, uu      = self.get_vars2D(['rho', 'uu'], iphi)
        u_gas, u_rad = self.get_vars2D(['u_gas', 'u_rad'], iphi)
        B, gcov      = self.get_vars2D(['B', 'gcov'], iphi)
        b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        c  = self.c
        gamma_adi = 5./3.
        pgas = (gamma_adi - 1.) * u_gas
        w  = rho * c * c + b2 / (4. * np.pi) + u_gas + pgas + (4./3.) * u_rad
        # Calculate uu in orthonormal spherical coordinates
        uu_sph = self.Coords2D.int_to_ortho_sph_vec(uu)
        vphi   = c * (uu_sph[3] / uu_sph[0])
        return (c / vphi) * np.sqrt(b2 / (4. * np.pi * w))

    def calc_HoR_rad(self, iphi):
        rho, uu      = self.get_vars2D(['rho', 'uu'], iphi)
        u_gas, u_rad = self.get_vars2D(['u_gas', 'u_rad'], iphi)
        c  = self.c
        gamma_adi = 5./3.
        pgas  = (gamma_adi - 1.) * u_gas
        prad  = (1./3.) * u_rad
        ptot  = pgas + prad
        rhoc2 = rho * c * c
        cs_rad = c * np.sqrt((4./3.) * ptot / (rhoc2 + u_rad + u_gas))
        # Calculate uu in orthonormal spherical coordinates
        uu_sph = self.Coords2D.int_to_ortho_sph_vec(uu)
        vphi   = c * (uu_sph[3] / uu_sph[0])
        return cs_rad / vphi
    
    def calc_HoR_tot(self, iphi):
        rho, Te, Ti, uu = self.get_vars2D(['rho', 'Te', 'Ti', 'uu'], iphi)
        u_gas, u_rad    = self.get_vars2D(['u_gas', 'u_rad'], iphi)
        B, gcov         = self.get_vars2D(['B', 'gcov'], iphi)

        # Thermal
        kB   = hr.CONSTANTS['kB']
        mp   = hr.CONSTANTS['mp'] 
        x_AB = hr.CONSTANTS['X_AB']
        y_AB = hr.CONSTANTS['Y_AB']
        # Equation 11 in Sadowski+ (2017)
        mu_i = 4.0 / (4.0 * x_AB + y_AB)
        cs     = np.sqrt(kB * (Ti + Te) / (mu_i * mp))

        # Radiation
        c      = self.c
        gamma_adi = 5./3.
        pgas   = (gamma_adi - 1.) * u_gas
        prad   = (1./3.) * u_rad
        ptot   = pgas + prad
        rhoc2  = rho * c * c
        cs_rad = c * np.sqrt((4./3.) * ptot / (rhoc2 + u_rad + u_gas))
        
        # Magnetic
        b2 = self.Field.calc_b2(B, B, gcov, from_B=True, uu=uu)
        w  = rhoc2 + (b2 / (4. * np.pi)) + u_gas + pgas + (4./3.) * u_rad
        vA = c * np.sqrt(b2 / (4. * np.pi * w))
        vtot = np.sqrt(vA * vA + cs_rad * cs_rad + cs * cs)
        # Calculate uu in orthonormal spherical coordinates
        uu_sph = self.Coords2D.int_to_ortho_sph_vec(uu)
        vphi   = c * (uu_sph[3] / uu_sph[0])
        return vtot/vphi

    def calc_Q(self, iphi):
        """MRI Quality Factors"""
        keys = ['rho', 'uu', 'B', 'u_gas', 'gcov', 
                'level_x1', 'level_x2', 'level_x3']
        rho, uu, B, u_gas, gcov, level_x1, level_x2, level_x3 = (
            self.get_vars2D(keys, iphi))

        return self.MRI.calc_Q(rho, uu, B, u_gas, gcov,
            amr=self.H.amr, 
            level_x1=level_x1, level_x2=level_x2, level_x3=level_x3,
            lowres1=self.H.lowres1, lowres2=self.H.lowres2, lowres3=self.H.lowres3)

# Compute quantities at constant theta
class Theta_Slice(Phi_Slice):
    def __init__(self, directory, fnum, **kwargs):
        self.H = hr.Read(directory, **kwargs)
        self.H.read_grid(fnum, direct=True)
        self.H.read_prims(fnum, skip_read_params=True)

# Compute quantities in the miplane (averaging over two nearest cells)
class Midplane_Slice(Phi_Slice):
    def __init__(self, directory, fnum, **kwargs):
        H = hr.Read(directory, **kwargs)
        H.read_params(fnum)
        nth = int(H.params['nb2'] * H.params['bs2'] * H.upscalex2 / H.lowres2)
        if 'j_min' in kwargs: kwargs.pop('j_min')
        if 'j_max' in kwargs: kwargs.pop('j_max')

        self.H = hr.Read(directory, j_min = nth/2-1, j_max = nth/2+1, **kwargs)
        self.H.read_grid( fnum, direct=True)
        self.H.read_prims(fnum, skip_read_params=True)

        self.cgs  = kwargs.get('cgs_data', True)
        self.MBH  = kwargs.get('MBH', 10.0) if self.cgs else 1.
        self.Msun = hr.CONSTANTS['Msun']
        self.rho0 = self.H.params['density_scale'] if self.cgs else 1. # g/cm^3
        self.c    = hr.CONSTANTS['c'] if self.cgs else 1. # cm/s
        self.G    = hr.CONSTANTS['G'] if self.cgs else 1. # cm^3/g s^2
        self.Rg   = (self.G * self.MBH * self.Msun / (self.c * self.c)
                     if self.cgs else 1.)

        self.r    = self.H.grid['r'] / self.Rg  
        self.th   = self.H.grid['theta']
        self.phi  = self.H.grid['phi']
        
        self.two_temp  = True if 'Te' and 'Ti' in self.H.data else False
        self.radiation = True if 'u_rad' and 'uu_rad' in self.H.data else False

        acc_args = {
            'grid'      : self.H.grid,
            'data'      : self.H.data,
            'pars'      : self.H.params,
            'CONSTANTS' : hr.CONSTANTS,
            'cgs'       : self.cgs,
            'MBH'       : self.MBH,
            'radiation' : self.radiation
        }
        class_args = {
            'pars'      : self.H.params,
            'CONSTANTS' : hr.CONSTANTS,
            'cgs'       : self.cgs
        }

        # Initialize Accretion class
        self.Acc   = ana.Accretion(**acc_args)
        # Initialize Fluid class
        self.Fluid = ana.Fluid(**class_args)
        # Initialize Field class
        self.Field = ana.Field(**class_args)
        # Initialize Radiation class
        self.Rad   = rad.Radiation(**class_args)
        # Initialize Analytic class
        self.Analytic = theory.Analytic(hr.CONSTANTS, cgs=self.cgs)

    def mid_val(self, q):
        return (q[:, 0, :] + q[:, 1, :]) / 2.

    def calc_XY(self):
        r   = self.r
        th  = self.th
        phi = self.phi
        
        # Convert to Cartesian coordinates
        # Note reflection for left panel
        self.x =  (r*np.sin(th)*np.cos(phi))
        self.y =  (r*np.sin(th)*np.sin(phi))
        self.z =  (r*np.cos(th)*np.ones(phi.shape))
        
        x, y = {}, {}
        x['r'] = (self.x[0, :, 0, :] + self.x[0, :, 1, :]) / 2.
        y['r'] = (self.y[0, :, 0, :] + self.y[0, :, 1, :]) / 2.
        return x, y
    
    def calc_XY_uni(self, box_lim, npts):
        # Make uniform grid for streamline plots
        x1D = np.linspace(-box_lim, box_lim, npts)
        y1D = np.linspace(-box_lim, box_lim, npts)
        X_uni, Y_uni = np.meshgrid(x1D, y1D)
        return X_uni, Y_uni

    def get_vars3D(self, keys):
        variables = []
        for key in keys:
            if key in self.H.data.keys():
                shape = list(self.H.data[f'{key}'].shape)
                if len(shape) == 4:
                    variables.append(self.H.data[f'{key}'][0])
                if len(shape) == 5:
                    variables.append(self.H.data[f'{key}'][:, 0])
            elif key in self.H.grid.keys():
                shape = list(self.H.grid[f'{key}'].shape)
                # Note that this should not be used to retrieve r,th,phi in 2D
                if len(shape) == 4:
                    variables.append(self.H.grid[f'{key}'][0, :, :, :])
                if len(shape) == 6:
                    variables.append(self.H.grid[f'{key}'][:, :, 0, :, :, :])
            else:
                return ValueError("Invalid variable key.")
        return variables[0] if len(variables) == 1 else variables
    
    def get_vars2D(self, keys, iphi):
        return self.get_vars3D(keys)
    
    def set_Coords3D(self):
        ones  = np.ones(self.H.data['rho'].shape)
        r3D   = (ones * self.r)[  0]
        th3D  = (ones * self.th)[ 0]
        phi3D = (ones * self.phi)[0]
        self.Coords3D = coords.Coordinates(r3D, th3D, phi3D, 
                                           self.H.params, hr.CONSTANTS)
        return None
    
    def set_Coords2D(self, iphi):
        self.set_Coords3D()

    # Thermodynamic quantities
    def calc_rho(self):
        # Density
        return self.mid_val(super().calc_rho(0))
    
    def calc_Te(self):
        # Electron temperature
        return self.mid_val(super().calc_Te(0))
    
    def calc_Ti(self):
        # Ion temperature
        return self.mid_val(super().calc_Ti(0))
    
    def calc_Te_Ti(self):
        # Electron temperature, ion temperature
        Te = self.mid_val(super().calc_Te(0))
        Ti = self.mid_val(super().calc_Ti(0))
        return Te, Ti
    
    def calc_sigma(self):
        # sigma
        return self.mid_val(super().calc_sigma(0))

    def calc_BetaInv(self):
        # Beta^(-1)
        return self.mid_val(super().calc_BetaInv(0))

    def calc_radiation_energy_density(self):
        # radiation energy density
        return self.mid_val(super().calc_radiation_energy_density(0))
    
    def calc_u_t(self):
        return self.mid_val(super().calc_u_t(0))

    def calc_ion_density(self):
        return self.mid_val(super().calc_ion_density(0))
    
    def calc_lepton_density(self):
        return self.mid_val(super().calc_lepton_density(0))

    # Pressures
    def calc_pgas(self, **kwargs):
        return self.mid_val(super().calc_pgas(0, **kwargs))

    def calc_pmag(self):
        return self.mid_val(super().calc_pmag(0))

    def calc_prad(self):
        return self.mid_val(super().calc_prad(0))
    
    def calc_pmag_pradgas(self, **kwargs):
        pmag = self.mid_val(super().calc_pmag(0))
        prad = self.mid_val(super().calc_prad(0))
        pgas = self.mid_val(super().calc_pgas(0, **kwargs))
        return pmag / (prad + pgas)

    def calc_pmag_prad(self):
        pmag = self.mid_val(super().calc_pmag(0))
        prad = self.mid_val(super().calc_prad(0))
        return pmag / prad
    
    def calc_prad_pgas(self, **kwargs):
        prad = self.mid_val(super().calc_prad(0))
        pgas = self.mid_val(super().calc_pgas(0, **kwargs))
        return prad / pgas

    # Magnetic Fields
    def calc_Bphi(self):
        B = self.get_vars3D(['B'])
        self.set_Coords3D()
        B_cyl = self.Coords3D.int_to_cyl_vec(B)
        return self.mid_val(B_cyl[2])

    def calc_Bz(self):
        return self.mid_val(super().calc_Bz(0))

    def calc_B_cyl(self, **kwargs):
        B = self.get_vars3D(['B'])
        self.set_Coords3D()

        # Calculate B in cylindrical coordinates
        B_cyl = self.Coords3D.int_to_cyl_vec(B)
        if not kwargs.get('lr', False):
            BR   = self.mid_val(B_cyl[1])
            Bphi = self.mid_val(B_cyl[2])
            Bz   = self.mid_val(B_cyl[3])
            return BR, Bphi, Bz
        else:
            BR, Bphi, Bz = {}, {}, {}
            BR['r']   = self.mid_val(B_cyl[1])
            Bphi['r'] = self.mid_val(B_cyl[2])
            Bz['r']   = self.mid_val(B_cyl[3])
            if not kwargs.get('in_out', False):
                return BR, Bphi, Bz
            else:
                in_out = {'r' : np.sign(B[1, :, 0, :] + B[1, :, 1, :])}
                return BR, Bphi, Bz, in_out

    def calc_B_cart(self, **kwargs):
        B = self.get_vars3D(['B'])
        self.set_Coords3D()

        # Calculate B in Cartesian coordinates
        B_cart = self.Coords3D.int_to_cart_vec(B) / self.Field.B0
        if not kwargs.get('lr', False):
            Bx = self.mid_val(B_cart[1])
            By = self.mid_val(B_cart[2])
            return Bx, By
        else:
            Bx, By = {}, {}
            Bx['r'] = self.mid_val(B_cart[1])
            By['r'] = self.mid_val(B_cart[2])
            if not kwargs.get('in_out', False):
                return Bx, By
            else:
                in_out = {'r' : np.sign(B[1, :, 0, :] + B[1, :, 1, :])}
                return Bx, By, in_out

    def calc_bpol(self):
        return self.mid_val(super().calc_bpol(0))

    def calc_btor(self):
        return self.mid_val(super().calc_btor(0))

    def calc_bpol_btor(self):
        # Returns signed ratio of poloidal to toroidal field strength
        B, gcov, uu = self.get_vars3D(['B', 'gcov', 'uu'])
        self.set_Coords3D()

        b    = self.Field.calc_b(B, uu, gcov)
        bcyl = self.Coords3D.int_to_cyl_vec(b, dtype=np.float64) 
        bpol = np.copy(b)
        btor = np.copy(b)
        bpol[0], bpol[3]  = 0., 0.
        btor[0], btor[1], btor[2] = 0., 0., 0.

        sgn     = np.sign(bcyl[3])
        b2pol   = self.Field.calc_b2(bpol, bpol, gcov)
        b2tor   = self.Field.calc_b2(btor, btor, gcov)
        bpol_btor = sgn * np.sqrt(b2pol / b2tor)
        return 0.5 * (bpol_btor[:, 0, :] + bpol_btor[:, 1, :])
    
    def calc_J2(self):
        c    = self.c
        uu   = self.H.data['uu']
        B    = self.H.data['B']
        gcov = self.H.grid['gcov']
        gdet = self.H.grid['gdet']
        J    = self.Field.calc_J(uu, B, gcov, gdet, Rg=self.Rg)
        uu_cov = ana.Tools.contract_vector(uu, gcov, cpp=True)
        delta  = np.eye(4, dtype=uu.dtype).reshape(4, 4, *([1] * (uu.ndim - 1)))
        h      = delta + np.einsum('i...,j...->ij...', uu/c, uu_cov/c)
        J_perp = ana.Tools.contract_vector(J, h, cpp=True)
        return self.mid_val(ana.Tools.dot(J_perp, J_perp, gcov)[0])
    
    # Fluid velocities
    def calc_vr(self):
        return self.mid_val(super().calc_vr(0))

    def calc_vr_sph(self):
        return self.mid_val(super().calc_vr_sph(0))

    def calc_vphi_cyl(self):
        return self.mid_val(super().calc_vphi_sph(0))

    def calc_vR_vKep(self):
        return self.mid_val(super().calc_vR_vKep(0))

    def calc_v_cyl(self, **kwargs):
        uu = self.get_vars3D(['uu'])
        self.set_Coords3D()

        # Calculate uu in cylindrical coordinates
        uu_cyl = self.Coords3D.int_to_cyl_vec(uu) / self.c
        if not kwargs.get('lr', False):
            ut   = self.mid_val(uu_cyl[0])
            uR   = self.mid_val(uu_cyl[1])
            uphi = self.mid_val(uu_cyl[2])
            uz   = self.mid_val(uu_cyl[3])
            return uR/ut, uphi/ut, uz/ut
        else:
            vR, vphi, vz = {}, {}, {}
            ut   = self.mid_val(uu_cyl[0])
            uR   = self.mid_val(uu_cyl[1])
            uphi = self.mid_val(uu_cyl[2])
            uz   = self.mid_val(uu_cyl[3])
            vR['r']   = uR / ut
            vphi['r'] = uphi / ut
            vz['r']   = uz / ut
            if not kwargs.get('in_out', False):
                return vR, vphi, vz
            else:
                in_out = {'r' : np.sign(uu[1, :, 0, :] + uu[1, :, 1, :])}
                return vR, vphi, vz, in_out

    def calc_uu_cart(self, **kwargs):
        uu = self.get_vars3D(['uu'])
        self.set_Coords3D()

        # Calculate uu in Cartesian coordinates
        uu_cart = self.Coords3D.int_to_cart_vec(uu) / self.c
        if not kwargs.get('lr', False):
            uux = (uu_cart[1][:, 0, :] + uu_cart[1][:, 1, :]) / 2.
            uuy = (uu_cart[2][:, 0, :] + uu_cart[2][:, 1, :]) / 2.
            return uux, uuy
        else:
            uux, uuy = {}, {}
            uux['r'] = (uu_cart[1][:, 0, :] + uu_cart[1][:, 1, :]) / 2.
            uuy['r'] = (uu_cart[2][:, 0, :] + uu_cart[2][:, 1, :]) / 2.
            if not kwargs.get('in_out', False):
                return uux, uuy
            else:
                in_out = {}
                in_out['r'] = np.sign(uu[1, :, 0, :] + uu[1, :, 1, :])
                return uux, uuy, in_out

    def calc_vpol_vtor(self):
        uu, gcov = self.get_vars3D(['uu', 'gcov'])
        self.set_Coords3D()

        uucyl = self.Coords3D.int_to_cyl_vec(uu, dtype=np.float64) 
        uupol = np.copy(uu)
        uutor = np.copy(uu)
        uupol[0], uupol[3]  = 0., 0.
        uutor[0], uutor[1], uutor[2] = 0., 0., 0.

        sgn     = np.sign(uucyl[3])
        v2pol   = ana.Tools.dot(uupol, uupol, gcov)
        v2tor   = ana.Tools.dot(uutor, uutor, gcov)
        vratio  = sgn * np.sqrt(v2pol / v2tor)
        return (vratio[:, 0, :] + vratio[:, 1, :]) / 2.
    
    # Radiation 4-velocity
    def calc_uu_rad_cart(self, **kwargs):
        uu_rad = self.get_vars3D(['uu_rad'])
        self.set_Coords3D()

        # Calculate uu_rad in Cartesian coordinates
        uuR_cart = self.Coords3D.int_to_cart_vec(uu_rad) / self.c
        if not kwargs.get('lr', False):
            uuRx = self.mid_val(uuR_cart[1])
            uuRy = self.mid_val(uuR_cart[2])
            return uuRx, uuRy
        else:
            uuRx, uuRy = {}, {}
            uuRx['r'] = self.mid_val(uuR_cart[1])
            uuRy['r'] = self.mid_val(uuR_cart[2])
            if not kwargs.get('in_out', False):
                return uuRx, uuRy
            else:
                in_out = {'r' : np.sign(uu_rad[1,:,0,:] + uu_rad[1,:,1,:])}
                return uuRx, uuRy, in_out

    # Cooling rates    
    # Inverse-Compton cooling rate
    def calc_qIC(self, **kwargs):
        return self.mid_val(super().calc_qIC(0, **kwargs))

    # Coulomb cooling rate
    def calc_qie(self, **kwargs):
        return self.mid_val(super().calc_qie(0, midplane=True, **kwargs))

    # Synchrotron cooling rate
    def calc_qsync(self, **kwargs):
        return self.mid_val(super().calc_qsync(0, **kwargs)) 
    
    # Radiation source term in the coordinate frame (G_t)
    # Note that the radiation is sourced by -G_t
    def calc_qrad(self):
        return self.mid_val(super().calc_qrad(0))
 
    # Cooling timescales
    def calc_teq(self, **kwargs):  
        return self.mid_val(super().calc_teq(0, midplane=True, **kwargs))
    
    def calc_tii(self, **kwargs):
        return self.mid_val(super().calc_tii(0, midplane=True, **kwargs))
    
    def calc_tee(self, **kwargs):
        return self.mid_val(super().calc_tee(0, midplane=True, **kwargs))
    
    def calc_tIC(self, **kwargs):   
        return self.mid_val(super().calc_tIC(0, midplane=True, **kwargs))
    
    # Orbital frequency
    def calc_OmegaK(self):
        return self.Analytic.calc_OmegaK(
            self.MBH, self.H.params['a'], self.r)[0, :, 0, :]

    # Opacities    
    def calc_kappa_eff(self, **kwargs):
        return self.mid_val(super().calc_kappa_eff(0, **kwargs))

    def calc_kappa_sync(self, **kwargs):
        return self.mid_val(super().calc_kappa_sync(0, **kwargs))

    def calc_kappa_ff(self, **kwargs):
        return self.mid_val(super().calc_kappa_ff(0, **kwargs))

class Load_Data():
    """
    Manages radial profiles and poloidal slices data products. 
    Slices, Profiles, Vertical_Profiles, Stresses, etc. classes all inherit
    methods from this parent class.
    """ 
    def _get_data_dir(self, averages_dir, file_type, reduced):
        """Helper to build the standard data directory path."""
        resolution = "Reduced_Resolution" if reduced else "Full_Resolution"
        return f"{averages_dir}/{resolution}/{file_type}" 

    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        """Retrieve file numbers based on file_number, start_file, and end_file."""
        file_type = kwargs.get('file_type', 'Profiles')
        reduced   = kwargs.get('reduced', False)
        data_dir  = self._get_data_dir(averages_dir, file_type, reduced)
        
        # Single file selection
        if file_number != -1:
            return [file_number]
            
        # Scan directory and extract available file numbers safely
        names = glob.glob(f"{file_type}*", root_dir=data_dir)
        if not names:
            return []
            
        prefix = f"{file_type}_"
        nums = sorted([
            int(name.removeprefix(prefix).removesuffix('.npz')) 
            for name in names
        ])
        
        first = nums[0]  if start_file == -1 else start_file
        last  = nums[-1] if end_file   == -1 else end_file
        
        fnums = list(range(first, last + 1))
        return self.check_averaged_files_exist(fnums, nums, file_type)

    def check_averaged_files_exist(self, fnums, data_nums, file_type):
        """Filter out and report missing files from the requested fnums list."""
        data_set = set(data_nums)
        valid_fnums = []
        
        for fnum in fnums:
            if fnum in data_set:
                valid_fnums.append(fnum)
            else:
                print(f"{file_type} {fnum} is missing.")
                
        return valid_fnums

    def get_last_file_num(self, averages_dir, file_type, **kwargs):
        """Determine the number of the last file."""
        reduced  = kwargs.get('reduced', False)
        data_dir = self._get_data_dir(averages_dir, file_type, reduced)
        names    = glob.glob(f"{file_type}*", root_dir=data_dir)
        
        prefix = f"{file_type}_"
        nums = [
            int(name.removeprefix(prefix).removesuffix('.npz')) 
            for name in names
        ]
        return max(nums)

    def load_data(self, averages_dir, fnums, **kwargs):
        """
        Create or update a slices/profiles dictionary. 
    
        Parameters
        ----------
        averages_dir : str
            Directory where averaged data is stored (e.g. Averages/Simulation).
        fnums : list of int
            File numbers to place into slices/profiles dictionary.
        **kwargs : dict
            Optional configurations. Accepted keys include:

            file_type : str
                File type used for the data product 
                (e.g. Profiles, Slices, Stresses, etc.)
            reduced : bool
                Use reduced data? (Default: False).
            show_loading : bool
                Whether or not to show "Loading file fnum ..."
                on the screen. This is used for reducing screen clutter.
            keys : list of str
                List of variable keys to be added to the slices/profiles
                dictionary.
            profiles : dict
                Profiles dictionary already in format [file_number][key].
                This allows an existing dictionary to be updated.
            slices : dict
                Slices dictionary already in format [file_number][key].
                This allows an existing dictionary to be updated.
        
        Returns
        -------
        dict
            Dictionary of profiles (1D) or slice (2D) arrays in format
            profiles[file_number][key].
        """
        file_type      = kwargs.get('file_type', 'Profiles')
        reduced        = kwargs.get('reduced', False)
        show_loading   = kwargs.get('show_loading', True)
        requested_keys = kwargs.get('keys', None)
        
        data_dir = self._get_data_dir(averages_dir, file_type, reduced)
        if (show_loading):
            print(f"Loading from {data_dir}...")
        
        # Determine zero-padding width from existing files
        names = glob.glob(f"{file_type}*", root_dir=data_dir)
        if not names:
            raise FileNotFoundError(
                f"No files matching '{file_type}*' found in {data_dir}")
        prefix = f"{file_type}_"
        nzeros = max([
            len(name.removeprefix(prefix).removesuffix('.npz')) 
            for name in names
        ])

        # Either load in existing data dictionary or create empty one
        data = kwargs.get('profiles', kwargs.get('slices', {}))
        
        # Set up quick conditional flags to keep the inner loop fast
        slice_objs   = {'Slices', 'Heating_Slices', 'Stresses'}
        print2screen = show_loading and file_type in slice_objs

        for fnum in fnums:
            if print2screen:
                print(f"File number {fnum}...", flush=True)
                
            data_file = f"{data_dir}/{file_type}_{fnum:0{nzeros}d}.npz"
            fnum_str  = str(fnum)
            
            with np.load(data_file) as f:
                if requested_keys:
                    if fnum_str not in data:
                        data[fnum_str] = {}
                    
                    for key in requested_keys:
                        data[fnum_str][key] = f[key] 
                else:
                    # Grab all keys
                    data[fnum_str] = dict(f.items())
                    
        return data

    def calc_dt(self, averages_dir, fnums, **kwargs):
        """Calculate dt weights for time averaging."""
        if not kwargs.get('dt_weight', True):
            return np.ones(len(fnums))
            
        file_type = kwargs.get('file_type', 'Profiles')
        
        # Route based on whether it's a slice type or profile type
        if 'Slices' in file_type or 'Stresses' in file_type:
            return self.calc_dt_slices(averages_dir, fnums, **kwargs)

        return self.calc_dt_profs(averages_dir, fnums, **kwargs)

    def calc_dt_profs(self, averages_dir, fnums, **kwargs):
        """Calculate dt weights specifically for profiles."""
        if 'profiles' not in kwargs:
            raise ValueError("Must supply profiles to use calc_dt_profs.")
        kwargs.setdefault('file_type', 'Profiles')
        kwargs.setdefault('show_loading', False)
        return self._calc_dt_uni(averages_dir, fnums, 
                                 data_key='profiles', **kwargs)

    def calc_dt_slices(self, averages_dir, fnums, **kwargs):
        """Calculate dt weights specifically for slices."""
        if 'slices' not in kwargs:
            raise ValueError("Must supply slices to use calc_dt_slices.")
        kwargs.setdefault('file_type', 'Slices')
        return self._calc_dt_uni(averages_dir, fnums, 
                                 data_key='slices', **kwargs)

    def _calc_dt_uni(self, averages_dir, fnums, data_key, **kwargs):
        """Unified methods to calculate dt for both profiles and slices."""
        file_type = kwargs['file_type']
        reduced   = kwargs.get('reduced', False)
        data_dict = kwargs.pop(data_key)

        TS        = Time_Series()
        max_fnum  = int(max(fnums))
        last_file = self.get_last_file_num(
            averages_dir, file_type, reduced=reduced)

        # Remove keys if passed since we are loading data again.
        kwargs.pop('keys', None)
        
        if max_fnum != last_file:
            # Look ahead to find the next available file
            file_list = list(range(max_fnum + 1, last_file + 1))
            # Note this is the global function not from self
            nxt_file  = check_files_exist(averages_dir, file_list, 
                file_type=file_type, reduced=reduced)[0]
            
            # Load the timestamp 't' for the next file if not already present
            if 't' not in data_dict[str(fnums[0])]:
                time_dict = self.load_data(averages_dir, fnums + [nxt_file], 
                                           keys=['t'], **kwargs)
            else:
                kwargs['show_loading'] = False
                time_dict = dict.fromkeys(data_dict.keys())
                for fnum in time_dict.keys():
                    time_dict[fnum] = {'t' : data_dict[fnum]['t']}
                kwargs[data_key] = time_dict
                time_dict = self.load_data(averages_dir, [nxt_file], 
                                           keys=['t'], **kwargs)
                 
            _, time = TS.calc_time_series('t', time_dict)
            dt = np.diff(time)
        else:
            # If at the last file, duplicate the final dt
            if 't' not in data_dict[str(fnums[0])]:
                time_dict = self.load_data(
                    averages_dir, fnums, keys=['t'], **kwargs)
            else:
                time_dict = dict.fromkeys(data_dict.keys())
                for fnum in time_dict.keys():
                    time_dict[fnum] = {'t' : data_dict[fnum]['t']}
                
            _, time = TS.calc_time_series('t', time_dict)
            
            if len(time) == 1:
                dt = np.array([1.0])
            else:
                dt = np.diff(time)
                dt = np.insert(dt, -1, dt[-1]) # Duplicate dt[-1]
        
        return dt

class Slices(Load_Data):
    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        """Retrieve file numbers based on file_number, start_file, and end file."""
        return super().get_fnums(averages_dir, file_number, start_file, end_file, 
                                 file_type='Slices', **kwargs)

    def load_slices(self, averages_dir, fnums, **kwargs):
        """Create slices dictionary: slices[file_number][key]."""
        return super().load_data(averages_dir, fnums, 
                                 file_type='Slices', **kwargs)

    def load_slices_time_averaged(self, averages_dir, fnums, **kwargs):
        """Create slices dictionary (time-averaged) slices[key]."""
        reduced   = kwargs.pop('reduced', False)
        file_type = kwargs.pop('file_type', 'Slices')
        
        # Pull coordinate/time/spin information for all file numbers
        coords = self.load_slices(averages_dir, fnums,
            keys=['t', 'r', 'theta', 'phi', 'a', 'dx1', 'dx2'],
            reduced=reduced, show_loading=False)
        # Stash per-file times
        self.t = np.array([coords[str(fnum)]['t'] for fnum in fnums])
        # Determine if resolution changes between any file numbers
        self.check_AMR(fnums, coords)
        
        # Keys and whether to divide by volume required for time averaging
        requested_keys = kwargs.get('keys')
        vols           = kwargs.get('vols')
        if not requested_keys:
            raise ValueError("Must supply keys for time averaging.")
        if not vols:
            raise ValueError("Must supply vols for time averaging.")

        # Pre-allocate time-averaged slices dictionary
        max_shape = (max(self.nr), max(self.nth))
        slices = {key: np.zeros(max_shape) for key in requested_keys}

        # Calculate time weighting
        # Time average is <w>_t = int w dt / int dt if dt_weight=True (default)
        # otherwise the time average is <w>_t = sum w / n_files
        dt = self.calc_dt(averages_dir, fnums, slices=coords, 
                          reduced=reduced, file_type=file_type, **kwargs)
        dT = np.sum(dt)

        # Build slice (s) keys
        skeys = list(requested_keys)
        if 'r' not in skeys and 'theta' not in skeys:
            skeys.extend(['r', 'theta'])
        if 'Vol' not in skeys and any(vols.values()):
            skeys.append('Vol')

        # Clean up kwargs so keys doesn't conflict during underlying file loads
        kwargs_clean = kwargs.copy()
        kwargs_clean.pop('keys')

        for i, fnum in enumerate(fnums):
            # Load the single time frame data
            single_slice = self.load_slices(averages_dir, [fnum], keys=skeys, 
                                            reduced=reduced, **kwargs_clean)
            
            dimensionless = kwargs.get('dedim')
            if dimensionless is not None:
                single_slice = self.remove_units(single_slice, [fnum], 
                                                 reduced, averages_dir) 

            for key in requested_keys:
                # "Time average" takes care of upscaling for single file number
                slices[key] += (dt[i] / dT) * self.time_avg(
                    key, single_slice, vol=vols[key], dt=dt[i], 
                    run_check_AMR=False
                )

        # Store coordinate information from file at maximum refinement level.
        # 'r' clipped at r_trim if --r_max cuts at odd number of zones.
        r_trim = self.r_trim.get(self.fnums0, 0)
        for coord_key in ['r', 'theta', 'phi', 'a', 'dx1', 'dx2']:
            coord = coords[str(self.fnums0)][coord_key]
            if coord_key == 'r' and r_trim:
                coord = coord[:coord.shape[0] - r_trim]
            slices[coord_key] = coord
            
        # Store dt's for time update
        slices['dt'] = dt

        # Downsample for speed if requested (will not work with update_slices)
        lr1 = kwargs.get('lowres1')
        lr2 = kwargs.get('lowres2')
        if lr1 or lr2:
            for key in requested_keys:
                slices[key] = slices[key][::lr1, ::lr2]
            slices['r']     = slices['r'][::lr1]
            slices['theta'] = slices['theta'][::lr2]

        if kwargs.get('dedim') is not None:
            slices['dedim'] = True
        return slices

    def remove_units(self, slice, fnums, reduced, averages_dir):
        ONE    = np.float32(1.)
        FOURPI = np.float32(4. * np.pi)
        for fnum in fnums:
            rho0_info = self.load_slices(averages_dir, [fnum], keys=['rho0'], 
                                        reduced=reduced, show_loading=False)
            fnum = str(fnum)
            rho0 = rho0_info[fnum]['rho0']
            c    = hr.CONSTANTS['c']
            B0   = np.sqrt(FOURPI) * np.sqrt(rho0) * c 
            keys = [
                'wuR',  'wuz', 'wu_phi', 'wur', 'wuth',
                'bR',   'bz',  'b_phi',  'br',  'bth', 'w'
            ]
            slice_keys = slice[fnum].keys()
            valid_keys = [key for key in keys if key in slice_keys]
            for key in valid_keys:
                if 'wu' in key:
                    slice[fnum][key] *= ONE / (rho0 * c * c * c)
                if 'b' in key:
                    slice[fnum][key] *= ONE / B0
            if 'w' in valid_keys:
                slice[fnum]['w'] *= ONE / (rho0 * c * c)
        return slice

    def check_AMR(self, fnums, slices):
        """Check if refinement levels changed between slices."""
        self.nums = list(fnums)
        self.nr   = [len(slices[str(fnum)]['r'])     for fnum in fnums]
        self.nth  = [len(slices[str(fnum)]['theta']) for fnum in fnums]

        # When --r_max is used, radial cell counts aren't always exact
        # powers of 2. In that instance, clip off the end radial zones
        # that prevent AMR upscaling. 
        n_min = min(self.nr)
        self.r_trim = {}
        for fnum, nr in zip(self.nums, self.nr):
            level  = round(math.log2(nr / n_min)) if nr > n_min else 0
            target = n_min * (2 ** level)
            self.r_trim[fnum] = max(nr - target, 0)
        self.nr = [nr - self.r_trim[fnum] for fnum, nr in zip(self.nums, self.nr)]

        # Check if grid resolutions vary across file numbers
        self.AMRx1 = len(set(self.nr))  > 1
        self.AMRx2 = len(set(self.nth)) > 1

        if self.AMRx1 or self.AMRx2:
            # Find file number matching the maximum spatial dimensions
            target_sizes = np.array(self.nr if self.AMRx1 else self.nth)
            max_size     = max(target_sizes)
            
            # Filter file numbers matching max resolution
            nr_eq_max   = np.array(self.nums)[target_sizes == max_size]
            self.fnums0 = nr_eq_max[0] if nr_eq_max.size > 0 else nr_eq_max
        else:
            # Grid resolution is completely uniform
            self.fnums0 = self.nums[0]

    def time_avg(self, quant, slices, **kwargs):
        """
        Compute volume and time-weighted averages. Supports both single-frame 
        inline processing and multi-frame standalone execution.
        """
        fnums = kwargs.get('fnums', sorted([int(k) for k in slices.keys()]))
        if kwargs.get('run_check_AMR', True):
            self.check_AMR(fnums, slices)
 
        max_shape = (max(self.nr), max(self.nth))
        q_avg = np.zeros(max_shape)
        
        # Supports both single frames for use in load_slice_time_averaged
        # and multiple frames for standalone use.
        dt = np.atleast_1d(kwargs.get('dt', 1))
        if 'dt' not in kwargs or 'dT' not in kwargs:
            if 'dt' not in kwargs:
                dt = np.ones(len(fnums))
            dT = np.sum(dt)
        else:
            dT = kwargs.get('dT')
            
        use_vol = kwargs.get('vol', False)
        has_amr = self.AMRx1 or self.AMRx2

        for i, fnum in enumerate(fnums):
            fnum_str = str(fnum)
            
            q   = slices[fnum_str][quant]
            vol = slices[fnum_str]['Vol'] if use_vol else 1.0

            trim = self.r_trim.get(fnum, 0)
            if trim:
                q = q[:q.shape[0] - trim]
                if use_vol:
                    vol = vol[:vol.shape[0] - trim]

            if has_amr:
                q = self.upscale(q)
                if use_vol:
                    vol = self.upscale(vol)
            
            q_avg += (dt[i] / dT) * (q / vol)
            
        return q_avg

    def upscale(self, q, **kwargs):
        """Upscale 2D array using vectorized block replication."""
        max_nr  = kwargs.get('max_nr')
        max_nth = kwargs.get('max_nth')
        
        if max_nr  is None: max_nr  = max(self.nr)
        if max_nth is None: max_nth = max(self.nth)
        
        # Determine upscale scaling factors
        scale_x1 = int(max_nr  / q.shape[0])
        scale_x2 = int(max_nth / q.shape[1])
        
        if scale_x1 == 1 and scale_x2 == 1:
            return q
            
        # Vectorized block expansion via NumPy tiling
        return np.repeat(np.repeat(q, scale_x1, axis=0), scale_x2, axis=1)

    def downscale(self, q, **kwargs):
        """
        Downscale a 2D array using vectorized local block averaging 
        (AMR Restriction).
        """
        min_nr  = kwargs.get('min_nr')
        min_nth = kwargs.get('min_nth')
        if min_nr  is None: min_nr  = min(self.nr)
        if min_nth is None: min_nth = min(self.nth)

        current_nr, current_nth = q.shape
        
        # Calculate scaling factors
        scale_x1 = int(current_nr / min_nr)
        scale_x2 = int(current_nth / min_nth)
        
        # If already at the target shape or smaller, return as-is
        if scale_x1 <= 1 and scale_x2 <= 1:
            return q
            
        # Vectorized block averaging via reshaping and taking the mean
        # Blocks of (target_rows, scale_x1, target_cols, scale_x2)
        reshaped = q.reshape(min_nr, scale_x1, min_nth, scale_x2)
        
        # Average across the spatial scaling axes (axes 1 and 3)
        return reshaped.mean(axis=(1, 3))
 
    def update_slices(self, averages_dir, fnums, slices, **kwargs):
        """
        Update time-averaged slices by subtracting old frames and 
        adding new ones.
        """
        reduced   = kwargs.get('reduced', False)
        file_type = kwargs.get('file_type', 'Slices')
        fnums_old = kwargs.get('fnums_old', fnums)
        
        # Pull parameters for rolling window updates
        sub_fnum   = str(fnums_old[0])
        add_fnum   = str(fnums[-1])
        load_fnums = [fnums_old[0], fnums[-1]]
        
        # Enforce validation guards upfront
        vols = kwargs.get('vols')
        if vols is None:
            raise ValueError("Cannot update slice without passing vols dict.")
        if 'dt' not in slices:
            raise ValueError("Must supply dt to use update_slices")
            
        dt_old = np.atleast_1d(slices['dt'])
        dT_old = np.sum(dt_old)

        if len(fnums_old) == len(fnums):
            # Calculate updated time-weighting metrics
            if kwargs.get('dt_weight', True):
                last_file = self.get_last_file_num(
                    averages_dir, file_type, reduced=reduced)
                
                next_fnum = int(fnums[-1] + 1)
                exist_args = {
                    'path_to_averages' : averages_dir, 
                    'file_list'        : [next_fnum],
                    'file_type'        : file_type,
                    'reduced'          : reduced
                }
                has_next = fnums[-1] != last_file and check_files_exist(**exist_args)
                t_fnums  = fnums + [next_fnum] if has_next else fnums
                coords   = self.load_slices(averages_dir, t_fnums, keys=['t'], 
                                            reduced=reduced, show_loading=False)
                
                TS = Time_Series()
                _, time = TS.calc_time_series('t', coords)
                
                dt_new = np.diff(time)
                if not has_next:
                    dt_new = np.insert(dt_new, -1, dt_new[-1])
            else:
                dt_new = dt_old
                
            dT_new = np.sum(dt_new)

            # Retrieve arrays to swap out and swap in
            requested_keys = kwargs.get('keys', list(slices.keys()))
            skeys          = requested_keys.copy()
            if 'r' not in skeys and 'theta' not in skeys:
                skeys.extend(['r', 'theta'])

            switch_slices = self.load_slices(averages_dir, load_fnums, keys=skeys)
            
            dimensionless = kwargs.get('dedim')
            if dimensionless is not None:
                switch_slices = self.remove_units(switch_slices, load_fnums, 
                                                  reduced, averages_dir)
            
            # Determine maximum resolution among current, sub, and add slices
            self.check_AMR(load_fnums, switch_slices)
            max_nr  = max(max(self.nr),  slices[requested_keys[0]].shape[0])
            max_nth = max(max(self.nth), slices[requested_keys[0]].shape[1])
            
            # Subtract out sub_frame and add in add_frame
            sub_frame = switch_slices[sub_fnum]
            add_frame = switch_slices[add_fnum]

            for key in requested_keys:
                if key in {'dt', 'r', 'theta', 'a', 'dx1', 'dx2'}:
                    continue  # Skip coordinate metadata frames
                
                use_vol = vols[key]
                sub_vol = sub_frame['Vol'] if use_vol else 1.0
                add_vol = add_frame['Vol'] if use_vol else 1.0
                
                # Check shapes and handle upscaling if they mismatch
                dt_sub  = dt_old[0]
                dt_add  = dt_new[-1]
                qdt_sub = (sub_frame[key] / sub_vol) * dt_sub
                qdt_add = (add_frame[key] / add_vol) * dt_add

                # Handle upscaling to highest-resolution target grid
                target_shape = (max_nr, max_nth)
                upscale_args = {'max_nr': max_nr, 'max_nth': max_nth} 
                
                if slices[key].shape != target_shape:
                    slices[key] = self.upscale(slices[key], **upscale_args)
                if qdt_sub.shape != target_shape:
                    qdt_sub = self.upscale(qdt_sub, **upscale_args)
                if qdt_add.shape != target_shape:
                    qdt_add = self.upscale(qdt_add, **upscale_args)

                # Execute sliding-window update in place
                slices[key] = (slices[key] * dT_old - qdt_sub + qdt_add) / dT_new
    
            slices['dt'] = dt_new
            
        return slices

class Custom_Slices(Slices):
    def __init__(self, file_type):
        self.file_type = file_type

    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        """Retrieve file numbers based on file_number, start_file, and end file."""
        return super(Slices, self).get_fnums(averages_dir, file_number, 
                                             start_file, end_file, 
                                             file_type=self.file_type, **kwargs)

    def load_slices(self, averages_dir, fnums, **kwargs):
        """Create slices dictionary custom_slices[file_number][key]."""
        return super(Slices, self).load_data(averages_dir, fnums, 
                                             file_type=self.file_type, **kwargs)
    
    def load_slices_time_averaged(self, averages_dir, fnums, **kwargs):
        """Create slices dictionary (time-averaged) custom_slices[key]."""
        kwargs.setdefault('file_type', self.file_type)
        return super().load_slices_time_averaged(averages_dir, fnums, **kwargs)
    
    def update_slices(self, averages_dir, fnums, slices, **kwargs):
        """
        Update time-averaged slices by subtracting old frames and 
        adding new ones.
        """
        kwargs.setdefault('file_type', self.file_type)
        return super().update_slices(averages_dir, fnums, slices, **kwargs)

class Stresses(Custom_Slices):
    def __init__(self):
        super().__init__(file_type='Stresses')

class Heating_Slices(Custom_Slices):
    def __init__(self):
        super().__init__(file_type='Heating_Slices')
        
class Load_Vertical_Data():
    def _gen_slices(self, averages_dir, fnums, **kwargs):
        """
        Generator to unify time-averaged and single-dump data layouts.
        Yields tuple of (fnum, data_dict). fnum=None means time-averaged.
        """
        time_avg = kwargs.get('time_average', False)
        slices   = kwargs.get('slices', None)

        if slices:
            self.fnums0 = 0
            keys = kwargs.get('keys', None)
            if keys is not None:
                slices = {k: slices[k] for k in (*keys, 'r', 'theta', 'a')}
            yield None, slices
        else:
            if time_avg:
                slices = self.load_slices_time_averaged(averages_dir, fnums, **kwargs)
                # Yield a single item with a None key to signify time-averaged.
                yield None, slices
            else:
                slices = self.load_slices(averages_dir, fnums, **kwargs)
                for fnum in slices.keys():
                    # Pauses after each fnum.
                    yield str(fnum), slices[str(fnum)]

    def load_theta_profiles(self, averages_dir, fnums, r0, **kwargs):
        """Load theta profiles at a given target radius r0."""
        profiles = {}
        time_avg = kwargs.get('time_average', False)

        for fnum_str, data in self._gen_slices(averages_dir, fnums, **kwargs):
            keys = [k for k in data.keys() if k not in ('r', 'theta')]

            out_dict = {}
            r   = data['r']
            idx = np.argmin(np.abs(r - r0))
            
            for key in keys:
                q = data[key]
                out_dict[key] = q[idx, :] if q.ndim == 2 else q
                
            out_dict['theta'] = data['theta']
            out_dict['r']     = r
            
            if time_avg:
                profiles = out_dict
            else:
                profiles[fnum_str] = out_dict
                
        return profiles

    def load_z_profiles(self, averages_dir, fnums, r0, **kwargs):
        """Interpolate data onto a uniform cylindrical Z-grid at radius r0."""
        npts = int(kwargs.get('npts', 100.))
        dR   = kwargs.get('dR', 0.1 * r0)
        
        # Grid setup
        R1D  = np.linspace(r0 - dR, r0 + dR, int(npts / 10) + 1)
        z1D  = np.linspace(-1. * r0, 1. * r0, int(100 * npts))
        R_uni, Z_uni = np.meshgrid(R1D, z1D)
        idx  = np.argmin(np.abs(R_uni[0, :] - r0))

        profiles = {}
        time_avg = kwargs.get('time_average', False)
        
        z_min, z_max = 1.1 * np.amin(z1D), 1.1 * np.amax(z1D)

        # Loop over file numbers or time-average.
        for fnum_str, data in self._gen_slices(averages_dir, fnums, **kwargs):
            keys = [k for k in data.keys() if k not in ('r', 'theta')]

            out_dict = {}
            r        = data['r']
            theta    = data['theta']

            R2D = r[:, None] * np.sin(theta[None, :])
            Z2D = r[:, None] * np.cos(theta[None, :])
                        
            R_cut = (R2D < 2. * r0) & (R2D > r0 - dR)
            Z_cut = (Z2D < z_max) & (Z2D > z_min)
            mask  = R_cut & Z_cut

            R_dat_flat = R2D[mask].ravel()
            Z_dat_flat = Z2D[mask].ravel()
                    
            grid_args = {
                'points' : (R_dat_flat, Z_dat_flat), 
                'xi'     : (R_uni, Z_uni), 
                'method' : 'cubic'
            }
            
            for key in keys:
                print(f"Loading {key}...", flush=True)
                q = data[key]
                
                if q.ndim == 2:
                    q_dat_flat = q[mask].ravel()
                    grid_args['values'] = q_dat_flat
                    q_uni = griddata(**grid_args)
                    out_dict[key] = q_uni[:, idx]
                else:
                    out_dict[key] = q
            
            out_dict['z_r']   = z1D / R1D[idx]
            out_dict['theta'] = theta
            out_dict['r']     = r
            
            if time_avg:
                profiles = out_dict
            else:
                profiles[fnum_str] = out_dict
                
        return profiles

class Vertical_Profiles(Load_Vertical_Data, Slices):
    pass

class Vertical_Stress_Profiles(Load_Vertical_Data, Stresses):
    pass

class Profiles(Load_Data):
    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        """Retrieve file numbers based on file_number, start_file, and end file."""
        return super().get_fnums(averages_dir, file_number, start_file, end_file, 
                                 file_type='Profiles', **kwargs)
    
    def load_profiles(self, averages_dir, fnums, **kwargs):
        """Create profiles dictionary: profiles[file_number][key]."""
        self.averages_dir = averages_dir
        self.reduced      = kwargs.get('reduced', False)
        return super().load_data(averages_dir, fnums, 
            file_type='Profiles', **kwargs)

    def time_avg(self, quant, profiles, **kwargs):
        """
        Compute volume and time-weighted averages. 
        """
        file_type = kwargs.get('file_type', 'Profiles')
        fnums = kwargs.get('fnums', sorted([int(k) for k in profiles.keys()]))
        self.check_AMR(fnums, profiles)
 
        q_avg = np.zeros(max(self.nr))
        
        # Time average is <w>_t = int w dt / int dt if dt_weight=True (default)
        # otherwise the time average is <w>_t = sum w / n_files
        dt = self.calc_dt(self.averages_dir, fnums, 
            file_type=file_type, reduced=self.reduced, profiles=profiles)
        dT = np.sum(dt)
        
        use_vol = kwargs.get('vol', False)
        has_amr = self.AMR

        for i, fnum in enumerate(fnums):
            fnum_str = str(fnum)
            q   = profiles[fnum_str][quant]
            vol = profiles[fnum_str]['Vol'] if use_vol else 1.0
            if has_amr:
                q = self.upscale(q)
                if use_vol:
                    vol = self.upscale(vol)
            
            q_avg += (dt[i] / dT) * (q / vol) 
        
        return q_avg
    
    def check_AMR(self, fnums, profiles):
        """Check if refinement levels changed between profiles."""
        self.nums = list(fnums)
        self.nr   = [len(profiles[str(fnum)]['r']) for fnum in fnums]
        
        # Check if grid resolutions vary across file numbers
        self.AMR = len(set(self.nr))  > 1
        
        if self.AMR:
            # Find file number matching the maximum spatial dimensions
            target_sizes = np.array(self.nr)
            max_size     = max(target_sizes)
            
            # Filter file numbers matching max resolution
            nr_eq_max   = np.array(self.nums)[target_sizes == max_size]
            self.fnums0 = nr_eq_max[0] if nr_eq_max.size > 0 else nr_eq_max
        else:
            # Grid resolution is completely uniform
            self.fnums0 = self.nums[0]
    
    def upscale(self, q, **kwargs):
        """Upscale 1D array using vectorized replication."""
        max_nr = kwargs.get('max_nr')
        if max_nr is None: max_nr = max(self.nr)
        
        # Determine upscale scaling factors
        scale_x1 = int(max_nr  / q.shape[0])
        
        if scale_x1 == 1:
            return q
            
        # Vectorized block expansion via NumPy tiling
        return np.repeat(q, scale_x1, axis=0)

    def downscale(self, q, **kwargs):
        """
        Downscale a 1D array using vectorized local averaging 
        (AMR Restriction).
        """
        min_nr  = kwargs.get('min_nr')
        if min_nr is None: min_nr = min(self.nr)
        current_nr = q.shape
         
        # Calculate scaling factors
        scale_x1 = int(current_nr / min_nr)
        
        # If already at the target shape or smaller, return as-is
        if scale_x1 <= 1:
            return q
            
        # Vectorized averaging via reshaping and taking the mean
        # Blocks of (target_rows, scale_x1)
        reshaped = q.reshape(min_nr, scale_x1)
        
        # Average across the spatial scaling axes (axis 1)
        return reshaped.mean(axis=1)

class Custom_Profiles(Profiles):
    def __init__(self, file_type):
        self.file_type = file_type

    def get_fnums(self, averages_dir, file_number, start_file, end_file, **kwargs):
        """Retrieve file numbers based on file_number, start_file, and end file."""
        return super(Profiles, self).get_fnums(averages_dir, file_number, 
                                               start_file, end_file, 
                                               file_type=self.file_type, **kwargs)

    def load_profiles(self, averages_dir, fnums, **kwargs):
        """Create profiles dictionary custom_profiles[file_number][key]."""
        self.averages_dir = averages_dir
        self.reduced      = kwargs.get('reduced', False)
        return super(Profiles, self).load_data(averages_dir, fnums, 
                                            file_type=self.file_type, **kwargs)

    def time_avg(self, quant, profiles, **kwargs):
        """
        Compute volume and time-weighted averages. 
        """
        kwargs.setdefault('file_type', self.file_type)
        return super().time_avg(quant, profiles, **kwargs)

class Heating_Profiles(Custom_Profiles):
    def __init__(self):
        super().__init__(file_type='Heating_Profiles')

class Time_Series(Profiles):
    def calc_time_series(self, key, profiles, **kwargs):
        fnums = list([int(fnum) for fnum in profiles.keys()])
        fnums.sort()
        time  = np.zeros(len(fnums))
        q     = np.zeros(len(fnums))
        r0    = kwargs.get('r0', 0.)
        isarr = sum((profiles[f'{fnums[0]}'][f'{key}']).shape)
        if isarr:
            for i, fnum in enumerate(fnums):
                r       = profiles[f'{fnum}']['r']
                idx     = np.argmin(np.abs(r - r0))
                time[i] = profiles[f'{fnum}']['t']
                vol     = (profiles[f'{fnum}']['Vol'] 
                           if kwargs.get('vol', False) else 1.)
                q[i]    = (profiles[f'{fnum}'][f'{key}']/vol)[idx]
        else:
            for i, fnum in enumerate(fnums):
                time[i] = profiles[f'{fnum}']['t']
                vol     = (profiles[f'{fnum}']['Vol'] 
                           if kwargs.get('vol', False) else 1.)
                q[i]    = (profiles[f'{fnum}'][f'{key}']/vol)
        return time, q