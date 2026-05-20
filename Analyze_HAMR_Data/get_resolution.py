import sys
import os
import glob
import argparse
import numpy as np
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{file_dir}')
asys_path = os.path.join(os.path.dirname(file_dir), "Analyze_HAMR_Data")
sys.path.insert(0, f'{asys_path}')
import compute_variables as cv
import hamr_read as hr

def main(**kwargs):
    sim       = kwargs.get('sim')
    start_file= int(kwargs.get('start_file'))
    end_file  = int(kwargs.get('end_file'))
    reduced   = kwargs.get('reduced')
    shared    = kwargs.get('shared')
    title     = kwargs.get('title')
    radiation = kwargs.get('radiation')
    two_temp  = kwargs.get('two_temp')
    
    cgs = True if (two_temp or radiation) else False

    # Determine data extraction directory
    data_dir = cv.get_data_dir(sim, reduced, shared)

    # Get total file list
    size, rank = 1, 0
    file_list = list(range(start_file, end_file + 1))
    file_list_local = cv.get_file_list_local(file_list, size, rank, data_dir)
    fnums, nr, nth, nphi, t = [], [], [], [], []
    for file_num in file_list_local:
        read_args = {
            'cgs_grid' : cgs,
            'cgs_data' : cgs,
            'coords1D' : True,
            'MBH'      : 10.
        }        
        
        H = hr.Read(data_dir, **read_args)
        H.read_params(file_num)

        fnums.append(file_num)
        nr.append(  int(H.params['nb1'] * H.params['bs1'] * H.upscalex1 / H.lowres1))
        nth.append( int(H.params['nb2'] * H.params['bs2'] * H.upscalex2 / H.lowres2))
        nphi.append(int(H.params['nb3'] * H.params['bs3'] * H.upscalex3 / H.lowres3))
        t.append(H.params['t'])

    AMRx1 = (nr.count(  nr[0])   != len(nr))
    AMRx2 = (nth.count( nth[0])  != len(nth))
    AMRx3 = (nphi.count(nphi[0]) != len(nphi))
    
    nr    = np.array(nr)
    nth   = np.array(nth)
    nphi  = np.array(nphi)
    fnums = np.array(fnums)
    t     = np.array(t)

    if (AMRx1):
        nr_set = set(nr)
        nr_change = []
        for nr0 in nr_set:
            target_fnums = fnums[nr == (len(nr) * [nr0])]
            try:
                len_target = len(target_fnums)
            except:
                len_target = 0
            nr_change.append(target_fnums[0] if len_target != 0 else target_fnums)
    
    if (AMRx2):
        nth_set = set(nth)
        nth_change = []
        for nth0 in nth_set:
            target_fnums = fnums[nth == (len(nth) * [nth0])]
            try:
                len_target = len(target_fnums)
            except:
                len_target = 0
            nth_change.append(target_fnums[0] if len_target != 0 else target_fnums)
    
    if (AMRx3):
        nphi_set = set(nphi)
        nphi_change = []
        for nphi0 in nphi_set:
            target_fnums = fnums[nphi == (len(nphi) * [nphi0])]
            try:
                len_target = len(target_fnums)
            except:
                len_target = 0
            nphi_change.append(target_fnums[0] if len_target != 0 else target_fnums)

    if (AMRx1):
        for i, nr0 in enumerate(nr_set):
            print(f"r-resolution at {nr0} at dump {nr_change[i]}.")
    if (AMRx2):
        for i, nth0 in enumerate(nth_set):
            print(f"theta-resolution at {nth0} at dump {nth_change[i]}.")
    if (AMRx3):
        for i, nphi0 in enumerate(nphi_set):
            print(f"phi-resolution at {nphi0} at dump {nphi_change[i]}.")
    
    if not AMRx1 and not AMRx2 and not AMRx3:
        print(f"Resolution of simulation is {nr[0]} x {nth[0]} x {nphi[0]}.")

    tstart = np.amin(t) / 1.e4
    tfinal = np.amax(t) / 1.e4
    duration = tfinal - tstart

    print(f"The start simulation time is {tstart} x 10^4 Rg/c")
    print(f"The final simulation time is {tfinal} x 10^4 Rg/c")
    print(f"The simulation duration is {duration} x 10^4 Rg/c")

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Get the resolution of all simulations""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('start_file', type=int, help = """dump number
                        of start file in profile extraction""")
    parser.add_argument('end_file', type=int, help = """dump number
                        of end file in profile extraction""")
    parser.add_argument('--radiation', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if radiation quantities
                        should be included in profile extraction.""")
    parser.add_argument('--two_temp', action=argparse.BooleanOptionalAction,
                        default=True, 
                        help="""boolean to determine if two-temperature 
                        physics should be included in profile extraction.""")     
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if profile extraction
                        should be done on reduced data.""")
    parser.add_argument('--shared', action='store_true', required=False, 
                        help="""boolean set to determine if proj-shared 
                        should be used as the location of the data directory""")
    parser.add_argument('--title', type=str, required=False, default=None, 
                        help="""title of simulation used for Averages directory
                        if different from the simulation title in proj-shared""")
    args = parser.parse_args()
    main(**vars(args))