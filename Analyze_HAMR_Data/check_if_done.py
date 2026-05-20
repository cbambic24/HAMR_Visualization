import sys
import os
import glob
import argparse
import numpy as np
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{file_dir}')
import compute_variables as cv

def main(**kwargs):
    sim            = kwargs.get('sim')
    file_number    = int(kwargs.get('file_number'))
    start_file     = int(kwargs.get('start_file'))
    end_file       = int(kwargs.get('end_file'))
    reduced        = kwargs.get('reduced')
    check_profs    = kwargs.get('profiles')
    check_slices   = kwargs.get('slices')
    check_stress   = kwargs.get('stresses')
    check_heat     = kwargs.get('heating')
    check_all      = kwargs.get('all')
    check_custom   = kwargs.get('custom')

    if check_all:
        check_profs  = True
        check_slices = True
        check_stress = True

    # Determine averages directory
    averages_dir = cv.user_path + f'/Averages/{sim}'
    file_info = {
        'averages_dir' : averages_dir,
        'file_number'  : file_number,
        'start_file'   : start_file,
        'end_file'     : end_file,
        'reduced'      : reduced
    }
    
    keys = ['done', 'k_max', 'slice_num_restart']

    if (check_profs):
        TS         = cv.Time_Series()
        prof_fnums = TS.get_fnums(**file_info)
        prof_args = {
            'averages_dir' : averages_dir,
            'fnums'        : prof_fnums,
            'keys'         : keys,
            'reduced'      : reduced
        }
        profiles = TS.load_profiles(**prof_args)
        
        for fnum in prof_fnums:
            if not profiles[f'{fnum}']['done']:
                kmax = profiles[f'{fnum}']['k_max']
                snum = profiles[f'{fnum}']['slice_num_restart']
                done = profiles[f'{fnum}']['done']
                print(f"Profiles for file {fnum} are incomplete at " + 
                      f"k_max = {kmax} and slice number {snum} " +
                      f"with done flag {done}")
    
    if (check_slices):
        Slice       = cv.Slices() 
        slice_fnums = Slice.get_fnums(**file_info)
        slice_args = {
            'averages_dir' : averages_dir,
            'fnums'        : slice_fnums,
            'keys'         : keys,
            'reduced'      : reduced
        }
        slices = Slice.load_slices(**slice_args)
        for fnum in slice_fnums:
            if not slices[f'{fnum}']['done']:
                kmax = slices[f'{fnum}']['k_max']
                snum = slices[f'{fnum}']['slice_num_restart']
                done = slices[f'{fnum}']['done']
                print(f"Slices for file {fnum} are incomplete at " + 
                      f"k_max = {kmax} and slice number {snum} " +
                      f"with done flag {done}")
    
    if (check_stress):
        Stress       = cv.Stresses() 
        stress_fnums = Stress.get_fnums(**file_info)
        stress_args = {
            'averages_dir' : averages_dir,
            'fnums'        : stress_fnums,
            'keys'         : keys,
            'reduced'      : reduced
        }
        stresses = Stress.load_slices(**stress_args)
        for fnum in stress_fnums:
            if not stresses[f'{fnum}']['done']:
                kmax = stresses[f'{fnum}']['k_max']
                snum = stresses[f'{fnum}']['slice_num_restart']
                done = stresses[f'{fnum}']['done']
                print(f"Stresses for file {fnum} are incomplete at " + 
                      f"k_max = {kmax} and slice number {snum} " +
                      f"with done flag {done}")
    
    if (check_heat):
        Heating    = cv.Heating_Slices()  
        heat_fnums = Heating.get_fnums(**file_info)
        heat_args = {
            'averages_dir' : averages_dir,
            'fnums'        : heat_fnums,
            'keys'         : keys,
            'reduced'      : reduced
        }
        htg_slices = Heating.load_slices(**heat_args)
        for fnum in heat_fnums:
            if not htg_slices[f'{fnum}']['done']:
                kmax = htg_slices[f'{fnum}']['k_max']
                snum = htg_slices[f'{fnum}']['slice_num_restart']
                done = htg_slices[f'{fnum}']['done']
                print(f"Heating Slices for file {fnum} are incomplete at " + 
                      f"k_max = {kmax} and slice number {snum} " +
                      f"with done flag {done}")
    
    if (check_custom):
        Rad_Slice    = cv.Custom_Slices('RIAF_Slices') 
        custom_fnums = Rad_Slice.get_fnums(**file_info)
        custom_args = {
            'averages_dir' : averages_dir,
            'fnums'        : custom_fnums,
            'keys'         : keys,
            'reduced'      : reduced
        }
        rad_slices = Rad_Slice.load_slices(**custom_args)
        for fnum in custom_fnums:
            if not rad_slices[f'{fnum}']['done']:
                kmax = rad_slices[f'{fnum}']['k_max']
                snum = rad_slices[f'{fnum}']['slice_num_restart']
                done = rad_slices[f'{fnum}']['done']
                print(f"Custom Slices for file {fnum} are incomplete at " + 
                      f"k_max = {kmax} and slice number {snum} " +
                      f"with done flag {done}")

if __name__ == '__main__':
    parser  = argparse.ArgumentParser(
                        description="""Check if data has been extracted completely.""")
    parser.add_argument('sim', type=str, help = """simulation name""")
    parser.add_argument('--file_number', type=int, required=False, 
                        default=-1, help = """dump number of profile plot""")
    parser.add_argument('--start_file', type=int, required=False,
                        default=-1, help = """dump number
                        of start file in profile average/evolution""")
    parser.add_argument('--end_file', type=int, required=False,
                        default=-1, help = """dump number
                        of end file in profile average/evolution""")
    parser.add_argument('--reduced', action='store_true', required=False,
                        help="""boolean to determine if profile extraction
                        should be done on reduced data.""")
    parser.add_argument('--profiles', action=argparse.BooleanOptionalAction,
                        default=False, help="""check radial profiles?""")
    parser.add_argument('--slices',   action=argparse.BooleanOptionalAction,
                        default=False, help="""check poloidal slices?""")
    parser.add_argument('--stresses', action=argparse.BooleanOptionalAction,
                        default=False, help="""check stress slices?""")
    parser.add_argument('--heating', action=argparse.BooleanOptionalAction,
                        default=False, help="""check heating slices?""")
    parser.add_argument('--all',     action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""check profiles + slices + stresses?""")
    parser.add_argument('--custom',     action=argparse.BooleanOptionalAction,
                        default=False, 
                        help="""check custom slices Rti_Fluid??""")
    args = parser.parse_args()
    main(**vars(args))

    '''
        save_dir = cv.user_path + '/PDF'
    Stress = cv.Stresses() 
    stress_fnums = Stress.get_fnums(**file_info)
    keys     = ['k_max', 'slice_num_restart']
    stresses = Stress.load_slices(averages_dir, stress_fnums, keys=keys, reduced=reduced)
    for fnum in stress_fnums:
        if not stresses[f'{fnum}']['slice_num_restart'] == 8:
            print(f"Stresses for file {fnum} are incomplete at k_max = {stresses[f'{fnum}']['k_max']} and slice number {stresses[f'{fnum}']['slice_num_restart']}")

    '''