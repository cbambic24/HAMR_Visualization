# H-AMR Analysis

This is a beta version of the analysis pipeline for _H-AMR_ ([Liska et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJS..263...26L/abstract)), a new GPU-accelerated general relativistic magnetohydrodynamics (GRMHD) code optimized to leverage exascale GPU-computing architectures.

_H-AMR_'s 3D adaptive mesh refinement (AMR) and adaptive time-stepping capabilities enable the code to perform state-of-the-art computations of accretion onto black holes at some of the highest resolutions achieved to date (e.g. [Liska et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474L..81L/abstract), [Ripperda et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...924L..32R/abstract)). However, the enormous data sets produced by _H-AMR_ require a unique read-in pipeline that can interpret the custom binary formats used for optimal memory storage on modern supercomputers.

This pipeline, written in Python and interfaced with C++ through a Cython-based API, allows the user to read in _H-AMR_ binary data and upscale the data to the highest level of mesh refinement via a nearest-neighbor interpolation scheme. More advanced interpolation routines, specifically tailored to allow post-facto ray-tracing of the data, are currently under development. The pipeline provides support for many of _H-AMR_'s physics modules, including M1 radiation transport ([Sadowski et al. 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.429.3533S/abstract), [McKinney et al. 2014](https://ui.adsabs.harvard.edu/abs/2014MNRAS.441.3177M/abstract), [Liska et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...944L..48L/abstract)), "two-temperature" thermodynamics ([Ressler et al. 2015](https://ui.adsabs.harvard.edu/abs/2015MNRAS.454.1848R/abstract); [Sadowski et al. 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..705S/abstract); [Liska et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...935L...1L/abstract)) which separately evolves ion and electron temperatures according to sub-grid schemes for allocating dissipated energy between the species, a photon-number conserving treatment of Comptonization ([Sadowski & Narayan 2015](https://ui.adsabs.harvard.edu/abs/2015MNRAS.454.2372S/abstract); [Liska et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...935L...1L/abstract)), resistive GRMHD schemes that separately evolve the electric field and magnetic field (e.g. [Ripperda et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJS..244...10R/abstract)), passive-scalar "tracer" fluids, and M1 neutrino transport ([Issa et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv241002852I/abstract)). The pipeline can be easily extended to handle additional physics modules, many of which are currently under active development.  

## Getting Started 

To use this pipeline, clone the repository with 
```
git clone https://github.com/cbambic24/HAMR_Analysis.git
```
The pipeline assumes that you have a main directory (e.g. __RUN__) where your analysis scripts are stored. While sample analysis and plotting routines are provided in _analysis.py_ and _figs.py_, the user is strongly encouraged to develop their own scripts. Within the __RUN__ directory, the pipeline assumes that the scripts _hamr_read.cpp_, _hamr_read.h_, _hamr_read_c.pyx_, _hamr_read.py_, and _setup.py_ are all stored in __RUN/HAMR_Visualization/Read_HAMR_Data__. Changing this directory structure is straightforward so long as  _hamr_read.cpp_, _hamr_read.h_, _hamr_read_c.pyx_, and _hamr_read.py_ are in the same directory as _setup.py_.

To compile the code, either execute the following command in the command line from the __RUN__ directory,
```
python HAMR_Visualization/Read_HAMR_Data/setup.py build_ext --inplace
```
or if executing the code from a Jupyter notebook situated in the __RUN__ directory, use the command
```
%run -i HAMR_Visualization/Read_HAMR_Data/setup.py build_ext --inplace
```

### Prerequisites

The pipeline has primarily been tested in a Windows environment, so you may run into issues if you're using a Linux or Mac OS. For a Linux OS, the main modifications are to the _setup.py_ script. A known combination that works in the _setup.py_ script is
```
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("hamr_readc",
                           sources=[read_dir + '/' + 'hamr_read_c.pyx',
                                    read_dir + '/' + 'hamr_read.cpp'],
                           language="c++",
                           include_dirs=[np.get_include()],
                           extra_compile_args=["-fopenmp", "-std=c++17",
                                               "-Wno-unused-result",
                                               "-Wno-deprecated-declarations"],
                           extra_link_args=["-fopenmp"])
```
However, you may run into issues with OpenMP (used to parallelize the read-in of large data files), particularly for the Mac OS. I am working on fixing this.

Another potential issue is that you could arrive at a segmentation fault while running the code because of using an older C++ compiler (e.g. older versions of g++ or MSVC) that is not fully compliant with C++17. Here, the problem originated with the std::file_system library, which was only introduced in recent versions of C++. You can circumvent this issue by using the more basic C command for checking if a file exists,
```
if(access(filename, 0) != -1)
```
where filename is an appropriately defined string for the file you're looking for. This can serve as a temporary work-around if you are in an environment where you cannot upgrade your g++ compiler to comply with modern C++ standards. 

### Reading in a Data Set

Suppose you were interested in loading in a state-of-the-art two-temperature radiation GRMHD simulation stored in __RUN/DATA/Exciting_Simulation__. To read in output 100 (or in _H-AMR_ parlance, dump 100) with your Jupyter notebook or Python script situated in __RUN__, you would execute
```
import sys
import os
import numpy as np
sys.path.insert(0, 'PATH_TO_RUN/RUN/HAMR_Visualization/Read_HAMR_Data')
import hamr_read as hr

H = hr.Read('DATA/Exciting_Simulation')
H.read_grid(100, direct=True)
H.read_prims(100, skip_read_params=True)
```
The flags "direct=True" and "skip_read_params=True" are unnecessary if you have properly written grid dump (gdumps) files in __RUN/DATA/Exciting_Simulation/gdumps__. Otherwise, the "direct=True" flag implies that you would like to analytically compute the Kerr metric for a spinning black hole in spherical Kerr-Schild coordinates without the additional step of waiting for the grid to be read in. The "skip_read_params=True" flag is a convenience that prevents read_prims from doing double-work and re-reading in the parameters file when read_grid has already read-in and stored the parameters. 

Additional flags/options available to the Read class can be found in the init() function in _hamr_read.py_. The user is encouraged to explore these settings and consult the Wiki on this repository for further instructions. Finally, for immediate assistance, please consult the parameter glossary included at the bottom of _hamr_read.py_. 

### Using the data

The user should only need access to the following dictionaries, where we assume that hamr_read was imported as "hr" and the Read class was assigned to "H" during the instantiation: hr.CONSTANTS, hr.params, H.grid, and H.data. List the keys in these dictionaries to determine what they contain, and consult the parameter glossary at the bottom of _hamr_read.py_ for assistance. 

As an example, suppose we wanted to read in our __Exciting_Simulation__ of a black hole of 10 solar masses, output in spherical Kerr-Schild coordinates ($r, \theta, \varphi$). If we wanted to read in the data and grid variables with CGS units, and only read in the data within a radial range of 1.2 to 300 gravitational radii ($R_g$), downsampled by a factor of 4 in the radial ($r$) coordinate, 4 in the poloidal ($\theta$) coordinate, and 16 in the toroidal ($\varphi$) coordinate, then the following commands would provide everything needed to plot the density ($\rho$ or "rho"):
 ```
H = hr.Read('DATA/Exciting_Simulation', cgs_data = True, cgs_grid = True, MBH = 10.0,
            do_box = 1, 
            r_min = 1.2, r_max = 300.0,
            lowres1 = 4, lowres2 = 4, lowres3 = 16)
H.read_grid(100, direct=True)
H.read_prims(100, skip_read_params=True)

MBH  = 10.0 # black hole mass in solar masses
Msun = hr.CONSTANTS['Msun']
c    = hr.CONSTANTS['c']
G    = hr.CONSTANTS['G']
Rg   = G*MBH*Msun/(c*c)

r, th, phi = H.grid['r']/Rg, H.grid['theta'], H.grid['phi']
rho = H.data['rho']
```
The coordinate variables (r, th, phi) and scalar variables like density (rho) are all 4D arrays with slots that correspond to (block number, $x_1$ coordinate index, $x_2$ coordinate index, $x_3$ coordinate index). For vectors like the 4-velocity $u^{\mu}$ (H.data['uu'] in the code) or the 3-magnetic field __B__ (H.data['B'] in the code), the arrays are 5D with slots (direction, block number, $x_1$ coordinate index, $x_2$ coordinate index, $x_3$ coordinate index), where direction $\in (0, 1, 2, 3)$ corresponds to the basis vectors in internal coordinates $(x_0, x_1, x_2, x_3)$. Similarly, for rank-2 tensors like the covariant metric tensor $g_{\mu \nu}$ (H.grid['gcov'] in the code), there are 6 slots: (direction, direction, block number, $x_1$ coordinate index, $x_2$ coordinate index, $x_3$ coordinate index). The block number is always 0 unless the by_block flag is used in hr.Read. In that case, the arrays are stored such that the coordinate indices correspond to a given block rather than the global array upscaled to the highest refinement level. Thus, the by_block option decreases the memory occupied by each of the arrays stored in the H.grid and H.data dictionaries. 

## Credits

_H-AMR_ as well as the original analysis pipeline was developed by Dr. Matthew Liska. The new pipeline contained in this repository was written by Dr. Christopher Bambic. Dr. Bambic is particularly grateful to Nicholas Kaaz for useful discussions and assistance. 
