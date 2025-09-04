# H-AMR Analysis

This is the analysis and visualization toolkit for _H-AMR_ ([Liska et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJS..263...26L/abstract)), a new GPU-accelerated general relativistic magnetohydrodynamics (GRMHD) code optimized to leverage exascale GPU-computing architectures.

_H-AMR_'s 3D adaptive mesh refinement (AMR) and adaptive time-stepping capabilities enable the code to perform state-of-the-art computations of accretion onto black holes at some of the highest resolutions achieved to date (e.g. [Liska et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474L..81L/abstract), [Ripperda et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...924L..32R/abstract)). However, the enormous data sets produced by _H-AMR_ require a unique read-in pipeline that can interpret the custom binary formats used for optimal memory storage on modern supercomputers.

This pipeline, written in Python and interfaced with C++ through a Cython-based API, allows the user to read in _H-AMR_ binary data and upscale the data to the highest level of mesh refinement via a nearest-neighbor interpolation scheme. More advanced interpolation routines, specifically tailored to allow post-facto ray-tracing of the data, are currently under development. The pipeline provides support for many of _H-AMR_'s physics modules, including M1 radiation transport ([Sadowski et al. 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.429.3533S/abstract), [McKinney et al. 2014](https://ui.adsabs.harvard.edu/abs/2014MNRAS.441.3177M/abstract), [Liska et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...944L..48L/abstract)), "two-temperature" thermodynamics ([Ressler et al. 2015](https://ui.adsabs.harvard.edu/abs/2015MNRAS.454.1848R/abstract); [Sadowski et al. 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..705S/abstract); [Liska et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...935L...1L/abstract)) which separately evolves ion and electron temperatures according to sub-grid schemes for allocating dissipated energy between the species, a photon-number conserving treatment of Comptonization ([Sadowski & Narayan 2015](https://ui.adsabs.harvard.edu/abs/2015MNRAS.454.2372S/abstract); [Liska et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...935L...1L/abstract)), resistive GRMHD schemes that separately evolve the electric field and magnetic field (e.g. [Ripperda et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJS..244...10R/abstract)), passive-scalar "tracer" fluids, and M1 neutrino transport ([Issa et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv241002852I/abstract)). The pipeline can be easily extended to handle additional physics modules, many of which are currently under active development.  

## Getting Started 

To use this toolkit, set up an SSH key on your GitHub account, then clone the repository with 
```
git clone git@github.com:cbambic24/HAMR_Visualization.git
```
This process will create a directory called __HAMR_Visualization__, with three sub-directories: __Analyze_HAMR_Data__, __Read_HAMR_Data__, and __Render_Movies__. All of the primary analysis tools, as well as the C++ acceleration routines for all of the calculations specific to radiation and two-temperature physics, are contained within the __Analyze_HAMR_Data__ directory. Reading in of the data and compilation of the toolkit is accomplished by the files contained within __Read_HAMR_Data__. We include a large number of sample scripts for producing images/movies in the __Render_Movies__ directory. The file __Analyze_HAMR_Data__/_compute_variables.py_ is the only file that the user should need to modify. There, I define a number of paths that are specific to my setup on Oakridge National Laboratory's _Frontier_ supercomputer. You should modify these directories as it suits your set-up.

We recommend you create the __HAMR_Visualization__ directory within the _user_path_ directory (for me, this is the path under the heading "cbambic"). At this same level (i.e. in the _user_path_ directory), we recommend you create the following directories: __Averages__, __DATA__, __Images__, __Movies__, __PDF__, __Logs__, and __RUN__. Generally, we store all of the profile and slice averages in the __Averages__ directory, hold all of the _H-AMR_ dumps files inside of the __DATA__/_Simulation_ directory, where here _Simulation_ refers to a specific _H-AMR_ simulation that you're analyzing, and we create the PNGs that are later turned into movies inside the __Images__ directory. We store folders containing LaTeX libraries, ffmpeg, etc. inside of __RUN__ and we will always execute code from the __RUN__ directory. In this respect, Jupyter notebooks are usually stored in the __RUN__ directory. Final PDFs that go into papers are stored in the __PDF__ directory. This set-up is what is expected by the toolkit, and these directories will be produced by the analysis scripts as they are needed if they do not already exist.  

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
]
)
```
Three _setup.py_ files are provided: _setup.py_, _setup_OLCF.py_, and _setup_Windows.py_. The first two are identical and use the above commands to work in a Linux environment. The _setup_Windows.py_ file has been tested in a Windows environment.

On Macs, we managed to get the following combination to work:
```
import os
import numpy as np
# Cython specific functions
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
os.environ["CXX"] = "/opt/homebrew/bin/g++-14"
read_dir = os.path.dirname(os.path.abspath(__file__))

##############################################################################

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("hamr_readc",
                           sources=[read_dir + '/' + 'hamr_read_c.pyx',
                                    read_dir + '/' + 'hamr_read.cpp'],
                           language="c++",
                           include_dirs=[np.get_include()],
                           extra_compile_args=["-std=c++20",
                           "-fopenmp"])
]
)
```
Note that we had to use the Homebrew version of g++ (be careful not to use the Apple g++ as it is just Clang++ aliased) and set the CXX environment variable. If you try to use Clang/Clang++ with OpenMP (used to parallelize the read-in of large data files), you may run into problems.

Another potential issue is that you could arrive at a segmentation fault while running the code because of using an older C++ compiler (e.g. older versions of g++ or MSVC) that is not fully compliant with C++17. Here, the problem originated with the std::file_system library, which was only introduced in recent versions of C++. You can circumvent this issue by using the more basic C command for checking if a file exists,
```
if(access(filename, 0) != -1)
```
where filename is an appropriately defined string for the file you're looking for. This can serve as a temporary work-around if you are in an environment where you cannot upgrade your g++ compiler to comply with modern C++ standards. 

### Reading in a Data Set

Suppose you were interested in loading in a state-of-the-art two-temperature radiation GRMHD simulation stored in __DATA__/__Exciting_Simulation__. First, you should edit the paths appropriately in __HAMR_Visuaization__/__Analyze_HAMR_Data__/_compute_variables.py_ such that the Python path includes all of the necessary analysis files. Then, to read in output 100 (or in _H-AMR_ parlance, dump 100) with your Jupyter notebook or Python script situated in __RUN__, you would execute
```
import sys
import os
import numpy as np
sys.path.insert(0, 'PATH_TO_HAMR_Visualization/HAMR_Visualization/Analyze_HAMR_Data')
import compute_variables as cv
import hamr_read as hr

H = hr.Read('DATA/Exciting_Simulation')
H.read_grid(100, direct=True)
H.read_prims(100, skip_read_params=True)
```
The flags "direct=True" and "skip_read_params=True" are unnecessary if you have properly written grid dump (gdumps) files in __DATA__/__Exciting_Simulation__/__gdumps__. Otherwise, the "direct=True" flag implies that you would like to analytically compute the Kerr metric for a spinning black hole in spherical Kerr-Schild coordinates without the additional step of waiting for the grid to be read in. The "skip_read_params=True" flag is a convenience that prevents read_prims from doing double-work and re-reading in the parameters file when read_grid has already read-in and stored the parameters. 

Additional flags/options available to the Read class can be found in the init() function in _hamr_read.py_. The user is encouraged to explore these settings and consult the Wiki on this repository for further instructions. Finally, for immediate assistance, please consult the parameter glossary included at the bottom of _hamr_read.py_. 

### Using the Data

The user should only need access to the following dictionaries, where we assume that hamr_read was imported as "hr" and the Read class was assigned to "H" during the instantiation: hr.CONSTANTS, H.params, H.grid, and H.data. List the keys in these dictionaries to determine what they contain, and consult the parameter glossary at the bottom of _hamr_read.py_ for assistance. 

As an example, suppose we wanted to read in our __Exciting_Simulation__ of a black hole of 10 solar masses, output in spherical Kerr-Schild coordinates ($r, \theta, \varphi$). If we wanted to read in the data and grid variables with CGS units, and only read in the data within a radial range of 1.2 to 300 gravitational radii ($R_g$), downsampled by a factor of 4 in the radial ($r$) coordinate, 4 in the poloidal ($\theta$) coordinate, and 16 in the toroidal ($\varphi$) coordinate, then the following commands would provide everything needed to plot the density ($\rho$ or "rho"):
 ```
H = hr.Read('DATA/Exciting_Simulation', cgs_data = True, cgs_grid = True, MBH = 10.0,
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

### Extracting Radial Profiles or Poloidal Slices
For any of the executable scripts we have written, the usage instructions should be available through the inclusion of the -h (for "help") flag at the end of the script. For instance, if you were to execute the _radial_profiles_time.py_ script from within the __RUN__ directory, 
```
python ../HAMR_Visualization/Analyze_HAMR_Data/radial_profiles_time.py -h
```
you would see the following information:
```
python ../HAMR_Visualization/Analyze_HAMR_Data/radial_profiles_time.py -h
usage: radial_profiles_time.py [-h] [--nslices NSLICES] [--lowres1 LOWRES1] [--lowres2 LOWRES2] [--lowres3 LOWRES3] [--radiation | --no-radiation] [--two_temp | --no-two_temp] [--reduced] [--restart] [--shared] [--title TITLE] sim start_file end_file

Calculate 1D profiles for all simulation outputs.

positional arguments:
  sim                   simulation name
  start_file            dump number of start file in profile extraction
  end_file              dump number of end file in profile extraction

options:
  -h, --help            show this help message and exit
  --nslices NSLICES     number of slices in phi used to load the data
  --lowres1 LOWRES1     downsampling parameter in x1
  --lowres2 LOWRES2     downsampling parameter in x2
  --lowres3 LOWRES3     downsampling parameter in x3
  --radiation, --no-radiation
                        boolean to determine if radiation quantities should be included in profile extraction. (default: True)
  --two_temp, --no-two_temp
                        boolean to determine if two-temperature physics should be included in profile extraction. (default: True)
  --reduced             boolean to determine if profile extraction should be done on reduced data.
  --restart             boolean set to determine if profile integration should continue from last slice_num
  --shared              boolean set to determine if proj-shared should be used as the location of the data directory
  --title TITLE         title of simulation used for Averages directory if different from the simulation title in proj-shared
```
The most common trick for analyzing _H-AMR_ data is to downsample in each direction independently using the --lowres flags. In addition, the --reduced flag tells the script to analyze the data in the __Data__/__Exciting_Simulation__/__reduced__ folder, which is usually downsampled ahead of time to reduce the amount of data that has to be moved through memory. For profile and slice extraction (where here, "slice" refers to an azimuthal average of the 3D data, which results in a 2D r-theta "slice"), the scripts are both OpenMP and MPI parallelized, with data chunks (for _radial_profiles.py_ and _poloidal_slices.py_) or individual dumps (for _radial_profiles_time.py_ and _poloidal_slices_time.py_) spread among different processors, while the read-in and computations on the separate data chunks/dumps are multithreaded with OpenMP, particularly whenever the cpp=True flag is passed to a function. Please consult the Wiki for invidiual function calls/behaviors. 

### Creating Images/Movies
As an example for creating a movie of the midplane density, you can run the script
```
python ../HAMR_Visualization/Render_Movies/Midplane/density_midplane.py -h
```
This will yield the following output:
```
python ../HAMR_Visualization/Render_Movies/Midplane/density_midplane.py -h
usage: density_midplane.py [-h] [--lowres1 LOWRES1] [--lowres2 LOWRES2] [--lowres3 LOWRES3] [--radiation | --no-radiation] [--two_temp | --no-two_temp] [--fieldlines] [--streamlines] [--rad_streamlines] [--reduced] [--shared] [--title TITLE] [--movie_title MOVIE_TITLE] sim start_file end_file

Midplane slice of fluid-frame density.

positional arguments:
  sim                   simulation name
  start_file            dump number of start file in movie extraction
  end_file              dump number of end file in movie extraction

options:
  -h, --help            show this help message and exit
  --lowres1 LOWRES1     downsampling parameter in x1
  --lowres2 LOWRES2     downsampling parameter in x2
  --lowres3 LOWRES3     downsampling parameter in x3
  --radiation, --no-radiation
                        boolean to determine if radiation quantities should be included in movie extraction. (default: True)
  --two_temp, --no-two_temp
                        boolean to determine if two-temperature physics should be included in movie extraction. (default: True)
  --fieldlines          boolean to determine if magnetic-field lines should be drawn (default: False).
  --streamlines         boolean to determine if 4-velocity streamlines should be drawn (default: False).
  --rad_streamlines     boolean to determine if radiation 4-velocity streamlines should be drawn (default: False).
  --reduced             boolean to determine if movie extraction should be done on reduced data.
  --shared              boolean set to determine if proj-shared should be used as the location of the data directory
  --title TITLE         title of simulation used for Movie directory if different from the simulation title in proj-shared
  --movie_title MOVIE_TITLE
                        title of movie (default: Density)
```
The script can be used to plot the density for frames start_file to end_file along with streamlines of the 3-vector magnetic field (--fieldlines), the 4-velocity (--streamlines) or the radiation 4-velocity (--rad_streamlines). 

Similarly, the same can be done with movies in the X-Z (poloidal) plane. Here, we can look at the density script usng the help (-h) flag to find,
```
python ../HAMR_Visualization/Render_Movies/Poloidal/density_poloidal.py -h
usage: density_poloidal.py [-h] [--lowres1 LOWRES1] [--lowres2 LOWRES2] [--lowres3 LOWRES3] [--radiation | --no-radiation] [--two_temp | --no-two_temp] [--fieldlines] [--streamlines] [--rad_streamlines] [--reduced] [--shared] [--sigma_contour SIGMA_CONTOUR] [--scattering_photosphere] [--effective_photosphere]
                           [--integrate_in_z] [--title TITLE] [--movie_title MOVIE_TITLE]
                           sim start_file end_file

Poloidal slice fluid-frame density.

positional arguments:
  sim                   simulation name
  start_file            dump number of start file in movie extraction
  end_file              dump number of end file in movie extraction

options:
  -h, --help            show this help message and exit
  --lowres1 LOWRES1     downsampling parameter in x1
  --lowres2 LOWRES2     downsampling parameter in x2
  --lowres3 LOWRES3     downsampling parameter in x3
  --radiation, --no-radiation
                        boolean to determine if radiation quantities should be included in movie extraction. (default: True)
  --two_temp, --no-two_temp
                        boolean to determine if two-temperature physics should be included in movie extraction. (default: True)
  --fieldlines          boolean to determine if magnetic-field lines should be drawn (default: False).
  --streamlines         boolean to determine if 4-velocity streamlines should be drawn (default: False).
  --rad_streamlines     boolean to determine if radiation 4-velocity streamlines should be drawn (default: False).
  --reduced             boolean to determine if movie extraction should be done on reduced data.
  --shared              boolean set to determine if proj-shared should be used as the location of the data directory
  --sigma_contour SIGMA_CONTOUR
                        draw sigma contour for specified value of sigma (default: 0)
  --scattering_photosphere
                        boolean to determine if scattering photosphere should be drawn
  --effective_photosphere
                        boolean to determine if effective absorption photosphere should be drawn (based on all opacities)
  --integrate_in_z      boolean to determine if photospheres should be computed from theta integral (default) vs z-integral
  --title TITLE         title of simulation used for Movie directory if different from the simulation title in proj-shared
  --movie_title MOVIE_TITLE
                        title of movie (default: Density)
```
For plots in the poloidal plane, we have included a number of additional options, namely the ability to draw the scattering photosphere and the effective photosphere (defined for simplicity as the optical depth $\tau = 1$ surface) as well as a contour for sigma. Further, integrals for computing the optical depths/photosheres can be computed either by integrating in $\theta$ or by integrating in $z$ through the flag --integrate_in_z.

## Credits

_H-AMR_ as well as the original analysis pipeline was developed by Dr. Matthew Liska. The new pipeline contained in this repository was written by Dr. Christopher Bambic. Dr. Bambic is particularly grateful to Nicholas Kaaz for useful discussions and assistance. 
