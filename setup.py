import os
import numpy as np
# Cython specific functions
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
read_dir = os.path.dirname(os.path.abspath(__file__))

##############################################################################

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
# "/openmp", 
#extra_link_args=["/lgomp"]
