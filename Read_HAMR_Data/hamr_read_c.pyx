# cython: language_level=3
# distutils: language = c++

import ctypes
import numpy as np
cimport numpy as np
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "../Analyze_HAMR_Data/hamr_analysis.h":
    cdef cppclass Fluid:
        Fluid() except + 
        Fluid(unordered_map[string, double] params,
            unordered_map[string, double] CONSTANTS) except +
        
        void calcQie(double* qie, float* rho, float* Te, float* Ti,
            int nb, int nx1, int nx2, int nx3)
    
    cdef cppclass Opacity:
        Opacity() except + 
        Opacity(unordered_map[string, double] params,
            unordered_map[string, double] CONSTANTS) except +

        void kappaSynchrotron(double* kappa, float* rho, float* Te, 
            float* uRad, float* b2, int emi, int XRB, 
            int nb, int nx1, int nx2, int nx3)

        void kappaAbsTotal(double* kappa, float* rho, float* Te, 
            float* Tr, float *b2, int XRB, 
            int nb, int nx1, int nx2, int nx3)

        void kappaEmiTotal(double* kappa, float* rho, float* Te, 
            float* Tr, float *b2, int XRB, 
            int nb, int nx1, int nx2, int nx3)
    
    cdef cppclass RIAF_Opacity:
        RIAF_Opacity() except + 
        RIAF_Opacity(unordered_map[string, double] params,
            unordered_map[string, double] CONSTANTS) except +

        void kappaSynchrotron(double* kappa, float* rho, float* Te, 
            float* uRad, float* b2, int emi, int XRB, 
            int nb, int nx1, int nx2, int nx3)
        
        void kappaAbsTotal(double* kappa, float* rho, float* Te, 
            float* Tr, float *b2, int XRB, 
            int nb, int nx1, int nx2, int nx3)

        void kappaEmiTotal(double* kappa, float* rho, float* Te, 
            float* Tr, float *b2, int XRB, 
            int nb, int nx1, int nx2, int nx3)

    cdef cppclass Radiation:
        Radiation() except + 
        Radiation(unordered_map[string, double] params,
            unordered_map[string, double] CONSTANTS) except +

        void qAbs(double* qabs, float* rho, float* Te, 
            float* Tr, float *b2, int XRB, int RIAF,
            int nb, int nx1, int nx2, int nx3)

        void qEmi(double* qemi, float* rho, float* Te, 
            float* Tr, float *b2, int XRB, int RIAF,
            int nb, int nx1, int nx2, int nx3)

cdef extern from "hamr_read.h":
    cdef struct GridArrays:
        float* x1
        float* x2
        float* x3
        float* r
        float* theta
        float* phi
        float* gcov
        float* gcon
        float* gdet
        float* dxdxp

    cdef struct GridCoords:
        float* x1
        float* x2
        float* x3
        float* r
        float* theta
        float* phi

    cdef struct GridResolution:
        int* x1
        int* x2
        int* x3

    cdef cppclass Read:
        Read() except +
        Read(string dir, int axisym, int by_block, 
            int export_raytracing,
            unordered_map[string, double] amr,
            unordered_map[string, vector[int]] amr_info,
            unordered_map[string, double] block_info,
            unordered_map[string, double] options,
            unordered_map[string, double] params,
            unordered_map[string, int] physics) except +

        # `except +` is what turns a C++ throw into a Python exception. The
        # readers validate their input files and throw std::runtime_error on
        # a missing or truncated dump; without this Cython does not generate
        # the translating try/catch and the throw is undefined behaviour at
        # the boundary instead of a catchable RuntimeError.
        void readGrid(const GridArrays& grid) except +

        void readGridCoords(const GridCoords& coords) except +

        void readGridResolution(const GridResolution& res_level) except +

        void readPrims(int fnum,
            const unordered_map[string, float*]& prims) except +
    
    cdef cppclass Tools:
        Tools() except + 

        void invert4x4(float* A, float* B, 
            int nb, int nx1, int nx2, int nx3) 

        void det4x4(float* A, float* B, 
            int nb, int nx1, int nx2, int nx3)

        void lowerIndex(float* u, float* u_mu, float* gcov,
            int nb, int nx1, int nx2, int nx3, int axisym)
        
        void lowerIndex(float* u, float* u_mu, float* gcov,
            int nb, int nx1, int nx2, int nx3, int axisym, int l)

        void boostVector(float* u, float* u_boost, float* uu_c,
            int nb, int nx1, int nx2, int nx3, int inverse)

        void contractTensor(float* tnsr, float* tnsr_out, float* M,
            int nb, int nx1, int nx2, int nx3, int axisym_tnsr, int axisym_M)

def pass_dict_to_cpp(my_dict):
    cdef unordered_map[string, double] cpp_dict
    for key, value in my_dict.items():
        cpp_dict[key.encode('utf-8')] = value
    return cpp_dict

def pass_int_dict_to_cpp(my_dict):
    cdef unordered_map[string, int] cpp_dict
    for key, value in my_dict.items():
        cpp_dict[key.encode('utf-8')] = value
    return cpp_dict

def pass_int_vec_dict_to_cpp(my_dict):
    cdef unordered_map[string, vector[int]] cpp_dict
    for key, value in my_dict.items():
        cpp_dict[key.encode('utf-8')] = value
    return cpp_dict

# wrapper class for C++
cdef class PyFluid:
    cdef Fluid *c_fluid

    def __init__(self, params, CONSTANTS):
        params_cpp    = pass_dict_to_cpp(params)
        CONSTANTS_cpp = pass_dict_to_cpp(CONSTANTS)

        self.c_fluid  = new Fluid(params_cpp, CONSTANTS_cpp)

    def __dealloc__(self):
        if self.c_fluid != NULL:
            del self.c_fluid
            self.c_fluid = NULL

    def calc_qie(self, 
        np.ndarray[np.float64_t, ndim=4, mode="c"] qie,
        np.ndarray[np.float32_t, ndim=4, mode="c"] rho,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Te,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Ti, 
        nb, nx1, nx2, nx3):
        return self.c_fluid.calcQie(&qie[0,0,0,0],
            &rho[0,0,0,0], &Te[0,0,0,0], &Ti[0,0,0,0],
            nb, nx1, nx2, nx3)

# wrapper class for C++
cdef class PyOpacity:
    cdef Opacity *c_opacity

    def __init__(self, params, CONSTANTS):
        params_cpp    = pass_dict_to_cpp(params)
        CONSTANTS_cpp = pass_dict_to_cpp(CONSTANTS)

        self.c_opacity= new Opacity(params_cpp, CONSTANTS_cpp)

    def __dealloc__(self):
        if self.c_opacity != NULL:
            del self.c_opacity
            self.c_opacity = NULL
    
    def calc_kappa_abs_tot(self,
        np.ndarray[np.float64_t, ndim=4, mode="c"] kappa_abs,
        np.ndarray[np.float32_t, ndim=4, mode="c"] rho,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Te,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Tr,
        np.ndarray[np.float32_t, ndim=4, mode="c"] b2,
        int XRB, int nb, int nx1, int nx2, int nx3):
        return self.c_opacity.kappaAbsTotal(
            &kappa_abs[0,0,0,0], &rho[0,0,0,0], &Te[0,0,0,0],
            &Tr[0,0,0,0], &b2[0,0,0,0], XRB, 
            nb, nx1, nx2, nx3)
    
    def calc_kappa_emi_tot(self,
        np.ndarray[np.float64_t, ndim=4, mode="c"] kappa_emi,
        np.ndarray[np.float32_t, ndim=4, mode="c"] rho,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Te,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Tr,
        np.ndarray[np.float32_t, ndim=4, mode="c"] b2,
        int XRB, int nb, int nx1, int nx2, int nx3):
        return self.c_opacity.kappaEmiTotal(
            &kappa_emi[0,0,0,0], &rho[0,0,0,0], &Te[0,0,0,0],
            &Tr[0,0,0,0], &b2[0,0,0,0], XRB, 
            nb, nx1, nx2, nx3)
        
    def calc_kappa_sync(self,
        np.ndarray[np.float64_t, ndim=4, mode="c"] kappa_sync,
        np.ndarray[np.float32_t, ndim=4, mode="c"] rho,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Te,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Tr,
        np.ndarray[np.float32_t, ndim=4, mode="c"] b2,
        int emi, int XRB, int nb, int nx1, int nx2, int nx3):
        return self.c_opacity.kappaSynchrotron(
            &kappa_sync[0,0,0,0], &rho[0,0,0,0], &Te[0,0,0,0],
            &Tr[0,0,0,0], &b2[0,0,0,0], emi, XRB, 
            nb, nx1, nx2, nx3)

# wrapper class for C++
cdef class PyRIAF_Opacity:
    cdef RIAF_Opacity *c_opacity

    def __init__(self, params, CONSTANTS):
        params_cpp    = pass_dict_to_cpp(params)
        CONSTANTS_cpp = pass_dict_to_cpp(CONSTANTS)

        self.c_opacity= new RIAF_Opacity(params_cpp, CONSTANTS_cpp)

    def __dealloc__(self):
        if self.c_opacity != NULL:
            del self.c_opacity
            self.c_opacity = NULL
    
    def calc_kappa_abs_tot(self,
        np.ndarray[np.float64_t, ndim=4, mode="c"] kappa_abs,
        np.ndarray[np.float32_t, ndim=4, mode="c"] rho,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Te,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Tr,
        np.ndarray[np.float32_t, ndim=4, mode="c"] b2,
        int XRB, int nb, int nx1, int nx2, int nx3):
        return self.c_opacity.kappaAbsTotal(
            &kappa_abs[0,0,0,0], &rho[0,0,0,0], &Te[0,0,0,0],
            &Tr[0,0,0,0], &b2[0,0,0,0], XRB, 
            nb, nx1, nx2, nx3)
    
    def calc_kappa_emi_tot(self,
        np.ndarray[np.float64_t, ndim=4, mode="c"] kappa_emi,
        np.ndarray[np.float32_t, ndim=4, mode="c"] rho,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Te,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Tr,
        np.ndarray[np.float32_t, ndim=4, mode="c"] b2,
        int XRB, int nb, int nx1, int nx2, int nx3):
        return self.c_opacity.kappaEmiTotal(
            &kappa_emi[0,0,0,0], &rho[0,0,0,0], &Te[0,0,0,0],
            &Tr[0,0,0,0], &b2[0,0,0,0], XRB, 
            nb, nx1, nx2, nx3)
        
    def calc_kappa_sync(self,
        np.ndarray[np.float64_t, ndim=4, mode="c"] kappa_sync,
        np.ndarray[np.float32_t, ndim=4, mode="c"] rho,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Te,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Tr,
        np.ndarray[np.float32_t, ndim=4, mode="c"] b2,
        int emi, int XRB, int nb, int nx1, int nx2, int nx3):
        return self.c_opacity.kappaSynchrotron(
            &kappa_sync[0,0,0,0], &rho[0,0,0,0], &Te[0,0,0,0],
            &Tr[0,0,0,0], &b2[0,0,0,0], emi, XRB, 
            nb, nx1, nx2, nx3)

# wrapper class for C++
cdef class PyRadiation:
    cdef Radiation *c_radiation

    def __init__(self, params, CONSTANTS):
        params_cpp    = pass_dict_to_cpp(params)
        CONSTANTS_cpp = pass_dict_to_cpp(CONSTANTS)

        self.c_radiation = new Radiation(params_cpp, CONSTANTS_cpp)

    def __dealloc__(self):
        if self.c_radiation != NULL:
            del self.c_radiation
            self.c_radiation = NULL
    
    def calc_q_abs_tot(self,
        np.ndarray[np.float64_t, ndim=4, mode="c"] qabs,
        np.ndarray[np.float32_t, ndim=4, mode="c"] rho,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Te,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Tr,
        np.ndarray[np.float32_t, ndim=4, mode="c"] b2,
        int XRB, int RIAF, int nb, int nx1, int nx2, int nx3):
        return self.c_radiation.qAbs(&qabs[0,0,0,0], 
            &rho[0,0,0,0], &Te[0,0,0,0], &Tr[0,0,0,0], &b2[0,0,0,0], 
            XRB, RIAF, nb, nx1, nx2, nx3)
    
    def calc_q_emi_tot(self,
        np.ndarray[np.float64_t, ndim=4, mode="c"] qemi,
        np.ndarray[np.float32_t, ndim=4, mode="c"] rho,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Te,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Tr,
        np.ndarray[np.float32_t, ndim=4, mode="c"] b2,
        int XRB, int RIAF, int nb, int nx1, int nx2, int nx3):
        return self.c_radiation.qEmi(&qemi[0,0,0,0], 
            &rho[0,0,0,0], &Te[0,0,0,0], &Tr[0,0,0,0], &b2[0,0,0,0], 
            XRB, RIAF, nb, nx1, nx2, nx3)
     
# wrapper class for C++
cdef class PyRead:
    cdef Read *c_read

    def __init__(self, dir, axisym, by_block, export_raytracing, 
                 amr, amr_info, block_info, options, params, physics):
        amr_cpp        = pass_dict_to_cpp(amr)
        amr_info_cpp   = pass_int_vec_dict_to_cpp(amr_info)
        block_info_cpp = pass_dict_to_cpp(block_info)
        options_cpp    = pass_dict_to_cpp(options)
        params_cpp     = pass_dict_to_cpp(params)
        physics_cpp    = pass_int_dict_to_cpp(physics)

        self.c_read    = new Read(dir.encode('utf-8'), axisym, by_block,
            export_raytracing, amr_cpp, amr_info_cpp, block_info_cpp,
            options_cpp, params_cpp, physics_cpp)

    def __dealloc__(self):
        if self.c_read != NULL:
            del self.c_read
            self.c_read = NULL

    def read_grid(self, 
        np.ndarray[np.float32_t, ndim=4, mode="c"] x1,
        np.ndarray[np.float32_t, ndim=4, mode="c"] x2,
        np.ndarray[np.float32_t, ndim=4, mode="c"] x3,
        np.ndarray[np.float32_t, ndim=4, mode="c"] r,
        np.ndarray[np.float32_t, ndim=4, mode="c"] theta,
        np.ndarray[np.float32_t, ndim=4, mode="c"] phi,
        np.ndarray[np.float32_t, ndim=6, mode="c"] gcov,
        np.ndarray[np.float32_t, ndim=6, mode="c"] gcon,
        np.ndarray[np.float32_t, ndim=4, mode="c"] gdet,
        np.ndarray[np.float32_t, ndim=6, mode="c"] dxdxp):
        # The Python-facing signature is unchanged, and deliberately keeps
        # the typed np.ndarray declarations: they are what enforce float32,
        # the right ndim, and C-contiguity at call time. Bundling into the
        # struct happens after those checks, so the aggregate costs nothing
        # in safety -- unlike handing C++ raw .ctypes.data pointers.
        cdef GridArrays grid
        grid.x1    = &x1[0,0,0,0]
        grid.x2    = &x2[0,0,0,0]
        grid.x3    = &x3[0,0,0,0]
        grid.r     = &r[0,0,0,0]
        grid.theta = &theta[0,0,0,0]
        grid.phi   = &phi[0,0,0,0]
        grid.gcov  = &gcov[0,0,0,0,0,0]
        grid.gcon  = &gcon[0,0,0,0,0,0]
        grid.gdet  = &gdet[0,0,0,0]
        grid.dxdxp = &dxdxp[0,0,0,0,0,0]
        return self.c_read.readGrid(grid)

    def read_grid_coords(self, 
        np.ndarray[np.float32_t, ndim=4, mode="c"] x1,
        np.ndarray[np.float32_t, ndim=4, mode="c"] x2,
        np.ndarray[np.float32_t, ndim=4, mode="c"] x3,
        np.ndarray[np.float32_t, ndim=4, mode="c"] r,
        np.ndarray[np.float32_t, ndim=4, mode="c"] theta,
        np.ndarray[np.float32_t, ndim=4, mode="c"] phi):
        cdef GridCoords coords
        coords.x1    = &x1[0,0,0,0]
        coords.x2    = &x2[0,0,0,0]
        coords.x3    = &x3[0,0,0,0]
        coords.r     = &r[0,0,0,0]
        coords.theta = &theta[0,0,0,0]
        coords.phi   = &phi[0,0,0,0]
        return self.c_read.readGridCoords(coords)

    def read_grid_resolution(self, 
        np.ndarray[int, ndim=4, mode="c"] res_level_x1,
        np.ndarray[int, ndim=4, mode="c"] res_level_x2,
        np.ndarray[int, ndim=4, mode="c"] res_level_x3):
        cdef GridResolution res_level
        res_level.x1 = &res_level_x1[0,0,0,0]
        res_level.x2 = &res_level_x2[0,0,0,0]
        res_level.x3 = &res_level_x3[0,0,0,0]
        return self.c_read.readGridResolution(res_level)

    def read_prims(self, fnum, dict prims):
        """prims maps H.data-style names to float32 C-contiguous arrays.

        Only the arrays the dump's physics flags call for need be present;
        readPrims raises naming the key if a required one is missing. This
        replaces a 26-argument signature in which every disabled field had
        to be filled with a dummy alias of rho.

        The old per-argument np.ndarray[np.float32_t, ndim=N, mode="c"]
        declarations were what enforced dtype/contiguity, so those checks
        are made explicitly here rather than being silently dropped.
        """
        cdef unordered_map[string, float*] cpp_prims
        cdef np.ndarray arr
        for key, value in prims.items():
            arr = value
            if arr.dtype != np.float32:
                raise TypeError(
                    f"read_prims: '{key}' must be float32, got {arr.dtype}.")
            if not arr.flags['C_CONTIGUOUS']:
                raise ValueError(
                    f"read_prims: '{key}' must be C-contiguous.")
            cpp_prims[key.encode('utf-8')] = <float*>arr.data
        return self.c_read.readPrims(fnum, cpp_prims)

# wrapper class for C++
cdef class PyTools:
    cdef Tools *c_tools

    def __init__(self):
        self.c_tools = new Tools()

    def __dealloc__(self):
        if self.c_tools != NULL:
            del self.c_tools
            self.c_tools = NULL

    def invert_4x4(self, np.ndarray[np.float32_t, ndim=6, mode="c"] A,
        np.ndarray[np.float32_t, ndim=6, mode="c"] B, nb, nx1, nx2, nx3):
        return self.c_tools.invert4x4(&A[0,0,0,0,0,0], &B[0,0,0,0,0,0], 
                                      nb, nx1, nx2, nx3)

    def det_4x4(self, np.ndarray[np.float32_t, ndim=6, mode="c"] A,
        np.ndarray[np.float32_t, ndim=4, mode="c"] B, nb, nx1, nx2, nx3):
        return self.c_tools.det4x4(&A[0,0,0,0,0,0], &B[0,0,0,0], 
                                   nb, nx1, nx2, nx3)

    def lower_index(self, np.ndarray[np.float32_t, ndim=5, mode="c"] u,
        np.ndarray[np.float32_t, ndim=5, mode="c"] u_mu,
        np.ndarray[np.float32_t, ndim=6, mode="c"] gcov,
        nb, nx1, nx2, nx3, axisym):
        return self.c_tools.lowerIndex(&u[0,0,0,0,0], &u_mu[0,0,0,0,0],
            &gcov[0,0,0,0,0,0], nb, nx1, nx2, nx3, axisym)

    def lower_index_mu(self, np.ndarray[np.float32_t, ndim=5, mode="c"] u,
        np.ndarray[np.float32_t, ndim=4, mode="c"] u_mu,
        np.ndarray[np.float32_t, ndim=6, mode="c"] gcov, mu, 
        nb, nx1, nx2, nx3, axisym):
        return self.c_tools.lowerIndex(&u[0,0,0,0,0], &u_mu[0,0,0,0],
            &gcov[0,0,0,0,0,0], nb, nx1, nx2, nx3, axisym, mu)
    
    def boost_vector(self, np.ndarray[np.float32_t, ndim=5, mode="c"] u,
        np.ndarray[np.float32_t, ndim=5, mode="c"] u_boost,
        np.ndarray[np.float32_t, ndim=5, mode="c"] uu_c,
        nb, nx1, nx2, nx3, inverse):
        return self.c_tools.boostVector(&u[0,0,0,0,0], &u_boost[0,0,0,0,0],
            &uu_c[0,0,0,0,0], nb, nx1, nx2, nx3, inverse)

    def contract_tensor(self, np.ndarray[np.float32_t, ndim=6, mode="c"] tnsr,
        np.ndarray[np.float32_t, ndim=6, mode="c"] tnsr_out,
        np.ndarray[np.float32_t, ndim=6, mode="c"] M,
        nb, nx1, nx2, nx3, axisym_tnsr, axisym_M):
        return self.c_tools.contractTensor(&tnsr[0,0,0,0,0,0],
            &tnsr_out[0,0,0,0,0,0], &M[0,0,0,0,0,0],
            nb, nx1, nx2, nx3, axisym_tnsr, axisym_M)