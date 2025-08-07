# cython: language_level=3
# distutils: language = c++

import ctypes
cimport numpy as np
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "hamr_analysis.h":
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
    
    cdef cppclass Radiation:
        Radiation() except + 
        Radiation(unordered_map[string, double] params,
            unordered_map[string, double] CONSTANTS) except +

        void qAbs(double* qabs, float* rho, float* Te, 
            float* Tr, float *b2, int XRB, 
            int nb, int nx1, int nx2, int nx3)

        void qEmi(double* qemi, float* rho, float* Te, 
            float* Tr, float *b2, int XRB, 
            int nb, int nx1, int nx2, int nx3)

cdef extern from "hamr_read.h":
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

        void readGrid(float* x1, float* x2, float* x3, 
            float* r, float* theta, float* phi, 
            float* gcov, float* gcon, float* gdet, float* dxdxp)

        void readGridCoords(float* x1, float* x2, float* x3, 
            float* r, float* theta, float* phi)

        void readGridResolution(int* res_level_x1, int* res_level_x2, 
            int* res_level_x3)

        void readPrims(int fnum, 
            float* rho, float* uGas, float* uu, float* B, 
            float* E, float* uRad, float* uuRad, 
            float* Te, float* Ti, float* phNum,
            float* Ye, float* xAlpha, float* xAtm, 
            float* uNu1, float* uNu2, float* uNu3, 
            float* uuNu1, float* uuNu2, float* uuNu3,
            float* nuNum1, float* nuNum2, float* nuNum3, 
            float* tr1, float* tr2)
    
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
    
    def calc_q_abs_tot(self,
        np.ndarray[np.float64_t, ndim=4, mode="c"] qabs,
        np.ndarray[np.float32_t, ndim=4, mode="c"] rho,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Te,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Tr,
        np.ndarray[np.float32_t, ndim=4, mode="c"] b2,
        int XRB, int nb, int nx1, int nx2, int nx3):
        return self.c_radiation.qAbs(&qabs[0,0,0,0], 
            &rho[0,0,0,0], &Te[0,0,0,0], &Tr[0,0,0,0], &b2[0,0,0,0], 
            XRB, nb, nx1, nx2, nx3)
    
    def calc_q_emi_tot(self,
        np.ndarray[np.float64_t, ndim=4, mode="c"] qemi,
        np.ndarray[np.float32_t, ndim=4, mode="c"] rho,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Te,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Tr,
        np.ndarray[np.float32_t, ndim=4, mode="c"] b2,
        int XRB, int nb, int nx1, int nx2, int nx3):
        return self.c_radiation.qEmi(&qemi[0,0,0,0], 
            &rho[0,0,0,0], &Te[0,0,0,0], &Tr[0,0,0,0], &b2[0,0,0,0], 
            XRB, nb, nx1, nx2, nx3)
     
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
        return self.c_read.readGrid(
            &x1[0,0,0,0], &x2[0,0,0,0], &x3[0,0,0,0],
            &r[0,0,0,0], &theta[0,0,0,0], &phi[0,0,0,0],
            &gcov[0,0,0,0,0,0], &gcon[0,0,0,0,0,0],
            &gdet[0,0,0,0], &dxdxp[0,0,0,0,0,0])

    def read_grid_coords(self, 
        np.ndarray[np.float32_t, ndim=4, mode="c"] x1,
        np.ndarray[np.float32_t, ndim=4, mode="c"] x2,
        np.ndarray[np.float32_t, ndim=4, mode="c"] x3,
        np.ndarray[np.float32_t, ndim=4, mode="c"] r,
        np.ndarray[np.float32_t, ndim=4, mode="c"] theta,
        np.ndarray[np.float32_t, ndim=4, mode="c"] phi):
        return self.c_read.readGridCoords(
            &x1[0,0,0,0], &x2[0,0,0,0], &x3[0,0,0,0],
            &r[0,0,0,0], &theta[0,0,0,0], &phi[0,0,0,0])

    def read_grid_resolution(self, 
        np.ndarray[int, ndim=4, mode="c"] res_level_x1,
        np.ndarray[int, ndim=4, mode="c"] res_level_x2,
        np.ndarray[int, ndim=4, mode="c"] res_level_x3):
        return self.c_read.readGridResolution(
            &res_level_x1[0,0,0,0], &res_level_x2[0,0,0,0], 
            &res_level_x3[0,0,0,0])

    def read_prims(self, fnum,    
        np.ndarray[np.float32_t, ndim=4, mode="c"] rho,
        np.ndarray[np.float32_t, ndim=4, mode="c"] uGas,
        np.ndarray[np.float32_t, ndim=5, mode="c"] uu,
        np.ndarray[np.float32_t, ndim=5, mode="c"] B,
        np.ndarray[np.float32_t, ndim=5, mode="c"] E,
        np.ndarray[np.float32_t, ndim=4, mode="c"] uRad,
        np.ndarray[np.float32_t, ndim=5, mode="c"] uuRad,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Te,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Ti,
        np.ndarray[np.float32_t, ndim=4, mode="c"] phNum,
        np.ndarray[np.float32_t, ndim=4, mode="c"] Ye,
        np.ndarray[np.float32_t, ndim=4, mode="c"] xAlpha,
        np.ndarray[np.float32_t, ndim=4, mode="c"] xAtm,
        np.ndarray[np.float32_t, ndim=4, mode="c"] uNu1,
        np.ndarray[np.float32_t, ndim=4, mode="c"] uNu2,
        np.ndarray[np.float32_t, ndim=4, mode="c"] uNu3,
        np.ndarray[np.float32_t, ndim=5, mode="c"] uuNu1,
        np.ndarray[np.float32_t, ndim=5, mode="c"] uuNu2,
        np.ndarray[np.float32_t, ndim=5, mode="c"] uuNu3,
        np.ndarray[np.float32_t, ndim=4, mode="c"] nuNum1,
        np.ndarray[np.float32_t, ndim=4, mode="c"] nuNum2,
        np.ndarray[np.float32_t, ndim=4, mode="c"] nuNum3,
        np.ndarray[np.float32_t, ndim=4, mode="c"] tr1,
        np.ndarray[np.float32_t, ndim=4, mode="c"] tr2):
        return self.c_read.readPrims(fnum, 
            &rho[0,0,0,0], &uGas[0,0,0,0], &uu[0,0,0,0,0],
            &B[0,0,0,0,0], &E[0,0,0,0,0], &uRad[0,0,0,0],
            &uuRad[0,0,0,0,0], &Te[0,0,0,0], &Ti[0,0,0,0],
            &phNum[0,0,0,0], &Ye[0,0,0,0],
            &xAlpha[0,0,0,0], &xAtm[0,0,0,0],
            &uNu1[0,0,0,0], &uNu2[0,0,0,0], &uNu3[0,0,0,0],
            &uuNu1[0,0,0,0,0], &uuNu2[0,0,0,0,0], &uuNu3[0,0,0,0,0],
            &nuNum1[0,0,0,0], &nuNum2[0,0,0,0], &nuNum3[0,0,0,0],
            &tr1[0,0,0,0], &tr2[0,0,0,0])

# wrapper class for C++
cdef class PyTools:
    cdef Tools *c_tools

    def __init__(self):
        self.c_tools = new Tools()

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