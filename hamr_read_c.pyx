# cython: language_level=3
# distutils: language = c++

import ctypes
cimport numpy as np
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "hamr_read.h":
    void kernel_read_grid(string dir, int axisym, int by_block, 
        int export_raytracing,
        unordered_map[string, double] amr,
        unordered_map[string, vector[int]] amr_info,
        unordered_map[string, double] block_info,
        unordered_map[string, double] params,
        unordered_map[string, double] options,
        float* x1, float* x2, float* x3, float* r, float* theta, float* phi,
        float* gcov, float* gcon, float* gdet, float* dxdxp)

    void kernel_read_prims(string dir, int fnum, int by_block, 
        int export_raytracing,
        unordered_map[string, double] amr,
        unordered_map[string, vector[int]] amr_info,
        unordered_map[string, double] block_info,
        unordered_map[string, double] params,
        unordered_map[string, int] physics,
        unordered_map[string, double] options,
        float* rho, float* uGas, float* uu, float* B, float* E,
        float* uRad, float* uuRad, float* Ti, float* Te, float* phNum,
        float* Ye, float* xAlpha, float* xAtm, 
        float* uNu1, float* uNu2, float* uNu3, 
        float* uuNu1, float* uuNu2, float* uuNu3,
        float* nuNum1, float* nuNum2, float* nuNum3, float* tr1, float* tr2) 

    void kernel_invert_4x4(float* A, float* B, 
        int nb, int nx1, int nx2, int nx3) 

    void kernel_det_4x4(float* A, float* B, 
        int nb, int nx1, int nx2, int nx3)

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

def read_grid(dir, axisym, by_block, export_raytracing,
    amr, amr_info, block_info, params, options,
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
    amr_cpp        = pass_dict_to_cpp(amr)
    amr_info_cpp   = pass_int_vec_dict_to_cpp(amr_info)
    block_info_cpp = pass_dict_to_cpp(block_info)
    params_cpp     = pass_dict_to_cpp(params)
    options_cpp    = pass_dict_to_cpp(options)
    return kernel_read_grid(dir.encode('utf-8'), 
        axisym, by_block, export_raytracing,
        amr_cpp,
        amr_info_cpp,
        block_info_cpp,
        params_cpp,
        options_cpp,
        &x1[0,0,0,0], &x2[0,0,0,0], &x3[0,0,0,0], 
        &r[0,0,0,0], &theta[0,0,0,0], &phi[0,0,0,0],
        &gcov[0,0,0,0,0,0], &gcon[0,0,0,0,0,0],
        &gdet[0,0,0,0], &dxdxp[0,0,0,0,0,0]
    )

def read_prims(dir, fnum, by_block, export_raytracing,
    amr, amr_info, block_info, params, physics, options,
    np.ndarray[np.float32_t, ndim=4, mode="c"] rho,
    np.ndarray[np.float32_t, ndim=4, mode="c"] uGas,
    np.ndarray[np.float32_t, ndim=5, mode="c"] uu,
    np.ndarray[np.float32_t, ndim=5, mode="c"] B,
    np.ndarray[np.float32_t, ndim=5, mode="c"] E,
    np.ndarray[np.float32_t, ndim=4, mode="c"] uRad,
    np.ndarray[np.float32_t, ndim=5, mode="c"] uuRad,
    np.ndarray[np.float32_t, ndim=4, mode="c"] Ti,
    np.ndarray[np.float32_t, ndim=4, mode="c"] Te,
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
    amr_cpp        = pass_dict_to_cpp(amr)
    amr_info_cpp   = pass_int_vec_dict_to_cpp(amr_info)
    block_info_cpp = pass_dict_to_cpp(block_info)
    params_cpp     = pass_dict_to_cpp(params)
    physics_cpp    = pass_int_dict_to_cpp(physics)
    options_cpp    = pass_dict_to_cpp(options)
    return kernel_read_prims(dir.encode('utf-8'), 
        fnum, by_block, export_raytracing,
        amr_cpp,
        amr_info_cpp,
        block_info_cpp,
        params_cpp,
        physics_cpp,
        options_cpp,
        &rho[0,0,0,0], &uGas[0,0,0,0], &uu[0,0,0,0,0],
        &B[0,0,0,0,0], &E[0,0,0,0,0], &uRad[0,0,0,0],
        &uuRad[0,0,0,0,0], &Ti[0,0,0,0], &Te[0,0,0,0],
        &phNum[0,0,0,0], &Ye[0,0,0,0], 
        &xAlpha[0,0,0,0], &xAtm[0,0,0,0], 
        &uNu1[0,0,0,0], &uNu2[0,0,0,0], &uNu3[0,0,0,0],
        &uuNu1[0,0,0,0,0], &uuNu2[0,0,0,0,0], &uuNu3[0,0,0,0,0],
        &nuNum1[0,0,0,0], &nuNum2[0,0,0,0], &nuNum3[0,0,0,0],
        &tr1[0,0,0,0], &tr2[0,0,0,0])

def invert_4x4(np.ndarray[np.float32_t, ndim=6, mode="c"] A,
    np.ndarray[np.float32_t, ndim=6, mode="c"] B, nb, nx1, nx2, nx3):
    return kernel_invert_4x4(&A[0,0,0,0,0,0], &B[0,0,0,0,0,0], 
    nb, nx1, nx2, nx3)

def det_4x4(np.ndarray[np.float32_t, ndim=6, mode="c"] A,
    np.ndarray[np.float32_t, ndim=4, mode="c"] B, nb, nx1, nx2, nx3):
    return kernel_det_4x4(&A[0,0,0,0,0,0], &B[0,0,0,0], 
    nb, nx1, nx2, nx3)

