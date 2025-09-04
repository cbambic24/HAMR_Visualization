import numpy as np
from numba import jit
from numba import prange
from scipy.interpolate import RegularGridInterpolator
import tracemalloc
from analysis import Tools

class Coordinates():
    def __init__(self, r, theta, phi, pars, CONSTANTS, **kwargs):
        self.MBH = kwargs.get('MBH', 1.0)
        self.dx1, self.dx2, self.dx3 = pars['dx1'], pars['dx2'], pars['dx3']
        if (kwargs.get('cgs', False)):
            self.c    = CONSTANTS['c']        # cm/s
            self.G    = CONSTANTS['G']        # cm^3/g s^2
            self.Msun = CONSTANTS['Msun']     # g
        else:
            self.c    = 1.0
            self.G    = 1.0
            self.Msun = 1.0
            self.MBH  = 1.0

        Rg = (self.G * self.MBH * self.Msun) / (self.c * self.c)

        self.r  = r / Rg
        self.th = theta
        self.ph = phi

    # -------------------------------------
    # Spherical to Cartesian (orthonormal)
    # -------------------------------------
    # (t, r, theta, phi) -> (t, x, y, z)
    def sph_to_cart_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        drdx = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        # Transformation to Cartesian Kerr Schild from Spherical Kerr Schild
        drdx[0, 0] = 1.0
        drdx[1, 1] = (np.sin(self.th) * np.cos(self.ph))
        drdx[1, 2] = (self.r * np.cos(self.th) * np.cos(self.ph))
        drdx[1, 3] = (-self.r * np.sin(self.th) * np.sin(self.ph))
        drdx[2, 1] = (np.sin(self.th) * np.sin(self.ph))
        drdx[2, 2] = (self.r * np.cos(self.th) * np.sin(self.ph))
        drdx[2, 3] = (self.r * np.sin(self.th) * np.cos(self.ph))
        drdx[3, 1] = (np.cos(self.th))
        drdx[3, 2] = (-self.r * np.sin(self.th))

        if not kwargs.get('cpp', False):
            return np.einsum('ij...,j...->i...', drdx, vec)
        else:
            return Tools.contract_vector_cpp(vec, drdx)

    # (t, r, theta, phi) -> (t, x, y, z)
    def sph_to_cart_one_form(self, one_form, **kwargs):
        return self.cart_to_sph_vec(one_form, **kwargs)
    
    # (t, r, theta, phi) -> (t, x, y, z)
    def sph_to_cart_tnsr(self, tnsr, **kwargs):
        shape = list(self.r.shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        drdx = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        
        # Transformation to Cartesian Kerr Schild from Spherical Kerr Schild
        drdx[0, 0] = 1.0
        drdx[1, 1] = (np.sin(self.th) * np.cos(self.ph))
        drdx[1, 2] = (self.r * np.cos(self.th) * np.cos(self.ph))
        drdx[1, 3] = (-self.r * np.sin(self.th) * np.sin(self.ph))
        drdx[2, 1] = (np.sin(self.th) * np.sin(self.ph))
        drdx[2, 2] = (self.r * np.cos(self.th) * np.sin(self.ph))
        drdx[2, 3] = (self.r * np.sin(self.th) * np.cos(self.ph))
        drdx[3, 1] = (np.cos(self.th))
        drdx[3, 2] = (-self.r * np.sin(self.th))

        return np.einsum('kl...,ki...,lj...->ij...', tnsr, drdx, drdx)

    # (t, r, theta, phi) -> (t, x, y, z)
    def sph_to_cart_cotnsr(self, cotnsr, **kwargs):
        return self.cart_to_sph_tnsr(cotnsr, **kwargs)

    # -------------------------------------
    # Cartesian (orthonormal) to Spherical
    # -------------------------------------
    # (t, x, y, z) -> (t, r, theta, phi)
    def cart_to_sph_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        dxdr = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        # Transformation to Spherical Kerr Schild from Cartesian Kerr Schild
        dxdr[0, 0] = 1.0
        dxdr[1, 1] = (np.sin(self.th) * np.cos(self.ph))
        dxdr[1, 2] = (np.sin(self.th) * np.sin(self.ph))
        dxdr[1, 3] = (np.cos(self.th))
        dxdr[2, 1] = ((1. / self.r) * (np.cos(self.th) * np.cos(self.ph)))
        dxdr[2, 2] = ((1. / self.r) * (np.cos(self.th) * np.sin(self.ph)))
        dxdr[2, 3] = (-(1. / self.r) * np.sin(self.th))
        dxdr[3, 1] = (-(1. / self.r) * np.sin(self.ph) / np.sin(self.th))
        dxdr[3, 2] = ((1. / self.r) * np.cos(self.ph) / np.sin(self.th))

        if not kwargs.get('cpp', False):
            return np.einsum('ij...,j...->i...', dxdr, vec)
        else:
            return Tools.contract_vector_cpp(vec, dxdr)
    
    # (t, x, y, z) -> (t, r, theta, phi)
    def cart_to_sph_one_form(self, one_form, **kwargs):
        return self.sph_to_cart_vec(one_form, **kwargs)

    # (t, x, y, z) -> (t, r, theta, phi)
    def cart_to_sph_tnsr(self, tnsr, **kwargs):
        shape = list(self.r.shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dxdr = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        # Transformation to Spherical Kerr Schild from Cartesian Kerr Schild
        dxdr[0, 0] = 1.0
        dxdr[1, 1] = (np.sin(self.th) * np.cos(self.ph))
        dxdr[1, 2] = (np.sin(self.th) * np.sin(self.ph))
        dxdr[1, 3] = (np.cos(self.th))
        dxdr[2, 1] = ((1. / self.r) * (np.cos(self.th) * np.cos(self.ph)))
        dxdr[2, 2] = ((1. / self.r) * (np.cos(self.th) * np.sin(self.ph)))
        dxdr[2, 3] = (-(1. / self.r) * np.sin(self.th))
        dxdr[3, 1] = (-(1. / self.r) * np.sin(self.ph) / np.sin(self.th))
        dxdr[3, 2] = ((1. / self.r) * np.cos(self.ph) / np.sin(self.th))
        
        return np.einsum('kl...,ki...,lj...->ij...', tnsr, dxdr, dxdr)
    
    def cart_to_sph_cotnsr(self, cotnsr, **kwargs):
        return self.sph_to_cart_tnsr(cotnsr, **kwargs)

    # -------------------------------------
    # Spherical to Cylindrical (orthonormal)
    # -------------------------------------
    # (t, r, theta, phi) -> (t, R, phi, z)
    def sph_to_cyl_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        drdz = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        # Transformation to Cylindrical Kerr Schild from Spherical Kerr Schild
        drdz[0, 0] = 1.
        drdz[1, 1] = np.sin(self.th)
        drdz[1, 2] = self.r * np.cos(self.th)
        drdz[2, 3] = self.r * np.sin(self.th)
        drdz[3, 1] = np.cos(self.th)
        drdz[3, 2] = -self.r * np.sin(self.th)

        if not kwargs.get('cpp', False):
            return np.einsum('ij...,j...->i...', drdz, vec)
        else:
            return Tools.contract_vector_cpp(vec, drdz)
    
    # (t, r, theta, phi) -> (t, R, phi, z)
    def sph_to_cyl_one_form(self, one_form, **kwargs):
        return self.cyl_to_sph_vec(one_form, **kwargs)

    # (t, r, theta, phi) -> (t, R, phi, z)
    def sph_to_cyl_tnsr(self, tnsr, **kwargs):
        shape = list(self.r.shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        drdz = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        
        # Transformation to Cylindrical Kerr Schild from Spherical Kerr Schild
        drdz[0, 0] = 1.
        drdz[1, 1] = np.sin(self.th)
        drdz[1, 2] = self.r * np.cos(self.th)
        drdz[2, 3] = self.r * np.sin(self.th)
        drdz[3, 1] = np.cos(self.th)
        drdz[3, 2] = -self.r * np.sin(self.th)
        
        return np.einsum('kl...,ki...,lj...->ij...', tnsr, drdz, drdz)

    # (t, r, theta, phi) -> (t, R, phi, z)
    def sph_to_cyl_cotnsr(self, cotnsr, **kwargs):
        return self.cyl_to_sph_tnsr(cotnsr, **kwargs)

    # -------------------------------------
    # Cylindrical (orthonormal) to Spherical
    # -------------------------------------
    # (t, R, phi, z) -> (t, r, theta, phi)
    def cyl_to_sph_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        dzdr = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        
        # Transformation to Spherical Kerr Schild from Cylindrical Kerr Schild
        dzdr[0, 0] = 1.
        dzdr[1, 1] = np.sin(self.th)
        dzdr[1, 2] = 0.
        dzdr[1, 3] = np.cos(self.th)
        dzdr[2, 1] = (1. / self.r) * np.cos(self.th)
        dzdr[2, 3] = -(1. / self.r) * np.sin(self.th)
        dzdr[3, 2] = (1. / self.r) * (1. / np.sin(self.th))

        if not kwargs.get('cpp', False):
            return np.einsum('ij...,j...->i...', dzdr, vec)
        else:
            return Tools.contract_vector_cpp(vec, dzdr)

    # (t, R, phi, z) -> (t, r, theta, phi)
    def cyl_to_sph_one_form(self, one_form, **kwargs):
        return self.sph_to_cyl_vec(one_form, **kwargs)

    # (t, R, phi, z) -> (t, r, theta, phi)
    def cyl_to_sph_tnsr(self, tnsr, **kwargs):
        shape = list(self.r.shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dzdr = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        
        # Transformation to Spherical Kerr Schild from Cylindrical Kerr Schild
        dzdr[0, 0] = 1.
        dzdr[1, 1] = np.sin(self.th)
        dzdr[1, 2] = 0.
        dzdr[1, 3] = np.cos(self.th)
        dzdr[2, 1] = (1. / self.r) * np.cos(self.th)
        dzdr[2, 3] = -(1. / self.r) * np.sin(self.th)
        dzdr[3, 2] = (1. / self.r) * (1. / np.sin(self.th))
        
        return np.einsum('kl...,ki...,lj...->ij...', tnsr, dzdr, dzdr)

    # (t, R, phi, z) -> (t, r, theta, phi)
    def cyl_to_sph_cotnsr(self, cotnsr, **kwargs):
        return self.sph_to_cyl_tnsr(cotnsr, **kwargs)

    # -------------------------------------
    # Internal to Spherical
    # -------------------------------------
    # (t, x1, x2, x3) -> (t, r, theta, phi) 
    def int_to_sph_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        dxdxp = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        # Transformation to Spherical Kerr Schild from internal coordinates
        dxdxp[0, 0] = 1.0
        dxdxp[1, 1] = self.r
        dxdxp[2, 2] = np.pi / 2.0
        dxdxp[3, 3] = 1.0

        if not kwargs.get('cpp', False):
            return np.einsum('ij...,j...->i...', dxdxp, vec)
        else:
            return Tools.contract_vector_cpp(vec, dxdxp)

    # (t, x1, x2, x3) -> (t, r, theta, phi) 
    def int_to_sph_one_form(self, one_form, **kwargs):
        return self.sph_to_int_vec(one_form, **kwargs)

    # (t, x1, x2, x3) -> (t, r, theta, phi) 
    def int_to_sph_tnsr(self, tnsr, **kwargs):
        shape = list(self.r.shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dxdxp = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        # Transformation to Spherical Kerr Schild from internal coordinates
        dxdxp[0, 0] = 1.0
        dxdxp[1, 1] = self.r
        dxdxp[2, 2] = np.pi / 2.0
        dxdxp[3, 3] = 1.0

        return np.einsum('kl...,ki...,lj...->ij...', tnsr, dxdxp, dxdxp)
    
    # (t, x1, x2, x3) -> (t, r, theta, phi) 
    def int_to_sph_cotnsr(self, cotnsr, **kwargs):
        return self.sph_to_int_tnsr(cotnsr, **kwargs)

    # -------------------------------------
    # Spherical to Internal
    # -------------------------------------
    # (t, r, theta, phi) -> (t, x1, x2, x3) 
    def sph_to_int_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        dxpdx = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        # Transformation to internal coordinates from Spherical Kerr Schild
        dxpdx[0, 0] = 1.0
        dxpdx[1, 1] = 1. / self.r
        dxpdx[2, 2] = 2.0 / np.pi
        dxpdx[3, 3] = 1.0

        if not kwargs.get('cpp', False):
            return np.einsum('ij...,j...->i...', dxpdx, vec)
        else:
            return Tools.contract_vector_cpp(vec, dxpdx)

    # (t, r, theta, phi) -> (t, x1, x2, x3) 
    def sph_to_int_one_form(self, one_form, **kwargs):
        return self.int_to_sph_vec(one_form, **kwargs)

    # (t, r, theta, phi) -> (t, x1, x2, x3) 
    def sph_to_int_tnsr(self, tnsr, **kwargs):
        shape = list(self.r.shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dxpdx = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        
        # Transformation to internal coordinates from Spherical Kerr Schild
        dxpdx[0, 0] = 1.0
        dxpdx[1, 1] = 1. / self.r
        dxpdx[2, 2] = 2.0 / np.pi
        dxpdx[3, 3] = 1.0
        
        return np.einsum('kl...,ki...,lj...->ij...', tnsr, dxpdx, dxpdx)

    # (t, r, theta, phi) -> (t, x1, x2, x3) 
    def sph_to_int_cotnsr(self, cotnsr, **kwargs):
        return self.int_to_sph_tnsr(cotnsr, **kwargs)

    # -------------------------------------
    # Internal to Cartesian (orthonormal)
    # -------------------------------------
    # (t, x1, x2, x3) -> (t, x, y, z) 
    def int_to_cart_vec(self, vec, **kwargs):
        return self.sph_to_cart_vec(
            self.int_to_sph_vec(vec, **kwargs), **kwargs)

    # (t, x1, x2, x3) -> (t, x, y, z) 
    def int_to_cart_one_form(self, one_form, **kwargs):
        return self.sph_to_cart_one_form(
            self.int_to_sph_one_form(one_form, **kwargs), **kwargs)

    # (t, x1, x2, x3) -> (t, x, y, z) 
    def int_to_cart_tnsr(self, tnsr, **kwargs):
        return self.sph_to_cart_tnsr(
            self.int_to_sph_tnsr(tnsr, **kwargs), **kwargs)

    # (t, x1, x2, x3) -> (t, x, y, z) 
    def int_to_cart_cotnsr(self, cotnsr, **kwargs):
        return self.sph_to_cart_cotnsr(
            self.int_to_sph_cotnsr(cotnsr, **kwargs), **kwargs)

    # -------------------------------------
    # Internal to Cylindrical (othornormal)
    # -------------------------------------
    # (t, x1, x2, x3) -> (t, R, phi, z) 
    def int_to_cyl_vec(self, vec, **kwargs):
        return self.sph_to_cyl_vec(
            self.int_to_sph_vec(vec, **kwargs), **kwargs)
    
    # (t, x1, x2, x3) -> (t, R, phi, z) 
    def int_to_cyl_one_form(self, one_form, **kwargs):
        return self.sph_to_cyl_one_form(
            self.int_to_sph_one_form(one_form, **kwargs), **kwargs)
    
    # (t, x1, x2, x3) -> (t, R, phi, z) 
    def int_to_cyl_tnsr(self, tnsr, **kwargs):
        return self.sph_to_cyl_tnsr(
            self.int_to_sph_tnsr(tnsr, **kwargs), **kwargs)
    
    # (t, x1, x2, x3) -> (t, R, phi, z) 
    def int_to_cyl_cotnsr(self, cotnsr, **kwargs):
        return self.sph_to_cyl_cotnsr(
            self.int_to_sph_cotnsr(cotnsr, **kwargs), **kwargs)

    # -------------------------------------
    # Internal to Spherical (orthonormal)
    # -------------------------------------
    # (t, x1, x2, x3) -> (t, r, theta, phi) 
    def int_to_ortho_sph_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        dxdxp = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        # Transformation to orthonormal Spherical Kerr Schild 
        # from internal coordinates
        dxdxp[0, 0] = 1.0
        dxdxp[1, 1] = self.r
        dxdxp[2, 2] = (np.pi / 2.0) * self.r
        dxdxp[3, 3] = self.r

        if not kwargs.get('cpp', False):
            return np.einsum('ij...,j...->i...', dxdxp, vec)
        else:
            return Tools.contract_vector_cpp(vec, dxdxp)

    # (t, x1, x2, x3) -> (t, r, theta, phi) 
    def int_to_ortho_sph_one_form(self, one_form, **kwargs):
        return self.ortho_sph_to_int_vec(one_form, **kwargs)

    # (t, x1, x2, x3) -> (t, r, theta, phi) 
    def int_to_ortho_sph_tnsr(self, tnsr, **kwargs):
        shape = list(self.r.shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dxdxp = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        # Transformation to Spherical Kerr Schild from internal coordinates
        dxdxp[0, 0] = 1.0
        dxdxp[1, 1] = self.r
        dxdxp[2, 2] = (np.pi / 2.0) * self.r
        dxdxp[3, 3] = self.r

        return np.einsum('kl...,ki...,lj...->ij...', tnsr, dxdxp, dxdxp)
    
    # (t, x1, x2, x3) -> (t, r, theta, phi) 
    def int_to_ortho_sph_cotnsr(self, cotnsr, **kwargs):
        return self.ortho_sph_to_int_tnsr(cotnsr, **kwargs) 

    # -------------------------------------
    # Spherical (orthonormal) to Internal
    # -------------------------------------
    #  (t, r, theta, phi) -> (t, x1, x2, x3)
    def ortho_sph_to_int_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        dxpdx = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        # Transformation to orthonormal Spherical Kerr Schild 
        # from internal coordinates
        dxpdx[0, 0] = 1.0
        dxpdx[1, 1] = 1. / self.r
        dxpdx[2, 2] = (2.0 / np.pi) * (1. / self.r)
        dxpdx[3, 3] = 1. / self.r

        if not kwargs.get('cpp', False):
            return np.einsum('ij...,j...->i...', dxpdx, vec)
        else:
            return Tools.contract_vector_cpp(vec, dxpdx)

    # (t, r, theta, phi) -> (t, x1, x2, x3)
    def ortho_sph_to_int_one_form(self, one_form, **kwargs):
        return self.int_to_ortho_sph_vec(one_form, **kwargs)

    # (t, r, theta, phi) -> (t, x1, x2, x3)
    def ortho_sph_to_int_tnsr(self, tnsr, **kwargs):
        shape = list(self.r.shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dxpdx = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        # Transformation to Spherical Kerr Schild from internal coordinates
        dxpdx[0, 0] = 1.0
        dxpdx[1, 1] = 1. / self.r
        dxpdx[2, 2] = (2.0 / np.pi) * (1. / self.r)
        dxpdx[3, 3] = 1. / self.r

        return np.einsum('kl...,ki...,lj...->ij...', tnsr, dxpdx, dxpdx)
    
    # (t, r, theta, phi) -> (t, x1, x2, x3)
    def ortho_sph_to_int_cotnsr(self, cotnsr, **kwargs):
        return self.int_to_ortho_sph_tnsr(cotnsr, **kwargs) 

    # Compute dot product between two vectors in the desired
    # coordinate system. Assumes vec_a, vec_b, and gcov are in
    # internal coordinates. Memory intensive on 3D arrays.
    # As an alternative, one could compute the one-form of the vector a,
    # and then convert to the desired coordinate system; however, this
    # requires computing the inverse of the coordinate transformation matrix.
    def dot(self, vec_a_in, vec_b_in, gcov, coord_system, **kwargs):
        vec_a = np.copy(vec_a_in)
        vec_b = np.copy(vec_b_in)
        if (coord_system == 'internal'):
            if 'component' in kwargs:
                vec_a, vec_b = self.isolate_component(vec_a, vec_b, **kwargs)
            return Tools.dot(vec_a, vec_b, gcov, **kwargs)
        elif (coord_system == 'spherical'):
            gcov_sph  = self.int_to_sph_cotnsr(gcov)
            vec_a_sph = self.int_to_sph_vec(vec_a)
            vec_b_sph = self.int_to_sph_vec(vec_b)
            if 'component' in kwargs:
                vec_a_sph, vec_b_sph = self.isolate_component(
                    vec_a_sph, vec_b_sph, **kwargs)
            return Tools.dot(vec_a_sph, vec_b_sph, gcov_sph, **kwargs)
        elif (coord_system == 'cylindrical'):
            gcov_cyl  = self.int_to_cyl_cotnsr(gcov)
            vec_a_cyl = self.int_to_cyl_vec(vec_a)
            vec_b_cyl = self.int_to_cyl_vec(vec_b)
            if 'component' in kwargs:
                vec_a_cyl, vec_b_cyl = self.isolate_component(
                    vec_a_cyl, vec_b_cyl, **kwargs)
            return Tools.dot(vec_a_cyl, vec_b_cyl, gcov_cyl, **kwargs)
        elif (coord_system == 'Cartesian'):
            gcov_cart  = self.int_to_cart_cotnsr(gcov)
            vec_a_cart = self.int_to_cart_vec(vec_a)
            vec_b_cart = self.int_to_cart_vec(vec_b)
            if 'component' in kwargs:
                vec_a_cart, vec_b_cart = self.isolate_component(
                    vec_a_cart, vec_b_cart, **kwargs)
            return Tools.dot(vec_a_cart, vec_b_cart, gcov_cart, **kwargs)

    # Zero all components other than kwargs.get('component')
    def isolate_component(self, vec_a, vec_b, **kwargs):
        indices_to_zero = [1, 2, 3]
        for i in range(0, 4):
            if (kwargs.get('component') == int(i)):
                if (int(i) != 0):
                    indices_to_zero.remove(int(i))
        for index in indices_to_zero:
            vec_a[index] = 0.
            vec_b[index] = 0.
        return vec_a, vec_b

class Integrals(Coordinates):
    def __init__(self, r, theta, phi, pars, CONSTANTS, **kwargs):
        super().__init__(r, theta, phi, pars, CONSTANTS, **kwargs)

    # symmetric \int_{0}^{\theta} dth q(r, th, phi)
    def cum_integrate_th(self, q, gcov22, **kwargs):
        int_qdth = np.zeros(q.shape)
        qdth = q * np.sqrt(gcov22) * self.dx2 * (
            self.sigma_cut(**kwargs) * self.theta_cut(**kwargs))
        shape = list(q.shape)
        if (len(shape) == 2):
            ith0 = np.argwhere(self.th[0, :] < np.pi/2)[-1][0] + 1
            int_qdth[:, :ith0] = np.cumsum(qdth[:, :ith0], axis=-1)
            int_qdth[:, ith0:] = np.cumsum(qdth[:, ith0:][:, ::-1],
                                           axis=-1)[:, ::-1]
        elif (len(shape) == 3):
            ith0 = np.argwhere(self.th[0, :, 0] < np.pi/2)[-1][0] + 1
            int_qdth[:, :ith0, :] = np.cumsum(qdth[:, :ith0, :], axis=1)
            int_qdth[:, ith0:, :] = np.cumsum(qdth[:, ith0:, :][:, ::-1, :],
                                              axis=1)[:, ::-1, :]
        elif (len(shape) == 4):
            ith0 = np.argwhere(self.th[0, 0, :, 0] < np.pi/2)[-1][0] + 1
            int_qdth[:, :, :ith0, :] = np.cumsum(qdth[:, :, :ith0, :], axis=2)
            int_qdth[:, :, ith0:, :] = np.cumsum(
                qdth[:, :, ith0:, :][:, :, ::-1, :], axis=2)[:, :, ::-1, :]
        return int_qdth

    # -\int_{\infty}^{r} dr q(r, th, phi)
    # Note: expects gcov in internal coordinates
    def cum_integrate_r(self, q, gcov11, **kwargs):
        int_qdr = np.zeros(q.shape)
        qdr = q * np.sqrt(gcov11) * self.dx1 * (
            self.sigma_cut(**kwargs) * self.theta_cut(**kwargs))
        shape = list(q.shape)
        if (len(shape) == 2):
            int_qdr = np.cumsum(qdr[::-1, :], axis=0)[::-1, :]
        elif (len(shape) == 3):
            int_qdr = np.cumsum(qdr[::-1, :, :], axis=0)[::-1, :, :]
        elif (len(shape) == 4):
            int_qdr = np.cumsum(qdr[:, ::-1, :, :], axis=1)[:, ::-1, :, :]
        return int_qdr

    def cum_integrate_z(self, q, gcovZZ, **kwargs):
        # note that this only works with a 2D array
        if (kwargs.get('interpolate', False)):
            return self.cum_integrate_z_interpolated(q, gcovZZ, **kwargs)
        int_qdz = np.zeros(q.shape)
        z       = self.r * np.cos(self.th)
        dz      = np.sqrt(gcovZZ) * self.r * (
            np.abs(self.dx1 * np.cos(self.th)) + 
            np.abs((np.pi/2.) * self.dx2 * np.sin(self.th))
        )
        R_cyl   = self.r * np.sin(self.th)
        dR_cyl  = self.r * self.dx1 * np.sin(self.th) 
        Rcyl_min= R_cyl - 0.5 * dR_cyl
        Rcyl_max= R_cyl + 0.5 * dR_cyl

        qdz = q * dz * self.sigma_cut(**kwargs) * self.theta_cut(**kwargs)
        int_qdz = self.int_dz_loop(int_qdz, qdz, self.r, self.th, z, 
                                   R_cyl, Rcyl_min, Rcyl_max)
        
        return int_qdz

    @staticmethod
    @jit(nopython=True, parallel=True)
    def int_dz_loop(int_qdz, qdz, r, th, z, R_cyl, Rcyl_min, Rcyl_max):
        for ir in prange(len(r[:, 0])):
            for ith in range(len(th[0, :])):
                temp_sum = 0.0
                for jr in range(len(r[:, 0])):
                    for jth in range(len(th[0, :])):
                        if ((R_cyl[jr, jth] < Rcyl_max[ir, ith]) and
                            (R_cyl[jr, jth] > Rcyl_min[ir, ith])):
                            if (z[ir, ith] >= 0.):
                                if (z[jr, jth] > z[ir, ith]):
                                    temp_sum += qdz[jr, jth]
                            else:
                                if (z[jr, jth] < z[ir, ith]):
                                    temp_sum += qdz[jr, jth]

                int_qdz[ir, ith] = temp_sum
        return int_qdz
            
    def cum_integrate_z_interpolated(self, q, gcovZZ, **kwargs):
        R_cyl = self.r * np.sin(self.th)
        z_cyl = self.r * np.cos(self.th)
        f     = RegularGridInterpolator((R_cyl, z_cyl), (q * np.sqrt(gcovZZ)))
        R1d   = self.r[:, 0]
        z1d   = np.linspace(np.amin(z_cyl), np.amax(z_cyl), len(self.th[0, :]))
        R_grid, z_grid = np.meshgrid(R1d, z1d)
        q_grid         = f([R_grid, z_grid])
        # This is wrong
        dz_grid        = np.concatenate([np.diff(z_grid), np.diff(z_grid)[-1]])
        int_qdz        = np.cumsum(q_grid * dz_grid)
        return int_qdz

    # Return where sigma < sigma_cut
    def sigma_cut(self, **kwargs):
        if 'sigma_cut' in kwargs and 'sigma' in kwargs:
            sigma = kwargs.get('sigma')
            return (sigma < kwargs.get('sigma_cut'))
        elif 'sigma_cut' in kwargs and 'sigma' not in kwargs:
            raise TypeError("Missing sigma in kwargs.")
        elif 'sigma_cut' not in kwargs and 'sigma' in kwargs:
            raise TypeError("Must specify value of sigma_cut.")
        else:
            return 1.0

    def theta_cut(self, **kwargs):
        if 'theta_cut' in kwargs:
            return ((self.th > kwargs.get('theta_cut')) & 
                    (self.th < np.pi - kwargs.get('theta_cut')))
        else:
            return 1.0

class Kerr_Schild(Coordinates):
    def __init__(self, r, theta, phi, pars, CONSTANTS, **kwargs):
        super().__init__(r, theta, phi, pars, CONSTANTS, **kwargs)
        
        # Appendix C of White (2016) Ph.D. Thesis
        self.a     = pars['a']
        self.Xi    = (self.r * self.r) - (pars['a'] * np.cos(self.th)) ** 2
        self.Sigma = (self.r * self.r) + (pars['a'] * np.cos(self.th)) ** 2
        self.Delta = (self.r * self.r) - 2. * self.r + (self.a * self.a)

    def calc_lapse(self, **kwargs):
        return (1. + 2. * self.r / self.Sigma) ** (-1/2)

    def calc_shift(self, **kwargs):
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 3)
        beta    = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        beta[0] = 2. * (self.r / self.Sigma) / (1. + 2. * self.r / self.Sigma)
        return beta
    
    def calc_Minkowski(self, **kwargs):
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        g_Mink = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        g_Mink[0, 0] = -1.
        g_Mink[1, 1] = 1.
        g_Mink[2, 2] = self.r * self.r
        g_Mink[3, 3] = (self.r * np.sin(self.th)) ** 2.
        return g_Mink
    
class Tetrads(Kerr_Schild):
    def __init__(self, r, theta, phi, pars, CONSTANTS, **kwargs):
        super().__init__(r, theta, phi, pars, CONSTANTS, **kwargs)
        shape = [4, 4] + list((self.r * self.th * self.ph).shape)
        self.e = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        system = kwargs.get('system', False)
        if (system == 'Takahashi_Kerr_Schild'):
            self.calc_Takahashi_KS()
        else:
            print("Please specify a valid coordinate system.")

    def calc_Takahashi_KS(self):
        alpha   = self.calc_lapse()
        beta    = self.calc_shift()
        bigA    = ((self.r * self.r + self.a * self.a) ** 2 - 
                   (self.a * self.a) * self.Delta * (np.sin(self.th)) ** 2)
        gammarr = 1. + 2. * self.r / self.Sigma
        gammarp = -self.a * (np.sin(self.th)) ** 2 * (1. + 2. * self.r / self.Sigma)
        gammapp = bigA * (np.sin(self.th)) ** 2 / self.Sigma
        
        # Equations 11-14 in Takahashi (2008)
        # Eq. 11
        self.e[0, 0] = alpha
        # Eq. 12
        self.e[1, 0] = beta[0] / np.sqrt(gammarr)
        self.e[1, 1] = 1. / np.sqrt(gammarr)
        # Eq. 13
        self.e[2, 2] = np.sqrt(self.Sigma)
        # Eq. 14
        self.e[3, 0] = beta[0] * gammarp / np.sqrt(gammapp)
        self.e[3, 1] = gammarp / np.sqrt(gammapp)
        self.e[3, 3] = np.sqrt(gammapp)

class Tilt(Coordinates):
    def __init__(self, r, theta, phi, pars, CONSTANTS, **kwargs):
        super().__init__(r, theta, phi, pars, CONSTANTS, **kwargs)

class Christoffel_KS(Kerr_Schild):
    def __init__(self, r, theta, phi, pars, CONSTANTS, **kwargs):
        super().__init__(r, theta, phi, pars, CONSTANTS, **kwargs)
        shape = [4, 4, 4] + list((self.r * self.th * self.ph).shape)
        self.g = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        self.calc_gt()
        self.calc_gr()
        self.calc_gth()
        self.calc_gphi()

        self.g_int = self.sph_to_int_Christoffel(self.g, **kwargs)

    # Equations C74 - C77 in White (2016) Ph.D. Thesis
    def calc_gt(self):
        a = self.a
        # Equations C.74a - C.74d
        self.g[0, 0, 0] = 2. * self.Xi * self.r / (self.Sigma ** 3)
        self.g[0, 0, 1] = (self.Xi / self.Sigma ** 2) * (
            1. + 2. * self.r / self.Sigma)
        self.g[0, 1, 0] = self.g[0, 0, 1]
        self.g[0, 0, 2] = -(2. * a * a * self.r / self.Sigma ** 2) * (
            np.sin(self.th) * np.cos(self.th)
        )
        self.g[0, 2, 0] = self.g[0, 0, 2]
        self.g[0, 0, 3] = -(2. * a * self.r * self.Xi / self.Sigma ** 3) * (
            np.sin(self.th) ** 2
        )
        self.g[0, 3, 0] = self.g[0, 0, 3]

        # Equations C.74e - C.74g
        self.g[0, 1, 1] = (2. * self.Xi / self.Sigma ** 2) * (
            1. + self.r / self.Sigma
        )
        self.g[0, 1, 2] = -(2. * a * a * self.r / self.Sigma ** 2) * (
            np.sin(self.th) * np.cos(self.th)
        )
        self.g[0, 2, 1] = self.g[0, 1, 2]
        self.g[0, 1, 3] = -(a * self.Xi / self.Sigma ** 2) * (
            1. + 2. * self.r / self.Sigma) * (np.sin(self.th) ** 2)
        self.g[0, 3, 1] = self.g[0, 1, 3]

        # Equations C.74h - C.74j
        self.g[0, 2, 2] = -2. * self.r * self.r / self.Sigma
        self.g[0, 2, 3] = (2. * a * a * a * self.r / self.Sigma ** 2) * (
            np.sin(self.th) ** 3 * self.cos(self.th)
        )
        self.g[0, 3, 2] = self.g[0, 2, 3]
        self.g[0, 3, 3] = -(2. * self.r / self.Sigma) * (
            self.r - (a*a * self.Xi / self.Sigma ** 2) * np.sin(self.th) ** 2
        ) * np.sin(self.th) ** 2

    def calc_gr(self):
        a = self.a
        # Equations C.75a - 
        self.g[1, 0, 0] = self.Delta * self.Xi / self.Sigma ** 3
        self.g[1, 0, 1] = self.Xi * (self.Delta - self.Sigma) / self.Sigma**3
        self.g[1, 1, 0] = self.g[1, 0, 1]
        self.g[1, 0, 3] = -(a * self.Delta * self.Xi / self.Sigma ** 3) * (
            np.sin(self.th) ** 2
        )
        self.g[1, 3, 0] = self.g[1, 0, 3]
        self.g[1, 1, 1] = (
            self.Xi * (self.Delta - 2. * self.Sigma) / self.Sigma ** 3
        )
        self.g[1, 1, 2] = -(a*a/self.Sigma) * np.sin(self.th)*np.cos(self.th)
        self.g[1, 2, 1] = self.g[1, 1, 2]
        self.g[1, 1, 3] = (a / self.Sigma ** 3) * (
            self.r ** 3 * (self.r **2 + 2.) + a * a * (
            self.r * a * a * np.power(np.cos(self.th), 4) + (
            (2. * self.r ** 3 + self.Delta - self.Sigma) * np.cos(self.th) ** 2
            - (self.r * self.r * (np.sin(self.th) ** 2))
            ) 
            )
        )
        self.g[1, 3, 1] = self.g[1, 1, 3]
        self.g[1, 2, 2] = -self.r * self.Delta / self.Sigma
        self.g[1, 3, 3] = -(self.Delta / self.Sigma) * (
            self.r - (a * a * self.Xi / self.Sigma ** 2) * np.sin(self.th) ** 2
        ) * np.sin(self.th) ** 2

    def calc_gth(self):
        a = self.a

        self.g[2, 0, 0] = -(2. * a * a * self.r / self.Sigma ** 3) * (
            np.sin(self.th) * np.cos(self.th)
        )
        self.g[2, 0, 1] = self.g[2, 0, 0]
        self.g[2, 1, 0] = self.g[2, 0, 0]
        self.g[2, 0, 3] = (2. * self.a * self.r / self.Sigma ** 3) * (
            (self.r * self.r + a * a) * np.sin(self.th) * np.cos(self.th)
        )
        self.g[2, 3, 0] = self.g[2, 0, 3]
        self.g[2, 1, 1] = self.g[2, 0, 0]
        self.g[2, 1, 2] = self.r / self.Sigma
        self.g[2, 2, 1] = self.g[2, 1, 2]
        self.g[2, 1, 3] = (a / self.Sigma) * (
            (1. + (2. * self.r / self.Sigma ** 2) * (self.r **2 + a**2))) * (
            np.sin(self.th) * np.cos(self.th)
        )
        self.g[2, 2, 2] = -(a * a / self.Sigma) * (
            np.sin(self.th) * np.cos(self.th)
        )
        self.g[2, 3, 3] = -(1. / self.Sigma) * (
            self.Delta + (2. * self.r / self.Sigma ** 2) * (
            np.square(self.r **2 + a*a))) * np.sin(self.th) * np.cos(self.th)

    def calc_gphi(self):
        a = self.a
        self.g[3, 0, 0] = a * self.Xi / self.Sigma ** 3
        self.g[3, 0, 1] = self.g[3, 0, 0]
        self.g[3, 1, 0] = self.g[3, 0, 0]
        self.g[3, 0, 2] = -(2. * a * self.r / self.Sigma ** 2) * (
            1. / np.tan(self.th)
        )
        self.g[3, 2, 0] = self.g[3, 0, 2]
        self.g[3, 0, 3] = -(a * a * self.Xi / self.Sigma ** 3) * (
            np.sin(self.th) ** 2
        )
        self.g[3, 3, 0] = self.g[3, 0, 3]
        self.g[3, 1, 1] = self.g[3, 0, 0]
        self.g[3, 1, 2] = -(a/self.Sigma**2) * (2. * self.r + self.Sigma) * (
            1. / np.tan(self.th)
        )
        self.g[3, 2, 1] = self.g[3, 1, 2]
        self.g[3, 1, 3] = (1. / self.Sigma ** 3) * (
            self.r * self.Sigma**2 - (a*a * self.Xi * (np.sin(self.th) ** 2))
        )
        self.g[3, 3, 1] = self.g[3, 1, 3]
        self.g[3, 2, 2] = -a * self.r / self.Sigma
        self.g[3, 2, 3] = ((2. * a * a * self.r / self.Sigma ** 2) * (
            np.sin(self.th) * np.cos(self.th)) + (1. / np.tan(self.th))
        )
        self.g[3, 3, 2] = self.g[3, 2, 3]
        self.g[3, 3, 3] = (a/self.Sigma ** 3) * (
            (a*a) * self.Xi * (np.sin(self.th) ** 2) - self.r * (self.Sigma**2)
        ) * (np.sin(self.th) ** 2)
        
    def sph_to_int_Christoffel(self, Gamma, **kwargs):
        shape = [4, 4] + list((self.r * self.th * self.ph).shape)
        dxdxp = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        dxpdx = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        # Transformation to Spherical Kerr Schild from internal coordinates
        # for contravariant index
        dxdxp[0, 0] = 1.0
        dxdxp[1, 1] = self.r
        dxdxp[2, 2] = np.pi / 2.0
        dxdxp[3, 3] = 1.0
        # Transformation to Spherical Kerr Schild from internal coordinates
        # for covariant index
        dxpdx[0, 0] = 1.0
        dxpdx[1, 1] = 1. / self.r
        dxpdx[2, 2] = 2.0 / np.pi
        dxpdx[3, 3] = 1.0

        return np.einsum('lmn...,li...,mj...,nk...->ijk...', 
                         Gamma, dxdxp, dxdxp, dxpdx)