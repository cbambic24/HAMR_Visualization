import numpy as np
from numba import jit
from numba import prange
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import tracemalloc
from analysis import Accretion
from analysis import Tools

class Coordinates():
    def __init__(self, r, theta, phi, pars, CONSTANTS, **kwargs):
        self.dx1, self.dx2, self.dx3 = pars['dx1'], pars['dx2'], pars['dx3']

        self.cgs  = kwargs.get('cgs', False)
        self.MBH  = kwargs.get('MBH', 1.0) if self.cgs else 1. # g
        self.c    = CONSTANTS['c']         if self.cgs else 1. # cm/s
        self.G    = CONSTANTS['G']         if self.cgs else 1. # cm^3/g s^2
        self.Msun = CONSTANTS['Msun']      if self.cgs else 1. # g
        Rg        = (self.G * self.MBH * self.Msun) / (self.c * self.c)

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
        shape = list((self.r * self.th * self.ph).shape)
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
        shape = list((self.r * self.th * self.ph).shape)
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
        shape = list((self.r * self.th * self.ph).shape)
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
        shape = list((self.r * self.th * self.ph).shape)
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
        shape = list((self.r * self.th * self.ph).shape)
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
        shape = list((self.r * self.th * self.ph).shape)
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
        shape = list((self.r * self.th * self.ph).shape)
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
    
    # -------------------------------------
    # Spherical (orthonormal) to Cylindrical (orthonormal)
    # -------------------------------------
    # (t, r, theta, phi) -> (t, R, phi, z)
    def ortho_sph_to_cyl_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        drdz = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))

        # Transformation to Cylindrical Kerr Schild from Spherical Kerr Schild
        drdz[0, 0] = 1.
        drdz[1, 1] = np.sin(self.th)
        drdz[1, 2] = np.cos(self.th)
        drdz[2, 3] = np.sin(self.th)
        drdz[3, 1] = np.cos(self.th)
        drdz[3, 2] = -np.sin(self.th)

        if not kwargs.get('cpp', False):
            return np.einsum('ij...,j...->i...', drdz, vec)
        else:
            return Tools.contract_vector_cpp(vec, drdz)
    
    # (t, r, theta, phi) -> (t, R, phi, z)
    def ortho_sph_to_cyl_one_form(self, one_form, **kwargs):
        return self.cyl_to_ortho_sph_vec(one_form, **kwargs)

    # (t, r, theta, phi) -> (t, R, phi, z)
    def ortho_sph_to_cyl_tnsr(self, tnsr, **kwargs):
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        drdz = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        
        # Transformation to Cylindrical Kerr Schild from Spherical Kerr Schild
        drdz[0, 0] = 1.
        drdz[1, 1] = np.sin(self.th)
        drdz[1, 2] = np.cos(self.th)
        drdz[2, 3] = np.sin(self.th)
        drdz[3, 1] = np.cos(self.th)
        drdz[3, 2] = np.sin(self.th)
        
        return np.einsum('kl...,ki...,lj...->ij...', tnsr, drdz, drdz)

    # (t, r, theta, phi) -> (t, R, phi, z)
    def ortho_sph_to_cyl_cotnsr(self, cotnsr, **kwargs):
        return self.cyl_to_ortho_sph_tnsr(cotnsr, **kwargs)

    # -------------------------------------
    # Cylindrical (orthonormal) to Spherical (orthonormal)
    # -------------------------------------
    # (t, R, phi, z) -> (t, r, theta, phi)
    def cyl_to_ortho_sph_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        dzdr = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        
        # Transformation to Spherical Kerr Schild from Cylindrical Kerr Schild
        dzdr[0, 0] = 1.
        dzdr[1, 1] = np.sin(self.th)
        dzdr[1, 2] = 0.
        dzdr[1, 3] = np.cos(self.th)
        dzdr[2, 1] = np.cos(self.th)
        dzdr[2, 3] = -np.sin(self.th)
        dzdr[3, 2] = (1. / np.sin(self.th))

        if not kwargs.get('cpp', False):
            return np.einsum('ij...,j...->i...', dzdr, vec)
        else:
            return Tools.contract_vector_cpp(vec, dzdr)

    # (t, R, phi, z) -> (t, r, theta, phi)
    def cyl_to_ortho_sph_one_form(self, one_form, **kwargs):
        return self.ortho_sph_to_cyl_vec(one_form, **kwargs)

    # (t, R, phi, z) -> (t, r, theta, phi)
    def cyl_to_ortho_sph_tnsr(self, tnsr, **kwargs):
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dzdr = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        
        # Transformation to Spherical Kerr Schild from Cylindrical Kerr Schild
        dzdr[0, 0] = 1.
        dzdr[1, 1] = np.sin(self.th)
        dzdr[1, 2] = 0.
        dzdr[1, 3] = np.cos(self.th)
        dzdr[2, 1] = np.cos(self.th)
        dzdr[2, 3] = -np.sin(self.th)
        dzdr[3, 2] = (1. / np.sin(self.th))
        
        return np.einsum('kl...,ki...,lj...->ij...', tnsr, dzdr, dzdr)

    # (t, R, phi, z) -> (t, r, theta, phi)
    def cyl_to_sph_cotnsr(self, cotnsr, **kwargs):
        return self.ortho_sph_to_cyl_tnsr(cotnsr, **kwargs)

    # Calculate the Lorentz boost matrix Lambda^mu_alpha 
    # See Sadowski+2013 Equation 10 (with - signs on t-i components)
    # Boost is from the unprimed to the primed frame
    # uu is the velocity of the primed frame as measured in the unprimed frame
    # uu is assumed dimensionless
    def calc_Lorentz_boost(self, uu, **kwargs):
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        Lambda = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        Lambda[0,  0] =  uu[0]
        Lambda[1:, 0] = -uu[1:]
        Lambda[0, 1:] = -uu[1:]

        if not kwargs.get('inverse', False):
            v = uu / uu[0]
        elif kwargs.get('inverse', False):
            v = -uu / uu[0]
        v2 = v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
        for mu in range(3):
            for nu in range(3):
                Lambda[mu+1, nu+1] = (((uu[0] - 1.) / v2) * (
                    v[mu + 1] * v[nu + 1]) + (mu == nu))
        return Lambda

    def boost_vector(self, vec, uu_c, **kwargs):
        if (kwargs.get('cpp', False)):
            return self.boost_vector_cpp(vec, uu_c, **kwargs)
        Lambda = self.calc_Lorentz_boost(uu_c, **kwargs)
        return np.einsum('ij...,j...->i...', Lambda, vec)

    def boost_vector_cpp(self, vec, uu_c, **kwargs):
        import hamr_readc
        Toolsc = hamr_readc.PyTools()
        dims   = list(vec.shape)
        dtype  = np.float32 # C++ layer expects float 
        if (len(dims) != 5 or dims[0] != 4):
            raise ValueError("This 4-vector with shape " +
                             str(dims) + " cannot be passed to C++.")
        [ndim, nb, nx1, nx2, nx3] = dims
        vec_boost = np.zeros(vec.shape, dtype=dtype)
        uu_dims   = list(uu_c.shape)
        if (len(uu_dims) != 5 or uu_dims[0] != 4):
            raise ValueError("The 4-velocity with shape " +
                             str(uu_dims) + " cannot be passed to C++.")
        inverse = 1 if kwargs.get('inverse', False) else 0
        Toolsc.boost_vector(np.float32(vec), vec_boost, np.float32(uu_c), 
            nb, nx1, nx2, nx3, inverse)
        return vec_boost

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

    # symmetric -\int_{\infty}^{z} dz q(z, R, phi)
    # Note: expects gcov in cylindrical coordinates
    def cum_integrate_z(self, q, gcovZZ, **kwargs):
        dth   = self.th[0, 0, 1, 0] - self.th[0, 0, 0, 0]

        # Apply sigma cut before integrating in z
        if 'sigma_cut' in kwargs:
            sigma = kwargs.get('sigma')
            R_uni, Z_uni, sigma_uni = self.interpolate_sph_to_cyl_3D(
                sigma, self.r[0, 0, 0, 0] * dth, **kwargs)
            kwargs['sigma'] = sigma_uni
            self.sigma_uni = sigma_uni

        R_uni, Z_uni, q_uni = self.interpolate_sph_to_cyl_3D(
            q * np.sqrt(gcovZZ), self.r[0, 0, 0, 0] * dth, **kwargs)

        shape     = list(q.shape)
        int_qdz   = np.zeros(q_uni.shape)

        # Apply theta cut before integrating in z
        if 'theta_cut' in kwargs:
            rsq       = R_uni * R_uni + Z_uni * Z_uni
            theta_uni = np.arccos(Z_uni / np.sqrt(rsq))
            kwargs['theta_uni'] = theta_uni
            kwargs['uni']       = True

        iz0 = np.argwhere(Z_uni[:, 0] < 0)[-1][0] + 1
        dz  = self.get_dz(Z_uni, iz0)
        qdz = q_uni * dz[:, None, None] * (
            self.sigma_cut(**kwargs) * self.theta_cut(**kwargs))

        # Integrate inward from |z_max|
        int_qdz[:iz0,:,:]= np.cumsum(qdz[:iz0,:,:], axis=0)
        int_qdz[iz0:,:,:]= np.cumsum(qdz[iz0:,:,:][::-1,:,:], axis=0)[::-1,:,:]

        int_qdz = self.interpolate_cyl_to_sph_3D(int_qdz, R_uni, Z_uni, **kwargs)
        int_qdz = np.swapaxes(int_qdz, 0, 1)
        if len(shape) == 4:
            return int_qdz[None, :, :, :]
        else:
            return int_qdz

    def get_dz(self, z, iz0):
        # calculate dz using knowledge that grid is sym-logarithmic in z
        dlogz        = np.zeros(z[:, 0].shape)
        dlogz[:-1]   = np.log(np.abs(z[1:, 0]/z[:-1, 0]))
        dlogz[-1]    = dlogz[-2]
        dlogz[iz0-1] = np.abs(dlogz[iz0])
        return z[:, 0] * dlogz

    def interpolate_sph_to_cyl_3D(self, q, rdth, **kwargs):
        # Interpolate q(r, theta, phi) -> q(Z, R, phi)
        r1D, th1D, phi1D = self.get_coords1D()
        
        # Resolution factor in z (default: 2 so 4x as many points in z as th)
        res   = kwargs.get('res', 2)
        # z_max restricts integral from -z_max to z_max (default: r_max)
        z_max = kwargs.get('z_max', r1D[-1])
        zpos  = np.logspace(
            np.log(rdth), np.log(z_max), int(res * len(th1D)), base=np.e)
        z1D  = np.concatenate([-zpos[::-1], zpos])
        # Create uniform R-Z grid (nz, nr)
        R_cyl, Z_cyl = np.meshgrid(r1D, z1D)
        q_cyl        = np.zeros([len(z1D), len(r1D), len(phi1D)])

        # Calculate true R-Z points
        R2D = r1D[:, None] * np.sin(th1D[None, :])
        Z2D = r1D[:, None] * np.cos(th1D[None, :])

        # Return q(th, r, phi) interpolated to new R-Z grid
        q_cyl = self.interpolate3D(q_cyl, R_cyl, Z_cyl, q,
            R2D.ravel(), Z2D.ravel(), len(phi1D), **kwargs)

        return R_cyl, Z_cyl, q_cyl

    def interpolate_cyl_to_sph_3D(self, q, R, Z, **kwargs):
        # Interpolate q(Z, R, phi) -> q(theta, r, phi)
        r1D, th1D, phi1D = self.get_coords1D()
        
        # Calculate true r-theta points
        r2D  = np.sqrt((R * R) + (Z * Z))
        th2D = np.arccos(Z / r2D)

        # Create uniform log(r)-cos(theta) grid (nth, nr)
        logr, costh = np.meshgrid(np.log(r1D), np.cos(th1D))
        q_sph       = np.zeros([len(th1D), len(r1D), len(phi1D)])

        q_sph = self.interpolate3D(q_sph, logr, costh, q[None, :, :, :],
            np.log(r2D).ravel(), np.cos(th2D).ravel(), len(phi1D), **kwargs)

        return q_sph

    def interpolate3D(self, q_uni, x_uni, y_uni, q, x1D, y1D, nk, **kwargs):
        for k in range(nk):
            q_uni[:,:,k]=griddata((x1D,y1D), q[0,:,:,k].ravel(), (x_uni,y_uni), 
                fill_value=0., method=kwargs.get('method','linear'))
        return q_uni

    def get_coords1D(self):
        r1D   = self.r[ 0, :, 0, 0]
        th1D  = self.th[0, 0, :, 0]
        phi1D = self.ph[0, 0, 0, :]
        return r1D, th1D, phi1D

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
        if 'theta_cut' in kwargs and not kwargs.get('uni', False):
            return ((self.th > kwargs.get('theta_cut')) & 
                    (self.th < np.pi - kwargs.get('theta_cut')))
        elif 'theta_cut' in kwargs and kwargs.get('uni', False):
            if 'theta_uni' in kwargs:
                th_uni = kwargs.get('theta_uni')
                return ((th_uni[:, :, None] > kwargs.get('theta_cut')) & 
                        (th_uni[:, :, None] < np.pi - kwargs.get('theta_cut')))
            else:
                raise ValueError("Must supply theta_uni to use uni=True.")
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
    
    def calc_gcov(self, **kwargs):
        # Calculate covariant metric in spherical Kerr-Schild coordinates
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        gcov  = np.zeros(shape, dtype=kwargs.get('dtype', np.float32))
        
        cth  = np.cos(self.th)
        sth  = np.sin(self.th)
        s2   = sth * sth
        rho2 = self.r * self.r + self.a * self.a * cth * cth
        
        gcov[0, 0, 0, :, :, 0] = (-1. + 2. * self.r / rho2)[0, :, :, 0]
        gcov[0, 1, 0, :, :, 0] = (2. * self.r / rho2)[0, :, :, 0]
        gcov[0, 3, 0, :, :, 0] = (-2. * self.a * self.r * s2 / rho2
                                  )[0, :, :, 0]
        gcov[1, 0, 0, :, :, 0] = gcov[0, 1, 0, :, :, 0]
        gcov[1, 1, 0, :, :, 0] =  (1. + 2. * self.r / rho2)[0, :, :, 0]
        gcov[1, 3, 0, :, :, 0] = (
            -self.a * s2 * (1. + 2. * self.r / rho2))[0, :, :, 0]
        gcov[2, 2, 0, :, :, 0] = rho2[0, :, :, 0]
        gcov[3, 0, 0, :, :, 0] = gcov[0, 3, 0, :, :, 0]
        gcov[3, 1, 0, :, :, 0] = gcov[1, 3, 0, :, :, 0]
        gcov[3, 3, 0, :, :, 0] = (s2 * (
            rho2 + self.a * self.a * s2 * (1. + 2. * self.r / rho2))
            )[0, :, :, 0] 
        return gcov
    
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

class Tilt(Accretion, Coordinates):
    def __init__(self, grid, data, pars, CONSTANTS, **kwargs):
        # Tilt angle as a function of radius (1D array)
        if 'tilt_angle' in kwargs:
            self.tilt_angle = kwargs.get('tilt_angle')
        super()._init(self, grid, data, pars, CONSTANTS, **kwargs)
        self.two_temp = True if ('Ti' in data and 'Te' in data) else False
        self.radiation= True if ('u_rad' in data and 'uu_rad' in data) else False

    # Equation 16 of Fragile & Anninos (2005)
    def calc_angular_momentum_vector(self, L, S, S_cov):
        levi_civita = Tools.calc_Levi_Civita(4, np.ones_like(S[1]))
        return np.einsum('ijkl...,ij...,k...', levi_civita, L, S) / (
            np.sqrt(-np.einsum('i...,i...->...', S_cov, S))
        )

    # Equation 17 of Fragile & Anninos (2005)
    def calc_angular_momentum_tnsr(self, **kwargs):
        if 'THD' not in kwargs:
            EOS = '2T' if self.two_temp else False
            self.stress_energy_HD = self.calc_THD_cart(EOS=EOS, **kwargs)
        else:
            self.stress_energy_HD = kwargs.get('THD')
        x    = np.zeros(self.data['uu'].shape)
        x[1] = self.r * np.sin(self.th) * np.cos(self.phi)
        x[2] = self.r * np.sin(self.th) * np.sin(self.phi)
        x[3] = self.r * np.cos(self.th) * np.ones(self.phi.shape)
        return np.einsum('i...,j...->ij...', x, self.stress_energy_HD[:,0]) - (
               np.einsum('j...,i...->ij...', x, self.stress_energy_HD[:,0]))
    
    # Angular momentum flux (see below Eq. 17 in Fragile & Anninos 2005)
    def calc_total_momentum_vector(self, **kwargs):
        if 'THD' not in kwargs:
            EOS = '2T' if self.two_temp else False
            self.stress_energy_HD = self.calc_THD_cart(EOS=EOS, **kwargs)
        else:
            self.stress_energy_HD = kwargs.get('THD')
        return self.stress_energy_HD[:, 0]

    # Hyrodynamic stress-energy tensor in Cartesian coordinates
    # See White, Quataert & Blaes (2019), Equation 5
    def calc_THD_cart(self, **kwargs):
        gamma_adi = 5./3.
        from analysis import Fluid
        Flu    = Fluid(self.pars, self.CONSTANTS, cgs=self.kwargs.get('cgs'))
        Coords = Coordinates(self.r, self.th, self.phi, self.pars, self.CONSTANTS)
        return Coords.int_to_cart_tnsr(Flu.calc_THD(**self.data, **self.grid,
            pgas=(gamma_adi - 1.) * self.data['u_gas'], **kwargs))

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
