import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
import tracemalloc
import hamr_read as hr
from analysis import Accretion
from analysis import Field
from analysis import Fluid
from analysis import Tools

# Numerical constants for performance
ONE       = np.float32(1.)

class Coordinates():
    """
    Coordinates class manages all coordinate transformations for 
    4-vectors (vec), one-forms/co-vectors (one_form), rank 2 tensors (tnsr)
    with two up indices, and rank 2 tensors with two down indices (cotnsr),
    such as the metric tensor. Lorentz boosts are included along with
    calculation of the inner product in all coordinate systems.

    Attributes
    ----------
    r : arr of float
        Coordinate radius array from grid object, grid['r'].
    theta : arr of float
        Coordinate theta array from grid object, grid['theta'].
    phi : arr of float
        Coordinate phi array from grid object, grid['phi'].
    pars : dict from H-AMR read object
        Parameters dictionary (dx1, dx2, density_scale, etc.).
    CONSTANTS : dict
        Constants dictionary (hr.CONSTANTS, usually).
    **kwargs : dict
        Optional configurations (e.g. cgs).

    """
    def __init__(self, r, theta, phi, pars, CONSTANTS, **kwargs):
        self.dx1, self.dx2, self.dx3 = pars['dx1'], pars['dx2'], pars['dx3']

        self.cgs  = kwargs.get('cgs', False)
        self.MBH  = kwargs.get('MBH', ONE) if self.cgs else ONE # g
        self.c    = CONSTANTS['c']         if self.cgs else ONE # cm/s
        self.G    = CONSTANTS['G']         if self.cgs else ONE # cm^3/g s^2
        self.Msun = CONSTANTS['Msun']      if self.cgs else ONE # g
        Rg        = (self.G * self.MBH * self.Msun) / (self.c * self.c)

        self.r  = r / Rg
        self.th = theta
        self.ph = phi

    # -------------------------------------
    # Spherical to Cartesian (orthonormal)
    # -------------------------------------
    def sph_to_cart_vec(self, vec, **kwargs):
        """(t, r, theta, phi) -> (t, x, y, z)"""
        shape = list(vec.shape)
        shape.insert(0, 4)
        drdx = hr.Tools.zeroNd(shape, 1, **kwargs)

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

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(vec, drdx)
        else:
            return np.einsum('ij...,j...->i...', drdx, vec)

    def sph_to_cart_one_form(self, one_form, **kwargs):
        """(t, r, theta, phi) -> (t, x, y, z)"""
        shape = list(one_form.shape)
        shape.insert(0, 4)
        dxdr = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Cartesian Kerr Schild from Spherical Kerr Schild
        dxdr[0, 0] = 1.0
        dxdr[1, 1] = (np.sin(self.th) * np.cos(self.ph))
        dxdr[2, 1] = (np.sin(self.th) * np.sin(self.ph))
        dxdr[3, 1] = (np.cos(self.th))
        dxdr[1, 2] = ((1. / self.r) * (np.cos(self.th) * np.cos(self.ph)))
        dxdr[2, 2] = ((1. / self.r) * (np.cos(self.th) * np.sin(self.ph)))
        dxdr[3, 2] = (-(1. / self.r) * np.sin(self.th))
        dxdr[1, 3] = (-(1. / self.r) * np.sin(self.ph) / np.sin(self.th))
        dxdr[2, 3] = ((1. / self.r) * np.cos(self.ph) / np.sin(self.th))

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(one_form, dxdr)
        else:
            return np.einsum('ij...,j...->i...', dxdr, one_form)
    
    def sph_to_cart_tnsr(self, tnsr, **kwargs):
        """(t, r, theta, phi) -> (t, x, y, z)"""
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        drdx = hr.Tools.zeroNd(shape, 1, **kwargs)
        
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

        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(tnsr, drdx)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', tnsr, drdx, drdx)

    def sph_to_cart_cotnsr(self, cotnsr, **kwargs):
        """(t, r, theta, phi) -> (t, x, y, z)"""
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dxdr = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Cartesian Kerr Schild from Spherical Kerr Schild
        dxdr[0, 0] = 1.0
        dxdr[1, 1] = (np.sin(self.th) * np.cos(self.ph))
        dxdr[2, 1] = (np.sin(self.th) * np.sin(self.ph))
        dxdr[3, 1] = (np.cos(self.th))
        dxdr[1, 2] = ((1. / self.r) * (np.cos(self.th) * np.cos(self.ph)))
        dxdr[2, 2] = ((1. / self.r) * (np.cos(self.th) * np.sin(self.ph)))
        dxdr[3, 2] = (-(1. / self.r) * np.sin(self.th))
        dxdr[1, 3] = (-(1. / self.r) * np.sin(self.ph) / np.sin(self.th))
        dxdr[2, 3] = ((1. / self.r) * np.cos(self.ph) / np.sin(self.th))

        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(cotnsr, dxdr)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', cotnsr, dxdr, dxdr)

    # -------------------------------------
    # Cartesian (orthonormal) to Spherical
    # -------------------------------------
    def cart_to_sph_vec(self, vec, **kwargs):
        """(t, x, y, z) -> (t, r, theta, phi)"""
        shape = list(vec.shape)
        shape.insert(0, 4)
        dxdr = hr.Tools.zeroNd(shape, 1, **kwargs)

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

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(vec, dxdr)
        else:
            return np.einsum('ij...,j...->i...', dxdr, vec)
    
    def cart_to_sph_one_form(self, one_form, **kwargs):
        """(t, x, y, z) -> (t, r, theta, phi)"""
        shape = list(one_form.shape)
        shape.insert(0, 4)
        drdx = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Spherical Kerr Schild from Cartesian Kerr Schild
        drdx[0, 0] = 1.0
        drdx[1, 1] = (np.sin(self.th) * np.cos(self.ph))
        drdx[2, 1] = (self.r * np.cos(self.th) * np.cos(self.ph))
        drdx[3, 1] = (-self.r * np.sin(self.th) * np.sin(self.ph))
        drdx[1, 2] = (np.sin(self.th) * np.sin(self.ph))
        drdx[2, 2] = (self.r * np.cos(self.th) * np.sin(self.ph))
        drdx[3, 2] = (self.r * np.sin(self.th) * np.cos(self.ph))
        drdx[1, 3] = (np.cos(self.th))
        drdx[2, 3] = (-self.r * np.sin(self.th))

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(one_form, drdx)
        else:
            return np.einsum('ij...,j...->i...', drdx, one_form)

    def cart_to_sph_tnsr(self, tnsr, **kwargs):
        """(t, x, y, z) -> (t, r, theta, phi)"""
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dxdr = hr.Tools.zeroNd(shape, 1, **kwargs)

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
        
        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(tnsr, dxdr)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', tnsr, dxdr, dxdr)
    
    def cart_to_sph_cotnsr(self, cotnsr, **kwargs):
        """(t, x, y, z) -> (t, r, theta, phi)"""
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        drdx = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Spherical Kerr Schild from Cartesian Kerr Schild
        drdx[0, 0] = 1.0
        drdx[1, 1] = (np.sin(self.th) * np.cos(self.ph))
        drdx[2, 1] = (self.r * np.cos(self.th) * np.cos(self.ph))
        drdx[3, 1] = (-self.r * np.sin(self.th) * np.sin(self.ph))
        drdx[1, 2] = (np.sin(self.th) * np.sin(self.ph))
        drdx[2, 2] = (self.r * np.cos(self.th) * np.sin(self.ph))
        drdx[3, 2] = (self.r * np.sin(self.th) * np.cos(self.ph))
        drdx[1, 3] = (np.cos(self.th))
        drdx[2, 3] = (-self.r * np.sin(self.th))

        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(cotnsr, drdx)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', cotnsr, drdx, drdx)
    
    # -------------------------------------
    # Cartesian (orthonormal) to internal
    # -------------------------------------
    def cart_to_int_vec(self, vec, **kwargs):
        """(t, x, y, z) -> (t, x1, x2, x3)"""
        return self.sph_to_int_vec(
            self.cart_to_sph_vec(vec, **kwargs), **kwargs)
    
    def cart_to_int_one_form(self, one_form, **kwargs):
        """(t, x, y, z) -> (t, x1, x2, x3)"""
        return self.sph_to_int_one_form(
            self.cart_to_sph_one_form(one_form, **kwargs), **kwargs)
    
    def cart_to_int_tnsr(self, tnsr, **kwargs):
        """(t, x, y, z) -> (t, x1, x2, x3)"""
        return self.sph_to_int_tnsr(
            self.cart_to_sph_tnsr(tnsr, **kwargs), **kwargs)
    
    def cart_to_int_cotnsr(self, cotnsr, **kwargs):
        """(t, x, y, z) -> (t, x1, x2, x3)"""
        return self.sph_to_int_cotnsr(
            self.cart_to_sph_cotnsr(cotnsr, **kwargs), **kwargs)

    # -------------------------------------
    # Spherical to Cylindrical (orthonormal)
    # -------------------------------------
    def sph_to_cyl_vec(self, vec, **kwargs):
        """(t, r, theta, phi) -> (t, R, phi, z)"""
        shape = list(vec.shape)
        shape.insert(0, 4)
        drdz = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Cylindrical Kerr Schild from Spherical Kerr Schild
        drdz[0, 0] = 1.
        drdz[1, 1] = np.sin(self.th)
        drdz[1, 2] = self.r * np.cos(self.th)
        drdz[2, 3] = self.r * np.sin(self.th)
        drdz[3, 1] = np.cos(self.th)
        drdz[3, 2] = -self.r * np.sin(self.th)

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(vec, drdz)
        else:
            return np.einsum('ij...,j...->i...', drdz, vec)
    
    def sph_to_cyl_one_form(self, one_form, **kwargs):
        """(t, r, theta, phi) -> (t, R, phi, z)"""
        shape = list(one_form.shape)
        shape.insert(0, 4)
        dzdr = hr.Tools.zeroNd(shape, 1, **kwargs)
        
        # Transformation to Cylindrical Kerr Schild from Spherical Kerr Schild
        dzdr[0, 0] = 1.
        dzdr[1, 1] = np.sin(self.th)
        dzdr[3, 1] = np.cos(self.th)
        dzdr[1, 2] = (1. / self.r) * np.cos(self.th)
        dzdr[3, 2] = -(1. / self.r) * np.sin(self.th)
        dzdr[2, 3] = (1. / self.r) * (1. / np.sin(self.th))

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(one_form, dzdr)
        else:
            return np.einsum('ij...,j...->i...', dzdr, one_form)

    def sph_to_cyl_tnsr(self, tnsr, **kwargs):
        """(t, r, theta, phi) -> (t, R, phi, z)"""
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        drdz = hr.Tools.zeroNd(shape, 1, **kwargs)
        
        # Transformation to Cylindrical Kerr Schild from Spherical Kerr Schild
        drdz[0, 0] = 1.
        drdz[1, 1] = np.sin(self.th)
        drdz[1, 2] = self.r * np.cos(self.th)
        drdz[2, 3] = self.r * np.sin(self.th)
        drdz[3, 1] = np.cos(self.th)
        drdz[3, 2] = -self.r * np.sin(self.th)
        
        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(tnsr, drdz)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', tnsr, drdz, drdz)

    def sph_to_cyl_cotnsr(self, cotnsr, **kwargs):
        """(t, r, theta, phi) -> (t, R, phi, z)"""
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dzdr = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Cylindrical Kerr Schild from Spherical Kerr Schild
        dzdr[0, 0] = 1.
        dzdr[1, 1] = np.sin(self.th)
        dzdr[3, 1] = np.cos(self.th)
        dzdr[1, 2] = (1. / self.r) * np.cos(self.th)
        dzdr[3, 2] = -(1. / self.r) * np.sin(self.th)
        dzdr[2, 3] = (1. / self.r) * (1. / np.sin(self.th))

        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(cotnsr, dzdr)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', cotnsr, dzdr, dzdr)

    # -------------------------------------
    # Cylindrical (orthonormal) to Spherical
    # -------------------------------------
    def cyl_to_sph_vec(self, vec, **kwargs):
        """(t, R, phi, z) -> (t, r, theta, phi)"""
        shape = list(vec.shape)
        shape.insert(0, 4)
        dzdr = hr.Tools.zeroNd(shape, 1, **kwargs)
        
        # Transformation to Spherical Kerr Schild from Cylindrical Kerr Schild
        dzdr[0, 0] = 1.
        dzdr[1, 1] = np.sin(self.th)
        dzdr[1, 2] = 0.
        dzdr[1, 3] = np.cos(self.th)
        dzdr[2, 1] = (1. / self.r) * np.cos(self.th)
        dzdr[2, 3] = -(1. / self.r) * np.sin(self.th)
        dzdr[3, 2] = (1. / self.r) * (1. / np.sin(self.th))

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(vec, dzdr)
        else:
            return np.einsum('ij...,j...->i...', dzdr, vec)

    def cyl_to_sph_one_form(self, one_form, **kwargs):
        """(t, R, phi, z) -> (t, r, theta, phi)"""
        shape = list(one_form.shape)
        shape.insert(0, 4)
        drdz = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Spherical Kerr Schild from Cylindrical Kerr Schild
        drdz[0, 0] = 1.
        drdz[1, 1] = np.sin(self.th)
        drdz[2, 1] = self.r * np.cos(self.th)
        drdz[3, 2] = self.r * np.sin(self.th)
        drdz[1, 3] = np.cos(self.th)
        drdz[2, 3] = -self.r * np.sin(self.th)

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(one_form, drdz)
        else:
            return np.einsum('ij...,j...->i...', drdz, one_form)

    def cyl_to_sph_tnsr(self, tnsr, **kwargs):
        """(t, R, phi, z) -> (t, r, theta, phi)"""
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dzdr = hr.Tools.zeroNd(shape, 1, **kwargs)
        
        # Transformation to Spherical Kerr Schild from Cylindrical Kerr Schild
        dzdr[0, 0] = 1.
        dzdr[1, 1] = np.sin(self.th)
        dzdr[1, 2] = 0.
        dzdr[1, 3] = np.cos(self.th)
        dzdr[2, 1] = (1. / self.r) * np.cos(self.th)
        dzdr[2, 3] = -(1. / self.r) * np.sin(self.th)
        dzdr[3, 2] = (1. / self.r) * (1. / np.sin(self.th))

        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(tnsr, dzdr)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', tnsr, dzdr, dzdr)

    def cyl_to_sph_cotnsr(self, cotnsr, **kwargs):
        """(t, R, phi, z) -> (t, r, theta, phi)"""
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        drdz = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Spherical Kerr Schild from Cylindrical Kerr Schild
        drdz[0, 0] = 1.
        drdz[1, 1] = np.sin(self.th)
        drdz[2, 1] = self.r * np.cos(self.th)
        drdz[3, 2] = self.r * np.sin(self.th)
        drdz[1, 3] = np.cos(self.th)
        drdz[2, 3] = -self.r * np.sin(self.th)

        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(cotnsr, drdz)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', cotnsr, drdz, drdz)

    # -------------------------------------
    # Internal to Spherical
    # -------------------------------------
    def int_to_sph_vec(self, vec, **kwargs):
        """(t, x1, x2, x3) -> (t, r, theta, phi)"""
        shape = list(vec.shape)
        shape.insert(0, 4)
        dxdxp = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Spherical Kerr Schild from internal coordinates
        dxdxp[0, 0] = 1.0
        dxdxp[1, 1] = self.r
        dxdxp[2, 2] = np.pi / 2.0
        dxdxp[3, 3] = 1.0

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(vec, dxdxp)
        else:
            return np.einsum('ij...,j...->i...', dxdxp, vec)

    def int_to_sph_one_form(self, one_form, **kwargs):
        """(t, x1, x2, x3) -> (t, r, theta, phi)"""
        # Matrix is diagonal so no transpose necessary
        return self.sph_to_int_vec(one_form, **kwargs)

    def int_to_sph_tnsr(self, tnsr, **kwargs):
        """(t, x1, x2, x3) -> (t, r, theta, phi)"""
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dxdxp = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Spherical Kerr Schild from internal coordinates
        dxdxp[0, 0] = 1.0
        dxdxp[1, 1] = self.r
        dxdxp[2, 2] = np.pi / 2.0
        dxdxp[3, 3] = 1.0

        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(tnsr, dxdxp)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', tnsr, dxdxp, dxdxp)
    
    def int_to_sph_cotnsr(self, cotnsr, **kwargs):
        """(t, x1, x2, x3) -> (t, r, theta, phi)"""
        # Matrix is diagonal so no transpose necessary
        return self.sph_to_int_tnsr(cotnsr, **kwargs)

    # -------------------------------------
    # Spherical to Internal
    # -------------------------------------
    # (t, r, theta, phi) -> (t, x1, x2, x3) 
    def sph_to_int_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        dxpdx = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to internal coordinates from Spherical Kerr Schild
        dxpdx[0, 0] = 1.0
        dxpdx[1, 1] = 1. / self.r
        dxpdx[2, 2] = 2.0 / np.pi
        dxpdx[3, 3] = 1.0

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(vec, dxpdx)
        else:
            return np.einsum('ij...,j...->i...', dxpdx, vec)

    # (t, r, theta, phi) -> (t, x1, x2, x3) 
    def sph_to_int_one_form(self, one_form, **kwargs):
        # Matrix is diagonal so no transpose necessary
        return self.int_to_sph_vec(one_form, **kwargs)

    # (t, r, theta, phi) -> (t, x1, x2, x3) 
    def sph_to_int_tnsr(self, tnsr, **kwargs):
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dxpdx = hr.Tools.zeroNd(shape, 1, **kwargs)
        
        # Transformation to internal coordinates from Spherical Kerr Schild
        dxpdx[0, 0] = 1.0
        dxpdx[1, 1] = 1. / self.r
        dxpdx[2, 2] = 2.0 / np.pi
        dxpdx[3, 3] = 1.0
        
        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(tnsr, dxpdx)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', tnsr, dxpdx, dxpdx)

    # (t, r, theta, phi) -> (t, x1, x2, x3) 
    def sph_to_int_cotnsr(self, cotnsr, **kwargs):
        # Matrix is diagonal so no transpose necessary
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
        dxdxp = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to orthonormal Spherical Kerr Schild 
        # from internal coordinates
        dxdxp[0, 0] = 1.0
        dxdxp[1, 1] = self.r
        dxdxp[2, 2] = (np.pi / 2.0) * self.r
        dxdxp[3, 3] = self.r

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(vec, dxdxp)
        else:
            return np.einsum('ij...,j...->i...', dxdxp, vec)

    # (t, x1, x2, x3) -> (t, r, theta, phi) 
    def int_to_ortho_sph_one_form(self, one_form, **kwargs):
        # Matrix is diagonal so no transpose necessary
        return self.ortho_sph_to_int_vec(one_form, **kwargs)

    # (t, x1, x2, x3) -> (t, r, theta, phi) 
    def int_to_ortho_sph_tnsr(self, tnsr, **kwargs):
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dxdxp = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Spherical Kerr Schild from internal coordinates
        dxdxp[0, 0] = 1.0
        dxdxp[1, 1] = self.r
        dxdxp[2, 2] = (np.pi / 2.0) * self.r
        dxdxp[3, 3] = self.r

        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(tnsr, dxdxp)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', tnsr, dxdxp, dxdxp)
    
    # (t, x1, x2, x3) -> (t, r, theta, phi) 
    def int_to_ortho_sph_cotnsr(self, cotnsr, **kwargs):
        # Matrix is diagonal so no transpose necessary
        return self.ortho_sph_to_int_tnsr(cotnsr, **kwargs) 

    # -------------------------------------
    # Spherical (orthonormal) to Internal
    # -------------------------------------
    #  (t, r, theta, phi) -> (t, x1, x2, x3)
    def ortho_sph_to_int_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        dxpdx = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to orthonormal Spherical Kerr Schild 
        # from internal coordinates
        dxpdx[0, 0] = 1.0
        dxpdx[1, 1] = 1. / self.r
        dxpdx[2, 2] = (2.0 / np.pi) * (1. / self.r)
        dxpdx[3, 3] = 1. / self.r

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(vec, dxpdx)
        else:
            return np.einsum('ij...,j...->i...', dxpdx, vec)

    # (t, r, theta, phi) -> (t, x1, x2, x3)
    def ortho_sph_to_int_one_form(self, one_form, **kwargs):
        # Matrix is diagonal so no transpose necessary
        return self.int_to_ortho_sph_vec(one_form, **kwargs)

    # (t, r, theta, phi) -> (t, x1, x2, x3)
    def ortho_sph_to_int_tnsr(self, tnsr, **kwargs):
        shape = list(self.r.shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dxpdx = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Spherical Kerr Schild from internal coordinates
        dxpdx[0, 0] = 1.0
        dxpdx[1, 1] = 1. / self.r
        dxpdx[2, 2] = (2.0 / np.pi) * (1. / self.r)
        dxpdx[3, 3] = 1. / self.r

        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(tnsr, dxpdx)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', tnsr, dxpdx, dxpdx)
    
    # (t, r, theta, phi) -> (t, x1, x2, x3)
    def ortho_sph_to_int_cotnsr(self, cotnsr, **kwargs):
        # Matrix is diagonal so no transpose necessary
        return self.int_to_ortho_sph_tnsr(cotnsr, **kwargs) 
    
    # -------------------------------------
    # Spherical (orthonormal) to Cylindrical (orthonormal)
    # -------------------------------------
    # (t, r, theta, phi) -> (t, R, phi, z)
    def ortho_sph_to_cyl_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        drdz = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Cylindrical Kerr Schild from Spherical Kerr Schild
        drdz[0, 0] = 1.
        drdz[1, 1] = np.sin(self.th)
        drdz[1, 2] = np.cos(self.th)
        drdz[2, 3] = np.sin(self.th)
        drdz[3, 1] = np.cos(self.th)
        drdz[3, 2] = -np.sin(self.th)

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(vec, drdz)
        else:
            return np.einsum('ij...,j...->i...', drdz, vec)
    
    # (t, r, theta, phi) -> (t, R, phi, z)
    def ortho_sph_to_cyl_one_form(self, one_form, **kwargs):
        shape = list(one_form.shape)
        shape.insert(0, 4)
        dzdr = hr.Tools.zeroNd(shape, 1, **kwargs)
        
        # Transformation to Cylindrical Kerr Schild from Spherical Kerr Schild
        dzdr[0, 0] = 1.
        dzdr[1, 1] = np.sin(self.th)
        dzdr[3, 1] = np.cos(self.th)
        dzdr[1, 2] = np.cos(self.th)
        dzdr[3, 2] = -np.sin(self.th)
        dzdr[2, 3] = (1. / np.sin(self.th))

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(one_form, dzdr)
        else:
            return np.einsum('ij...,j...->i...', dzdr, one_form)

    # (t, r, theta, phi) -> (t, R, phi, z)
    def ortho_sph_to_cyl_tnsr(self, tnsr, **kwargs):
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        drdz = hr.Tools.zeroNd(shape, 1, **kwargs)
        
        # Transformation to Cylindrical Kerr Schild from Spherical Kerr Schild
        drdz[0, 0] = 1.
        drdz[1, 1] = np.sin(self.th)
        drdz[1, 2] = np.cos(self.th)
        drdz[2, 3] = np.sin(self.th)
        drdz[3, 1] = np.cos(self.th)
        drdz[3, 2] = np.sin(self.th)
        
        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(tnsr, drdz)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', tnsr, drdz, drdz)

    # (t, r, theta, phi) -> (t, R, phi, z)
    def ortho_sph_to_cyl_cotnsr(self, cotnsr, **kwargs):
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dzdr = hr.Tools.zeroNd(shape, 1, **kwargs)
        
        # Transformation to Cylindrical Kerr Schild from Spherical Kerr Schild
        dzdr[0, 0] = 1.
        dzdr[1, 1] = np.sin(self.th)
        dzdr[3, 1] = np.cos(self.th)
        dzdr[1, 2] = np.cos(self.th)
        dzdr[3, 2] = -np.sin(self.th)
        dzdr[2, 3] = (1. / np.sin(self.th))

        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(cotnsr, dzdr)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', cotnsr, dzdr, dzdr)

    # -------------------------------------
    # Cylindrical (orthonormal) to Spherical (orthonormal)
    # -------------------------------------
    # (t, R, phi, z) -> (t, r, theta, phi)
    def cyl_to_ortho_sph_vec(self, vec, **kwargs):
        shape = list(vec.shape)
        shape.insert(0, 4)
        dzdr = hr.Tools.zeroNd(shape, 1, **kwargs)
        
        # Transformation to Spherical Kerr Schild from Cylindrical Kerr Schild
        dzdr[0, 0] = 1.
        dzdr[1, 1] = np.sin(self.th)
        dzdr[1, 2] = 0.
        dzdr[1, 3] = np.cos(self.th)
        dzdr[2, 1] = np.cos(self.th)
        dzdr[2, 3] = -np.sin(self.th)
        dzdr[3, 2] = (1. / np.sin(self.th))

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(vec, dzdr)
        else:
            return np.einsum('ij...,j...->i...', dzdr, vec)

    # (t, R, phi, z) -> (t, r, theta, phi)
    def cyl_to_ortho_sph_one_form(self, one_form, **kwargs):
        shape = list(one_form.shape)
        shape.insert(0, 4)
        drdz = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Spherical Kerr Schild from Cylindrical Kerr Schild
        drdz[0, 0] = 1.
        drdz[1, 1] = np.sin(self.th)
        drdz[2, 1] = np.cos(self.th)
        drdz[3, 2] = np.sin(self.th)
        drdz[1, 3] = np.cos(self.th)
        drdz[2, 3] = -np.sin(self.th)

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(one_form, drdz)
        else:
            return np.einsum('ij...,j...->i...', drdz, one_form)

    # (t, R, phi, z) -> (t, r, theta, phi)
    def cyl_to_ortho_sph_tnsr(self, tnsr, **kwargs):
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        dzdr = hr.Tools.zeroNd(shape, 1, **kwargs)
        
        # Transformation to Spherical Kerr Schild from Cylindrical Kerr Schild
        dzdr[0, 0] = 1.
        dzdr[1, 1] = np.sin(self.th)
        dzdr[1, 3] = np.cos(self.th)
        dzdr[2, 1] = np.cos(self.th)
        dzdr[2, 3] = -np.sin(self.th)
        dzdr[3, 2] = (1. / np.sin(self.th))
        
        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(tnsr, dzdr)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', tnsr, dzdr, dzdr)

    # (t, R, phi, z) -> (t, r, theta, phi)
    def cyl_to_ortho_sph_cotnsr(self, cotnsr, **kwargs):
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        drdz = hr.Tools.zeroNd(shape, 1, **kwargs)

        # Transformation to Spherical Kerr Schild from Cylindrical Kerr Schild
        drdz[0, 0] = 1.
        drdz[1, 1] = np.sin(self.th)
        drdz[2, 1] = np.cos(self.th)
        drdz[3, 2] = np.sin(self.th)
        drdz[1, 3] = np.cos(self.th)
        drdz[2, 3] = -np.sin(self.th)

        if kwargs.get('cpp', False):
            return Tools.contract_tensor_cpp(cotnsr, drdz)
        else:
            return np.einsum('kl...,ik...,jl...->ij...', cotnsr, drdz, drdz)

    # -------------------------------------
    # Cylindrical (orthonormal) to Internal
    # -------------------------------------
    # (t, R, phi, z) -> (t, x1, x2, x3)
    def cyl_to_int_vec(self, vec, **kwargs):
        return self.sph_to_int_vec(
            self.cyl_to_sph_vec(vec, **kwargs), **kwargs)

    # (t, R, phi, z) -> (t, x1, x2, x3)
    def cyl_to_int_one_form(self, one_form, **kwargs):
        return self.sph_to_int_one_form(
            self.cyl_to_sph_one_form(one_form, **kwargs), **kwargs)

    # (t, R, phi, z) -> (t, x1, x2, x3)
    def cyl_to_int_tnsr(self, tnsr, **kwargs):
        return self.sph_to_int_tnsr(
            self.cyl_to_sph_tnsr(tnsr, **kwargs), **kwargs)

    # (t, R, phi, z) -> (t, x1, x2, x3)
    def cyl_to_int_cotnsr(self, cotnsr, **kwargs):
        return self.sph_to_int_cotnsr(
            self.cyl_to_sph_cotnsr(cotnsr, **kwargs), **kwargs)

    # Calculate the Lorentz boost matrix Lambda^mu_alpha
    # See Sadowski+2013 Equation 10 (with - signs on t-i components)
    # Boost is from the unprimed to the primed frame
    # uu is the velocity of the primed frame as measured in the unprimed frame
    # uu is assumed dimensionless
    def calc_Lorentz_boost(self, uu, **kwargs):
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        Lambda = hr.Tools.zeroNd(shape, 1, **kwargs)

        if not kwargs.get('inverse', False):
            v = uu / uu[0]
        else:
            v = -uu / uu[0]
        v2 = v[1] * v[1] + v[2] * v[2] + v[3] * v[3]

        Lambda[0,  0] = uu[0]
        Lambda[1:, 0] = -uu[0] * v[1:]
        Lambda[0, 1:] = -uu[0] * v[1:]
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

    def dot(self, vec_a_in, vec_b_in, gcov, coord_system, **kwargs):
        """
        Compute dot product between two vectors in the desired
        coordinate system. Assumes vec_a, vec_b, and gcov are in
        internal coordinates. Memory intensive on 3D arrays.
        As an alternative, one could compute the one-form of the vector a,
        and then convert to the desired coordinate system; however, this
        requires computing the inverse of the coordinate transformation matrix.
        """
        transforms = {
            'internal'    : (lambda g, **kw: g,      lambda v, **kw: v),
            'spherical'   : (self.int_to_sph_cotnsr,  self.int_to_sph_vec),
            'cylindrical' : (self.int_to_cyl_cotnsr,  self.int_to_cyl_vec),
            'Cartesian'   : (self.int_to_cart_cotnsr, self.int_to_cart_vec),
        }
        cotnsr_transform, vec_transform = transforms[coord_system]

        vec_a = vec_transform(np.copy(vec_a_in), **kwargs)
        vec_b = vec_transform(np.copy(vec_b_in), **kwargs)
        gcov  = cotnsr_transform(gcov, **kwargs)

        if 'component' in kwargs:
            vec_a, vec_b = self.isolate_component(vec_a, vec_b, **kwargs)

        return Tools.dot(vec_a, vec_b, gcov, **kwargs)

    def isolate_component(self, vec_a, vec_b, **kwargs):
        """Zero all spatial components other than kwargs.get('component')"""
        component = kwargs.get('component')
        for index in (1, 2, 3):
            if index != component:
                vec_a[index] = 0.
                vec_b[index] = 0.
        return vec_a, vec_b

class Integrals(Coordinates):
    def __init__(self, r, theta, phi, pars, CONSTANTS, **kwargs):
        super().__init__(r, theta, phi, pars, CONSTANTS, **kwargs)

    # symmetric \int_{0}^{\theta} dth q(r, th, phi)
    def cum_integrate_th(self, q, gcov22, **kwargs):
        qdth = q * np.sqrt(gcov22) * self.dx2 * self.apply_selection(**kwargs)

        # theta's axis shifts by one once a leading block dimension
        # is present (q.ndim == 4).
        th_slices = {
            2: (self.th[0, :],      -1),
            3: (self.th[0, :, 0],    1),
            4: (self.th[0, 0, :, 0], 2),
        }
        th_1D, th_axis = th_slices[q.ndim]
        ith0 = np.argwhere(th_1D < np.pi/2)[-1][0] + 1

        lower = [slice(None)] * q.ndim
        upper = [slice(None)] * q.ndim
        lower[th_axis] = slice(None, ith0)
        upper[th_axis] = slice(ith0, None)
        lower, upper  = tuple(lower), tuple(upper)

        flip = [slice(None)] * q.ndim
        flip[th_axis] = slice(None, None, -1)
        flip = tuple(flip)

        # Write cumsum results directly into (a view of) int_qdth,
        # rather than into a throwaway temporary that then gets copied in.
        int_qdth = np.empty_like(qdth)
        np.cumsum(qdth[lower],       axis=th_axis, out=int_qdth[lower])
        np.cumsum(qdth[upper][flip], axis=th_axis, out=int_qdth[upper][flip])
        return int_qdth

    # -\int_{\infty}^{r} dr q(r, th, phi)
    # Note: expects gcov in internal coordinates
    def cum_integrate_r(self, q, gcov11, **kwargs):
        qdr = q * np.sqrt(gcov11) * self.dx1 * self.apply_selection(**kwargs)

        # r's axis shifts by one once a leading block dimension
        # is present (q.ndim == 4).
        r_axis = 0 if q.ndim < 4 else 1

        # Writing into a flipped view of int_qdr both reverses the
        # cumsum direction and un-reverses the result, with no temporary.
        int_qdr = np.empty_like(qdr)
        np.cumsum(np.flip(qdr, axis=r_axis), axis=r_axis,
                  out=np.flip(int_qdr, axis=r_axis))
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
        qdz = q_uni * dz[:, None, None] * self.apply_selection(**kwargs)

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
        if 'sigma_cut' not in kwargs and 'sigma' not in kwargs:
            return 1.0
        if 'sigma' not in kwargs:
            raise TypeError("Missing sigma in kwargs.")
        if 'sigma_cut' not in kwargs:
            raise TypeError("Must specify value of sigma_cut.")
        return kwargs['sigma'] < kwargs['sigma_cut']

    def theta_cut(self, **kwargs):
        if 'theta_cut' not in kwargs:
            return 1.0

        theta_cut = kwargs.get('theta_cut')
        if not kwargs.get('uni', False):
            th = self.th
        elif 'theta_uni' in kwargs:
            th = kwargs.get('theta_uni')[:, :, None]
        else:
            raise ValueError("Must supply theta_uni to use uni=True.")

        return (th > theta_cut) & (th < np.pi - theta_cut)

    def apply_selection(self, **kwargs):
        """Applies sigma and theta selections, combined multiplicatively."""
        selection_funcs = {
            ('sigma_cut',) : self.sigma_cut,
            ('theta_cut',) : self.theta_cut,
        }

        keep = ONE
        for trigger_keys, sel_func in selection_funcs.items():
            if any(key in kwargs for key in trigger_keys):
                keep = keep * sel_func(**kwargs)

        return keep

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
        beta    = hr.Tools.zeroNd(shape, 1, **kwargs)
        beta[0] = 2. * (self.r / self.Sigma) / (1. + 2. * self.r / self.Sigma)
        return beta
    
    def calc_Minkowski(self, **kwargs):
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        g_Mink = hr.Tools.zeroNd(shape, 1, **kwargs)
        g_Mink[0, 0] = -1.
        g_Mink[1, 1] = 1.
        g_Mink[2, 2] = self.r * self.r
        g_Mink[3, 3] = (self.r * np.sin(self.th)) ** 2.
        return g_Mink

    def calc_gdet(self, **kwargs):
        return self.Sigma * np.abs(np.sin(self.th))
 
    def calc_gcov(self, **kwargs):
        """Calculate metric in spherical Kerr-Schild coordinates."""
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        gcov  = hr.Tools.zeroNd(shape, 1, **kwargs)
        
        cth  = np.cos(self.th)
        sth  = np.sin(self.th)
        s2   = sth * sth
        rho2 = self.r * self.r + self.a * self.a * cth * cth
        
        gcov[0, 0][..., 0] = (-1. + 2. * self.r / rho2)[..., 0]
        gcov[0, 1][..., 0] = (2. * self.r / rho2)[..., 0]
        gcov[0, 3][..., 0] = (-2. * self.a * self.r * s2 / rho2)[..., 0]
        gcov[1, 0][..., 0] = gcov[0, 1][..., 0]
        gcov[1, 1][..., 0] =  (1. + 2. * self.r / rho2)[..., 0]
        gcov[1, 3][..., 0] = (-self.a * s2 * (1. + 2. * self.r / rho2))[..., 0]
        gcov[2, 2][..., 0] = rho2[..., 0]
        gcov[3, 0][..., 0] = gcov[0, 3][..., 0]
        gcov[3, 1][..., 0] = gcov[1, 3][..., 0]
        gcov[3, 3][..., 0] = (s2 * (
            rho2 + self.a * self.a * s2 * (1. + 2. * self.r / rho2))
            )[..., 0] 
        return gcov

    def calc_gcon(self, **kwargs):
        """Calculate metric inverse in spherical Kerr-Schild coordinates."""
        shape = list((self.r * self.th * self.ph).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        gcon  = hr.Tools.zeroNd(shape, 1, **kwargs)
        
        cth  = np.cos(self.th)
        sth  = np.sin(self.th)
        s2   = sth * sth
        
        rho2 = self.r * self.r + self.a * self.a * cth * cth
        gcon[0, 0][..., 0] = (-1. - 2. * self.r / rho2)[..., 0]
        gcon[0, 1][..., 0] = (2. * self.r / rho2)[..., 0] 
        gcon[1, 0][..., 0] = gcon[0, 1][..., 0]
        gcon[1, 1][..., 0] = (self.Delta / rho2)[..., 0]
        gcon[1, 3][..., 0] = (self.a / rho2)[..., 0] 
        gcon[2, 2][..., 0] = (1. / rho2)[..., 0] 
        gcon[3, 1][..., 0] = gcon[1, 3][..., 0]
        gcon[3, 3][..., 0] = (1. / (rho2 * s2))[..., 0]
        return gcon
 
class Tetrads(Kerr_Schild):
    def __init__(self, r, theta, phi, pars, CONSTANTS, **kwargs):
        super().__init__(r, theta, phi, pars, CONSTANTS, **kwargs)
        shape = [4, 4] + list((self.r * self.th * self.ph).shape)
        self.e = hr.Tools.zeroNd(shape, 1, **kwargs)

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
        """
        Class for managing remapping of data to tilted frame.
        For angle definitions, see Wolfram MathWorld (WMW) convention.
        Precession (prec) angle: "phi" on WMW.
        Inclination (inc) angle: "theta" on WMW.
        """
        Accretion.__init__(  self, grid, data, pars, CONSTANTS, **kwargs)
        Coordinates.__init__(self, **grid, pars=pars, CONSTANTS=CONSTANTS, 
                             **kwargs)
        self.two_temp = True if ('Ti' in data and 'Te' in data) else False
        self.radiation= True if ('u_rad' in data and 'uu_rad' in data) else False

        self.set_frame(**kwargs)

    def set_frame(self, **kwargs):
        self.data_tilted, self.grid_tilted = False, False
        
        if kwargs.get('tilted_frame', False):
            # Euler angles as a function of radius (1D array)
            prec = kwargs.get('precession') 
            inc  = kwargs.get('inclination')
            if prec is None or inc is None:
                raise ValueError("Must specify precession, inclination " + 
                                 "angle to use tilted_frame=True.")

            prec = np.asarray(prec, dtype=np.float32)
            inc  = np.asarray(inc,  dtype=np.float32)

            if prec.ndim == 1:
                prec = prec[:, None, None]
            if inc.ndim == 1:
                inc  = inc[:, None, None]

            prec *= (np.pi / 180)
            inc  *= (np.pi / 180)

            self.prec = np.broadcast_to(prec, self.r.shape)
            self.inc  = np.broadcast_to(inc,  self.r.shape)

            self.remap_data(**kwargs)

    def remap_data(self, **kwargs):
        # Define position vector
        shape = (self.r * self.th * self.phi).shape
        xvec  = np.zeros((4,) + shape)
        xvec[1] = self.r * np.sin(self.th) * np.cos(self.phi)
        xvec[2] = self.r * np.sin(self.th) * np.sin(self.phi)
        xvec[3] = self.r * np.cos(self.th)

        if self.prec is None or self.inc is None:
            raise ValueError("Must specify precession and inclination " + 
                             "angle to use tilted_frame=True. Run " + 
                             "Tilt.calc_tilt_angles first.")
        self.prec = self.prec * np.ones(shape)
        self.inc  = self.inc * np.ones(shape)

        # Convert vectors to Cartesian coordinates before remapping.
        for key in ['uu', 'B', 'uu_rad']:
            if key in self.data.keys():
                self.data[key] = self.int_to_cart_vec(self.data[key], **kwargs)
            
        axisym = True if self.grid['gcov'].shape[-1] == 1 else False
        
        self.tilt_grid = kwargs.get('tilt_grid', True)
        if self.tilt_grid:
            # Convert tensor to Cartesian coordinates before remapping.
            self.grid['gcov'] = self.int_to_cart_cotnsr(self.grid['gcov'], 
                                                        **kwargs)
            self.grid['gcon'] = self.int_to_cart_tnsr(self.grid['gcon'], 
                                                      **kwargs)

        xvec_tilt = self.rotate_vector(xvec, inverse=True, **kwargs)

        r_tilt    = np.sqrt(xvec_tilt[1] * xvec_tilt[1] + 
                            xvec_tilt[2] * xvec_tilt[2] + 
                            xvec_tilt[3] * xvec_tilt[3]
        ) 
        th_tilt   = np.arccos(np.clip(xvec_tilt[3] / r_tilt, -1.0, 1.0))
        phi_tilt  = np.atan2(xvec_tilt[2], xvec_tilt[1]) 

        if shape[0] == 1:
            r, th, phi = self.r[0,:,0,0], self.th[0,0,:,0], self.phi[0,0,0,:]

        th_min,  dth  = th[0],  th[1]  - th[0] 
        phi_min, dphi = phi[0], phi[1] - phi[0]
            
        # r index (needs to be mapped to 3D)
        r_idx = np.broadcast_to(np.arange(len(r))[None, :, None, None], shape)

        # Deal with phi range
        phi_wrapped = np.mod(phi_tilt, 2 * np.pi)

        # theta and phi indices
        th_idx  = (th_tilt - th_min) / dth
        phi_idx = (phi_wrapped - phi_min) / dphi

        th_idx = np.clip(th_idx, 0.0, len(th) - 1.0)

        coords_tilt = np.stack([r_idx, th_idx, phi_idx], axis=0)
        
        map_args   = {
            'coordinates' : coords_tilt,
            'order'       : 1,
            'mode'        : 'nearest'
        }

        for key in self.data.keys():
            # To handle scalars, 4-vectors, and tensors, 
            # collapse all leading (component/block) axes into 
            # one flat batch dim (..., nr, nth, nphi) -> (nvar, nr, nth, nphi)
            # where nvar = n_components (1 or 4 or 16) x nblocks.
            orig_shape = self.data[key].shape
            arr        = self.data[key].reshape(-1, *orig_shape[-3:])
            remapped   = np.empty_like(arr)

            for n in range(arr.shape[0]):
                map_args['input'] = arr[n]
                remapped[n] = map_coordinates(**map_args)
            
            self.data[key] = remapped.reshape(orig_shape)

        if self.tilt_grid:
            nphi = shape[-1]
            for key in ['gcov', 'gcon', 'gdet']:
                if axisym:
                    target_shape = self.grid[key].shape[:-1] + (nphi,)
                    self.grid[key] = np.broadcast_to(self.grid[key], 
                                                     target_shape).copy()
                orig_shape = self.grid[key].shape
                arr        = self.grid[key].reshape(-1, *orig_shape[-3:])
                remapped   = np.empty_like(arr)

                for n in range(arr.shape[0]):
                    map_args['input'] = arr[n]
                    remapped[n] = map_coordinates(**map_args)
                
                self.grid[key] = remapped.reshape(orig_shape)
 
        # Tilt vectors after remapping
        self.r   = r_tilt 
        self.th  = th_tilt
        self.phi = phi_wrapped
        for key in ['uu', 'B', 'uu_rad']:
            if key in self.data.keys():
                self.data[key] = self.cart_to_int_vec(
                    self.rotate_vector(self.data[key], **kwargs), **kwargs)

        self.data_tilted = True

        # Free memory
        self.grid['dxdxp'] = None

        if self.tilt_grid:
            # Tilt metric after remapping
            self.grid['gcov'] = self.cart_to_int_cotnsr(self.rotate_tensor(
                self.grid['gcov'], inverse=True, **kwargs), **kwargs)
            # Tilt inverse metric after remapping
            self.grid['gcon'] = self.cart_to_int_tnsr(self.rotate_tensor(
                self.grid['gcon'], **kwargs), **kwargs)
            
            self.grid_tilted = True
            
    def calc_tilt_angles(self, L, **kwargs):
        rho0   = self.rho0
        c      = self.c
        rho0c2 = rho0 * c * c
        
        J = kwargs.get('J')
        if J is None:
            J = self.calc_angular_momentum_vector(L, **kwargs)
        J *= ONE / rho0c2
        if J.shape[0] == 3:
            Jmag = np.sqrt(J[0] * J[0] + J[1] * J[1] + J[2] * J[2])
            Jx, Jy, Jz = J
        elif J.shape[0] == 4:
            Jmag = np.sqrt(J[1] * J[1] + J[2] * J[2] + J[3] * J[3])
            Jx, Jy, Jz = J[1:]
        
        prec = np.atan2(Jy, Jx)
        inc  = np.arccos(Jz / Jmag)
        
        if 'J' in kwargs:
            J *= rho0c2

        if self.data_tilted or self.grid_tilted:
            return np.degrees(prec), np.degrees(inc)
        else:
            self.prec = prec[None, :, None, None] if prec.ndim == 1 else prec
            self.inc  = inc[ None, :, None, None] if inc.ndim == 1 else inc
            return np.degrees(prec), np.degrees(inc)
 
    def calc_angular_momentum_vector(self, L, **kwargs):
        """
        Angular momentum vector J
        Equation 16 of Fragile & Anninos (2005)
        If bulk_momentum=True, uses relativistic expression summing over
        t, x, y, z indices. If bulk_momentum=False (default), uses
        flat-spacetime expression, summing over x, y, z indices.
        """
        rho0   = self.rho0
        c      = self.c
        rho0c2 = rho0 * c * c
        
        # De-dimensionalize
        L *= ONE / rho0c2
        if kwargs.get('bulk_momentum', False):
            S     = kwargs.get('S')
            S_cov = kwargs.get('S_cov')
            if S is None or S_cov is None:
                raise ValueError("Must specify S and S_cov 4-vectors " +
                                 "when using bulk_momentum=True.")

            levi_civita = Tools.calc_Levi_Civita(4, np.ones_like(S[1]))
            S     *= ONE / rho0c2
            S_cov *= ONE / rho0c2

            # Calculate J
            J = np.einsum('ijkl...,ij...,k...', levi_civita, L, S) / (
                2. * np.sqrt(-np.einsum('i...,i...->...', S_cov, S))
            ) 

            S     *= rho0c2
            S_cov *= rho0c2
        else:
            levi_civita = Tools.calc_Levi_Civita(3, np.ones_like(L[0][0]))
            
            # Calculate J
            J = np.einsum('ijk...,ij...->k...', levi_civita, L) / 2.
            
        # Redimensionalize
        L *= rho0c2
        return J * rho0c2

    def calc_angular_momentum_tnsr(self, **kwargs):
        """
        Angular momentum tensor L^{mu nu}
        Equation 17 of Fragile & Anninos (2005)
        """
        self.stress_energy_HD = kwargs.get('THD')
        if self.stress_energy_HD is None:
            self.stress_energy_HD = self.calc_THD_cart(**kwargs)

        r   = self.r   * np.ones_like(self.data['rho'])
        th  = self.th  * np.ones_like(self.data['rho'])
        phi = self.phi * np.ones_like(self.data['rho'])
        
        x    = np.zeros(self.data['uu'].shape)
        x[1] = r * np.sin(th) * np.cos(phi)
        x[2] = r * np.sin(th) * np.sin(phi)
        x[3] = r * np.cos(th) * np.ones(phi.shape)
        return np.einsum('i...,j...->ij...', x, self.stress_energy_HD[:,0]) - (
               np.einsum('j...,i...->ij...', x, self.stress_energy_HD[:,0]))
    
    def calc_total_momentum_vector(self, **kwargs):
        """
        Angular momentum flux 
        (see below Eq. 17 in Fragile & Anninos 2005)
        """
        if 'THD' not in kwargs:
            EOS = '2T' if self.two_temp else False
            self.stress_energy_HD = self.calc_THD_cart(EOS=EOS, **kwargs)
        else:
            self.stress_energy_HD = kwargs.get('THD')
        return self.stress_energy_HD[:, 0]

    def calc_THD_cart(self, **kwargs):
        """
        Hyrodynamic stress-energy tensor in Cartesian coordinates
        See White, Quataert & Blaes (2019), Equation 5
        """
        pgas = kwargs.pop('pgas', None)
        if pgas is None:
            gamma_adi = 5./3.
            pgas      = (gamma_adi - 1.) * self.data['u_gas']
        Flu = Fluid(self.pars, self.CONSTANTS, cgs=self.kwargs.get('cgs'))
        return self.int_to_cart_tnsr(Flu.calc_THD(**self.data, **self.grid,
                                                  pgas=pgas, **kwargs))
    
    # Rotations 
    def rotate_vector(self, vec, **kwargs):
        """
        Rotate 4-vector based on precession and inclination angles.
        See Equation A1 of Kaaz et al. (2024).
        T : inclination (tilt) angle (T)
        P : precession angle (P)
        """
        shape = list(vec.shape)
        shape.insert(0, 4)
        rot = hr.Tools.zeroNd(shape, 1, **kwargs)
        if not kwargs.get('inverse', False):
            rot[0, 0] = 1.
            rot[1, 1] =  np.cos(self.inc) * np.cos(self.prec)
            rot[1, 2] = -np.cos(self.inc) * np.sin(self.prec)
            rot[1, 3] = -np.sin(self.inc) 
            rot[2, 1] =  np.sin(self.prec)
            rot[2, 2] =  np.cos(self.prec)
            rot[2, 3] =  0.
            rot[3, 1] =  np.sin(self.inc) * np.cos(self.prec)
            rot[3, 2] = -np.sin(self.inc) * np.sin(self.prec)
            rot[3, 3] =  np.cos(self.inc)
        else:
            rot[0, 0] = 1.
            rot[1, 1] =  np.cos(self.inc) * np.cos(self.prec)
            rot[2, 1] = -np.cos(self.inc) * np.sin(self.prec)
            rot[3, 1] = -np.sin(self.inc) 
            rot[1, 2] =  np.sin(self.prec)
            rot[2, 2] =  np.cos(self.prec)
            rot[3, 2] =  0.
            rot[1, 3] =  np.sin(self.inc) * np.cos(self.prec)
            rot[2, 3] = -np.sin(self.inc) * np.sin(self.prec)
            rot[3, 3] =  np.cos(self.inc)

        if kwargs.get('cpp', False):
            return Tools.contract_vector_cpp(vec, rot)
        else:
            return np.einsum('ij...,j...->i...', rot, vec)
    
    def rotate_tensor(self, tnsr, **kwargs):
        """
        Rotate tensor based on precession and inclination angles.
        See Equation A1 of Kaaz et al. (2024).
        T : inclination (tilt) angle (T)
        P : precession angle (P)
        """
        shape = list((self.r * self.th * self.phi).shape)
        shape.insert(0, 4)
        shape.insert(0, 4)
        rot = hr.Tools.zeroNd(shape, 1, **kwargs)
        if not kwargs.get('inverse', False):
            rot[0, 0] = 1.
            rot[1, 1] =  np.cos(self.inc) * np.cos(self.prec)
            rot[1, 2] = -np.cos(self.inc) * np.sin(self.prec)
            rot[1, 3] = -np.sin(self.inc) 
            rot[2, 1] =  np.sin(self.prec)
            rot[2, 2] =  np.cos(self.prec)
            rot[2, 3] =  0.
            rot[3, 1] =  np.sin(self.inc) * np.cos(self.prec)
            rot[3, 2] = -np.sin(self.inc) * np.sin(self.prec)
            rot[3, 3] =  np.cos(self.inc)
        else:
            rot[0, 0] = 1.
            rot[1, 1] =  np.cos(self.inc) * np.cos(self.prec)
            rot[2, 1] = -np.cos(self.inc) * np.sin(self.prec)
            rot[3, 1] = -np.sin(self.inc) 
            rot[1, 2] =  np.sin(self.prec)
            rot[2, 2] =  np.cos(self.prec)
            rot[3, 2] =  0.
            rot[1, 3] =  np.sin(self.inc) * np.cos(self.prec)
            rot[2, 3] = -np.sin(self.inc) * np.sin(self.prec)
            rot[3, 3] =  np.cos(self.inc)

        return np.einsum('kl...,ki...,lj...->ij...', tnsr, rot, rot)
    
class Christoffel_KS(Kerr_Schild):
    def __init__(self, r, theta, phi, pars, CONSTANTS, **kwargs):
        super().__init__(r, theta, phi, pars, CONSTANTS, **kwargs)
        shape = [4, 4, 4] + list((self.r * self.th * self.ph).shape)
        self.g = hr.Tools.zeroNd(shape, 1, **kwargs)

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
        dxdxp = hr.Tools.zeroNd(shape, 1, **kwargs)
        dxpdx = hr.Tools.zeroNd(shape, 1, **kwargs)
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