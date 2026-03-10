import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from matplotlib import patches

class Cartesian():
    def __init__(self, params, **kwargs):
        self.params = params
    
    def plot_xy(self, fig, ax, X, Y, D, cmap, FS, **kwargs):
        im = {}
        vmin, vmax = self.get_vmin_vmax(D, **kwargs)
        norm       = self.get_norm(vmin, vmax, **kwargs)

        if 'fix_pole' in kwargs:
            self.check_fix_pole(**kwargs)            
            th0   = kwargs.get('theta0', np.pi/2.)
            r     = kwargs.get('r')
            theta = kwargs.get('theta')
            phi   = kwargs.get('phi')

            FP = Fix_Pole()
            FP.fix_cart_xy(X, Y, D, r, theta, phi, theta0 = th0, 
                           fix_pole=kwargs.get('fix_pole', 'constant'))

        im['r'] = ax.pcolormesh(X['r'], Y['r'], D['r'],
                                norm = norm, cmap = cmap, shading = 'gouraud')
        cax = self.get_cax(fig, ax, im, FS, **kwargs)

        if 'fix_pole' in kwargs:
            FP.remove_poles_xy(X, Y, D)

        contours = kwargs.get('contours', False)
        if (contours):
            self.plot_contours(ax, im['r'], X['r'], Y['r'], contours)

        return im, cax
    
    def plot_Rz(self, fig, ax, R, Z, D, cmap, FS, **kwargs):
        im = {}
        vmin, vmax = self.get_vmin_vmax(D, **kwargs)
        norm       = self.get_norm(vmin, vmax, **kwargs)

        if 'fix_pole' in kwargs:
            self.check_fix_pole(**kwargs) 
            th0   = kwargs.get('theta0', np.pi/2.)
            r     = kwargs.get('r')
            theta = kwargs.get('theta')
            phi   = kwargs.get('phi')

            FP = Fix_Pole()
            FP.fix_cart_Rz(R, Z, D, r, theta, phi, theta0 = th0, 
                           fix_pole=kwargs.get('fix_pole', 'constant'))

        im['r'] = ax.pcolormesh(R['r'], Z['r'], D['r'],
                                norm = norm, cmap = cmap, shading = 'gouraud')
        cax = self.get_cax(fig, ax, im, FS, **kwargs)

        if 'fix_pole' in kwargs:
            FP.remove_poles_Rz(R, Z, D)

        contours = kwargs.get('contours', False)
        if (contours):
            self.plot_contours(ax, im['r'], R['r'], Z['r'], contours)

        return im, cax
    
    def plot_xz(self, fig, ax, X, Z, D, cmap, FS, **kwargs):
        im = {}
        vmin, vmax = self.get_vmin_vmax(D, **kwargs)
        norm       = self.get_norm(vmin, vmax, **kwargs)
        contours_l = kwargs.get('contours_left', False)
        contours_r = kwargs.get('contours_right', False)

        if 'fix_pole' in kwargs:
            self.check_fix_pole(**kwargs)
            phi0  = kwargs.get('phi0', 0.0)
            r     = kwargs.get('r')
            theta = kwargs.get('theta')
            phi   = kwargs.get('phi')

            FP = Fix_Pole()
            FP.fix_cart_xz(X, Z, D, r, theta, phi, phi0 = phi0, 
                           fix_pole=kwargs.get('fix_pole', 'constant'))

        im['l'] = ax.pcolormesh(X['l'], Z['l'], D['l'],
                                norm = norm, cmap = cmap, shading = 'gouraud')
        im['r'] = ax.pcolormesh(X['r'], Z['r'], D['r'],
                                norm = norm, cmap = cmap, shading = 'gouraud')
        cax = self.get_cax(fig, ax, im, FS, **kwargs)

        if 'fix_pole' in kwargs:
            FP.remove_poles_xz(X, Z, D)

        if (contours_l):
            self.plot_contours(ax, im['l'], X['l'], Z['l'], contours_l)
        if (contours_r):
            self.plot_contours(ax, im['r'], X['r'], Z['r'], contours_r)

        return im, cax

    def plot_xz_mirror(self, fig, ax, X, Z, Dl, Dr, cmaps, FS, **kwargs):
        im   = {}
        vmin_l, vmax_l = self.get_vmin_vmax(Dl, 
            lim=kwargs.get('lim_left', False), 
            sym=kwargs.get('sym_left', False))
        norm_l         = self.get_norm(vmin_l, vmax_l,
            log=kwargs.get('log_left', False), 
            sym=kwargs.get('sym_left', False),
            linthresh=kwargs.get('linthresh_left', False))
        contours_l     = kwargs.get('contours_left', False)
        extend_l       = kwargs.get('extend_left', 'neither')

        vmin_r, vmax_r = self.get_vmin_vmax(Dr,
            lim=kwargs.get('lim_right', False), 
            sym=kwargs.get('sym_right', False))
        norm_r         = self.get_norm(vmin_r, vmax_r,
            log=kwargs.get('log_right', False), 
            sym=kwargs.get('sym_right', False),
            linthresh=kwargs.get('linthresh_right', False))
        contours_r     = kwargs.get('contours_right', False)
        extend_r       = kwargs.get('extend_right', 'neither')

        if 'fix_pole' in kwargs:
            self.check_fix_pole(**kwargs) 
            phi0  = kwargs.get('phi0', 0.0)
            r     = kwargs.get('r')
            theta = kwargs.get('theta')
            phi   = kwargs.get('phi')

            FP = Fix_Pole()
            FP.fix_cart_xz_mirror(X, Z, Dl, Dr, r, theta, phi, phi0 = phi0, 
                           fix_pole=kwargs.get('fix_pole', 'constant'))

        im['l'] = ax.pcolormesh(X['l'], Z['l'], Dl['l'],
                                norm = norm_l, cmap = cmaps[0], 
                                shading = 'gouraud')
        im['r'] = ax.pcolormesh(X['r'], Z['r'], Dr['r'],
                                norm = norm_r, cmap = cmaps[1], 
                                shading = 'gouraud')
        
        cax_r = self.get_cax(fig, ax, im, FS, extend=extend_r, **kwargs)
        cax_l = self.get_cax(fig, ax, im, FS, extend=extend_l, 
                             cbar_loc='top', **kwargs)

        if 'fix_pole' in kwargs:
            FP.remove_poles_xz_mirror(X, Z, Dl, Dr)

        if (contours_l):
            self.plot_contours(ax, im['l'], X['l'], Z['l'], contours_l)
        if (contours_r):
            self.plot_contours(ax, im['r'], X['r'], Z['r'], contours_r)

        return im, cax_l, cax_r

    def plot_contours(self, ax, im, X, Y, contours):
        if 'linestyles' not in contours:
            contours['linestyles'] = len(contours['quantities']) * ['-']
        for (quantity, level, linewidth, linestyle, color) in zip(
            contours['quantities'], contours['levels'], 
            contours['linewidths'], contours['linestyles'], 
            contours['colors']):
            if not (isinstance(level, np.ndarray) or isinstance(level, list)):
                plt_level = [level] 
            else:
                plt_level = level
            ax.contour(X, Y, quantity, levels=plt_level, 
                linewidths=linewidth, linestyles=linestyle, colors=color)
        return None

    def plot_xy_vec(self, fig, ax, X, Y, X_uni, Y_uni, VX, VY, **kwargs):
        method= kwargs.get('interp_method', 'cubic')
        xlim  = 1.1 * np.amax(np.abs(X_uni))
        ylim  = 1.1 * np.amax(np.abs(Y_uni))

        x_dat  = X['r'].ravel()
        y_dat  = Y['r'].ravel()
        VX_dat = VX['r'].ravel()
        VY_dat = VY['r'].ravel()

        selec   = (np.abs(x_dat) <= xlim) & (np.abs(y_dat) <= ylim)
        x_data  = x_dat[selec]
        y_data  = y_dat[selec]
        VX_data = VX_dat[selec]
        VY_data = VY_dat[selec]

        VX_uni = griddata((x_data, y_data), VX_data, 
                          (X_uni, Y_uni), method=method)
        VY_uni = griddata((x_data, y_data), VY_data, 
                          (X_uni, Y_uni), method=method)

        if 'mask' in kwargs:
            mask      = kwargs.get('mask')
            mask_dat  = (mask['r'].astype(int)).ravel()
            mask_data = mask_dat[selec]
            mask_uni  = griddata((x_data, y_data), mask_data, 
                                 (X_uni, Y_uni), method='nearest')
            VX_uni = np.ma.array(VX_uni, mask=mask_uni)
            VY_uni = np.ma.array(VY_uni, mask=mask_uni)
        
        if 'color_streamlines' in kwargs:
            color_streamlines = kwargs.get('color_streamlines')
            values            = color_streamlines['values']
            cmap              = color_streamlines['cmap']

            values_dat  = values['r'].ravel()
            values_data = values_dat[selec]
            values_uni  = griddata((x_data, y_data), values_data, 
                                   (X_uni, Y_uni), method=method)
        
        dens = kwargs.get('dens', 1.0)

        streams = ax.streamplot(X_uni, Y_uni, VX_uni, VY_uni, 
            density = (dens, dens),
            broken_streamlines = kwargs.get('broken_streamlines', True),
            linewidth = kwargs.get('linewidth', 1.0), 
            color = (kwargs.get('color', 'white') 
                     if 'color_streamlines' not in kwargs else values_uni), 
            cmap = None if 'color_streamlines' not in kwargs else cmap,
            arrowsize = kwargs.get('arrowsize', 1.0))

        if kwargs.get('return_streamlines', False):
            return streams
        
    def plot_Rz_vec(self, fig, ax, R, Z, R_uni, Z_uni, VR, VZ, **kwargs):
        return self.plot_xy_vec(fig, ax, R, Z, R_uni, Z_uni, VR, VZ, **kwargs)

    def plot_xz_vec(self, fig, ax, X, Z, X_uni, Z_uni, VX, VZ, **kwargs):
        # Do not allow streamlines to cross pole unless cross_pole=True
        if (kwargs.get('cross_pole', False)):
            return self.plot_xz_vec_cross_pole(fig, ax, X, Z, X_uni, Z_uni, 
                                               VX, VZ, **kwargs)
        X_l, Z_l   = {'r' : X['l']},  {'r' : Z['l']}
        VX_l, VZ_l = {'r' : VX['l']}, {'r' : VZ['l']}
        r1D, z1D   = X_uni[0, :], Z_uni[:, 0]
        r1D_pos, r1D_neg = r1D[r1D > 0], r1D[r1D < 0]
        X_uni_l, Z_uni_l = np.meshgrid(r1D_neg, z1D)
        X_uni_r, Z_uni_r = np.meshgrid(r1D_pos, z1D)

        kwargs_l = kwargs.copy()
        if 'mask' in kwargs:
            mask     = {'r' : kwargs.get('mask')['l']}
            kwargs_l['mask'] = mask
        if 'color_streamlines' in kwargs:
            values   = {'r' : kwargs.get('values')['l']}
            kwargs_l['values'] = values
        if 'color' in kwargs:
            color = kwargs.get('color')
            if isinstance(color, dict):
                kwargs_l['color'] = color['l']
                kwargs['color']   = color['r']

        self.plot_Rz_vec(fig, ax, X_l, Z_l, X_uni_l, Z_uni_l, VX_l, VZ_l, **kwargs_l)
        return self.plot_Rz_vec(fig, ax, X, Z, X_uni_r, Z_uni_r, VX, VZ, **kwargs)
            
    def plot_xz_vec_cross_pole(self, fig, ax, X, Z, X_uni, Z_uni, VX, VZ, **kwargs):
        method= kwargs.get('interp_method', 'cubic')
        xlim  = 1.1 * np.amax(np.abs(X_uni))
        zlim  = 1.1 * np.amax(np.abs(Z_uni))

        x_dat  = np.concatenate([X['r'].ravel(),  X['l'].ravel()])
        z_dat  = np.concatenate([Z['r'].ravel(),  Z['l'].ravel()])
        VX_dat = np.concatenate([VX['r'].ravel(), VX['l'].ravel()])
        VZ_dat = np.concatenate([VZ['r'].ravel(), VZ['l'].ravel()])

        selec   = (np.abs(x_dat) <= xlim) & (np.abs(z_dat) <= zlim)
        x_data  = x_dat[selec]
        z_data  = z_dat[selec]
        VX_data = VX_dat[selec]
        VZ_data = VZ_dat[selec]

        VX_uni = griddata((x_data, z_data), VX_data, 
                          (X_uni, Z_uni), method=method)
        VZ_uni = griddata((x_data, z_data), VZ_data, 
                          (X_uni, Z_uni), method=method)

        if 'mask' in kwargs:
            mask      = kwargs.get('mask')
            mask_dat  = np.concatenate([(mask['r'].astype(int)).ravel(),
                                        (mask['l'].astype(int)).ravel()])
            mask_data = mask_dat[selec]
            mask_uni  = griddata((x_data, z_data), mask_data, 
                                 (X_uni, Z_uni), method='nearest')
            VX_uni = np.ma.array(VX_uni, mask=mask_uni)
            VZ_uni = np.ma.array(VZ_uni, mask=mask_uni)

        if 'color_streamlines' in kwargs:
            color_streamlines = kwargs.get('color_streamlines')
            values            = color_streamlines['values']
            cmap              = color_streamlines['cmap']

            values_dat  = np.concatenate(
                [values['r'].ravel(), values['l'].ravel()])
            values_data = values_dat[selec]
            values_uni  = griddata((x_data, z_data), values_data, 
                                   (X_uni, Z_uni), method=method)

        dens = kwargs.get('dens', 1.0)
        
        ax.streamplot(X_uni, Z_uni, VX_uni, VZ_uni, 
            density = (dens, dens),
            broken_streamlines = kwargs.get('broken_streamlines', True),
            linewidth = kwargs.get('linewidth', 1.0), 
            color = (kwargs.get('color', 'white') 
                if 'color_streamlines' not in kwargs else values_uni), 
            cmap = (None if 'color_streamlines' not in kwargs else cmap),
            arrowsize = kwargs.get('arrowsize', 1.0))      
                   
    def plot_xz_mirror_vec(self, fig, ax, X, Z, 
                           X_uni, Z_uni, VX, VZ, **kwargs): 
        method= kwargs.get('interp_method', 'cubic')
        xlim  = 1.1 * np.amax(np.abs(X_uni))
        zlim  = 1.1 * np.amax(np.abs(Z_uni))

        x_dat  = X['r'].ravel()
        z_dat  = Z['r'].ravel()
        VX_dat = VX['r'].ravel()
        VZ_dat = VZ['r'].ravel()

        selec   = (np.abs(x_dat) <= xlim) & (np.abs(z_dat) <= zlim)
        x_data  = x_dat[selec]
        z_data  = z_dat[selec]
        VX_data = VX_dat[selec]
        VZ_data = VZ_dat[selec]

        VX_uni = griddata((x_data, z_data), VX_data, 
                          (X_uni, Z_uni), method=method)
        VZ_uni = griddata((x_data, z_data), VZ_data, 
                          (X_uni, Z_uni), method=method)

        if 'mask' in kwargs:
            mask      = kwargs.get('mask')
            mask_dat  = (mask['r'].astype(int)).ravel()
            mask_data = mask_dat[selec]
            mask_uni  = griddata((x_data, z_data), mask_data, 
                                 (X_uni, Z_uni), method='nearest')
            VX_uni = np.ma.array(VX_uni, mask=mask_uni)
            VZ_uni = np.ma.array(VZ_uni, mask=mask_uni)
        
        if 'color_streamlines' in kwargs:
            color_streamlines = kwargs.get('color_streamlines')
            values            = color_streamlines['values']
            cmap              = color_streamlines['cmap']

            values_dat  = values['r'].ravel()
            values_data = values_dat[selec]
            values_uni  = griddata((x_data, z_data), values_data, 
                                   (X_uni, Z_uni), method=method)
        
        dens = kwargs.get('dens', 1.0)

        ax.streamplot(X_uni, Z_uni, VX_uni, VZ_uni, 
                      density = (dens, dens),
                      broken_streamlines = (
                          kwargs.get('broken_streamlines', True)),
                      linewidth = kwargs.get('linewidth', 1.0), 
                      color = (kwargs.get('color', 'white') 
                        if 'color_streamlines' not in kwargs else values_uni), 
                      cmap = (
                          None if 'color_streamlines' not in kwargs else cmap),
                      arrowsize = kwargs.get('arrowsize', 1.0))

        streams = ax.streamplot(X_uni, Z_uni, VX_uni, VZ_uni, 
                      density = (dens, dens),
                      broken_streamlines = (
                          kwargs.get('broken_streamlines', True)),
                      linewidth = kwargs.get('linewidth', 1.0), 
                      color = (kwargs.get('color', 'white') 
                        if 'color_streamlines' not in kwargs else values_uni), 
                      cmap = (
                          None if 'color_streamlines' not in kwargs else cmap),
                      arrowsize = kwargs.get('arrowsize', 1.0))
        new_lines = []
        for line in streams.lines.get_segments():
            new_line = line.copy()
            new_line[:, 0] *= -1
            new_lines.append(new_line)

        streams.lines.set_segments(new_lines)

    def standard_cbar(self, fig, ax, images, FS, **kwargs):
        divider = make_axes_locatable(ax)
        extend  = kwargs.get('extend', 'neither')
        if (kwargs.get('cbar_loc') == 'top'):
            cax = divider.append_axes("top", size="4%", pad=0.2)
            cbar = fig.colorbar(images, cax=cax, extend=extend, 
                                orientation='horizontal')
            cax.xaxis.set_ticks_position("top")
        else:
            cax = divider.append_axes("right", size="4%", pad=0.2)
            cbar = fig.colorbar(images, cax=cax, extend=extend)
        if(kwargs.get('title')):
            title = kwargs.get('title')
            cbar.ax.set_xlabel(title, labelpad=20, fontsize=FS)
        return cax

    def get_vmin_vmax(self, D, **kwargs):
        if ('l' in D and 'r' in D):
            DD = np.concatenate((D['l'], D['r']))
        elif ('l' in D and 'r' not in D): 
            DD = D['l']
        else: 
            DD = D['r']
        if (kwargs.get('lim')):
            lim = kwargs.get('lim')
            return lim[0], lim[1]
        else:
            if (kwargs.get('sym')):
                if (np.abs(np.amin(DD)) > np.amax(DD)): 
                    return np.amin(DD), np.abs(np.amin(DD))
                else:
                    return -np.amax(DD), np.amax(DD)
            else: 
                minD, maxD = np.percentile(DD, 5), np.percentile(DD, 95)
                return minD, maxD

    def get_norm(self, vmin, vmax, **kwargs):
        if (kwargs.get('log')):
            if (kwargs.get('sym')):
                print('Symmetric log')
                return colors.SymLogNorm(linthresh = kwargs.get('linthresh'),
                                         vmin = vmin, vmax = vmax)
            else:
                print('Positive log')
                return colors.LogNorm(vmin = vmin, vmax = vmax)
        else:
            return colors.Normalize(vmin = vmin, vmax = vmax)
        
    def get_cax(self, fig, ax, im, FS, **kwargs):
        if (kwargs.get('nocbar')):
            return 'None'
        else:
            if (kwargs.get('cbar_loc') == 'left'):
                return self.standard_cbar(fig, ax, im['l'], FS, **kwargs)
            elif (kwargs.get('cbar_loc') == 'top'):
                return self.standard_cbar(fig, ax, im['l'], FS, **kwargs)
            else:
                return self.standard_cbar(fig, ax, im['r'], FS, **kwargs)

    def check_fix_pole(self, **kwargs):
        if ('r' not in kwargs or 'theta' not in kwargs or 
            'phi' not in kwargs):
            raise ValueError("r, theta, and phi " + 
                             "must be specified to use fix_pole")
        if (kwargs.get('fix_pole', 'constant') != 'constant' and
            kwargs.get('fix_pole', 'constant') != 'average'):
            raise ValueError("Only constant and average kwarg for " +
                             "fix_pole has been implemented currently.")

class Fix_Pole():
    def __init__(self):
        pass

    def fix_cart_xy(self, X, Y, D, r, theta, phi, **kwargs):
        r1D, th1D, phi1D = self.get1D(r, theta, phi)
        phi_min, phi_max = np.amin(phi1D), np.amax(phi1D)
        phi_strt = 0. * np.ones(len(r1D))
        dphi     = np.diff(phi1D)[-1]
        phi_end  = (2. * np.pi) * np.ones(len(r1D))
        dat_strt, dat_end = D['r'][:, 0], D['r'][:, -1]
        th0      = kwargs.get('theta0', np.pi/2.)
        x_strt   = r1D * np.sin(th0) * np.cos(phi_strt)
        x_end    = r1D * np.sin(th0) * np.cos(phi_end)
        y_strt   = r1D * np.sin(th0) * np.sin(phi_strt)
        y_end    = r1D * np.sin(th0) * np.sin(phi_end)

        X['r']   = self.concatenate(X['r'], x_strt, x_end)
        Y['r']   = self.concatenate(Y['r'], y_strt, y_end)
        if kwargs.get('fix_pole', 'constant') == 'constant':
            D['r'] = self.concatenate(D['r'], dat_strt, dat_end)
        elif kwargs.get('fix_pole', 'constant') == 'average':
            dat_avg = 0.5 * (dat_strt + dat_end)
            D['r'] = self.concatenate(D['r'], dat_avg, dat_avg)

    def fix_cart_xz(self, X, Z, D, r, theta, phi, **kwargs):
        r1D, th1D, phi1D = self.get1D(r, theta, phi)
        th_strt, th_end     = 0., np.pi
        rdat_strt, rdat_end = D['r'][:, 0], D['r'][:, -1]
        ldat_strt, ldat_end = D['l'][:, 0], D['l'][:, -1]
        x_strt  = r1D * np.sin(th_strt)
        x_end   = r1D * np.sin(th_end)
        z_strt  = r1D * np.cos(th_strt)
        z_end   = r1D * np.cos(th_end) 

        X['r']  = self.concatenate(X['r'], x_strt, x_end)
        Z['r']  = self.concatenate(Z['r'], z_strt, z_end)
        X['l']  = self.concatenate(X['l'], x_strt, x_end)
        Z['l']  = self.concatenate(Z['l'], z_strt, z_end)

        if kwargs.get('fix_pole', 'constant') == 'constant':
            D['r'] = self.concatenate(D['r'], rdat_strt, rdat_end)
            D['l'] = self.concatenate(D['l'], ldat_strt, ldat_end)
        elif kwargs.get('fix_pole', 'constant') == 'average':
            print("Warning: you're averaging two separate poles.")
            dat_avg = 0.5 * (rdat_strt + rdat_end)
            D['r'] = self.concatenate(D['r'], dat_avg, dat_avg)  
            dat_avg = 0.5 * (ldat_strt + ldat_end)
            D['l'] = self.concatenate(D['l'], dat_avg, dat_avg)  

    def fix_cart_Rz(self, R, Z, D, r, theta, phi, **kwargs):
        r1D, th1D, phi1D = self.get1D(r, theta, phi)
        th_strt, th_end  = 0., np.pi
        dat_strt, dat_end = D['r'][:, 0], D['r'][:, -1]
        R_strt  = r1D * np.sin(th_strt)
        R_end   = r1D * np.sin(th_end)
        z_strt  = r1D * np.cos(th_strt)
        z_end   = r1D * np.cos(th_end) 

        R['r']  = self.concatenate(R['r'], R_strt, R_end)
        Z['r']  = self.concatenate(Z['r'], z_strt, z_end)

        if kwargs.get('fix_pole', 'constant') == 'constant':
            D['r'] = self.concatenate(D['r'], dat_strt, dat_end)
        elif kwargs.get('fix_pole', 'constant') == 'average':
            print("Warning: you're averaging two separate poles.")
            dat_avg = 0.5 * (dat_strt + dat_end)
            D['r'] = self.concatenate(D['r'], dat_avg, dat_avg)  
  
    def fix_cart_xz_mirror(self, X, Z, Dl, Dr, r, theta, phi, **kwargs):
        r1D, th1D, phi1D      = self.get1D(r, theta, phi)
        th_strt, th_end  = 0., np.pi
        ldat_strt, ldat_end = Dl['l'][:, 0], Dl['l'][:, -1]
        rdat_strt, rdat_end = Dr['r'][:, 0], Dr['r'][:, -1]
        phi0    = kwargs.get('phi0', 0.0)
        x_strt  = r1D * np.sin(th_strt)
        x_end   = r1D * np.sin(th_end)
        z_strt  = r1D * np.cos(th_strt)
        z_end   = r1D * np.cos(th_end)

        X['r']  = self.concatenate(X['r'], x_strt, x_end)
        Z['r']  = self.concatenate(Z['r'], z_strt, z_end)
        X['l']  = self.concatenate(X['l'], -x_strt, -x_end)
        Z['l']  = self.concatenate(Z['l'], z_strt, z_end)
        if kwargs.get('fix_pole', 'constant') == 'constant':
            Dr['r'] = self.concatenate(Dr['r'], rdat_strt, rdat_end)
            Dl['l'] = self.concatenate(Dl['l'], ldat_strt, ldat_end)
        elif kwargs.get('fix_pole', 'constant') == 'average':
            rdat_avg = 0.5 * (rdat_strt + rdat_end)
            ldat_avg = 0.5 * (ldat_strt + ldat_end)
            Dr['r'] = self.concatenate(Dr['r'], rdat_avg, rdat_avg)
            Dl['l'] = self.concatenate(Dl['l'], ldat_avg, ldat_avg)

    def remove_poles_xy(self, X, Y, D):
        X['r'] = X['r'][:, 1:-1]
        Y['r'] = Y['r'][:, 1:-1]
        D['r'] = D['r'][:, 1:-1]

    def remove_poles_xz(self, R, Z, D):
        R['l'] = R['l'][:, 1:-1]
        Z['l'] = Z['l'][:, 1:-1]
        R['r'] = R['r'][:, 1:-1]
        Z['r'] = Z['r'][:, 1:-1]
        D['l'] = D['l'][:, 1:-1]
        D['r'] = D['r'][:, 1:-1]

    def remove_poles_Rz(self, R, Z, D):
        return self.remove_poles_xy(R, Z, D)

    def remove_poles_xz_mirror(self, X, Z, Dl, Dr):
        X['l']  = X['l'][:, 1:-1]
        Z['l']  = Z['l'][:, 1:-1]
        X['r']  = X['r'][:, 1:-1]
        Z['r']  = Z['r'][:, 1:-1]
        Dl['l'] = Dl['l'][:, 1:-1]
        Dr['r'] = Dr['r'][:, 1:-1]

    def concatenate(self, q, q_strt, q_end):
        return np.concatenate((q_strt[:, None], q, q_end[:, None]), axis=1)
    
    def get1D(self, r, theta, phi):
        return self.get_r1D(r), self.get_theta1D(theta), self.get_phi1D(phi)
    
    def get_r1D(self, r):
        match len(r.shape):
            case 1:
                return r
            case 2:
                return r[0]
            case 3:
                return r[:, 0, 0]
            case 4:
                return r[0, :, 0, 0]
            
    def get_theta1D(self, theta):
        match len(theta.shape):
            case 1:
                return theta
            case 2:
                return theta[0, :]
            case 3:
                return theta[0, :, 0]
            case 4:
                return theta[0, 0, :, 0]
            
    def get_phi1D(self, phi):
        match len(phi.shape):
            case 1:
                return phi
            case 2:
                return phi[:, 0]
            case 3:
                return phi[0, 0, :]
            case 4:
                return phi[0, 0, 0, :]

class HAMR_Cartesian(Cartesian):
    # Create plots assuming 3D data (similar to original pipeline)
    def __init__(self):
        super().__init__([])

    def plot_xy(self, fig, ax, Q, r, theta, phi, **kwargs):
        kwargs = self.update_kwargs(r, theta, phi, **kwargs)
        FS   = kwargs.pop('FS', 20)
        cmap = kwargs.pop('cmap','viridis')
        ith  = int(len(theta[0, 0, :, 0]) / 2)
        x3D  = r * np.sin(theta) * np.cos(phi)
        y3D  = r * np.sin(theta) * np.sin(phi)
        X, Y, D = {}, {}, {}
        X['r'] = 0.5 * (x3D[0, :, ith-1, :] + x3D[0, :, ith, :])
        Y['r'] = 0.5 * (y3D[0, :, ith-1, :] + y3D[0, :, ith, :])
        D['r'] = 0.5 * (Q[  0, :, ith-1, :] + Q[  0, :, ith, :])
        return super().plot_xy(fig, ax, X, Y, D, cmap, FS, **kwargs)

    def plot_Rz(self, fig, ax, Q, r, theta, phi, **kwargs):
        # phi0 specifies the target phi value of the slice
        # iphi overrides phi0 to select a specific index
        kwargs = self.update_kwargs(r, theta, phi, **kwargs)
        FS   = kwargs.pop('FS', 20)
        cmap = kwargs.pop('cmap','viridis')
        phi0 = kwargs.get('phi0', 0.)
        if 'iphi' in kwargs:
            iphi = int(kwargs.get('iphi', 0))
        else:
            iphi = int(np.argmin(np.abs(phi[0, 0, 0, :] - phi0)))

        R3D = r * np.sin(theta) * np.ones_like(phi)
        z3D = r * np.cos(theta) * np.ones_like(phi)
        R, Z, D = {}, {}, {}
        R['r'] = R3D[0, :, :, iphi]
        Z['r'] = z3D[0, :, :, iphi]
        D['r'] = Q[  0, :, :, iphi]
        return super().plot_Rz(fig, ax, R, Z, D, cmap, FS, **kwargs)

    def plot_xz(self, fig, ax, Q, r, theta, phi, **kwargs):
        kwargs = self.update_kwargs(r, theta, phi, **kwargs)
        FS   = kwargs.pop('FS', 20)
        cmap = kwargs.pop('cmap','viridis')
        phi0 = kwargs.get('phi0', 0.)
        if 'iphi' in kwargs:
            iphi = int(kwargs.get('iphi', 0))
        else:
            iphi = int(np.argmin(np.abs(phi[0, 0, 0, :] - phi0)))
        if (iphi < len(phi[0, 0, 0, :]) / 2):
            ipi = int(iphi + len(phi[0, 0, 0, :]) / 2)
        else:
            iphi = int(iphi - len(phi[0, 0, 0, :]) / 2)
            ipi  = int(iphi + len(phi[0, 0, 0, :]) / 2)
        R3D = r * np.sin(theta) * np.ones_like(phi)
        z3D = r * np.cos(theta) * np.ones_like(phi)
        X, Z, D = {}, {}, {}
        X['r'] = R3D[ 0, :, :, iphi]
        Z['r'] = z3D[ 0, :, :, iphi]
        D['r'] = Q[   0, :, :, iphi]
        X['l'] = -R3D[0, :, :, ipi]
        Z['l'] = z3D[ 0, :, :, ipi]
        D['l'] = Q[   0, :, :, ipi]
        return super().plot_xz(fig, ax, X, Z, D, cmap, FS, **kwargs)

    def plot_xz_mirror(self, fig, ax, Ql, Qr, r, theta, phi, **kwargs):
        # Plots R-z slices mirrored
        # phi0 specifies the target phi value of the slice
        # iphi overrides phi0 to select a specific index
        kwargs = self.update_kwargs(r, theta, phi, **kwargs)
        FS    = kwargs.pop('FS', 20)
        cmaps = kwargs.pop('cmaps',['viridis', 'viridis'])
        phi0  = kwargs.get('phi0', 0.)
        if 'iphi' in kwargs:
            iphi = int(kwargs.get('iphi', 0))
        else:
            iphi = int(np.argmin(np.abs(phi[0, 0, 0, :] - phi0)))

        R3D = r * np.sin(theta) * np.ones_like(phi)
        z3D = r * np.cos(theta) * np.ones_like(phi)
        X, Z, Dl, Dr = {}, {}, {}, {}
        X['r']  = R3D[ 0, :, :, iphi]
        Z['r']  = z3D[ 0, :, :, iphi]
        Dr['r'] = Qr[  0, :, :, iphi]
        X['l']  = R3D[ 0, :, :, iphi]
        Z['l']  = -z3D[0, :, :, iphi]
        Dl['l'] = Ql[  0, :, :, iphi]
        return super().plot_xz_mirror(fig, ax, X, Z, Dl, Dr, 
                                      cmaps, FS, **kwargs)
    
    def update_kwargs(self, r, theta, phi, **kwargs):
        if kwargs.get('fix_pole', False):
            kwargs['r']     = r
            kwargs['theta'] = theta
            kwargs['phi']   = phi
        return kwargs

class HAMR_Cartesian_Block(Cartesian):
    # Create plots with block boundaries assuming 3D data 
    # (similar to original pipeline)
    def __init__(self, params):
        self.params = params
        self.dth    = (np.pi / 2.) * self.params['dx2']
        self.dphi   = self.params['dx3']

    def plot_xy(self, fig, ax, Q, r, theta, phi, **kwargs):
        theta0 = kwargs.get('theta0', np.pi/2.)
        FS   = kwargs.pop('FS', 20)
        cmap = kwargs.pop('cmap','viridis')
        im = {}
        Q_plt = {'r' : Q}
        vmin, vmax = self.get_vmin_vmax(Q_plt, **kwargs)
        norm       = self.get_norm(vmin, vmax, **kwargs)
        
        plt_args = {
            'norm'    : norm,
            'cmap'    : cmap,
            'shading' : 'gouraud'
        }
        
        nb  = Q.shape[0]
        x3D = r * np.sin(theta) * np.cos(phi)
        y3D = r * np.sin(theta) * np.sin(phi)
        for i in range(nb):
            if (np.any(theta[i, 0, :, 0] > theta0) and 
                np.any(theta[i, 0, :, 0] < theta0 + self.dth)):
                ith = np.argmin(np.abs(theta[i, 0, :, 0] - theta0))
                X = x3D[i, :, ith, :]
                Y = y3D[i, :, ith, :]
                D = Q[  i, :, ith, :]

                im['r'] = ax.pcolormesh(X, Y, D, **plt_args)

                if kwargs.get('show_grid', 0):
                    ds = kwargs.get('show_grid', 0)
                    self.plot_grid(ax, 'xy', ds, r[i], theta[i], phi[i], 
                                   ith=ith, **kwargs)
        
        cax = self.get_cax(fig, ax, im, FS, **kwargs)
        return im, cax

    def plot_Rz(self, fig, ax, Q, r, theta, phi, **kwargs):
        phi0 = kwargs.get('phi0', 0.)
        FS   = kwargs.pop('FS', 20)
        cmap = kwargs.pop('cmap','viridis')
        im    = {}
        Q_plt = {'r' : Q}
        vmin, vmax = self.get_vmin_vmax(Q_plt, **kwargs)
        norm       = self.get_norm(vmin, vmax, **kwargs)
        
        plt_args = {
            'norm'    : norm,
            'cmap'    : cmap,
            'shading' : 'gouraud'
        }

        nb  = Q.shape[0]
        R3D = r * np.sin(theta) * np.ones_like(phi)
        z3D = r * np.cos(theta) * np.ones_like(phi)
        for i in range(nb):
            dphi = np.diff(phi[i, 0, 0, :])[0]
            if (np.any(phi[i, 0, 0, :] > phi0) and 
                np.any(phi[i, 0, 0, :] < phi0 + dphi)):
                iphi = np.argmin(np.abs(phi[i, 0, 0, :] - phi0))
                R = R3D[i, :, :, iphi]
                Z = z3D[i, :, :, iphi]
                D = Q[  i, :, :, iphi]
                im['r'] = ax.pcolormesh(R, Z, D, **plt_args)

                if kwargs.get('show_grid', 0):
                    ds = kwargs.get('show_grid', 0)
                    self.plot_grid(ax, 'Rz', ds, r[i], theta[i], phi[i], 
                                   iphi=iphi, **kwargs)
        
        cax = self.get_cax(fig, ax, im, FS, **kwargs)
        return im, cax

    def plot_xz(self, fig, ax, Q, r, theta, phi, **kwargs):
        phi0 = kwargs.get('phi0', 0.)
        if (phi0 < np.pi):
            pi0 = phi0 + np.pi
        else:
            phi0 = phi0 - np.pi
            pi0  = phi0 + np.pi
        FS   = kwargs.pop('FS', 20)
        cmap = kwargs.pop('cmap','viridis')
        im    = {}
        Q_plt = {'r' : Q}
        vmin, vmax = self.get_vmin_vmax(Q_plt, **kwargs)
        norm       = self.get_norm(vmin, vmax, **kwargs)
        
        plt_args = {
            'norm'    : norm,
            'cmap'    : cmap,
            'shading' : 'gouraud'
        }

        nb  = Q.shape[0]
        x3D = r * np.sin(theta) * np.cos(phi)
        z3D = r * np.cos(theta) * np.ones_like(phi)
        for i in range(nb):
            dphi = np.diff(phi[i, 0, 0, :])[0]
            if (np.any(phi[i, 0, 0, :] > phi0) and 
                np.any(phi[i, 0, 0, :] < phi0 + dphi)):
                iphi = np.argmin(np.abs(phi[i, 0, 0, :] - phi0))
                X = x3D[i, :, :, iphi]
                Z = z3D[i, :, :, iphi]
                D = Q[  i, :, :, iphi]
                im['r'] = ax.pcolormesh(X, Z, D, **plt_args)

                if kwargs.get('show_grid', 0):
                    ds = kwargs.get('show_grid', 0)
                    self.plot_grid(ax, 'xz', ds, r[i], theta[i], phi[i], 
                                   iphi=iphi, **kwargs)

            if (np.any(phi[i, 0, 0, :] > pi0 - dphi) and 
                np.any(phi[i, 0, 0, :] < pi0)):
                ipi = np.argmin(np.abs(phi[i, 0, 0, :] - pi0))
                X = x3D[i, :, :, ipi]
                Z = z3D[i, :, :, ipi]
                D = Q[  i, :, :, ipi]
                im['l'] = ax.pcolormesh(X, Z, D, **plt_args)

                if kwargs.get('show_grid', 0):
                    ds = kwargs.get('show_grid', 0)
                    self.plot_grid(ax, 'xz', ds, r[i], theta[i], phi[i], 
                                   iphi=ipi, **kwargs)
        
        cax = self.get_cax(fig, ax, im, FS, **kwargs)
        return im, cax

    def plot_grid(self, ax, orientation, ds, r, theta, phi, **kwargs):
        grid_color = kwargs.get('grid_color', 'white')
        grid_width = kwargs.get('grid_width', 0.1)
        dr   = np.diff(r[:, 0, 0])[0]
        dth  = np.diff(theta[0, :, 0])[0]
        dphi = np.diff(phi[0, 0, :])[0]
        nr   = len(r[:, 0, 0])
        nth  = len(theta[0, :, 0])
        nphi = len(phi[0, 0, :])
        if orientation == 'xz' or orientation == 'Rz':
            iphi = kwargs.get('iphi', 0)
            if nr % ds != 0 or nth % ds != 0:
                raise ValueError("show_grid must divide nr and nth")
            xshift = r * np.sin(theta - dth/2.) * np.cos(phi)
            zshift = r * np.cos(theta - dth/2.) * np.ones_like(phi)
            for ith in range(0, nth, ds):
                xline = xshift[:, ith, iphi]
                zline = zshift[:, ith, iphi]
                ax.plot(xline, zline, color=grid_color, lw=grid_width)
            for ir in range(0, nr, ds):
                r0     = r[ir, 0, 0] - dr/2.
                th_min = np.amin(theta[0, :, 0]) - dth/2.
                th_max = np.amax(theta[0, :, 0]) + dth/2.
                th1D   = np.linspace(th_min, th_max, 100)
                phi0   = phi[0, 0,  iphi]
                pm     = int(np.cos(phi0) > 0.) - int(np.cos(phi0) < 0.)
                xcurve = pm * r0 * np.sin(th1D) 
                zcurve = r0 * np.cos(th1D)
                ax.plot(xcurve, zcurve, color=grid_color, lw=grid_width)
        elif orientation == 'xy':
            ith = kwargs.get('ith', 0)
            if nr % ds != 0 or nphi % ds != 0:
                raise ValueError("show_grid must divide nr and nphi")
            xshift = r * np.cos(phi - dphi/2.)
            yshift = r * np.sin(phi - dphi/2.)
            for iphi in range(0, nphi, ds):
                xline = xshift[:, ith, iphi]
                yline = yshift[:, ith, iphi]
                ax.plot(xline, yline, color=grid_color, lw=grid_width)
            for ir in range(0, nr, ds):
                r0     = r[ir, 0, 0] - dr/2.
                phi_min = np.amin(phi[0, 0, :]) - dphi/2.
                phi_max = np.amax(phi[0, 0, :]) + dphi/2.
                phi1D   = np.linspace(phi_min, phi_max, 100)
                th0     = theta[0, 0,  ith]
                xcurve = r0 * np.cos(phi1D) 
                ycurve = r0 * np.sin(phi1D)
                ax.plot(xcurve, ycurve, color=grid_color, lw=grid_width)
                
if __name__ == "__main__":
    print("Executing figs.py")