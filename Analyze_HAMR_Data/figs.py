import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

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

        vmin_r, vmax_r = self.get_vmin_vmax(Dr,
            lim=kwargs.get('lim_right', False), 
            sym=kwargs.get('sym_right', False))
        norm_r         = self.get_norm(vmin_r, vmax_r,
            log=kwargs.get('log_right', False), 
            sym=kwargs.get('sym_right', False),
            linthresh=kwargs.get('linthresh_right', False))
        contours_r     = kwargs.get('contours_right', False)

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
        
        cax_r = self.get_cax(fig, ax, im, FS, **kwargs)
        cax_l = self.get_cax(fig, ax, im, FS, cbar_loc='top', **kwargs)

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
            ax.contour(X, Y, quantity, levels=[level], 
                linewidths=linewidth, linestyles=linestyle, colors=color)
        return None

    def plot_xy_vec(self, fig, ax, X, Y, X_uni, Y_uni, VX, VY, **kwargs):
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
                          (X_uni, Y_uni), method='cubic')
        VY_uni = griddata((x_data, y_data), VY_data, 
                          (X_uni, Y_uni), method='cubic')

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
                                   (X_uni, Y_uni), method='cubic')
        
        dens = kwargs.get('dens', 1.0)

        ax.streamplot(X_uni, Y_uni, VX_uni, VY_uni, 
                      density = (dens, dens),
                      broken_streamlines = (
                          kwargs.get('broken_streamlines', True)),
                      linewidth = kwargs.get('linewidth', 1.0), 
                      color = (kwargs.get('color', 'white') 
                        if 'color_streamlines' not in kwargs else values_uni), 
                      cmap = (
                          None if 'color_streamlines' not in kwargs else cmap),
                      arrowsize = kwargs.get('arrowsize', 1.0))
        
    def plot_Rz_vec(self, fig, ax, R, Z, R_uni, Z_uni, VR, VZ, **kwargs):
        return self.plot_xy_vec(fig, ax, R, Z, R_uni, Z_uni, VR, VZ, **kwargs)
    
    def plot_xz_vec(self, fig, ax, X, Z, X_uni, Z_uni, VX, VZ, **kwargs):
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
                          (X_uni, Z_uni), method='cubic')
        VZ_uni = griddata((x_data, z_data), VZ_data, 
                          (X_uni, Z_uni), method='cubic')

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
                                   (X_uni, Z_uni), method='cubic')

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
        
    def plot_xz_mirror_vec(self, fig, ax, X, Z, 
                           X_uni, Z_uni, VX, VZ, **kwargs): 
        
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
                          (X_uni, Z_uni), method='cubic')
        VZ_uni = griddata((x_data, z_data), VZ_data, 
                          (X_uni, Z_uni), method='cubic')

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
                                   (X_uni, Z_uni), method='cubic')
        
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
                return np.amin(DD), np.amax(DD)

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
        th_strt, th_end  = 0., np.pi
        dat_strt, dat_end = D['r'][:, 0], D['r'][:, -1]
        x_strt  = r1D * np.sin(th_strt)
        x_end   = r1D * np.sin(th_end)
        z_strt  = r1D * np.cos(th_strt)
        z_end   = r1D * np.cos(th_end) 

        X['r']  = self.concatenate(X['r'], x_strt, x_end)
        Z['r']  = self.concatenate(Z['r'], z_strt, z_end)

        if kwargs.get('fix_pole', 'constant') == 'constant':
            D['r'] = self.concatenate(D['r'], dat_strt, dat_end)
        elif kwargs.get('fix_pole', 'constant') == 'average':
            print("Warning: you're averaging two separate poles.")
            dat_avg = 0.5 * (dat_strt + dat_end)
            D['r'] = self.concatenate(D['r'], dat_avg, dat_avg)  

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
        theta_min, theta_max  = np.amin(th1D), np.amax(th1D)
        dth     = np.diff(th1D)[-1]
        th_strt = (theta_min / 2.) * np.ones(len(r1D))
        th_end  = (np.pi - dth/2) * np.ones(len(r1D))
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
        return self.remove_poles_xy(R, Z, D)

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


if __name__ == "__main__":
    print("Executing figs.py")


