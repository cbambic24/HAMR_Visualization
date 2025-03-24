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

        im['r'] = ax.pcolormesh(X['r'], Y['r'], D['r'],
                                norm = norm, cmap = cmap, shading = 'gouraud')
        cax = self.get_cax(fig, ax, im, FS, **kwargs)
        
        contours = kwargs.get('contours', False)
        if (contours):
            self.plot_contours(ax, im['r'], X['r'], Y['r'], contours)

        return im, cax
    
    def plot_xz(self, fig, ax, X, Z, D, cmap, FS, **kwargs):
        im = {}
        vmin, vmax = self.get_vmin_vmax(D, **kwargs)
        norm       = self.get_norm(vmin, vmax, **kwargs)

        im['l'] = ax.pcolormesh(X['l'], Z['l'], D['l'],
                                norm = norm, cmap = cmap, shading = 'gouraud')
        im['r'] = ax.pcolormesh(X['r'], Z['r'], D['r'],
                                norm = norm, cmap = cmap, shading = 'gouraud')
        cax = self.get_cax(fig, ax, im, FS, **kwargs)
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

        im['l'] = ax.pcolormesh(X['l'], Z['l'], Dl['l'],
                                norm = norm_l, cmap = cmaps[0], 
                                shading = 'gouraud')
        im['r'] = ax.pcolormesh(X['r'], Z['r'], Dr['r'],
                                norm = norm_r, cmap = cmaps[1], 
                                shading = 'gouraud')
        
        cax_l = self.get_cax(fig, ax, im, FS, cbar_loc='top', **kwargs)
        cax_r = self.get_cax(fig, ax, im, FS, **kwargs)

        if (contours_l):
            self.plot_contours(ax, im['l'], X['l'], Z['l'], contours_l)
        if (contours_r):
            self.plot_contours(ax, im['r'], X['r'], Z['r'], contours_r)

        return im, cax_l, cax_r

    def plot_contours(self, ax, im, X, Y, contours):
        for (quantity, level, linewidth, color) in zip(
            contours['quantities'], contours['levels'], 
            contours['linewidths'], contours['colors']):
            ax.contour(X, Y, quantity, levels=[level], 
                       linewidths=linewidth, colors=color)
        return None

    def plot_xy_vec(self, fig, ax, X, Y, 
                    X_uni, Y_uni, VX, VY, **kwargs):
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
    
    def plot_xz_vec(self, fig, ax, X, Z, 
                    X_uni, Z_uni, VX, VZ, **kwargs):
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

    def standard_cbar(self, fig, ax, images, FS, **kwargs):
        divider = make_axes_locatable(ax)
        if (kwargs.get('cbar_loc') == 'top'):
            cax = divider.append_axes("top", size="4%", pad=0.2)
            cax.xaxis.set_ticks_position("top")
            cbar = fig.colorbar(images, cax=cax, location="top")
        else:
            cax = divider.append_axes("right", size="4%", pad=0.2)
            cbar = fig.colorbar(images, cax=cax, location="right")
        #if (kwargs.get('cbar_loc') == 'left'):
        #    cbar.ax.yaxis.set_ticks_position('left')
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
                                         vmin = vmin, vmax = vmax, base=10)
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


if __name__ == "__main__":
    print("Executing figs.py")


