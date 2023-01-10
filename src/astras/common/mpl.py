# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:59:54 2022

@author: bittmans
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import colour
import re
# %%
""" Matplotlib functions and classes """


# plot colors
def set_line_cycle_properties(numlines, reverse_z=False, color_order=None,
                              cycle=None, interpolate=0, **kwargs):
    # a function used to manipulate an existing color cycle,
    # e.g. changing order, interpolating between the colors
    # (interpolate = number of steps).
    # ways to manipulate further can be added easily.
    # if cycle is None, the current cycle from rcParams will be used.
    # returns zorder for the lines (can be reversed) and the cycle
    def interp_cycle(n_lines, cyc_in, n_colors):
        if n_colors > len(cyc_in) - 1:
            n_colors = len(cyc_in) - 1
        colors = [cyc_in[0]]
        for i in range(n_colors):
            c1 = cyc_in[i]
            c2 = cyc_in[i + 1]
            interp_colors = [c.hex for c in colour.Color(
                c1).range_to(colour.Color(c2), n_lines)]
            colors.extend(interp_colors[1:])
        return [colors[i] for i in range(0, n_lines*n_colors, n_colors)]

    if reverse_z:
        zord = [i for i in range(numlines - 1, -1, -1)]
    else:
        zord = [i for i in range(numlines)]
    if cycle is None:
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cyc = [c for c in cycle]
    while len(cyc) < numlines:
        cyc = cyc + cycle
    if interpolate:
        colors = interp_cycle(numlines, cyc, interpolate)
        if colors is not None:
            cyc = colors
    if type(color_order) is dict:
        it = range(numlines)
        try:
            if color_order['reverse']:
                it = range(numlines - 1, -1, -1)
        except Exception:
            pass
        try:
            len(color_order['ind'])
        except Exception:
            try:
                while len(cyc) < numlines + np.abs(color_order['ind']):
                    cyc = cyc + \
                        plt.rcParams['axes.prop_cycle'].by_key()['color']
            except Exception:
                pass
            else:
                cyc = [cyc[color_order['ind'] + i] for i in it]
        else:
            try:
                c = cyc
                while len(cyc) <= np.max(np.abs(color_order['ind'])):
                    cyc.extend(c)
                c = [cyc[i] for i in color_order['ind']]
                cyc = c
            except Exception:
                pass
    return zord, cyc


class PlotColorManager2D():
    # a class to manipulate the colormaps for 2D mpl plots.
    # can be used to set color limits, invert cmaps, change symmetry of the
    # DivergingNorm norm (see mpl.colors.DivergingNorm)
    #
    # Used for TkMplFigure class (main figure class of package)
    def __init__(self, clims=[-10, 10], cmap=plt.get_cmap('RdBu_r'),
                 invert_cmap=True, cmap_sym=True, clim_sym=True):
        self.clims = clims
        self.cmap = cmap
        self.cmap_name = 'Red Blue'
        self.cmap_dict = {'Red Blue': 'RdBu',
                          'Seismic': 'seismic',
                          'Spectral': 'Spectral',
                          'Cool Warm': 'coolwarm',
                          'Twilight': 'twilight_shifted',
                          'Brown Green': 'BrBG',
                          'Red Grey': 'RdGy',
                          'Inferno': 'inferno',
                          'Hot': 'hot',
                          'Terrain': plt.cm.terrain}
        self.invert_cmap = invert_cmap
        self.div_norm = None
        self.map_center_zero = True
        self.cmap_sym = cmap_sym
        self.clim_sym = clim_sym

    def update(self, *args, **kwargs):
        if 'clims' in kwargs.keys():
            self.clims = kwargs['clims']
        if 'cmap_center_zero' in kwargs.keys():
            self.map_center_zero = kwargs['cmap_center_zero']
        if 'invert_cmap' in kwargs.keys():
            self.invert_cmap = kwargs['invert_cmap']
        if 'cmap_sym' in kwargs.keys():
            self.cmap_sym = kwargs['cmap_sym']
        if 'cmap' in kwargs.keys():
            cmap = kwargs['cmap']
        else:
            cmap = None
        if 'clim_sym' in kwargs.keys():
            self.clim_sym = kwargs['clim_sym']
        self.update_cmap(cmap)

    def update_cmap(self, name=None):
        if name is not None:
            self.cmap_name = name
        if re.search('terrain', self.cmap_name, re.I):
            self.cmap = plt.cm.terrain
        else:
            try:
                cm = self.cmap_dict[self.cmap_name]
            except KeyError:
                cm = 'RdBu'
            except Exception:
                raise
            if self.invert_cmap:
                cm += '_r'
            self.cmap = plt.get_cmap(cm)
        self.update_div_norm()

    def update_div_norm(self):
        if sum(self.clims) != 0 and self.map_center_zero:
            # diverging colormap with the center at 0
            # assuming self.clims[0] < self.clims[1]
            # for proper functioning
            try:
                ratio = -self.clims[0]/self.clims[1]
            except Exception:
                ratio = np.inf
            if ratio > 0 and np.isfinite(ratio):
                n_lower = int(np.round(512/(1 + 1/ratio)))
                n_upper = int(np.round(512/(1 + ratio)))
                if n_lower + n_upper != 512:
                    n_upper = 512 - n_lower
                if self.cmap_sym:
                    if np.abs(self.clims[0]) > np.abs(self.clims[1]):
                        neg_vals = self.cmap(np.linspace(0, 0.5, n_lower))
                        pos_vals = self.cmap(np.linspace(
                            0.5, 0.5*(1 + 1/ratio), n_upper))
                    else:
                        neg_vals = self.cmap(np.linspace(
                            0.5*(1 - ratio), 0.5, n_lower))
                        pos_vals = self.cmap(np.linspace(0.5, 1, n_upper))
                else:
                    neg_vals = self.cmap(np.linspace(0, 0.5, n_lower))
                    pos_vals = self.cmap(np.linspace(0.5, 1, n_upper))
                color_stack = np.vstack((neg_vals, pos_vals))
            else:
                if np.abs(self.clims[0]) > np.abs(self.clims[1]):
                    color_stack = self.cmap(np.linspace(
                        0, 0.5*(1 - 1/np.abs(ratio)), 512))
                else:
                    color_stack = self.cmap(np.linspace(
                        0.5*(1 + np.abs(ratio)), 1, 512))
            self.cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'newmap', color_stack)

    def get_kwargs(self):
        if self.div_norm:
            dct = {'cmap': self.cmap, 'norm': self.div_norm}
        else:
            dct = {'vmin': self.clims[0], 'vmax': self.clims[1],
                   'cmap': self.cmap}
        return dct

    def set_auto_clim(self, dat, opt='symmetric', contrast=1, offset=0.0):
        if opt == 'symmetric':
            self.clims = np.round(
                np.array([-contrast, contrast])*np.max(np.max(np.abs(
                    np.ma.array(dat, mask=np.isnan(dat)))))*(1+offset))
        elif opt == 'asymmetric':
            self.clims = np.round(
                [np.min(np.min(np.ma.array(dat, mask=np.isnan(dat))))
                 * (1+offset) * contrast,
                 np.max(np.max(np.ma.array(dat, mask=np.isnan(dat))))
                 * (1+offset) * contrast])
        self.update_div_norm()
        return self.clims


# Normalization class for diverging colormaps
class MidpointNormalizeCmap(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, symmetric=True,
                 clip=False):
        self.vcenter = vcenter
        self.symmetric = symmetric
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if self.symmetric:
            v_ext = np.max([np.abs(self.vmin), np.abs(self.vmax)])
            x, y = [-v_ext, self.vcenter, v_ext], [0, 0.5, 1]
        else:
            x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


class MidpointNormalizeFair(mpl.colors.Normalize):
    """ From: https://matplotlib.org/users/colormapnorms.html"""

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        print('hi')
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)

        vlargest = max(abs(self.vmax - self.midpoint),
                       abs(self.vmin - self.midpoint))
        x, y = [self.midpoint - vlargest, self.midpoint,
                self.midpoint + vlargest], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# helper and/or wrapper functions for plotting with matplotlib


def get_fill_list(fill, numlines):
    if type(fill) is list or type(fill) is np.ndarray:
        fill_list = [0 for i in range(numlines)]
        for i, f in enumerate(fill):
            try:
                fill_list[i] = f
            except Exception:
                break
    else:
        fill_list = [fill for i in range(numlines)]
    return fill_list


def plot_1d(func, x, y, *plot_args, transpose=False, **plot_kwargs):
    if transpose:
        _x = x
        x = y
        y = _x
    return func(x, y, *plot_args, **plot_kwargs)


def plot(ax, *plot_args, **plot_kwargs):
    return plot_1d(ax.plot, *plot_args,  **plot_kwargs)


def fill_between(ax, *plot_args, **plot_kwargs):
    return plot_1d(ax.fill_between, *plot_args, **plot_kwargs)


def step_plot(ax, *args, **kwargs):
    return plot_1d(ax.step, *args, **kwargs)


def text_in_plot(ax, *args, **kwargs):
    return plot_1d(ax.text, *args, **kwargs)


def plot_2d(func, x, y, z, *plot_args, transpose=False, **plot_kwargs):
    if transpose:
        _x = x
        x = y
        y = _x
        z = np.transpose(z)
    return func(x, y, z, *plot_args, **plot_kwargs)


def pcolor(ax, x, y, z, *args, **kwargs):
    return plot_2d(ax.pcolormesh, x, y, z, *args, **kwargs)

