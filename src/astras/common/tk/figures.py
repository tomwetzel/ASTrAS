# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:49:39 2020

@author: bittmans

"""

import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import copy
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import matplotlib.animation as animation

from ..helpers import GlobalSettings
from ..dataobjects import TATrace
from ..mpl import (
    PlotColorManager2D, set_line_cycle_properties, get_fill_list, plot,
    fill_between)
from ..helpers import BlankObject, idle_function
from .general import (
    CustomFrame, move_toplevel_to_default, center_toplevel,
    save_box, load_box, GroupBox)
from .traces import TraceSelection


# %%
def tk_mpl_plot_function(x, y, *args, fig=None, ax=None, fill=0,
                         include_fit=False, fit_x=None, fit_y=None,
                         fit_kwargs=None, show_legend=False, reverse_z=False,
                         color_order=None, color_cycle=None,
                         interpolate_colors=False, cla=True,
                         min_cycle_len=None, fit_comp_alpha=0.15,
                         show_fit_legend=False, transpose=False,
                         function_extension=None, ymode=None, **plot_kwargs):
    def idlefunc(i, j, *args, **kwargs):
        return i, []

    if ax is None:
        return
    if function_extension is None:
        function_extension = idlefunc
    numlines = np.shape(y)[0]
    if np.shape(x) != np.shape(y):
        x = [x for i in range(numlines)]
    fill = get_fill_list(fill, numlines)
    if include_fit:
        if not (fit_x is None or fit_y is None):
            if fit_kwargs is None:
                fit_kwargs = {'color': 'black'}
            if not show_fit_legend:
                fit_kwargs['label'] = '_nolegend_'
            if len(np.shape(fit_y)) < 2:
                fit_y = [fit_y]
            if np.shape(fit_x) != np.shape(fit_y):
                fit_x = [fit_x for i in range(numlines)]
        else:
            include_fit = False
    if cla:
        ax.cla()
    if not min_cycle_len:
        min_cycle_len = numlines * 2
    zord, cycle = set_line_cycle_properties(
        min_cycle_len, reverse_z, color_order, cycle=color_cycle,
        interpolate=interpolate_colors)
    lines = []
    i = 0
    for j in range(numlines):
        x_line = x[j]
        y_line = y[j]
        lines.append(plot(ax, x_line, y_line, transpose=transpose,
                          zorder=zord[i], color=cycle[i], **plot_kwargs))
        if fill[i]:
            lines.append(fill_between(ax, x_line, y_line,
                                      zorder=zord[i],
                                      color=cycle[i],
                                      alpha=fill[i],
                                      transpose=transpose,
                                      label='_nolegend_'))
        if include_fit:
            try:
                lines.append(
                    plot(ax, fit_x[j], fit_y[j], transpose=transpose,
                         **fit_kwargs))
            except Exception:
                pass
        i += 1
        i, line = function_extension(i, j, ax, zord, cycle,
                                     fit_comp_alpha=fit_comp_alpha,
                                     fig=fig, transpose=transpose)
        lines.extend(line)
    return lines


class FigureEditor(tk.Tk, tk.Toplevel):
    def __init__(self, *args, config_filepath='config.txt', parent=None, **kwargs):
        if parent is None:
            tk.Tk.__init__(self, *args, **kwargs)
        else:
            tk.Toplevel.__init__(self, parent, *args, **kwargs)
        settings = GlobalSettings(config_path=config_filepath)
        kw = {}
        kw['dim'] = settings['figure_std_size']
        for key in ('fit_kwargs', 'plot_style', 'ticklabel_format'):
            kw[key] = settings[key]
        self.fr = FigFrame(self, controller=self, editable=True, **kw)
        self.fr.grid(row=0, column=0)
        self.menubar = tk.Menu(self)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Figure", menu=self.filemenu)
        self.filemenu.add_command(label="Load",
                                  command=self.fr.open_fig_obj)
        self.filemenu.add_command(label="Save",
                                  command=self.fr.save_figure_obj)
        self.title("Figure Editor")
        tk.Tk.config(self, menu=self.menubar)


# %%
class TkMplFigure(tk.Frame):
    def __init__(self, parent=None, color_obj=None, dim=None, row=0,
                 tight_layout=True, toolbar=True, callbacks=None,
                 plot_type='2D', plot_function=None, plot_kwargs=None,
                 grid_kwargs=None, legend_kwargs=None, num_subplots=1,
                 fontsize=None, num_rows=1, num_columns=1, xlimits=None,
                 ylimits=None, plotstyle=None, fit_kwargs=None,
                 xlabel_pos=None, ylabel_pos=None, axes_grid_on=None,
                 axes_frame_on=True, invert_yaxis=None, transpose=False,
                 texts=None, text_kwargs=None, **kwargs):
        self.style_dict = {'Fast': {'name': 'fast',
                                    'grid': False},
                           'Classic': {'name': 'classic',
                                       'grid': False},
                           'bmh': {'name': 'bmh',
                                   'grid': True},
                           'Dark': {'name': 'dark_background',
                                    'grid': False},
                           '538': {'name': 'fivethirtyeight',
                                   'grid': True},
                           'GG Plot': {'name': 'ggplot',
                                       'grid': True},
                           'Tableau Colorblind': {
                               'name': 'tableau-colorblind10',
                               'grid': True},
                           'Solarize': {'name': 'Solarize_Light2',
                                        'grid': True},
                           'Seaborn': {'name': 'seaborn',
                                       'grid': True}}
        seaborn_substyles = ['colorblind', 'white', 'paper', 'dark-palette',
                             'darkgrid', 'deep', 'pastel']
        for sbst in seaborn_substyles:
            self.style_dict['-'.join(['Seaborn', sbst.title()])] = {
                'name': '-'.join(['seaborn', sbst]),
                'grid': bool(re.search('grid', sbst, re.I))}

        self.plotstyle = 'fast' if plotstyle is None else plotstyle

        self.parent = parent

        self.parent_row = row

        self.twiny_conversion_func = lambda x: [1/val for val in x]
        self.twinx_conversion_func = lambda y: [1/val for val in y]
        self.axes = []
        self.cbar = []
        self.images = []
        self._num_subplots = num_subplots
        # numerus insensitive keywords
        for key in ('xlabel', 'ylabel', 'clabel', 'legend', 'axes_title'):
            if not re.search(key, "".join(kwargs.keys())):
                kwargs[key] = None

        if invert_yaxis is None:
            invert_yaxis = plot_type == '2D'
        if axes_grid_on is None:
            axes_grid_on = mpl.rcParams['axes.grid']
        if fontsize is None:
            fontsize = mpl.rcParams['font.size']
        self._set_subplot_attributes(
            num_subplots, num_rows=num_rows, num_columns=num_columns,
            plot_type=plot_type, plot_kwargs=plot_kwargs,
            grid_kwargs=grid_kwargs, legend_kwargs=legend_kwargs,
            axes_grid_on=axes_grid_on, axes_frame_on=axes_frame_on,
            transpose=transpose, color_obj=color_obj, xlimits=xlimits,
            ylimits=ylimits, invert_yaxis=invert_yaxis, fit_kwargs=fit_kwargs,
            fontsize=fontsize, texts=texts, text_kwargs=text_kwargs, **kwargs)

        self.tight_layout = tight_layout
        self.dim = dim
        self.show_toolbar = toolbar
        self.canvas_callbacks = callbacks
        self.init_figure()
        self.current_axes_no = 0

        if plot_function is None:
            self.plot_function = idle_function
        else:
            self.plot_function = plot_function

        if xlabel_pos is not None:
            self.set_for_all_axes(self.set_label_pos, *xlabel_pos, case='x',
                                  update_canvas=False)
        if ylabel_pos is not None:
            self.set_for_all_axes(self.set_label_pos, *ylabel_pos, case='y',
                                  update_canvas=False)

        self.set_axes_lim_all(update_canvas=False)
        self.set_ticklabel_format_all(update_canvas=False)
        self.set_fontsize(update_canvas=False)
        if parent is None:
            self.canvas = BlankObject()
            self.canvas.draw = idle_function
        else:
            self.init_tk_widgets()

    def set_num_subplots(self, num_subplots, num_rows=None, num_columns=None,
                         **kwargs):
        del self.images
        del self.cbar
        for i, ax in enumerate(self.axes):
            self.figure.delaxes(ax)
        try:
            self._set_subplot_attributes(num_subplots, num_rows=num_rows,
                                         num_columns=num_columns, **kwargs)
        except Exception:
            raise
        finally:
            self.init_figure(new_figure=False)

    def _set_subplot_attributes(self, num_subplots, num_rows=None,
                                num_columns=None, **kwargs):
        self._num_subplots = num_subplots
        if not num_rows:
            if not num_columns:
                num_columns = int(np.sqrt(self._num_subplots))
            num_rows = num_columns

        while num_rows * num_columns < self._num_subplots:
            if num_rows > num_columns:
                num_columns += 1
            else:
                num_rows += 1

        self.num_rows = num_rows
        self.num_columns = num_columns
        if self._num_subplots > 1:
            if (num_rows < 1 or num_columns < 1 or
                    (num_rows == 1 and num_columns == 1)):
                self.num_columns = (3 if self._num_subplots > 2
                                    else self._num_subplots)
                self.num_rows = int(self._num_subplots/self.num_columns) + int(
                    np.mod(self._num_subplots, self.num_columns) > 0)

        try:
            plot_type = kwargs['plot_type']
        except Exception:
            pass
        else:
            if not type(plot_type) is list:
                self.plot_type = [plot_type for i in range(self._num_subplots)]
            elif len(plot_type) != self._num_subplots:
                self.plot_type = [plot_type[0]
                                  for i in range(self._num_subplots)]
            else:
                self.plot_type = plot_type

        try:
            plot_kwargs = kwargs['plot_kwargs']
        except Exception:
            pass
        else:
            if plot_kwargs is None:
                self.plot_kwargs = [{} for i in range(self._num_subplots)]
            else:
                if type(plot_kwargs) is list:
                    self.plot_kwargs = plot_kwargs
                else:
                    self.plot_kwargs = [
                        plot_kwargs for i in range(self._num_subplots)]

        try:
            fit_kwargs = kwargs['fit_kwargs']
        except Exception:
            pass
        else:
            if fit_kwargs is None:
                self.fit_kwargs = [{} for i in range(self._num_subplots)]
            else:
                if type(fit_kwargs) is list:
                    self.fit_kwargs = fit_kwargs
                else:
                    self.fit_kwargs = [
                        fit_kwargs for i in range(self._num_subplots)]

        try:
            grid_kwargs = kwargs['grid_kwargs']
        except Exception:
            pass
        else:
            if grid_kwargs is None:
                self.grid_kwargs = [{} for i in range(self._num_subplots)]
            else:
                self.grid_kwargs = [
                    grid_kwargs for i in range(self._num_subplots)]
        try:
            legend_kwargs = kwargs['legend_kwargs']
        except Exception:
            pass
        else:
            if legend_kwargs is None:
                self.legend_kwargs = [{"fontsize":
                                       mpl.rcParams["legend.fontsize"]}
                                      for i in range(self._num_subplots)]
            else:
                self.legend_kwargs = [
                    legend_kwargs for i in range(self._num_subplots)]

        try:
            axes_grid_on = kwargs['axes_grid_on']
        except Exception:
            pass
        else:
            if type(axes_grid_on) is bool or type(axes_grid_on) is int:
                self.axes_grid_on = [
                    axes_grid_on for i in range(self._num_subplots)]
            else:
                self.axes_grid_on = axes_grid_on

        try:
            invert_yaxis = kwargs['invert_yaxis']
        except Exception:
            pass
        else:
            if type(invert_yaxis) is list:
                self.invert_yaxis = invert_yaxis
            else:
                self.invert_yaxis = [
                    invert_yaxis for i in range(self._num_subplots)]

        try:
            axes_frame_on = kwargs['axes_frame_on']
        except Exception:
            pass
        else:
            if type(axes_frame_on) is bool or type(axes_frame_on) is int:
                self.axes_frame_on = [
                    axes_frame_on for i in range(self._num_subplots)]
            else:
                self.axes_frame_on = axes_frame_on

        try:
            color_obj = kwargs['color_obj']
        except Exception:
            pass
        else:
            if color_obj is not None:
                if type(color_obj) is list:
                    self.color_obj = color_obj
                else:
                    self.color_obj = [
                        color_obj for i in range(self._num_subplots)]
            else:
                self.color_obj = []
                for i in range(self._num_subplots):
                    self.color_obj.append(PlotColorManager2D())

        try:
            xlimits = kwargs['xlimits']
        except Exception:
            pass
        else:
            if xlimits is None:
                self.xlimits = [None for i in range(self._num_subplots)]
            else:
                if type(np.shape(xlimits)) is int:
                    self.xlimits = [xlimits for i in range(self._num_subplots)]
                elif len(np.shape(xlimits)) == 1:
                    self.xlimits = [xlimits for i in range(self._num_subplots)]
                else:
                    self.xlimits = xlimits

        try:
            ylimits = kwargs['ylimits']
        except Exception:
            pass
        else:
            if ylimits is None:
                self.ylimits = [None for i in range(self._num_subplots)]
            else:
                if type(np.shape(ylimits)) is int:
                    self.ylimits = [ylimits for i in range(self._num_subplots)]
                elif len(np.shape(ylimits)) == 1:
                    self.ylimits = [ylimits for i in range(self._num_subplots)]
                else:
                    self.ylimits = ylimits

        try:
            try:
                ylabels = kwargs['ylabels']
            except Exception:
                try:
                    ylabels = kwargs['ylabel']
                except Exception:
                    raise
        except Exception:
            pass
        else:
            if ylabels is None:
                self.ylabels = ["" for i in range(self._num_subplots)]
            elif not type(ylabels) is list:
                self.ylabels = [ylabels for i in range(self._num_subplots)]
            else:
                self.ylabels = ylabels

        try:
            try:
                xlabels = kwargs['xlabels']
            except Exception:
                try:
                    xlabels = kwargs['xlabel']
                except Exception:
                    raise
        except Exception:
            pass
        else:
            if xlabels is None:
                self.xlabels = ["" for i in range(self._num_subplots)]
            elif not type(xlabels) is list:
                self.xlabels = [xlabels for i in range(self._num_subplots)]
            else:
                self.xlabels = xlabels

        try:
            try:
                clabels = kwargs['clabels']
            except Exception:
                try:
                    clabels = kwargs['clabel']
                except Exception:
                    raise
        except Exception:
            pass
        else:
            if clabels is None:
                self.clabels = ["" for i in range(self._num_subplots)]
            elif not type(clabels) is list:
                self.clabels = [clabels for i in range(self._num_subplots)]
            else:
                self.clabels = clabels

        try:
            try:
                axes_titles = kwargs['axes_titles']
            except Exception:
                try:
                    axes_titles = kwargs['axes_title']
                except Exception:
                    axes_titles = None
        except Exception:
            pass
        else:
            if axes_titles is None:
                self.axes_titles = ["" for i in range(self._num_subplots)]
            elif not type(axes_titles) is list:
                self.axes_titles = [
                    axes_titles for i in range(self._num_subplots)]
            else:
                self.axes_titles = axes_titles

        try:
            fontsize = kwargs['fontsize']
        except Exception:
            pass
        else:
            self.fontsize = [fontsize for i in range(self._num_subplots)]
            self.cbar_fontsize = [fontsize -
                                  2 for i in range(self._num_subplots)]
            self.ticklabelsize = [mpl.rcParams["xtick.labelsize"]
                                  for i in range(self._num_subplots)]

        try:
            try:
                legends = kwargs['legends']
            except Exception:
                try:
                    legends = kwargs['legend']
                except Exception:
                    raise
        except Exception:
            pass
        else:
            if legends is None:
                self.legends = ["" for i in range(self._num_subplots)]
            else:
                self.legends = legends
            if legends is False:
                self.legends = [False for i in range(self._num_subplots)]
            self.legend_visible = [True for i in range(self._num_subplots)]
            for i, pt in enumerate(self.plot_type):
                if re.search('2D', pt, re.I):
                    self.legends[i] = False
                    self.legend_visible[i] = False

        try:
            transpose = kwargs['transpose']
        except Exception:
            pass
        else:
            if type(transpose) is list:
                self.transpose = transpose
            else:
                self.transpose = [transpose for i in range(self._num_subplots)]

        if 'ticklabel_format' in kwargs.keys():
            tlf = kwargs['ticklabel_format']
            if type(tlf) is list:
                self.ticklabel_format = tlf
            else:
                self.ticklabel_format = [
                    tlf for i in range(self._num_subplots)]
            for tlf in self.ticklabel_format:
                t = {}
                if ('x' not in tlf.keys()) and ('y' not in tlf.keys()):
                    for key, val in tlf.items():
                        t[key] = val
                    for ax in ('x', 'y'):
                        tlf[ax] = t
                else:
                    for ax in ('x', 'y'):
                        if ax in tlf.keys():
                            t[ax] = tlf
                        else:
                            t[ax] = {}
                    tlf = t
        else:
            tlf = {'x': {}, 'y': {}}
            self.ticklabel_format = [tlf for i in range(self._num_subplots)]

        self.line_fill = [{} for i in range(self._num_subplots)]

        try:
            try:
                texts = kwargs['texts']
            except Exception:
                texts = kwargs['text']
        except Exception:
            pass
        else:
            if type(texts) is list:
                self.texts = texts
            else:
                self.texts = [texts for i in range(self._num_subplots)]

        try:
            text_kwargs = kwargs['text_kwargs']
        except Exception:
            pass
        else:
            if type(text_kwargs) is list:
                self.text_kwargs = text_kwargs
            else:
                self.text_kwargs = [
                    text_kwargs for i in range(self._num_subplots)]

        self.tick_params = [{} for i in range(self._num_subplots)]

    def copy_attributes(self, fig_obj, exclude=[], deep=True):
        def copy_attr(*args):
            return copy.deepcopy(getattr(*args))
        if deep:
            copy_func = copy_attr
        else:
            copy_func = getattr
        for attr in ['plot_kwargs', 'grid_kwargs', 'legend_kwargs', 'num_rows',
                     'num_columns', 'axes_titles', '_num_subplots',
                     'tight_layout', 'dim', 'show_toolbar', 'current_axes_no',
                     'plotstyle', 'axes_frame_on', 'axes_grid_on', 'xlabels',
                     'ylabels', 'xlimits', 'ylimits', 'clabels', 'fit_kwargs',
                     'color_obj', 'legends', 'plot_type', 'invert_yaxis']:
            if attr not in exclude:
                try:
                    val = copy_func(fig_obj, attr)
                    setattr(self, attr, val)
                except Exception:
                    pass
        for attr in ['canvas_callbacks', 'plot_function']:
            if attr not in exclude:
                try:
                    val = getattr(fig_obj, attr)
                    setattr(self, attr, val)
                except Exception:
                    pass

    def copy_color_settings(self, color_object, exclude=[]):
        def copy_attr(copy_to, copy_from):
            for attr in ['clims', 'invert_cmap', 'cmap_sym', 'cmap',
                         'cmap_name', 'div_norm', 'cmap_dict']:
                if attr not in exclude:
                    try:
                        exec('copy_to.' + attr + ' = copy_from.' + attr)
                    except Exception as e:
                        print('Unable to copy color settings attribute ' +
                              attr + ':\n' + str(e))

        if type(color_object) is list:
            for i in range(len(self.color_obj)):
                copy_attr(self.color_obj[i], color_object[i])
        else:
            for obj in self.color_obj:
                copy_attr(obj, color_object)

    def init_figure(self, new_figure=True):
        if new_figure:
            self.figure = plt.figure()
            plt.close()
        self.axes = []
        self.cbar = []
        self.images = []
        for i in range(self._num_subplots):
            self.axes.append(self.figure.add_subplot(
                self.num_rows, self.num_columns, i + 1))
            self.cbar.append(None)
            self.images.append(None)
        self.figure.set_tight_layout(self.tight_layout)

    def load_figure(self, filepath):
        # loaded properties
        with open(filepath, 'rb') as f:
            objects = pickle.load(f)
            f.close()
        self.figure = objects[0]
        try:
            if type(objects[1]) is dict:
                attributes = objects[1]
            else:
                raise
        except Exception:
            attributes = {}
        self.axes = self.figure.axes
        self._num_subplots = len(self.axes)
        plot_types = [set() for i in range(self._num_subplots)]
        self.images = [None for i in range(self._num_subplots)]
        for i, ax in enumerate(self.axes):
            for child in ax.get_children():
                if type(child) is mpl.collections.QuadMesh:
                    plot_types[i].add("2D")
                    self.images[i] = child
                elif type(child) is mpl.lines.Line2D:
                    plot_types[i].add("line")
                if "2D" in plot_types[i] and "line" in plot_types[i]:
                    break
        self.plot_type = ["_".join(t) for t in plot_types]
        remove_ax = []
        self.cbar = []
        i = 0
        while i < len(self.images):
            im = self.images[i]
            try:
                im.colorbar.remove()
            except Exception:
                self.cbar.append(None)
                i += 1
            else:
                self.cbar.append(self.figure.colorbar(im, ax=self.axes[i]))
                remove_ax.append(i + 1)
                i += 2
        for i in remove_ax:
            self.axes.pop(i)
            self.images.pop(i)
            self.plot_type.pop(i)
        self._num_subplots = len(self.axes)
        self.invert_yaxis = [False for i in range(self._num_subplots)]
        self.transpose = [False for i in range(self._num_subplots)]
        self.ticklabelsize = [mpl.rcParams["xtick.labelsize"]
                              for i in range(self._num_subplots)]
        self.tick_params = [{} for i in range(self._num_subplots)]
        # get properties from loaded axes where possible
        self.xlabels = []
        self.ylabels = []
        self.axes_titles = []
        self.xlimits = []
        self.ylimits = []
        for ax in self.axes:
            self.xlabels.append(ax.get_xlabel())
            self.ylabels.append(ax.get_ylabel())
            self.axes_titles.append(ax.get_title())
            self.xlimits.append(ax.get_xlim())
            self.ylimits.append(ax.get_ylim()[::-1])
        # get remaining properties from current values
        for attr in ('axes_grid_on', 'axes_frame_on', 'grid_kwargs'):
            setattr(self, attr, [getattr(self, attr)[0]
                    for i in range(self._num_subplots)])
        # overwrite default values from file:
        for key, attr in attributes.items():
            setattr(self, key, attr)
        # (re-)initialize tk widgets
        canvas = self.canvas
        tb = self.toolbar
        self.init_tk_widgets()
        self._remove_widgets(canvas=canvas, toolbar=tb)
        self.grid()
        self.plot_function = idle_function

    def save_figure(self):
        file = save_box(title="Save Figure", fname="figure.pkl",
                        fext=".pkl",
                        filetypes=[("PKL", ".pkl"), ("PNG", ".png"),
                                   ("SVG", ".svg"), ("JPG", ".jpg"),
                                   ("PDF", ".pdf")],
                        parent=self.parent)
        try:
            filepath = file.name
        except Exception:
            return
        else:
            if filepath.endswith('pkl'):
                attributes = {}
                for attr in ('axes_grid_on', 'axes_frame_on', 'grid_kwargs'):
                    attributes[attr] = getattr(self, attr)
                with open(filepath, 'wb') as f:
                    pickle.dump([self.figure, attributes], f)
                    f.close()
            elif filepath.endswith('png') or filepath.endswith('jpg'):
                current_size = self.figure.get_size_inches()
                dpi = 600
                canvas_dim = [self.canvas.get_tk_widget().winfo_width(),
                              self.canvas.get_tk_widget().winfo_height()]
                canv_dim_in = [c/dpi for c in canvas_dim]
                scaling = [current_size[0]/canv_dim_in[0],
                           current_size[1]/canv_dim_in[1]]
                pad = 0.075
                new_size = [current_size[0]*scaling[0],
                            current_size[1]*scaling[1]]
                self.figure.set_size_inches(*[n * (1-pad) for n in new_size])
                self.figure.savefig(filepath, bbox_inches='tight',
                                    pad_inches=max([n * pad for n in new_size]),
                                    dpi=dpi)
                self.figure.set_size_inches(*current_size)
            else:
                self.figure.savefig(filepath)

    def init_tk_widgets(self):
        if self.parent is not None:
            self.canvas = FigureCanvasTkAgg(self.figure, self.parent)
            if self.dim is not None:
                self.canvas.get_tk_widget().config(
                    width=self.dim[0], height=self.dim[1])
            self.canvas.draw()
            if self.show_toolbar:
                self.toolbar_frame = tk.Frame(self.parent)
                self.toolbar = NavigationToolbar2Tk(
                    self.canvas, self.toolbar_frame)
                self.toolbar.update()
                self.toolbar.children['!button5'].config(
                    command=self.save_figure)
            else:
                self.toolbar = False
            if not type(self.canvas_callbacks) is dict:
                self.set_callback(self.canvas_callbacks, 'button_press_event')
            else:
                if 'button_press_event' not in self.canvas_callbacks.keys():
                    self.set_callback(None, 'button_press_event')
                for event, callback in self.canvas_callbacks.items():
                    self.set_callback(callback, event)

    def grid(self, row=None, sticky='wnse', rowspan=1, **kwargs):
        if row is None:
            row = self.parent_row
        else:
            self.parent_row = row
        self.canvas.get_tk_widget().grid(sticky=sticky)
        self.canvas._tkcanvas.grid(row=row, rowspan=rowspan, **kwargs)
        if self.toolbar:
            self.toolbar_frame.grid(row=row + rowspan, sticky='w', **kwargs)

    def _remove_widgets(self, canvas=None, toolbar=None):
        if canvas is None:
            canvas = self.canvas
        if toolbar is None:
            toolbar = self.toolbar
        canvas.get_tk_widget().destroy()
        toolbar.grid_forget()

    def set_style(self, style, update_canvas=True):
        if style in self.style_dict.keys():
            self.plotstyle = self.style_dict[style]['name']
            self.axes_grid_on = [self.style_dict[style]['grid']
                                 for i in range(self._num_subplots)]
            canvas = self.canvas
            tb = self.toolbar
            with plt.style.context(self.plotstyle):
                self.init_figure()
            self.init_tk_widgets()
            self._remove_widgets(canvas=canvas, toolbar=tb)
            self.grid()
            self.plot_all()

    def set_grid(self, i=None, grid_on=None, frame_on=None, update_canvas=True,
                 **grid_kwargs):
        if i is None:
            i = self.current_axes_no
        if grid_on is not None:
            self.axes_grid_on[i] = grid_on
        if frame_on is not None:
            self.set_frame(frame_on, i=i, update_canvas=False)
        for key, val in grid_kwargs.items():
            self.grid_kwargs[i][key] = val
        if self.axes_grid_on[i]:
            self.axes[i].grid(True, **self.grid_kwargs[i])
        else:
            self.axes[i].grid(False)
        if update_canvas:
            self.canvas.draw()

    def set_frame(self, frame_on=None, i=None, update_canvas=True):
        if i is None:
            i = self.current_axes_no
        if frame_on is None:
            frame_on = self.axes_frame_on[i]
        else:
            self.axes_frame_on[i] = frame_on
        self.axes[i].set_frame_on(frame_on)
        if update_canvas:
            self.canvas.draw()

    def set_grid_all(self, **kwargs):
        for i in range(self._num_subplots):
            self.set_grid(i, **kwargs, update_canvas=False)
        self.plot_all()

    def set_axes_aspect_ratio(self, ratio='equal', i=None,
                              adjustable='box', update_canvas=False,
                              **kwargs):
        if i is None:
            i = self.current_axes_no
        self.axes[i].set_aspect(ratio, adjustable=adjustable)
        if update_canvas:
            self.canvas.draw()

    def set_axes_lim_all(self, **kwargs):
        self.set_for_all_axes(self.set_axes_lim, **kwargs)

    def set_xlim(self, xlimits, i=None, **kwargs):
        self.set_axes_lim(x=xlimits, i=i, **kwargs)

    def set_ylim(self, ylimits, i=None, **kwargs):
        self.set_axes_lim(y=ylimits, i=i, **kwargs)

    def set_axes_lim(self, i=None, x=None, y=None, update_canvas=True):
        if i is None:
            i = self.current_axes_no
        if self.transpose[i]:
            x_func = self.axes[i].set_ylim
            y_func = self.axes[i].set_xlim
            xinvert = self.invert_yaxis[i]
            yinvert = False
        else:
            x_func = self.axes[i].set_xlim
            y_func = self.axes[i].set_ylim
            yinvert = self.invert_yaxis[i]
            xinvert = False
        if x is not None:
            if x is False:
                self.xlimits[i] = None
            else:
                if xinvert:
                    xlim = x[::-1]
                else:
                    xlim = x
                x_func(xlim)
                self.xlimits[i] = x
        if y is not None:
            if y is False:
                self.ylimits[i] = None
            else:
                if yinvert:
                    ylim = y[::-1]
                else:
                    ylim = y
                y_func(ylim)
                self.ylimits[i] = y
        if update_canvas:
            self.canvas.draw()

    def set_invert_yaxis(self, invert, i=None, update_canvas=True):
        if i is None:
            i = self.current_axes_no
        self.invert_yaxis[i] = invert
        self.set_ylim(self.ylimits[i], i=i, update_canvas=update_canvas)

    def get_axes(self, case='current'):
        if case == 'current':
            return self.axes[self.current_axes_no]
        elif case == 'all':
            return self.axes
        elif type(case) is int:
            return self.axes[case]

    def plot_text(self, i, text=None, update_canvas=True, text_kwargs=None):
        if text is None:
            text = self.texts[i]
        if text_kwargs is None:
            text_kwargs = self.text_kwargs[i]
            if text_kwargs is None:
                text_kwargs = {'x': 0.1, 'y': 0.9}
        if self.transpose[i]:
            x = text_kwargs['y']
            y = text_kwargs['x']
        else:
            x = text_kwargs['x']
            y = text_kwargs['y']
        kwargs = {}
        for k in text_kwargs.keys():
            if k in ('x', 'y'):
                continue
            kwargs[k] = text_kwargs[k]
        self.axes[i].text(x, y, text, **kwargs,
                          transform=self.axes[i].transAxes)
        if update_canvas:
            self.canvas.draw()

    def plot_all(self, *args, update_canvas=True, **kwargs):
        for i, ax in enumerate(self.axes):
            self._plot(i, *args, **kwargs)
        if update_canvas:
            self.canvas.draw()

    def plot_simple(self, i, *args, update_canvas=True, **kwargs):
        if 'plot_kwargs' in kwargs.keys():
            self.plot_kwargs[i] = kwargs['plot_kwargs']
        if self.plot_type[i] == '2D':
            color_kwargs = self.color_obj[i].get_kwargs()
        else:
            color_kwargs = {}
        with plt.style.context(self.plotstyle):
            try:
                self.images[i] = self.plot_function(
                    i, *args, ax=self.axes[i], fig=self.figure,
                    transpose=self.transpose[i], **self.plot_kwargs[i],
                    **color_kwargs)
            except (AttributeError, TypeError):
                self.images[i] = self.plot_function(
                    i, *args, ax=self.axes[i], fig=self.figure,
                    transpose=self.transpose[i])
                self.plot_kwargs[i] = {}
                raise
            except Exception:
                raise
            if not self.texts[i] is None:
                self.plot_text(i, update_canvas=False)
        if update_canvas:
            self.canvas.draw()

    def _plot(self, i, *args, xlimits=None, ylimits=None, xlabel=None,
              ylabel=None, clabel=None, update_canvas=False, **kwargs):
        if 'plot_kwargs' in kwargs.keys():
            self.plot_kwargs[i] = kwargs['plot_kwargs']
        if 'fit_kwargs' in kwargs.keys():
            self.fit_kwargs[i] = kwargs['fit_kwargs']
        if 'grid_kwargs' not in kwargs.keys():
            grid_kwargs = self.grid_kwargs[i]
        else:
            self.grid_kwargs[i] = kwargs['grid_kwargs']
        if self.plot_type[i] == '2D':
            kwargs = self.color_obj[i].get_kwargs()
        elif self.plot_type[i] == 'linefit':
            kwargs = {'fit_kwargs': self.fit_kwargs[i]}
        else:
            kwargs = {}
        with plt.style.context(self.plotstyle):
            try:
                self.images[i] = self.plot_function(
                    i, *args, ax=self.axes[i], fig=self.figure,
                    transpose=self.transpose[i], **self.plot_kwargs[i],
                    **kwargs)
            except (AttributeError, TypeError):
                self.images[i] = self.plot_function(
                    i, *args, ax=self.axes[i], fig=self.figure,
                    transpose=self.transpose[i])
                self.plot_kwargs[i] = {}
            except Exception:
                raise
            if self.texts[i] is not None:
                self.plot_text(i, update_canvas=False)

        if self.plot_type[i] == '2D':
            self.set_colorbar(i=i, label=clabel, update_canvas=False)
        else:
            self.remove_colorbar(i=i, update_canvas=False)

        if xlimits is None:
            self.set_axes_lim(i, x=self.xlimits[i], update_canvas=False)
        elif xlimits is not False:
            self.set_axes_lim(i, x=xlimits, update_canvas=False)

        if ylimits is None:
            self.set_axes_lim(i, y=self.ylimits[i], update_canvas=False)
        elif ylimits is not False:
            self.set_axes_lim(i, y=ylimits, update_canvas=False)

        if ylabel is None:
            ylabel = self.ylabels[i]
        if xlabel is None:
            xlabel = self.xlabels[i]
        try:
            self.set_axes_label(y=ylabel, i=i, update_canvas=False)
        except Exception:
            pass
        try:
            self.set_axes_label(x=xlabel, i=i, update_canvas=False)
        except Exception:
            pass
        self.set_axes_title(i=i, update_canvas=False)
        self.set_legend(i=i, update_canvas=False)
        self.set_fontsize(i=i, update_canvas=False)
        self.set_ticklabel_format(i=i, axis='both', update_canvas=False)
        self.set_grid(i=i, update_canvas=False, **grid_kwargs)
        self.set_frame(
            i=i, frame_on=self.axes_frame_on[i], update_canvas=update_canvas)

    def set_colorbar(self, i=None, label=None, fontsize=None,
                     update_canvas=True):
        if i is None:
            i = self.current_axes_no
        if self.cbar[i] is not None:
            self.remove_colorbar(i=i, update_canvas=False)
        self.cbar[i] = self.figure.colorbar(self.images[i], ax=self.axes[i])
        self.set_clabel(i=i, label=label, fontsize=fontsize,
                        update_canvas=update_canvas)

    def remove_colorbar(self, i=None, update_canvas=True):
        if i is None:
            i = self.current_axes_no
        try:
            self.cbar[i].remove()
        except Exception:
            pass
        self.cbar[i] = None
        if update_canvas:
            self.canvas.draw()

    def plot(self, *args, i=None, update_canvas=True, **kwargs):
        if i is None:
            i = self.current_axes_no
        self._plot(i, *args, **kwargs)
        if update_canvas:
            self.canvas.draw()

    def set_img_contrast(self, value, i=None, update_canvas=True):
        if i is None:
            i = self.current_axes_no
        if self.plot_type[i] == '2D':
            self.images[i].set_clim(
                vmin=np.exp(-(np.double(value))) * self.color_obj[i].clims[0],
                vmax=np.exp(-(np.double(value))) * self.color_obj[i].clims[1])
        if update_canvas:
            self.canvas.draw()

    def set_img_contrast_all(self, *args, **kwargs):
        self.set_for_all_axes(self.set_img_contrast, *args, **kwargs)

    def get_ylim(self, **kwargs):
        return self.get_attribute(self.ylimits, **kwargs)

    def get_xlim(self, **kwargs):
        return self.get_attribute(self.xlimits, **kwargs)

    def get_xlabel(self, **kwargs):
        return self.get_attribute(self.xlabels, **kwargs)

    def get_ylabel(self, **kwargs):
        return self.get_attribute(self.ylabels, **kwargs)

    def get_title(self, **kwargs):
        return self.get_attribute(self.axes_titles, **kwargs)

    def get_clabel(self, **kwargs):
        return self.get_attribute(self.clabels, **kwargs)

    def get_clim(self, i=None):
        if i is None:
            i = self.current_axes_no
        return self.color_obj[i].clims

    def get_attribute(self, attribute, i=None):
        if i is None:
            i = self.current_axes_no
        return attribute[i]

    def idle_function(self, *args, **kwargs):
        return None, None

    def set_callback(self, callback, event='button_press_event'):
        if event == 'button_press_event':
            self.canvas.callbacks.connect(
                event, lambda *args, **kwargs: self.axes_focus(
                    *args, callback=callback, **kwargs))
        else:
            self.canvas.callbacks.connect(event, callback)

    def axes_focus(self, event_args, callback=None):
        for i in range(len(self.axes)):
            if event_args.inaxes == self.axes[i]:
                self.current_axes_no = i
                break
        if callback is not None:
            callback(event_args)

    def set_canvas_size(self, dim):
        self.canvas.get_tk_widget().config(width=dim[0],
                                           height=dim[1])
        self.canvas.draw()

    def set_twin_axes(self, case='x', conversion_func=None, digits=0,
                      tick_locs=None, i=None, update_canvas=True):
        if i is None:
            i = self.current_axes_no
        if tick_locs is None:
            tick_locs = self.axes[i].get_xticks()
        if case.lower() == 'x':
            ax2 = self.axes[i].twiny()
            if conversion_func is None:
                conversion_func = self.twiny_conversion_func
            new_locs = [str(np.round(loc, digits))
                        for loc in conversion_func(tick_locs)]
            ax2.set_xlim(self.axes[i].get_xlim())
            ax2.set_xticks(tick_locs)
            ax2.set_xticklabels(new_locs)
        else:
            ax2 = self.axes[i].twinx()
            if conversion_func is None:
                conversion_func = self.twinx_conversion_func
            new_locs = [str(np.round(loc, digits))
                        for loc in conversion_func(tick_locs)]
            ax2.set_ylim(self.axes[i].get_ylim())
            ax2.set_yticks(tick_locs)
            ax2.set_yticklabels(new_locs)
        if update_canvas:
            self.canvas.draw()

    def set_axes_title(self, title=None, i=None, update_canvas=True):
        if i is None:
            i = self.current_axes_no
        if title is None:
            title = self.axes_titles[i]
        else:
            self.axes_titles[i] = title
        self.axes[i].title.set_text(title)
        if update_canvas:
            self.canvas.draw()

    def set_fontsize(self, i=None, fontsize=None, ticklabelsize=None,
                     update_canvas=True):
        if i is None:
            i = self.current_axes_no
        if fontsize is None:
            fontsize = self.fontsize[i]
        else:
            self.fontsize[i] = fontsize

        if ticklabelsize is None:
            ticklabelsize = self.ticklabelsize[i]
        else:
            self.ticklabelsize[i] = ticklabelsize

        for item in (self.axes[i].xaxis.label, self.axes[i].yaxis.label,
                     self.axes[i].title):
            item.set_fontsize(fontsize)
        self.set_tick_params(i=i, labelsize=ticklabelsize,
                             update_canvas=update_canvas)

        if update_canvas:
            self.canvas.draw()

    def set_ticklabel_format_all(self, *args, **kwargs):
        self.set_for_all_axes(self.set_ticklabel_format, *args, **kwargs)

    def set_ticklabel_format(self, axis='x', update_canvas=True, **kwargs):
        if re.search('both', axis, re.I):
            for ax in ('x', 'y'):
                self._set_ticklabel_format(
                    axis=ax, update_canvas=False, **kwargs)
            if update_canvas:
                self.canvas.draw()
        else:
            self._set_ticklabel_format(
                axis=axis, update_canvas=update_canvas, **kwargs)

    def _set_ticklabel_format(self, i=None, axis='x', update_canvas=True,
                              **kwargs):
        if i is None:
            i = self.current_axes_no
        for key, val in kwargs.items():
            self.ticklabel_format[i][axis][key] = val
        self.axes[i].ticklabel_format(
            axis=axis, **self.ticklabel_format[i][axis])
        if update_canvas:
            self.canvas.draw()

    def set_legend(self, entries=None, i=None, visible=None, linewidth=None,
                   update_canvas=True, handles=None, **legend_kwargs):
        if i is None:
            i = self.current_axes_no
        if entries is None:
            entries = self.legends[i]
        if visible is not None:
            self.legend_visible[i] = bool(visible)
        if legend_kwargs:
            self.set_legend_kwargs(i=i, **legend_kwargs)
        if entries is False or entries is None:
            self.legends[i] = False
        else:
            self.legends[i] = entries
            with plt.style.context(self.plotstyle):
                if handles is None:
                    self.axes[i].legend(entries, **self.legend_kwargs[i])
                else:
                    self.axes[i].legend(
                        handles, entries, **self.legend_kwargs[i])
            self.axes[i].get_legend().set_visible(self.legend_visible[i])
            if linewidth is not None:
                for line in self.axes[i].get_legend().get_lines():
                    line.set_linewidth(linewidth)
        if update_canvas:
            self.canvas.draw()

    def get_legend(self, i=None):
        if i is None:
            i = self.current_axes_no
        return [txt.get_text()
                for txt in self.axes[i].get_legend().get_texts()]

    def set_legend_visibility(self, case='toggle', i=None, update_canvas=True):
        if i is None:
            i = self.current_axes_no
        if case == 'toggle':
            case = not self.axes[i].get_legend().get_visible()
        try:
            self.axes[i].get_legend().set_visible(case)
        except Exception:
            return
        self.legend_visible[i] = case
        if update_canvas:
            self.canvas.draw()

    def set_legend_visibility_all(self, *args, **kwargs):
        self.set_for_all_axes(self.set_legend_visibility, *args, **kwargs)

    def set_axes_label(self, i=None, x=None, y=None, update_canvas=True):
        if i is None:
            i = self.current_axes_no
        if self.transpose[i]:
            x_func = self.axes[i].set_ylabel
            y_func = self.axes[i].set_xlabel
        else:
            x_func = self.axes[i].set_xlabel
            y_func = self.axes[i].set_ylabel
        if x:
            self.xlabels[i] = x
        x_func(self.xlabels[i])
        if y:
            self.ylabels[i] = y
        y_func(self.ylabels[i])
        if update_canvas:
            self.canvas.draw()

    def set_xlabel(self, label=None, i=None, update_canvas=True):
        if i is None:
            i = self.current_axes_no
        self.set_axes_label(i=i, x=label, update_canvas=update_canvas)

    def set_ylabel(self, label=None, i=None, update_canvas=True):
        if i is None:
            i = self.current_axes_no
        self.set_axes_label(i=i, y=label, update_canvas=update_canvas)

    def set_ylabel_pos(self, *args, **kwargs):
        self.set_label_pos(*args, case='y', **kwargs)

    def set_xlabel_pos(self, *args, **kwargs):
        self.set_label_pos(*args, case='x', **kwargs)

    def set_label_pos(self, x_pos, y_pos, case='x', i=None,
                      update_canvas=True):
        if i is None:
            i = self.current_axes_no
        if case.lower() == 'x':
            ax = self.axes[i].xaxis
        elif case.lower() == 'y':
            ax = self.axes[i].yaxis
        else:
            return
        ax.set_label_coords(x_pos, y_pos)
        if update_canvas:
            self.canvas.draw()

    def set_clabel(self, label=None, i=None, fontsize=None,
                   update_canvas=True):
        if i is None:
            i = self.current_axes_no
        if label is None and not self.clabels[i] is None:
            label = self.clabels[i]
        if fontsize is None:
            fontsize = self.cbar_fontsize[i]
        else:
            self.cbar_fontsize[i] = fontsize
        try:
            self.cbar[i].set_label(label, fontsize=fontsize)
        except Exception:
            pass
        else:
            self.cbar[i].ax.tick_params(labelsize=fontsize)
            if update_canvas:
                self.canvas.draw()

    def set_legend_kwargs(self, i=None, update_canvas=False, **kwargs):
        if i is None:
            i = self.current_axes_no
        for key, val in kwargs.items():
            self.legend_kwargs[i][key] = val

    def set_plot_kwargs_all(self, update_canvas=False, **kwargs):
        for i in range(self._num_subplots):
            self.set_plot_kwargs(i, **kwargs)

    def set_transpose(self, transpose, i=None, update_canvas=False, **kwargs):
        if i is None:
            i = self.current_axes_no
        self.transpose[i] = transpose

    def set_plot_kwargs(self, i=None, update_canvas=False, **kwargs):
        if i is None:
            i = self.current_axes_no
        for key, val in kwargs.items():
            self.plot_kwargs[i][key] = val

    def set_fit_kwargs(self, i=None, update_canvas=False, **kwargs):
        if i is None:
            i = self.current_axes_no
        for key, val in kwargs.items():
            self.fit_kwargs[i][key] = val

    def set_fit_kwargs_all(self, update_canvas=False, **kwargs):
        for i in range(self._num_subplots):
            self.set_fit_kwargs(i, **kwargs)

    def set_tick_params_all(self, **kwargs):
        self.set_for_all_axes(self.set_tick_params, **kwargs)

    def set_tick_params(self, i=None, case='major', update_canvas=True,
                        **kwargs):
        if i is None:
            i = self.current_axes_no
        if not case == 'major':
            for axis in (self.axes[i].xaxis, self.axes[i].yaxis):
                axis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        if 'labelsize' in kwargs.keys():
            self.ticklabelsize[i] = kwargs['labelsize']
        self.axes[i].tick_params(**kwargs)
        for key, val in kwargs.items():
            self.tick_params[i][key] = val
        if update_canvas:
            self.canvas.draw()

    def set_frame_width_all(self, **kwargs):
        self.set_for_all_axes(self.set_frame_width, **kwargs)

    def set_frame_width(self, i=None, width=1, update_canvas=True):
        if i is None:
            i = self.current_axes_no
        for axis in ['top', 'bottom', 'left', 'right']:
            self.axes[i].spines[axis].set_linewidth(width)
        if update_canvas:
            self.canvas.draw()

    def update_color_obj(self, i=None, **kwargs):
        if i is None:
            i = self.current_axes_no
        self.color_obj[i].update(**kwargs)

    def set_clim(self, clims=None, i=None, update_canvas=True):
        if i is None:
            i = self.current_axes_no
        if clims is None:
            clims = self.color_obj[i].clims
        else:
            self.update_color_obj(i=i, clims=clims)
        self.images[i].set_clim(clims)
        self.cbar[i].update_normal(self.images[i])
        if update_canvas:
            self.canvas.draw()

    def set_cmap(self, i=None, update_canvas=True, **kwargs):
        if i is None:
            i = self.current_axes_no
        self.color_obj[i].update_cmap(**kwargs)
        self.images[i].set_cmap(self.color_obj[i].cmap)
        self.cbar[i].update_normal(self.images[i])
        if update_canvas:
            self.canvas.draw()

    def set_data_cursor_2d(self, xval, yval, i=None, reverse_x=False,
                           update_canvas=True):
        if i is None:
            i = self.current_axes_no
        self.axes[i].format_coord = PcolorCoordFormatter(
            self.images[i], xval, yval, reverse_x=reverse_x)
        if update_canvas:
            self.canvas.draw()

    def set_auto_clim(self, dat, i=None, update_canvas=True, **kwargs):
        if i is None:
            i = self.current_axes_no
        clims = self.color_obj[i].set_auto_clim(dat, **kwargs)
        self.images[i].set_clim(vmin=clims[0], vmax=clims[1])
        if update_canvas:
            self.canvas.draw()
        return clims

    def get_lines_properties(self, i=None, j=0, props=None):
        if i is None:
            i = self.current_axes_no
        if props is None:
            props = ['linestyle', 'linewidth', 'color', 'drawstyle', 'alpha',
                     'label', 'marker', 'markersize', 'fillstyle',
                     'markerfacecolor', 'markeredgewidth', 'markeredgecolor']
        try:
            line = self.get_axes(i).get_lines()[j]
        except Exception:
            return {}
        else:
            properties = {}
            for prop in props:
                try:
                    exec(prop.join(["properties[\'", "\'] = line.get_", "()"]))
                except Exception:
                    properties[prop] = None
            return properties

    def set_lines_properties(self, i=None, j=0, update_canvas=True, **kwargs):
        if i is None:
            i = self.current_axes_no
        try:
            line = self.get_axes(i).get_lines()[j]
        except Exception as e:
            return {"Error retrieving line": e}
        else:
            return self.set_child_properties(
                i, line, update_canvas=update_canvas, **kwargs)

    def set_lines_properties_all(self, i=None, update_canvas=True, 
                                 exclude_tag=None, **props):
        if i is None:
            i = self.current_axes_no
        lines = self.get_axes(i).get_lines()
        if exclude_tag is None:
            for line in lines:
                self.set_child_properties(
                    i, line, update_canvas=False, **props)
        else:
            for line in lines:
                if not re.search(exclude_tag, line.get_label()):
                    self.set_child_properties(
                        i, line, update_canvas=False, **props)
        if update_canvas:
            self.canvas.draw()

    def set_child_properties(self, ax_num, obj, update_canvas=True, **props):
        errors = {}
        try:
            obj.set(**props)
        except Exception:
            for prop, value in props.items():
                try:
                    obj.set(prop=value)
                except Exception:
                    try:
                        exec(prop.join(["errors[\'", "\'] = obj.get_", "()"]))
                    except Exception:
                        pass
        self.set_legend(i=ax_num, update_canvas=update_canvas)
        return errors

    def set_collection_properties(self, i=None, j=0, update_canvas=True,
                                  **kwargs):
        if i is None:
            i = self.current_axes_no
        try:
            coll = self.get_axes(i).collections[j]
        except Exception as e:
            return {"Error retrieving collection": e}
        else:
            return self.set_child_properties(
                i, coll, update_canvas=update_canvas, **kwargs)

    def set_line_fill(self, i=None, alpha=None, lines="all",
                      update_canvas=True, **fill_kwargs):
        if i is None:
            i = self.current_axes_no
        if lines == "all":
            lines = self.get_lines_list(i, lines)
        if lines is None:
            return False
        if alpha is None:
            alpha = 0
        self.plot_kwargs[i]['fill'] = alpha
        if not type(alpha) is list:
            alpha = [alpha for i in range(len(lines))]
        else:
            while len(alpha) < len(lines):
                alpha.extend(alpha)
        for j, line in enumerate(lines):
            x = line.get_xdata()
            y = line.get_ydata()
            coll = fill_between(self.axes[i], x, y, alpha=alpha[j],
                                color=line.get_color(),
                                zorder=line.get_zorder(),
                                **fill_kwargs)
            self.line_fill[i][line] = coll
        if update_canvas:
            self.canvas.draw()
        return True

    def get_lines_list(self, i, lines="all"):
        all_lines = self.axes[i].get_lines()
        if type(lines) is list:
            try:
                lines = [all_lines[j] for j in lines]
            except Exception:
                return
        else:
            if lines == "all":
                lines = all_lines
            else:
                try:
                    lines = [all_lines[lines]]
                except Exception:
                    return
        return lines

    def clear_fill(self, i=None, lines="all", update_canvas=True):
        if i is None:
            i = self.current_axes_no
        lines = self.get_lines_list(i, lines)
        if lines is None:
            return
        for line in lines:
            if line in self.line_fill[i].keys():
                self.line_fill[i][line].remove()
                del self.line_fill[i][line]
        if update_canvas:
            self.canvas.draw()
        return True

    def set_for_current_axes(self, func, *func_args, **func_kwargs):
        func(*func_args, i=self.current_axes_no, **func_kwargs)

    def set_for_all_axes(self, func, *func_args, update_canvas=True,
                         **func_kwargs):
        for i in range(self._num_subplots):
            func(*func_args, i=i, update_canvas=False, **func_kwargs)
        if update_canvas:
            self.canvas.draw()

    def get_num_subplots(self):
        return self._num_subplots


class TkMplMovieFigure(TkMplFigure):
    # incomplete
    def __init__(self, movie_function, **figure_kwargs):
        TkMplFigure.__init__(self, **figure_kwargs)


# %%
class FigureOptionWindow(tk.Toplevel):
    def __init__(self, parent, *args, title="Figure Options",
                 controller=None, **kwargs):
        tk.Toplevel.__init__(self, parent)
        self.frame = FigureOptionFrame(
            self, *args, controller=controller, **kwargs)
        self.frame.grid(sticky='wnse')
        self.title(title)
        if not controller:
            controller = parent
        center_toplevel(self, controller)


class FigureOptionFrame(tk.Frame):
    def __init__(self, parent, figure, dim=None, controller=None,
                 max_canvas_dim=2000, max_fontsize=40, editable=True,
                 get_wdgt_val_from_fig=False, external_figure=False,
                 **fig_kwargs):
        tk.Frame.__init__(self, parent)
        self.figure = figure
        self.max_canvas_dim = max_canvas_dim
        self.max_fontsize = max_fontsize
        self.external_figure = external_figure
        self.controller = controller
        # init non-input parameters
        self.frames = {}
        self.widgets = {}
        self.vars = {}
        self.labels = {}
        self.opt_panels = {}
        if dim is None:
            dim = [600, 400]
        # check which plot types are contained in figure
        if editable:
            self.set_plot_type(get_wdgt_val_from_fig=get_wdgt_val_from_fig)
            self.init_edit_opts(self, canvas_dim=dim, row=0, rowspan=3,
                                cbar_opts=self.check_2d)
        else:
            self._get_plot_types()

    def init_edit_opts(self, parent, row=0, column=1, pady=10, padx=10,
                       sticky=tk.W, border=False, cbar_opts=None,
                       canvas_dim=None, **frame_kwargs):
        obj = self.figure
        frame_row = row
        if cbar_opts is None:
            cbar_opts = self.check_2d
        row = 0
        self.frames['edit_opts'] = CustomFrame(
            parent, dim=(2, 16), border=border)
        tk.Label(self.frames['edit_opts'], text="Title:").grid(
            row=row, column=0, sticky=tk.W, columnspan=2)
        row += 1
        self.vars['title'] = tk.StringVar(value=obj.axes[0].get_title())
        self.widgets['title'] = tk.ttk.Entry(
            self.frames['edit_opts'], textvariable=self.vars['title'],
            width=30)
        self.widgets['title'].bind('<Return>', self.set_title)
        self.widgets['title'].grid(row=row, column=0, sticky=tk.W, pady=10,
                                   columnspan=2)
        row += 1

        tk.Label(self.frames['edit_opts'], text="Font Size:").grid(
            row=row, column=0, sticky=tk.W)
        row += 1
        self.vars['fontsize'] = tk.DoubleVar(value=mpl.rcParams['font.size'])
        self.widgets['fontsize'] = tk.ttk.Entry(
            self.frames['edit_opts'], textvariable=self.vars['fontsize'],
            width=5)
        self.widgets['fontsize'].bind('<Return>', self.set_fontsize)
        self.widgets['fontsize'].grid(row=row, column=0, pady=10,
                                      sticky=tk.W)
        if self.check_2d:
            tk.Label(self.frames['edit_opts'],
                     text='Colorbar label Size:').grid(
                         row=row - 1, column=1, sticky=tk.W)
            self.vars['cbar_fontsize'] = tk.DoubleVar(
                value=np.round(self.vars['fontsize'].get() * 0.8))
            self.widgets['cbar_fontsize'] = tk.ttk.Entry(
                self.frames['edit_opts'],
                textvariable=self.vars['cbar_fontsize'], width=5)
            self.widgets['cbar_fontsize'].bind('<Return>', self.set_clabel)
            self.widgets['cbar_fontsize'].grid(row=row, column=1, pady=10,
                                               sticky=tk.W)
        row += 1
        self.labels['xlabel'] = tk.Label(
            self.frames['edit_opts'], text="X label:")
        self.labels['xlabel'].grid(row=row, column=0, pady=10,
                                   sticky=tk.W, columnspan=2)
        row += 1
        self.vars['xlabel'] = tk.StringVar(value=obj.axes[0].get_xlabel())
        self.widgets['xlabel'] = tk.ttk.Entry(
            self.frames['edit_opts'], textvariable=self.vars['xlabel'],
            width=30)
        self.widgets['xlabel'].bind('<Return>', self.set_xlabel)
        self.widgets['xlabel'].grid(row=row, column=0, pady=10,
                                    sticky=tk.W, columnspan=2)
        row += 1
        self.labels['ylabel'] = tk.Label(
            self.frames['edit_opts'], text="Y label:")
        self.labels['ylabel'].grid(row=row, column=0, sticky=tk.W,
                                   columnspan=2)
        row += 1
        self.vars['ylabel'] = tk.StringVar(value=obj.axes[0].get_ylabel())
        self.widgets['ylabel'] = tk.ttk.Entry(
            self.frames['edit_opts'], textvariable=self.vars['ylabel'],
            width=30)
        self.widgets['ylabel'].bind('<Return>', self.set_ylabel)
        self.widgets['ylabel'].grid(row=row, column=0, pady=10,
                                    sticky=tk.W, columnspan=2)
        row += 1

        if cbar_opts:
            row = self.init_cbar_opt(row)
        self.vars['xlower'] = tk.DoubleVar()
        self.vars['xupper'] = tk.DoubleVar()
        self.vars['ylower'] = tk.DoubleVar()
        self.vars['yupper'] = tk.DoubleVar()
        try:
            self.vars['xlower'].set(np.round(obj.xlimits[0][0], 3))
            self.vars['xupper'].set(np.round(obj.xlimits[0][1], 3))
        except Exception:
            pass
        try:
            self.vars['ylower'].set(np.round(obj.ylimits[0][0], 3))
            self.vars['yupper'].set(np.round(obj.ylimits[0][1], 3))
        except Exception:
            pass

        for k in ['xlower', 'xupper', 'ylower', 'yupper']:
            self.widgets[k] = tk.ttk.Entry(
                self.frames['edit_opts'], textvariable=self.vars[k], width=10,
                justify=tk.RIGHT)
        for k in ['xlower', 'xupper']:
            self.widgets[k].bind(
                '<Return>', lambda *args: self.set_axlim_callback(
                    *args, case='x'))
        for k in ['ylower', 'yupper']:
            self.widgets[k].bind(
                '<Return>', lambda *args: self.set_axlim_callback(
                    *args, case='y'))
        self.labels['xlim'] = tk.Label(
            self.frames['edit_opts'], text="X limits:")
        self.labels['xlim'].grid(row=row, column=0, pady=10, sticky=tk.W)
        row += 1
        self.widgets['xlower'].grid(row=row, column=0, sticky=tk.W)
        self.widgets['xupper'].grid(row=row, column=1, sticky=tk.W)
        row += 1
        self.labels['ylim'] = tk.Label(
            self.frames['edit_opts'], text="Y limits:")
        self.labels['ylim'].grid(row=row, column=0, pady=10, sticky=tk.W)
        row += 1
        self.widgets['ylower'].grid(row=row, column=0, sticky=tk.W)
        self.widgets['yupper'].grid(row=row, column=1, sticky=tk.W)
        row += 1

        self.vars['transpose_axes'] = tk.IntVar(value=0)
        self.widgets['transpose_axes'] = tk.ttk.Checkbutton(
            self.frames['edit_opts'], text="Transpose Axes",
            variable=self.vars['transpose_axes'],
            command=self.transpose_axes_callback)
        self.widgets['transpose_axes'].grid(row=row,
                                            column=0, sticky='w', pady=5)
        self.vars['invert_y'] = tk.IntVar()
        self.vars['invert_y'].set(1 if re.search('2d', self.figure.plot_type[
            self.figure.current_axes_no], re.I) else 0)
        tk.ttk.Checkbutton(self.frames['edit_opts'],
                           text="Invert Y",
                           variable=self.vars['invert_y'],
                           command=self.invert_y_callback).grid(
                               row=row, column=1, sticky='w', pady=5)
        row += 1
        self.labels['ydisp_mode'] = tk.ttk.Label(self.frames['edit_opts'],
                                                 text="Y axis display:")
        self.labels['ydisp_mode'].grid(row=row, column=0, sticky='w',
                                       pady=5)
        self.vars['ydisp_mode'] = tk.StringVar(value="Values")
        self.widgets['ydisp_mode'] = tk.ttk.OptionMenu(
            self.frames['edit_opts'], self.vars['ydisp_mode'],
            self.vars['ydisp_mode'].get(), "Values", "Values log10",
            "Points", command=lambda *args: self.y_disp_mode_callback())
        self.widgets['ydisp_mode'].grid(row=row, column=1, sticky='w',
                                        pady=5)
        row += 1
        row = self.init_canvas_opts(row=row, canvas_dim=canvas_dim)
        self.frames['edit_opts'].grid(row=frame_row, column=column, padx=padx,
                                      pady=pady, sticky=sticky, **frame_kwargs)

    def set_edit_mode(self, *args, external_figure=None):
        # external_figure: boolean
        if external_figure is None:
            external_figure = self.external_figure
        else:
            self.external_figure = external_figure
        self.vars['transpose_axes'].set(0)
        for key in ['transpose_axes', 'ydisp_mode']:
            self.widgets[key].config(
                state='disabled' if external_figure else 'normal')

    def set_plot_type(self, plot_type=None, get_wdgt_val_from_fig=True):
        self._get_plot_types(plot_type=plot_type)
        for panel in self.opt_panels.values():
            panel.grid_remove()
        try:
            self.contrast_slider.grid_remove()
        except Exception:
            pass
        row = 0
        # 2D plot options
        if self.check_2d:
            if 'color' in self.opt_panels.keys():
                self.opt_panels['color'].grid(row=row)
                self.contrast_slider.grid(row=row, column=0, rowspan=2)
            else:
                self.init_color_opts(
                    self, self, row=row, rowspan=1,
                    contrast_slider_kwargs={'row': row, 'column': 0})
            row += 1
            if not self.check_line:
                if 'axes_only' not in self.opt_panels.keys():
                    self.opt_panels['axes_only'] = AxesEditor(
                        self, self.figure, controller=self.controller)
                self.opt_panels['axes_only'].grid(row=row,
                                                  column=2, sticky='nwe')
                self.plot_opts = self.opt_panels['axes_only']
        # line plot options
        if self.check_line:
            self.init_lineplot_opts(self, self.controller, rowspan=2, row=row,
                                    include_fit=self.check_fit)
        if get_wdgt_val_from_fig:
            self.axes_focus_update_ui()

    def _get_plot_types(self, plot_type=None):
        if plot_type is None:
            plot_type = self.figure.plot_type
        else:
            plot_type = [plot_type]
        self.check_2d = False
        self.check_line = False
        self.check_fit = False
        for t in plot_type:
            if re.search('2D', t, re.I):
                self.check_2d = True
            if re.search('line', t, re.I):
                self.check_line = True
            if re.search('fit', t, re.I):
                self.check_fit = True
            if self.check_2d and self.check_line and self.check_fit:
                break

    def axes_focus_update_ui(self, *args, callback=None):
        i = self.figure.current_axes_no
        try:
            self.get_widget_values_from_figure(i)
        except (TypeError, AttributeError) as e:
            print(e)
        except Exception:
            raise
        if callback is not None:
            callback(*args)

    def get_widget_values_from_figure(self, i):
        for fun in (
                lambda: self.vars['title'].set(self.figure.axes_titles[i]),
                lambda: self.vars['xlabel'].set(self.figure.xlabels[i]),
                lambda: self.vars['ylabel'].set(self.figure.ylabels[i]),
                lambda: self.vars['xlower'].set(self.figure.xlimits[i][0]),
                lambda: self.vars['xupper'].set(self.figure.xlimits[i][1]),
                lambda: self.vars['ylower'].set(self.figure.ylimits[i][0]),
                lambda: self.vars['yupper'].set(self.figure.ylimits[i][1]),
                lambda: self.opt_panels['color'].get_color_obj_info(
                    self.figure.color_obj[i])):
            try:
                fun()
            except (TypeError, KeyError):
                pass
            except Exception as e:
                print(e)
        try:
            self.plot_opts.get_options_from_figure(i)
        except Exception:
            pass

    def idle_function(self, *args, **kwargs):
        return

    def init_canvas_opts(self, parent=None, row=0, column=0, pady=10, padx=0,
                         canvas_dim=None, center=True):
        if canvas_dim is None:
            canvas_dim = [600, 400]
        if parent is None:
            parent = self.frames['edit_opts']
        if center:
            sticky = ['e', 'w']
        else:
            sticky = ['w', 'w']
        tk.Label(parent, text='Canvas Size:').grid(
            row=row, column=0, pady=10,
            sticky=None if center else 'w', columnspan=2)
        row += 1

        for i, key in enumerate(('canvas_width', 'canvas_height')):
            self.vars[key] = tk.IntVar(value=canvas_dim[i])
            self.widgets[key] = tk.ttk.Entry(
                parent, textvariable=self.vars[key], width=10,
                justify=tk.RIGHT)
            self.widgets[key].grid(row=row, column=column + i, pady=10,
                                   padx=padx,
                                   sticky=sticky[i])
            self.widgets[key].bind('<Return>', self.set_canvas_size)
        return row

    def init_lineplot_opts(self, parent, controller, *plot_args, row=0,
                           column=2, padx=10, pady=10, rowspan=1,
                           include_fit=False, **plot_kwargs):
        self.opt_panels['lines'] = LinePlotEditor(
            parent, self.figure, *plot_args, include_fit=include_fit,
            controller=controller, **plot_kwargs)
        self.opt_panels['lines'].grid(row=row, column=column, padx=padx,
                                      pady=10, rowspan=rowspan)
        self.plot_opts = self.opt_panels['lines']

    def init_color_opts(self, parent, controller=None, row=0, column=2,
                        padx=10, pady=10,
                        contrast_slider_kwargs={'row': 0, 'column': 0},
                        **grid_kwargs):
        self.contrast_slider = ContrastSlider(self, self.figure)
        self.contrast_slider.grid(**contrast_slider_kwargs)
        self.opt_panels['color'] = ColorMapEditor(
            parent, self.figure, horizontal=False,
            contrast_slider=self.contrast_slider,
            num_sub_plots=len(self.figure.axes))
        self.opt_panels['color'].grid(row=row, column=column, padx=padx,
                                      pady=pady, **grid_kwargs)
        try:
            self.opt_panels['color'].clim_callback([])
        except Exception:
            pass

    def init_cbar_opt(self, row, column=0):
        tk.ttk.Label(self.frames['edit_opts'], text="Colorbar label:").grid(
            row=row, column=column, pady=10, sticky=tk.W,
            columnspan=2)
        self.vars['clabel'] = tk.StringVar(value="$\Delta$ Abs.")
        self.widgets['clabel'] = tk.ttk.Entry(
            self.frames['edit_opts'], textvariable=self.vars['clabel'],
            width=30)
        self.widgets['clabel'].bind('<Return>', self.set_clabel)
        self.widgets['clabel'].grid(row=row + 1, column=column, pady=10,
                                    sticky=tk.W, columnspan=2)
        return row + 2

    def set_canvas_size(self, *args, dim=None):
        if dim is None:
            dim = []
            for var in self.vars['canvas_width'], self.vars['canvas_height']:
                if var.get() > self.max_canvas_dim:
                    var.set(self.max_canvas_dim)
                dim.append(var.get())
        self.figure.set_canvas_size(dim)

    def set_legend(self, *args):
        self.figure.set_legend(entries=self.vars['legend'].get().split(","))

    def axes_aspect_callback(self, *args):
        ratio = 1 if self.vars['equal_axes_aspect'].get() else 'auto'
        self.figure.set_axes_aspect_ratio(ratio=ratio, update_canvas=True)

    def transpose_axes_callback(self, *args):
        self.vars['invert_y'].set(int(self.vars['transpose_axes'].get() == 0))
        if self.plot_opts.vars['edit_all']:
            nums = range(len(self.figure.axes))
        else:
            nums = [self.figure.current_axes_no]
        for n in nums:
            self.figure.transpose[n] = bool(self.vars['transpose_axes'].get())
            self.figure.invert_yaxis[n] = bool(self.vars['invert_y'].get())
        self.figure.plot()
        swap_dct = {'x': 'y', 'y': 'x'}
        for lb in self.labels.values():
            text = list(lb.cget("text"))
            text[0] = swap_dct[text[0].lower()].title()
            lb.config(text="".join(text))

    def invert_y_callback(self, *args):
        if self.plot_opts.vars['edit_all']:
            self.figure.set_for_all_axes(self.figure.set_invert_yaxis,
                                         self.vars['invert_y'].get())
        else:
            self.figure.set_invert_yaxis(self.vars['invert_y'].get())

    def y_disp_mode_callback(self, *args):
        ydispmode = self.vars['ydisp_mode'].get()
        if re.search('value', ydispmode, re.I):
            ymode = 'values'
        else:
            ymode = 'points'
        if re.search('log', ydispmode, re.I):
            ymode += re.search('log\d*', ydispmode, re.I).group(0)
        i = self.figure.current_axes_no
        ax = self.figure.axes[i]
        if re.search('2d', self.figure.plot_type[i], re.I):
            case = 'y'
        else:
            case = 'x'
        kwargs = {case + 'mode': ymode}
        self.figure.set_plot_kwargs(**kwargs)
        kwargs = {case + 'limits': False}
        self.figure.plot(update_canvas=False, **kwargs)
        exec(
            case.join(["self.figure.set_",
                       "lim(ax.get_",
                       "lim(), update_canvas=False)"]))
        self.get_widget_values_from_figure(self.figure.current_axes_no)
        if ydispmode.lower() == "points":
            getattr(ax, case.join(['set_', 'label']))(ydispmode)
            self.vars[case + 'label'].set(ydispmode)

        self.figure.canvas.draw()

    def set_axlim_callback(self, *args, case='x', update=True):
        if case == 'x':
            try:
                self.figure.set_axes_lim(x=[self.vars['xlower'].get(),
                                            self.vars['xupper'].get()],
                                         update_canvas=update)
            except Exception:
                self.vars['xlower'].set(
                    self.figure.xlimits[self.figure.current_axes_no][0])
                self.vars['xupper'].set(
                    self.figure.xlimits[self.figure.current_axes_no][1])
                raise
        elif case == 'y':
            try:
                self.figure.set_axes_lim(y=[self.vars['ylower'].get(),
                                            self.vars['yupper'].get()],
                                         update_canvas=update)
            except Exception:
                self.vars['ylower'].set(
                    self.figure.ylimits[self.figure.current_axes_no][0])
                self.vars['yupper'].set(
                    self.figure.ylimits[self.figure.current_axes_no][1])
                raise

    def set_clabel(self, *args, update=True):
        self.figure.set_clabel(label=self.vars['clabel'].get(),
                               fontsize=None, update_canvas=update)

    def set_ylabel(self, *args, update=True):
        try:
            self.figure.set_axes_label(
                y=self.vars['ylabel'].get(), update_canvas=update)
        except Exception:
            self.vars['ylabel'].set(
                self.figure.ylabels[self.figure.current_axes_no])

    def set_xlabel(self, *args, update=True):
        try:
            self.figure.set_axes_label(
                x=self.vars['xlabel'].get(), update_canvas=update)
        except Exception:
            self.vars['xlabel'].set(
                self.figure.xlabels[self.figure.current_axes_no])

    def set_title(self, *args, update=True):
        try:
            self.figure.set_axes_title(
                self.vars['title'].get(), update_canvas=update)
        except Exception:
            self.vars['title'].set(
                self.figure.axes_titles[self.figure.current_axes_no])

    def set_fontsize(self, *args, update=True):
        try:
            if self.vars['fontsize'].get() > self.max_fontsize:
                self.vars['fontsize'].set(self.max_fontsize)
        except Exception:
            return
        else:
            self.figure.set_clabel(fontsize=self.vars['fontsize'].get()*0.9,
                                   update_canvas=False)
            self.figure.set_fontsize(
                fontsize=self.vars['fontsize'].get(), update_canvas=update)

    def adjust_contrast(self, *args):
        self.figure.set_img_contrast_all(*args)


# %%
class FigFrame(tk.Frame):
    def __init__(self, parent, fig_obj=None, controller=None, dim=None,
                 plot_type='2D', plotstyle='fast', editable=False, xvals=None,
                 yvals=None, plot_func=None, header=None, button_dict=None,
                 xlimits=None, ylimits=None, canvas_callback=None,
                 num_subplots=1, num_rows=1, num_columns=1, data_obj=None,
                 show_plot=False, get_wdgt_val_from_fig=False, **kwargs):
        # set up container frame
        tk.Frame.__init__(self, parent)
        for i in range(1 + int(editable) * 2):
            self.columnconfigure(i, weight=1)
        for i in range(3 + bool(header) + bool(button_dict)):
            self.rowconfigure(i, weight=1)

        if dim is None:
            dim = [600, 400]
        # init non-input parameters
        self.frames = {}
        self.widgets = {}
        self.vars = {}
        self.labels = {}
        # read input parameters
        # simple copy
        self.parent = parent
        self.controller = controller
        self.editable = editable
        # header label
        row = 0
        if header is not None:
            if 'font' not in header.keys():
                header['font'] = "Helvetica"
            if 'fontsize' not in header.keys():
                header['fontsize'] = 15
            tk.Label(self, text=header['text'],
                     font=(header['font'],
                           header['fontsize'])).grid(
                               row=row, column=0, columnspan=3)
            row += 1
        if self.editable:
            if canvas_callback is None:
                cb = {'button_press_event':
                      lambda *args: self.axes_focus_update_ui(*args)}
            else:
                cb = {}
                for event, fun in canvas_callback.items():
                    if event == 'button_press_event':
                        cb[event] = lambda *args: self.axes_focus_update_ui(
                            *args, callback=fun)
                    else:
                        cb[event] = fun
        else:
            cb = canvas_callback
        if fig_obj is None:
            # initialize figure object
            self.figure = TkMplFigure(
                self, dim=dim, row=row, plot_function=plot_func,
                num_subplots=num_subplots, num_rows=num_rows,
                num_columns=num_columns, callbacks=cb, plot_type=plot_type,
                plotstyle=plotstyle, xlimits=xlimits, ylimits=ylimits,
                **kwargs)
        else:
            self.figure = fig_obj
            self.figure.parent = self
            self.figure.parent_row = row
            self.figure.dim = dim
            if cb is not None:
                self.figure.canvas_callbacks = cb
                for event, callback in self.figure.canvas_callbacks.items():
                    self.figure.set_callback(callback, event)
            self.figure.init_tk_widgets()
        self.figure.grid()
        if show_plot:
            self.figure.plot_all()
        # initialize option frame
        self.opts = FigureOptionFrame(
            self, self.figure, dim=dim, controller=controller,
            editable=self.editable,
            get_wdgt_val_from_fig=get_wdgt_val_from_fig)
        self.opts.grid(row=row, column=1, rowspan=2,
                       sticky='wnse', padx=5, pady=5)
        if self.editable:
            if self.opts.check_2d:
                if (xvals is not None) and (yvals is not None):
                    for ax, im in zip(self.figure.axes, self.figure.images):
                        ax.format_coord = PcolorCoordFormatter(im,
                                                               xvals, yvals)
            if type(data_obj) is TATrace:
                self.parent.traces = data_obj
                self.init_trace_opts(data_obj)

        # set up command buttons if provided
        if button_dict is not None:
            self.frames['save_buttons'] = CustomFrame(
                self, dim=(len(button_dict.keys()), 1), border=False)
            i = 0
            for k in button_dict.keys():
                ttk.Button(self.frames['save_buttons'], text=k,
                           command=button_dict[k]).grid(row=0, column=i)
                i = i + 1
            self.frames['save_buttons'].grid(
                row=row + 2, column=0, columnspan=2, sticky='wnse',
                pady=(0, 10))

    def axes_focus_update_ui(self, *args, callback=None, **kwargs):
        try:
            self.opts.axes_focus_update_ui(*args, **kwargs)
        except AttributeError:
            pass
        except Exception:
            raise
        if callback:
            callback(*args, **kwargs)

    def init_trace_opts(self, traces, row=0, column=3,
                        padx=10, pady=10, rowspan=3):
        select_frame = TraceSelection(
            self, traces, fig_obj=self.figure, update_func=None,
            header="Select curves", layout='vertical')
        select_frame.grid(row=row, column=column,
                          padx=padx, pady=10, rowspan=rowspan)
        select_frame.update_box()
        try:
            select_frame.update_func = self.figure.plot
        except Exception:
            pass

    def open_fig_obj(self, *args):
        file = load_box(fext=".pkl", filetypes=[
                       ('Figure', '*.pkl')], parent=self)
        try:
            file[0].name
        except Exception:
            return
        else:
            self.figure.load_figure(file[0].name)
            self.opts.set_edit_mode(external_figure=True)
            self.opts.set_plot_type()
            try:
                title = ["..."]
                title.extend(file[0].name.split("/")[-3:])
                title = "Figure " + "/".join(title)
                self.parent.title(title)
            except Exception:
                pass

    def save_figure_obj(self, *args):
        self.figure.save_figure()


# %%
class ContrastSlider(tk.Scale):
    def __init__(self, parent, figure, from_=4, to=-3,
                 resolution=0.1, length=300, label='Contrast',
                 **scale_kwargs):
        tk.Scale.__init__(self, parent, from_=from_, to=to,
                          resolution=resolution, length=length, label=label,
                          command=self.adjust_contrast, **scale_kwargs)
        self.figure = figure

    def adjust_contrast(self, *args):
        self.figure.set_img_contrast_all(*args)


# %%
class AxesEditor(CustomFrame):
    def __init__(self, parent,
                 figure, *plot_args, controller=None, style_edit=True,
                 grid_edit=True, plot_func=None,
                 adv_opts=True, border=True, dim=None, row=0,
                 column=0, plotstyle=None, **plot_kwargs):
        if dim is None:
            dim = (2, 6)
        CustomFrame.__init__(self, parent, border=border, dim=dim)
        self.vars = {}
        self.widgets = {}
        self.optmenus = {}
        self.widgets = {}
        self.frames = {}
        self.grid_kwargs = {}
        self.adv_opts_vals = None
        self.controller = controller
        self.figure = figure
        self.style_edit = style_edit
        self.grid_edit = grid_edit
        if plotstyle is None:
            plotstyle = self.figure.plotstyle

        if plot_func is None:
            self.callback = lambda *args, **kwargs: self.idle_function(
                *args, **kwargs)
        else:
            self.callback = lambda *args, **kwargs: plot_func(
                *args, *plot_args, **plot_kwargs, **kwargs)
        self.vars['edit_all'] = tk.IntVar(value=1)
        self.widgets['edit_all'] = tk.ttk.Checkbutton(
            self, text="Change for all subplots",
            variable=self.vars['edit_all'])
        self.widgets['edit_all'].grid(row=row, column=0)
        row += 1
        if self.style_edit:
            row = self.init_style_editor(row, plotstyle=plotstyle)
        if self.grid_edit:
            row = self.init_grid_editor(row)
        if adv_opts:
            row = self.place_adv_opts_button(row)
        self.row = row

    def init_style_editor(self, row=0, column=0, plotstyle='fast'):
        self.vars['plotstyle'] = tk.StringVar(value=plotstyle.title())
        self.widgets['plotstyle'] = tk.ttk.OptionMenu(
            self, self.vars['plotstyle'], self.vars['plotstyle'].get(),
            *self.figure.style_dict.keys(), command=self.change_style)
        tk.Label(self, text='Plot Style:').grid(row=row, column=column,
                                                sticky='wn')
        row += 1
        self.widgets['plotstyle'].grid(row=row, column=column,
                                       sticky='wne')
        return row + 1

    def init_grid_editor(self, row=0, column=0):
        self.frames['grid_editor'] = CustomFrame(
            self, dim=(2, 1), border=False)
        self.vars['grid_on'] = tk.IntVar(value=0)
        self.widgets['grid_on'] = tk.ttk.Checkbutton(
            self.frames['grid_editor'], variable=self.vars['grid_on'],
            text='Show grid', command=self.update_grid)
        self.widgets['grid_on'].grid(row=0, column=0, sticky='w')
        self.vars['frame_on'] = tk.IntVar(value=1)
        self.widgets['frame_on'] = tk.ttk.Checkbutton(
            self.frames['grid_editor'], variable=self.vars['frame_on'],
            text='Frame on', command=self.update_grid)
        self.widgets['frame_on'].grid(row=0, column=1, sticky='w')
        self.frames['grid_editor'].grid(
            row=row, column=column, sticky='wnse')
        return row + 1

    def update_grid(self):
        if self.vars['edit_all'].get():
            self.figure.set_for_all_axes(self.figure.set_grid,
                                         grid_on=self.vars['grid_on'].get(),
                                         frame_on=self.vars['frame_on'].get())
        else:
            self.figure.set_grid(grid_on=self.vars['grid_on'].get(),
                                 frame_on=self.vars['frame_on'].get())

    def place_adv_opts_button(self, row=0, column=0):
        ttk.Button(self, text='Advanced Options',
                   command=self._open_adv_opts).grid(row=row, column=column)
        return row + 1

    def _open_adv_opts(self):
        if type(self.adv_opts_vals) is list:
            try:
                value_dict = self.adv_opts_vals[self.figure.current_axes_no]
            except Exception:
                value_dict = None
        else:
            value_dict = self.adv_opts_vals
        AdvancedPlotOptions(self, self.figure, controller=self.controller,
                            value_dict=value_dict)

    def change_style(self, *args):
        self.figure.set_style(self.vars['plotstyle'].get())
        if self.grid_edit and self.style_edit:
            self.vars['grid_on'].set(
                self.figure.axes_grid_on[self.figure.current_axes_no])

    def get_options_from_figure(self, i):
        self._get_fig_opts(i)

    def _get_fig_opts(self, i):
        if not self.vars['edit_all'].get():
            for fun in (
                lambda: self.vars['grid_on'].set(self.figure.axes_grid_on[i]),
                lambda: self.vars['frame_on'].set(self.figure.axes_frame_on[i])
            ):
                try:
                    fun()
                except Exception as e:
                    print(e)

    def idle_function(self, *args, **kwargs):
        return


# %%
class AdvancedPlotOptions(tk.Toplevel):
    def __init__(self, parent, figure, controller=None, value_dict=None,
                 edit_all=False, update_func=None):
        tk.Toplevel.__init__(self, parent)
        self.frames = {}
        self.vars = {}
        self.optmenus = {}
        self.widgets = {}
        self.grid_kwargs = {}
        self.legend_kwargs = {}
        self.tick_kwargs = {'width': 0.5}
        self.misc_params = {'frame_width': 1}
        self.figure = figure
        self.parent = parent
        self.reverse_color = False
        self.value_dict = {'grid_style': '-',
                           'grid_width': 0.9,
                           'grid_axis': 'both',
                           'frame_width': 1,
                           'ticks_which': 'major',
                           'ticks_show': 1,
                           'ticks_bottom': 1,
                           'ticks_top': 0,
                           'ticks_left': 1,
                           'ticks_right': 0,
                           'ticks_labelsize': 11,
                           'ticks_size': 0.5,
                           'ticks_width': 0.5,
                           'legend_anchor_x': 0.5,
                           'legend_anchor_y': 0.5,
                           'reverse_color': 0,
                           'shift_color_cycle': 0,
                           'xlabel_pos_x': 0.5,
                           'xlabel_pos_y': -0.1,
                           'ylabel_pos_x': -0.1,
                           'ylabel_pos_y': 0.5,
                           'sci_not_ord_mag_lower': -3,
                           'sci_not_ord_mag_upper': 3}

        for k, l in zip(('grid_style', 'grid_width', 'grid_axis'),
                        ('linestyle', 'linewidth', 'axis')):
            try:
                self.value_dict[k] = self.figure.grid_kwargs[
                    self.figure.current_axes_no][l]
            except KeyError:
                pass
            except Exception:
                raise

        for k in ('which', 'show', 'bottom', 'top', 'left', 'right',
                  'labelsize', 'width'):
            try:
                self.value_dict['ticks_' + k] = self.figure.tick_params[
                    self.figure.current_axes_no][k]
            except KeyError:
                pass
            except Exception:
                raise
        try:
            self.value_dict['ticks_labelsize'] = self.figure.ticklabelsize[
                self.figure.current_axes_no]
        except Exception:
            pass
        if value_dict is not None:
            for key, val in value_dict.items():
                self.value_dict[key] = val

        if update_func is not None:
            self.update_parent = update_func

        row = 0

        # grid
        self.frames['grid'] = GroupBox(
            self, text='Grid options', border=True, dim=(2, 4))

        tk.Label(self.frames['grid'], text='Style').grid(row=0, column=0,
                                                         sticky='w')
        tk.Label(self.frames['grid'], text='Width').grid(row=1, column=0,
                                                         sticky='w')
        tk.Label(self.frames['grid'], text='Axis').grid(row=2, column=0,
                                                        sticky='w')

        self.vars['grid_style'] = tk.StringVar(
            value=self.value_dict['grid_style'])
        self.optmenus['grid_style'] = tk.ttk.OptionMenu(
            self.frames['grid'], self.vars['grid_style'],
            self.vars['grid_style'].get(), '-', '-.', '--', ':',
            command=self.update_grid_opts)
        self.optmenus['grid_style'].config(width=7)
        self.optmenus['grid_style'].grid(row=0, column=1, sticky='w')

        self.vars['grid_width'] = tk.DoubleVar(
            value=self.value_dict['grid_width'])
        self.widgets['grid_width'] = tk.ttk.Entry(
            self.frames['grid'], textvariable=self.vars['grid_width'], width=5)
        self.widgets['grid_width'].grid(row=1, column=1, sticky='w')
        self.widgets['grid_width'].bind('<Return>', self.update_grid_opts)

        self.vars['grid_axis'] = tk.StringVar(
            value=self.value_dict['grid_axis'])
        self.optmenus['grid_axis'] = tk.ttk.OptionMenu(
            self.frames['grid'], self.vars['grid_axis'],
            self.vars['grid_axis'].get(), 'both', 'x', 'y',
            command=self.update_grid_opts)
        self.optmenus['grid_axis'].config(width=7)
        self.optmenus['grid_axis'].grid(row=2, column=1, sticky='w')

        tk.Label(self.frames['grid'], text='Frame width:').grid(
            row=3, column=0, sticky='w')
        self.vars['frame_width'] = tk.DoubleVar(
            value=self.value_dict['frame_width'])
        self.widgets['frame_width'] = tk.ttk.Entry(
            self.frames['grid'], textvariable=self.vars['frame_width'],
            width=5)
        self.widgets['frame_width'].bind('<Return>', self.set_framewidth)
        self.widgets['frame_width'].grid(row=3, column=1, sticky='w')

        self.frames['grid'].grid(row=row, column=0, sticky='wnse',
                                 padx=5, pady=5)

        # ticks
        self.frames['ticks'] = GroupBox(
            self, dim=(2, 6), text="Axes ticks", border=True)

        tk.Label(self.frames['ticks'], text='Edit:').grid(
            row=0, column=0, sticky='w')
        self.vars['ticks_which'] = tk.StringVar(
            value=self.value_dict['ticks_which'])
        self.optmenus['ticks_which'] = tk.ttk.OptionMenu(
            self.frames['ticks'], self.vars['ticks_which'],
            self.vars['ticks_which'].get(), 'major', 'minor', 'both',
            command=self.update_ticks)
        self.optmenus['ticks_which'].config(width=7)
        self.optmenus['ticks_which'].grid(row=0, column=1, sticky='w')

        self.frames['ticks_show'] = CustomFrame(
            self.frames['ticks'], border=False, dim=(4, 2))

        self.vars['ticks_show'] = tk.IntVar(
            value=self.value_dict['ticks_show'])
        self.widgets['ticks_show'] = tk.ttk.Checkbutton(
            self.frames['ticks_show'], text='Show',
            variable=self.vars['ticks_show'],
            command=self.ticks_show_check_callback)
        self.widgets['ticks_show'].grid(row=0, column=0, sticky='w')

        for i, key in enumerate(('bottom', 'top', 'left', 'right')):
            self.vars['ticks_' +
                      key] = tk.IntVar(value=self.value_dict['ticks_' + key])
            self.widgets['ticks_' + key] = tk.ttk.Checkbutton(
                self.frames['ticks_show'], text=key,
                variable=self.vars['ticks_' + key], command=self.update_ticks)
            self.widgets['ticks_' + key].grid(row=1, column=i, sticky='w')

        self.frames['ticks_show'].grid(
            row=1, column=0, columnspan=2, sticky='wns')

        tk.Label(self.frames['ticks'], text='Font size:').grid(
            row=2, column=0, sticky='w')
        self.vars['ticks_labelsize'] = tk.DoubleVar(
            value=self.value_dict['ticks_labelsize'])
        self.widgets['ticks_labelsize'] = tk.ttk.Entry(
            self.frames['ticks'], textvariable=self.vars['ticks_labelsize'],
            width=5)
        self.widgets['ticks_labelsize'].bind('<Return>', self.update_ticks)
        self.widgets['ticks_labelsize'].grid(row=2, column=1, sticky='w')

        tk.Label(self.frames['ticks'], text='Tick size:').grid(
            row=3, column=0, sticky='w')
        self.vars['ticks_size'] = tk.DoubleVar(
            value=self.value_dict['ticks_size'])
        self.widgets['ticks_size'] = tk.ttk.Entry(
            self.frames['ticks'], textvariable=self.vars['ticks_size'],
            width=5)
        self.widgets['ticks_size'].bind('<Return>', self.update_ticks)
        self.widgets['ticks_size'].grid(row=3, column=1, sticky='w')

        tk.Label(self.frames['ticks'], text='Width:').grid(
            row=4, column=0, sticky='w')
        self.vars['ticks_width'] = tk.DoubleVar(
            value=self.value_dict['ticks_width'])
        self.widgets['ticks_width'] = tk.ttk.Entry(
            self.frames['ticks'], textvariable=self.vars['ticks_width'],
            width=5)
        self.widgets['ticks_width'].bind('<Return>', self.update_ticks)
        self.widgets['ticks_width'].grid(row=4, column=1, sticky='w')

        self.frames['ticks_sci_not'] = CustomFrame(
            self.frames['ticks'], border=False, dim=(3, 2))
        self.vars['sci_not'] = tk.IntVar(value=1)
        self.widgets['sci_not'] = tk.ttk.Checkbutton(
            self.frames['ticks_sci_not'], variable=self.vars['sci_not'],
            text='Sci. Notation')
        self.widgets['sci_not'].grid(row=0, column=0, sticky='w')
        tk.ttk.Label(self.frames['ticks_sci_not'], text="Axis:").grid(
            row=0, column=1, sticky='w')
        self.vars['axis_sci_notation'] = tk.StringVar(value="X")
        self.optmenus['axis_sci_notation'] = tk.ttk.OptionMenu(
            self.frames['ticks_sci_not'], self.vars['axis_sci_notation'],
            "X", "X", "Y", "Both")
        self.optmenus['axis_sci_notation'].config(width=10)
        self.optmenus['axis_sci_notation'].grid(row=0, column=2, sticky='w')
        self.vars['axis_sci_notation'].trace(
            'w', self.axis_sci_notation_callback)
        tk.ttk.Label(self.frames['ticks_sci_not'], text="Ord. of Mag.").grid(
            row=1, column=0, sticky='w')
        self.vars['sci_not_ord_mag_lower'] = tk.IntVar(
            value=self.value_dict['sci_not_ord_mag_lower'])
        self.vars['sci_not_ord_mag_upper'] = tk.IntVar(
            value=self.value_dict['sci_not_ord_mag_upper'])
        self.widgets['sci_not_ord_mag_lower'] = tk.ttk.Entry(
            self.frames['ticks_sci_not'],
            textvariable=self.vars['sci_not_ord_mag_lower'],
            width=5)
        self.widgets['sci_not_ord_mag_lower'].grid(
            row=1, column=1, sticky='e', padx=5)
        self.widgets['sci_not_ord_mag_upper'] = tk.ttk.Entry(
            self.frames['ticks_sci_not'],
            textvariable=self.vars['sci_not_ord_mag_upper'],
            width=5)
        self.widgets['sci_not_ord_mag_upper'].grid(
            row=1, column=2, sticky='w', padx=5)
        self.widgets['sci_not_ord_mag_lower'].bind(
            '<Return>', self.sci_not_callback)
        self.widgets['sci_not_ord_mag_upper'].bind(
            '<Return>', self.sci_not_callback)
        self.vars['sci_not'].trace('w', self.sci_not_callback)

        self.ticks_show_check_callback()

        self.frames['ticks_sci_not'].grid(
            row=5, column=0, columnspan=2, sticky='wnse')

        self.frames['ticks'].grid(row=row, column=1, sticky='wnse',
                                  padx=5, pady=5, rowspan=2)

        row += 1
        # legend
        self.frames['legend'] = GroupBox(
            self, text='Legend options', border=True, dim=(3, 1))

        tk.Label(self.frames['legend'], text='Anchor:').grid(row=0, column=0,
                                                             sticky='w')
        for i, key in enumerate(['_x', '_y']):
            key = 'legend_anchor' + key
            self.vars[key] = tk.DoubleVar(value=self.value_dict[key])
            self.widgets[key] = tk.ttk.Entry(
                self.frames['legend'], textvariable=self.vars[key], width=5)
            self.widgets[key].bind('<Return>', self.update_legend_opts)
            self.widgets[key].grid(row=0, column=i + 1, padx=2)

        self.frames['legend'].grid(row=row, column=0, sticky='wnse',
                                   padx=5, pady=5)

        # axis labels
        self.frames['label_pos'] = GroupBox(self, text="Label Positions",
                                            border=True, dim=(4, 2))
        tk.ttk.Label(self.frames['label_pos'], text='x label:').grid(
            row=0, column=0, columnspan=2)
        tk.ttk.Label(self.frames['label_pos'], text='y label:').grid(
            row=0, column=2, columnspan=2)
        i = 0
        for ax in ['x', 'y']:
            for dim in ['x', 'y']:
                key = ax + 'label_pos_' + dim
                self.vars[key] = tk.DoubleVar(value=self.value_dict[key])
                self.widgets[key] = tk.ttk.Entry(
                    self.frames['label_pos'], textvariable=self.vars[key],
                    width=5)
                self.widgets[key].bind('<Return>', lambda *args, c=ax:
                                       self.update_label_pos(*args, case=c))
                self.widgets[key].grid(row=1, column=i)
                i += 1
        self.frames['label_pos'].grid(row=row + 1, column=1, sticky='wnse',
                                      padx=5, pady=5)

        row += 1

        # line color
        self.frames['color'] = GroupBox(
            self, border=True, text='Color options', dim=(3, 1))

        self.vars['reverse_color'] = tk.IntVar(
            value=self.value_dict['reverse_color'])
        ttk.Button(self.frames['color'],
                   command=self.reverse_color_callback,
                   text='Reverse cycle').grid(row=0, column=0, sticky='w')
        tk.Label(self.frames['color'], text='Shift cycle by:').grid(
            row=0, column=1, sticky='w')
        self.vars['shift_color_cycle'] = tk.IntVar(
            value=self.value_dict['shift_color_cycle'])
        self.widgets['shift_color_cycle'] = tk.ttk.Entry(
            self.frames['color'], textvariable=self.vars['shift_color_cycle'],
            width=5)
        self.widgets['shift_color_cycle'].bind(
            '<Return>', lambda *args, update=True:
                self.update_color_opts(update_canvas=update))
        self.widgets['shift_color_cycle'].grid(row=0, column=2, sticky='w')

        self.vars['enable_custom_colorcycle'] = tk.IntVar(value=0)
        self.widgets['enable_custom_colorcycle'] = tk.ttk.Checkbutton(
            self.frames['color'],
            variable=self.vars['enable_custom_colorcycle'],
            text='Custom color cycle (hex)')
        self.vars['enable_custom_colorcycle'].trace(
            'w', self.enable_custom_color_callback)
        self.widgets['enable_custom_colorcycle'].grid(
            row=1, column=0, columnspan=3, sticky='w')
        with plt.style.context(self.figure.plotstyle):
            cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.vars['custom_colorcycle'] = tk.StringVar(value="")
        try:
            self.vars['custom_colorcycle'].set(", ".join(cycle))
        except Exception:
            pass
        self.widgets['custom_colorcycle'] = tk.ttk.Entry(
            self.frames['color'], textvariable=self.vars['custom_colorcycle'],
            width=25, state='disabled')
        self.widgets['custom_colorcycle'].bind(
            "<Return>", self.set_custom_color_cycle)
        self.widgets['custom_colorcycle'].grid(row=2, column=0, columnspan=3,
                                               sticky='we', padx=2, pady=2)

        self.frames['color'].grid(row=row, column=0, sticky='wnse',
                                  padx=5, pady=5)

        row += 1

        ttk.Button(self, text='Done', command=self.close_window).grid(
            row=row, column=0, columnspan=2,
            padx=5, pady=5)
        self.protocol("WM_DELETE_WINDOW", self.close_window)
        if controller:
            center_toplevel(self, controller)

    def enable_custom_color_callback(self, *args):
        if self.vars['enable_custom_colorcycle'].get():
            self.widgets['custom_colorcycle'].config(state='normal')
            self.set_custom_color_cycle()
        else:
            self.widgets['custom_colorcycle'].config(state='disabled')
            self.figure.set_plot_kwargs(color_cycle=None)
            self.figure.plot()

    def sci_not_callback(self, *args):
        kwargs = {}
        kwargs['style'] = 'sci' if self.vars['sci_not'].get() else 'plain'
        kwargs['scilimits'] = (self.vars['sci_not_ord_mag_lower'].get(
        ), self.vars['sci_not_ord_mag_upper'].get())
        if re.search('both', self.vars['axis_sci_notation'].get(), re.I):
            axis = ['x', 'y']
        else:
            axis = [self.vars['axis_sci_notation'].get().lower()]
        func = (self.figure.set_ticklabel_format
                if self.parent.vars['edit_all'].get()
                else self.figure.set_ticklabel_format_all)
        for ax in axis:
            func(axis=ax, update_canvas=False, **kwargs)
        self.figure.canvas.draw()

    def axis_sci_notation_callback(self, *args):
        ax = self.vars['axis_sci_notation'].get().lower()
        for i, lbl in enumerate(['sci_not_ord_mag_lower',
                                 'sci_not_ord_mag_upper']):
            self.vars[lbl].set(
                self.figure.ticklabel_format[
                    self.figure.current_axes_no][ax]['scilimits'][i])

    def close_window(self):
        self.save_widget_values()
        self.destroy()

    def ticks_show_check_callback(self, *args):
        if not self.vars['ticks_show'].get():
            default = (0, 0, 0, 0)
            state = 'disabled'
        else:
            default = (1, 0, 1, 0)
            state = 'normal'
        for i, key in enumerate(('bottom', 'top', 'left', 'right')):
            self.vars['ticks_' + key].set(default[i])
            self.widgets['ticks_' + key].config(state=state)
        self.update_ticks()

    def update_ticks(self, *args, update_canvas=True):
        tick_kwargs = {'which': self.vars['ticks_which'].get()}
        for key in ('bottom', 'top', 'left', 'right'):
            tick_kwargs[key] = self.vars['ticks_' + key].get()
            tick_kwargs['label' + key] = self.vars['ticks_' + key].get()
        tick_kwargs['labelsize'] = self.vars['ticks_labelsize'].get()
        tick_kwargs['width'] = self.vars['ticks_width'].get()

        if self.parent.vars['edit_all'].get():
            self.figure.set_tick_params_all(
                case=self.vars['ticks_which'].get(),
                update_canvas=update_canvas, **tick_kwargs)
        else:
            self.figure.set_tick_params(
                case=self.vars['ticks_which'].get(),
                update_canvas=update_canvas, **tick_kwargs)

    def set_framewidth(self, *args, update_canvas=True):
        if self.parent.vars['edit_all'].get():
            self.figure.set_frame_width_all(
                width=self.vars['frame_width'].get(),
                update_canvas=update_canvas)
        else:
            self.figure.set_frame_width(width=self.vars['frame_width'].get(),
                                        update_canvas=update_canvas)

    def save_widget_values(self):
        opts = {}
        for key, val in self.vars.items():
            opts[key] = val.get()
        if self.parent.vars['edit_all'].get():
            self.parent.adv_opts_vals = opts
        else:
            if not type(self.parent.adv_opts_vals) is list:
                self.parent.adv_opts_vals = [
                    self.value_dict for i in range(self.figure._num_subplots)]
            self.parent.adv_opts_vals[self.figure.current_axes_no] = opts

    def reverse_color_callback(self):
        self.reverse_color = not self.reverse_color
        self.update_color_opts()

    def update_color_opts(self, *args, update_canvas=True):
        color_order = {'reverse': self.reverse_color,
                       'ind': self.vars['shift_color_cycle'].get()}
        self.figure.set_plot_kwargs(color_order=color_order)
        if update_canvas:
            self.figure.plot()

    def update_legend_opts(self, *args, update_canvas=True):
        self.figure.set_legend(
            bbox_to_anchor=[self.vars['legend_anchor' + key].get()
                            for key in ['_x', '_y']],
            update_canvas=update_canvas)

    def set_custom_color_cycle(self, *args, update_canvas=True):
        self.figure.set_plot_kwargs(
            color_cycle=[
                s.strip()
                for s in self.vars['custom_colorcycle'].get().split(",")])
        try:
            self.figure.plot()
        except Exception:
            self.reset_color_cycle()

    def reset_color_cycle(self, *args):
        self.figure.set_plot_kwargs(color_cycle=None)
        self.figure.plot()
        with plt.style.context(self.figure.plotstyle):
            cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        try:
            self.vars['custom_colorcycle'].set(", ".join(cycle))
        except Exception:
            pass

    def update_grid_opts(self, *args, update_canvas=True):
        self.figure.set_grid(linestyle=self.vars['grid_style'].get(),
                             linewidth=self.vars['grid_width'].get(),
                             axis=self.vars['grid_axis'].get(),
                             update_canvas=update_canvas)

    def update_label_pos(self, *args, case='x', update_canvas=True):
        x_pos = self.vars[case + 'label_pos_x'].get()
        y_pos = self.vars[case + 'label_pos_y'].get()
        self.figure.set_label_pos(
            x_pos, y_pos, case=case, update_canvas=update_canvas)

    def update_all(self, *args):
        self.update_grid_opts(update_canvas=False)
        self.update_legend_opts(update_canvas=False)
        self.update_color_opts(update_canvas=False)
        self.update_ticks(update_canvas=False)
        self.set_custom_color_cycle(update_canvas=True)

    def update_parent(self, *args, **kwargs):
        return


# %%
class PolyCollectEditor(tk.Toplevel):
    def __init__(self, parent, figure, controller=None):
        tk.Toplevel.__init__(self, parent)
        container = tk.Frame(self)
        self.parent = parent
        self.figure = figure
        self.controller = controller
        self.vars = {}
        self.widgets = {}
        self.output = None

        coll = self.figure.get_axes().collections

        self.current_collect = tk.IntVar(value=1)
        row = 0
        self.place_widget(container, "current_collect",
                          "Collection No.:", *[str(i + 1)
                                               for i in range(len(coll))],
                          widget_type="opt", var=self.current_collect,
                          command=self.get_props_from_fig, row=row)
        row += 1
        self.place_widget(container, "alpha", "Alpha:",
                          vartype=tk.DoubleVar,
                          row=row)
        row += 1
        self.place_widget(container, "facecolor", "Face color:",
                          row=row)
        row += 1
        self.place_widget(container, "edgecolor", "Edge color:",
                          row=row)
        row += 1
        self.place_widget(container, "linestyle", "Line style:",
                          "-", "--", "-.", ":",
                          widget_type="opt",
                          row=row)
        row += 1
        self.place_widget(container, "linewidth", "Line width:",
                          vartype=tk.DoubleVar,
                          row=row)
        row += 1
        self.place_widget(container, "zorder", "Z-Order:",
                          vartype=tk.IntVar,
                          row=row)
        row += 1
        self.place_widget(container, "label", "Label:",
                          vartype=tk.StringVar, row=row)
        row += 1
        self.get_props_from_fig()

        tk.ttk.Button(container, text='Apply and Close',
                      command=self.applyandclose).grid(
            row=row, column=0,
            columnspan=2)

        container.grid(row=0, column=0, sticky='wnse', padx=10, pady=10)

    def get_props_from_fig(self, *args, j=None):
        if j is None:
            j = self.current_collect.get() - 1
        coll = self.figure.get_axes().collections[j]
        for key in ['alpha', 'zorder', 'label']:
            try:
                exec(key.join(["self.vars[key].set(coll.get_", "())"]))
            except Exception as e:
                print(e)
        self.vars['facecolor'].set(mpl.colors.to_hex(coll.get_facecolor()[0]))
        self.vars['edgecolor'].set(mpl.colors.to_hex(coll.get_edgecolor()[0]))
        self.vars['linewidth'].set(coll.get_linewidth()[0])
        ls = coll.get_linestyle()[0][1]
        if ls is None:
            self.vars['linestyle'].set('-')
        else:
            ls_dct = {
                1.0: ':',
                3.7: '--',
                6.4: '-.'}
            try:
                self.vars['linestyle'].set(ls_dct[ls[0]])
            except Exception:
                self.vars['linestyle'].set('-')

    def applyandclose(self):
        self.set_properties()
        self.output = []
        lines = self.figure.get_axes().get_lines()
        for j in range(len(lines)):
            self.get_props_from_fig(j=j)
            dct = {}
            for key, val in self.vars.items():
                try:
                    dct[key] = val.get()
                except Exception:
                    dct[key] = None
            self.output.append(dct)
        self.destroy()

    def set_properties(self, *args):
        props = {}
        for key, var in self.vars.items():
            props[key] = var.get()
        self.figure.set_collection_properties(j=self.current_collect.get() - 1,
                                              **props)

    def place_widget(self, parent, name, label, *widget_args,
                     vartype=tk.StringVar,
                     widget_type="entry", var=None,
                     row=0, sticky_lbl='w', sticky_wdgt='e',
                     padx=5, pady=5, command=None,
                     **widget_kwargs):
        if var is None:
            try:
                var = self.vars[name]
            except Exception:
                self.vars[name] = vartype()
                var = self.vars[name]
        if command is None:
            command = self.set_properties
        if widget_type == 'entry':
            self.widgets[name] = tk.ttk.Entry(
                parent, *widget_args, textvariable=var, **widget_kwargs)
            if command:
                self.widgets[name].bind('<Return>', command)
        elif widget_type == 'opt':
            self.widgets[name] = tk.ttk.OptionMenu(
                parent, var, var.get(), *widget_args, command=command,
                **widget_kwargs)
        else:
            return
        tk.ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky=sticky_lbl, padx=padx, pady=pady)
        self.widgets[name].grid(row=row, column=1,
                                sticky=sticky_wdgt, padx=padx, pady=pady)


# %%
class EditorWindow(tk.Toplevel):
    def __init__(self, parent, figure, editor, controller=None,
                 reset_function=None):
        tk.Toplevel.__init__(self, parent)
        self.output = None
        self.parent = parent
        self.figure = figure

        tk.ttk.Button(self, text="Apply & Close",
                      command=self.applyandclose).grid(
                          row=1, column=0, padx=5, pady=5)
        if reset_function is None:
            self.reset = idle_function
            cspan_editor = 1
        else:
            self.reset = reset_function
            tk.ttk.Button(self, text="Reset", command=self.reset).grid(
                row=1, column=1, padx=5, pady=5)
            cspan_editor = 2

        self.editor = editor(self, figure, dim=(cspan_editor, 2))
        self.editor.grid(sticky='wnse', row=0, column=0, padx=5, pady=5,
                         columnspan=cspan_editor)
        if controller:
            center_toplevel(self, controller)

    def applyandclose(self):
        self.output = self.editor.get_output()
        self.destroy()


class LegendEditorWindow(EditorWindow):
    def __init__(self, parent, figure, controller=None):
        EditorWindow.__init__(self, parent, figure, LegendEditor,
                              controller=controller)

    def applyandclose(self):
        self.output = self.editor.get_output()
        self.destroy()


class Editor(CustomFrame):
    def __init__(self, parent, figure, **kwargs):
        CustomFrame.__init__(self, parent, **kwargs)
        self.figure = figure
        self.vars = {}
        self.widgets = {}

    def place_widget(self, parent, name, label, *widget_args,
                     vartype=tk.StringVar,
                     widget_type="entry", var=None,
                     value=None,
                     row=0, sticky_lbl='w', sticky_wdgt='e',
                     padx=5, pady=5, command=None,
                     **widget_kwargs):
        if var is None:
            try:
                var = self.vars[name]
            except Exception:
                self.vars[name] = vartype()
                var = self.vars[name]
                var.set(value)
        if command is None:
            command = self.default_callback
        if widget_type == 'entry':
            self.widgets[name] = tk.ttk.Entry(
                parent, *widget_args, textvariable=var, **widget_kwargs)
            if command:
                self.widgets[name].bind('<Return>', command)
        elif widget_type == 'opt':
            self.widgets[name] = tk.ttk.OptionMenu(
                parent, var, var.get(), *widget_args, command=command,
                **widget_kwargs)
        else:
            return
        tk.ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky=sticky_lbl, padx=padx, pady=pady)
        self.widgets[name].grid(row=row, column=1,
                                sticky=sticky_wdgt, padx=padx, pady=pady)

    def default_callback(self, *args, **kwargs):
        return


class LegendEditor(Editor):
    def __init__(self, parent, figure, **frame_kwargs):
        Editor.__init__(self, parent, figure, **frame_kwargs)
        self.figure.set_legend(entries=self.figure.get_legend(),
                               visible=True, update_canvas=False)

        self.selected_handle = tk.StringVar(value="All")
        self.vars['linewidth'] = tk.DoubleVar()
        self.vars['linestyle'] = tk.StringVar()
        self.vars['marker'] = tk.StringVar()
        self.vars['markersize'] = tk.DoubleVar()
        self.vars['alpha'] = tk.StringVar()
        self._get_handles()
        self.handle_select_callback()
        self.get_current_values()

        self.default_callback = self.set_handle_properties
        tk.ttk.OptionMenu(
            self, self.selected_handle, self.selected_handle.get(), "All",
            *np.arange(1, len(self._handles) + 1),
            command=self.handle_select_callback).grid(
                row=0, column=1, sticky='w')
        tk.ttk.Label(self, text="Edit handles:").grid(
            row=0, column=0, sticky='w')

        self.edit_opts = GroupBox(self, text="Properties", dim=(2, 5))
        row = 0
        self.place_widget(self.edit_opts, "linewidth", "Line width:",
                          row=row)
        row += 1
        self.place_widget(self.edit_opts, "linestyle", "Line style:",
                          row=row)
        row += 1
        self.place_widget(self.edit_opts, "marker", "Marker:", row=row)

        row += 1
        self.place_widget(self.edit_opts, "markersize", "Marker size:",
                          row=row)
        row += 1
        self.place_widget(self.edit_opts, "alpha", "Alpha:", row=row)

        self.edit_opts.grid(row=1, column=0, columnspan=2, sticky='wnse')

    def set_marker_properties(self, *args, update_canvas=True):
        for h in self._current_handles:
            getattr(h, "set_" + "markersize")(self.vars['markersize'].get())
        if update_canvas:
            self.figure.set_legend()

    def set_line_properties(self, *args, update_canvas=True):
        for h in self._current_handles:
            getattr(h, "set_" + "linewidth")(self.vars['linewidth'].get())
        if update_canvas:
            self.figure.set_legend()

    def set_handle_properties(self, *args, update_canvas=True):
        keys = [key for key in self.vars.keys() if key != 'alpha']
        alpha = self.vars['alpha'].get()
        if alpha.lower() == 'none':
            alpha = 1
        else:
            alpha = np.double(alpha)
        for h in self._current_handles:
            for key in keys:
                getattr(h, "set_" + key)(self.vars[key].get())
            h.set_alpha(alpha)
        if update_canvas:
            self.figure.set_legend()

    def handle_select_callback(self, *args):
        if self.selected_handle.get().lower() == "all":
            self._current_handles = [h for h in self._helpers]
        else:
            ind = int(self.selected_handle.get()) - 1
            self._current_handles = [self._helpers[ind]]

    def _get_handles(self):
        lines = self.figure.get_axes().get_lines()
        lgnd = self.figure.get_axes().get_legend()
        self._handles = [h for h in lgnd.legendHandles]
        self._lines = [line for line in lines
                       if not line.get_label() == "_nolegend_"]
        xlim = self.figure.get_axes().get_xlim()
        x = xlim[1] + np.abs(np.sum(xlim))
        y = np.abs(np.sum(self.figure.get_axes().get_ylim()))
        self._helpers = []
        props = list(self.vars.keys())
        props.extend(["color"])
        for i, line in enumerate(self._lines):
            if re.search("legendhelper", line.get_label(), re.I):
                self._helpers.append(line)
            else:
                kwargs = {}
                for prop in props:
                    kwargs[prop] = getattr(line, "get_" + prop)()
                helper, = self.figure.get_axes().plot(
                    x, y, label="_legendhelper" + str(i), **kwargs)
                self._helpers.append(helper)
                line.set_label('_nolegend_')
        self.figure.set_xlim(xlim, update_canvas=False)
        self.figure.set_legend(entries=self.figure.get_legend())

    def get_current_values(self):
        h = self._current_handles[0]
        for key, var in self.vars.items():
            var.set(getattr(h, "get_" + key)())

    def clear_helpers(self):
        for line in self.figure.get_axes().get_lines():
            if re.search("legendhelper", line.get_label(), re.I):
                line.remove()

    def get_output(self):
        return


# %%
class LineEditorWindow(EditorWindow):
    def __init__(self, parent, figure, controller=None):
        EditorWindow.__init__(self, parent, figure, LineEditor,
                              controller=controller, reset_function=self.reset)

    def reset(self, *args):
        self.figure.plot()
        self.get_lines_properties()
        self.controller.lines_properties = {}
        self.line_fill_alpha = {}


class LineEditor(CustomFrame):
    def __init__(self, parent, figure, **frame_kwargs):
        CustomFrame.__init__(self, parent, **frame_kwargs)
        self.figure = figure
        self.vars = {}

        self.current_line = tk.IntVar(value=1)
        self.fill_alpha = tk.DoubleVar(value=0.0)
        self.vars['linestyle'] = tk.StringVar(value='-')
        self.vars['linewidth'] = tk.DoubleVar(value=1.5)
        self.vars['color'] = tk.StringVar()
        self.vars['drawstyle'] = tk.StringVar(value='default')
        self.vars['alpha'] = tk.DoubleVar()
        self.vars['zorder'] = tk.IntVar()
        self.vars['label'] = tk.StringVar()

        self.vars['marker'] = tk.StringVar(value='o')
        self.vars['markersize'] = tk.DoubleVar()
        self.vars['fillstyle'] = tk.StringVar(value='full')
        self.vars['markerfacecolor'] = tk.StringVar()
        self.vars['markeredgecolor'] = tk.StringVar()
        self.vars['markeredgewidth'] = tk.DoubleVar()

        self.get_fill()

        self.output = None
        self.widgets = {}
        tk.ttk.Label(self, text="Line No.:").grid(row=0, column=0,
                                                  sticky='w')

        self.line_select = tk.ttk.OptionMenu(
            self, self.current_line, self.current_line.get(),
            *[i + 1 for i in range(len(self.figure.get_axes().get_lines()))])
        self.line_select.grid(row=0, column=1, sticky='w')
        self.line_options = GroupBox(self, text="Line Options", dim=(2, 5))
        row = 0
        self.place_widget(self.line_options, 'linestyle', 'Line style:',
                          '-', '--', ':', '-.', 'None',
                          widget_type='opt',
                          command=lambda *args: self.set_properties(),
                          row=row)
        row += 1
        self.place_widget(self.line_options, 'linewidth', 'Width:',
                          command=lambda *args: self.set_properties(),
                          width=8, row=row)
        row += 1
        self.place_widget(self.line_options, 'color', 'Color:',
                          row=row, command=lambda *args: self.set_properties(),
                          width=8)
        row += 1
        self.place_widget(self.line_options, 'drawstyle', 'Draw style:',
                          'default', 'steps', 'steps-pre', 'steps-mid',
                          'steps-post', widget_type='opt',
                          command=lambda *args: self.set_properties(),
                          row=row)
        row += 1
        self.place_widget(self.line_options, 'alpha', 'Alpha:',
                          command=lambda *args: self.set_properties(),
                          width=8, row=row)
        row += 1
        self.place_widget(self.line_options, 'zorder', 'Z-order',
                          command=lambda *args: self.set_properties(), width=8,
                          row=row)
        row += 1
        self.place_widget(self.line_options, 'label', 'Label',
                          command=lambda *args: self.set_properties(), width=8,
                          row=row)
        self.line_options.grid(row=1, column=0,
                               padx=5, pady=5,
                               sticky='wne')

        self.marker_options = GroupBox(self, text="Markers", dim=(2, 5))
        row = 0
        self.place_widget(self.marker_options, 'marker', "Style:",
                          "None", "o", "x", "v", "^", "<", widget_type="opt",
                          sticky_wdgt='e', row=row)
        row += 1
        tk.ttk.Label(self.marker_options, text="Adv. style:").grid(
            row=row, column=0, sticky='w', padx=5)
        self.widgets['adv_marker'] = tk.ttk.Entry(
            self.marker_options, textvariable=self.vars['marker'], width=5)
        self.widgets['adv_marker'].bind('<Return>', self.set_adv_marker)
        self.widgets['adv_marker'].grid(row=row, column=1,
                                        sticky='w', padx=2)

        row += 1
        self.place_widget(self.marker_options, "markersize", "Size:",
                          row=row, width=5)
        row += 1
        self.place_widget(self.marker_options, "fillstyle", "Fillstyle:",
                          'full', 'left', 'right', 'bottom', 'top', 'none',
                          widget_type='opt', row=row)
        row += 1
        self.place_widget(self.marker_options, "markerfacecolor", "Facecolor:",
                          row=row, width=8)
        row += 1
        self.place_widget(self.marker_options, "markeredgewidth", "Edgewidth:",
                          row=row, width=5)
        row += 1
        self.place_widget(self.marker_options, "markeredgecolor", "Edgecolor:",
                          row=row, width=8)
        row += 1

        self.marker_options.grid(row=1, column=1, rowspan=2,
                                 padx=5, pady=5, sticky='wnse')

        self.effect_options = GroupBox(self, text="Effects",
                                       dim=(2, 5))
        self.fill_options = GroupBox(
            self.effect_options, text="Fill", dim=(2, 3))
        row = 0
        self.place_widget(self.fill_options, 'fillalpha', "Alpha:",
                          var=self.fill_alpha, command=self.set_fill,
                          row=row, width=5)
        self.fill_options.grid(row=0, column=0,
                               sticky='wnse', padx=2, pady=2)
        self.effect_options.grid(row=2, column=0,
                                 sticky='wne', padx=5, pady=5)

        self.current_line.trace('w', lambda *args, gf=True:
                                self.get_lines_properties(get_fill=gf))
        self.get_lines_properties()

    def place_widget(self, parent, name, label, *widget_args,
                     widget_type="entry", var=None,
                     row=0, sticky_lbl='w', sticky_wdgt='e',
                     padx=5, pady=5, command=None,
                     **widget_kwargs):
        if command is None:
            command = self.set_marker_properties
        elif command is False:
            command = None

        if var is None:
            try:
                var = self.vars[name]
            except Exception:
                return
        if widget_type == 'entry':
            self.widgets[name] = tk.ttk.Entry(
                parent, *widget_args, textvariable=var, **widget_kwargs)
            if command:
                self.widgets[name].bind('<Return>', command)
        elif widget_type == 'opt':
            self.widgets[name] = tk.ttk.OptionMenu(
                parent, var, var.get(), *widget_args, command=command,
                **widget_kwargs)
        else:
            return
        tk.ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky=sticky_lbl, padx=padx, pady=pady)
        self.widgets[name].grid(
            row=row, column=1, sticky=sticky_wdgt, padx=padx, pady=pady)

    def get_output(self):
        self.set_properties()
        output = []
        lines = self.figure.get_axes().get_lines()
        for j in range(len(lines)):
            self.get_lines_properties(j=j)
            dct = {}
            for key, val in self.vars.items():
                try:
                    dct[key] = val.get()
                except Exception:
                    dct[key] = None
            output.append(dct)
        return output

    def get_lines_properties(self, *args, j=None, properties="all",
                             get_fill=False):
        if j is None:
            j = self.current_line.get() - 1

        if properties == "all":
            properties = None
        else:
            properties = self._get_property_list(properties)
        props = self.figure.get_lines_properties(j=j, props=properties)
        for key, value in props.items():
            try:
                self.vars[key].set(value)
            except Exception as e:
                print(e)
        if get_fill:
            self.get_fill()

    def get_fill(self, *args):
        lines = self.figure.get_axes().get_lines()
        try:
            self.fill_alpha.set(
                self.figure.line_fill[self.figure.current_axes_no][
                    lines[self.current_line.get() - 1]].get_alpha())
        except Exception:
            self.fill_alpha.set(0)

    def set_fill(self, *args):
        line_num = self.current_line.get() - 1
        self.figure.clear_fill(lines=line_num,
                               update_canvas=False)
        self.figure.set_line_fill(alpha=self.fill_alpha.get(),
                                  lines=line_num)

    def _get_property_list(self, properties, case='lines', valid_props=None):
        if valid_props is None:
            valid_props = {'lines': ['linestyle', 'linewidth', 'color',
                                     'drawstyle', 'alpha', 'zorder', 'label'],
                           'marker': ['marker', 'markersize',
                                      'markerfacecolor',
                                      'markeredgewidth',
                                      'markeredgecolor',
                                      'fillstyle']}[case]
        if type(properties) is list:
            properties = [p for p in properties if p in valid_props]
        elif properties.lower() == 'all':
            properties = valid_props
        elif properties.lower() in valid_props:
            properties = [properties.lower()]
        else:
            properties = None
        return properties

    def set_marker_properties(self, *args, **kwargs):
        return self.set_properties(*args, case='marker', **kwargs)

    def set_adv_marker(self, *args, value=None, **kwargs):
        curr_marker = self.get_lines_properties(properties=['marker'])
        if value is not None:
            self.vars['marker'].set(value)
        try:
            self.set_marker_properties(properties='marker')
        except Exception:
            self.vars['marker'].set(curr_marker)

    def set_properties(self, *args, properties='all',
                       case='lines',
                       update_canvas=True):
        properties = self._get_property_list(properties, case)
        self._set_lines_properties(properties, update_canvas=update_canvas)

    def _set_lines_properties(self, properties, update_canvas=True):
        prop_dct = {}
        j = self.current_line.get() - 1
        curr_props = self.figure.get_lines_properties(j=j)
        for prop in properties:
            try:
                prop_dct[prop] = self.vars[prop].get()
            except Exception:
                self.vars[prop].set(curr_props[prop])
        try:
            errors = self.figure.set_lines_properties(
                j=j, update_canvas=update_canvas, **prop_dct)
        except Exception:
            self.figure.set_lines_properties(j=j, update_canvas=update_canvas,
                                             **curr_props)
            self.get_lines_properties(j=j, properties=properties)
        else:
            for key, val in errors.items():
                self.vars[key].set(val)


# %%
class LinePlotEditor(AxesEditor):
    def __init__(self, parent, figure, *plot_args, controller=None,
                 plot_func=None, include_fit=False, **plot_kwargs):
        AxesEditor.__init__(self, parent,
                            figure, *plot_args, controller=controller,
                            plot_func=plot_func, border=True,
                            dim=(2, 8), adv_opts=False, **plot_kwargs)
        row = self.row
        self.include_fit = include_fit
        self.fill_alpha = 0
        self.reverse_z = False

        self.frames['legend_opts'] = CustomFrame(self, border=True, dim=(2, 6))
        self.vars['show_legend'] = tk.IntVar(value=1)
        self.widgets['show_legend'] = tk.ttk.Checkbutton(
            self.frames['legend_opts'], text='Show Legend',
            variable=self.vars['show_legend'],
            command=self.toggle_legend)
        self.widgets['show_legend'].grid(row=0, column=0, sticky='w')
        tk.Label(self.frames['legend_opts'], text='Label font size:').grid(
            row=1, column=0, sticky='w')
        self.vars['legend_fontsize'] = tk.DoubleVar(value=13)
        self.widgets['legend_fontsize'] = tk.ttk.Entry(
            self.frames['legend_opts'],
            textvariable=self.vars['legend_fontsize'],
            width=7)
        self.widgets['legend_fontsize'].bind(
            '<Return>', lambda *args, ge=False:
                self.set_legend(*args, get_entry=ge))
        self.widgets['legend_fontsize'].grid(row=1, column=1, sticky='e')

        tk.ttk.Label(self.frames['legend_opts'],
                     text='Label line width:').grid(
                         row=2, column=0, sticky='w')
        self.vars['legend_linewidth'] = tk.DoubleVar(
            value=mpl.rcParams['lines.linewidth'])
        self.widgets['legend_linewidth'] = tk.ttk.Entry(
            self.frames['legend_opts'],
            textvariable=self.vars['legend_linewidth'], width=5)
        self.widgets['legend_linewidth'].bind(
            '<Return>', lambda *args, ge=False:
                self.set_legend(*args, get_entry=ge))
        self.widgets['legend_linewidth'].grid(row=2, column=1, sticky='e')

        self.vars['legend_loc'] = tk.StringVar(value='best')
        self.optmenus['legend_loc'] = tk.ttk.OptionMenu(
            self.frames['legend_opts'], self.vars['legend_loc'],
            self.vars['legend_loc'].get(), 'best', 'upper right',
            'upper left', 'lower left', 'lower right', 'center left',
            'center right', 'lower center', 'upper center', 'center',
            command=lambda *args: self.set_legend(*args, get_entry=False))
        tk.Label(self.frames['legend_opts'], text='Position:').grid(
            row=3, column=0, sticky='w')
        self.optmenus['legend_loc'].grid(row=3, column=1, sticky='w')

        tk.Label(self.frames['legend_opts'],
                 text='Set Legend Entries (comma sep.)').grid(
                     row=4, column=0, sticky='w', columnspan=2)
        self.vars['legend'] = tk.StringVar()
        self.widgets['legend'] = tk.ttk.Entry(
            self.frames['legend_opts'], textvariable=self.vars['legend'],
            width=20)
        self.widgets['legend'].bind('<Return>', self.set_legend)
        self.widgets['legend'].grid(row=5, column=0, padx=2, pady=2,
                                    sticky='w', columnspan=2)

        tk.ttk.Button(self.frames['legend_opts'], text="Custom",
                      command=self.open_legend_opts).grid(
                          row=6, column=0, columnspan=2, padx=2, pady=2)

        self.frames['legend_opts'].grid(
            row=row, column=0, sticky='wnse', padx=2, pady=2)
        row += 1

        self.frames['linestyle_opts'] = CustomFrame(
            self, border=True, dim=(1, 1))

        tk.Label(self.frames['linestyle_opts'],
                 text="Line Style:").grid(row=0, column=0, sticky='w')
        self.vars['linestyle'] = tk.StringVar(value='-')
        linestyles = ['-', '--', ':', '-.', None]
        self.optmenus['linestyle'] = tk.ttk.OptionMenu(
            self.frames['linestyle_opts'], self.vars['linestyle'],
            self.vars['linestyle'].get(), *linestyles,
            command=self.set_lines_properties)
        self.optmenus['linestyle'].grid(row=0, column=1, sticky='w')
        tk.Label(self.frames['linestyle_opts'],
                 text="Line Width:").grid(row=1, column=0, sticky='w')
        self.vars['linewidth'] = tk.DoubleVar(value=2)
        self.widgets['linewidth'] = tk.ttk.Entry(
            self.frames['linestyle_opts'], textvariable=self.vars['linewidth'],
            width=5)
        self.widgets['linewidth'].grid(row=1, column=1, sticky='w')
        self.widgets['linewidth'].bind('<Return>', self.set_lines_properties)

        tk.Label(self.frames['linestyle_opts'],
                 text='Marker Style:').grid(row=3, column=0, sticky='w')
        markerstyles = [None, 'o', 'x']
        self.vars['marker'] = tk.StringVar(value=None)
        self.optmenus['marker'] = tk.ttk.OptionMenu(
            self.frames['linestyle_opts'], self.vars['marker'],
            self.vars['marker'].get(), *markerstyles,
            command=self.set_lines_properties)
        self.optmenus['marker'].grid(row=3, column=1, sticky='w')

        tk.Label(self.frames['linestyle_opts'],
                 text='Marker Size:').grid(row=4, column=0, sticky='w')
        self.vars['markersize'] = tk.IntVar(value=12)
        self.widgets['markersize'] = tk.ttk.Entry(
            self.frames['linestyle_opts'],
            textvariable=self.vars['markersize'],
            width=5)
        self.widgets['markersize'].bind('<Return>', self.set_lines_properties)
        self.widgets['markersize'].grid(row=4, column=1, sticky='w')

        self.vars['fill_curve'] = tk.IntVar(value=0)
        self.vars['fill_curve_alpha'] = tk.StringVar(value="0.15")
        self.widgets['fill_curve'] = tk.ttk.Checkbutton(
            self.frames['linestyle_opts'], variable=self.vars['fill_curve'],
            text="Fill Curve(s)", command=self.fill_curve_callback)
        self.widgets['fill_curve'].grid(row=5, column=0, sticky='w')
        self.widgets['fill_curve_alpha'] = tk.ttk.Entry(
            self.frames['linestyle_opts'],
            textvariable=self.vars['fill_curve_alpha'], state='disabled',
            width=5)
        self.widgets['fill_curve_alpha'].bind('<Return>',
                                              self.fill_curve_callback)
        self.widgets['fill_curve_alpha'].grid(row=5, column=1, sticky='w')

        self.frames['line_color'] = GroupBox(
            self.frames['linestyle_opts'], text='Line Color', border=True,
            dim=(3, 3))

        self.vars['interpolate_colors'] = tk.IntVar(value=0)
        self.widgets['interpolate_colors'] = tk.ttk.Checkbutton(
            self.frames['line_color'],
            variable=self.vars['interpolate_colors'],
            text="Interpolate",
            command=self.interp_color_callback)
        self.widgets['interpolate_colors'].grid(row=0, column=0)
        self.color_interp_label = tk.ttk.Label(
            self.frames['line_color'], text='Interp. Steps:')
        self.color_interp_label.grid(row=0, column=1, sticky='e')
        self.vars['color_interp_val'] = tk.IntVar(value=1)
        self.widgets['color_interp_val'] = tk.ttk.Entry(
            self.frames['line_color'],
            textvariable=self.vars['color_interp_val'],
            width=5)
        self.widgets['color_interp_val'].bind(
            '<Return>', self.interp_color_callback)
        self.widgets['color_interp_val'].grid(row=0, column=2, sticky='w')
        if self.include_fit:
            self.add_fit_opts(row=1)

        self.frames['line_color'].grid(row=6, column=0, columnspan=2,
                                       sticky='wnse', padx=2, pady=2)
        self.frames['linestyle_opts'].grid(row=row, column=0, sticky='wnse',
                                           padx=2, pady=2)
        row += 1

        tk.ttk.Button(self, text="Edit indiv. lines",
                      command=self.open_line_edit).grid(row=row, column=0)
        row += 1
        tk.ttk.Button(self, text="Edit collections/fill",
                      command=self.open_collect_edit).grid(row=row, column=0)
        row += 1

        row = self.place_adv_opts_button(row=row, column=0)

        if len(self.figure.axes) == 1:
            ttk.Button(self, text="Reverse Plot Order",
                       command=self.revert_order).grid(
                           row=row, column=0, padx=5, pady=5)
        self.row = row + 1
        self.lines_properties = {}
        self.collection_properties = {}

    def open_line_edit(self, *args):
        win = LineEditorWindow(self, self.figure, controller=self.controller)
        self.wait_window(win)
        if win.output is not None:
            self.lines_properties[self.figure.current_axes_no] = win.output

    def open_collect_edit(self, *args):
        coll = self.figure.get_axes().collections
        if len(coll) > 0:
            win = PolyCollectEditor(
                self, self.figure, controller=self.controller)
            self.wait_window(win)
            if win.output is not None:
                self.collection_properties[
                    self.figure.current_axes_no] = win.output

    def open_legend_opts(self, *args):
        self.vars['show_legend'].set(1)
        LegendEditorWindow(self, self.figure, controller=self.controller)

    def _set_indiv_lines_properties(self, *args, update_canvas=True):
        errors = {}
        for axnum, lines in self.lines_properties.items():
            errors[axnum] = {}
            for j, props in enumerate(lines):
                errors[axnum][j] = self.figure.set_lines_properties(
                    i=axnum, j=j, update_canvas=False, **props)
        if update_canvas:
            self.figure.canvas.draw()

    def _set_collect_properties(self, *args, update_canvas=True):
        errors = {}
        for axnum, lines in self.collection_properties.items():
            errors[axnum] = {}
            for j, props in enumerate(lines):
                errors[axnum][j] = self.figure.set_collect_properties(
                    i=axnum, j=j, update_canvas=False, **props)
        if update_canvas:
            self.figure.canvas.draw()

    def revert_order(self):
        self.reverse_z = not self.reverse_z
        self.update_plot()

    def add_fit_opts(self, row=6, default_color='Black'):
        self.vars['fit_color'] = tk.StringVar(value=default_color)
        self.widgets['fit_color'] = tk.ttk.Entry(
            self.frames['line_color'], textvariable=self.vars['fit_color'],
            width=10)
        self.widgets['fit_color'].bind('<Return>', self.update_plot)
        self.widgets['fit_color'].grid(row=row, column=1, columnspan=2,
                                       sticky='w')
        tk.Label(self.frames['line_color'], text="Fit Color:").grid(
            row=row, column=0, sticky='w')
        tk.ttk.Label(self.frames['line_color'], text='Fit Comp. Alpha:').grid(
            row=row + 1, column=0, sticky='w')
        self.vars['fit_comp_alpha'] = tk.StringVar(value="0")
        self.widgets['fit_comp_alpha'] = tk.ttk.Entry(
            self.frames['line_color'],
            textvariable=self.vars['fit_comp_alpha'],
            width=5)
        self.widgets['fit_comp_alpha'].bind('<Return>', self.update_plot)
        self.widgets['fit_comp_alpha'].grid(row=row + 1, column=1, sticky='w')

    def fill_curve_callback(self, *args, update_canvas=True):
        self.fill_alpha = bool(self.vars['fill_curve'].get())
        if self.fill_alpha:
            self.widgets['fill_curve_alpha'].config(state='normal')
            alpha = self.vars['fill_curve_alpha'].get().replace(
                " ", "").split(",")
            try:
                self.fill_alpha = np.double(alpha)
                if len(self.fill_alpha) < 2:
                    self.fill_alpha = self.fill_alpha[0]
                else:
                    alpha = []
                    for a in self.fill_alpha:
                        if a < 0:
                            alpha.append("0")
                        elif a > 1:
                            alpha.append("1")
                        else:
                            alpha.append(str(a))
                    self.vars['fill_curve_alpha'].set(",".join(alpha))
            except Exception:
                self.fill_alpha = 0
                self.vars['fill_curve_alpha'].set("0")
                return
        else:
            self.widgets['fill_curve_alpha'].config(state='disabled')
            self.fill_alpha = 0
        self.set_fill(update_canvas=update_canvas)

    def interp_color_callback(self, *args):
        if self.vars['interpolate_colors'].get():
            self.widgets['color_interp_val'].config(state='normal')
            self.color_interp_label.config(state='normal')
        else:
            self.widgets['color_interp_val'].config(state='disabled')
            self.color_interp_label.config(state='disabled')
        self.update_plot()

    def add_plot_ci_option(self, value=1):
        self.vars['plot_ci'] = tk.IntVar(value=value)
        self.widgets['plot_ci'] = tk.ttk.Checkbutton(
            self, variable=self.vars['plot_ci'], text="Plot conf. int.",
            command=self.plot_ci_callback)
        self.widgets['plot_ci'].grid(row=self.row, column=0, sticky='w')

    def plot_ci_callback(self, *args):
        plot_ci = bool(self.vars['plot_ci'].get())
        if self.vars['edit_all'].get():
            self.figure.set_plot_kwargs_all(plot_fit_error=plot_ci)
            self.figure.plot_all()
        else:
            self.figure.set_plot_kwargs(plot_fit_error=plot_ci)
            self.figure.plot()

    def toggle_legend(self, *args, update_canvas=True):
        if self.vars['edit_all'].get():
            self.figure.set_legend_visibility_all(
                case=self.vars['show_legend'].get(),
                update_canvas=update_canvas)
        else:
            self.figure.set_legend_visibility(
                case=self.vars['show_legend'].get(),
                update_canvas=update_canvas)

    def get_options_from_figure(self, i):
        # legend options
        self._get_fig_opts(i)
        if self.vars['edit_all'].get():
            j = self.figure.current_axes_no
        else:
            j = i
        self.vars['show_legend'].set(self.figure.legend_visible[j])
        lg_kwargs = self.figure.legend_kwargs[j]
        if 'fontsize' in lg_kwargs.keys():
            self.vars['legend_fontsize'].set(lg_kwargs['fontsize'])
        if 'loc' in lg_kwargs.keys():
            self.vars['legend_loc'].set(lg_kwargs['loc'])
        try:
            legend_texts = self.figure.get_legend(i=j)
        except Exception:
            pass
        else:
            self.vars['legend'].set(','.join(legend_texts))
        # mpl rc parameters
        for key in ('linestyle', 'linewidth', 'marker', 'markersize'):
            if key in self.figure.plot_kwargs[i].keys():
                self.vars[key].set(self.figure.plot_kwargs[i][key])
            else:
                self.vars[key].set(mpl.rcParams['.'.join(['lines', key])])

    def set_legend(self, *args, get_entry=True, update_canvas=True):
        self.figure.set_legend_kwargs(
            fontsize=self.vars['legend_fontsize'].get(),
            loc=self.vars['legend_loc'].get(),
            bbox_to_anchor=None)
        if get_entry:
            self.figure.set_legend(
                entries=[s.strip()
                         for s in self.vars['legend'].get().split(",")],
                update_canvas=True,
                linewidth=self.vars['legend_linewidth'].get())
            self.vars['show_legend'].set(1)
        else:
            self.figure.set_legend(
                entries=self.figure.get_legend(),
                linewidth=self.vars['legend_linewidth'].get())

    def set_lines_properties(self, *args, properties='all',
                             update_canvas=True):
        if not type(properties) is list:
            if properties == 'all':
                properties = ['linestyle', 'linewidth', 'marker',
                              'markersize']
            else:
                properties = [properties]
        props = {}
        for key in properties:
            props[key] = self.vars[key].get()
        
        self.figure.set_for_all_axes(self.figure.set_lines_properties_all,
                                     update_canvas=update_canvas, 
                                     exclude_tag='fit|_nolegend',
                                     **props)

    def set_fill(self, *args, update_canvas=True):
        try:
            self.figure.clear_fill(lines="all", update_canvas=False)
        except ValueError:
            pass
        except Exception:
            raise
        lines = self.figure.get_lines_list(self.figure.current_axes_no)
        i = 0
        for line in lines:
            if re.search('fit|nolegend', line.get_label()):
                lines.pop(i)
            else:
                i += 1
        self.figure.set_line_fill(lines=lines,
                                  alpha=self.fill_alpha,
                                  update_canvas=update_canvas)

    # redefining inherited function specifically for line plots
    def update_plot(self, *args):
        plot_kwargs = {'linestyle': self.vars['linestyle'].get(),
                       'linewidth': self.vars['linewidth'].get(),
                       'marker': self.vars['marker'].get(),
                       'markersize': self.vars['markersize'].get(),
                       'fill': self.fill_alpha,
                       'reverse_z': self.reverse_z}
        if self.include_fit:
            try:
                if not self.vars['fit_color'].get()[0] == '#':
                    color = self.vars['fit_color'].get().lower()
                else:
                    color = self.vars['fit_color'].get()
            except Exception:
                color = 'black'
                include = False
            else:
                include = not color == 'none'
            plot_kwargs['include_fit'] = include
            plot_kwargs['fit_comp_alpha'] = [
                np.double(s) for s in
                self.vars['fit_comp_alpha'].get().replace(" ", "").split(",")]
            fit_kwargs = {'color': color}
        else:
            fit_kwargs = {}
            plot_kwargs['fit_comp_alpha'] = 0
        if self.vars['interpolate_colors'].get():
            plot_kwargs['interpolate_colors'] = int(
                self.vars['color_interp_val'].get())
        else:
            plot_kwargs['interpolate_colors'] = False
        legend_kwargs = {'fontsize': self.vars['legend_fontsize'].get(),
                         'loc': self.vars['legend_loc'].get()}
        if self.vars['edit_all'].get():
            self.figure.set_plot_kwargs_all(**plot_kwargs)
            self.figure.set_fit_kwargs_all(**fit_kwargs)
            self.figure.set_for_all_axes(
                self.figure.set_legend_kwargs, **legend_kwargs)
            self.figure.plot_all(update_canvas=False)
        else:
            self.figure.set_plot_kwargs(**plot_kwargs)
            self.figure.set_fit_kwargs(**fit_kwargs)
            self.figure.set_legend_kwargs(**legend_kwargs)
            self.figure.plot(update_canvas=False)
        self._set_indiv_lines_properties(update_canvas=True)


# %%
class ToplevelFigure(tk.Toplevel):
    def __init__(self, parent, controller=None, window_title=None,
                 **fig_frame_kwargs):
        tk.Toplevel.__init__(self, parent)
        if controller is not None:
            move_toplevel_to_default(self, controller)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.fr = FigFrame(self, controller=controller, **fig_frame_kwargs)
        self.parent = parent
        self.controller = controller
        self.fr.grid(row=0, column=0, sticky='wnse')

        self.menubar = tk.Menu(self)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Figure", menu=self.filemenu)
        self.filemenu.add_command(
            label="Load", command=self.fr.open_fig_obj)
        self.filemenu.add_command(
            label="Save", command=self.fr.save_figure_obj)

        tk.Tk.config(self, menu=self.menubar)
        if window_title:
            self.title(window_title)
        if controller is not None:
            center_toplevel(self, controller)


# %%
class ColorMapEditor(tk.Frame):
    def __init__(self, parent, figure, contrast_slider=None, horizontal=True,
                 auto_callback=None, settings_dict=None, update_function=None,
                 num_sub_plots=1, plot_index=0,
                 title="Color Map"):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.frames = {}
        self.optmenus = {}
        self.widgets = {}
        self.entries = {}
        self.buttons = {}
        self.vars = {}
        self.figure = figure
        if contrast_slider is None:
            self.contrast_slider = BlankObject()
            self.contrast_slider.set = idle_function()
        else:
            self.contrast_slider = contrast_slider

        if update_function is None:
            self.update_function = self.idle_function
        else:
            self.update_function = update_function

        self.frames['main'] = GroupBox(
            self, dim=(3, 2) if horizontal else (1, 4), text=title)
        self.frames['limits'] = GroupBox(
            self.frames['main'], dim=(5, 2) if horizontal else(2, 5),
            text='Color limits')
        if horizontal:
            self.widget_pos = {'cscheme': {'row': 0, 'column': 0},
                               'cmap_sym': {'row': 0, 'column': 1},
                               'invert_color': {'row': 1, 'column': 1},
                               'cmap_center_zero': {'row': 2, 'column': 1},
                               'limits': {'row': 0, 'column': 2},
                               'clim_sym': {'row': 0, 'column': 0},
                               'auto_button': {'row': 0, 'column': 4},
                               'lower_lbl': {'row': 1, 'column': 0},
                               'clim_low': {'row': 1, 'column': 1},
                               'upper_lbl': {'row': 1, 'column': 2},
                               'clim_up': {'row': 1, 'column': 3}}
        else:
            self.widget_pos = {'cscheme': {'row': 1, 'column': 0},
                               'cmap_sym': {'row': 2, 'column': 0},
                               'invert_color': {'row': 3, 'column': 0},
                               'cmap_center_zero': {'row': 4, 'column': 0},
                               'limits': {'row': 5, 'column': 0},
                               'clim_sym': {'row': 0, 'column': 0},
                               'auto_button': {'row': 3, 'column': 0},
                               'lower_lbl': {'row': 1, 'column': 0},
                               'clim_low': {'row': 1, 'column': 1},
                               'upper_lbl': {'row': 2, 'column': 0},
                               'clim_up': {'row': 2, 'column': 1}}
        if settings_dict is None:
            c_obj = self.figure.color_obj[plot_index]
            self.vars['lower_clim'] = tk.DoubleVar(value=c_obj.clims[0])
            self.vars['upper_clim'] = tk.DoubleVar(value=c_obj.clims[1])
            self.vars['cmap_sym'] = tk.IntVar(value=int(c_obj.cmap_sym))
            self.vars['clim_sym'] = tk.IntVar(value=int(c_obj.clim_sym))
            self.vars['color_scheme'] = tk.StringVar(value=c_obj.cmap_name)
            self.vars['invert_color'] = tk.IntVar(
                value=int(c_obj.invert_cmap))
            self.vars['cmap_center_zero'] = tk.IntVar(
                value=int(c_obj.map_center_zero))
        else:
            self.vars['lower_clim'] = tk.DoubleVar(
                value=settings_dict['plot_kwargs']['vmin'])
            self.vars['upper_clim'] = tk.DoubleVar(
                value=settings_dict['plot_kwargs']['vmax'])
            self.vars['cmap_sym'] = tk.IntVar(
                value=1 if settings_dict['sym_map'] else 0)
            self.vars['clim_sym'] = tk.IntVar(
                value=1 if settings_dict['sym_lim'] else 0)
            self.vars['color_scheme'] = tk.StringVar(
                value=settings_dict['cmap_name'])
            self.vars['invert_color'] = tk.IntVar(
                value=int(settings_dict['invert_cmap']))
            self.vars['cmap_center_zero'] = tk.IntVar(
                value=int(settings_dict['map_center_zero']))

        self.set_axes_to_edit()

        self.optmenus['cscheme'] = tk.ttk.OptionMenu(
            self.frames['main'], self.vars['color_scheme'],
            self.vars['color_scheme'].get(),
            *self.figure.color_obj[plot_index].cmap_dict.keys(),
            command=self.change_color_scheme)
        self.optmenus['cscheme'].config(width=12)
        self.optmenus['cscheme'].grid(
            **self.widget_pos['cscheme'],
            sticky=tk.E,
            rowspan=2 if horizontal else 1)
        self.widgets['cmap_sym'] = tk.ttk.Checkbutton(
            self.frames['main'], text="symmetric map",
            variable=self.vars['cmap_sym'], command=self.change_color_scheme)
        self.widgets['cmap_sym'].grid(
            **self.widget_pos['cmap_sym'],
            sticky=tk.W)

        self.widgets['invert_color'] = tk.ttk.Checkbutton(
            self.frames['main'], text="invert map",
            variable=self.vars['invert_color'],
            command=self.change_color_scheme)
        self.widgets['invert_color'].grid(
            **self.widget_pos['invert_color'],
            sticky=tk.W)
        self.widgets['cmap_center_zero'] = tk.ttk.Checkbutton(
            self.frames['main'], text="center zero",
            variable=self.vars['cmap_center_zero'],
            command=self.center_zero_callback)
        self.widgets['cmap_center_zero'].grid(
            **self.widget_pos['cmap_center_zero'],
            sticky='w')

        self.frames['limits'].grid(
            **self.widget_pos['limits'],
            rowspan=3 if horizontal else 1, sticky='nswe', pady=5, padx=5)

        self.widgets['clim_sym'] = tk.ttk.Checkbutton(
            self.frames['limits'], text='symmetric limits',
            variable=self.vars['clim_sym'], command=self.clim_callback)
        self.widgets['clim_sym'].grid(
            **self.widget_pos['clim_sym'],
            columnspan=3,
            sticky='w')

        for key in ('cmap_center_zero', 'cmap_sym'):
            self.widgets[key].config(state='disabled')

        if auto_callback is not None:
            self.widgets['auto_button'] = ttk.Button(
                self.frames['limits'], text="auto",
                command=auto_callback)
            self.widgets['auto_button'].grid(
                    **self.widget_pos['auto_button'],
                    sticky=tk.W+tk.E,
                    padx=5, pady=2, rowspan=2,
                    columnspan=1 if horizontal else 2)
        tk.Label(self.frames['limits'], text="lower:").grid(
            **self.widget_pos['lower_lbl'],
            sticky=tk.E if horizontal else tk.W)
        self.entries['clim_low'] = tk.ttk.Entry(
            self.frames['limits'], textvariable=self.vars['lower_clim'],
            width=8)
        self.entries['clim_low'].grid(
            **self.widget_pos['clim_low'],
            ipady=1, pady=2,
            sticky=tk.E if horizontal else tk.W, padx=5)
        self.entries['clim_low'].bind(
            '<Return>', lambda event: self.clim_callback(event, case='lower'))

        tk.Label(self.frames['limits'], text="upper:").grid(
            **self.widget_pos['upper_lbl'],
            sticky=tk.E if horizontal else tk.W)
        self.entries['clim_up'] = tk.ttk.Entry(
            self.frames['limits'], textvariable=self.vars['upper_clim'],
            width=8)
        self.entries['clim_up'].grid(
            **self.widget_pos['clim_up'],
            ipady=1, pady=2,
            sticky=tk.E if horizontal else tk.W, padx=5)
        self.entries['clim_up'].bind(
            '<Return>', lambda event: self.clim_callback(event, case='upper'))

        self.frames['main'].grid(sticky='wnse', row=0, column=0)
        for wdgt in self.entries.values():
            wdgt.config(justify=tk.RIGHT)

    def idle_function(self, *args, **kwargs):
        return

    def clim_callback(self, *args, case='lower'):
        if self.vars['clim_sym'].get():
            self.entries['clim_low'].config(state='disabled')
            for key in ('cmap_center_zero', 'cmap_sym'):
                self.vars[key].set(1)
                self.widgets[key].config(state='disabled')
            if case == 'lower' and self.vars['lower_clim'].get() < 0:
                clims = [self.vars['lower_clim'].get(
                ), -self.vars['lower_clim'].get()]
                self.vars['upper_clim'].set(-self.vars['lower_clim'].get())
            elif case == 'upper' and self.vars['upper_clim'].get() > 0:
                clims = [-self.vars['upper_clim'].get(),
                         self.vars['upper_clim'].get()]
                self.vars['lower_clim'].set(-self.vars['upper_clim'].get())
            else:
                c = np.max(
                    np.abs([self.vars['upper_clim'].get(),
                            self.vars['lower_clim'].get()]))
                clims = [-c, c]
                self.vars['lower_clim'].set(-c)
                self.vars['upper_clim'].set(c)
            self.update_color_obj(clims=clims, cmap=None,
                                  clim_sym=self.vars['clim_sym'].get())
            self.update_image()
        else:
            self.entries['clim_low'].config(state='normal')
            for key in ('cmap_center_zero', 'cmap_sym'):
                self.widgets[key].config(state='normal')
            if self.vars['lower_clim'].get() >= self.vars['upper_clim'].get():
                clims = [-np.abs(self.vars['lower_clim'].get()),
                         np.abs(self.vars['lower_clim'].get())]
                self.vars['lower_clim'].set(clims[0])
                self.vars['upper_clim'].set(clims[1])
                messagebox.showerror(
                    message="Lower color limit must be smaller than upper.")
            else:
                clims = [self.vars['lower_clim'].get(
                ), self.vars['upper_clim'].get()]
            self.update_color_obj(clims=clims, cmap=None,
                                  clim_sym=self.vars['clim_sym'].get())
            self.update_image()
        try:
            self.contrast_slider.set(0.0)
        except Exception:
            pass

    def set_axes_to_edit(self, i=None):
        self.update_all_plots = False
        try:
            if i is None:
                self.update_color_obj = self.figure.update_color_obj
            elif i == "all":
                self.update_color_obj = (
                    lambda **kwargs: self.figure.set_for_all_axes(
                        self.figure.update_color_obj, **kwargs))
                self.update_all_plots = True
            elif i < self.figure.get_num_subplots():
                self.update_color_obj = (
                    lambda **kwargs: self.figure.update_color_obj(
                        i=i, **kwargs))
            else:
                self.update_color_obj = self.figure.update_color_obj
        except Exception:
            self.update_color_obj = self.figure.update_color_obj

    def center_zero_callback(self, *args):
        if not self.vars['cmap_center_zero'].get():
            self.vars['cmap_sym'].set(1)
            self.widgets['cmap_sym'].config(state='disabled')
        else:
            self.widgets['cmap_sym'].config(state='normal')
        self.change_color_scheme()

    def change_color_scheme(self, *args):
        self.update_color_obj(
            cmap=self.vars['color_scheme'].get(),
            cmap_sym=bool(self.vars['cmap_sym'].get()),
            invert_cmap=self.vars['invert_color'].get(),
            cmap_center_zero=self.vars['cmap_center_zero'].get())
        self.update_image()

    def get_color_obj_info(self, c_obj):
        self.vars['color_scheme'].set(value=c_obj.cmap_name)
        self.vars['invert_color'].set(value=int(c_obj.invert_cmap))
        self.vars['lower_clim'].set(value=c_obj.clims[0])
        self.vars['upper_clim'].set(value=c_obj.clims[1])
        self.vars['clim_sym'].set(value=int(c_obj.clim_sym))
        self.vars['cmap_sym'].set(value=int(c_obj.cmap_sym))

    def update_image(self, clims=True, cmap=True, update_canvas=True):
        funcs = []
        if clims:
            funcs.append(self.figure.set_clim)
        if cmap:
            funcs.append(self.figure.set_cmap)
        if self.update_all_plots:
            for func in funcs:
                self.figure.set_for_all_axes(func, update_canvas=update_canvas)
        else:
            for func in funcs:
                func(update_canvas=False)
            if update_canvas:
                self.figure.canvas.draw()

    def update_plot(self, *args, replot=False, **kwargs):
        if self.update_all_plots:
            if replot:
                self.figure.plot_all()
            else:
                self.figure.set_for_all_axes(self.figure.set_clim)
        else:
            if replot:
                self.figure.plot()
            else:
                self.figure.set_clim()


# %%
class PcolorCoordFormatter(object):
    def __init__(self, im, x, y, reverse_x=False):
        self.im = im
        self.x = x
        self.y = y
        self.reverse_x = reverse_x

    def __call__(self, x, y):
        try:
            if self.reverse_x:
                ind = (np.where(self.y >= y)[0][0]-1,
                       np.where(self.x >= x)[0][0-1*int(self.reverse_x)]-1)
            else:
                ind = (np.where(self.y >= y)[0][0-1*int(self.reverse_x)]-1,
                       np.where(self.x >= x)[0][0-1*int(self.reverse_x)]-1)
            z = self.im._A.reshape(self.im._meshHeight,
                                   self.im._meshWidth)[ind]
        except Exception:
            z = np.nan
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)


def open_topfigure(parent, fig_obj=None, color_obj=None, plot_type='2D',
                   **fig_kwargs):
    # wrapper function for class ToplevelFigure.
    # accepts TkMplFigure obj as fig_obj and
    # PlotColorManager2D as color_obj.
    # either will be copied, not passed directly, so the original object will
    # not change if figure in ToplevelFigure is edited.
    if fig_obj is not None:
        plot_type = fig_obj.plot_type
        fig_kwargs['num_subplots'] = fig_obj._num_subplots
        plot_on_frame_init = False
    else:
        plot_on_frame_init = True
    fig = ToplevelFigure(parent, plot_type=plot_type,
                         get_wdgt_val_from_fig=plot_on_frame_init,
                         show_plot=plot_on_frame_init,
                         **fig_kwargs)
    if fig_obj is not None:
        exclude = ['color_obj'] if color_obj is not None else []
        if 'plot_func' in fig_kwargs.keys():
            exclude.append('plot_func')
        fig.fr.figure.copy_attributes(fig_obj, exclude=exclude)
    if color_obj is not None:
        fig.fr.figure.copy_color_settings(color_obj)
    if not plot_on_frame_init:
        fig.fr.figure.plot_all()
        fig.fr.axes_focus_update_ui()
    return fig


def init_figure_frame(parent, fig_obj=None, color_obj=None, plot_type='2D',
                      **fig_kwargs):
    if fig_obj is not None:
        plot_type = fig_obj.plot_type
        fig_kwargs['num_subplots'] = fig_obj._num_subplots
    fr = FigFrame(parent, plot_type=plot_type, **fig_kwargs)
    if fig_obj is not None:
        fr.figure.copy_attributes(
            fig_obj, exclude=['color_obj'] if color_obj is not None else [])
        fr.axes_focus_update_ui()
    if color_obj is not None:
        fr.figure.copy_color_settings(color_obj)
    fr.figure.plot_all()
    return fr


class PlotMovie(ToplevelFigure):
    def __init__(self, parent, controller=None, fig_obj=None, data=None,
                 movie_function=None, dim=600, fit=None, show_movie=True,
                 title=None, fit_plotstyle='line', legend_entries=None,
                 frame_labels=None, plot_type=None, frame_interval=50,
                 movie_iterable=None, plot_kwargs=None, **fig_kwargs):
        if movie_iterable is None:
            self.movie_iterable = list(range(3))
        else:
            self.movie_iterable = movie_iterable
        if movie_function is None:
            if data is None:
                messagebox.showerror(
                    "Please provide either data set or a movie function"
                    + " for PlotMovie init.")
                return
            self.movie_function = self.data_movie
        else:
            self.movie_function = lambda *args, **kwargs: movie_function(
                self, *args, **kwargs)
        if legend_entries is None:
            legend_entries = False
        if not re.search('line', str(plot_type), re.I):
            plot_type = 'line' if fit is None else 'linefit'
        if not plot_kwargs:
            plot_kwargs = {'include_fit': fit is not None}
        else:
            plot_kwargs['include_fit'] = fit is not None
        ToplevelFigure.__init__(self, parent, controller, fig_obj=None,
                                clims=None, plot_type=plot_type, cbar=None,
                                dim=[dim, dim], editable=True, title=None,
                                plot_func=self.movie, legends=legend_entries,
                                plot_kwargs=plot_kwargs, **fig_kwargs)
        self.fr.opts.plot_opts.vars['show_legend'].set(0)
        self.frame_counter = 0
        self.frame_labels = frame_labels
        self.frame_label_coord = (0.05, 0.95)
        self.frame_label_kwargs = {}
        self.plot_title = title
        self.fit = fit
        if data is not None:
            data_type = type(data)
            if data_type is TATrace:
                self.data_type = 'traces'
                self.data = data
                self.num_lines = 1
                self.num_fit_lines = 1
            else:
                self.data_type = 'matrix'
                self.x = data[0]
                self.data = data[1]
                if len(np.shape(self.data)) > 1:
                    self.num_lines = np.shape(self.data)[0]
                else:
                    self.num_lines = 1
                if self.fit is not None:
                    self.fit_x = fit[0]
                    self.fit = fit[1]
                    self.num_fit_lines = np.shape(self.fit)[0]

        self.frame_interval = tk.IntVar(value=frame_interval)
        self.fr.frames['movie_opts'] = CustomFrame(
            self, dim=(8, 1), border=False)
        tk.Label(self.fr.frames['movie_opts'],
                 text="Frame interval (ms):").grid(
                     row=1, column=0, padx=1, sticky='we')
        self.frame_interval_entry = tk.ttk.Entry(
            self.fr.frames['movie_opts'],
            textvariable=self.frame_interval, width=10)
        self.frame_interval_entry.grid(row=1, column=1, padx=2, sticky=tk.W)
        self.frame_interval_entry.bind('<Return>', self.reset)

        ttk.Button(
            self.fr.frames['movie_opts'], text="Replay",
            command=self.replay).grid(row=1, column=2, padx=2, pady=2)

        ttk.Button(
            self.fr.frames['movie_opts'], text="Stop", command=self.stop).grid(
                row=1, column=3, padx=2, pady=2)

        ttk.Button(
            self.fr.frames['movie_opts'], text="Continue",
            command=self.contin).grid(row=1, column=4, padx=2, pady=2)
        ttk.Button(
            self.fr.frames['movie_opts'], text="Reverse time",
            command=self.reverse).grid(row=1, column=5, padx=2, pady=2)
        ttk.Button(
            self.fr.frames['movie_opts'], text="Reset",
            command=self.reset).grid(row=1, column=6, padx=2, pady=2)
        save_button = ttk.Button(
            self.fr.frames['movie_opts'], text="Save movie",
            command=self.save_animation)
        save_button.configure(state='disabled')
        save_button.grid(row=1, column=7, padx=2, pady=2)

        self.fr.frames['movie_opts'].grid(row=2, column=0, columnspan=2,
                                          sticky='wnse', padx=10, pady=5)
        self.fr.opts.opt_panels['lines'].set_fill = self.set_fill
        self.show_movie = self.fr.figure.plot
        if show_movie:
            self.show_movie()
        self.fr.figure.set_legend_visibility(case=False)

    def movie(self, *args, **kwargs):
        self.ani, self.movie_iterable, self.frame_counter = (
            self.movie_function(
                *args, frame_interval=self.frame_interval.get(),
                iterable=self.movie_iterable, counter=self.frame_counter,
                **kwargs))

    def data_movie(self, *args, frame_interval=100, ax=None,
                   fig=None, fill=0, reverse_z=False, color_order=None,
                   color_cycle=None, interpolate_colors=False,
                   # legend_kwargs={},
                   include_fit=True, fit_kwargs=None, iterable=None,
                   counter=0, transpose=False, 
                   linestyle=None, marker=None, **plot_kwargs):
        def get_plotdata_matrix(i, j, *args):
            return self.x, self.data[i, j, :]

        def get_fitdata_matrix(i, j, *args):
            return self.fit_x, self.fit[i, j, :]

        def get_plotdata_trace(i, j, keys):
            return self.data.xdata, self.data.tr[keys[j]]['y']

        def get_fitdata_trace(i, j, keys):
            return self.data.tr[keys[j]]['fit_x'], self.data.tr[keys[j]]['fit']

        if not fit_kwargs:
            fit_kwargs = {'color': 'black',
                          'linestyle': '-'}
        if linestyle is None:
            linestyle = ['-' for i in range(self.num_lines)]
        elif type(linestyle) is not list:
            linestyle = [linestyle for i in range(self.num_lines)]
        if type(marker) is not list:
            marker = [marker for i in range(self.num_lines)]
        if self.data_type == 'matrix':
            get_data = get_plotdata_matrix
            get_fit = get_fitdata_matrix
            args = None
            default_iter = list(range(np.shape(self.data)[1]))
        elif self.data_type == 'traces':
            get_data = get_plotdata_trace
            get_fit = get_fitdata_trace
            args = self.data.active_traces
            default_iter = list(range(len(args)))
        else:
            return None, self.movie_iterable, self.frame_counter

        fill = get_fill_list(fill, numlines=self.num_lines)

        ax.cla()
        if self.fit is None:
            include_fit = False
        lines = []
        cyc_len = self.num_lines if color_order is None else np.max(
            color_order['ind']) + 1
        zord, cycle = set_line_cycle_properties(
            cyc_len, reverse_z, color_order, cycle=color_cycle,
            interpolate=interpolate_colors)

        l, = ax.plot(get_data(0, 0, args)[0], get_data(0, 0, args)[1],
                     zorder=zord[0], color=cycle[0], 
                     linestyle=linestyle[0], marker=marker[0],
                     **plot_kwargs)
        lines.append(l)

        if self.movie_iterable[1] < self.movie_iterable[0]:
            self.movie_iterable = default_iter[::-1]
        else:
            self.movie_iterable = default_iter

        for i in range(1, self.num_lines):
            l, = ax.plot(get_data(i, 0, args)[0], get_data(i, 0, args)[1],
                         zorder=zord[i], color=cycle[i],
                         linestyle=linestyle[i], marker=marker[i],
                         **plot_kwargs)
            lines.append(l)
        if include_fit:
            for i in range(self.num_fit_lines):
                fit = get_fit(i, 0, args)
                l, = ax.plot(fit[0], fit[1], label='fit', **fit_kwargs)
                lines.append(l)

        def init():
            self.frame_counter = 0
            j = 0
            for i in range(self.num_lines):
                x, y = get_data(i, self.movie_iterable[0], args)
                lines[j].set_data(x, y)
                j += 1
            if include_fit:
                for i in range(self.num_fit_lines):
                    x, y = get_fit(i, self.movie_iterable[0], args)
                    lines[j].set_data(x, y)
                    j += 1
            if fill:
                for i in range(self.num_lines):
                    x, y = get_data(i, self.movie_iterable[0], args)
                    ax.fill_between(x, y, alpha=fill[i],
                                    color=lines[i].get_color())
            if self.frame_labels is not None:
                self.text = ax.text(*self.frame_label_coord,
                                    self.frame_labels[0],
                                    transform=ax.transAxes,
                                    **self.frame_label_kwargs)
            return lines

        def ani_fill(i):
            ax.collections.clear()
            for j in range(self.num_lines):
                x, y = get_data(j, i, args)
                ax.fill_between(x, y, alpha=fill[j],
                                color=lines[j].get_color())

        def ani_text(i):
            self.text.set_text(self.frame_labels[i])

        def ani_plot(i):
            try:
                k = 0
                for j in range(self.num_lines):
                    x, y = get_data(
                        j, self.movie_iterable[self.frame_counter], args)
                    lines[k].set_data(x, y)
                    k += 1
                if include_fit:
                    for j in range(self.num_fit_lines):
                        x, y = get_fit(
                            j, self.movie_iterable[self.frame_counter], args)
                        lines[k].set_data(x, y)
                        k += 1
            except Exception:
                raise
            return lines

        def animate(i):
            try:
                lines = ani_plot(self.movie_iterable[self.frame_counter])
            except Exception:
                self.ani.event_source.stop()
                return
            self.frame_counter += 1
            return lines

        def animate_text(i):
            try:
                ani_text(self.movie_iterable[self.frame_counter])
            except Exception:
                self.ani.event_source.stop()
                return
            lines = animate(i)
            return lines

        def animate_fill(i):
            try:
                ani_fill(self.movie_iterable[self.frame_counter])
            except Exception:
                self.ani.event_source.stop()
                return
            lines = animate(i)
            return lines

        def animate_fill_text(i):
            try:
                ani_fill(self.movie_iterable[self.frame_counter])
            except Exception:
                self.ani.event_source.stop()
                return
            lines = animate_text(i)
            return lines

        if fill:
            if self.frame_labels is None:
                mov_func = animate_fill
            else:
                mov_func = animate_fill_text
        else:
            if self.frame_labels is None:
                mov_func = animate
            else:
                mov_func = animate_text

        self.ani = animation.FuncAnimation(
            fig, mov_func, init_func=init, interval=frame_interval,
            save_count=len(self.movie_iterable))
        return self.ani, self.movie_iterable, self.frame_counter

    def replay(self, *args):
        try:
            self.ani.pause()
            self.frame_counter = 0
            self.ani.resume()
        except Exception:
            pass

    def reset(self, *args):
        try:
            self.ani.pause()
            self.fr.figure.plot()
        except Exception:
            raise

    def stop(self, *args):
        try:
            self.ani.pause()
        except Exception:
            pass

    def contin(self):
        try:
            self.ani.resume()
        except Exception:
            pass

    def reverse(self):
        self.frame_counter = len(self.movie_iterable) - self.frame_counter
        self.movie_iterable = self.movie_iterable[::-1]
        
    def set_fill(self, *args, update_canvas=True):
        try:
            self.fr.figure.clear_fill(lines="all", update_canvas=False)
        except ValueError:
            pass
        except Exception:
            raise
        self.fr.figure.set_line_fill(
            lines="all",
            alpha=self.fr.opts.opt_panels['lines'].fill_alpha,
            update_canvas=update_canvas)
        self.reset()
        self.ani.pause()
        self.fr.opts.opt_panels['lines'].set_lines_properties()
        self.ani.resume()

    def save_animation(self):
        if not self.plot_title:
            self.plot_title = "TA_movie"
        file, ext = save_box(
            title="Save movie", fname=self.plot_title + ".gif",
            fext=".gif", filetypes=[("GIF", ".gif")], return_ext=True)
        try:
            file.name
        except Exception:
            return
        else:
            fps = 1000/self.frame_interval.get()
            try:
                if ext == '.gif':
                    writer = animation.PillowWriter(fps=fps)
                elif ext in ('.mp4', '.avi', '.mov'):
                    writer = animation.FFMpegWriter(fps=fps)
                else:
                    messagebox.showerror("", "File extension not recognized.")
                    return
                self.ani.save(file.name, writer)
            except Exception as e:
                messagebox.showerror("", "Error saving movie:\n" + str(e))


def open_figure_options(*args, default_vals=None, **kwargs):
    if default_vals is not None:
        if 'dim' not in kwargs.keys():
            kwargs['dim'] = default_vals['figure_std_size']
        for key in ('fit_kwargs', 'plotstyle', 'ticklabel_format'):
            if key not in kwargs.keys():
                kwargs[key] = default_vals[key]
    return FigureOptionWindow(*args, **kwargs)
