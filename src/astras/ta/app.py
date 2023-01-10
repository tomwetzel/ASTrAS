# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:46:47 2019

@author: Simon Bittmann


python 3.6

required non-up-to-date package versions:
scipy 1.3.0

to do next:
    constraints for kinetic models in global fit

longer term ideas:
    different types of kinetic models in global fit
    wavelength dependent sigma values for global fit if feasible...
    wavelet analysis
    global fit: wavelet analysis for residual, need good test data though

commonly used abbreviations in variable and method names:
    ind(s) - index/indices
    win(s) - window(s)
    underscores often omitted after single chars
        such as x, y, z, t (= time) or c (= color)
    lbl(s) - label(s)
    lim - limit(s)
    del - delete
    opt(s) - option(s)
    ops - operations
    fig - figure
    obj = object
    disp - display
    resid - residual

"""
# %%
from ..common.helpers import (
    GlobalSettings, BlankObject, ThreadedTask, idle_function)
from ..common.dataobjects import TAData, TATrace
from ..common.mpl import (pcolor, plot, fill_between, get_fill_list,
                          set_line_cycle_properties)
from ..common.fitmethods import lmfitreport_to_dict

from ..common.tk.general import (
    save_box, load_box, CustomProgressbarWindow, CustomTimer, GroupBox,
    MultiDisplayWindow, MultiDisplay, ScrollFrame, CustomFrame, general_error,
    no_data_error, move_toplevel_to_default, center_toplevel,
    BinaryDialog, MultipleOptionsWindow, enable_disable_child_widgets)
from ..common.tk.linefit import (
    FitParaOptsWindow, FitParaEntryWindowLmfit, FitTracePage,
    KineticModelOptions, FitResultsDisplay)
from ..common.tk.figures import (
    TkMplFigure, open_topfigure, FigureOptionWindow, ColorMapEditor,
    ContrastSlider, init_figure_frame, PlotMovie, tk_mpl_plot_function)
from ..common.tk.traces import TraceSelection

from queue import Queue
from matplotlib.widgets import RectangleSelector
# from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pickle
import re
import traceback
import threading
# math and plot modules
from scipy import ndimage
from uncertainties import ufloat
import numpy as np
import lmfit
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['font.family'] = 'serif'


__all__ = ['AppMain', 'MainPage', 'TraceManager', 'AveragingOptionsWindow',
           'AveragingOptions', 'OutlierDisplay', 'AveragingManager',
           'ChirpCorrPage', 'ManualChirpWindow', 'DataInspector',
           # 'KineticModelOptions',
           # 'SpectralFitOptions', 'KineticFitOptions', 'FitTracePage',
           'GlobalFitPage', 'GlobalFitResults', 'TraceManagerWindow',
           'SVDAnalysis', 'CPMFit', 'DynamicLineShapeFit',
           'ModifyData', 'FitResultsDisplay']


# %% Main tkinter app (controller)
class AppMain(tk.Tk, tk.Toplevel):
    def __init__(self, *args, scrollable=None, geometry=None,
                 init_canv_geom=None, config_filepath=None, 
                 parent=None, **kwargs):
        if parent is None:
            tk.Tk.__init__(self, *args, **kwargs)
        else:
            tk.Toplevel.__init__(self, parent)
        if config_filepath is None:
            config_filepath = 'config.txt'
        elif not re.search("\.", config_filepath):
            config_filepath += '.txt'
        self.title("ASTrAS - Transient Absorption Analysis")
        self.settings = GlobalSettings(config_path=config_filepath)
        # main data object
        data = TAData(
            time_delay_precision=int(self.settings['time_delay_precision']),
            input_time_conversion_factor=np.double(
                self.settings['input_time_conversion_factor']))
        # ttk styles
        if scrollable is None:
            try:
                scrollable = self.settings['scrollable']
            except Exception:
                scrollable = False
        if geometry is None:
            try:
                geometry = self.settings['geometry']
            except Exception:
                geometry = False
        if init_canv_geom is None:
            try:
                init_canv_geom = self.settings['init_canv_geom']
            except Exception:
                init_canv_geom = "175x160"
        style = ttk.Style()
        style.theme_use("vista")
        style.configure("Custom.TLabelframe.Label", foreground="black")
        style.configure("Custom.TLabelframe", foreground="grey")
        self.app_settings = {}
        self.settings_loaded = False
        # tkinter widgets
        container = tk.Frame(self)
        if scrollable:
            canvas = tk.Canvas(container)
            self.canvas = canvas
            self.set_canvas_size(init_canv_geom)
            main_frame = tk.Frame(canvas)
            main_frame.grid(sticky='wnse', column=0, row=0)
            vsb = tk.ttk.Scrollbar(container, orient="vertical",
                                   command=canvas.yview)
            hsb = tk.ttk.Scrollbar(container, orient="horizontal",
                                   command=canvas.xview)
            canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            vsb.grid(row=0, column=1, sticky='sne')
            hsb.grid(row=1, column=0, sticky='wse')
            canvas.grid(sticky='wnse', row=0, column=0)
            self.frame_window = canvas.create_window((4, 4), window=main_frame,
                                                     anchor='n',
                                                     tags="main_frame")
            main_frame.bind(
                "<Configure>", lambda *args: canvas.config(
                    scrollregion=canvas.bbox("all")))
            canvas.bind(
                "<Configure>", lambda *args: canvas.config(
                    scrollregion=canvas.bbox("all")))
        else:
            self.columnconfigure(0, weight=1)
            self.rowconfigure(0, weight=1)
            container.grid_columnconfigure(0, weight=1)
            for i in range(3):
                container.grid_rowconfigure(i, weight=1)
            main_frame = container

        # initialize menu bar
        self.menubar = tk.Menu(container)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.pagemenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Pages", menu=self.pagemenu)
        self.appmenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Apps", menu=self.appmenu)
        tk.Tk.config(self, menu=self.menubar)

        # Define frames
        self.frames = {}

        self.buttons = {}
        self.frames['nav'] = CustomFrame(main_frame, dim=(8, 1), border=True)
        self.frames['info'] = CustomFrame(main_frame, border=True)
        self.file_label = tk.Label(self.frames['info'],
                                   text='Please load data via file menu.')
        self.file_label.grid(row=0, column=0, sticky=tk.W, padx=5)

        for lbl, page in zip(("Home", "Chirp Correction", "Global Fit"),
                             (MainPage, ChirpCorrPage, GlobalFitPage)):
            frame = page(main_frame, self, data)
            self.frames[page] = frame
            self.buttons[frame] = tk.Button(self.frames['nav'], text=lbl,
                                            command=lambda fr=page:
                                                self.show_frame(fr))
            self.pagemenu.add_command(
                label=lbl, command=lambda fr=page: self.show_frame(fr))
            frame.grid(row=1, column=0, sticky="nsew", pady=5)
        frame = FitTracePage(
            main_frame, controller=self, figure_settings=self.settings,
            main_page=self.frames[MainPage])
        self.frames[FitTracePage] = frame
        self.buttons[frame] = tk.Button(self.frames['nav'], text="Trace Fit",
                                        command=lambda fr=FitTracePage:
                                            self.show_frame(fr))
        self.pagemenu.add_command(
            label="Trace Fit",
            command=lambda fr=FitTracePage: self.show_frame(fr))
        frame.grid(row=1, column=0, sticky="nsew", pady=5)

        col = 0
        self.buttons[self.frames[MainPage]].grid(
            column=col, row=0, padx=5, pady=5)
        col += 1
        self.buttons[self.frames[ChirpCorrPage]].grid(
            column=col, row=0, padx=5, pady=5)
        col += 1
        self.buttons[self.frames[FitTracePage]].grid(
            column=col, row=0, padx=5, pady=5)
        col += 1
        self.buttons[self.frames[GlobalFitPage]].grid(
            column=col, row=0, padx=5, pady=5)
        col += 1
        self.buttons['data_inspect'] = tk.Button(
            self.frames['nav'], text='Data Inspector',
            command=self.frames[MainPage].open_data_inspector)
        self.buttons['data_inspect'].grid(column=col, row=0, padx=5, pady=5)
        col += 1
        self.buttons['svd'] = tk.Button(
            self.frames['nav'], text='SVD Analysis',
            command=self.frames[MainPage].open_svd_window)
        self.buttons['svd'].grid(column=col, row=0, padx=5, pady=5)
        col += 1
        self.buttons['lineshape'] = tk.Button(
            self.frames['nav'], text="Line shape",
            command=self.frames[MainPage].open_lineshape_win)
        self.buttons['lineshape'].grid(column=col, row=0, padx=5, pady=5)
        col += 1
        self.buttons['cpm_fit'] = tk.Button(
            self.frames['nav'], text='CPM Fit',
            command=self.frames[MainPage].open_cpm_window)
        self.buttons['cpm_fit'].grid(column=col, row=0, padx=5, pady=5)

        self.frames['nav'].grid(row=0, column=0, sticky='nsew')
        self.current_frame = frame
        self.show_frame(MainPage)
        self.frames['info'].grid(row=2, column=0, sticky='nsew')

        container.grid(row=0, column=0, padx=20, pady=5, sticky='wnse')
        if bool(geometry):
            self.geometry(geometry)
        if scrollable:
            self.get_window_size()
            self.bind("<Configure>",
                      lambda event: self.canvas_resize(canvas, event))

        self.protocol("WM_DELETE_WINDOW", self._closing)

    def _closing(self):
        if self.settings['ui_closing_warning']:
            if messagebox.askokcancel(
                    "Quit Application",
                    "All unsaved data will be lost. Quit anyway?",
                    parent=self):
                self.destroy()
        else:
            self.destroy()

    def get_window_size(self):
        self.wd = self.winfo_width()
        self.ht = self.winfo_height()

    def set_canvas_size(self, geometry):
        self.canvas_wd = int(re.search('(\d*)(?=x)', geometry)[0])
        self.canvas_ht = int(re.search('(?<=x)(\d*)', geometry)[0])
        self.canvas.config(width=self.canvas_wd, height=self.canvas_ht)

    def canvas_resize(self, canvas, event):
        def update_size(event):
            wd = self.winfo_width()
            ht = self.winfo_height()
            wd_diff = wd - self.wd
            ht_diff = ht - self.ht
            self.wd = wd
            self.ht = ht
            self.canvas_wd += wd_diff
            self.canvas_ht += ht_diff
            canvas.config(width=self.canvas_wd, height=self.canvas_ht)
        if event.widget is self:
            if (np.abs(self.wd - event.width) > 20
                    or np.abs(self.ht - event.height) > 20):
                update_size(event)

    def save_app_settings(self):
        for key in self.frames.keys():
            self.app_settings[key] = {}
            try:
                self.frames[key].write_app_settings()
            except Exception as e:
                print(key, e)
        file = save_box(fext=".pkl", filetypes=[
                       ('Python setting file', '*.pkl')])
        try:
            file.name
        except Exception:
            return
        else:
            with open(file.name, 'wb') as f:
                pickle.dump(self.app_settings, f)
                f.close()

    def load_app_settings(self):
        file = load_box(fext=".pkl", filetypes=[
                       ('Python setting file', '*.pkl')])
        try:
            file[0].name
        except Exception as e:
            print(e)
        else:
            with open(file[0].name, 'rb') as f:
                self.app_settings = pickle.load(f)
                f.close()
            for key in self.frames.keys():
                try:
                    self.frames[key].load_app_settings()
                except Exception as e:
                    print(key, e)
            self.settings_loaded = True

    def show_frame(self, cont):
        curr_fr = self.current_frame
        if self.current_frame._leave_page():
            self.current_frame = self.frames[cont]
            if self.current_frame._enter_page():
                self.buttons[curr_fr].config(fg='black')
                self.current_frame.tkraise()
                self.buttons[self.current_frame].config(fg='blue')
            else:
                self.current_frame = curr_fr

    def open_topfigure_wrap(self, *args, default_vals=None, **kwargs):
        if default_vals is None:
            default_vals = self.settings
        if 'dim' not in kwargs.keys():
            kwargs['dim'] = default_vals['figure_std_size']
        for key in ('fit_kwargs', 'plotstyle', 'ticklabel_format'):
            if key not in kwargs.keys():
                try:
                    kwargs[key] = default_vals[key]
                except Exception:
                    pass
        return open_topfigure(*args, **kwargs)

    def tk_mpl_figure_wrap(self, *args, **kwargs):
        if 'dim' not in kwargs.keys():
            kwargs['dim'] = self.settings['figure_std_size']
        for key in ('fit_kwargs', 'plotstyle'):
            if key not in kwargs.keys():
                kwargs[key] = self.settings[key]
        if 'plot_type' not in kwargs.keys():
            kwargs['plot_type'] = '2D'
        if kwargs['plot_type'].lower() == '2d':
            for key, settingskey in zip(('transpose', 'invert_yaxis'),
                                        ('2Dmap_transpose', '2Dmap_invert_y')):
                if key not in kwargs.keys():
                    kwargs[key] = self.settings[settingskey]
        return TkMplFigure(*args, **kwargs)

    def open_figure_options(self, *args, default_vals=None, **kwargs):
        if default_vals is None:
            default_vals = self.settings
        if 'dim' not in kwargs.keys():
            kwargs['dim'] = default_vals['figure_std_size']
        for key in ('fit_kwargs', 'plotstyle', 'ticklabel_format'):
            if key not in kwargs.keys():
                kwargs[key] = default_vals[key]
        return FigureOptionWindow(*args, **kwargs)


# %%
""" ################# Pages of Main app ###################
    TOC:
        MainPage
        ChirpCorrection
        FitTracePage
        GlobalFitPage

"""


class MainPage(tk.Frame):
    def __init__(self, parent, controller, data_obj, master=None):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.data_obj = data_obj
        # variables
        trace_dict = self.get_trace_dict()
        traces = trace_dict['Kinetic']
        # self.centroidLabel = ['Center of gravity']
        # self.plot_centroid_in_map = False
        self.centroids = None
        self.plot_kwargs = {}
        self.entries = {}
        self.buttons = {}
        self.vars = {}
        self.optmenus = {}
        self.checks = {}
        self.frames = {}
        self.labels = {}
        self.widgets = {}

        # Frames
        self.frames['main_opts'] = CustomFrame(
            self, dim=(6, 5), border=False)
        self.frames['filter_opts'] = GroupBox(self.frames['main_opts'],
                                              dim=(6, 2), text="Filter")
        for i in range(5):
            self.rowconfigure(i, weight=1)
        for i in range(3):
            self.columnconfigure(i, weight=1)

        # Figures
        self.main_figure = self.controller.tk_mpl_figure_wrap(
            self,
            # dim=settings['figure_std_size'],
            callbacks={'button_press_event':
                       self.main_plot_click_callback},
            plot_function=self.data_obj.plot_ta_map,
            xlabels=self.controller.settings['xlabel_default'],
            ylabels=self.controller.settings['ylabel_default'],
            clabels=self.data_obj.zlabel)
        self.main_figure.grid(row=0, column=0, padx=2, pady=2, sticky='wnse')
        self.trace_figure = self.controller.tk_mpl_figure_wrap(
            self,
            # dim=settings['figure_std_size'],
            callbacks={'button_press_event':
                       self.trace_plot_click_callback},
            plot_function=traces.plot,
            xlabels=self.controller.settings['ylabel_default'],
            ylabels=self.data_obj.zlabel,
            plot_type='linefit',
            plot_kwargs={'include_fit': False})
        self.trace_figure.grid(row=0, column=2, columnspan=2,
                               padx=2, pady=2, sticky='wnse')
        # panel: Main plot options
        # x Axis options
        xopts = GroupBox(self.frames['main_opts'],
                         text='Spectral Axis', dim=(2, 3))
        self.vars['lambda_lim_low'] = tk.DoubleVar(
            value=self.controller.settings['xlim_lower'])
        self.vars['lambda_lim_up'] = tk.DoubleVar(
            value=self.controller.settings['xlim_upper'])
        self.buttons['lambda_lim'] = ttk.Button(
            xopts, text="Set spectral range", command=self.set_xlimits)
        self.buttons['lambda_lim'].grid(column=0, row=0, sticky='we', padx=5)

        self.entries['lambda_lim_low'] = tk.Entry(
            xopts, textvariable=self.vars['lambda_lim_low'], width=7)
        self.entries['lambda_lim_low'].grid(
            column=1, row=0, sticky='w', padx=5)
        self.entries['lambda_lim_low'].bind('<Return>', self.set_xlimits)
        self.entries['lambda_lim_up'] = tk.Entry(
            xopts, width=7, textvariable=self.vars['lambda_lim_up'])
        self.entries['lambda_lim_up'].grid(column=2, row=0, sticky='w', padx=5)
        self.entries['lambda_lim_up'].bind('<Return>', self.set_xlimits)

        tk.Label(xopts, text='Spectral quantity:').grid(row=1, column=0,
                                                        padx=5)
        self.vars['spec_mode'] = tk.StringVar(
                value=self.controller.settings['input_spectral_quantity'])
        self.optmenus['spec_unit'] = tk.ttk.OptionMenu(
            xopts, self.vars['spec_mode'],
            self.controller.settings['input_spectral_quantity'],
            'wavelength', 'wavenumber', 'energy',
            command=self.change_xunit)
        self.optmenus['spec_unit'].grid(row=1, column=1, columnspan=2,
                                        sticky='we')
        self.optmenus['spec_unit'].config(width=12)

        self.buttons['lambda_shift'] = ttk.Button(
            xopts, text='Shift wavelength:', command=self.shift_wavelength)
        self.buttons['lambda_shift'].grid(column=0, row=2, padx=5,
                                          columnspan=1, sticky='we')
        self.vars['lambda_shift'] = tk.DoubleVar(value='0')
        self.entries['lambda_shift'] = tk.Entry(
            xopts, textvariable=self.vars['lambda_shift'], width=7)
        self.entries['lambda_shift'].bind('<Return>', self.shift_wavelength)
        self.entries['lambda_shift'].grid(column=1, row=2, sticky='w',
                                          padx=(5, 0))
        self.labels['lambda_shift_unit'] = tk.Label(
            xopts, text=self.controller.settings['input_spectral_unit'])
        self.labels['lambda_shift_unit'].grid(column=2, row=2, sticky='w',
                                              padx=5)
        xopts.grid(row=0, column=0, sticky='wnse', padx=2, pady=2)

        # y axis options

        yopts = GroupBox(self.frames['main_opts'], text='Time Axis',
                         dim=(2, 3))
        self.buttons['td_lim'] = ttk.Button(yopts, text="Set time range",
                                            command=self.set_tlim)
        self.buttons['td_lim'].grid(row=0, column=0, sticky='we', padx=5,
                                    columnspan=1)
        self.vars['tlower'] = tk.DoubleVar(value=0)
        self.vars['tupper'] = tk.DoubleVar(value=1)
        self.entries['tlower'] = tk.Entry(
            yopts, textvariable=self.vars['tlower'], width=7)
        self.entries['tupper'] = tk.Entry(
            yopts, textvariable=self.vars['tupper'], width=7)
        self.entries['tlower'].bind('<Return>', self.set_tlim)
        self.entries['tupper'].bind('<Return>', self.set_tlim)
        self.entries['tlower'].grid(row=0, column=1, sticky='w', padx=(5, 0))
        self.entries['tupper'].grid(row=0, column=2, sticky='w', padx=5)

        tk.Label(yopts, text='Input time unit:').grid(
            row=1, column=0, columnspan=1, padx=5, sticky='we')
        self.vars['tunit'] = tk.StringVar(
            value=self.controller.settings['input_time_unit'])
        self.entries['tunit'] = tk.Entry(
            yopts, textvariable=self.vars['tunit'], width=7, justify=tk.RIGHT)
        self.entries['tunit'].bind('<Return>', self.t_unit_select_cb)
        self.entries['tunit'].grid(row=1, column=1, sticky='w', padx=5)

        self.vars['t0shift'] = tk.DoubleVar(value=0)
        self.buttons['t0shift'] = ttk.Button(yopts,
                                             text='Time zero shift',
                                             command=self.shift_t0)
        self.buttons['t0shift'].grid(column=0, row=2, sticky='we', padx=5)
        self.entries['t0shift'] = tk.Entry(yopts,
                                           textvariable=self.vars['t0shift'],
                                           width=7)
        self.entries['t0shift'].grid(column=1, row=2, sticky='w', padx=5)
        self.entries['t0shift'].bind('<Return>', self.shift_t0)
        self.labels['t0shift_unit'] = tk.Label(
            yopts, text=self.controller.settings['input_time_unit'])
        self.labels['t0shift_unit'].grid(column=2, row=2, sticky='w', padx=5)

        yopts.grid(row=0, column=1, sticky='wnse', padx=2, pady=2)

        # filter subpanel

        tk.Label(self.frames['filter_opts'], text="Sigma:").grid(
            column=1, row=0, sticky='w', padx=5)
        tk.Label(self.frames['filter_opts'], text="Order:").grid(
            column=1, row=1, sticky='w', padx=5)
        self.buttons['gauss_filt'] = ttk.Button(
            self.frames['filter_opts'], text="Gaussian Filter",
            command=self.filter_button_callback)
        self.buttons['gauss_filt'].grid(column=0, row=0, sticky='we', padx=5,
                                        pady=2, rowspan=2)
        self.buttons['gauss_filt_reset'] = ttk.Button(
            self.frames['filter_opts'], text="Reset Filter",
            command=self.reset_filter_callback)
        self.buttons['gauss_filt_reset'].grid(column=3, row=0, padx=5,
                                              sticky='we', rowspan=2, pady=2)

        self.vars['gauss_filt_sigma'] = tk.DoubleVar(value=0.8)
        self.vars['gauss_filt_order'] = tk.IntVar(value=0)
        self.entries['gauss_filt_sigma'] = tk.Entry(
            self.frames['filter_opts'],
            textvariable=self.vars['gauss_filt_sigma'],
            width=7)
        self.entries['gauss_filt_sigma'].grid(
            column=2, row=0, sticky=tk.W, padx=5)
        self.entries['gauss_filt_order'] = tk.Entry(
            self.frames['filter_opts'],
            textvariable=self.vars['gauss_filt_order'],
            width=7)
        self.entries['gauss_filt_order'].grid(
            column=2, row=1, sticky=tk.W, padx=5)

        self.checks['gauss_filt'] = tk.ttk.Checkbutton(
            self.frames['filter_opts'])

        tk.Label(self.frames['filter_opts'], text='Pixel binning').grid(
            column=4, row=0, rowspan=2, sticky='w', padx=(5, 0))
        self.vars['pixel_binning'] = tk.IntVar(value=1)
        self.entries['pixel_binning'] = tk.Entry(
            self.frames['filter_opts'],
            textvariable=self.vars['pixel_binning'],
            width=7)
        self.entries['pixel_binning'].grid(column=5, row=0, sticky='w',
                                           padx=5, rowspan=2)
        self.entries['pixel_binning'].bind(
            '<Return>', self.bin_pixel_callback)

        self.frames['filter_opts'].grid(row=3, column=0, columnspan=2,
                                        sticky='wnse', padx=5, pady=2)

        # subpanel: plot color

        self.frames['plot_color'] = ColorMapEditor(
            self.frames['main_opts'], self.main_figure, horizontal=True,
            auto_callback=self.auto_clim_callback, update_function=(
                lambda *args: self.update_color_settings()))
        self.frames['plot_color'].vars['invert_color'].set(1)
        self.frames['plot_color'].grid(row=4, column=0, columnspan=2, padx=5,
                                       pady=5, sticky='wnse')

        self.buttons['advanced_edit'] = tk.ttk.Button(
            self.frames['main_opts'], text='Advanced',
            command=self.open_adv_editor)
        self.buttons['advanced_edit'].grid(row=5, column=0, padx=5, sticky='w')

        self.frames['main_opts'].grid(row=2, column=0, sticky='wsne', ipady=5,
                                      ipadx=2, pady=5)

        # panel: trace plot options

        self.trace_opts = TraceManager(
            self, self.controller, self.trace_figure, self.data_obj,
            traces=trace_dict,
            spec_unit=self.controller.settings['input_spectral_unit'])
        self.trace_opts.model_select_callbacks['Integral'] = self.integrate
        self.trace_opts.model_select_callbacks['Centroid'] = self.calc_centroid
        self.trace_opts.model_select_callback_general = (
            self.remove_centroid)
        self.trace_opts.select.grid_forget()
        self.trace_opts.vars['trace_mode'].trace(
            'w', self._trace_type_callback)
        self.trace_opts.select.grid(
            row=0, column=1, rowspan=2, sticky='wnse', padx=5, pady=5)

        self.trace_opts.grid(row=2, column=2, sticky='nswe', pady=5, padx=5)

        # panel: main page apps

        self.frames['inpage_apps'] = CustomFrame(
            self.trace_opts, dim=(2, 2), border=True)
        self.frames['inpage_apps'].grid(row=1, column=0, sticky='nswe',
                                        pady=5, padx=5)
        self.buttons['integrate'] = ttk.Button(
            self.frames['inpage_apps'], text="Integrate",
            command=self.integrate_callback)
        self.buttons['integrate'].grid(column=0, row=0, padx=5, pady=5)

        self.vars['integral_mode'] = tk.StringVar(value="Spectrum")
        self.optmenus['integral_mode'] = tk.ttk.OptionMenu(
                                                self.frames['inpage_apps'],
                                                self.vars['integral_mode'],
                                                "Spectrum", "Spectrum", "Time")
        self.optmenus['integral_mode'].grid(row=0, column=1, padx=5, pady=5)

        self.buttons['centroid'] = ttk.Button(
            self.frames['inpage_apps'], text="Centroid",
            command=self.centroid_callback)
        self.buttons['centroid'].grid(column=0, row=1, padx=5, pady=5)
        self.buttons['movie'] = ttk.Button(self.frames['inpage_apps'],
                                           text="Show movie",
                                           command=self.show_ta_movie)
        self.buttons['movie'].grid(column=1, row=1, padx=5, pady=5)

        # menu bar
        controller.filemenu.add_command(
            label="Load single Scan", command=self.load_scan)
        controller.filemenu.add_command(
            label="Load & Average", command=self.load_and_average)
        controller.filemenu.add_command(
            label="Advanced Scan Averaging", command=self.average_advanced)
        controller.filemenu.add_command(
            label="Load settings", command=self.controller.load_app_settings)
        controller.filemenu.add_command(
            label="Load wavelength file", command=self.load_wavelength_file)
        controller.filemenu.add_separator()
        controller.filemenu.add_command(
            label="Save data", command=self.save_data)
        controller.filemenu.add_command(
            label="Save settings", command=self.controller.save_app_settings)
        controller.appmenu.add_command(
            label="Inspect Data", command=self.open_data_inspector)
        controller.appmenu.add_command(
            label="SVD Analysis", command=self.open_svd_window)
        controller.appmenu.add_command(
            label="Dynamic Line Shape Fit", command=self.open_lineshape_win)
        controller.appmenu.add_command(
            label="CPM Fit", command=self.open_cpm_window)
        tk.Tk.config(controller, menu=controller.menubar)

        # miscellaneous
        for obj in (self, self.frames['plot_color']):
            for widget in (list(obj.entries.values())
                           + list(obj.buttons.values())
                           + list(obj.widgets.values())
                           + list(obj.optmenus.values())):
                widget.config(state='disabled')
        for widget in self.checks.values():
            widget.config(state='disabled')
        for widget in self.entries.values():
            widget.config(justify=tk.RIGHT)
        self.entries['tunit'].config(justify=tk.LEFT)

    def write_app_settings(self):
        for var in (self.vars, self.frames['plot_color'].vars):
            for key, val in var.items():
                try:
                    self.controller.app_settings[MainPage][key] = val.get()
                except Exception as e:
                    print(e)

    def load_app_settings(self):
        current_settings = {}
        for key in ('spec_mode', 't0shift', 'lambda_lim_up', 'lambda_lim_low'):
            current_settings[key] = self.vars[key].get()
        for var in (self.vars, self.frames['plot_color'].vars):
            for key in var.keys():
                try:
                    var[key].set(self.controller.app_settings[MainPage][key])
                except Exception as e:
                    print(e)
        if self.vars['lambda_shift'].get() != 0:
            self.shift_wavelength(update_main_figure=False)
        self.t_unit_select_cb(update_main_figure=False)
        if current_settings['spec_mode'] != self.vars['spec_mode'].get():
            self.change_xunit(update_figures=False)
        if current_settings['t0shift'] != self.vars['t0shift'].get():
            self.shift_t0(update_main_figure=False)
        self.set_tlim(update_main_figure=False)
        if ((current_settings['lambda_lim_up']
             != self.vars['lambda_lim_up'].get())
            or (current_settings['lambda_lim_low']
                != self.vars['lambda_lim_low'].get())):
            self.set_xlimits(update_clim=False, update_main_figure=False)
        self.frames['plot_color'].change_color_scheme()

    def set_tlim(self, *args, update_main_figure=True):
        upper = np.max(self.data_obj.time_delays)
        lower = np.min(self.data_obj.time_delays)
        if (self.vars['tupper'].get() > upper
                or self.vars['tupper'].get() < lower):
            self.vars['tupper'].set(np.round(upper, 3))
        if (self.vars['tlower'].get() < lower
                or self.vars['tlower'].get() > upper):
            self.vars['tlower'].set(np.round(lower, 3))
        ylim = [self.vars['tlower'].get(), self.vars['tupper'].get()]
        self.main_figure.set_ylim(ylim, update_canvas=update_main_figure)
        self.update_data_cursor()
        if self.trace_opts.trace_mode not in ('Spectral', 'Right SV'):
            self.trace_figure.set_xlim(ylim, update_canvas=update_main_figure)
        for key in ['Kinetic', 'Left SV', 'Centroid']:
            try:
                self.trace_opts.traces[key].xrange = ylim
            except Exception:
                pass
        self.trace_opts.update_traces(xlim=ylim)
        self.data_obj.set_time_win_indices(ylim)

    def set_xlimits(self, *args, xlim=None, update_clim=True,
                    update_main_figure=True):
        if xlim is None:
            xlim = [self.vars['lambda_lim_low'].get(),
                    self.vars['lambda_lim_up'].get()]
        try:
            self.data_obj.set_xlimits(np.sort(xlim))
            self.main_figure.set_xlim(xlim)
            self.update_data_cursor()
            for key in ['Spectral', 'Right SV']:
                try:
                    self.trace_opts.traces[key].xrange = xlim
                except Exception:
                    pass
            if re.search('wave|energ',
                         self.trace_opts.current_traces.xlabel) and len(
                    self.trace_opts.current_traces.active_traces) > 0:
                self.trace_opts.update_traces()
            if update_clim:
                self.get_auto_clim(update_global=True)
        except Exception as e:
            raise
            if not self.data_obj.get_is_loaded():
                no_data_error()
            else:
                general_error(e)

    def update_data_cursor(self):
        self.main_figure.set_data_cursor_2d(
                self.data_obj.spec_axis[self.data_obj.get_x_mode()],
                self.data_obj.time_delays,
                reverse_x=self.data_obj.get_x_mode() != 'wavelength')

    def save_data(self):
        win = BinaryDialog(
            self, controller=self, yes_button_text="Complete",
            prompt="Save complete matrix or user-defined time range?",
            no_button_text="Range")
        file = save_box(fext='.mat',
                        filetypes=[('text files', '.txt'),
                                   ('Matlab files', '.mat')],
                        fname='TA_matrix',
                        parent=self)
        if file is not None:
            if not win.output:
                self.data_obj.save_data_matrix(
                    file, matrix=self.data_obj.delA[self.data_obj.td_slice, :],
                    y=self.data_obj.time_delays[self.data_obj.td_slice])
            else:
                self.data_obj.save_data_matrix(file, matrix=self.data_obj.delA)

    # color editor
    def update_color_settings(self):
        self.data_obj.color = self.main_figure.color_obj[0]
        for page in (GlobalFitPage, ChirpCorrPage):
            self.controller.frames[page].ta_map_figure.copy_color_settings(
                self.data_obj.color)
            self.controller.frames[page].ta_map_figure.set_cmap(
                update_canvas=False)
            self.controller.frames[page].ta_map_figure.set_clim(
                update_canvas=True)

    def auto_clim_callback(self, update_global=False):
        self.get_auto_clim(update_global=update_global)
        self.frames['plot_color'].change_color_scheme()

    def get_auto_clim(self, update_global=False):
        clims = self.main_figure.set_auto_clim(
            self.data_obj.delA[:, self.data_obj.get_xlim_slice()],
            opt='symmetric' if self.frames['plot_color'].vars['clim_sym'].get()
                else 'asymmetric')
        self.frames['plot_color'].vars['lower_clim'].set(clims[0])
        self.frames['plot_color'].vars['upper_clim'].set(clims[1])
        if update_global:
            self.update_color_settings()

    def main_plot_click_callback(self, event):
        if self.data_obj.get_is_loaded():
            mode = self.trace_opts.trace_mode
            if (event.button == 3
                    and mode.lower() in ('kinetic', 'spectral')):
                try:
                    if (((not self.main_figure.transpose[0])
                            and mode.lower() == 'kinetic')
                        or (self.main_figure.transpose[0]
                            and mode.lower() == 'spectral')):
                        point = event.xdata
                        rnd = 1
                    else:
                        point = event.ydata
                        rnd = 3
                    self.trace_opts.get_and_plot_traces(point)
                    self.trace_opts.manual_point.set(
                        np.round(point, rnd))
                except Exception as e:
                    raise
                    if not self.data_obj.get_is_loaded():
                        no_data_error()
                    else:
                        general_error(e)
            elif event.dblclick:
                if self.centroids:
                    plot_func = self.centroid_map_plot_func
                else:
                    plot_func = self.data_obj.plot_ta_map
                win = self.controller.open_topfigure_wrap(
                    self, plot_func=plot_func, editable=True,
                    fig_obj=self.main_figure,
                    color_obj=self.main_figure.color_obj)

                if self.centroids:
                    win.fr.opts.plot_opts.grid_forget()
                    win.fr.opts.init_lineplot_opts(
                        win.fr.opts, self.controller, row=1)
                    win.fr.figure.grid()
                    win.fr.figure.plot_function = plot_func
                    win.fr.figure.plot_all()
            elif event.button == 3:
                self.controller.open_figure_options(
                    self, self.main_figure, controller=self.controller)

    def trace_plot_click_callback(self, event):
        if self.data_obj.get_is_loaded():
            if event.button == 3:
                self.controller.open_figure_options(
                    self, self.trace_figure, controller=self.controller)
            elif event.dblclick:
                self.controller.open_topfigure_wrap(
                    self, fig_obj=self.trace_figure, editable=True,
                    data_obj=self.trace_opts.current_traces)

    def change_xunit(self, *args, update_figures=True):
        try:
            self.data_obj.set_xlimits(
                np.sort([self.vars['lambda_lim_low'].get(),
                         self.vars['lambda_lim_up'].get()]))
            self.data_obj.set_x_mode(self.vars['spec_mode'].get())
            try:
                self.data_obj.time_zeros[:, 0] = self.data_obj.spec_axis[
                    self.data_obj.get_x_mode()]
            except Exception:
                pass
            xlim_new = self.data_obj.get_xlim()
            self.vars['lambda_lim_low'].set(xlim_new[0])
            self.vars['lambda_lim_up'].set(xlim_new[1])

            if update_figures:
                self.update_main_plot()
            if re.search('spec', self.trace_opts.trace_mode, re.I):
                self.trace_opts.unit_lbl.config(
                    text=self.data_obj.time_unit)
            elif re.search('kinetic', self.trace_opts.trace_mode, re.I):
                self.trace_opts.unit_lbl.config(
                    text=self.data_obj.spec_unit)
            else:
                self.trace_opts.unit_lbl.config(text="")
            for key in ['Spectral', 'Right SV']:
                try:
                    self.trace_opts.traces[key].set_x_mode(
                        self.vars['spec_mode'].get())
                    self.trace_opts.traces[key].xrange = xlim_new
                except Exception:
                    pass
            if len(self.trace_opts.current_traces.active_traces) > 0:
                self.trace_opts.update_traces()
            else:
                self.update_axes_labels(update_canvas=update_figures)
            self._update_secondary_pages()
            self.update_data_cursor()
        except Exception as e:
            traceback.print_exc()
            if not self.data_obj.get_is_loaded():
                no_data_error()
                self.vars['spec_mode'].set(
                    self.controller.settings['input_spectral_quantity'])
            else:
                general_error(e)

    def shift_wavelength(self, *args, update_main_figure=True):
        try:
            self.data_obj.shift_wavelengths(self.vars['lambda_shift'].get())
        except Exception as e:
            if not self.data_obj.get_is_loaded():
                no_data_error()
            else:
                general_error(e)
        else:
            self.update_trace_spec_axis()
            self._update_secondary_pages()
            if update_main_figure:
                self.update_main_plot()

    def update_trace_spec_axis(self):
        self.trace_opts.traces['Spectral'].xdata = self.data_obj.spec_axis[
            self.data_obj.get_x_mode()]
        self.trace_opts.traces['Kinetic'] = TATrace()
        self.trace_opts.update_traces()
        self.controller.frames[FitTracePage]._update_content()

    def update_axes_labels(self, update_canvas=True):
        # main figure
        self.data_obj.time_unit = self.vars['tunit'].get()
        self.main_figure.set_axes_label(
            y='time delay (' + self.data_obj.time_unit + ')',
            x=(self.data_obj.get_x_mode()
               + ' (' + self.data_obj.spec_unit_label + ')'),
            update_canvas=update_canvas)
        # trace figure
        if self.trace_opts.trace_mode in ('Spectral', 'Right SV'):
            self.trace_figure.set_xlabel(self.main_figure.get_xlabel(),
                                         update_canvas=update_canvas)
        else:
            self.trace_figure.set_xlabel(self.main_figure.get_ylabel(),
                                         update_canvas=update_canvas)
        self.controller.frames[FitTracePage]._update_content()

    def t_unit_select_cb(self, *args, update_main_figure=True):
        self.update_axes_labels(update_canvas=update_main_figure)
        if self.trace_opts.trace_mode.lower() == 'spectral':
            self.unit_lbl.config(text=self.data_obj.time_unit)
        self.labels['t0shift_unit'].config(text=self.data_obj.time_unit)
        self.trace_opts.traces['Kinetic'].set_xunit(self.data_obj.time_unit)
        self.trace_opts.traces['Centroid'].set_xunit(self.data_obj.time_unit)
        self.controller.frames[ChirpCorrPage].t0_figure.set_ylabel(
                self.main_figure.get_ylabel(), update_canvas=True)

    def load_wavelength_file(self):
        if self.data_obj.get_is_loaded():
            filename = filedialog.askopenfilename()
            try:
                self.data_obj.load_wavelength_file(filename)
            except Exception:
                return
            else:
                self.update_main_plot()
                self.vars['lambda_shift'].set(0)
                self.vars['spec_mode'].set(
                    self.controller.settings['input_spectral_quantity'])
        else:
            messagebox.showerror(
                parent=self,
                message="Please load data before loading wavelength file.")

    def load_scan(self):
        filename = filedialog.askopenfilename()
        try:
            fext = filename[-re.search("\.", filename[::-1]).end():]
        except Exception:
            pass
        else:
            if self.data_obj.get_is_loaded():
                self.clear_data()
            self.data_obj.load_single_scan(filename, filetype=fext)
            self._after_loading()
            self.set_infolabel(filename, case='file')

    def clear_data(self):
        for ax in (self.main_figure.axes[0],
                   self.controller.frames[GlobalFitPage].ta_map_figure.axes[0],
                   self.trace_figure.axes[0],
                   self.controller.frames[FitTracePage].fit_figure.axes[0]):
            ax.cla()
        # self.data_obj.clear()
        try:
            self.main_figure.cbar[0].remove()
        except Exception:
            pass
        self.data_obj.__init__()

    def set_infolabel(self, path, case='file'):
        case_dict = {'file': 'Loaded file ',
                     'dir': 'Averaged files in directory ',
                     'concat': 'Concatenated files in directory '}
        self.controller.file_label.config(text=case_dict[case] + path)

    def load_and_average(self):
        # user input
        path = filedialog.askdirectory()
        try:
            filetypes = self.get_filetypes(path)
        except Exception:
            pass
        else:
            if len(filetypes) > 0:
                if len(filetypes) > 1:
                    window = MultipleOptionsWindow(
                        self, self.controller,
                        filetypes,
                        text=("Choose file types for averaging\n"
                              + " (Multiple possible, but not recommended)"),
                        buttontext="Load & Average")
                    filetypes = window.output
            else:
                filetypes = [".mat", ".scan", ".txt"]
            win = AveragingOptionsWindow(self, self.controller)
            self.wait_window(win)
            if win.output is None:
                return
        # Begin loading and averaging
            if self.data_obj.get_is_loaded():
                self.clear_data()
            pbar = CustomProgressbarWindow(self, controller=self.controller,
                                           text="Scan averaging",
                                           cancel_button=True)
            pbar.start_timer()
            try:
                outliers = self.data_obj.load_and_average_scans(
                    path, filetypes, progressbar=pbar, **win.output)
                pbar.update_timer()
        # update UI
                if outliers is not False:
                    self._after_loading()
                    if win.output['remove_outliers'] and outliers is not None:
                        OutlierDisplay(self, self.controller, outliers)
                else:
                    self.main_figure.axes[0].cla()
                    self.main_figure.canvas.draw()
                pbar.destroy()
            except Exception as e:
                messagebox.showerror(message=e, parent=self)
                pbar.destroy()
            else:
                self.set_infolabel(path, case='dir')

    def average_advanced(self):
        # user input
        path = filedialog.askdirectory()
        try:
            filetypes = self.get_filetypes(path)
        except Exception:
            pass
        else:
            if len(filetypes) > 0:
                if len(filetypes) > 1:
                    window = MultipleOptionsWindow(
                        self, self.controller,
                        filetypes,
                        text=("Choose file types for averaging\n"
                              + " (Multiple possible, but not recommended)"),
                        buttontext="Load & Average")
                    filetypes = window.output
            else:
                filetypes = [".mat", ".scan", ".txt"]
        # Loading scans
            if self.data_obj.get_is_loaded():
                self.clear_data()
            pbar = CustomProgressbarWindow(self, controller=self.controller,
                                           text="Loading Files",
                                           cancel_button=True)
            pbar.start_timer()
            all_timesteps_equal, all_timesteps_same = (
                self.data_obj.read_scan_files(
                    path, filetypes, progressbar=pbar))
        # preliminary average
            self.data_obj.average_scans(
                all_timesteps_equal, all_timesteps_same, interpolate_nan=False,
                progressbar=pbar)
        # start averaging manager
            pbar.destroy()

            window = AveragingManager(
                self, self.controller, all_timesteps_equal,
                all_timesteps_same, self.data_obj,
                canvas_dim=[self.controller.settings['figure_std_size'][0],
                            self.controller.settings['figure_std_size'][1]*2])
            self.wait_window(window)
        # update UI after averaging
            if window.success:
                self._after_loading()
                self.set_infolabel(path)

    def get_filetypes(self, path):
        file_types = []
        for t in (".mat", ".scan", ".txt"):
            for f in os.scandir(path):
                if re.search(t, f.name):
                    file_types.append(t)
                    break
        return file_types

    def _after_loading(self):
        for obj in (self, self.frames['plot_color']):
            for widget in (list(obj.entries.values())
                           + list(obj.buttons.values())
                           + list(obj.widgets.values())
                           + list(obj.optmenus.values())):
                widget.config(state='normal')
        if self.frames['plot_color'].vars['clim_sym'].get():
            self.frames['plot_color'].entries['clim_low'].config(
                state='disabled')
            self.frames['plot_color'].widgets['cmap_sym'].config(
                state='disabled')
            self.frames['plot_color'].widgets['cmap_center_zero'].config(
                state='disabled')
        for widget in self.checks.values():
            widget.config(state='normal')

        self.trace_opts.plot_opt_enable_disable()
        self.vars['lambda_shift'].set(0)
        self.vars['t0shift'].set(0)
        self.update_data_cursor()
        self.auto_time_win()
        self.main_figure.plot(xlimits=self.data_obj.get_xlim(),
                              ylimits=[self.vars['tlower'].get(),
                                       self.vars['tupper'].get()])
        # reset trace manager and set data
        self.trace_opts.clear_all_traces()
        self.trace_opts.set_data(dat=self.data_obj)
        self.trace_opts.set_trace_mode(trace_mode='Kinetic')
        # reset other pages
        self._update_secondary_pages()

    def update_main_plot(self):
        self.main_figure.plot(xlimits=[self.vars['lambda_lim_low'].get(),
                                       self.vars['lambda_lim_up'].get()],
                              ylimits=[self.vars['tlower'].get(),
                                       self.vars['tupper'].get()])
        self.update_data_cursor()

    def auto_time_win(self):
        self.vars['tupper'].set(np.round(
            np.max(self.data_obj.time_delays), 3))
        self.vars['tlower'].set(np.round(
            np.min(self.data_obj.time_delays), 3))
        self.data_obj.set_time_win_indices(
            [self.vars['tlower'].get(), self.vars['tupper'].get()])

    def get_trace_dict(self):
        self.data_obj.integral.ylabel = 'Integrated intensity (a.u.)'
        self.data_obj.centroid.ylabel = ('Centroid ('
                                         + self.data_obj.spec_unit_label + ')')
        trace_dict = {'Kinetic': TATrace(trace_type='kinetic',
                                         ylabel=self.data_obj.zlabel),
                      'Spectral': TATrace(trace_type='spectral',
                                          ylabel=self.data_obj.zlabel),
                      'Integral': self.data_obj.integral,
                      'Centroid': self.data_obj.centroid}
        return trace_dict

    def shift_t0(self, *args, update_main_figure=True):
        try:
            prev_shift = self.data_obj.time_zero_shift
            self.data_obj.shift_timezero(self.vars['t0shift'].get())
        except Exception as e:
            if not self.data_obj.get_is_loaded():
                no_data_error()
            else:
                raise
                general_error(e)
        else:
            if update_main_figure:
                self.update_main_plot()
                self.update_data_cursor()
            if self.trace_opts.trace_mode not in ('Spectral', 'Right SV'):
                self.trace_opts.current_traces.xdata = (np.array(
                                    self.trace_opts.current_traces.xdata)
                                    - prev_shift
                                    + self.data_obj.time_zero_shift)
                try:
                    self.trace_opts.update_traces()
                except Exception:
                    self.trace_opts.current_traces = TATrace(
                        xdata=self.data_obj.time_delays)
                    self.trace_opts.update_traces(set_current_traces=False)
            else:
                self.update_axes_labels(update_canvas=update_main_figure)

    def bin_pixels(self):
        try:
            self.data_obj.bin_pixels(self.vars['pixel_binning'].get())
        except Exception as e:
            if self.data_obj.get_is_loaded():
                messagebox.showerror(
                                "Error",
                                "Error attempting to bin data.\n" + str(e),
                                parent=self)
            else:
                no_data_error()
            self.data_obj.delA = self.data_obj.non_filt_delA
            self.vars['pixel_binning'].set(1)
        else:
            for key in ('Spectral', 'Right SV'):
                try:
                    self.trace_opts.traces[key].clear_all_traces()
                except KeyError:
                    pass
                except Exception:
                    raise
            self.trace_opts.set_trace_mode()

    def gaussian_filter(self):
        try:
            self.data_obj.delA = ndimage.gaussian_filter(
                self.data_obj.delA,
                sigma=self.vars['gauss_filt_sigma'].get(),
                order=self.vars['gauss_filt_order'].get())
        except Exception as e:
            if self.data_obj.get_is_loaded():
                messagebox.showerror(
                    "Error",
                    "Invalid input parameter(s) for Gaussian filter.\n"
                    + str(e),
                    parent=self)
            else:
                no_data_error()
        # else:
        #     self.data_obj.is_filtered = True

    def bin_pixel_callback(self, *args):
        if self.vars['pixel_binning'].get() < 1:
            self.vars['pixel_binning'].set(1)
            self.data_obj.delA = self.data_obj.non_filt_delA
        else:
            self.bin_pixels()
            # if self.data_obj.is_filtered:
            #     self.gaussian_filter()
        self.frames['plot_color'].clim_callback([])
        self.update_main_plot()

    def filter_button_callback(self, *args):
        try:
            self.bin_pixels()
            self.gaussian_filter()
        except Exception:
            pass
        else:
            self.frames['plot_color'].clim_callback([])
            self.update_main_plot()

    def reset_filter_callback(self):
        self.data_obj.delA = self.data_obj.non_filt_delA
        if self.data_obj.pixel_binned:
            self.data_obj.reset_bin()
        self.frames['plot_color'].clim_callback([])
        self.update_main_plot()
        # self.data_obj.is_filtered = False

    # traces, Integral & Centroid
    def _trace_type_callback(self, *args):
        self.controller.frames[FitTracePage]._update_content()

    def integrate_callback(self, *args):
        self.trace_opts.set_trace_mode(trace_mode='Integral')

    def integrate(self):
        try:
            # run integration
            if self.vars['integral_mode'].get() == 'Spectrum':
                key = ('Integral '
                       + str(self.vars['lambda_lim_low'].get()) + '-'
                       + str(self.vars['lambda_lim_up'].get()) + ' '
                       + self.data_obj.spec_unit_label)
                self.data_obj.integrate_spectrum(label=key)
            elif self.vars['integral_mode'].get() == 'Time':
                key = ('Integral '
                       + str(self.vars['tlower'].get()) + '-'
                       + str(self.vars['tupper'].get()) + ' '
                       + self.entries['tunit'].get())
                self.data_obj.integrate_time(label=key)
            else:
                return

        except Exception as e:
            if not self.data_obj.get_is_loaded():
                no_data_error()
            else:
                general_error(e)
        else:
            self.trace_opts.traces['Integral'] = self.data_obj.integral
            self.trace_opts.trace_mode = 'Integral'
            self.trace_opts.unit_lbl.config(text="")
            if self.vars['integral_mode'].get() == 'Time':
                self.trace_figure.set_xlim(
                    [self.vars['lambda_lim_low'].get(),
                     self.vars['lambda_lim_up'].get()])
            else:
                self.trace_figure.set_xlim(
                    [min(self.data_obj.time_delays),
                     max(self.data_obj.time_delays)])

    def centroid_callback(self, *args):
        self.trace_opts.set_trace_mode(trace_mode='Centroid')

    def calc_centroid(self, *args):
        try:
            key = ('Center of gravity '
                   + str(self.vars['lambda_lim_low'].get()) + '-'
                   + str(self.vars['lambda_lim_up'].get()) + ' '
                   + self.data_obj.spec_unit_label)
            self.data_obj.calculate_centroid(label=key)

        except Exception as e:
            if not self.data_obj.get_is_loaded():
                no_data_error()
            else:
                general_error(e)
        else:
            self.data_obj.centroid.ylabel = (
                'Centroid (' + self.data_obj.spec_unit_label + ')')
            self.data_obj.centroid.xlabel = self.main_figure.get_ylabel()
            self.trace_opts.traces['Centroid'] = self.data_obj.centroid
            self.trace_opts.trace_mode = 'Centroid'
            self.trace_opts.unit_lbl.config(text="")
            self.trace_figure.set_xlim(
                [self.data_obj.time_delays[0], self.data_obj.time_delays[-1]])
            self.remove_centroid(update_canvas=False)
            self.plot_centroid_mainplot(update_canvas=True)
            return {'ylim': self.main_figure.get_xlim()}

    def plot_centroid_mainplot(self, axes=None, update_canvas=False):
        if axes is None:
            axes = self.main_figure.axes[0]
        self.centroids = self._plot_centroid(axes)
        if update_canvas:
            self.main_figure.canvas.draw()

    def _plot_centroid(self, axes):
        key = self.data_obj.centroid.active_traces[0]
        centroids, = axes.plot(self.data_obj.centroid.tr[key]['y'],
                               self.data_obj.centroid.xdata,
                               marker='o', linestyle='None',
                               markersize=5, color='black')
        return centroids

    def centroid_map_plot_func(self, *args, ax=None, **kwargs):
        image = self.data_obj.plot_ta_map(*args, ax=ax, **kwargs)
        self._plot_centroid(ax)
        return image

    def remove_centroid(self, update_canvas=True):
        if self.centroids:
            self.main_figure.axes[0].lines.remove(self.centroids)
            self.centroids = []
            if update_canvas:
                self.main_figure.canvas.draw()

    # page navigation & Apps
    def open_data_inspector(self):
        if self._load_check():
            DataInspector(self, self.controller, self.data_obj,
                          xlim=self.main_figure.get_xlim(),
                          ylim=self.main_figure.get_ylim(),
                          zlim=self.data_obj.color.clims)

    def show_ta_movie(self):
        if self._load_check():
            frame_labels = [str(np.round(td, 3)) + ' '
                            + self.data_obj.time_unit
                            for td in self.data_obj.time_delays]
            PlotMovie(self, self.controller,
                      data=[self.data_obj.wavelengths,
                            np.array([self.data_obj.delA])],
                      xlimits=self.main_figure.get_xlim(),
                      ylimits=self.main_figure.get_clim(),
                      frame_labels=frame_labels,
                      xlabels=self.main_figure.get_xlabel(),
                      ylabels=self.main_figure.get_clabel())

    def open_lineshape_win(self):
        if self._load_check():
            self.data_obj.plot_data.write_properties(
                x=self.data_obj.wavenumbers,
                xlims=self.data_obj.calc_wavenumber(
                    self.main_figure.get_xlim()),
                ylims=self.main_figure.get_ylim(),
                xlabel='wavenumber (cm$^{-1}$)',
                ylabel=self.main_figure.get_ylabel(),
                clims=self.data_obj.color.clims,
                z=self.data_obj.delA,
                y=self.data_obj.time_delays,
                plot_kwargs=self.data_obj.color.get_kwargs(),
                clabel=self.data_obj.zlabel)
            DynamicLineShapeFit(self, self.controller, self.data_obj.plot_data,
                                self.data_obj)
            self.data_obj.plot_data.write_properties(x=self.data_obj.spec_axis[
                self.data_obj.get_x_mode()],
                xlabel=self.main_figure.get_xlabel(),
                xlims=self.main_figure.get_xlim(),
                z=self.data_obj.delA, y=self.data_obj.time_delays)

    def _write_plot_data_for_ext_app(self, write_color_kwargs=True):
        self.data_obj.plot_data.write_properties(
            x=self.data_obj.spec_axis[self.data_obj.get_x_mode()][
                self.data_obj.get_xlim_slice()],
            # self.data_obj._lambda_lim_index[0]
            # :self.data_obj._lambda_lim_index[1]],
            xlims=self.main_figure.get_xlim(),
            ylims=self.main_figure.get_ylim(),
            xlabel=self.main_figure.get_xlabel(),
            ylabel=self.main_figure.get_ylabel(),
            clims=self.data_obj.color.clims,
            z=self.data_obj.delA[:, self.data_obj.get_xlim_slice()],
            y=self.data_obj.time_delays)
        if write_color_kwargs:
            self.data_obj.plot_data.write_properties(
                plot_kwargs=self.main_figure.color_obj[0].get_kwargs())

    def open_svd_window(self):
        # note: currently only available for wavelength as x axis because of
        # compatibility with spectral trace fit. To be adjusted soon
        if self._load_check():
            self._write_plot_data_for_ext_app()
            SVDAnalysis(self, self.controller, self.data_obj.plot_data,
                        self.data_obj,
                        xlim=self.main_figure.get_xlim(),
                        ylim=self.main_figure.get_ylim())

    # deprecated, to be deleted
    # def openSpecShiftWindow(self):
    #     if self._load_check():
    #         self._write_plot_data_for_ext_app()
    #         SpectralShift(self, self.controller, self.data_obj.plot_data)
    def open_cpm_window(self):
        if self._load_check():
            self._write_plot_data_for_ext_app(write_color_kwargs=False)
            CPMFit(self, self.controller, self.data_obj.plot_data,
                   self.data_obj, color_obj=self.main_figure.color_obj[0])

    def open_adv_editor(self):
        if self._load_check():
            ModifyData(self, self.data_obj, controller=self.controller)

    def _load_check(self):
        if self.data_obj.get_is_loaded():
            return True
        else:
            no_data_error()
            return False

    def _update_secondary_pages(self):
        for page in (ChirpCorrPage, GlobalFitPage, FitTracePage):
            self.controller.frames[page]._update_content()

    # page navigation functions
    def _leave_page(self):
        return self._load_check()

    def _enter_page(self):
        return True


# %%
class TraceManager(tk.Frame):
    def __init__(self, parent, controller, trace_figure, data_obj,
                 spec_unit=None, traces=None, dat=None,
                 model_select_callbacks=None, residual_widgets=False):
        tk.Frame.__init__(self, parent)
        self.trace_figure = trace_figure
        self.parent = parent
        self.controller = controller
        self.data_obj = data_obj
        self.set_data(dat=dat)
        if spec_unit is None:
            spec_unit = 'nm'
        if traces is None:
            self.traces = {'Kinetic': TATrace(trace_type='kinetic',
                                              ylabel=self.data_obj.zlabel),
                           'Spectral': TATrace(trace_type='spectral',
                                               ylabel=self.data_obj.zlabel)}
        else:
            self.traces = traces
        if model_select_callbacks is None:
            self.init_callbacks()
        else:
            self.model_select_callbacks = model_select_callbacks

        self.model_select_callback_general = idle_function

        self.optmenus = {}
        self.vars = {}
        self.entries = {}
        self.checks = {}
        self.buttons = {}

        self.add_trace_box = GroupBox(self, text="Add Traces", dim=(3, 6))
        tk.Label(self.add_trace_box, text='Type:').grid(
            row=0, column=0, sticky=tk.W)
        self.trace_mode = 'Kinetic'
        if self.trace_mode not in self.traces.keys():
            self.trace_mode = list(self.traces.keys())[0]
        self.current_traces = self.traces[self.trace_mode]
        self.vars['trace_mode'] = tk.StringVar(value=self.trace_mode)
        self.optmenus['trace_mode'] = tk.ttk.OptionMenu(
                                                self.add_trace_box,
                                                self.vars['trace_mode'],
                                                'Kinetic',
                                                *self.traces.keys(),
                                                command=self.set_trace_mode)
        self.optmenus['trace_mode'].config(width=11)
        self.optmenus['trace_mode'].grid(
            row=0, column=1, columnspan=2, sticky=tk.W)

        tk.Label(self.add_trace_box, text='Add trace(s) at:').grid(
            row=1, column=0, sticky=tk.W)
        self.manual_point = tk.DoubleVar(value=450)
        self.entries['manual_point'] = tk.Entry(
                            self.add_trace_box,
                            textvariable=self.manual_point,
                            width=7, state='disabled')
        self.entries['manual_point'].grid(row=1, column=1, sticky=tk.W)
        self.entries['manual_point'].bind(
            '<Return>', self.trace_plot_manual_callback)

        self.unit_lbl = tk.Label(
                                        self.add_trace_box,
                                        text=spec_unit)
        self.unit_lbl.grid(row=1, column=2, sticky=tk.W)

        self.vars['trace_plot_increment'] = tk.DoubleVar(value=50)
        self.entries['trace_plot_increment'] = tk.Entry(
                                self.add_trace_box,
                                textvariable=self.vars['trace_plot_increment'],
                                width=7, state='disabled')
        self.entries['trace_plot_increment'].bind(
            '<Return>', self.trace_plot_manual_callback)
        self.entries['trace_plot_increment'].grid(row=3, column=1, sticky=tk.W)

        self.vars['trace_plot_num_increments'] = tk.IntVar(value=1)
        self.entries['trace_plot_num_increments'] = tk.Entry(
            self.add_trace_box,
            textvariable=self.vars['trace_plot_num_increments'],
            width=7, state='disabled')
        self.entries['trace_plot_num_increments'].bind(
            '<Return>', self.trace_plot_manual_callback)
        self.entries['trace_plot_num_increments'].grid(
            row=2, column=1, sticky=tk.W)
        tk.Label(self.add_trace_box, text="No. of traces:").grid(
            row=2, column=0, sticky=tk.W)
        tk.Label(self.add_trace_box, text='Interval:').grid(
            row=3, column=0, sticky=tk.W)

        self.vars['add_trace_plots'] = tk.IntVar(value=1)
        self.checks['add_trace_plots'] = tk.ttk.Checkbutton(
                                        self.add_trace_box,
                                        text="Add trace plots",
                                        variable=self.vars['add_trace_plots'])
        self.checks['add_trace_plots'].grid(row=4, column=0, sticky=tk.W)

        self.buttons['save_traces'] = ttk.Button(self.add_trace_box,
                                                 text='Save traces',
                                                 command=self.save_traces)

        self.vars['subtract_traces'] = tk.IntVar(value=0)
        self.checks['subtract_traces'] = tk.ttk.Checkbutton(
                                        self.add_trace_box,
                                        text="Subtract from first trace",
                                        variable=self.vars['subtract_traces'],
                                        command=self.subtract_traces_callback)
        self.checks['subtract_traces'].grid(
            row=5, column=0, columnspan=2, sticky=tk.W)

        self.buttons['save_traces'].grid(
            row=4, column=1, columnspan=2, sticky=tk.W)

        self.select = TraceSelection(
                                self, self.traces,
                                update_func=self.trace_sel_update,
                                fig_obj=self.trace_figure,
                                header='Select Traces',
                                layout='horizontal',
                                delete_button=True)

        self.add_trace_box.grid(row=0, column=0, sticky='wnse', padx=5, pady=5)

        self.select.grid(row=0, column=1, sticky='wnse', padx=5, pady=5)

        if residual_widgets:
            resid_grpbx = GroupBox(self, text="Residual")
            self.buttons['plot_resid'] = tk.ttk.Button(
                        resid_grpbx,
                        text="Show Residual",
                        command=(lambda *args, fft=False, **kwargs:
                                 self.open_resid_figure(
                                     *args, fft=fft, **kwargs)))
            self.buttons['plot_resid_fft'] = tk.ttk.Button(
                        resid_grpbx,
                        text="Resid. FFT",
                        command=(lambda *args, fft=True, **kwargs:
                                 self.open_resid_figure(
                                     *args, fft=fft, **kwargs)))
            self.buttons['plot_resid'].grid(row=0, column=0)
            self.buttons['plot_resid_fft'].grid(row=0, column=1)
            waveletbox = GroupBox(resid_grpbx, text="Wavelet Analysis")
            self.buttons['wavelet_resid'] = tk.ttk.Button(
                waveletbox, text="Transform",
                command=self.run_wavelet_analysis)
            self.buttons['wavelet_resid'].grid(row=1, column=0)
            tk.ttk.Label(waveletbox, text="Time range:").grid(row=2, column=0)
            self.vars['wavelet_lower'] = tk.DoubleVar()
            self.vars['wavelet_upper'] = tk.DoubleVar()
            tk.ttk.Entry(waveletbox, textvariable=self.vars['wavelet_lower'],
                         width=8).grid(row=2, column=1)
            tk.ttk.Entry(waveletbox, textvariable=self.vars['wavelet_upper'],
                         width=8).grid(row=2, column=2)
            tk.ttk.Label(waveletbox, text="Wavelet:").grid(row=3, column=0)
            self.vars['wavelet_type'] = tk.StringVar(value="Morlet")
            self.optmenus['wavelet_type'] = tk.ttk.OptionMenu(
                waveletbox, self.vars['wavelet_type'],
                self.vars['wavelet_type'].get(),
                self.vars['wavelet_type'].get())
            self.optmenus['wavelet_type'].config(state='disabled')
            self.optmenus['wavelet_type'].grid(row=3, column=1, columnspan=2)
            tk.ttk.Label(waveletbox, text="No. of Scales:").grid(
                row=4, column=0)
            self.vars['num_scales'] = tk.IntVar(value=1000)
            tk.ttk.Entry(waveletbox, textvariable=self.vars['num_scales'],
                         width=5).grid(row=4, column=1, columnspan=2)

            waveletbox.grid(row=1, column=0, sticky='wnse')
            # disable wavelet analysis methods until proper testing
            enable_disable_child_widgets(waveletbox, case='disabled')
            resid_grpbx.grid(row=1, column=0, sticky='wnse')

    def open_resid_figure(self, *args, fft=False):
        tr = self.traces[self.trace_mode]
        if fft:
            tr.residual_fft()
            plot_func = (lambda *args, pc='resid_fft', **kwargs:
                         tr.plot_residual(*args, plot_case=pc, **kwargs))
            title = 'FFT of Residual'
            xlbl = "wavenumber (cm$^{-1}$)"
        else:
            plot_func = (lambda *args, pc='resid', **kwargs:
                         tr.plot_residual(*args, plot_case=pc, **kwargs))
            title = 'Residual'
            xlbl = tr.xlabel
        self.controller.open_topfigure_wrap(
            self, plot_func=plot_func, plot_type='line', editable=True,
            legends=[tr.active_traces], xlabels=xlbl, ylabels=tr.ylabel,
            axes_titles=title)

    def run_wavelet_analysis(self, *args):
        # def plot_func(*args, **kwargs):
        #     return self.data_obj.plot_ta_map(*args, dat=cwtmat, yvalues=freq,
        #                            xvalues = tr.xdata[xrange[0]:xrange[1]],
        #                            **kwargs)
        tr = self.traces[self.trace_mode]
        xrange = np.where(
                    np.logical_and(
                        tr.xdata >= self.vars['wavelet_lower'].get(),
                        tr.xdata <= self.vars['wavelet_upper'].get()))
        xrange = [xrange[0][0], xrange[0][-1]]
        cwtmat, freq, error = self.data_obj.wavelet_analysis(
                                dat=tr.tr[tr.active_traces[-1]]['residual'],
                                x=tr.xdata, xrange=xrange,
                                num_scales=self.vars['num_scales'].get())
        self.controller.open_topfigure_wrap(
            self, plot_func=lambda *args, **kwargs: self.data_obj.plot_ta_map(
                *args,
                dat=cwtmat,
                yvalues=freq,
                xvalues=tr.xdata[xrange[0]:xrange[1]],
                **kwargs),
            editable=True)

    def init_callbacks(self):
        def update_kinetic():
            self.trace_figure.set_xlim(
                np.sort([self.data_obj.time_delays[0],
                         self.data_obj.time_delays[-1]]))
            self.unit_lbl.config(text=self.data_obj.spec_unit)

        def update_spectral():
            self.trace_figure.set_xlim(self.data_obj.get_xlim())
            self.unit_lbl.config(text=self.data_obj.time_unit)

        self.model_select_callbacks = {'Kinetic': update_kinetic,
                                       'Spectral': update_spectral}

    def set_data(self, dat=None):
        if type(dat) is BlankObject or type(dat) is TAData:
            self.dat = dat
        elif dat is None:
            self.dat = self.data_obj
        elif type(dat) is dict:
            self.dat = BlankObject()
            for attr in ['x', 'y', 'z', 'xlabel', 'ylabel', 'zlabel', 'fit']:
                try:
                    setattr(self.dat, attr, dat[attr])
                except Exception:
                    setattr(self.dat, attr, None)
        if type(self.dat) is TAData:
            self.get_trace_at_point = self.get_trace_at_point_tadata
        else:
            self.get_trace_at_point = self.get_trace_at_point_general

    def clear_all_traces(self):
        for key, tr in self.traces.items():
            tr.clear_all_traces()

    def set_trace_mode(self, *args, trace_mode=None):
        if trace_mode:
            self.vars['trace_mode'].set(trace_mode)
        self.trace_mode = self.vars['trace_mode'].get()
        self.parent.trace_mode = self.trace_mode
        self.model_select_callback_general()
        self.trace_figure.axes[0].cla()
        self.trace_figure.plot_function = self.traces[self.trace_mode].plot
        try:
            del self.diff_traces
        except Exception:
            pass
        self.vars['subtract_traces'].set(0)
        try:
            self.plot_opt_enable_disable()
            self.update_axes_labels(update_canvas=False)
        except Exception:
            pass
        try:
            update_kwargs = self.model_select_callbacks[self.trace_mode]()
        except KeyError:
            update_kwargs = {}
        except Exception:
            raise
        if update_kwargs is None:
            update_kwargs = {}
        self.update_traces(**update_kwargs)

    def trace_plot_manual_callback(self, *args):
        def get_y(case):
            if case is TAData:
                return self.data_obj.time_delays[self.data_obj.td_slice]
            else:
                return self.dat.y

        def get_x(case):
            if case is TAData:
                return self.data_obj.spec_axis[self.data_obj.get_x_mode()][
                    self.data_obj.get_xlim_indices()]
            else:
                return self.dat.x

        if self.trace_mode in ('Spectral', 'Kinetic'):
            if self.trace_mode == 'Spectral':
                check = get_y(type(self.dat))
            else:
                check = get_x(type(self.dat))
            try:
                if self.manual_point.get() > max(check):
                    self.manual_point.set(np.round(max(check), 3))
                elif self.manual_point.get() < min(check):
                    self.manual_point.set(np.round(min(check), 3))
                self.get_and_plot_traces(
                    self.manual_point.get(), increment=True)
            except Exception as e:
                if not self.data_obj.get_is_loaded():
                    no_data_error()
                else:
                    general_error(e)

    def subtract_traces_callback(self, *args):
        if self.vars['subtract_traces'].get():
            self.subtract_traces()
        else:
            self.update_traces(set_current_traces=True)

    def subtract_traces(self):
        traces = self.traces[self.trace_mode]
        init_kwargs = {'xdata': traces.xdata, 'xlabel': traces.xlabel,
                       'ylabel': "$\Delta$" + traces.ylabel}
        try:
            self.diff_traces
        except Exception:
            self.diff_traces = TATrace(**init_kwargs)
        else:
            if not self.vars['add_trace_plots'].get():
                self.diff_traces.__init__(**init_kwargs)
        ref_key = traces.active_traces[0]
        keys = traces.active_traces[1:]
        for key in keys:
            diff = traces.tr[key]['y'] - traces.tr[ref_key]['y']
            key = 'Difference ' + key + ' - ' + ref_key
            self.diff_traces.tr[key] = {'y': diff}
            self.diff_traces.active_traces.append(key)
        self.current_traces = self.diff_traces
        self.update_traces(set_current_traces=False)

    def plot_opt_enable_disable(self, *args):
        state = 'disabled' if self.trace_mode in (
            'Integral', 'Centroid') else 'normal'
        for key in ('trace_plot_increment',
                    'trace_plot_num_increments',
                    'manual_point'):
            self.entries[key].config(state=state)

    def get_trace_at_point_general(self, point):
        try:
            xunit = re.search('(?<=\().*(?=\))',
                              self.dat.get_xlabel()).group(0)
        except Exception:
            try:
                xunit = re.search('(?<=\().*(?=\))', self.dat.xlabel).group(0)
            except Exception:
                raise
                xunit = ''
        try:
            yunit = re.search('(?<=\().*(?=\))',
                              self.dat.get_ylabel()).group(0)
        except Exception:
            try:
                yunit = re.search('(?<=\().*(?=\))', self.dat.ylabel).group(0)
            except Exception:
                yunit = ''
        if self.trace_mode == 'Kinetic':
            p = np.where(self.dat.x >= point)[0][0]
            tr = self.dat.z[:, p]
            try:
                fit = self.dat.fit[:, p]
            except Exception:
                fit = None
                self.trace_figure.set_plot_kwargs(include_fit=False)
            else:
                self.trace_figure.set_plot_kwargs(include_fit=True)
            self.traces[self.trace_mode].xdata = self.dat.y
            self.traces[self.trace_mode].xlabel = self.dat.ylabel
            lb = " ".join([str(np.round(point, 3)), xunit])
            self.traces[self.trace_mode].set_xunit(yunit)
        elif self.trace_mode == 'Spectral':
            p = np.where(self.dat.y >= point)[0][0]
            tr = self.dat.z[p, :]
            try:
                fit = self.dat.fit[p, :]
            except Exception:
                fit = None
                self.trace_figure.set_plot_kwargs(include_fit=False)
            else:
                self.trace_figure.set_plot_kwargs(include_fit=True)
            self.traces[self.trace_mode].xdata = self.dat.x
            self.traces[self.trace_mode].xlabel = self.dat.xlabel
            self.traces[self.trace_mode].set_xunit(xunit)
            lb = " ".join([str(np.round(point, 3)), yunit])
        else:
            return

        self.traces[self.trace_mode].tr[lb] = {'y': tr, 'val': point}
        if fit is not None:
            self.traces[self.trace_mode].tr[lb]['fit'] = fit
            self.traces[self.trace_mode].tr[lb]['fit_x'] = self.traces[
                self.trace_mode].xdata
            self.traces[self.trace_mode].tr[lb][
                'residual'] = self.traces[self.trace_mode].tr[lb]['y'] - fit
        if lb not in self.traces[self.trace_mode].active_traces:
            self.traces[self.trace_mode].active_traces.append(lb)

    def get_trace_at_point_tadata(self, point):
        xmode = self.data_obj.get_x_mode()
        if self.trace_mode == 'Kinetic':
            p = np.where(self.data_obj.spec_axis[xmode] >= point)
            if (self.data_obj.spec_axis[xmode][1]
                    > self.data_obj.spec_axis[xmode][0]):
                ind = 0
            else:
                ind = -1
            tr = self.data_obj.delA[:, p[0][ind]]
            self.traces[self.trace_mode].xdata = self.data_obj.time_delays
            self.traces[self.trace_mode].set_xrange("all")
            self.traces[self.trace_mode].xlabel = self.data_obj.set_ylabel()
            lb = " ".join([str(round(
                point, 2 if re.search('energ', xmode, re.I) else 0)),
                           self.data_obj.spec_unit_label])
        elif self.trace_mode == 'Spectral':
            self.traces[self.trace_mode].xdata = self.data_obj.spec_axis[xmode]
            tr = self.data_obj.delA[np.where(
                self.data_obj.time_delays >= point)[0][0], :]
            self.traces[self.trace_mode].xlabel = self.data_obj.set_xlabel()
            self.traces[self.trace_mode].set_xrange(
                self.data_obj.get_xlim())
            lb = str(round(point, 3)) + " " + self.data_obj.time_unit
        self.traces[self.trace_mode].tr[lb] = {'y': tr, 'val': point}
        if lb not in self.traces[self.trace_mode].active_traces:
            self.traces[self.trace_mode].active_traces.append(lb)

    def get_traces(self, point, increment=True):
        if increment:
            num_steps = self.vars['trace_plot_num_increments'].get()
            incr = self.vars['trace_plot_increment'].get()
            if type(self.dat) is TAData:
                max_val = (self.dat.get_xlim()[1]
                           if self.trace_mode == 'Kinetic'
                           else np.max(
                               self.dat.time_delays[self.dat.td_slice]))
            else:
                max_val = max(
                    self.dat.x) if self.trace_mode == 'Kinetic' else max(
                        self.dat.y)
        else:
            num_steps = 1
            incr = 0
            max_val = np.inf
        for i in range(num_steps):
            if point > max_val:
                break
            else:
                self.get_trace_at_point(point)
                point += incr

    def update_traces(self, set_current_traces=True, **kwargs):
        if set_current_traces:
            self.current_traces = self.traces[self.trace_mode]
        self.parent.traces = self.current_traces
        self.select.traces = self.current_traces
        self.select.update_box(**kwargs)

    def get_and_plot_traces(self, point, increment=False):
        self.trace_mode = self.vars['trace_mode'].get()
        if self.vars['add_trace_plots'].get() == 0:
            self.traces[self.trace_mode].tr = {}
            self.traces[self.trace_mode].active_traces = []
        self.get_traces(point, increment=increment)
        self.subtract_traces_callback()

    def trace_sel_update(self, xlim=None, ylim=None, update_plot=False):
        xlim = self.current_traces.set_xrange(xlim)
        if ylim is None:
            ylim = self.current_traces.auto_ylim()
        self.trace_figure.set_axes_lim(x=xlim, y=ylim, update_canvas=False)
        self.trace_figure.set_axes_label(x=self.current_traces.xlabel)
        try:
            self.plot_traces(update_canvas=True, update_plot=update_plot)
        except Exception:
            pass

    def plot_traces(self, *args, fig=None, update_plot=True, **plot_kwargs):
        if fig is None:
            fig = self.trace_figure
        fig.set_legend(entries=self.current_traces.active_traces)
        if update_plot:
            fig.plot_function = self.current_traces.plot
            fig.plot(**plot_kwargs)

    def save_traces(self):
        file = save_box(filetypes=[('text files', '.txt'),
                                   ('Matlab files', '.mat')],
                        fext='.txt', parent=self)
        if file is not None:
            self.current_traces.save_traces(
                file, save_fit=False, spec_quantity=self.data_obj.get_x_mode(),
                spec_unit=self.data_obj.spec_unit_label,
                trace_type=self.trace_mode,
                ylabel=self.trace_figure.get_ylabel(), range_ind=(
                    self.data_obj.get_xlim_indices()
                    if self.trace_mode in ('Spectral', 'RightSV') else None))


# %%
class AveragingOptionsWindow(tk.Toplevel):
    def __init__(self, parent, controller):
        tk.Toplevel.__init__(self, parent)
        self.frame = AveragingOptions(self)
        self.frame.grid(row=0, column=0, padx=5, pady=5)

        button_frame = tk.Frame(self)
        tk.ttk.Button(button_frame, text="OK", command=self.ok).grid(
            row=0, column=0, padx=5, pady=5)
        tk.ttk.Button(button_frame,
                      text="Cancel",
                      command=self.cancel).grid(
                          row=0, column=1, padx=5, pady=5)
        button_frame.grid(row=1, column=0, padx=5, pady=5)
        center_toplevel(self, controller)

    def ok(self):
        self.frame.get_options()
        self.output = self.frame.options
        self.destroy()

    def cancel(self):
        self.output = None
        self.destroy()


class AveragingOptions(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.output = [0, 0, 0]
        grid_kw = {'padx': 5, 'pady': 5, 'sticky': 'w'}
        tk.Label(self, text="Select options for averaging").grid(
                        row=0, column=0, columnspan=2, padx=5, pady=5)
        self.interp_nan = tk.IntVar(value=0)
        tk.ttk.Checkbutton(self, variable=self.interp_nan,
                           text="Interpolate NaN",
                           command=self.interp_nan_callback).grid(
                               row=1, column=0, sticky='w', padx=5, pady=5,
                               columnspan=4)
        self.nan_window_label = tk.ttk.Label(self, text="x window:")
        self.nan_window_lower = tk.DoubleVar(value=380)
        self.nan_window_upper = tk.DoubleVar(value=700)
        self.nan_window_lower_entry = tk.ttk.Entry(
            self, textvariable=self.nan_window_lower, width=5)

        self.nan_window_upper_entry = tk.ttk.Entry(
            self, textvariable=self.nan_window_upper, width=5)
        self.remove_outliers = tk.IntVar(value=0)
        tk.ttk.Checkbutton(self,
                           variable=self.remove_outliers,
                           text="Auto Remove Outliers",
                           command=self.outlier_removal_callback).grid(
                               row=4, column=0, sticky='w', padx=5, pady=5,
                               columnspan=4)

        self.outlier_opts = tk.Frame(self)
        tk.Label(self.outlier_opts, text="Outlier threshold:").grid(
            row=0, column=0, **grid_kw)
        self.outlier_threshold = tk.DoubleVar(value=1)
        tk.Entry(self.outlier_opts,
                 textvariable=self.outlier_threshold,
                 width=5).grid(
                     row=0, column=1, **grid_kw)

        self.outlier_mode = tk.StringVar(value='Derivative')
        tk.ttk.OptionMenu(self.outlier_opts, self.outlier_mode, 'Derivative',
                          'Derivative', 'Statistical').grid(
                              row=1, column=0, **grid_kw)

        self.pixel_mode = tk.StringVar(value='ROI:')
        tk.ttk.OptionMenu(
            self.outlier_opts, self.pixel_mode, 'ROI:', 'ROI:', 'pixelwise',
            command=self.pixel_mode_callback).grid(
                row=2, column=0, **grid_kw)

        self.roi_frame = tk.Frame(self.outlier_opts)
        self.roi_center = tk.DoubleVar(value=420)
        tk.Label(self.roi_frame, text='ROI center (nm):').grid(
            row=0, column=0, **grid_kw)
        tk.Entry(self.roi_frame, textvariable=self.roi_center,
                 width=5).grid(row=0, column=1, **grid_kw)
        self.roi_width = tk.IntVar(value=20)
        tk.Label(self.roi_frame, text='Width (pix.):').grid(
            row=1, column=0, **grid_kw)
        tk.Entry(self.roi_frame, textvariable=self.roi_width,
                 width=5).grid(row=1, column=1, **grid_kw)

        self.roi_frame.grid(row=3, column=0, columnspan=2)
        self.outlier_opts.grid(row=5, column=1, columnspan=3)

        self.outlier_removal_callback()

        self.weight_power = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self, variable=self.weight_power,
            text="Weight by pump power").grid(
                row=3, column=0, sticky='w',
                padx=5, pady=5, columnspan=4)

    def get_options(self):
        self.options = {'interpolate_nan': self.interp_nan.get(),
                        'nan_wl_window': [self.nan_window_lower.get(),
                                          self.nan_window_upper.get()],
                        'power_weight': self.weight_power.get(),
                        'outlier_threshold': self.outlier_threshold.get(),
                        'remove_outliers': self.remove_outliers.get()}
        if self.remove_outliers.get():
            self.options['wavelength_of_interest'] = self.roi_center.get()
            self.options['wavelength_of_interest_window'] = int(
                self.roi_width.get()/2)
            if self.pixel_mode.get() == 'pixelwise':
                self.options['outlier_mode'] = (self.outlier_mode.get()
                                                + 'pixel')
            else:
                self.options['outlier_mode'] = (self.outlier_mode.get()
                                                + 'ROI')

    def interp_nan_callback(self):
        if self.interp_nan.get():
            self.nan_window_label.grid(row=2, column=1, sticky='w')
            self.nan_window_lower_entry.grid(row=2, column=2, sticky='w')
            self.nan_window_upper_entry.grid(row=2, column=3, sticky='w')
        else:
            self.nan_window_label.grid_remove()
            self.nan_window_lower_entry.grid_remove()
            self.nan_window_upper_entry.grid_remove()

    def outlier_removal_callback(self, *args):
        if self.remove_outliers.get():
            for child in self.outlier_opts.winfo_children():
                try:
                    child.configure(state='normal')
                except Exception:
                    pass
            self.pixel_mode_callback()
        else:
            for child in self.outlier_opts.winfo_children():
                try:
                    child.configure(state='disabled')
                except Exception:
                    pass
            for child in self.roi_frame.winfo_children():
                try:
                    child.configure(state='disabled')
                except Exception:
                    pass

    def pixel_mode_callback(self, *args):
        if self.pixel_mode.get() == 'ROI:':
            for child in self.roi_frame.winfo_children():
                child.configure(state='normal')
        else:
            for child in self.roi_frame.winfo_children():
                child.configure(state='disabled')


class OutlierDisplay(tk.Toplevel):
    def __init__(self, parent, controller, outliers, max_rows=10):
        tk.Toplevel.__init__(self, parent)
        grid_kwargs = {'padx': 5, 'pady': 5, 'sticky': tk.W}
        tk.Label(self, text="Removed Outliers").grid(row=0, column=0,
                                                     columnspan=2,
                                                     padx=5, pady=5)
        tk.Label(self, text="File").grid(row=1, column=0, **grid_kwargs)
        tk.Label(self, text="No. of outliers").grid(
            row=1, column=1, **grid_kwargs)
        row = 2
        col = 0
        for key, val in outliers.items():
            if row > max_rows + 1:
                col += 2
                row = 2
                tk.Label(self, text="File").grid(
                    row=1, column=col, **grid_kwargs)
                tk.Label(self, text="No. of outliers").grid(
                    row=1, column=col + 1, **grid_kwargs)
            try:
                value = len(val)
            except Exception:
                value = val
            try:
                if value > 0:
                    tk.Label(self, text=key).grid(
                        row=row, column=col, **grid_kwargs)
                    tk.Label(self, text=value).grid(
                        row=row, column=col + 1, **grid_kwargs)
                    row += 1
            except Exception:
                pass
        if col > 0:
            row = max_rows + 2
        tk.ttk.Button(self, text='OK', command=self.destroy).grid(
            row=row, column=0, columnspan=col + 2, padx=5, pady=5)
        center_toplevel(self, controller)


# %%
class AveragingManager(tk.Toplevel):
    def __init__(self, parent, controller, all_timesteps_equal,
                 all_timesteps_same, data_obj, canvas_dim=None):
        tk.Toplevel.__init__(self, parent)

        self.all_timesteps_equal = all_timesteps_equal
        self.all_timesteps_same = all_timesteps_same
        self.data_obj = data_obj
        self.success = False
        self.controller = controller
        self.parent = parent
        self.removed_scans = []
        self.current_file = self.data_obj.files[0]
        self.manual_outliers = {}
        self.files = []
        ylimits = [0, len(self.data_obj.time_delays)]
        xlimits = [np.min(self.data_obj.wavelengths),
                   np.max(self.data_obj.wavelengths)]
        fig_kwargs = {}
        if canvas_dim is not None:
            fig_kwargs['dim'] = canvas_dim

        for f in self.data_obj.files:
            self.manual_outliers[f] = []
            self.files.append(f)

        self.figure = self.controller.tk_mpl_figure_wrap(
            self, num_subplots=2, xlimits=xlimits, ylimits=ylimits,
            xlabel=self.controller.settings['xlabel_default'],
            ylabel=self.controller.settings['ylabel_default'],
            # dim=canvas_dim,
            axes_titles=[None, "Scan Average"],
            callbacks={'button_press_event':
                       self.single_scan_plot_callback},
            **fig_kwargs)
        self.figure.grid(row=0, column=0, padx=5, pady=5)

        plot_opt_frame = CustomFrame(self, dim=(5, 1), border=False)

        tk.Label(plot_opt_frame, text="Y-Axis:").grid(
            row=0, column=0, sticky='w', padx=5, pady=5)
        self.ymode = tk.StringVar(value="time points")
        tk.ttk.OptionMenu(
            plot_opt_frame, self.ymode, "time points", "time points",
            "time delays", command=self.ymode_callback).grid(
                row=0, column=1, sticky='w', padx=5, pady=5, columnspan=2)

        self.ylower = tk.DoubleVar(value=ylimits[0])
        self.yupper = tk.DoubleVar(value=ylimits[1])
        tk.Label(plot_opt_frame, text='Y Limits:').grid(
            row=1, column=0, sticky='w', padx=5, pady=5)
        self.yupper_entry = tk.Entry(
            plot_opt_frame, textvariable=self.yupper, width=8)
        self.yupper_entry.grid(
            row=1, column=2, sticky='w', padx=5, pady=5)
        self.yupper_entry.bind('<Return>', self.set_ylim)
        self.ylower_entry = tk.Entry(
            plot_opt_frame, textvariable=self.ylower, width=8)
        self.ylower_entry.grid(
            row=1, column=1, sticky='w', padx=5, pady=5)
        self.ylower_entry.bind('<Return>', self.set_ylim)
        tk.ttk.Button(plot_opt_frame, text="Full Window:",
                      command=lambda a=True: self.set_ylim(
                          autolim=a)).grid(
                              row=1, column=3, sticky='w', padx=5, pady=5)
        self.autolim_mode = tk.StringVar(value="Current Scan")
        tk.ttk.OptionMenu(plot_opt_frame, self.autolim_mode, "Current Scan",
                          "Current Scan", "Scan Average",
                          command=lambda *args, a=True: self.set_ylim(
                              autolim=a)).grid(
                                  row=1, column=4, sticky='w', padx=5, pady=5)

        self.xlower = tk.DoubleVar(value=xlimits[0])
        self.xupper = tk.DoubleVar(value=xlimits[1])
        tk.Label(plot_opt_frame, text='X Limits:').grid(
            row=2, column=0, sticky='w', padx=5, pady=5)
        self.xlower_entry = tk.Entry(plot_opt_frame, textvariable=self.xlower,
                                     width=8)
        self.xlower_entry.bind('<Return>', self.set_xlim)
        self.xlower_entry.grid(row=2, column=1, sticky='w', padx=5, pady=5)
        self.xupper_entry = tk.Entry(plot_opt_frame, textvariable=self.xupper,
                                     width=8)
        self.xupper_entry.bind('<Return>', self.set_xlim)
        self.xupper_entry.grid(row=2, column=2, sticky='w', padx=5, pady=5)

        plot_opt_frame.grid(row=2, column=0, sticky='wnse', padx=5, pady=5)

        self.contrast_slider = ContrastSlider(self, self.figure)
        self.contrast_slider.grid(row=0, column=1)

        tk.ttk.Button(self, text="Color Map",
                      command=self.open_color_editor).grid(
                          row=1, column=1, sticky='nwe')

        ui_frame = CustomFrame(self, dim=(2, 2), border=True)

        scan_select_frame = GroupBox(
            ui_frame, dim=(4, 4), text="Scan Selection")

        tk.Label(scan_select_frame, text="Included Scans").grid(
            row=0, column=0, padx=5, pady=5, columnspan=2)
        self.scan_select = tk.Listbox(scan_select_frame)
        self.scan_select.bind("<<ListboxSelect>>", self.scan_select_callback)
        self.scan_select.grid(row=1, column=0)

        self.scan_select_yscroll = tk.Scrollbar(scan_select_frame)

        self.scan_select.config(yscrollcommand=self.scan_select_yscroll.set)
        self.scan_select_yscroll.config(command=self.scan_select.yview)
        self.scan_select_yscroll.grid(row=1, column=1, sticky='nse')

        self.scan_select_xscroll = tk.Scrollbar(
            scan_select_frame, orient=tk.HORIZONTAL)

        self.scan_select.config(xscrollcommand=self.scan_select_xscroll.set)
        self.scan_select_xscroll.config(command=self.scan_select.xview)
        self.scan_select_xscroll.grid(row=2, column=0, sticky='nwe')

        tk.ttk.Button(scan_select_frame, text="Remove",
                      command=self.remove_scan).grid(
                          row=3, column=0)

        tk.Label(scan_select_frame, text="Removed Scans").grid(
            row=0, column=2, padx=5, pady=5, columnspan=2)
        self.removed_scans_select = tk.Listbox(scan_select_frame)
        self.removed_scans_select.grid(row=1, column=2)
        self.removed_scans_scroll = tk.Scrollbar(scan_select_frame)
        self.removed_scans_select.config(
            yscrollcommand=self.removed_scans_scroll.set)
        self.removed_scans_scroll.config(
            command=self.removed_scans_select.yview)
        self.removed_scans_scroll.grid(row=1, column=3, sticky='nse')

        tk.ttk.Button(
            scan_select_frame, text="Add", command=self.add_scan).grid(
                row=3, column=2)

        scan_select_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nwse',
                               columnspan=2)

        tk.Label(ui_frame, text="Scan processing")
        manual_outlier_frame = tk.Frame(ui_frame)
        tk.Label(manual_outlier_frame, text="Manual Outlier Selection").grid(
            row=0, column=0, columnspan=2)
        self.manual_outliers_select = tk.Listbox(manual_outlier_frame)
        self.manual_outliers_select.grid(row=1, column=0)
        self.outlier_select_scroll = tk.Scrollbar(manual_outlier_frame)
        self.manual_outliers_select.config(
            yscrollcommand=self.outlier_select_scroll.set)
        self.outlier_select_scroll.config(
            command=self.manual_outliers_select.yview)
        self.outlier_select_scroll.grid(row=1, column=1, sticky='nse')

        tk.ttk.Button(manual_outlier_frame, text="Unmark",
                      command=self.unmark_current_outlier).grid(
                          row=2, column=0, padx=5, pady=5)
        tk.Label(manual_outlier_frame,
                 text="Add/Remove time point:").grid(row=3, column=0,
                                                     padx=5, pady=5)
        self.manual_outliers_ = tk.DoubleVar()
        self.manual_outliers_entry = tk.Entry(
            manual_outlier_frame, textvariable=self.manual_outliers_, width=8)
        self.manual_outliers_entry.grid(row=4, column=0, padx=5, pady=5)
        self.manual_outliers_entry.bind('<Return>',
                                        self.outlier_entry_callback)

        manual_outlier_frame.grid(row=2, column=0, padx=5,
                                  pady=5, sticky='wnse')

        self.auto_opts = AveragingOptions(ui_frame)
        self.auto_opts.grid(row=2, column=1)

        self.auto_opts.outlier_mode.trace(
            'w', self.auto_outlier_mode_callback)

        button_frame = tk.Frame(ui_frame)
        tk.ttk.Button(button_frame, text="Update Average",
                      command=self.update_average).grid(
                          row=0, column=0, padx=5, pady=5)
        tk.ttk.Button(button_frame, text="Continue",
                      command=self.average_and_continue).grid(
                          row=0, column=1, padx=5, pady=5)
        button_frame.grid(columnspan=2, column=0, row=3, sticky='wnse')

        ui_frame.grid(row=0, column=2, rowspan=3, padx=5, pady=5,
                      sticky='nwse')

        self.fill_scan_select()
        self.scan_select.select_set(0)
        self.scan_select_callback()
        self.figure.plot(i=1)

    def auto_outlier_mode_callback(self, *args):
        if (self.auto_opts.outlier_mode.get() == 'Statistical'
                and len(self.files) <= 2):
            self.auto_opts.outlier_mode.set('Derivative')
            messagebox.showerror(
                parent=self,
                message="At least three spectra per time point required for"
                        + "statistical removal.\nUsing Derivative")
        elif (not self.all_timesteps_same
              and self.auto_opts.outlier_mode.get() == 'Statistical'):
            messagebox.showwarning(
                parent=self,
                message="At least three spectra per time point required for"
                        + "statistical removal.\n"
                        + "Outlier removal may not work properly.")

    def outlier_entry_callback(self, *args):
        try:
            self.toggle_outlier(self.manual_outliers_.get())
        except Exception as e:
            messagebox.showerror(message=e, parent=self)

    def unmark_current_outlier(self):
        try:
            ind, = self.manual_outliers_select.curselection()
        except Exception:
            raise
        else:
            self.unmark_outlier(
                self.data_obj.raw_scans[self.current_file]['time_delays'][
                    self.manual_outliers[self.current_file][ind]])

    def single_scan_plot_callback(self, event):
        if event.button == 3:
            self.toggle_outlier(event.ydata)

    def get_time_index(self, point):
        td = self.data_obj.raw_scans[self.current_file]['time_delays']
        try:
            upper = np.where(td >= point)[0][0]
            lower = np.where(td < point)[0][-1]
        except Exception:
            return None
        if np.abs(point - td[upper]) < np.abs(point - td[lower]):
            return upper
        else:
            return lower

    def toggle_outlier(self, point):
        ind = self.get_time_index(point)
        if ind in self.manual_outliers[self.current_file]:
            self.manual_outliers[self.current_file].remove(ind)
        else:
            self.manual_outliers[self.current_file].append(ind)
        self.update_outliers()

    def mark_outlier(self, point):
        ind = self.get_time_index(point)
        if ind not in self.manual_outliers[self.current_file]:
            self.manual_outliers[self.current_file].append(ind)
            self.update_outliers()

    def unmark_outlier(self, point):
        ind = self.get_time_index(point)
        if ind in self.manual_outliers[self.current_file]:
            self.manual_outliers[self.current_file].remove(ind)
            self.update_outliers()

    def update_outliers(self):
        self.manual_outliers_select.delete(0, tk.END)
        for ind in self.manual_outliers[self.current_file]:
            self.manual_outliers_select.insert(
                tk.END, self.data_obj.raw_scans[
                    self.current_file]['time_delays'][ind])

    def average_and_continue(self):
        if not self.success:
            self.update_average()
        self.destroy()

    def update_average(self):
        self.auto_opts.get_options()
        pbar = CustomProgressbarWindow(self, controller=self,
                                       text="Scan averaging",
                                       cancel_button=True)
        outliers = self.data_obj.average_scans(
            self.all_timesteps_equal, self.all_timesteps_same,
            progressbar=pbar, manual_outliers=self.manual_outliers,
            files_to_average=self.files, **self.auto_opts.options)
        self.update_plotfunc()
        self.figure.plot(i=1)
        if outliers is not False:
            self.success = True
            if (self.auto_opts.options['remove_outliers']
                    and outliers is not None):
                OutlierDisplay(self, self, outliers)
        pbar.destroy()

    def ymode_callback(self, *args):
        if re.search('delay', self.ymode.get(), re.I):
            self.yupper.set(np.max(self.data_obj.time_delays))
            self.ylower.set(np.min(self.data_obj.time_delays))
        else:
            self.yupper.set(len(self.data_obj.time_delays) - 1)
            self.ylower.set(0)
        self.figure.set_axes_lim_all(y=[self.ylower.get(), self.yupper.get()])
        self.update_plotfunc()
        self.figure.plot_all()

    def open_color_editor(self, *args):
        win = tk.Toplevel(self)
        center_toplevel(win, self)
        color_edit = ColorMapEditor(win, self.figure, horizontal=True,
                                    contrast_slider=self.contrast_slider,
                                    num_sub_plots=len(self.figure.axes))
        color_edit.set_axes_to_edit(i="all")
        color_edit.grid(padx=5, pady=5)

    def plot_function(self, y, i, *args, **kwargs):
        if i == 0:
            return self.data_obj.plot_raw_scan(
                self.current_file, *args, yvalues=y[0], **kwargs)
        else:
            return self.data_obj.plot_ta_map(*args, **kwargs, write_map=True,
                                             yvalues=y[1])

    def update_plotfunc(self):
        if re.search('delay', self.ymode.get(), re.I):
            y = [None, None]
            self.figure.set_ylabel("time delay", update_canvas=False)
        else:
            y = [list(range(len(
                self.data_obj.raw_scans[self.current_file]['time_delays'])))]
            y.append(list(range(len(self.data_obj.time_delays))))
            self.figure.set_ylabel("time point", update_canvas=False)
        self.figure.plot_function = lambda *args, **kwargs: self.plot_function(
            y, *args, **kwargs)
        title = self.current_file.split('/')[-1]
        self.figure.set_axes_title(title=title, i=0, update_canvas=False)

    def set_xlim(self, *args):
        self.figure.set_axes_lim_all(x=[self.xlower.get(), self.xupper.get()])

    def set_ylim(self, *args, autolim=False):
        if not autolim:
            ylimits = [self.ylower.get(), self.yupper.get()]
        else:
            if self.autolim_mode.get() == 'Scan Average':
                y = self.data_obj.time_delays
            elif self.autolim_mode.get() == 'Current Scan':
                y = self.data_obj.raw_scans[self.current_file]['time_delays']
            else:
                return
            if re.search('delay', self.ymode.get(), re.I):
                ylimits = [np.max(y), np.min(y)]
            else:
                ylimits = [len(y) - 1, 0]
            self.yupper.set(ylimits[1])
            self.ylower.set(ylimits[0])
        self.figure.set_axes_lim_all(y=ylimits)

    def fill_scan_select(self):
        self.scan_select.delete(0, tk.END)
        for f in self.files:
            self.scan_select.insert(tk.END, f)

    def fill_remove_scans_select(self):
        self.removed_scans_select.delete(0, tk.END)
        for f in self.removed_scans:
            self.removed_scans_select.insert(tk.END, f)

    def remove_scan(self):
        try:
            ind, = self.scan_select.curselection()
        except Exception:
            return
        self.removed_scans_select.insert(tk.END, self.files[ind])
        self.removed_scans.append(self.files[ind])
        self.files.pop(ind)
        self.fill_scan_select()

    def add_scan(self):
        try:
            ind, = self.removed_scans_select.curselection()
        except Exception:
            return
        self.scan_select.insert(tk.END, self.removed_scans[ind])
        self.files.append(self.removed_scans[ind])
        self.removed_scans.pop(ind)
        self.fill_remove_scans_select()

    def scan_select_callback(self, *args):
        try:
            ind, = self.scan_select.curselection()
        except Exception:
            return
        else:
            self.current_file = self.files[ind]
            self.update_plotfunc()
            self.figure.plot(i=0)
            self.update_outliers()


# %%
class ChirpCorrPage(CustomFrame):
    def __init__(self, parent, controller, data_obj):
        CustomFrame.__init__(self, parent, dim=(2, 3))

        self.controller = controller
        self.data_obj = data_obj
        # misc. variables

        self.t0_found = False
        self.t0_fit_done = False
        self._init_outlier_arrays()
        self.outlier_commands = {'mark': self.add_outlier,
                                 'remove': self.remove_outlier,
                                 None: lambda i: (),
                                 'mark_immune': self.immune_outlier}
        self.outlier_points = {}
        self.rect_select_mode = None

        self.frames = {}
        self.vars = {}
#        self.frames['mainToolbar'] = tk.Frame(self)
#        self.frames['t0Toolbar'] = tk.Frame(self)
        self.frames['commands'] = CustomFrame(self, dim=(3, 1))
        self.frames['find_t0'] = GroupBox(self.frames['commands'],
                                          text="Find time zeros",
                                          dim=(3, 4))
        self.frames['fit_t0'] = GroupBox(self.frames['commands'],
                                         text="Fit time zeros",
                                         dim=(3, 2))
        self.frames['correct_chirp'] = GroupBox(self.frames['commands'],
                                                text="Chirp correction",
                                                dim=(3, 4))
        self.frames['outliers'] = GroupBox(self.frames['fit_t0'],
                                           text="Outlier rejection",
                                           dim=(5, 2))
        self.frames['outliers_sub'] = CustomFrame(self.frames['outliers'],
                                                  dim=(4, 1))
        self.ta_map_figure = self.controller.tk_mpl_figure_wrap(
            self, plot_function=self.data_obj.plot_ta_map,
            xlabels=self.controller.settings['xlabel_default'],
            ylabels=self.controller.settings['ylabel_default'],
            clables=self.data_obj.zlabel,
            callbacks={'button_press_event':
                       self.map_figure_callback})
        self.ta_map_figure.grid(row=0, column=0, padx=(0, 10), sticky='wnse')
        self.t0_figure = self.controller.tk_mpl_figure_wrap(
            self, plot_function=None,
            xlabels=self.controller.settings['xlabel_default'],
            ylabel=self.controller.settings['ylabel_default'],
            plot_type='linefit',
            invert_zaxis=self.controller.settings['2Dmap_invert_y'],
            transpose=self.controller.settings['2Dmap_transpose'],
            callbacks={'button_press_event': self.t0_plot_callback})
        self.t0_figure.grid(row=0, column=1, sticky='wens')

        self.t0_axes = self.t0_figure.get_axes()
        ttk.Button(
            self.frames['correct_chirp'],
            text="Correct Chirp",
            command=lambda: self.run_correction(controller)).grid(
                row=0, column=0, columnspan=2, sticky='we', pady=5, padx=5)

        self.enable_max_timestep = tk.IntVar(value=0)
        self.enable_max_timestep_check = tk.ttk.Checkbutton(
            self.frames['correct_chirp'], text='Max. time step:',
            variable=self.enable_max_timestep,
            command=self.enable_max_timestep_callback)
        self.enable_max_timestep_check.grid(row=1, column=0, sticky='we')
        self.max_timestep = tk.DoubleVar(value=10)
        self.max_timestep_entry = tk.ttk.Entry(self.frames['correct_chirp'],
                                               textvariable=self.max_timestep,
                                               width=5)
        self.max_timestep_entry.config(state='disabled')
        self.max_timestep_entry.grid(row=1, column=1, sticky='w')

        self.cancel_button = tk.ttk.Button(self.frames['correct_chirp'],
                                           text='Cancel',
                                           command=self.cancel_correction)
        self.cancel_button.config(state='disabled')
        self.cancel_button.grid(
            row=2, column=0, columnspan=2, sticky='we', pady=5, padx=5)

        ttk.Button(
            self.frames['correct_chirp'],
            text="Reset Correction",
            command=self.reset_correction).grid(
                row=3, column=0, columnspan=2, pady=5, sticky='we', padx=5)

        self.timestep_disp = GroupBox(
            self.frames['correct_chirp'], text='Time steps')
        self.timestep_disp.grid(row=0, column=2, rowspan=4, padx=5, pady=5,
                                sticky='wne')
        self.timestep_labels = []
        # Find time zero panel

        self.t0_dict = {'Threshold': ['derivative',
                                      'threshold real',
                                      'threshold abs.'],
                        'Load chirp file': ['fit and time zeros',
                                            'time zeros only'],
                        'Manual': ['time zeros only']}
        self.t0_method = tk.StringVar(value='Threshold')
        self.t0_algo = tk.StringVar(value='derivative')

        tk.ttk.Label(self.frames['find_t0'], text="Method:").grid(
            row=0, column=0, sticky='w')
        t0_method_select = tk.ttk.OptionMenu(
            self.frames['find_t0'], self.t0_method, 'Threshold',
            *self.t0_dict.keys())
        t0_method_select.config(width=15)

        t0_method_select.grid(row=0, column=1, sticky='we')

        tk.Label(self.frames['find_t0'], text="Signal threshold:").grid(
            row=2, column=0, sticky='we')
        self.irf_threshold = tk.DoubleVar(value=10)
        self.irf_threshold_entry = tk.Entry(
            self.frames['find_t0'], textvariable=self.irf_threshold, width=5)
        self.irf_threshold_entry.grid(row=2, column=1, sticky=tk.E)

        tk.ttk.Label(self.frames['find_t0'],
                     text="Signal type:").grid(row=1, column=0, sticky='w')
        self.find_t0_algo_select = tk.ttk.OptionMenu(
                                    self.frames['find_t0'],
                                    self.t0_algo,
                                    self.t0_algo.get(),
                                    *self.t0_dict[self.t0_method.get()])
        self.find_t0_algo_select.config(width=15)
        self.find_t0_algo_select.grid(row=1, column=1, sticky='we')
        ttk.Button(self.frames['find_t0'],
                   text="Get time zeros",
                   command=self.get_time_zeros).grid(
                       row=3, column=0, sticky='we')
        ttk.Button(self.frames['find_t0'],
                   text="Save to file",
                   command=self.save_t0).grid(
                       row=3, column=1, sticky='we', pady=5,  padx=5)

        self.t0_method.trace('w', self.t0_method_callback)

        # Fit time zero panel
        tk.Label(self.frames['fit_t0'],
                 text="Fit function:").grid(row=0, column=0,
                                            sticky=tk.W, padx=5)
        self.t0_fit_func = tk.StringVar(value='5th order polyn.')
        tk.ttk.OptionMenu(
            self.frames['fit_t0'], self.t0_fit_func, self.t0_fit_func.get(),
            '3rd order polyn.', '5th order polyn.', '7th order polyn.').grid(
                row=0, column=1, sticky=tk.W)

        ttk.Button(
            self.frames['fit_t0'], text="Run fit",
            command=self.fit_time_zero).grid(
                row=0, column=2, sticky=tk.E)

        # outlier sub panel

        tk.Label(self.frames['outliers'],
                 text='Auto finding thresholds:').grid(row=0, column=0,
                                                       sticky=tk.W)
        tk.Label(self.frames['outliers'],
                 text='absolute:').grid(row=0, column=1, sticky=tk.E)

        self.t0_fit_outlier_thresh_abs = tk.DoubleVar(value=4)
        tk.Entry(
            self.frames['outliers'],
            textvariable=self.t0_fit_outlier_thresh_abs,
            width=5).grid(row=0, column=2, sticky=tk.W)
        tk.Label(self.frames['outliers'], text='relative:').grid(
            row=0, column=3, sticky=tk.E)
        self.t0_fit_outlier_thresh_rel = tk.DoubleVar(value=4)
        tk.Entry(self.frames['outliers'],
                 textvariable=self.t0_fit_outlier_thresh_rel,
                 width=5).grid(row=0, column=4, sticky=tk.W)

        ttk.Button(self.frames['outliers_sub'],
                   text="Mark outliers",
                   command=self.mark_outlier_callback).grid(
                       row=1, column=0, sticky='we', padx=2)
        ttk.Button(self.frames['outliers_sub'],
                   text="Remove marks",
                   command=self.remove_outlier_callback).grid(
                       row=1, column=3, columnspan=2, sticky='we', padx=2)
        ttk.Button(self.frames['outliers_sub'],
                   text="Mark to include",
                   command=self.include_outlier_callback).grid(
                       row=1, column=1, columnspan=2, sticky='we', padx=2)
        ttk.Button(self.frames['outliers_sub'],
                   text="Reset marks",
                   command=self.reset_outliers).grid(
                       row=1, column=5, sticky='we', padx=2)
        self.init_rect_select()

        # grid frames
        self.frames['outliers_sub'].grid(row=1, column=0, columnspan=5,
                                         sticky='wnse')
        self.frames['outliers'].grid(row=1, column=0, columnspan=3,
                                     sticky='wnse', padx=5, pady=5, rowspan=3)
        self.frames['find_t0'].grid(row=0, column=0,
                                    sticky='nswe', padx=(0, 5), pady=5)
        self.frames['fit_t0'].grid(row=0, column=1, sticky='nswe', padx=5,
                                   pady=5, columnspan=2)
        self.frames['correct_chirp'].grid(row=0, column=3, sticky='nswe',
                                          padx=(5, 0), pady=5)
        self.frames['commands'].grid(row=2, column=0, columnspan=2,
                                     sticky='n')

    def write_app_settings(self):
        for key, val in self.vars.items():
            try:
                self.controller.app_settings[ChirpCorrPage][key] = val.get()
            except Exception as e:
                print(e)

    def load_app_settings(self):
        for key in self.vars.keys():
            try:
                self.vars[key].set(
                    self.controller.app_settings[ChirpCorrPage][key])
            except Exception as e:
                print(e)

    # time zero finding/loading

    def t0_method_callback(self, *args):
        opts = self.t0_dict[self.t0_method.get()]
        self.t0_algo.set(opts[0])
        menu = self.find_t0_algo_select['menu']
        menu.delete(0, 'end')
        for opt in opts:
            menu.add_command(
                label=opt, command=lambda o=opt: self.t0_algo.set(o))
        self.get_time_zeros()

    def load_t0_file(self):
        file = filedialog.askopenfile(
            filetypes=[('Text files', '.txt .dat'), ('all files', '.*')])
        self.data_obj.load_time_zero_file(file)
        self.plot_t0()
        self.t0_found = True
        self.t0_fit_done = False

    def get_chirp_manually(self):
        window = ManualChirpWindow(
            self, self.controller.frames[MainPage].main_figure, self.data_obj,
            controller=self.controller)
        self.wait_window(window)
        self.manual_t0 = window.points
        if len(self.manual_t0) > 0:
            self.data_obj.time_zeros = np.zeros((len(self.manual_t0), 2))
            for i in range(len(self.manual_t0)):
                self.data_obj.time_zeros[i, 0] = self.manual_t0[i][0]
                self.data_obj.time_zeros[i, 1] = self.manual_t0[i][1]
            try:
                self.plot_t0()
            except Exception:
                messagebox.showerror('error',
                                     "Error plotting time zero values.",
                                     parent=self)
                raise
            else:
                self.t0_found = True
                self.t0_fit_done = False

    def get_time_zeros(self):
        if re.search('load', self.t0_method.get(), re.I):
            try:
                self.load_t0_file()
                self.t0_found = True
            except Exception as e:
                self.t0_found = False
                messagebox.showerror("Error", e, parent=self)
        elif re.search('manual', self.t0_method.get(), re.I):
            try:
                self.get_chirp_manually()
                self.t0_found = True
            except Exception as e:
                self.t0_found = False
                messagebox.showerror("Error", e, parent=self)
                raise
        else:
            self.auto_t0()
        self._init_outlier_arrays()

    def map_figure_callback(self, event):
        if (event.dblclick or event.button == 3):
            self.controller.open_figure_options(
                self, self.ta_map_figure, controller=self.controller)

    def auto_t0(self):
        try:
            self.data_obj.find_time_zeros(self.irf_threshold.get(),
                                          algorithm=self.t0_algo.get())
            self.t0_found = True
        except Exception:
            self.t0_found = False
        try:
            self.plot_t0()
            self.t0_fit_done = False
        except Exception as e:
            messagebox.showerror(message=e, parent=self)

    def plot_t0(self, only=True):
        self.t0_axes.cla()
        self.init_rect_select()
        self.t0_axes.plot(self.data_obj.time_zeros[:, 0],
                          self.data_obj.time_zeros[:, 1],
                          'o', color='blue', markersize=5)
        self.t0_figure.set_legend(entries=['time zeros'])
        self.t0_found = True
        if only:
            try:
                self.set_t0_axlim()
            except Exception:
                pass

    def set_t0_axlim(self):
        if self.t0_method.get() == 'Threshold':
            t0 = self.data_obj.time_zeros[self.data_obj.get_xlim_indices(), :]
        else:
            t0 = self.data_obj.time_zeros
        ylim = [np.min(np.ma.array(t0[:, 1], mask=np.isnan(t0[:, 1]))),
                np.max(np.ma.array(t0[:, 1], mask=np.isnan(t0[:, 1])))]
        xlim = [np.min(np.ma.array(t0[:, 0], mask=np.isnan(t0[:, 0]))),
                np.max(np.ma.array(t0[:, 0], mask=np.isnan(t0[:, 0])))]
        ydiff = np.abs(ylim[0] - ylim[1])
        xdiff = np.abs(xlim[0] - xlim[1])
        if ydiff == 0:
            ydiff = np.max(np.abs(ylim))
        ylim = [ylim[0]-0.1*ydiff, ylim[1]+0.1*ydiff]
        xlim = [xlim[0]-0.1*xdiff, xlim[1]+0.1*xdiff]
        self.t0_figure.set_axes_lim(x=xlim, y=ylim, update_canvas=True)
        self.ta_map_figure.set_ylim(ylim, update_canvas=True)

    # time zero fit
    def fit_time_zero(self):
        try:
            try:
                non_outliers = self.t0_non_outliers[:, 0]
            except Exception:
                non_outliers = []
            self.outliers = self.data_obj.fit_time_zeros(
                polyorder=self.t0_fit_func.get()[0],
                outlier_thresh_rel=self.t0_fit_outlier_thresh_rel.get(),
                ind=[0, len(self.data_obj.time_zeros) - 1],
                forced_outliers=np.transpose(
                    np.array(self.forced_t0_outliers)),
                non_outliers=non_outliers)
        except TypeError:
            raise
            m = 'Insufficient number of time zero values for fit.'
            if len(self.data_obj.time_zeros) == 0:
                m = m + "\nPlease run time zero finding first."
            messagebox.showerror(message=m, parent=self)
            self.t0_fit_done = False
        except Exception as e:
            raise
            messagebox.showerror(message=e, parent=self)
        else:
            if self.t0_fit_done:
                self.plot_t0(only=False)
            self.t0_axes.plot(
                self.data_obj.wavelengths[self.data_obj.get_xlim_slice()],
                self.data_obj.time_zero_fit[self.data_obj.get_xlim_slice()],
                color=self.controller.settings['fit_color'])
            self.t0_figure.set_xlim(self.data_obj.get_lambda_lim())
            self.plot_outliers()
            self.set_t0_axlim()
            self.t0_fit_done = True

    # time zero fit outlier handling
    def plot_outliers(self):
        try:
            [self.t0_figure.get_axes().lines.remove(p)
             for p in self.outlier_points.values()]
        except Exception:
            pass
        self.outlier_points = {}
        if self.t0_fit_done:
            leg = ['time zeros', 'fit']
        else:
            leg = ['time zeros']
        try:
            if len(self.outliers[:, 0]) > 0:
                self.outlier_points['auto'], = self.t0_axes.plot(
                    self.outliers[0], self.outliers[1], 'h', color='orange')
                leg.append('outliers')
        except Exception:
            pass
        try:
            if len(self.forced_t0_outliers[:, 0]) > 0:
                self.outlier_points['forced'], = self.t0_axes.plot(
                    self.forced_t0_outliers[:, 0],
                    self.forced_t0_outliers[:, 1],
                    'h', color='red')
                leg.append('forced outliers')
        except Exception:
            pass
        try:
            if len(self.t0_non_outliers[:, 0]) > 0:
                self.outlier_points['immune'], = self.t0_axes.plot(
                    self.t0_non_outliers[:, 0],
                    self.t0_non_outliers[:, 1],
                    'h', color='green')
                leg.append('immune')
        except Exception:
            pass
        self.t0_figure.set_legend(entries=leg, update_canvas=True)

    def reset_outliers(self):
        self._init_outlier_arrays()
        self.rs.set_visible(False)
        self.plot_outliers()

    def _init_outlier_arrays(self):
        self.forced_t0_outliers = np.array([[], []])
        self.t0_non_outliers = np.array([[], []])
        self.outliers = np.array([[], []])

    def include_outlier_callback(self):
        self.rect_select_mode = 'mark_immune'

    def mark_outlier_callback(self):
        self.rect_select_mode = 'mark'

    def remove_outlier_callback(self):
        self.rect_select_mode = 'remove'

    def init_rect_select(self):
        self.rs = RectangleSelector(self.t0_figure.get_axes(),
                                    self.line_select_callback, drawtype='box',
                                    useblit=False,
                                    button=[1], minspanx=1, minspany=1,
                                    spancoords='pixels', interactive=True)

    def line_select_callback(self, eclick, erelease):
        self.rs.set_visible(True)
        inds = []
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        xmax, xmin = np.max([x1, x2]), np.min([x1, x2])
        ymax, ymin = np.max([y1, y2]), np.min([y1, y2])
        bool_arr = np.logical_and(np.logical_and(
            self.data_obj.time_zeros[:, 0] >= xmin,
            self.data_obj.time_zeros[:, 0] <= xmax),
            np.logical_and(self.data_obj.time_zeros[:, 1] >= ymin,
                           self.data_obj.time_zeros[:, 1] <= ymax))
        xvals = self.data_obj.time_zeros[bool_arr, 0]
        yvals = self.data_obj.time_zeros[bool_arr, 1]
        for x, y in zip(xvals, yvals):
            arr = np.add(np.abs(self.data_obj.time_zeros[:, 0] - x),
                         np.abs(self.data_obj.time_zeros[:, 1] - y))
            inds.append(np.ma.array(arr, mask=np.isnan(arr)).argmin())
        for i in inds:
            self.outlier_commands[self.rect_select_mode](i)
        self.plot_outliers()

    def t0_plot_callback(self, event):
        if self.t0_found and event.button == 3:
            arr = np.add(
                np.abs(self.data_obj.time_zeros[:, 0] - event.xdata),
                np.abs(self.data_obj.time_zeros[:, 1] - event.ydata))
            i = np.ma.array(arr, mask=np.isnan(arr)).argmin()
            self.toggle_outlier(i)
            self.plot_outliers()
        elif event.button == 3 or event.dblclick:
            self.controller.open_figure_options(
                self, self.t0_figure, controller=self.controller)

    def toggle_outlier(self, idx):
        try:
            if (np.any(np.isin(self.forced_t0_outliers[:, 0],
                               self.data_obj.wavelengths[idx]))
                or np.any(np.isin(self.t0_non_outliers[:, 0],
                                  self.data_obj.wavelengths[idx]))):
                self.remove_outlier(idx)
            else:
                self.outlier_commands[self.rect_select_mode](idx)
        except Exception:
            self.outlier_commands[self.rect_select_mode](idx)

    def immune_outlier(self, idx):
        self.remove_outlier(idx)
        try:
            self.t0_non_outliers = np.vstack((
                np.append(self.t0_non_outliers[:, 0],
                          self.data_obj.time_zeros[idx, 0]),
                np.append(self.t0_non_outliers[:, 1],
                          self.data_obj.time_zeros[idx, 1]))).T
        except Exception:
            self.t0_non_outliers = np.array(
                [[self.data_obj.time_zeros[idx, 0],
                  self.data_obj.time_zeros[idx, 1]]])

    def add_outlier(self, idx):
        try:
            self.forced_t0_outliers = np.vstack((
                np.append(self.forced_t0_outliers[:, 0],
                          self.data_obj.time_zeros[idx, 0]),
                np.append(self.forced_t0_outliers[:, 1],
                          self.data_obj.time_zeros[idx, 1]))).T
        except Exception:
            self.forced_t0_outliers = np.array(
                [[self.data_obj.time_zeros[idx, 0],
                  self.data_obj.time_zeros[idx, 1]]])

    def remove_outlier(self, idx):
        try:
            self.forced_t0_outliers = self.forced_t0_outliers[np.isin(
                self.forced_t0_outliers[:, 0], self.data_obj.wavelengths[idx],
                invert=True)]
        except Exception:
            pass
        try:
            self.t0_non_outliers = self.t0_non_outliers[np.isin(
                self.t0_non_outliers[:, 0], self.data_obj.wavelengths[idx],
                invert=True)]
        except Exception:
            pass

    # chirp correction
    def enable_max_timestep_callback(self, *args):
        if self.enable_max_timestep.get():
            self.max_timestep_entry.config(state='normal')
        else:
            self.max_timestep_entry.config(state='disabled')

    def show_timestep_disp(self, visible=True):
        for lbl in self.timestep_labels:
            lbl.grid_forget()
            del lbl
        self.timestep_labels = []
        self.timestep_disp.grid_remove()
        if visible:
            row = 0
            col = 0
            for ts in self.data_obj.timesteps:
                if row > 5:
                    row = 0
                    col += 1
                self.timestep_labels.append(tk.ttk.Label(self.timestep_disp,
                                                         text=str(ts)))
                self.timestep_labels[-1].grid(row=row, column=col)
                row += 1
            self.timestep_disp.grid()

    def _update_progressbar(self, in_queue):
        while self._corr_running:
            d = in_queue.get()
            try:
                self.progbar.set_val_ratio(d['i']/d['n'])
            except Exception:
                pass
            if 'label' in d.keys():
                try:
                    self.progbar.update_label(d['label'])
                except Exception:
                    pass
            self.progbar.update_timer()

    def run_correction(self, controller):
        def start_thread(*args, **kwargs):
            # def func(*args, **kwargs):
            #     return self.data_obj.correct_chirp(
            #         *args,
            #         max_time_step=self.max_timestep.get()
            #         if self.enable_max_timestep.get() else None,
            #         **kwargs)
            self._corr_running = True
            self.queue = Queue()
            self.task = ThreadedTask(
                lambda *ar, **kw: self.data_obj.correct_chirp(
                    *ar,
                    max_time_step=self.max_timestep.get()
                    if self.enable_max_timestep.get() else None,
                    **kw),
                *args,
                after_finished_func=self._after_correction,
                interruptible=True, queue=self.queue,
                **kwargs)
            self.task.start()
            self.listener = threading.Thread(
                target=self._update_progressbar, args=(self.queue,))
            self.listener.start()
            self.queue.join()

        if self.t0_found is False:
            self.get_time_zeros()
        if self.t0_fit_done is False:
            self.fit_time_zero()
            if self.t0_fit_done is False:
                return
        self.progbar = CustomProgressbarWindow(self,
                                               controller=self.controller,
                                               text="Chirp Correction",
                                               start_timer=True,
                                               cancel_button=True)
        self.progbar.frame.cancel_callback = self.cancel_correction
        self.data_obj.delA = self.data_obj.raw_delA
        self.data_obj.time_delays = self.data_obj.time_delays_raw
        self.data_obj.chirp_corrected = False
        start_thread()

    def _after_correction(self):
        try:
            if self.task.output is not None:
                self.after(100, self._post_correction_ops)
        except Exception:
            raise
        finally:
            self._corr_running = False
            self.progbar.destroy()
            self._toggle_widget_state_during_corr(case='end')

    def _post_correction_ops(self):
        # print('cancelled', self._cancelled)
        # if self._cancelled:
        #     self.data_obj.delA = self.data_obj.raw_delA
        #     self.data_obj.time_delays = self.data_obj.time_delays_raw
        #     self.data_obj.chirp_corrected = False
        #     ylim = None
        # else:
        self.data_obj.non_filt_delA = self.data_obj.delA
        self.data_obj.chirp_corrected = True
        self.controller.frames[MainPage].vars['t0shift'].set(0)
        self._update_pages()
        ylim = [np.min(self.data_obj.time_delays),
                np.abs(self.t0_figure.get_ylim()[1] -
                       self.t0_figure.get_ylim()[0])]
        self.ta_map_figure.plot(
            ylimits=ylim,
            update_canvas=True)

    def _toggle_widget_state_during_corr(self, case='start'):
        return

    def cancel_correction(self):
        try:
            self.task.raise_exception(
                func=self.reset_correction(update_ui=False))
        except Exception as e:
            print(e)
        # else:
        #     print('reset')

    def reset_correction(self, update_ui=True):
        self.data_obj.delA = self.data_obj.raw_delA
        self.data_obj.time_delays = self.data_obj.time_delays_raw
        self.data_obj.chirp_corrected = False
        if update_ui:
            self.ta_map_figure.plot(
                ylimits=self.controller.frames[
                    MainPage].main_figure.get_ylim(),
                update_canvas=True)
            self._update_pages()
            self.controller.frames[MainPage].vars['t0shift'].set(
                self.data_obj.time_zero_shift)

    def save_t0(self):
        file = save_box(parent=self)
        try:
            file.name
        except Exception:
            pass
        else:
            self.data_obj.save_time_zero_file(file)

    def _update_pages(self):
        fr = self.controller.frames[MainPage]
        fr.auto_time_win()
        fr.set_tlim(update_main_figure=False)
        fr.update_axes_labels(update_canvas=False)
        fr.update_main_plot()
        self.data_obj.time_zero_shift = fr.vars['t0shift'].get()
        self.controller.frames[MainPage].trace_figure.axes[0].cla()
        self.controller.frames[GlobalFitPage]._update_content()
        fr.trace_figure.canvas.draw()

    def _update_content(self, replot=True, reset=True):
        mp = self.controller.frames[MainPage]
        if replot:
            self.ta_map_figure.plot(ylimits=mp.main_figure.get_ylim(),
                                    xlimits=mp.main_figure.get_xlim(),
                                    update_canvas=False)
        # self.ta_map_figure.set_xlabel(mp.main_figure.get_xlabel(),
        #                               update_canvas=False)
        # self.ta_map_figure.set_ylabel(mp.main_figure.get_ylabel(),
        #                               update_canvas=True)
        self.ta_map_figure.set_axes_label(x=mp.main_figure.get_xlabel(),
                                          y=mp.main_figure.get_ylabel())
        self.show_timestep_disp(visible=True)
        if reset:
            self.t0_found = False
            self.t0_fit_done = False
            self.t0_figure.get_axes().cla()
        self.t0_figure.set_axes_label(x=mp.main_figure.get_xlabel(),
                                      y=mp.main_figure.get_ylabel())

    # page navigation functions
    def _leave_page(self):
        return True

    def _enter_page(self):
        self.ta_map_figure.set_xlim(
            self.controller.frames[MainPage].main_figure.get_xlim(),
            update_canvas=False)
        self._update_content(replot=False, reset=False)
        return True


# %%
class ManualChirpWindow(tk.Toplevel):
    def __init__(self, parent, figure, data_obj, controller=None):
        tk.Toplevel.__init__(self, parent)
        if controller is None:
            controller = parent
        move_toplevel_to_default(self, controller)
        self.data_obj = data_obj
        self.fig_opt_frame = CustomFrame(self, dim=(4, 2))
        self.fig_frame = init_figure_frame(
            self, controller=self, fig_obj=figure,
            plot_func=self.data_obj.plot_ta_map, editable=False,
            canvas_callback={'button_press_event':
                             self.add_point_click})
        self.fig_frame.opts.init_color_opts(
            self.fig_frame.opts,
            contrast_slider_kwargs={'row': 0, 'column': 1})
        self.fig_frame.opts.init_canvas_opts(
            self.fig_opt_frame, row=0, column=0, padx=5)
        self.fig_frame.grid(row=0, column=0, columnspan=2, rowspan=2)

        tk.Label(self.fig_opt_frame, text='Y limits').grid(
            row=0, column=2, columnspan=2, pady=5)
        self.tlower = tk.DoubleVar(value=np.min(self.data_obj.time_delays))
        self.tlower_entry = tk.Entry(
            self.fig_opt_frame, textvariable=self.tlower, width=8)
        self.tlower_entry.grid(row=1, column=2, padx=5, sticky=tk.E)
        self.tlower_entry.bind('<Return>', self.set_ylim)
        self.tupper = tk.DoubleVar(value=np.max(self.data_obj.time_delays))
        self.tupper_entry = tk.Entry(
            self.fig_opt_frame, textvariable=self.tupper, width=8)
        self.tupper_entry.grid(row=1, column=3, padx=5, sticky=tk.W)
        self.tupper_entry.bind('<Return>', self.set_ylim)

        self.fig_opt_frame.grid(row=2, column=0, sticky='wnse')

        self.table_canvas = tk.Canvas(self)
        self.table = tk.Frame(self.table_canvas)
        self.table_scroll = tk.Scrollbar(self, orient="vertical",
                                         command=self.table_canvas.yview)
        self.table_canvas.configure(yscrollcommand=self.table_scroll.set)
        self.table_scroll.grid(row=0, column=3, sticky='nsw', pady=10)
        self.table_canvas.grid(row=0, column=2, sticky='nswe', padx=5, pady=10)
        self.table_canvas.create_window((4, 3), window=self.table, anchor='ne',
                                        tags="self.table")
        self.table.bind("<Configure>", self.config_tab_canvas)
        tk.Label(self.table, text=self.data_obj.get_x_mode() + " (" +
                 self.data_obj.spec_unit + ")").grid(
            row=0, column=1, sticky=tk.W, padx=2)
        tk.Label(self.table, text="time delay ("
                 + self.data_obj.time_unit + ")").grid(
            row=0, column=2, sticky=tk.W, padx=2)
        tk.Label(self.table, text=" ").grid(
            row=0, column=0, sticky=tk.W, padx=2)

        self.counter = 0
        self.points = []
        self.entries = []
        self.last_deleted_point = []
        self.plot_points = []

        ttk.Button(self, text='Done', command=self.destroy).grid(
            row=4, column=0, columnspan=3)
        tk.Label(self, text="Right click on map to add point.\n"
                 + "Added points can be removed by clicking on them. \n"
                 + "Last added point can be deleted by Ctrl+Z and re-added"
                 + " by Ctrl+Y.\nClose window to continue"
                 + " chirp correction.").grid(row=1, column=2)
        self.bind('<Control-z>', self.del_last_point)
        self.bind('<Control-y>', self.re_add_last_point)
        center_toplevel(self, controller)

    def config_tab_canvas(self, *args):
        self.table_canvas.config(scrollregion=self.table_canvas.bbox("all"))

    def set_ylim(self, *args):
        self.fig_frame.figure.set_ylim([self.tlower.get(), self.tupper.get()])

    def add_point_click(self, event):
        if event.button == 3:
            self.add_point([event.xdata, event.ydata])

    def add_point(self, point):
        self.points.append(point)
        self.entries.append([tk.Label(self.table, text=str(self.counter + 1)),
                             tk.Label(self.table, text=str(
                                 round(point[0], 2))),
                             tk.Label(self.table,
                                      text=str(round(point[1], 2)))])
        for i in range(3):
            self.entries[self.counter][i].grid(row=self.counter + 1, column=i,
                                               sticky=tk.W, padx=2)
            self.entries[self.counter][i].bind(
                "<Button-1>", lambda *args, p=self.counter:
                    self.del_point(*args, pos=p))
        self.counter = self.counter + 1
        self.plot_points.append(
            self.fig_frame.figure.axes[0].plot(point[0], point[1],
                                               marker='x', color='yellow'))
        self.fig_frame.figure.canvas.draw()

    def del_last_point(self, *args):
        self.del_point(pos=-1)

    def re_add_last_point(self, *args):
        try:
            self.add_point(self.last_deleted_point[-1])
            del self.last_deleted_point[-1]
        except Exception:
            pass

    def del_point(self, *args, pos=1):
        try:
            self.counter = self.counter - 1
            self.last_deleted_point.append(self.points[pos])
            del self.points[pos]
            for i in range(pos, len(self.points)):
                self.entries[i][0].config(text=str(i + 1))
                self.entries[i][1].config(
                    text=str(round(self.points[i][0], 2)))
                self.entries[i][2].config(
                    text=str(round(self.points[i][1], 2)))
                for j in range(3):
                    self.entries[i][j].grid_forget()
            for i in range(3):
                self.entries[-1][i].grid_forget()
            del self.entries[-1]
            for i in range(pos, len(self.points)):
                for j in range(3):
                    self.entries[i][j].grid(
                        row=i + 1, column=j, sticky=tk.W, padx=2)
            for i in range(len(self.points), -1, -1):
                del self.fig_frame.figure.axes[0].lines[i]
            self.plot_points = []
            for i in range(len(self.points)):
                self.plot_points.append(
                    self.fig_frame.figure.axes[0].plot(
                        self.points[i][0], self.points[i][1], marker='D',
                        color='yellow', markeredgecolor='black'))
            self.fig_frame.figure.canvas.draw()
        except Exception:
            pass


# %%
class DataInspector(tk.Toplevel):
    def __init__(self, parent, controller, data_obj, dataset=None,
                 data_labels=None, xunit=None, yunit=None, xlabel=None,
                 ylabel=None, zlabel=None, xlim=None, ylim=None, zlim=None,
                 invert_yaxis=True, transpose_y_plot=False):
        tk.Toplevel.__init__(self, parent)
        self.data_obj = data_obj
        self.controller = controller
        self.frames = {}
        self.figures = {}
        self.axes = {}
        self.canvas = {}
        self.entries = {}
        self.vars = {}
        self.title("Data Inspector")
        self.mouseover_active = False
        self.zlim = zlim
        self.invert_yaxis_main = invert_yaxis
        if dataset is None:
            dataset = [self.data_obj.delA]
            xvalues = self.data_obj.wavelengths
            yvalues = self.data_obj.time_delays
        else:
            xvalues = dataset['x']
            yvalues = dataset['y']
            dataset = dataset['z']
        if data_labels is None:
            if np.shape(dataset)[0] > 1:
                self.data_labels = ["Data" + str(i + 1)
                                for i in range(np.shape(dataset)[0])]
            else:
                self.data_labels = [""]
        else:
            self.data_labels = data_labels
        if xunit is None:
            self.xunit = self.data_obj.spec_unit_label
        else:
            self.xunit = xunit
        if xlabel is None:
            self.xlabel = self.data_obj.get_x_mode() + " (" + self.xunit + ")"
        else:
            self.xlabel = xlabel
        if yunit is None:
            self.yunit = self.data_obj.time_unit
        else:
            self.yunit = yunit
        if ylabel is None:
            self.ylabel = "time delay (" + self.yunit + ")"
        else:
            self.ylabel = ylabel
        if zlabel is None:
            self.zlabel = self.data_obj.zlabel
        else:
            self.zlabel = zlabel
        # Figures
        self.figures['main'] = self.controller.tk_mpl_figure_wrap(
            self, toolbar=False,
            # dim=settings['figure_std_size'],
            invert_yaxis=self.invert_yaxis_main,
            plot_function=lambda *args, **kwargs: self.data_obj.plot_ta_map(
                *args, dat=dataset[0], xvalues=xvalues, yvalues=yvalues,
                **kwargs),
            xlimits=xlim, ylimits=ylim, climits=self.zlim,
            xlabels=self.xlabel, ylabels=self.ylabel, clabels=self.zlabel,
            callbacks={'button_press_event': self.main_plot_click_callback,
                       'motion_notify_event': self.main_axes_mouseover})
        self.figures['main'].copy_color_settings(self.data_obj.color)
        self.figures['main'].canvas.get_tk_widget().config(cursor='tcross')
        self.figures['main'].plot()
        self.figures['main'].grid(row=0, column=0, padx=5,
                                  sticky='wnse')
        self.dataset = dataset
        if self.controller.settings['2Dmap_transpose']:
            self.xvalues = yvalues
            self.yvalues = xvalues
            self.dataset[0] = np.transpose(self.dataset[0])
            ylab = self.ylabel
            self.ylabel = self.xlabel
            self.xlabel = ylab
            self.xlim = ylim
            self.ylim = xlim
        else:
            self.xvalues = xvalues
            self.yvalues = yvalues
            self.xlim = xlim
            self.ylim = ylim
        for key in ['x', 'y']:
            self.figures[key] = plt.figure()
            plt.close()
            self.figures[key].set_tight_layout(True)
            self.axes[key] = self.figures[key].add_subplot(111)
            self.canvas[key] = FigureCanvasTkAgg(self.figures[key], self)
            self.canvas[key].get_tk_widget().grid(sticky='wnse')
            self.canvas[key].draw()
        self.canvas['x']._tkcanvas.grid(row=1, column=0, padx=5,
                                        sticky='wnse')
        self.canvas['y']._tkcanvas.grid(row=0, column=1, padx=5,
                                        sticky='wnse')
        self.axes['x'].set_xlabel(self.xlabel)
        self.axes['y'].set_xlabel(self.ylabel)
        self.axes['x'].set_ylabel(self.zlabel)
        self.axes['y'].set_ylabel(self.zlabel)
        # tkinter widgets
        widget_frame = tk.Frame(self)
        # disp_frame = tk.Frame(widget_frame)
        self.vars['xlower'] = tk.DoubleVar()
        self.vars['xupper'] = tk.DoubleVar()
        self.vars['ylower'] = tk.DoubleVar()
        self.vars['yupper'] = tk.DoubleVar()
        self.vars['zlower'] = tk.DoubleVar()
        self.vars['zupper'] = tk.DoubleVar()
        tk.Label(widget_frame, text="x limits:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        tk.Label(widget_frame, text="y limits:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5)
        tk.Label(widget_frame, text="z limits:").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=5)
        row = 1
        col = 1
        for key in ['xlower', 'xupper', 'ylower', 'yupper', 'zlower',
                    'zupper']:
            self.entries[key] = tk.Entry(
                widget_frame, textvariable=self.vars[key], width=8)
            self.entries[key].grid(row=row,
                                   column=col, sticky=tk.W, padx=5, pady=5)
            col += 1
            if col > 2:
                row += 1
                col = 1
        for key in ['xlower', 'xupper']:
            self.entries[key].bind(
                '<Return>', lambda *args, c='x':
                    self.limit_entry_callback(*args, case=c))
        for key in ['zlower', 'zupper']:
            self.entries[key].bind(
                '<Return>', lambda *args, c='z':
                    self.limit_entry_callback(*args, case=c))
        for key in ['ylower', 'yupper']:
            self.entries[key].bind(
                '<Return>', lambda *args, c='y':
                    self.limit_entry_callback(*args, case=c))

        self.vars['transpose_y_plot'] = tk.IntVar(value=int(transpose_y_plot))
        tk.ttk.Checkbutton(widget_frame, text="Transpose Y Plot",
                           variable=self.vars['transpose_y_plot'],
                           command=self.trans_y_callback).grid(
                               row=row + 1, column=1, columnspan=2, sticky='w')
        # disp_frame.grid(row=0, column=0, columnspan=3, sticky='wnse',
        #                padx=5, pady=5)
        widget_frame.grid(row=1, column=1, sticky='wnse', padx=5, pady=5)
        if self.xlim is None:
            self.xlim = self.figures['main'].axes[0].get_xlim()
        if self.ylim is None:
            self.ylim = self.figures['main'].axes[0].get_ylim()
        if self.zlim is None:
            self.zlim = self.figures['main'].get_clim()
        self.vars['xlower'].set(self.xlim[0])
        self.vars['xupper'].set(self.xlim[1])
        self.limit_entry_callback(case='x', update_canvas=False)
        self.vars['ylower'].set(self.ylim[0])
        self.vars['yupper'].set(self.ylim[1])
        self.limit_entry_callback(case='y', update_canvas=False)
        self.vars['zlower'].set(self.zlim[0])
        self.vars['zupper'].set(self.zlim[1])
        self.limit_entry_callback(case='z', update_canvas=False)
        self.lines_are_drawn = False
        center_toplevel(self, parent)

    def limit_entry_callback(self, *args, case='xlower', update_canvas=True):
        if case[0] == 'x':
            xlim = [self.vars['xlower'].get(), self.vars['xupper'].get()]
            try:
                if self.controller.settings['2Dmap_transpose']:
                    self.figures['main'].set_ylim(
                        xlim, update_canvas=update_canvas)
                else:
                    self.figures['main'].set_xlim(
                        xlim, update_canvas=update_canvas)
            except Exception:
                return
            else:
                self.axes['x'].set_xlim(*xlim)
                self.xlim = xlim
                if update_canvas:
                    self.canvas['x'].draw()
        elif case[0] == 'y':
            ylim = [self.vars['ylower'].get(), self.vars['yupper'].get()]
            try:
                if self.controller.settings['2Dmap_transpose']:
                    self.figures['main'].set_xlim(
                        ylim, update_canvas=update_canvas)
                else:
                    self.figures['main'].set_ylim(
                        ylim, update_canvas=update_canvas)
            except Exception:
                return
            else:
                if (self.invert_yaxis_main
                        and self.vars['transpose_y_plot'].get()):
                    ylim = ylim[::-1]
                self.axes['y'].set_xlim(*ylim)
                self.ylim = ylim
                if update_canvas:
                    self.canvas['y'].draw()
        elif case[0] == 'z':
            zlim = [self.vars['zlower'].get(), self.vars['zupper'].get()]
            self.figures['main'].set_clim(zlim)
            try:
                self.axes['x'].set_ylim(*zlim)
            except Exception:
                return
            else:
                if self.vars['transpose_y_plot'].get():
                    self.axes['y'].set_xlim(*zlim)
                else:
                    self.axes['y'].set_ylim(*zlim)
                self.zlim = zlim
                if update_canvas:
                    self.canvas['x'].draw()
                    self.canvas['y'].draw()

    def trans_y_callback(self, *args):
        self.axes['y'].cla()
        self.axes['x'].cla()
        if self.vars['transpose_y_plot'].get():
            self.axes['y'].set_xlabel(self.zlabel)
            self.axes['y'].set_ylabel(self.ylabel)
        else:
            self.axes['y'].set_ylabel(self.zlabel)
            self.axes['y'].set_xlabel(self.ylabel)
        self.limit_entry_callback(case='y', update_canvas=False)
        self.axes['x'].set_xlabel(self.xlabel)
        self.axes['x'].set_ylabel(self.zlabel)
        self.plot_xy(self.xpoint, self.ypoint)

    # def change_cursor(self, *args, cursor="wait"):
    #     self.config(cursor=cursor)

    def idle(self, *args, **kwargs):
        return

    def plot_xy(self, xpoint, ypoint):
        try:
            xind = np.where(self.xvalues >= xpoint)[0][0]
        except Exception:
            return
        try:
            yind = np.where(self.yvalues >= ypoint)[0][0]
        except Exception:
            return
        for ax in (self.axes['y'], self.axes['x']):
            for line in ax.lines:
                line.remove()
        try:
            for line in (self.yvline, self.xvline):
                line.remove()
        except Exception:
            pass
        trans_y = self.vars['transpose_y_plot'].get()
        if trans_y:
            self.yvline = self.axes['y'].axhline(y=ypoint, color='grey',
                                                 label='_nolegend_')
        else:
            self.yvline = self.axes['y'].axvline(x=ypoint, color='grey',
                                                 label='_nolegend_')
        self.xvline = self.axes['x'].axvline(x=xpoint, color='grey',
                                             label='_nolegend_')
        self.yplot = []
        self.xplot = []
        for i, dat in enumerate(self.dataset):
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
            yp, = plot(self.axes['y'], self.yvalues, dat[:, xind], color=color,
                       transpose=trans_y)
            self.yplot.append(yp)
            xp, = self.axes['x'].plot(self.xvalues, dat[yind, :], color=color)
            self.xplot.append(xp)
        if trans_y:
            self.axes['y'].set_ylim(*self.ylim)
            if self.zlim is not None:
                self.axes['y'].set_xlim(*self.zlim)
        else:
            self.axes['y'].set_xlim(*self.ylim)
            if self.zlim is not None:
                self.axes['y'].set_ylim(*self.zlim)
        self.axes['x'].set_xlim(*self.xlim)
        if self.zlim is not None:
            self.axes['x'].set_ylim(*self.zlim)
        self.axes['x'].legend(
            [" ".join([lbl, str(np.round(
                ypoint, self.data_obj.time_delay_precision)), self.yunit])
             for lbl in self.data_labels])
        self.axes['y'].legend(
            [" ".join([lbl, str(np.round(
                xpoint, self.data_obj.time_delay_precision)), self.xunit])
             for lbl in self.data_labels])
        self.canvas['x'].draw()
        self.canvas['y'].draw()
        self.lines_are_drawn = True
        self.ypoint = ypoint
        self.xpoint = xpoint

    def update_xy(self, xpoint, ypoint):
        try:
            xind = np.where(self.xvalues >= xpoint)[0][0]
        except Exception:
            return
        try:
            yind = np.where(self.yvalues >= ypoint)[0][0]
        except Exception:
            return
        try:
            for line in (self.yvline, self.xvline):
                line.remove()
        except Exception:
            pass
        if self.vars['transpose_y_plot'].get():
            self.yvline = self.axes['y'].axhline(y=ypoint, color='grey',
                                                 label='_nolegend_')
        else:
            self.yvline = self.axes['y'].axvline(x=ypoint, color='grey',
                                                 label='_nolegend_')
        self.xvline = self.axes['x'].axvline(x=xpoint, color='grey',
                                             label='_nolegend_')
        for i, dat in enumerate(self.dataset):
            if self.vars['transpose_y_plot'].get():
                yplt = [dat[:, xind], self.yvalues]
            else:
                yplt = [self.yvalues, dat[:, xind]]
            self.yplot[i].set_data(*yplt)
            self.xplot[i].set_data(self.xvalues, dat[yind, :])
        self.axes['x'].legend(
            [" ".join([lbl, str(np.round(
                ypoint, self.data_obj.time_delay_precision)), self.yunit])
             for lbl in self.data_labels])
        self.axes['y'].legend(
            [" ".join([lbl, str(np.round(
                xpoint, self.data_obj.time_delay_precision)), self.xunit])
             for lbl in self.data_labels])
        self.canvas['x'].draw()
        self.canvas['y'].draw()
        self.ypoint = ypoint
        self.xpoint = xpoint

    def main_axes_mouseover(self, event):
        if self.mouseover_active:
            if self.lines_are_drawn:
                self.update_xy(event.xdata, event.ydata)
            else:
                self.plot_xy(event.xdata, event.ydata)

    def main_plot_click_callback(self, event):
        if event.button == 3:
            self.mouseover_active = not self.mouseover_active
        if self.lines_are_drawn:
            self.update_xy(event.xdata, event.ydata)
        else:
            self.plot_xy(event.xdata, event.ydata)


# %%
class GlobalFitPage(tk.Frame):
    def __init__(self, parent, controller, data_obj, master=None):
        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.controller = controller
        self.data_obj = data_obj
        self.vars = {}
        self.xaxis = []
        self.previous_inits = None
        self.fit_done = False
        # Frames

        self.option_frame = CustomFrame(self, dim=(4, 4))
        self.win_opt_frame = CustomFrame(
            self.option_frame, dim=(6, 3), border=True)
        self.model_opt_frame = CustomFrame(
            self.option_frame, dim=(1, 1), border=True)
        self.svd_frame = CustomFrame(self.option_frame, dim=(2, 3),
                                     border=True)
        self.para_frame = CustomFrame(
            self.option_frame, dim=(3, 5), border=True)
        self.irf_option_frame = CustomFrame(
            self.option_frame, dim=(2, 3), border=True)
        self.algo_frame = CustomFrame(
            self.option_frame, dim=(3, 6), border=True)
        self.multistart_frame = tk.Frame(self.algo_frame)
        self.run_frame = CustomFrame(self.option_frame, dim=(3, 6),
                                     border=True)

        for i in range(2):
            self.columnconfigure(i, weight=2)
        for i in range(5):
            self.rowconfigure(i, weight=2)

        # Canvas and plots
        self.ta_map_figure = self.controller.tk_mpl_figure_wrap(
            self,
            xlabels=self.controller.settings['xlabel_default'],
            ylabels=self.controller.settings['ylabel_default'],
            clabels=self.data_obj.zlabel,
            callbacks={'button_press_event':
                       lambda *args: self.click_plot_callback(
                           controller, "Main", *args)}, row=1)
        self.ta_map_figure.grid(column=0, row=0)

        self.das_figure = self.controller.tk_mpl_figure_wrap(
            self,
            plot_function=self.das_plot_fun,
            plot_type='line',
            xlabels=self.controller.settings['xlabel_default'],
            ylabels=self.data_obj.zlabel,
            fit_kwargs=self.controller.settings['fit_kwargs'],
            callbacks={'button_press_event':
                       lambda *args: self.click_plot_callback(
                           controller, "DAS", *args)})
        self.das_figure.grid(column=1, row=0)

        # panels

        # option panel
        self.option_frame.grid(row=2, column=0, columnspan=2,
                               sticky='nswe', padx=(0, 15), pady=(0, 10))
        tk.Label(self.option_frame, text='Reconstruction Options').grid(
            row=0, column=1)

        #
        self.win_opt_frame.grid(
            row=1, column=0, columnspan=2, sticky='wens', pady=1, padx=1)
        tk.Label(self.win_opt_frame, text='Window').grid(
            row=0, column=0, sticky='wn')
        self.spec_win_lbl = tk.Label(
            self.win_opt_frame,
            text=self.controller.settings['xlabel_default'])
        self.spec_win_lbl.grid(
            row=1, column=0, columnspan=2, padx=5, sticky=tk.W)
        self.vars['lambda_lim_low'] = tk.DoubleVar(
            value=controller.frames[MainPage].vars['lambda_lim_low'].get())
        self.vars['lambda_lim_up'] = tk.DoubleVar(
            value=controller.frames[MainPage].vars['lambda_lim_up'].get())
        self.spec_win_lower_entry = tk.Entry(
            self.win_opt_frame,
            textvariable=self.vars['lambda_lim_low'],
            width=5)
        self.spec_win_upper_entry = tk.Entry(
            self.win_opt_frame,
            textvariable=self.vars['lambda_lim_up'],
            width=5)
        self.spec_win_lower_entry.grid(row=2, column=0)
        self.spec_win_upper_entry.grid(row=2, column=1)
        self.spec_win_lower_entry.bind('<Return>', self.spec_win_callback)
        self.spec_win_upper_entry.bind('<Return>', self.spec_win_callback)

        tk.Label(
            self.win_opt_frame,
            text=self.controller.settings['ylabel_default']).grid(
                row=1, column=2, columnspan=2, padx=5, sticky=tk.W)
        self.vars['tlower'] = tk.DoubleVar(value=0)
        self.vars['tupper'] = tk.DoubleVar(value=1)
        self.tlower_entry = tk.Entry(
            self.win_opt_frame,
            textvariable=self.vars['tlower'],
            width=5)
        self.tupper_entry = tk.Entry(
            self.win_opt_frame,
            textvariable=self.vars['tupper'],
            width=5)
        self.tlower_entry.grid(row=2, column=2, padx=(5, 0))
        self.tupper_entry.grid(row=2, column=3, padx=(0, 5))
        self.tlower_entry.bind('<Return>', self.time_win_callback)
        self.tupper_entry.bind('<Return>', self.time_win_callback)

        tk.Label(self.win_opt_frame,
                 text="time zero (fit)").grid(
                     row=1, column=4, sticky=tk.W, columnspan=2)
        self.vars['time_zero_fit'] = tk.DoubleVar(value=0)
        self.time_zero_fit_entry = tk.Entry(
            self.win_opt_frame,
            textvariable=self.vars['time_zero_fit'],
            width=5)
        self.time_zero_fit_entry.grid(row=2, column=4, sticky=tk.W)
        self.time_zero_fit_unit = tk.Label(self.win_opt_frame, text='(ps)')
        self.time_zero_fit_unit.grid(row=2, column=5, sticky=tk.W)
        #
        self.model_opt_frame.grid(
            row=2, column=0, sticky='wens', pady=1, padx=1)

        self.vars['inf_comp'] = tk.IntVar(value=0)
        self.vars['num_comp'] = tk.IntVar(value=2)
        self.model_opts = KineticModelOptions(self.model_opt_frame)
        self.model_opts.model_class.set("Parallel")
        self.model_opts.model.set("2 Comp.")
        self.model_opts.grid(row=0, column=0, sticky='wnse')
        self.model_opts.callback = self.model_options_callback

        self.sine_modul = tk.IntVar(value=0)
        self.sine_modul_check = tk.ttk.Checkbutton(
            self.model_opt_frame,
            text="Sine modulation",
            variable=self.sine_modul)

        self.svd_frame.grid(row=1, column=3, sticky='wens', pady=1, padx=1)
        tk.Label(self.svd_frame, text='SVD reduction').grid(
            row=0, column=0, sticky=tk.W, columnspan=2)
        self.vars['enable_svd'] = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.svd_frame, text="Enable",
            variable=self.vars['enable_svd']).grid(
                row=1, column=0, sticky=tk.W)
        tk.Label(self.svd_frame, text='Components:').grid(row=2, column=0)
        self.vars['svd_comp'] = tk.IntVar(value=5)
        self.svd_comp_select = tk.Entry(
            self.svd_frame, textvariable=self.vars['svd_comp'], width=5)
        self.svd_comp_select.grid(row=2, column=1)

        self.para_frame.grid(
            row=2, column=1, sticky='wens', pady=1, padx=1)
        tk.ttk.Label(self.para_frame,
                     text="Parameters").grid(row=0, column=0, sticky='nw')

        tk.ttk.Label(self.para_frame,
                     text="Param. Settings:").grid(row=1, column=0, sticky='w')
        self.vars['guess_opt'] = tk.StringVar(value='enter')
        self.guess_opt_select = tk.ttk.OptionMenu(
            self.para_frame, self.vars['guess_opt'],
            self.vars['guess_opt'].get(), 'enter', 'auto')
        self.guess_opt_select.config(width=10)
        self.guess_opt_select.grid(row=1, column=1, sticky='w', columnspan=2)

        self.vars['enable_nonl_constr'] = tk.IntVar(value=0)
        self.enable_nonl_constr_check = tk.ttk.Checkbutton(
            self.para_frame,
            text='Nonlin. Constraints:',
            variable=self.vars['enable_nonl_constr'],
            command=self.enable_nonl_constr_callback,
            state='disabled')
        self.enable_nonl_constr_check.grid(
            row=2, column=0, columnspan=3, sticky=tk.W)
        self.min_para_diff_label = tk.Label(
            self.para_frame, text="Minimum Parameter \n Difference (%):")
        self.min_para_diff_label.grid(
            row=3, column=0, sticky=tk.W, columnspan=2)
        self.vars['min_para_diff'] = tk.DoubleVar(value=10)
        self.min_para_diff_entry = tk.Entry(
            self.para_frame,
            textvariable=self.vars['min_para_diff'],
            width=5)
        self.min_para_diff_entry.grid(row=3, column=2, sticky=tk.W)
        self.max_das_amp_label = tk.Label(
            self.para_frame, text="Maximum relative \n DAS amplitude:")
        self.max_das_amp_label.grid(
            row=4, column=0, sticky=tk.W, columnspan=2)
        self.vars['max_das_amp'] = tk.DoubleVar(value=4)
        self.max_das_amp_entry = tk.Entry(
            self.para_frame,
            textvariable=self.vars['max_das_amp'],
            width=5)
        self.max_das_amp_entry.grid(row=4, column=2, sticky=tk.W)

        self.enable_nonl_constr_callback()

        self.irf_option_frame.grid(
            row=1, column=2, sticky='wens', pady=1, padx=1)
        tk.Label(self.irf_option_frame, text='IRF').grid(
            row=0, column=0, sticky='wn')
        self.vars['irf_opt'] = tk.IntVar(value=0)
        self.irf_opt_check = tk.ttk.Checkbutton(
            self.irf_option_frame,
            text="Include IRF",
            variable=self.vars['irf_opt'])
        self.irf_opt_check.grid(row=1, column=0, sticky=tk.W, columnspan=2)
        self.sigma_lbl = tk.Label(self.irf_option_frame,
                                  text='Sigma (fs):')
        self.sigma_lbl.grid(row=2, column=0, sticky=tk.W)
        self.vars['irf_val'] = tk.DoubleVar(value='100')
        self.irf_factor = 1e-3
        self.irf_val_entry = tk.Entry(
            self.irf_option_frame, textvariable=self.vars['irf_val'], width=6)
        self.irf_val_entry.grid(row=2, column=1, sticky=tk.W)

        self.algo_frame.grid(
            row=2, column=2, sticky='wens', pady=1, padx=1)
        tk.Label(self.algo_frame, text='Algorithm').grid(
            row=0, column=0, sticky='wn', columnspan=4)
        self.vars['algo_opt'] = tk.StringVar(value="least_squares")
        algos = ["nelder", "differential_evolution", "ampgo", "SLSQP",
                 "least_squares", "COBYLA", "basinhopping", "L-BFGS-B"]
        algo_opt_select = tk.ttk.OptionMenu(
            self.algo_frame, self.vars['algo_opt'],
            self.vars['algo_opt'].get(), *algos)
        algo_opt_select.config(width=20)
        algo_opt_select.grid(row=1, column=0, columnspan=3, sticky='we')

        self.multistart_enabled = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.algo_frame,
            text="Multistart",
            variable=self.multistart_enabled,
            command=self.enable_multistart_callback).grid(
                row=2, column=0, columnspan=3, sticky=tk.W)

        tk.Label(self.multistart_frame, text="No. of runs:").grid(
            row=0, column=0, sticky='w')
        self.multistart_num_runs = tk.IntVar(value=20)
        tk.Entry(
            self.multistart_frame,
            textvariable=self.multistart_num_runs,
            width=5).grid(row=0, column=1, sticky='w')

        tk.Label(self.multistart_frame, text="Guess var. (%):").grid(
            row=1, column=0, sticky='w')
        self.multistart_variance = tk.DoubleVar(value=10)
        tk.Entry(
            self.multistart_frame,
            textvariable=self.multistart_variance,
            width=5).grid(row=1, column=1, sticky='w')

        self.multistart_frame.grid(row=3, column=1, columnspan=2)

        tk.Label(self.algo_frame, text='Tolerance:').grid(
            row=5, column=0, sticky='w')
        tk.Label(self.algo_frame, text='1E').grid(
            row=5, column=1, sticky=tk.E)
        self.vars['fun_tol'] = tk.IntVar(value=-4)
        tk.Entry(self.algo_frame, textvariable=self.vars['fun_tol'],
                 width=5).grid(row=5, column=2, sticky=tk.W)

        self.vars['algo_opt'].trace('w', self.algo_opts_callback)

        #
        self.run_frame.grid(row=2, column=3, sticky='wnse', pady=1, padx=1)
        tk.Label(self.run_frame, text='Run Reconstruction').grid(
            row=0, column=0, columnspan=3, sticky='wn')

        button_frame = tk.Frame(self.run_frame)
        self.run_fit_button = ttk.Button(
            button_frame, text="Start", command=self.start_fit)
        self.run_fit_button.grid(
            row=0, column=0, sticky='wnse', padx=5, pady=5)

        self.cancel_fit_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_fit,
            state='disabled')
        self.cancel_fit_button.grid(
            row=0, column=2, sticky='wnse', padx=5, pady=5)

        button_frame.grid(row=6, column=0, columnspan=3, sticky='wnse')

        self.disp_conc = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.run_frame,
            text="Plot Concentrations Live",
            variable=self.disp_conc).grid(
                row=4, column=0, columnspan=3, sticky=tk.W)

        self.disp_results_after = tk.IntVar(value=1)
        tk.ttk.Checkbutton(
            self.run_frame,
            text="Show Results Window",
            variable=self.disp_results_after).grid(
                row=5, column=0, columnspan=3, sticky='w')

        self.show_results_button = tk.ttk.Button(
            self.run_frame,
            text="Show Results",
            command=self.show_results_win)
        self.show_results_button.config(state='disabled')
        self.show_results_button.grid(row=7, column=0, columnspan=3, pady=5)

        self.timer = CustomTimer(self.option_frame, start=False, text="")
        self.timer.grid(row=3, column=0, columnspan=3, sticky='w')

        self.enable_multistart_callback()

    def write_app_settings(self):
        for key, val in self.vars.items():
            try:
                self.controller.app_settings[GlobalFitPage][key] = val.get()
            except Exception as e:
                print(e)

    def load_app_settings(self):
        for key in self.vars.keys():
            try:
                self.vars[key].set(
                    self.controller.app_settings[GlobalFitPage][key])
            except Exception as e:
                print(e)

    # UI Callbacks
    def model_options_callback(self, *args):
        model = self.model_opts.get_output()
        model_str = model[0][:6] + model[1]
        self.vars['num_comp'].set(model[1][0])
        self.vars['inf_comp'].set(bool(re.search('inf', model_str)))
        self.model = [model[0], model_str]

    def spec_win_callback(self, *args, update_canvas=True):
        xlim = [self.vars['lambda_lim_low'].get(),
                self.vars['lambda_lim_up'].get()]
        self.ta_map_figure.set_xlim(xlim, update_canvas=update_canvas)

    def time_win_callback(self, *args, update_canvas=True):
        if (self.vars['tupper'].get() > np.max(
                [self.data_obj.time_delays[0], self.data_obj.time_delays[-1]])
                or self.vars['tupper'].get() < np.min(
                    [self.data_obj.time_delays[0],
                     self.data_obj.time_delays[-1]])):
            self.vars['tupper'].set(
                np.round(np.max([self.data_obj.time_delays[0],
                                 self.data_obj.time_delays[-1]]), 3))
        if (self.vars['tlower'].get() < np.min(
                [self.data_obj.time_delays[0], self.data_obj.time_delays[-1]])
                or self.vars['tlower'].get() > np.max(
                    [self.data_obj.time_delays[0],
                     self.data_obj.time_delays[-1]])):
            self.vars['tlower'].set(
                np.round(np.min([self.data_obj.time_delays[0],
                                 self.data_obj.time_delays[-1]]), 3))
        try:
            ylim = [self.vars['tlower'].get(),
                    self.vars['tupper'].get()]
            self.ta_map_figure.set_ylim(ylim, update_canvas=update_canvas)
        except Exception:
            if self.fit_done:
                self.vars['tlower'].set(
                    0 if (np.min(
                        self.controller.frames[MainPage].main_figure
                        .get_ylim()) < 0)
                    else round(np.min(
                        self.controller.frames[
                            MainPage].main_figure.get_ylim())))
                self.vars['tupper'].set(
                    round(np.max(
                        self.controller.frames[
                            MainPage].main_figure.get_ylim())))
            else:
                self.vars['tlower'].set(
                    0 if self.data_obj.time_delays[0] < 0
                    else round(np.min(self.data_obj.time_delays)))
                self.vars['tupper'].set(
                    round(np.max(self.data_obj.time_delays)))
            if update_canvas:
                self.ta_map_figure.canvas.draw()

    def algo_opts_callback(self, *args):
        # nonlin. constraints currently disabled since prior
        # implementation no longer works. to be reworked.
        # if self.vars['algo_opt'].get() in ('ampgo', 'least_squares',
        #                                    'nelder'):
        if True:
            self.enable_nonl_constr_check.config(state='disabled')
            self.vars['enable_nonl_constr'].set(0)
            self.enable_nonl_constr_callback()
        else:
            self.enable_nonl_constr_check.config(state='normal')
        if self.vars['algo_opt'].get() in ('differential_evolution'):
            self.vars['guess_opt'].set('enter')
            self.guess_opt_select.config(state='disabled')
        else:
            self.guess_opt_select.config(state='normal')

    def enable_nonl_constr_callback(self, *args):
        if self.vars['enable_nonl_constr'].get():
            for widget in [self.min_para_diff_entry, self.max_das_amp_entry,
                           self.min_para_diff_label, self.max_das_amp_label]:
                widget.config(state='normal')
        else:
            for widget in [self.min_para_diff_entry, self.max_das_amp_entry,
                           self.min_para_diff_label, self.max_das_amp_label]:
                widget.config(state='disabled')

    def enable_multistart_callback(self, *args):
        if self.multistart_enabled.get():
            for child in self.multistart_frame.winfo_children():
                child.config(state='normal')
        else:
            for child in self.multistart_frame.winfo_children():
                child.config(state='disabled')

    def click_plot_callback(self, cont, datatype, event):
        if event.button == 3 or event.dblclick:
            kwargs = {'editable': True}
            if datatype == "DAS":
                kwargs['fig_obj'] = self.das_figure
                kwargs['data_obj'] = self.das
            elif datatype == "Main":
                kwargs['fig_obj'] = self.ta_map_figure
            else:
                kwargs['fig_obj'] = None
                if datatype.plot_type == '2D':
                    kwargs['plot_func'] = datatype.plot_2d
                    kwargs['plot_type'] = '2D'
                else:
                    kwargs['plot_func'] = datatype.plot_line
                    kwargs['plot_type'] = 'line'
            self.controller.open_topfigure_wrap(
                self, controller=self.controller, **kwargs)

    # methods associated with running reconstruction

    def cancel_fit(self):
        try:
            self.task.raise_exception()
        except Exception:
            self.run_fit_button.config(state='normal')
            self.cancel_fit_button.config(state='disabled')

    def start_fit(self):
        self.timer.text = "Reconstruction running."
        self._start_fit_thread()

    def _start_fit_thread(self):
        def start_thread(*args, **kwargs):
            self._fit_running = True
            self.queue = Queue()
            self.task = ThreadedTask(self.data_obj.run_global_analysis, *args,
                                     after_finished_func=self._after_fit,
                                     interruptible=True,
                                     queue=self.queue,
                                     **kwargs)
            self.task.start()
            if self.disp_conc.get():
                self.listener = threading.Thread(
                    target=self._live_plot, args=(self.queue,))
                self.listener.start()
            self.queue.join()

        # get UI inputs

        self.time_win_callback()
        self.spec_win_callback()
        self.model_options_callback()
        if self.multistart_variance.get() >= 100:
            self.multistart_variance.set(95)

        # saving required information for post-reconstruction:
        self.current_fit_opts = {}
        for var in ['guess_opt', 'enable_svd']:
            self.current_fit_opts[var] = self.vars[var].get()
        self.current_fit_opts['model'] = self.model[1]

        tlower, tupper = self.data_obj.set_time_win_indices((
            self.vars['tlower'].get(),
            self.vars['tupper'].get()))
        tupper += 1         # indices will be used for slice
        xind = self.data_obj.get_xlim_indices(
            limits=[self.vars['lambda_lim_low'].get(),
                    self.vars['lambda_lim_up'].get()])
        if self.disp_conc.get():
            self.das_figure.axes[0].cla()
            self.das_figure.axes[0].set_title("Concentrations")
            self.conc_plot = []
            for i in range(self.vars['num_comp'].get()):
                line, = self.das_figure.axes[0].plot(
                    self.data_obj.time_delays[tlower:tupper],
                    np.zeros(tupper-tlower))
                self.conc_plot.append(line)
            self.conc_plot.append(self.das_figure.axes[0].text(
                0.05, 0.95, "No. of iterations: 0",
                transform=self.das_figure.axes[0].transAxes))
            self.das_figure.axes[0].set_ylim([-0.1, 1.1])

            self.das_figure.canvas.draw()
        fit_obj = self.data_obj.init_global_analysis(
            td_window=(tlower, tupper),
            spec_window=xind,
            amp_constraint=self.vars['max_das_amp'].get(),
            guesses='auto',
            bounds='auto',
            model=self.model[1],
            algorithm=self.vars['algo_opt'].get(),
            fun_tol=np.power(10.0, self.vars['fun_tol'].get()),
            para_constraint=self.vars['min_para_diff'].get(),
            t0=self.vars['time_zero_fit'].get(),
            irf_sigma=self.irf_factor *
            self.vars['irf_val'].get() if bool(
                self.vars['irf_opt'].get()) else None,
            svd_comp=self.vars['svd_comp'].get()
            if bool(self.vars['enable_svd'].get())
            else None,
            sine_modul=bool(self.sine_modul.get()),
            multistart=(self.multistart_num_runs.get()
                        if self.multistart_enabled.get() else False),
            multistart_variance=self.multistart_variance.get()/100,
            enable_nonl_constr=bool(self.vars['enable_nonl_constr'].get()))

        if re.search('enter', self.vars['guess_opt'].get(), re.I):
            if self.previous_inits:
                last_val = 0
                for i in range(self.vars['num_comp'].get()):
                    key = 'tau_' + str(i + 1)
                    try:
                        last_val = self.previous_inits[''][key]['val']
                    except Exception:
                        key = 'tau_' + str(i)
                        break
                if last_val == np.inf:
                    try:
                        del self.previous_inits[''][key]
                    except KeyError:
                        pass
                    except Exception:
                        raise
                if self.vars['inf_comp'].get():
                    try:
                        del self.previous_inits[''][
                            'tau_' + str(self.vars['num_comp'].get())]
                    except KeyError:
                        pass
                    except Exception:
                        raise
            win = FitParaOptsWindow(self, fit_obj.params,
                                    controller=self.controller,
                                    input_values=self.previous_inits)
            if self.vars['inf_comp'].get():
                for wdgt in win.fr.widgets['_tau_' + str(
                        self.vars['num_comp'].get())].values():
                    wdgt.config(state='disabled')
            self.wait_window(win)
            self.timer.start()
            if win.output is None:
                self.run_fit_button.config(state='normal')
                self.cancel_fit_button.config(state='disabled')
                self.timer.text = "Reconstruction cancelled."
                self.timer.stop()
                self.previous_inits = {'': {}}
                return
            else:
                fit_obj.params = win.write_parameters(fit_obj.params)
                self.previous_inits = win.output
        else:
            self.timer.start()

        # start reconstruction thread
        self.run_fit_button.config(state='disabled')
        self.cancel_fit_button.config(state='normal')
        start_thread(fit_obj=fit_obj)

    def _after_fit(self):
        try:
            if self.task.output is not None:
                self.fit_obj, self.das = self.task.output
                self._fit_running = False
                self.after(10, lambda: self._update_after_reconstr())
                self.timer.text = "Reconstruction finished."
            else:
                self._fit_running = False
                self.timer.text = "Reconstruction cancelled."
        except Exception:
            self._fit_running = False
        finally:
            self.timer.stop()
            self.run_fit_button.config(state='normal')
            self.cancel_fit_button.config(state='disabled')

    def _update_after_reconstr(self):
        if self.fit_obj.error is not None:
            messagebox.showerror(message=str(self.fit_obj.error), parent=self)
            return

        # set x values
        xind = self.data_obj.get_xlim_indices(
            limits=[self.vars['lambda_lim_low'].get(),
                    self.vars['lambda_lim_up'].get()])
        self.xaxis = self.data_obj.spec_axis[
            self.data_obj.get_x_mode()][slice(xind[0], xind[1] + 1)]

        # DAS plot

        self.das.xdata = np.array(self.xaxis)
        self.das.xlabel = self.ta_map_figure.get_xlabel()
        if re.search("paral", self.current_fit_opts['model'], re.I):
            if len(self.das.active_traces) > 1:
                tau = [self.fit_obj.result.params[p].value
                       for p in self.fit_obj.result.params if re.search(
                               'tau', p, re.I)]
                self.das.active_traces = [
                    self.das.active_traces[i] for i in np.argsort(tau)]
            self.das_figure.set_axes_title(
                "Decay-associated Spectra", update_canvas=False)
            self.das_figure.set_legend(
                entries=self.das.active_traces, update_canvas=False)
        else:
            self.das_figure.set_axes_title(
                "Species-associated Spectra", update_canvas=False)
            self.das_figure.set_legend(
                entries=[("Species " + "ABCDEFGHIJKLMNOPQ"[i])
                         for i in range(len(self.das.active_traces))],
                update_canvas=False)
        self.das_figure.set_axes_label(
            x=self.ta_map_figure.get_xlabel(), update_canvas=False)
        self.das_figure.plot_function = self.das.plot
        self.das_figure.set_plot_kwargs(fill=0.15)
        self.das_figure.plot()

        # show results if enabled
        self.show_results_button.config(state='normal')
        if self.disp_results_after.get():
            self.show_results_win()
        self.fit_done = True

    def _live_plot(self, in_queue):
        while self._fit_running:
            d = in_queue.get()
            y = d['conc']
            n = d['num_iter']
            for i in range(np.shape(y)[0]):
                self.conc_plot[i].set_ydata(y[i])
            self.conc_plot[i + 1].set_text("No. of iterations: " + str(n))
            try:
                self.after(1, lambda: self.das_figure.canvas.draw())
            except Exception as e:
                print(e)

    # Displaying results

    def show_results_win(self, *args):
        self.result_win = GlobalFitResults(self)
        self.wait_window(self.result_win)

    def das_plot_fun(self, *args, fig=None, ax=None, fill=0.15,
                     reverse_z=False,
                     # legend_kwargs=None,
                     color_order=None,
                     color_cycle=None, interpolate_colors=False,
                     transpose=False, **plot_kwargs):
        if fig is None:
            fig = self.das_figure.figure
        if ax is None:
            ax = self.das_figure.axes[0]
        ax.cla()
        numlines = len(self.das.active_traces)
        fill = get_fill_list(fill, numlines)
        zord, cycle = set_line_cycle_properties(
            numlines, reverse_z, color_order, cycle=color_cycle,
            interpolate=interpolate_colors)
        if np.any(fill):
            for i, key in enumerate(self.das.active_traces):
                ax.plot(self.das.xdata, self.das.tr[key]['y'], zorder=zord[i],
                        color=cycle[i], **plot_kwargs)
                if fill[i]:
                    ax.fill_between(self.das.xdata, self.das.tr[key]['y'],
                                    zorder=zord[i], label='_nolegend_',
                                    color=cycle[i], alpha=fill[i])
        else:
            for i, key in enumerate(self.das.active_traces):
                ax.plot(self.das.xdata, self.das.tr[key]['y'],
                        zorder=zord[i], color=cycle[i], **plot_kwargs)

    # page navigation

    def _leave_page(self):
        return True

    def _enter_page(self):
        self._update_window()
        return True

    def _update_window(self):
        if not self.controller.settings_loaded:
            mp = self.controller.frames[MainPage]
            self.vars['tupper'].set(mp.vars['tupper'].get())
            self.vars['tlower'].set(mp.vars['tlower'].get())
            self.vars['time_zero_fit'].set(
                0.0 if np.min(self.data_obj.time_delays) < 0
                else np.min(self.data_obj.time_delays))
            self.time_win_callback(update_canvas=False)
            self.vars['lambda_lim_low'].set(mp.vars['lambda_lim_low'].get())
            self.vars['lambda_lim_up'].set(mp.vars['lambda_lim_up'].get())
            self.spec_win_callback(update_canvas=True)

    def _update_content(self):
        mp = self.controller.frames[MainPage]
        self.ta_map_figure.copy_attributes(
            mp.main_figure, exclude=['canvas_callbacks', 'plot_function'])
        self.ta_map_figure.plot_function = self.data_obj.plot_ta_map
        self.ta_map_figure.plot()
        self.show_results_button.config(state='disabled')
        self.run_fit_button.config(state='normal')
        self.cancel_fit_button.config(state='disabled')
        self.spec_win_lbl.config(text=mp.main_figure.get_xlabel())
        self.time_zero_fit_unit.config(text='(' + mp.vars['tunit'].get() + ')')
        if self.data_obj.time_unit in self.data_obj.time_unit_factors.keys():
            self.sigma_lbl.config(
                text='Sigma ('
                + self.data_obj.time_unit_factors[self.data_obj.time_unit][1]
                + '):')
            self.irf_factor = 1e-3
            if self.controller.settings_loaded:
                self.vars['irf_val'].set(100)
        else:
            self.sigma_lbl.config(text=mp.vars['tunit'].get().join(['Sigma (',
                                                                    '):']))
            self.irf_factor = 1
            if self.controller.settings_loaded:
                self.vars['irf_val'].set(0.1)
        self._update_window()


# %%
class GlobalFitResults(tk.Toplevel):
    def __init__(self, parent):
        if not type(parent) is GlobalFitPage:
            print('Invalid parent for class GlobalFitResults. '
                  + 'Must be GlobalFitPage.')
            return
        tk.Toplevel.__init__(self, parent)
        self.title("Global Fit Results")
        self.parent = parent
        self.fit_obj = parent.fit_obj
        self.x = parent.xaxis
        self.y = self.fit_obj.y
        self.das = parent.das
        self.controller = parent.controller
        self.resid = self.fit_obj.fit_matrix-self.fit_obj.z
        self.resid_relative = np.divide(self.resid, self.fit_obj.z)*100

        self.fig_frame = CustomFrame(self, dim=(2, 4))

        self.main_figure = self.controller.tk_mpl_figure_wrap(
            self.fig_frame,
            clabels=self.parent.data_obj.zlabel,
            callbacks={'button_press_event':
                       lambda *args: self.click_plot_callback(
                           "Main", *args)})
        self.main_figure.grid(row=0, column=0)
        self.main_figure.copy_attributes(parent.ta_map_figure)
        self.main_figure.set_axes_title("Data")

        self.das_figure = self.controller.tk_mpl_figure_wrap(
            self.fig_frame,
            plot_function=parent.das_plot_fun,
            plot_type='line',
            xlabels=self.controller.settings['xlabel_default'],
            ylabels=self.parent.data_obj.zlabel,
            callbacks={'button_press_event':
                       lambda *args: self.click_plot_callback(
                           "DAS", *args)})
        self.das_figure.grid(column=0, row=2)
        self.das_figure.copy_attributes(parent.das_figure)

        self.fit_figure = self.controller.tk_mpl_figure_wrap(
            self.fig_frame,
            plot_function=self.fit_plot,
            axes_titles="Fit",
            xlabels=self.main_figure.get_xlabel(),
            ylabels=self.main_figure.get_ylabel(),
            xlimits=self.main_figure.get_xlim(),
            ylimits=self.main_figure.get_ylim(),
            color_obj=self.parent.data_obj.color,
            callbacks={'button_press_event':
                       lambda *args: self.click_plot_callback(
                           "Fit", *args)}, row=0)
        self.fit_figure.grid(column=1)

        self.resid_figure = self.controller.tk_mpl_figure_wrap(
            self.fig_frame,
            plot_function=self.residual_plot,
            axes_titles="Residual",
            xlabels=self.main_figure.get_xlabel(),
            ylabels=self.main_figure.get_ylabel(),
            xlimits=self.main_figure.get_xlim(),
            ylimits=self.main_figure.get_ylim(),
            clabels=self.parent.data_obj.zlabel,
            callbacks={'button_press_event':
                       lambda *args: self.click_plot_callback(
                           "Resid", *args)}, row=2)
        self.resid_figure.grid(column=1)

        self.ops_frame = CustomFrame(self, dim=(2, 5))
        self.model_display = GroupBox(
            self.ops_frame, text="Model", dim=(2, 2), border=True)
        self.results_display = GroupBox(
            self.ops_frame, text="Parameters", dim=(2, 1), border=True)
        self.advanced_plots = GroupBox(
            self.ops_frame, text="Plots", dim=(2, 3), border=True)
        self.plot_comps_frame = GroupBox(
            self.advanced_plots, text="Components", dim=(1, 2), border=True)
        self.resid_opt_frame = GroupBox(
            self.advanced_plots, text="Residual", dim=(2, 2), border=True)
        self.save_frame = GroupBox(
            self.ops_frame, text="Save Results", dim=(2, 1), border=True)
        self.report_box = GroupBox(self.ops_frame, dim=(
            1, 1), text="Fit Report", border=True)
        gbox_gkwargs = {'sticky': 'wnse', 'padx': 5, 'pady': 5}

        #
        self.model_info = self.fit_obj.model_obj.get_model()
        model_label = self.model_info['model'].title()
        try:
            model_label = "\n".join(
                [model_label, self.model_info['properties']['mechanism']])
        except Exception:
            pass
        self.model_display.grid(row=0, column=0, **gbox_gkwargs)
        tk.ttk.Label(self.model_display, text="Model:").grid(
            row=0, column=0, sticky='w')
        tk.ttk.Label(self.model_display, text=model_label).grid(
            row=0, column=1, sticky='w')
        tk.ttk.Label(self.model_display, text="No. of\nExp.:").grid(
            row=1, column=0, sticky='w')
        tk.ttk.Label(self.model_display,
                     text=self.fit_obj.number_of_decays).grid(
                         row=1, column=1, sticky='w')

        self.results_display.grid(row=0, column=1, **gbox_gkwargs)
        self.results_disp_canvas = tk.Canvas(
            self.results_display, width=200, height=60)
        self.results_display_frame = tk.Frame(self.results_disp_canvas)
        self.results_disp_scroll = tk.Scrollbar(
            self.results_display,
            orient="vertical",
            command=self.results_disp_canvas.yview)
        self.results_disp_canvas.configure(
            yscrollcommand=self.results_disp_scroll.set)
        self.results_disp_scroll.grid(row=0, column=1, sticky='ns')
        self.results_disp_canvas.grid(sticky='wnse', row=0, column=0)
        self.results_disp_canvas.create_window(
            (4, 4), window=self.results_display_frame, anchor='ne',
            tags="self.results_display_frame")
        self.results_display_frame.bind("<Configure>",
                                        self.config_results_canvas)
        self.results_display_widgets = []

        self.report_box.grid(row=1, column=0, columnspan=2, **gbox_gkwargs)
        self.report_frame = ScrollFrame(self.report_box, widget=MultiDisplay,
                                        width=350, height=300,
                                        scroll_dir='xy',
                                        input_dict={},
                                        mode='expandable',
                                        orient='vertical')
        self.report_disp = self.report_frame.widget
        self.report_frame.grid(sticky='wnse')

        self.advanced_plots.grid(row=2, column=0, columnspan=2, **gbox_gkwargs)
        tk.ttk.Button(self.advanced_plots,
                      text="Movie",
                      command=self.show_movie).grid(
                          row=1, column=0, pady=5)

        tk.ttk.Button(self.advanced_plots,
                      text="Inspect Fit",
                      command=self.inspect_fit).grid(
                          row=0, column=0, pady=5)

        tk.ttk.Button(
            self.advanced_plots, text="Plot Traces",
            command=self.plot_traces).grid(row=0, column=1, pady=5)

        tk.ttk.Button(self.advanced_plots,
                      text="Concentrations",
                      command=self.open_concentration_figure).grid(
                          row=1, column=1, pady=5)
        ttk.Button(self.plot_comps_frame,
                   text='Show',
                   command=self._plot_components).grid(
                       row=1, column=0, padx=5)
        self.comp_to_show = tk.StringVar(value="all")
        tk.Entry(
            self.plot_comps_frame, textvariable=self.comp_to_show).grid(
                row=0, column=0, padx=5, pady=1)
        self.plot_comps_frame.grid(row=2, column=1, **gbox_gkwargs)
        #
        self.resid_opt_frame.grid(row=2, column=0, **gbox_gkwargs)
        tk.Label(self.resid_opt_frame, text='Scale:').grid(
            row=0, column=0, sticky=tk.W)
        self.resid_scale_opt = tk.StringVar(value='absolute')
        tk.ttk.OptionMenu(
            self.resid_opt_frame, self.resid_scale_opt,
            self.resid_scale_opt.get(), 'absolute', 'relative',
            command=self.update_resid).grid(row=0, column=1, sticky='we')
        self.resid_fft_button = ttk.Button(self.resid_opt_frame, text='FFT',
                                           command=self.residual_fft)
        self.resid_fft_button.grid(row=1, column=0, columnspan=2)
        #
        self.save_frame.grid(row=3, column=0, columnspan=2, **gbox_gkwargs)

        ttk.Button(
            self.save_frame, text='Save', command=self.save_results).grid(
                row=0, column=1)
        self.what_to_save = tk.StringVar(value='Spectra')
        tk.ttk.OptionMenu(self.save_frame, self.what_to_save,
                          'Spectra', 'Spectra', 'Fit Map',
                          'Residual').grid(row=0, column=0)

        self.disp_results()

        self.fig_frame.grid(row=0, column=0, rowspan=2, **gbox_gkwargs)
        self.ops_frame.grid(row=0, column=1, **gbox_gkwargs)

    def fit_plot(self, *args, **kwargs):
        return self.parent.data_obj.plot_ta_map(
            *args, dat=self.fit_obj.fit_matrix, yvalues=self.y, xvalues=self.x,
            **kwargs)

    def residual_plot(self, *args, **kwargs):
        return self.parent.data_obj.plot_ta_map(
            *args, dat=self.res, yvalues=self.y, xvalues=self.x, **kwargs)

    def click_plot_callback(self, case, event):
        if event.button == 3 or event.dblclick:
            kwargs = {}
            if case == "DAS":
                fig_obj = self.das_figure
                kwargs['data_obj'] = self.das
            elif case == "Resid":
                fig_obj = self.resid_figure
            elif case == "Main":
                fig_obj = self.main_figure
            elif case == 'Fit':
                fig_obj = self.fit_figure
            self.controller.open_topfigure_wrap(
                self, fig_obj=fig_obj, editable=True, controller=self,
                **kwargs)

    def plot_traces(self, *args):
        dat = {'x': self.x,
               'y': self.y,
               'z': self.fit_obj.z,
               'xlabel': self.main_figure.get_xlabel(),
               'ylabel': self.main_figure.get_ylabel(),
               'zlabel': self.main_figure.get_clabel(),
               'fit': self.fit_obj.fit_matrix}
        TraceManagerWindow(
            self, self.controller, dat, self.parent.data_obj,
            map_figure=self.main_figure, title='Global Fit Traces',
            residual_widgets=True,
            spec_unit=self.controller.settings['input_spectral_unit'])

    def inspect_fit(self, *args):
        dataset = {'x': self.x,
                   'y': self.y,
                   'z': [self.fit_obj.z, self.fit_obj.fit_matrix]}
        DataInspector(self, self.controller, self.parent.data_obj,
                      dataset=dataset,
                      data_labels=["Data", "Fit"],
                      xlim=self.main_figure.get_xlim(),
                      ylim=[np.min(self.y), np.max(self.y)],
                      zlim=self.main_figure.get_clim(),
                      xlabel=self.main_figure.get_xlabel(),
                      ylabel=self.main_figure.get_ylabel(),
                      zlabel=self.main_figure.get_clabel())

    def show_movie(self, *args):
        frame_labels = [str(np.round(td, 3)) + ' ' +
                        self.parent.data_obj.time_unit
                        for td in self.fit_obj.y]
        PlotMovie(self, self.controller,
                  data=[self.x, np.array([self.fit_obj.z])],
                  fit=[self.x, np.array([self.fit_obj.fit_matrix])],
                  xlimits=self.main_figure.get_xlim(),
                  ylimits=self.main_figure.get_clim(),
                  frame_labels=frame_labels)

    def config_results_canvas(self, *args):
        self.results_disp_canvas.config(
            scrollregion=self.results_disp_canvas.bbox("all"))

    def update_resid(self, *args):
        if self.resid_scale_opt.get() == 'absolute':
            self.res = self.resid
            clim_resid = self.parent.data_obj.get_clim(self.res)
            clabel = '$\Delta \Delta$ Abs.'
        else:
            self.res = self.resid_relative
            clim_resid = [-10, 10]
            clabel = '$\Delta \Delta$ Abs. (%)'
        self.resid_figure.update_color_obj(clims=clim_resid)
        self.resid_figure.set_clabel(label=clabel, update_canvas=False)
        self.resid_figure.plot()

    def save_results(self):
        fnames = {'Fit Map': 'GA_fit',
                  'Residual': 'GA_residual',
                  'Spectra': 'GA_spectra'}
        file = save_box(fext='.mat',
                        filetypes=[('text files', '.txt'),
                                   ('Matlab files', '.mat')],
                        fname=fnames[self.what_to_save.get()],
                        parent=self)
        if file is not None:
            if self.what_to_save.get() == 'Spectra':
                self.das.save_traces(
                    file, trace_keys='all',
                    spec_quantity=re.search(
                        '.*(?=\s\()', self.das_figure.get_xlabel()).group(0),
                    spec_unit=re.search(
                        '(?<=\().*(?=\))',
                        self.das_figure.get_xlabel()).group(0),
                    trace_type='DAS' if re.search(
                        'parallel', self.fit_obj.model, re.I) else 'SAS',
                    ylabel=self.das_figure.get_ylabel(),
                    save_fit=False)
            elif self.what_to_save.get() == 'Fit Map':
                self.parent.data_obj.save_data_matrix(
                    file, matrix=self.fit_obj.fit_matrix, x=self.x,
                    y=self.fit_obj.y)
            elif self.what_to_save.get() == 'Residual':
                self.parent.data_obj.save_data_matrix(
                    file, matrix=self.res, x=self.x, y=self.fit_obj.y)
        self.lift()

    def residual_fft(self, plot=True):
        self.fft_map, self.fft_freq, err = self.parent.data_obj.fft_map(
            dat=np.transpose(self.resid), x=self.y)
        if err:
            messagebox.showerror(message=err, parent=self)
        if plot:
            self.controller.open_topfigure_wrap(
                self,
                plot_func=lambda *args, **kwargs:
                    self.parent.data_obj.plot_ta_map(
                        *args,
                        dat=np.transpose(self.fft_map),
                        yvalues=self.fft_freq,
                        xvalues=self.x,
                        **kwargs),
                color_obj=self.resid_figure.color_obj, editable=True,
                ylabels='wavenumber (cm$^{-1}$)',
                xlabels=self.resid_figure.get_xlabel(),
                clabels=self.main_figure.get_clabel(),
                ylimits=[np.min(self.fft_freq),
                         np.max(self.fft_freq)],
                xlimits=[np.min(self.x), np.max(self.x)],
                axes_titles='F.T. of Residual',
                controller=self,
                clims=self.parent.data_obj.get_clim(self.fft_map))

    def residual_wavelet(self, plot=True):
        wl = 350
        try:
            wl_ind = np.where(self.x >= wl)[0][0]
        except Exception:
            return
        cwtmat, freq, error = self.parent.data_obj.wavelet_analysis(
            dat=np.transpose(self.resid),
            x=self.y,
            index=wl_ind)
        if plot:
            self.controller.open_topfigure_wrap(
                self,
                plot_func=lambda *args, **kwargs:
                    self.parent.data_obj.plot_ta_map(
                        *args,
                        dat=cwtmat,
                        yvalues=freq,
                        xvalues=self.y,
                        **kwargs),
                editable=True,
                controller=self)

    def _plot_components(self):
        if self.comp_to_show.get() == 'all':
            to_plot = 'all'
        else:
            try:
                comp_inds = [
                    int(i) - 1
                    for i in self.comp_to_show.get().strip().split(",")]
                to_plot = [self.das.active_traces[i] for i in comp_inds]
            except Exception as e:
                print(e)
                to_plot = 'all'
                self.comp_to_show.set('all')
        if to_plot == 'all':
            to_plot = self.das.active_traces
            comp_inds = list(range(len(self.das.active_traces)))
        self.comp_data = np.zeros((len(to_plot),
                                  len(self.das.tr[to_plot[0]]['y']),
                                  np.shape(self.fit_obj.ycomps)[1]))
        self.comp_sum = np.zeros((len(self.das.tr[to_plot[0]]['y']),
                                  np.shape(self.fit_obj.ycomps)[1]))
        num_cols = 2
        num_rows = int(len(to_plot) / num_cols) + 1
        subplot_pos = []
        axes_titles = []
        # Calculate individual components
        for i, comp in enumerate(to_plot):
            das = self.das.tr[comp]['y']
            for pixel in range(len(das)):
                self.comp_data[i, pixel, :] = das[pixel] * \
                    self.fit_obj.ycomps[comp_inds[i]]
                self.comp_sum[pixel, :] += self.comp_data[i, pixel, :]
            subplot_pos.append([num_rows, num_cols, i + 1])
            axes_titles.append("Component No. " +
                               str(comp_inds[i] + 1) + ": " + comp)
        if len(to_plot) > 1:
            axes_titles.append("Sum of Components")
            subplot_pos.append([num_rows, num_cols, i + 2])
            num_subplots = len(to_plot) + 1
        else:
            num_subplots = len(to_plot)
        self.comp_sum_ind = len(to_plot)
        self.controller.open_topfigure_wrap(
            self, plot_func=self.comp_plot_function,
            num_subplots=num_subplots, num_rows=num_rows,
            num_columns=num_cols, plot_type='2D', editable=True,
            controller=self, axes_titles=axes_titles,
            xlabels=self.main_figure.get_xlabel(),
            ylabels=self.main_figure.get_ylabel(),
            ylimits=self.main_figure.get_ylim())

    def comp_plot_function(self, ind, *args, fig=None, ax=None,
                           color_obj=None, **plot_kwargs):
        if fig is None:
            fig = plt.figure()
            plt.close()
        if ax is None:
            ax = fig.add_subplot(111)
        for key in ('cmap', 'vmin', 'vmax'):
            if key not in plot_kwargs.keys():
                plot_kwargs[key] = self.parent.data_obj.color.get_kwargs()[key]
        if ind == self.comp_sum_ind:
            plot_data = np.transpose(self.comp_sum)
        else:
            plot_data = np.transpose(self.comp_data[ind, :, :])
        img = pcolor(ax, self.x, self.fit_obj.y, plot_data, **plot_kwargs)
        return img

    def open_concentration_figure(self, *args):
        self.controller.open_topfigure_wrap(
            self,
            plot_func=self.plot_concentrations,
            plot_type='line',
            ylabels="Concentration",
            xlabels=self.main_figure.get_ylabel(),
            axes_titles="Concentration Vectors",
            controller=self,
            editable=True,
            legends=self.das_figure.legends)

    def plot_concentrations(self, *args, fig=None, ax=None, fill=0,
                            reverse_z=False, color_order=None,
                            interpolate_colors=False, color_cycle=None,
                            transpose=False, **kwargs):
        if fig is None:
            fig = plt.figure()
            fig.set_tight_layout(True)
        if ax is None:
            ax.add_subplot(111)
        else:
            ax.cla()
        numlines = len(self.fit_obj.ycomps)
        fill = get_fill_list(fill, numlines)
        zord, cycle = set_line_cycle_properties(
            numlines, reverse_z, color_order, cycle=color_cycle,
            interpolate=interpolate_colors)
        if np.any(fill):
            for i, c in enumerate(self.fit_obj.ycomps):
                ax.plot(self.fit_obj.y, c,
                        zorder=zord[i], color=cycle[i], **kwargs)
                if fill[i]:
                    ax.fill_between(self.fit_obj.y, c, zorder=zord[i],
                                    label='_nolegend_',
                                    color=cycle[i], alpha=fill[i])
        else:
            for i, c in enumerate(self.fit_obj.ycomps):
                ax.plot(self.fit_obj.y, c,
                        zorder=zord[i], color=cycle[i], **kwargs)

    def disp_results(self):
        # plots
        self.main_figure.plot()
        self.das_figure.plot()
        self.fit_figure.plot()
        self.update_resid()
        # parameter display
        for w in self.results_display_widgets:
            w.grid_forget()
        paras = self.fit_obj.result.params
        names = {'t0': 'Time Zero',
                 'sigma_1': 'Sigma'}
        tau_list = self.model_info['properties']['tau_order']
        for i in range(self.fit_obj.number_of_decays):
            names['tau_' + str(i + 1)] = 'Tau_' + tau_list[i]
        for i, p in enumerate(paras):
            try:
                if paras[p].vary:
                    try:
                        stderr = paras[p].stderr
                        text = str(ufloat(paras[p].value, stderr))
                    except Exception:
                        text = self.parent.data_obj.time_to_str(paras[p].value)
                else:
                    continue
            except Exception:
                continue
            else:
                self.results_display_widgets.append(
                    tk.ttk.Label(self.results_display_frame, text=names[p]))
                self.results_display_widgets[-1].grid(
                    row=i, column=0, padx=5, pady=2, sticky='w')
                self.results_display_widgets.append(
                    tk.ttk.Label(
                        self.results_display_frame,
                        text=' '.join([text, self.parent.data_obj.time_unit])))
                self.results_display_widgets[-1].grid(
                    row=i, column=1, padx=5, pady=2, sticky='e')
        # Fit report
        try:
            for key, frame in self.report_disp.frames.items():
                frame.grid_forget()
                del self.report_disp.frames[key]
        except Exception:
            pass
        report_dict = lmfitreport_to_dict(self.fit_obj.fit_report())
        report_dict["content"]["Variables"]["visible"] = False
        report_dict["grid_kwargs"] = {'sticky': 'w'}
        self.report_disp.add_frames({"": report_dict})


# %%
class TraceManagerWindow(tk.Toplevel):
    def __init__(self, parent, controller, dat, data_obj, map_figure=None,
                 traces=None, title=None, **manager_kwargs):
        tk.Toplevel.__init__(self, parent)
        self.controller = controller
        self.title(title)
        try:
            fig_kwargs = {'dim': controller.settings['figure_std_size']}
        except Exception:
            fig_kwargs = {}
        row = 0
        if map_figure:
            self.main_figure = TkMplFigure(
                self,
                callbacks={'button_press_event':
                           self.map_figure_callback})
            try:
                self.main_figure.copy_attributes(map_figure)
            except Exception as e:
                messagebox.showerror(message=e, parent=self)
            else:
                self.main_figure.grid(
                    row=row, column=0, padx=0, pady=0, sticky='wnse')
                row += 1
                self.main_figure.plot()

        self.trace_figure = TkMplFigure(
            self,
            callbacks={'button_press_event':
                       self.trace_plot_click_callback},
            plot_function=None,
            # dim=settings['figure_std_size'],
            xlabels=dat['ylabel'],
            ylabels=dat['zlabel'],
            plot_type='line',
            plot_kwargs={'include_fit': False},
            **fig_kwargs)
        self.trace_figure.grid(row=row, column=0, sticky='wnse')
        self.traceManager = TraceManager(self, self.controller,
                                         self.trace_figure, data_obj,
                                         dat=dat, traces=traces,
                                         **manager_kwargs)
        self.traceManager.set_trace_mode()

        self.traceManager.grid(row=0, column=1, rowspan=2)
        center_toplevel(self, parent)

    def map_figure_callback(self, event):
        mode = self.traceManager.trace_mode
        if (event.button == 3
                and mode.lower() in ('kinetic', 'spectral')):
            if mode.lower() == 'kinetic':
                point = event.xdata
            else:
                point = event.ydata
            self.traceManager.get_and_plot_traces(point)
            self.traceManager.manual_point.set(np.round(point, 6))
            try:
                self.traceManager.vars['wavelet_lower'].set(np.min(
                    self.traceManager.traces[
                        self.traceManager.trace_mode].xdata))
            except Exception:
                pass
            else:
                self.traceManager.vars['wavelet_upper'].set(np.max(
                    self.traceManager.traces[
                        self.traceManager.trace_mode].xdata))

    def trace_plot_click_callback(self, event):
        if event.button == 3 or event.dblclick:
            self.controller.open_topfigure_wrap(
                self, fig_obj=self.trace_figure, editable=True,
                controller=self)


# %%
class SVDAnalysis(tk.Toplevel):
    def __init__(self, parent, controller, dat, data_obj, dim=(400, 300),
                 case='TA', xlim=None, ylim=None, run=True):
        tk.Toplevel.__init__(self, parent)
        self.title("SVD Analysis")
        self.data_obj = data_obj
        self.controller = controller
        if controller:
            move_toplevel_to_default(self, controller)
        self.parent = parent
        self.data = dat
        self.frames = {}
        self.entries = {}
        self.vars = {}
        self.buttons = {}

        # initialize frames
        self.frames['ui'] = CustomFrame(self, dim=(1, 4))
        self.frames['options'] = CustomFrame(
            self.frames['ui'], dim=(3, 2), border=True)
        self.frames['window'] = CustomFrame(
            self.frames['options'], dim=(3, 1), border=False)
        self.frames['commands'] = CustomFrame(
            self.frames['ui'], dim=(2, 3), border=True)
        # self.frames['command_sub'] = CustomFrame(
        #     self.frames['commands'], dim=(1, 2), border=True)
        self.frames['save'] = CustomFrame(
            self.frames['ui'], dim=(1, 4), border=True)
        # resizability
        for i in range(3):
            self.columnconfigure(i, weight=1)
        for i in range(2):
            self.rowconfigure(i, weight=1)

        # figures
        fig_kwargs = {'axes_titles': 'Data',
                      'callbacks': {'button_press_event':
                                    self.main_canvas_callback}}
        self.main_figure = self.controller.tk_mpl_figure_wrap(
            self, plot_function=self.data_obj.plot_ta_map,
            xlabels=self.data_obj.set_xlabel(),
            ylabels=self.data_obj.set_ylabel(),
            xlimits=xlim,
            ylimits=ylim,
            clabels=self.data_obj.zlabel,
            color_obj=self.data_obj.color, **fig_kwargs)
        self.main_figure.plot()
        self.main_figure.grid(row=1, column=0, padx=(20, 5), pady=5)

        fig_kwargs = {'axes_titles': 'Reduced Data',
                      'callbacks': {'button_press_event':
                                    self.red_dat_canvas_callback}}
        self.red_data_fig = self.controller.tk_mpl_figure_wrap(
            self,
            plot_function=self.plot_red_data,
            xlabels=self.data_obj.xlabel,
            clabels=self.data_obj.zlabel,
            ylabels=self.data_obj.ylabel,
            color_obj=self.data_obj.color, **fig_kwargs)
        self.red_data_fig.grid(row=1, column=1, padx=(5, 20), pady=5)

        lbl = self.data_obj.zlabel[::-1]
        if re.search('\(', lbl):
            lbl = "".join([lbl[:re.search('\)', lbl).span()[1]],
                           '.u.a',
                           lbl[re.search('\(', lbl).span()[0]:]])[::-1]

        self.left_svec_fig = self.controller.tk_mpl_figure_wrap(
            self,
            # dim=settings['figure_std_size'],
            plot_type='line',
            plot_function=lambda *args, c='Left', **kwargs:
                self.plot_svec(
                    *args, case=c, **kwargs),
                xlabels=self.data_obj.ylabel, ylabels=lbl,
                axes_titles="Left Singular Vectors",
                callbacks={'button_press_event': lambda *args, c='Left':
                           self.vector_canvas_callback(*args, case=c)})
        self.left_svec_fig.grid(row=3, column=0, padx=(20, 5), pady=5)

        self.right_svec_fig = self.controller.tk_mpl_figure_wrap(
            self,
            # dim=settings['figure_std_size'],
            plot_type='line',
            plot_function=lambda *args, c='Right', **kwargs:
            self.plot_svec(
                *args, case=c, **kwargs),
            xlabels=self.data_obj.xlabel, ylabels=lbl,
            axes_titles="Right Singular Vectors",
            callbacks={'button_press_event': lambda *args, c='Right':
                       self.vector_canvas_callback(*args, case=c)})
        self.right_svec_fig.grid(row=3, column=1, padx=(5, 20), pady=5)
        self.frames['ui'].grid(row=0, column=2, rowspan=5, sticky='wnse')

        # option panel
        self.frames['options'].grid(row=0, column=0, sticky='wnse',
                                    padx=2, pady=2)

        self.frames['window'].grid(row=0, column=0, sticky='wnse',
                                   columnspan=2)
        tk.Label(self.frames['window'], text='Window').grid(row=0, column=0,
                                                            sticky=tk.W)
        tk.Label(self.frames['window'], text='spec. range:').grid(
            row=1, column=0)
        if xlim is not None:
            self.xlower = tk.DoubleVar(value=xlim[0])
            self.xupper = tk.DoubleVar(value=xlim[1])
        else:
            self.xlower = tk.DoubleVar(
                value=np.round(np.min(np.ma.array(
                    self.data.x, mask=np.isnan(self.data.x))), 2))
            self.xupper = tk.DoubleVar(
                value=np.round(np.max(np.ma.array(
                    self.data.x, mask=np.isnan(self.data.x))), 2))
        self.entries['xlower'] = tk.Entry(
            self.frames['window'], textvariable=self.xlower, width=8)
        self.entries['xlower'].bind('<Return>', lambda *args:
                                    self.set_window(*args, case='x'))
        self.entries['xlower'] .grid(row=1, column=1)
        self.yupper = tk.DoubleVar(value=np.max(self.data.y))
        self.entries['xupper'] = tk.Entry(
            self.frames['window'], textvariable=self.xupper, width=8)
        self.entries['xupper'].bind('<Return>', lambda *args:
                                    self.set_window(*args, case='x'))
        self.entries['xupper'].grid(row=1, column=2)
        tk.Label(self.frames['window'], text='time range:').grid(
            row=2, column=0)
        if ylim:
            self.ylower = tk.DoubleVar(value=ylim[0])
            self.yupper = tk.DoubleVar(value=ylim[1])
        else:
            self.ylower = tk.DoubleVar(
                value=np.round(np.min(np.ma.array(
                    self.data.y, mask=np.isnan(self.data.y))), 3))
            self.yupper = tk.DoubleVar(
                value=np.round(np.max(np.ma.array(
                    self.data.y, mask=np.isnan(self.data.y))), 3))
        self.entries['ylower'] = tk.Entry(
            self.frames['window'], textvariable=self.ylower, width=8)
        self.entries['ylower'] .bind('<Return>', lambda *args:
                                     self.set_window(*args, case='y'))
        self.entries['ylower'] .grid(row=2, column=1)
        self.entries['yupper'] = tk.Entry(
            self.frames['window'], textvariable=self.yupper, width=8)
        self.entries['yupper'].bind('<Return>', lambda *args:
                                    self.set_window(*args, case='y'))
        self.entries['yupper'].grid(row=2, column=2)

        ttk.Button(self.frames['window'], text='Update', command=lambda *args:
                   self.set_window(*args, case='both')).grid(row=3, column=0,
                                                             columnspan=3)

        # command panel

        ttk.Button(self.frames['commands'], text="Display singular values",
                   command=self.disp_sval_callback).grid(
                       row=0, column=0, sticky='we', padx=5, pady=5,
                       columnspan=2)

        ttk.Button(self.frames['commands'],
                   text='Plot singular values',
                   command=self.plot_sval_callback).grid(
                       row=1, column=0, sticky='we',
                       padx=5, pady=5, columnspan=2)

        ttk.Button(self.frames['commands'],
                   text='Reset/Update',
                   command=self.run_svd).grid(
                       row=6, column=0, sticky='we',
                       padx=5, pady=5, columnspan=2)

        tk.Label(self.frames['commands'],
                 text='Show components:').grid(
                     row=2, column=0, sticky=tk.W, columnspan=2)
        self.vars['select_comp'] = tk.StringVar(value='2')
        self.entries['select_comp'] = tk.Entry(
            self.frames['commands'],
            textvariable=self.vars['select_comp'],
            width=10)
        self.entries['select_comp'].bind('<Return>', self.show_select_comp)
        self.entries['select_comp'].grid(row=3, column=0, sticky=tk.W,
                                         padx=5)
        self.comp_select_mode = tk.StringVar(value='Cutoff')
        comp_select_mode_select = tk.ttk.OptionMenu(
            self.frames['commands'], self.comp_select_mode,
            self.comp_select_mode.get(),
            'Cutoff', 'Select',
            command=self.show_option_callback)
        comp_select_mode_select.config(width=5)
        comp_select_mode_select.grid(row=3, column=1, sticky=tk.W, padx=5)
        self.buttons['swap_sign'] = ttk.Button(
            self.frames['commands'],
            text='Swap sign',
            command=self.swap_component_sign,
            state='disabled')
        self.buttons['swap_sign'].grid(row=4, column=1, padx=5)
        self.buttons['auto_sign'] = ttk.Button(
            self.frames['commands'],
            text='Auto sign',
            command=self.comp_auto_sign,
            state='disabled')
        self.buttons['auto_sign'].grid(row=4, column=0, padx=5)

        self.frames['commands'].grid(row=1, column=0, columnspan=2,
                                     sticky='wnse', padx=5, pady=5)
        self.frames['commands'].grid(row=1, column=0, sticky='wnse',
                                     padx=2, pady=2)

        # saving and communication
        tk.Label(self.frames['save'],
                 text='Save for analysis').grid(
                     row=0, column=0, sticky='nw')
        self.frames['save'].grid(row=2, column=0, sticky='wnse',
                                 padx=2, pady=2)
        self.save_opt = tk.StringVar(value='Left SV')
        tk.ttk.OptionMenu(
            self.frames['save'], self.save_opt,
            'Left SV', 'Left SV', 'Right SV').grid(
                row=1, column=0, sticky='we', padx=3, pady=3)
        if self.parent:
            self.buttons['send_to_main'] = ttk.Button(
                self.frames['save'],
                text='Send to main app',
                command=self.send_to_main_app,
                state='disabled')
            self.buttons['send_to_main'].grid(
                row=2, column=0, sticky='we', padx=3, pady=3)

        self.buttons['save'] = tk.ttk.Button(
            self.frames['save'],
            text='Save to file',
            command=lambda c='vect': self.save_to_file(case=c),
            state='disabled')
        self.buttons['save'].grid(row=2, column=1, sticky='we',
                                  padx=3, pady=3)

        self.buttons['save_red_data'] = ttk.Button(
            self.frames['save'],
            text='Save red. data',
            command=lambda c='red': self.save_to_file(case=c),
            state='disabled')
        self.buttons['save_red_data'].grid(row=3, column=0, columnspan=2,
                                           sticky='we', padx=3, pady=3)

        # misc. variables

        self.yr = slice(0, len(self.data.y))
        self.xr = slice(0, len(self.data.x))
        self.exclude = []
        self.svd_done = False
        self.comps = range(2)
        self.s = None
        if run:
            self.run_svd(showerror=False)

    def vector_canvas_callback(self, event, case='Left'):
        if self.svd_done and (event.button == 3 or event.dblclick):
            self.controller.open_topfigure_wrap(
                self, plot_func=lambda *args, **kwargs:
                    self.plot_svec(*args, case=case, **kwargs),
                    fig_obj=(self.left_svec_fig
                             if case.lower() == 'left'
                             else self.right_svec_fig),
                    editable=True, controller=self)

    def main_canvas_callback(self, event):
        if (event.button == 3 or event.dblclick):
            self.controller.open_topfigure_wrap(
                self, fig_obj=self.main_figure, plot_func=self.data.plot_2d,
                editable=True, controller=self, color_obj=self.data_obj.color)

    def red_dat_canvas_callback(self, event):
        if self.svd_done and (event.button == 3 or event.dblclick):
            self.controller.open_topfigure_wrap(
                self, fig_obj=self.red_data_fig,
                plot_func=self.plot_red_data, controller=self,
                editable=True, color_obj=self.data_obj.color)

    def show_select_comp(self, *args):
        try:
            self.set_components()
            self.calc_red_data()
            self.plot_svectors()
        except Exception:
            self.run_svd()

    def set_components(self, *args):
        try:
            self.comps = [
                int(x)-1 for x in self.vars['select_comp'].get().split(",")]
        except Exception:
            if self.vars['select_comp'].get() == "":
                self.comps = range(2)
            else:
                self.lift()
                messagebox.showerror("Invalid input",
                                     "Please enter component numbers "
                                     + "separated by commas",
                                     parent=self)
                self.lift()
        if self.comp_select_mode.get() == 'Cutoff':
            self.vars['select_comp'].set(str(np.max(self.comps) + 1))
            self.comps = range(int(self.vars['select_comp'].get()))

    def save_to_file(self, case='red'):
        file = save_box(fext='.mat',
                        filetypes=[('text files', '.txt'),
                                   ('Matlab files', '.mat')],
                        fname='TA_SVD_'
                        + str(len(list(self.comps))) + '_comp',
                        parent=self)
        if file is not None:
            if re.search('red', case, re.I):
                self.data_obj.save_data_matrix(file, matrix=self.reduced_data,
                                               x=self.data.x[self.xr],
                                               y=self.data.y[self.yr])
            else:
                case = self.save_opt.get()
                try:
                    self.write_trace_obj(case=case)
                except Exception:
                    return
                trace_keys = list(self.traces.tr.keys())
                trace_type = 'Kinetic' if re.search(
                    'left', case, re.I) else 'Spectral'
                self.traces.save_traces(
                    file, trace_keys=trace_keys, save_fit=False,
                    spec_quantity=self.data_obj.get_x_mode(),
                    spec_unit=self.data_obj.spec_unit_label,
                    trace_type=trace_type)

    def swap_component_sign(self):
        self.set_components()
        self.calc_red_data(update_plot=False)
        for i in range(len(self.comps)):
            self.ured[:, i] = -self.u[:, self.comps[i]]
            self.u[:, self.comps[i]] = -self.u[:, self.comps[i]]
            self.vred[:, i] = -self.v[self.comps[i], :]
            self.v[self.comps[i], :] = -self.v[self.comps[i], :]
        self.plot_svectors()

    def set_vec_sign(self, i, sign):
        self.ured[:, i] = sign*self.u[:, self.comps[i]]
        self.u[:, self.comps[i]] = sign*self.u[:, self.comps[i]]
        self.vred[:, i] = sign*self.v[self.comps[i], :]
        self.v[self.comps[i], :] = sign*self.v[self.comps[i], :]

    def comp_auto_sign(self):
        self.set_components()
        self.calc_red_data(update_plot=False)
        for i in range(len(self.comps)):
            if (np.max(self.u[np.argmax(np.abs(self.u[:, self.comps[i]])),
                              self.comps[i]]) >= 0):
                sign = 1
            else:
                sign = -1
            self.set_vec_sign(i, sign)
        self.plot_svectors()

    def show_option_callback(self, *args):
        if self.comp_select_mode.get() == 'Cutoff':
            self.vars['select_comp'].set("")
        else:
            e = ""
            for i in range(np.max(self.comps) + 1):
                e = e + str(i+1) + ","
            e = e[:-1]
            self.vars['select_comp'].set(e)
        self.show_select_comp()

    def write_trace_obj(self, case='left'):
        if re.search('left', case, re.I):
            self.traces = TATrace(trace_type='time',
                                  xdata=self.data.y[self.yr],
                                  xlabel=self.left_svec_fig.get_xlabel())
            for i in self.comps:
                self.traces.tr['Comp. ' + str(i+1)] = {
                    'y': self.u[:, i],
                    'val': self.comps[i - 1]}
        elif re.search('right', case, re.I):
            self.traces = TATrace(trace_type='spectral',
                                  xdata=self.data.x[self.xr],
                                  xlabel=self.right_svec_fig.get_xlabel())
            for i in self.comps:
                self.traces.tr['Comp. ' + str(i+1)] = {
                    'y': self.v[i, :],
                    'val': self.comps[i - 1]}
        else:
            return
        self.traces.ylabel = self.save_opt.get()

    def send_to_main_app(self):
        if self.svd_done:
            self.write_trace_obj(case=self.save_opt.get())
            self.parent.trace_opts.traces[self.save_opt.get()] = self.traces
            self.parent.trace_opts.set_trace_mode(
                trace_mode=self.save_opt.get())
            self.controller.frames[FitTracePage]._update_content()
            self.parent.update()

    def set_xlimits(self, *args, update=True):
        self.get_x_indices()
        self.xlower.set(np.round(self.data.x[self.xr][0], 2))
        self.xupper.set(np.round(self.data.x[self.xr][-1], 2))
        for fig in self.main_figure, self.red_data_fig:
            fig.set_xlim([self.xlower.get(), self.xupper.get()],
                         update_canvas=update)

    def get_x_indices(self):
        try:
            ind = [np.where(self.data.x >= self.xlower.get())[0][0],
                   np.where(self.data.x <= self.xupper.get())[0][-1]]
        except Exception:
            raise
            ind = [0, len(self.data.x)-1]
        self.xr = slice(ind[0], ind[1] + 1)

    def set_window(self, *args, case='x', update=True):
        if case == 'both':
            case = ['x', 'y']
        else:
            case = [case]
        try:
            for c in case:
                eval('self.set_' + c + 'limits(update = update)')
        except Exception:
            raise
        if update and self.svd_done:
            self.run_svd()

    def set_ylimits(self, *args, update=True):
        self.get_y_indices()
        self.ylower.set(self.data.y[self.yind[0]])
        self.yupper.set(self.data.y[self.yind[1]])
        for fig in self.main_figure, self.red_data_fig:
            fig.set_ylim([self.ylower.get(), self.yupper.get()],
                         update_canvas=update)

    def get_y_indices(self):
        try:
            self.yind = [np.where(self.data.y >= self.ylower.get())[0][0],
                         np.where(self.data.y <= self.yupper.get())[0][-1]]
        except Exception:
            self.yind = [0, len(self.data.y)-1]
        self.yr = slice(self.yind[0], self.yind[1] + 1)

    def plot_sval_callback(self, *args):
        if self._get_sing_values():
            self.controller.open_topfigure_wrap(
                self, plot_func=lambda *args, **kwargs:
                    self.plot_sval(*args, **kwargs),
                plot_type='line',
                controller=self,
                xlabels='Component No.',
                ylabels='Singular Value',
                axes_title='Singular Values',
                editable=True, dim=[500, 400])

    def _get_sing_values(self):
        self.get_y_indices()
        self.get_x_indices()
        try:
            self.u, self.s, self.v = np.linalg.svd(
                self.data.z[self.yr, self.xr], full_matrices=False)
        except Exception:
            return False
        else:
            return True

    def disp_sval_callback(self, *args):
        if self._get_sing_values():
            disp_dct = {}
            val_per_page = 10
            for i in range(10):
                key = "-".join([str(i*val_per_page + 1),
                               str((i+1)*val_per_page)])
                string = str(self.s[i*val_per_page])
                string += "\n".join([str(self.s[i*val_per_page + j])
                                    for j in range(1, val_per_page)])
                disp_dct[key] = {"type": "label",
                                 "content": string,
                                 "grid_kwargs": {"pady": 5, "padx": 5}}
            MultiDisplayWindow(
                self, header="Singular Values", input_dict=disp_dct)

    def plot_sval(self, *args, fig=None, ax=None, fill=0, marker='x',
                  reverse_z=False, color_order=None, color_cycle=None,
                  interpolate_colors=False, linestyle=None,
                  # legend_kwargs=None,
                  transpose=False, **plot_kwargs):
        if fig is None:
            fig = plt.figure()
            plt.close()
            fig.set_tight_layout(True)
        if ax is None:
            fig.add_subplot(111)
        ax.cla()
        ax.set_xlim(0, 10.5)
        sv_plot = ax.plot(np.array(range(1, len(self.s)+1)), self.s,
                          marker=marker, linestyle=linestyle, **plot_kwargs)
        ax.plot(np.array(self.comps) + 1,
                self.s[self.comps], 'h')
        ax.set_xlabel('Component No.')
        ax.set_ylabel('Singular Value')
        return sv_plot

    def run_svd(self, *args, showerror=True):
        # run SVD
        self.svd_done = False
        self.set_components()
        self.set_window(case='both', update=False)
        z = self.data.z[self.yr, self.xr]
        try:
            self.u, self.s, self.v = np.linalg.svd(
                np.ma.array(z, mask=np.isnan(z)), full_matrices=False)
            self.calc_red_data(update_plot=True)
        except Exception as e:
            if showerror:
                msg = str(e)
                if re.search('converge', msg, re.I):
                    msg += ".\nPlease try adjusting window."
                messagebox.showerror(parent=self, message=msg)
        else:
            self.plot_svectors()
            self.svd_done = True
            for widget in self.buttons.values():
                widget.config(state='normal')

    def calc_red_data(self, update_plot=True):
        self.ured = np.transpose([self.u[:, i] for i in self.comps])
        self.vred = [self.v[i, :] for i in self.comps]
        self.sred = [self.s[i] for i in self.comps]
        self.reduced_data = np.dot(
            self.ured, np.dot(np.diag(self.sred), self.vred))
        self.vred = np.transpose(self.vred)
        if update_plot:
            self.plot_reduced_data()

    def plot_reduced_data(self):
        if self.comp_select_mode.get() == 'Cutoff':
            title = str(np.max(self.comps) + 1) + ' component'
            if np.max(self.comps) > 0:
                title = title + 's'
        else:
            title = 'Component'
            if len(self.comps) > 1:
                title = title + 's '
                for c in self.comps[:-1]:
                    title = title + str(c + 1) + ', '
                title = title[:-2] + ' & ' + str(self.comps[-1] + 1)
            else:
                title = title + ' ' + str(self.comps[0] + 1)
        self.red_data_fig.set_axes_title(title, update_canvas=False)
        self.red_data_fig.plot()

    def plot_red_data(self, *args, fig=None, ax=None, fill=0, reverse_z=False,
                      color_order=None, color_cycle=None,
                      interpolate_colors=False,
                      # legend_kwargs=None,
                      **kwargs):
        if fig is None:
            fig = self.red_data_fig
        if ax is None:
            ax = self.red_data_fig.get_axes()
        try:
            im = pcolor(ax, self.data.x[self.xr], self.data.y[self.yr],
                        self.reduced_data, **kwargs)
        except ValueError as ve:
            self.lift()
            messagebox.showerror(message=ve, parent=self)
            self.lift()
        except Exception:
            raise
        else:
            return im

    def plot_svectors(self):
        for fig in (self.right_svec_fig, self.left_svec_fig):
            fig.set_legend(entries=["Comp. " + str(c + 1)
                           for c in self.comps], update_canvas=False)
        self.left_svec_fig.plot()
        self.right_svec_fig.plot()

    def plot_svec(self, *args, fig=None, ax=None, case='Left', fill=0,
                  reverse_z=False, color_order=None, color_cycle=None,
                  interpolate_colors=False, legend_kwargs=None,
                  transpose=False, **plot_kwargs):
        if legend_kwargs is None:
            legend_kwargs = {}
        if fig is None:
            fig = plt.figure()
            plt.close()
            fig.set_tight_layout(True)
        if ax is None:
            ax = fig.add_subplot(111)
        ax.cla()
        if case.lower() == 'left':
            xval = self.data.y[self.yr]
            vec = self.ured
            xlabel = self.main_figure.get_ylabel()
        else:
            xval = self.data.x[self.xr]
            vec = np.transpose(self.vred, axes=(0, 1))
            xlabel = self.main_figure.get_xlabel()
        numlines = np.shape(vec)[1]
        fill = get_fill_list(fill, numlines)
        lines = []
        zord, cycle = set_line_cycle_properties(
            numlines, reverse_z, color_order, cycle=color_cycle,
            interpolate=interpolate_colors)
        for i in range(numlines):
            lines.append(
                ax.plot(xval, vec[:, i], zorder=zord[i], color=cycle[i],
                        **plot_kwargs))
            if fill[i]:
                ax.fill_between(xval, vec[:, i],
                                label='_nolegend_',
                                zorder=zord[i], color=cycle[i], alpha=fill[i])
        ax.axhline(ls='--', color='grey')
        ax.set_xlabel(xlabel)
        ax.legend(['Comp. ' + str(i+1) for i in self.comps], **legend_kwargs)
        if len(self.comps) == 1:
            ax.set_title(case + ' singular vector')
        else:
            ax.set_title(case + ' singular vectors')
        return lines


# %% Deprecated
# class SpectralShift(tk.Toplevel):
#     def __init__(self, parent, controller, dat, dim=400, inputCentroid=False):
#         tk.Toplevel.__init__(self, parent)
#         self.title("Spectral Shift Tool")
#         self.controller = controller
#         move_toplevel_to_default(self, controller)
#         self.plot_data = dat
#         self.parent = parent
#         self.vars = {}
#         self.entries = {}
#         self.selects = {}
#         self.canvas = {}
#         self.figures = {}
#         self.axes = {}
#         self.toolbars = {}
#         self.plots = {}
#         self.inputCentroid = inputCentroid
#         self.t_range = None

#         self.frames = {'commands': CustomFrame(self, dim=(2, 1))}
#         self.frames['commandsShiftCurve'] = CustomFrame(self.frames['commands'],
#                                                         dim=(6, 1), border=True)
#         self.frames['commandsSpectralShift'] = CustomFrame(self.frames['commands'],
#                                                            dim=(3, 1), border=True)

#         for k, i in zip(('data', 'shift', 'result'), range(3)):
#             self.frames[k + 'PlotToolbar'] = CustomFrame(self, dim=(1, 1))
#             self.figures[k] = plt.figure()
#             plt.close()
#             self.figures[k].set_tight_layout(True)
#             self.axes[k] = self.figures[k].add_subplot(111)
#             self.axes[k].format_coord = lambda x, y: ""
#             self.canvas[k] = FigureCanvasTkAgg(self.figures[k], self)
#             self.canvas[k].get_tk_widget().config(width=dim, height=dim)
#             self.toolbars[k] = NavigationToolbar2Tk(
#                 self.canvas[k], self.frames[k + 'PlotToolbar'])
#             self.toolbars[k].update()
#             self.canvas[k].get_tk_widget().grid(row=0, column=i, sticky='wnse',
#                                                 padx=5, pady=5)
#             self.frames[k + 'PlotToolbar'].grid(row=1, column=i, sticky='wnse',
#                                                 padx=5)

#         self.canvas['result'].callbacks.connect(
#             'button_press_event', self.resultCanvCallback)

#         self.plots['data'] = self.plot_data.plot_2d(fig=self.figures['data'], ax=self.axes['data'],
#                                                      canv=self.canvas['data'], **self.parent.plot_kwargs)
#         self.canvas['data'].draw()

#         tk.Label(self.frames['commandsShiftCurve'], text='time window:').grid(
#             row=0, column=0, sticky=tk.W)
#         self.vars['tlower'] = tk.DoubleVar(value=np.min(self.plot_data.y)
#                                            if np.min(self.plot_data.y) > 0 else 0.0)
#         self.vars['tupper'] = tk.DoubleVar(value=np.max(self.plot_data.y))
#         for k, i in zip(('tlower', 'tupper'), (1, 2)):
#             self.entries[k] = tk.Entry(self.frames['commandsShiftCurve'],
#                                        textvariable=self.vars[k], width=6)
#             self.entries[k].grid(row=0, column=i, sticky=tk.W)
#             self.entries[k].bind('<Return>', self.showShiftCurve)

#         tk.Label(self.frames['commandsShiftCurve'], text='Shift curve:').grid(row=0, column=3,
#                                                                               sticky=tk.W)
#         self.vars['shiftCurveMode'] = tk.StringVar(value='Centroid')
#         self.selects['shiftCurveMode'] = tk.ttk.OptionMenu(self.frames['commandsShiftCurve'],
#                                                            self.vars['shiftCurveMode'], self.vars['shiftCurveMode'].get(
#         ),
#             'Centroid', 'Maximum', 'Minimum', command=self.showShiftCurve)
#         self.selects['shiftCurveMode'].grid(row=0, column=4, sticky=tk.W)

#         ttk.Button(self.frames['commandsShiftCurve'], text='Smooth',
#                    command=self.smoothShiftCurve).grid(row=0, column=5, sticky=tk.W)

#         ttk.Button(self.frames['commandsSpectralShift'], text='Shift',
#                    command=self.shiftSpectra).grid(row=0, column=0)
#         ttk.Button(self.frames['commandsSpectralShift'], text='Save to app',
#                    command=lambda *args: self.saveShift(*args, case='inapp')).grid(row=0, column=1)
#         ttk.Button(self.frames['commandsSpectralShift'], text='Save to file',
#                    command=lambda *args: self.saveShift(*args, case='file')).grid(row=0, column=2)

#         self.frames['commandsShiftCurve'].grid(row=0, column=0,
#                                                sticky='wnse', padx=1, pady=1)
#         self.frames['commandsSpectralShift'].grid(row=0, column=1,
#                                                   sticky='wnse', padx=1, pady=1)
#         self.frames['commands'].grid(row=2, column=0, columnspan=3,
#                                      sticky='wnse')

#         if not self.inputCentroid:
#             self.showShiftCurve()
#         else:
#             self.shiftCurve = self.inputCentroid
#             self.plotShiftCurve()

#     def saveShift(self, *args, case='inapp'):
#         if case.lower() == 'inapp':
#             self.data_obj.delA = self.shiftedMap
#             self.data_obj.wavelengths = self.plot_data.x
#             self.data_obj.spec_axis[self.data_obj.get_x_mode()] = self.plot_data.x
#             self.data_obj._lambda_lim_index = [0, len(self.data_obj.wavelengths) - 1]
#             self.parent.update_main_plot()
#         elif case.lower() == 'file':
#             self.data_obj.save_data_matrix(matrix=self.shiftedMap, x=self.plot_data.x,
#                                   y=self.plot_data.y, fname='shiftedMap',
#                                   header='Spectrally shifted data. Mode: ' +
#                                   self.vars['shiftCurveMode'].get(),
#                                   parent=self)

#     def showShiftCurve(self, *args):
#         self.t_range = [np.where(self.data_obj.time_delays >= self.vars['tlower'].get())[0][
#             0 - int(self.data_obj.time_delays[-1] < self.data_obj.time_delays[0])],
#             np.where(self.data_obj.time_delays <= self.vars['tupper'].get())[0][
#             -1 + int(self.data_obj.time_delays[-1] < self.data_obj.time_delays[0])]]
#         if self.vars['shiftCurveMode'].get() == 'Centroid':
#             self.t_range = ([np.where(self.data_obj.time_delays >= 0)[0][0 - int(self.data_obj.time_delays[-1] < self.data_obj.time_delays[0])],
#                             len(self.data_obj.time_delays)] if self.data_obj.time_delays[0] <= 0 else [0, len(self.data_obj.time_delays)])
#             self.data_obj.calculate_centroid(self.data_obj._lambda_lim_index, t_range=self.t_range)
#             self.shiftCurve = self.data_obj.centroid
#             self.rawShiftCurve = self.data_obj.centroid.tr[None]['y']
#         else:
#             self.shiftCurve = TATrace()
#             self.shiftCurve.xdata, y = self.data_obj.find_spec_maximum(self.data_obj._lambda_lim_index, t_range=self.t_range,
#                                                             mode='max' if self.vars['shiftCurveMode'].get()
#                                                             == 'Maximum' else 'min')
#             self.shiftCurve.tr[None] = {}
#             self.shiftCurve.tr[None]['y'] = y
#         self.plotShiftCurve()

#     def plotShiftCurve(self):
#         self.axes['shift'].cla()
#         self.axes['shift'].set_ylabel(self.axes['data'].get_ylabel())
#         self.axes['shift'].set_xlabel(self.axes['data'].get_xlabel())
#         self.axes['shift'].plot(self.shiftCurve.tr[None]
#                                 ['y'], self.shiftCurve.xdata)
#         self.axes['shift'].invert_yaxis()
#         self.canvas['shift'].draw()

#     def shiftSpectra(self):
#         shiftedMap = self.data_obj.spectral_shift(self.shiftCurve.tr[None]['y'],
#                                         spec=self.plot_data.x,
#                                         td=self.shiftCurve.xdata,
#                                         data_mat=self.plot_data.z[slice(*self.t_range), :])
#         self.shiftedMap = []
#         for z in self.plot_data.z:
#             self.shiftedMap.append(z)
#         self.shiftedMap = np.array(self.shiftedMap)
#         self.shiftedMap[slice(*self.t_range), :] = shiftedMap
#         self.updateShiftMap()

#     def resultCanvCallback(self, event):
#         if event.button == 3:
#             self.controller.open_topfigure_wrap(self, plot_func=lambda *args, **kwargs: self.plotShiftedMap(
#                 *args, **kwargs),
#                 editable=True, controller=self,
#                 clims=self.data_obj.color.clims,

#                 #                            colorSettingsDict = self.controller.colorSettings,
#                 color_obj=self.data_obj.color)

#     def updateShiftMap(self):
#         self.plots['result'] = self.plotShiftedMap(
#             fig=self.figures['result'], ax=self.axes['result'],
#             **self.data_obj.color.get_kwargs())
#         self.canvas['result'].draw()

#     def plotShiftedMap(self, *args, fig=None, ax=None, **kwargs):
#         if fig is None:
#             fig = plt.figure()
#             plt.close()
#             fig.set_tight_layout(True)
#         if ax is None:
#             ax = fig.add_subplot(111)
#         if 'cmap' not in kwargs.keys():
#             kwargs['cmap'] = self.plot_data.cmap
#         ax.cla()
#         img = pcolor(ax, self.plot_data.x, self.plot_data.y,
#                      self.shiftedMap, **kwargs)
#         ax.set_ylim(self.axes['data'].get_ylim())
#         ax.set_ylabel(self.axes['data'].get_ylabel())
#         ax.set_xlabel(self.axes['data'].get_xlabel())
#         return img


# %%
class CPMFit(tk.Toplevel):
    def __init__(self, parent, controller, dat, data_obj, dim=None,
                 color_obj=None):
        if dim is None:
            dim = controller.settings['figure_std_size']
        tk.Toplevel.__init__(self, parent)
        self.data_obj = data_obj
        self.title("CPM Analysis")
        self.controller = controller
        move_toplevel_to_default(self, controller)
        self.parent = parent
        self.frames = {}
        self.vars = {}
        self.dat = dat
        self.plot_func_dict = {'sigma': self.plot_sigma,
                               'time zero': self.plot_t0,
                               'fit map': self.plot_fit_map,
                               't0 in map': self.plot_t0_map}

        # Frames
        # self.frames['mainToolbar'] = tk.Frame(self)
        # self.frames['traceToolbar'] = tk.Frame(self)
        # self.frames['fitToolbar'] = tk.Frame(self)

        self.widget_frame = CustomFrame(self, border=False, dim=(2, 1))

        self.fit_frame = GroupBox(
            self.widget_frame, text='Fit Options', dim=(2, 2))
        self.model_frame = GroupBox(self.fit_frame, text='Model', dim=(3, 2))
        self.fit_win_frame = GroupBox(
            self.fit_frame, text='Window', dim=(3, 4))
        self.fit_opt_frame = CustomFrame(
            self.fit_frame, border=True, dim=(2, 6))
        self.t0_guess_frame = CustomFrame(
            self.fit_opt_frame, border=True, dim=(4, 2))

        self.results_frame = GroupBox(
            self.widget_frame, text='Results', dim=(2, 2))
        self.plot_frame = CustomFrame(
            self.results_frame, border=True, dim=(3, 4))
        self.navi_frame = CustomFrame(self.plot_frame, border=False,
                                      dim=(2, 1))
        self.savelist_frame = GroupBox(
            self.results_frame, text='Save', dim=(2, 3))
        self.fit_para_frame = CustomFrame(
            self.results_frame, border=True, dim=(2, 3))
        self.fit_para_display = CustomFrame(self.results_frame,
                                            border=True, dim=(1, 2),
                                            width=300)
        self.hermite_para_disp = CustomFrame(self.fit_para_display,
                                             border=True, dim=(5, 4),
                                             width=300)
        # self.hermite_para_disp_blank = CustomFrame(
        #     self.fit_para_display)

        # Figures

        self.main_figure = self.controller.tk_mpl_figure_wrap(
            self, dim=dim, plot_function=dat.plot_2d, xlabels=dat.xlabel,
            ylabels=dat.ylabel, clabels=dat.clabel, xlimits=dat.xlims,
            ylimits=dat.ylims,
            callbacks={'button_press_event':
                       self.main_plot_click_callback},
            color_obj=color_obj)

        self.main_figure.plot()
        self.main_figure.grid(row=1, column=0, padx=5, pady=5, sticky='wnse')

        self.trace_figure = self.controller.tk_mpl_figure_wrap(
            self,
            dim=dim,
            plot_function=self.plot_trace,
            xlabels=dat.ylabel,
            ylabels=dat.clabel,
            xlimits=dat.ylims,
            plot_type='linefit',
            # fit_kwargs=settings['fit_kwargs'],
            callbacks={'button_press_event':
                       self.open_trace_figure})

        self.trace_figure.grid(row=1, column=1, columnspan=1,
                               padx=5, pady=5, sticky='wnse')

        self.fit_figure = self.controller.tk_mpl_figure_wrap(
            self, dim=dim, plot_function=dat.plot_2d, xlabels=dat.xlabel,
            ylabels=dat.ylabel, clabels=dat.clabel, xlimits=dat.xlims,
            ylimits=dat.ylims, color_obj=color_obj,
            callbacks={'button_press_event':
                       self.fit_figure_callback})
        self.fit_figure.grid(row=1, column=2, padx=5, pady=5)

        # Fit frame
        self.fit_frame.grid(row=0, column=0, columnspan=1, sticky='wnse',
                            padx=5, pady=5)

        # fit window frame
        self.fit_win_frame.grid(
            row=0, column=0, columnspan=1, sticky='wnse', padx=2, pady=2)
        tk.Label(self.fit_win_frame,
                 text="Traces to fit:").grid(row=0, column=0, sticky='w')
        self.window_mode = tk.StringVar(value='Range')
        tk.ttk.OptionMenu(
            self.fit_win_frame, self.window_mode, 'Range', 'Range', 'Single',
            command=self.window_mode_callback).grid(
                row=0, column=1, columnspan=2, sticky=tk.W)

        x_lim = self.data_obj.get_xlim()
        tk.Label(self.fit_win_frame, text="x window:").grid(
            row=1, column=0, sticky=tk.W, columnspan=2)
        self.xlower = tk.DoubleVar(value=x_lim[0])
        self.xlower_entry = tk.Entry(
            self.fit_win_frame, textvariable=self.xlower, width=8)
        self.xupper = tk.DoubleVar(value=x_lim[1])
        self.xupper_entry = tk.Entry(
            self.fit_win_frame, textvariable=self.xupper, width=8)
        self.xlower_entry.bind(
            '<Return>', lambda *args: self.set_fit_window(case='lower'))
        self.xupper_entry.bind(
            '<Return>', lambda *args: self.set_fit_window(case='upper'))
        self.win_step_lbl = tk.Label(self.fit_win_frame, text="Steps:")
        self.xstep = tk.DoubleVar(
            value=np.round((x_lim[1]-x_lim[0]) / 20))
        self.xstep_entry = tk.Entry(
            self.fit_win_frame, textvariable=self.xstep, width=8)
        self.xlower_entry.grid(
            row=1, column=1, sticky=tk.W, padx=2, pady=2)
        self.xupper_entry.grid(
            row=1, column=2, sticky=tk.W, padx=2, pady=2)
        self.win_step_lbl.grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        self.xstep_entry.grid(row=2, column=1, sticky=tk.W, padx=2, pady=2)

        self.add_datasets = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.fit_win_frame,
            text='Add to current data set',
            variable=self.add_datasets).grid(
                row=3, column=0, padx=2, pady=2, sticky=tk.W, columnspan=2)

        self.fit_function = tk.StringVar(value="Hermite Polyn.")
        self.fit_function_dict = {'Hermite Polyn.': 'hermitepoly',
                                  'Gaussian': 'gaussian',
                                  'Gaussian * Sine': 'gaussiansine'}
        tk.ttk.OptionMenu(
            self.model_frame, self.fit_function, self.fit_function.get(),
            *self.fit_function_dict.keys(),
            command=self.fit_func_select_callback).grid(
                row=0, column=0, columnspan=3, sticky='w')

        tk.ttk.Label(self.model_frame, text="No. of comp.:").grid(
            row=1, column=0, sticky='w')
        self.num_comp = tk.IntVar(value=5)
        tk.ttk.OptionMenu(self.model_frame, self.num_comp,
                          5, *list(range(1, 10))).grid(
                              row=1, column=1, sticky='w')
        self.num_sine_label = tk.Label(
            self.model_frame, text="No. of sine comp.:")
        self.num_sine_label.grid(row=1, column=2, sticky='w')
        self.num_sine_label.config(state='disabled')
        self.num_sine_comps = tk.IntVar(value=0)
        self.num_sine_select = tk.ttk.OptionMenu(
            self.model_frame, self.num_sine_comps, 0, *list(range(5)))
        self.num_sine_select.grid(row=1, column=3, sticky='w')
        self.num_sine_select.config(state='disabled')

        self.offset = tk.IntVar(value=0)
        tk.ttk.Checkbutton(self.model_frame, text='Offset',
                           variable=self.offset).grid(
                               row=2, column=0, sticky=tk.W)

        self.model_frame.grid(row=1, column=0, sticky='wnse', padx=2, pady=2)

        # fit option frame
        self.fit_opt_frame.grid(
            row=0, column=1, rowspan=2, sticky='wnse', padx=2, pady=2)

        tk.Label(self.fit_opt_frame, text="Algorithm:").grid(
            row=0, column=0, sticky=tk.W)
        self.fit_algo = tk.StringVar(value='leastsq')
        self.fit_algo_select = tk.ttk.OptionMenu(
            self.fit_opt_frame, self.fit_algo, self.fit_algo.get(), 'leastsq',
            'SLSQP')
        self.fit_algo_select.grid(row=0, column=1, sticky='we', columnspan=2)

        tk.Label(self.fit_opt_frame, text='Guesses &\nBounds:').grid(
            row=1, column=0, sticky=tk.W)
        self.enter_guesses_opt = tk.StringVar(value='Enter')
        tk.ttk.OptionMenu(
            self.fit_opt_frame, self.enter_guesses_opt,
            self.enter_guesses_opt.get(), 'Enter', 'Auto',
            command=self.enter_guess_opt_callback).grid(
                row=1, column=1, padx=2, pady=2, columnspan=2, sticky='we')

        # tzero subframe

        self.t0_auto_guess_opt = tk.IntVar(value=1)
        self.t0_auto_guess_opt_check = tk.ttk.Checkbutton(
            self.fit_opt_frame, variable=self.t0_auto_guess_opt,
            text="Time zero auto guess",
            command=self.t0_auto_guess_callback)
        self.t0_auto_guess_opt_check.grid(row=2, column=0, sticky='wn',
                                          padx=2, pady=2, columnspan=2)

        self.auto_t0_thresh = tk.DoubleVar(value=10)
        tk.Entry(
            self.t0_guess_frame,
            textvariable=self.auto_t0_thresh,
            width=3).grid(row=0, column=1, sticky='w')
        tk.Label(self.t0_guess_frame,
                 text='Detection\nthreshold:').grid(
                     row=0, column=0, sticky='wn')
        self.auto_t0_fit_thresh = tk.DoubleVar(value=2)
        tk.Entry(
            self.t0_guess_frame,
            textvariable=self.auto_t0_fit_thresh,
            width=3).grid(row=0, column=3, sticky='w')
        tk.Label(self.t0_guess_frame,
                 text='Outlier\nthresh.:').grid(
                     row=0, column=2, sticky='wn')
        ttk.Button(
            self.t0_guess_frame,
            text='Display',
            command=self.disp_t0_guess_button_callback).grid(
                row=1, column=0, columnspan=4)

        self.t0_guess_frame.grid(row=3, column=0, sticky='wnse',
                                 padx=2, pady=2, columnspan=2)
        self.t0_guess_frame_blank = tk.Frame(self.fit_opt_frame)
        self.t0_guess_frame_blank.grid(row=3, column=0, sticky='wnse',
                                       padx=2, pady=2, columnspan=2)
        self.t0_guess_frame.tkraise()

        self.recursive_guess = tk.IntVar(value=1)
        tk.ttk.Checkbutton(
            self.fit_opt_frame,
            text='Recursive Initials',
            variable=self.recursive_guess).grid(
                row=4, column=0, padx=2, pady=2, sticky=tk.W, columnspan=1)
        self.disp_para_after_fit = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.fit_opt_frame,
            text='Display parameters',
            variable=self.disp_para_after_fit).grid(
                row=4, column=1, padx=2, pady=2, sticky=tk.W, columnspan=1)

        button_frame = CustomFrame(
            self.fit_opt_frame, dim=(2, 1), border=False)

        self.run_fit_button = ttk.Button(
            button_frame, text='Run', command=self.run_fit)
        self.run_fit_button.grid(row=0, column=0, padx=5, pady=5)

        self.cancel_fit_button = tk.ttk.Button(
            button_frame, text='Cancel', command=self.cancel_fit)
        self.cancel_fit_button.config(state='disabled')
        self.cancel_fit_button.grid(row=0, column=1, padx=5, pady=5)

        button_frame.grid(row=5, column=0, sticky='wnse')

        self.sigma_default_guess = 0.1
        if self.data_obj.time_unit in self.data_obj.time_unit_factors.keys():
            self.irf_factor = 1e-3
            self.sigma_lbl = (
                'Sigma (' + self.data_obj.time_unit_factors[
                    self.data_obj.time_unit][1] + '):')
        else:
            self.irf_factor = 1
            self.sigma_lbl = "Sigma (" + self.data_obj.time_unit + "):"

        # results frame
        self.results_frame.grid(row=0, column=1, columnspan=1,
                                sticky='wnse', padx=5, pady=5)
        # trace plot frame
        self.plot_frame.grid(row=0, column=0, sticky='wnse', padx=2, pady=2)
        tk.Label(self.plot_frame,
                 text="Enter value or\nright click on map.").grid(
                     row=1, column=2, columnspan=2, sticky=tk.W)
        tk.Label(self.plot_frame, text='Show trace at (nm):').grid(
            row=1, column=0, sticky=tk.W)
        self.show_trace_at = tk.DoubleVar(value=self.xlower.get())
        self.show_trace_at_entry = tk.Entry(
            self.plot_frame, textvariable=self.show_trace_at, width=8)
        self.show_trace_at_entry.grid(row=1, column=1, sticky=tk.W)
        self.show_trace_at_entry.bind(
            '<Return>', self.show_trace_at_entry_callback)
        self.navi_frame.grid(row=2, column=0, sticky='nsw',
                             columnspan=2)
        tk.Label(self.navi_frame,
                 text="Navigate through traces").grid(
                     row=0, column=0, sticky='we', columnspan=2)

        left_arrow = {'children': [('Button.leftarrow', None)]}
        right_arrow = {'children': [('Button.rightarrow', None)]}
        style = ttk.Style(self)
        style.layout('L.TButton', [('Button.focus', left_arrow)])
        style.layout('R.TButton', [('Button.focus', right_arrow)])
        ttk.Button(
            self.navi_frame,
            style='L.TButton',
            command=lambda *args: self.navigate_trace(case='previous')).grid(
                row=1, column=0, sticky=tk.E)
        ttk.Button(
            self.navi_frame,
            style='R.TButton',
            command=lambda *args: self.navigate_trace(case='next')).grid(
                row=1, column=1, sticky=tk.W)
        self.show_components = tk.IntVar(value=1)
        self.show_comps_check = tk.ttk.Checkbutton(
            self.plot_frame,
            text='Plot Components',
            variable=self.show_components,
            command=self.show_single_fit)
        self.show_comps_check.grid(row=2, column=2, sticky='w')

        # parameter display
        tk.ttk.Button(self.fit_para_display,
                      text="Show Report",
                      command=self.show_para_for_trace).grid(
                          row=1, column=0, padx=10, pady=10, sticky='w')
        tk.ttk.Label(self.fit_para_display,
                     text="Multiple Parameters:").grid(
                         row=1, column=1, padx=10, pady=10, sticky='w')
        self.multipara_opt = tk.StringVar(value="Average")
        multipara_opt_select = tk.ttk.OptionMenu(
            self.fit_para_display, self.multipara_opt,
            self.multipara_opt.get(), "Average", "First Comp.",
            command=self.multipara_callback)
        multipara_opt_select.config(width=10)
        multipara_opt_select.grid(
            row=1, column=2, padx=10, pady=10, sticky='w')

        self.fit_para_display.grid(row=1, column=0, sticky='wnse',
                                   padx=2, pady=2)
        self.hermite_para_disp.grid(row=0, column=0, sticky='wnse',
                                    padx=2, pady=2, columnspan=3)
        # self.hermite_para_disp_blank.grid(row=0, column=0, sticky='wnse',
        #                                        padx=2, pady=2)
        tk.Label(self.hermite_para_disp, text=self.sigma_lbl).grid(
            row=0, column=0, sticky=tk.W, columnspan=3)
        self.sigma_display = tk.Label(self.hermite_para_disp, text="     ")
        self.sigma_display.grid(row=0, column=3, columnspan=2, sticky=tk.W)
        tk.Label(self.hermite_para_disp,
                 text='t0 (' + self.data_obj.time_unit + '):').grid(
                     row=1, column=0, sticky=tk.W, columnspan=3)
        self.t0_display = tk.Label(self.hermite_para_disp, text="     ")
        self.t0_display.grid(row=1, column=3, columnspan=2, sticky=tk.W)
        tk.Label(self.hermite_para_disp, text='Amplitude(s):').grid(
            row=2, column=0, sticky=tk.W, columnspan=3)
        self.poly_amp_display = []
        for i in range(5):
            self.poly_amp_display.append(
                tk.Label(self.hermite_para_disp, text=" ", width=5))
            self.poly_amp_display[i].grid(row=3, column=i, sticky=tk.W)

        # self.hermite_para_disp.tkraise()

        # save list frame
        self.savelist_frame.grid(row=1, column=1, columnspan=1,
                                 pady=(5, 2), padx=2, sticky='wnse')
        ttk.Button(self.savelist_frame,
                   text='Add current',
                   command=self.add_current_to_save).grid(
                       row=0, column=0, padx=1, pady=1)
        ttk.Button(self.savelist_frame,
                   text='Remove current',
                   command=self.remove_from_savelist).grid(
                       row=0, column=1, padx=1, pady=1)

        self.add_all_to_savelist_button = ttk.Button(
            self.savelist_frame, text='Add all',
            command=self.add_all_to_savelist)
        self.add_all_to_savelist_button.grid(row=1, column=0, padx=2, pady=2)

        ttk.Button(self.savelist_frame,
                   text='Remove all',
                   command=self.clear_savelist).grid(
                       row=1, column=1, padx=2, pady=2)

        ttk.Button(
            self.savelist_frame,
            text='Show list',
            command=self.show_savelist).grid(row=2, column=0, padx=2, pady=2)

        ttk.Button(
            self.savelist_frame, text='Save', command=self.save_para).grid(
                row=2, column=1, padx=2, pady=2)

        # fit parameter frame
        self.fit_para_frame.grid(row=0, column=1, sticky='wnse', padx=2,
                                 pady=2)
        tk.Label(self.fit_para_frame, text="Plot:").grid(
            row=0, column=0, sticky=tk.W)
        self.result_disp_mode = tk.StringVar(value='fit map')
        self.result_disp_select = tk.ttk.OptionMenu(
            self.fit_para_frame, self.result_disp_mode,
            self.result_disp_mode.get(), 'fit map', 'sigma', 'time zero',
            't0 in map',
            command=self.result_disp_select_callback)
        self.result_disp_select.config(width=10)
        self.result_disp_select.grid(row=0, column=1, sticky='we',
                                     columnspan=2)
        tk.Label(self.fit_para_frame, text='Movie:').grid(
            row=1, column=0, sticky=tk.W)
        self.movie_mode = tk.StringVar(value='lambda')
        movie_mode_select = tk.ttk.OptionMenu(
            self.fit_para_frame, self.movie_mode, self.movie_mode.get(),
            'lambda', 'time')
        movie_mode_select.config(width=10)
        movie_mode_select.grid(row=1, column=1, sticky='we')

        ttk.Button(self.fit_para_frame,
                   text="Show Movie",
                   command=self.show_movie).grid(row=2, column=1)

        self.widget_frame.grid(row=3, columnspan=3, sticky='wnse')

        # progressbar
        self.progressbar_frame = tk.Frame(
            self, highlightbackground="grey", highlightthickness=1, bd=0)
        self.progressbar_frame.grid(
            row=4, column=0, columnspan=4, sticky='we', pady=5, padx=5)
        self.progress_val = tk.DoubleVar(value=0)
        self.progressbar = ttk.Progressbar(self.progressbar_frame,
                                           orient=tk.HORIZONTAL,
                                           length=400,
                                           mode='determinate',
                                           variable=self.progress_val)
        self.pb_len = self.progressbar.cget("length")
        self.progressbar.grid(row=0, column=1, sticky=tk.W, padx=20)

        # misc. variables
        self.show_trace_number = 0
        self.show_fit = ""
        self.auto_t0 = []
        self.fit_done = False
        self.previous_guess_entries = None
        self.report_window = None
        self.savelist = []
        self.reports = {}
        center_toplevel(self, controller)

    # UI Callbacks
    def set_fit_window(self, *args, case='lower'):
        low = self.xlower.get()
        up = self.xupper.get()
        if case == 'lower' or case == 'both':
            if low >= up and self.window_mode.get() == 'Range':
                low = up - 1
            if low < np.min(self.dat.x):
                low = np.min(self.dat.x)
            self.xlower.set(low)
        if case == 'upper' or case == 'both':
            if low >= up and self.window_mode.get() == 'Range':
                up = low + 1
            if up > np.max(self.dat.x):
                up = np.max(self.dat.x)
            self.xupper.set(up)
        xlim_curr = self.main_figure.get_xlim()
        xlim_new = None
        if low < min(xlim_curr) or up > max(xlim_curr):
            xlim_new = [min([low, *xlim_curr]), max([up, *xlim_curr])]
            self.main_figure.set_xlim(xlim_new)

    def fit_func_select_callback(self, *args):
        if re.search('sine', self.fit_function.get(), re.I):
            self.num_sine_label.config(state='normal')
            self.num_sine_select.config(state='normal')
            self.num_sine_comps.set(1)
            self.num_comp.set(1)
        else:
            self.num_sine_label.config(state='disabled')
            self.num_sine_select.config(state='disabled')
            self.num_sine_comps.set(0)
            if re.search('hermit', self.fit_function.get(), re.I):
                self.num_comp.set(5)
            else:
                self.num_comp.set(2)
        self.previous_guess_entries = None

    def enter_guess_opt_callback(self, *args):
        if self.enter_guesses_opt.get().lower() == 'auto':
            self.t0_auto_guess_opt.set(1)
            self.t0_auto_guess_opt_check.config(state='disabled')
            self.t0_auto_guess_callback()
        else:
            self.t0_auto_guess_opt_check.config(state='normal')

    def t0_auto_guess_callback(self, *args):
        if self.t0_auto_guess_opt.get():
            self.t0_guess_frame.tkraise()
        else:
            self.t0_guess_frame_blank.tkraise()

    def show_savelist(self):
        self._show_fit_para(self.savelist, title="Save List")

    def _show_fit_para(self, keys, title="Fit Results"):
        entry_dict = {}
        sigma = self.get_sigma(keys)
        t0 = self.get_t0(keys)
        for i, key in enumerate(keys):
            entry_dict[key] = [sigma[i]/self.irf_factor]
            entry_dict[key].append(t0[i])
        FitResultsDisplay(self, entry_dict,
                          controller=self,
                          headers=['wavelength', 'S' + self.sigma_lbl[1:],
                                   'Time zero'],
                          title=title)

    def show_para_for_trace(self, open_new=True):
        paras = self.traces.tr[self.show_fit]['fit_paras']
        headers = {
            "curve": (self.data_obj.get_x_mode()
                      + " (" + self.data_obj.spec_unit + ")"),
            "comp": "Parameter",
            "x": 'Time delay (' + self.data_obj.time_unit + ')',
            "y": 'Amplitude'}
        report_dict = {self.show_fit: lmfitreport_to_dict(
            self.reports[self.show_fit])}
        try:
            self.report_window.setup_table(
                curve_labels=[self.show_fit], params=paras)
        except Exception:
            if open_new:
                self.report_window = FitParaEntryWindowLmfit(
                    self, [self.show_fit],
                    headers=headers,
                    mode='display',
                    params=paras,
                    show_table=False)
                self.report_window.fr.set_axis_assignment(
                    'sigma', 'x', params=paras)
                self.report_window.report_disp = MultiDisplay(
                    self.report_window,
                    input_dict=report_dict,
                    mode='expandable',
                    orient='horizontal')
                self.report_window.report_disp.grid(
                    row=1, column=0, sticky='wnse')
            else:
                return
        else:
            self.report_window.report_disp.add_frames(report_dict)
            self.report_window.report_disp.select_frame(frame=self.show_fit)

    def add_current_to_save(self):
        if self.show_fit not in self.savelist:
            self.savelist.append(self.show_fit)

    def remove_from_savelist(self):
        try:
            self.savelist.remove(self.show_fit)
        except Exception:
            return

    def add_all_to_savelist(self):
        self.savelist = []
        for key in self.traces.active_traces:
            self.savelist.append(key)

    def clear_savelist(self):
        self.savelist = []

    def save_para(self):
        window = MultipleOptionsWindow(self, self,
                                       ['time zero', 'sigma', 'fit curves'],
                                       text='Choose variables to save',
                                       check_values=[1, 0, 0, 0],
                                       padx=5, pady=5)
        save_dict = {'time zero': self.save_t0,
                     'sigma': self.save_sigma,
                     'fit curves': self.save_fit}
        if 'all' in window.output:
            self.save_all_info()
        else:
            for o in window.output:
                save_dict[o]()

    def save_all_info(self):
        file = save_box(filetypes=[('Matlab files', '.mat')], fext='.mat',
                        parent=self)
        if file is not None:
            try:
                self.traces.save_traces_mat(file.name, save_fit=True,
                                            trace_keys=self.savelist,
                                            trace_type='CPM',
                                            custom_para={'PolynomialOrder': 5})
            except Exception as e:
                messagebox.showerror(message=e, parent=self)

    def save_t0(self):
        file = save_box(parent=self)
        try:
            file.name
        except Exception:
            pass
        else:
            self.data_obj.time_zeros = np.zeros((len(self.savelist), 2))
            self.data_obj.time_zero_fit = []
            t0 = self.get_t0(keys=self.savelist)
            for i, k in enumerate(self.savelist):
                self.data_obj.time_zeros[i, 0] = self.traces.tr[k]['val']
                self.data_obj.time_zeros[i, 1] = t0[i]
            self.data_obj.save_time_zero_file(file)

    def save_sigma(self):
        file = save_box(parent=self)
        try:
            file.name
        except Exception:
            pass
        else:
            save_data = np.zeros((len(self.savelist), 2))
            sigma = self.get_sigma(keys=self.savelist)
            for i, k in enumerate(self.savelist):
                save_data[i, 0] = self.traces.tr[k]['val']
                save_data[i, 1] = sigma[i]
            self.data_obj.save_plot_data(file, save_data)

    def save_fit(self):
        if len(self.savelist) > 0:
            file = save_box(filetypes=[('text files', '.txt'),
                                       ('Matlab files', '.mat')],
                            fext='.txt', parent=self)
            if file is not None:
                self.traces.xlabel = self.trace_figure.get_xlabel()
                self.traces.save_traces(file, save_fit=True,
                                        trace_keys=self.savelist,
                                        trace_type='CPM',
                                        custom_para={'PolynomialOrder': 5})
        else:
            messagebox.showerror(message="Please select traces to save.",
                                 parent=self)

    def main_plot_click_callback(self, *args):
        if args[0].button == 3 and self.fit_done:
            if self.main_figure.transpose[0]:
                point = args[0].ydata
            else:
                point = args[0].xdata
            self.get_nearest_trace(point)
            self.show_single_fit()

    def get_nearest_trace(self, val):
        self.show_fit, self.show_trace_number = (
            self.traces.get_nearest_trace(val))

    def plot_trace(self, *args, fig=None, ax=None, **kwargs):
        if fig is None:
            fig = self.trace_figure.figure
        if ax is None:
            ax = self.trace_figure.axes[0]
        self.traces.plot(self, *args, fig=fig, ax=ax,
                         active_traces=[self.show_fit], **kwargs)
#        return None, None

    def fit_figure_callback(self, event):
        if self.fit_done and (event.button == 3 or event.dblclick):
            if re.search('map', self.result_disp_mode.get(), re.I):
                plot_type = '2D'
            else:
                plot_type = 'linefit'
            figure = self.controller.open_topfigure_wrap(
                self, controller=self, fig_obj=self.fit_figure,
                plot_type=plot_type, editable=True)
            self.plot_func_dict[self.result_disp_mode.get().lower()](
                fig_obj=figure.fr.figure)

    def open_trace_figure(self, event):
        if self.fit_done and (event.button == 3 or event.dblclick):
            self.controller.open_topfigure_wrap(
                self, plot_func=self.plot_trace, controller=self,
                fig_obj=self.trace_figure, plot_type='linefit',
                editable=True)

    def result_disp_select_callback(self, *args):
        if self.fit_done:
            self.plot_func_dict[self.result_disp_mode.get().lower()]()

    def show_trace_at_entry_callback(self, *args):
        if self.fit_done:
            self.get_nearest_trace(self.show_trace_at.get())
            self.show_single_fit()

    def navigate_trace(self, *args, case='next'):
        try:
            self.show_fit = self.traces.active_traces[
                self.show_trace_number - 1 + 2*int(case == 'next')]
        except Exception:
            return
        else:
            self.show_trace_number += 2*int(case == 'next') - 1
            self.show_single_fit()

    def show_movie(self, *args):
        if self.movie_mode.get() == 'lambda':
            PlotMovie(self, controller=self,
                      data=self.traces,
                      xlimits=self.main_figure.get_ylim(), fit=True,
                      ylimits=self.main_figure.get_clim(),
                      ylabels=self.main_figure.get_clabel(),
                      xlabels=self.main_figure.get_ylabel(),
                      frame_interval=int(2000/len(self.xvalues)),
                      movie_iterable=list(
                          range(len(self.traces.active_traces))),
                      frame_labels=self.traces.active_traces)
        else:
            dat = [self.data_obj.spec_axis[
                self.data_obj.get_x_mode()], np.array([self.data_obj.delA])]
            fit_map = np.array(
                [np.transpose([self.traces.tr[key]['fit']
                               for key in self.traces.active_traces])])
            fit = [self.xvalues, fit_map]
            tlabels = [str(td) + ' (' + self.data_obj.time_unit +
                       ')' for td in self.traces.xdata]
            PlotMovie(self, controller=self, data=dat, fit=fit,
                      xlabel=self.main_figure.get_xlabel(),
                      ylabel=self.main_figure.get_clabel(),
                      xlimits=self.main_figure.get_xlim(),
                      ylimits=self.main_figure.get_clim(),
                      frame_interval=int(10000/len(self.traces.xdata)),
                      frame_labels=tlabels)

    def get_t0_fit(self):
        self.auto_t0s = self.data_obj.find_time_zeros(
            irf_threshold=self.auto_t0_thresh.get(),
            algorithm='derivative', write_to_obj=False)
        try:
            outliers, self.auto_t0 = self.data_obj.fit_time_zeros(
                polyorder=5, time_zeros=self.auto_t0s,
                write_to_obj=False, ind=[0, len(self.auto_t0s)-1],
                outlier_thresh_rel=10)
        except Exception:
            raise
            self.auto_t0 = np.ones(
                len(self.data_obj.wavelengths))*np.round(
                    (self.data_obj.time_delays[-1]
                     + self.data_obj.time_delays[0])/2, 3)
            outliers = []
        return outliers

    def disp_t0_guess_button_callback(self):
        outliers = self.get_t0_fit()
        self.trace_figure.axes[0].cla()
        self.trace_figure.axes[0].plot(
            self.auto_t0s[:, 0], self.auto_t0s[:, 1])
        try:
            self.trace_figure.axes[0].plot(outliers[0], outliers[1], 'h')
        except Exception:
            self.trace_figure.axes[0].plot(
                self.data_obj.wavelengths, self.auto_t0,
                color=self.controller.settings['fit_color'])
            self.trace_figure.axes[0].legend(
                ('time zeros found', 'fit for guess'))
        else:
            self.trace_figure.axes[0].plot(
                self.data_obj.wavelengths, self.auto_t0,
                color=self.controller.settings['fit_color'])
            self.trace_figure.axes[0].legend(
                ('time zeros found', 'outliers', 'fit for guess'))

        self.trace_figure.axes[0].set_xlim(self.main_figure.get_xlim())
        self.trace_figure.axes[0].set_ylim(self.main_figure.get_ylim()[::-1])
        self.trace_figure.axes[0].set_xlabel(self.main_figure.get_xlabel())
        self.trace_figure.axes[0].set_ylabel(self.main_figure.get_ylabel())
        self.trace_figure.canvas.draw()

    def multipara_callback(self, *args):
        if self.fit_done:
            self.show_single_fit()
            self.plot_func_dict[self.result_disp_mode.get().lower()]()

    def find_fit_indices(self):
        fit_points = []
        if self.window_mode.get() == 'Range':
            fit_inds = []
            if self.xstep.get() > 0:
                wl = self.xlower.get()
                upper = self.xupper.get()
            else:
                wl = self.xupper.get()
                upper = self.xlower.get()
            while wl <= upper:
                fit_inds.append(
                    np.where(self.data_obj.wavelengths >= wl)[0][0])
                fit_points.append(
                    str(np.round(
                        self.data_obj.wavelengths[fit_inds[-1]], 1)) + ' nm')
                wl = wl + self.xstep.get()
        elif self.window_mode.get() == 'Single':
            fit_inds = [
                np.where(self.data_obj.wavelengths
                         >= self.xlower.get())[0][0]]
            fit_points.append(
                str(np.round(
                    self.data_obj.wavelengths[fit_inds[0]], 1)) + ' nm')
        return fit_inds, fit_points

    def get_default_bounds(self, max_amp=20, min_amp=-20):
        t_window = np.abs(
            self.data_obj.time_delays[-1]-self.data_obj.time_delays[0])
        t0_default_bounds = [min(self.data_obj.time_delays) - t_window,
                             max(self.data_obj.time_delays) + t_window]
        sigma_default_bounds = [0, np.inf]
        poly_amp_bds = [10*min_amp] if min_amp < 0 else [0]
        poly_amp_bds.append(10*max_amp if max_amp > 0 else 0)
        return t0_default_bounds, sigma_default_bounds, poly_amp_bds

    def run_fit(self):
        def start_thread(*args, **kwargs):
            self._fit_running = True
            self.queue = Queue()
            self.task = ThreadedTask(self.traces.run_cpm_fit, *args,
                                     after_finished_func=self._after_fit,
                                     interruptible=True, queue=self.queue,
                                     **kwargs)
            self.task.start()
            self.listener = threading.Thread(
                target=self._update_progressbar, args=(self.queue,))
            self.listener.start()
            self.queue.join()
        self.progress_val.set(0)
        self.set_fit_window(case='both')
        try:
            self.report_window.destroy()
        except Exception:
            pass
        self.report_window = None
        # prepare variables
        if not self.add_datasets.get():
            if self.fit_done:
                del self.traces
            self.fit_done = False
        if not self.fit_done:
            self.traces = TATrace(xdata=self.data_obj.time_delays,
                                  xunit=self.data_obj.time_unit)
        # get indices for traces to fit
        fit_inds, fit_points = self.find_fit_indices()
        self.traces.active_traces = []
        max_y = []
        min_y = []
        for i, index in enumerate(fit_inds):
            self.traces.tr[fit_points[i]] = {
                'y': self.data_obj.delA[:, index],
                'val': self.data_obj.wavelengths[index]}
            self.traces.active_traces.append(fit_points[i])
            max_y.append(max(self.data_obj.delA[:, index]))
            min_y.append(min(self.data_obj.delA[:, index]))
        max_y = max(max_y)
        min_y = min(min_y)

        # set default parameter values
        general_guesses = {}
        bounds = {}

        # time zero guess
        if self.t0_auto_guess_opt.get():
            guess_window_exclude = ['x0_1']
            t0_guesses = {}
            if len(self.auto_t0) == 0:
                self.get_t0_fit()
            if re.search('hermit',
                         self.fit_function_dict[self.fit_function.get()],
                         re.I):
                for i, key in enumerate(self.traces.active_traces):
                    t0_guesses[key] = {
                        'x0_1': self.auto_t0[fit_inds][i]}
            else:
                for i, key in enumerate(self.traces.active_traces):
                    t0_guesses[key] = {}
                    for j in range(self.num_comp.get()):
                        t0_guesses[key]['gauss_x0_' + str(j + 1)] = (
                            self.auto_t0[fit_inds][i])
                        guess_window_exclude.append('gauss_x0_' + str(j + 1))
        else:
            t0_default = np.round(
                (self.data_obj.time_delays[-1]
                 + self.data_obj.time_delays[0])/2, 3)
            if re.search('hermit',
                         self.fit_function_dict[self.fit_function.get()],
                         re.I):
                general_guesses['x0_1'] = t0_default
            else:
                for j in range(self.num_comp.get()):
                    general_guesses['gauss_x0_' + str(j + 1)] = t0_default
            t0_guesses = None
            guess_window_exclude = None

        # misc. default guesses/bounds
        if self.previous_guess_entries is None:
            t0_default_bounds, sigma_default_bounds, poly_amp_bds = (
                self.get_default_bounds(max_amp=max_y, min_amp=min_y))

            if re.search('hermit',
                         self.fit_function_dict[self.fit_function.get()],
                         re.I):
                general_guesses['sigma_1'] = self.sigma_default_guess
                bounds['sigma_1'] = sigma_default_bounds
                bounds['x0_1'] = t0_default_bounds
                for i in range(self.num_comp.get()):
                    bounds['hermite_amp_' + str(i + 1)] = poly_amp_bds
            else:
                if re.search('sine',
                             self.fit_function_dict[self.fit_function.get()],
                             re.I):
                    for i in range(self.num_comp.get()):
                        general_guesses['osc_omega_' + str(i + 1)] = (
                            2 / self.sigma_default_guess)
                for i in range(self.num_comp.get()):
                    general_guesses['gauss_sigma_' +
                                    str(i + 1)] = self.sigma_default_guess
                    bounds['gauss_sigma_' + str(i + 1)] = sigma_default_bounds
                    bounds['gauss_x0_' + str(i + 1)] = t0_default_bounds
        else:
            for key, val in self.previous_guess_entries['inits'].items():
                general_guesses[key] = val
            for key, val in self.previous_guess_entries['bounds'].items():
                bounds[key] = val

        # run fit and process results

        # init fit object and set default inits, bounds
        self.fit_obj = self.traces.init_cpm_fit(
            func=self.fit_function_dict[self.fit_function.get()],
            method=self.fit_algo.get(),
            guesses=general_guesses,
            offset=bool(self.offset.get()),
            bounds=bounds,
            n=self.num_comp.get(),
            modul_n=self.num_sine_comps.get())
        # get parameter input if selected
        if self.enter_guesses_opt.get() == 'Enter':
            win = FitParaOptsWindow(self, self.fit_obj.params,
                                    controller=self.controller,
                                    exclude_para=guess_window_exclude)
            self.wait_window(win)
            if win.output is None:
                return
            self.fit_obj.params = win.write_parameters(self.fit_obj.params)
            self.previous_guess_entries = {'inits': {}, 'bounds': {}}
            for para in self.fit_obj.params:
                p = self.fit_obj.params[para]
                self.previous_guess_entries['inits'][para] = p.value
                self.previous_guess_entries['bounds'][para] = [p.min, p.max]
        self.cancel_fit_button.config(state='normal')
        self.run_fit_button.config(state='disabled')
        # run fit
        start_thread(fit_obj=self.fit_obj,
                     inits=t0_guesses, calculate_components=True,
                     recursive_guess=self.recursive_guess.get())

    def _post_fit_ops(self):
        self._enable_after_fit()
        self._fit_running = False
        if self.window_mode.get() == 'Single':
            self.show_fit = self.traces.active_traces[0]
        self.traces.active_traces, self.xvalues = self.traces.sort_by_value()
        if not self.window_mode.get() == 'Single':
            self.show_fit = self.traces.active_traces[0]
        self.show_trace_number = self.traces.active_traces.index(self.show_fit)
        self.show_single_fit()
        self.plot_func_dict[self.result_disp_mode.get().lower()]()
        self.progress_val.set(self.progressbar.cget("length"))
        self.progressbar.update_idletasks()
        self.update()
        self.auto_t0 = []
        self.fit_done = True
        if self.disp_para_after_fit.get():
            self._show_fit_para(self.traces.active_traces)

    def _after_fit(self):
        self._fit_running = False
        if self.task.output is not None:
            self.fit_obj, paras, prev_inits, fitreports, self.success = (
                self.task.output)
            for key, val in fitreports.items():
                self.reports[key] = val
            self.after(100, self._post_fit_ops)
        else:
            self.after(100, self._enable_after_fit)

    def _enable_after_fit(self):
        self.cancel_fit_button.config(state='disabled')
        self.run_fit_button.config(state='normal')

    def _update_progressbar(self, in_queue):
        while self._fit_running:
            q = in_queue.get()
            self.progress_val.set(q['point']/q['n']*self.pb_len)
            self.progressbar.update_idletasks()

    def cancel_fit(self):
        try:
            self.task.raise_exception()
        except Exception as e:
            messagebox.showerror(message=e, parent=self)

    def get_t0(self, keys=None):
        return self._get_parameter(keys=keys, case='x0')

    def get_sigma(self, keys=None):
        return self._get_parameter(keys=keys, case='sigma')

    def _get_parameter(self, keys=None, case='x0'):
        if keys is None:
            keys = self.traces.active_traces
        para_list = []
        if self.multipara_opt.get().lower() == "average":
            for k in keys:
                paras = []
                for para in self.traces.tr[k]['fit_paras'].keys():
                    if re.search(case, para, re.I):
                        paras.append(self.traces.tr[k]['fit_paras'][para])
                para_list.append(np.mean(paras))
        else:
            for k in keys:
                for para in self.traces.tr[k]['fit_paras'].keys():
                    if re.search(case, para, re.I):
                        para_list.append(
                            self.traces.tr[k]['fit_paras'][para].value)
                        break
        return para_list

    def plot_t0_map(self, *args, marker='x', markersize=7,
                    markercolor='black', fig_obj=None):
        if fig_obj is None:
            fig_obj = self.fit_figure
        self.plot_fit_map(update_canvas=False, fig_obj=fig_obj)
        t0 = self.get_t0()
        plot(fig_obj.axes[0], self.xvalues, t0, marker=marker,
             markersize=markersize, color=markercolor,
             transpose=self.controller.settings['2Dmap_transpose'])
        plot(fig_obj.axes[0], self.xvalues[self.show_trace_number],
             t0[self.show_trace_number], marker=marker,
             markersize=markersize*1.5, color='orange',
             mew=1.5, transpose=self.controller.settings['2Dmap_transpose'])
        fig_obj.set_legend(["Time Zero"], visible=True, update_canvas=True)

    def plot_sigma(self, fig_obj=None):
        if fig_obj is None:
            fig_obj = self.fit_figure
        sigma = self.get_sigma()
        fig_obj.plot_function = lambda *args, **kwargs: plot(
            fig_obj.axes[0], self.xvalues, sigma, ".")
        fig_obj.ylabels = [self.sigma_lbl[:-1]]
        fig_obj.plot_type = ['line']
        fig_obj.set_legend(["Sigma"], visible=True, update_canvas=False)
        fig_obj.axes[0].cla()
        fig_obj.plot(ylimits=False, xlimits=False)

    def plot_t0(self, fig_obj=None):
        if fig_obj is None:
            fig_obj = self.fit_figure
        t0 = self.get_t0()
        fig_obj.plot_function = (
            lambda *args, ax=None, fig=None, **kwargs: plot(
                fig_obj.axes[0], self.xvalues, t0, ".", **kwargs))
        fig_obj.set_for_all_axes(
            fig_obj.set_transpose,
            self.controller.settings['2Dmap_transpose'])
        fig_obj.ylabels = self.main_figure.ylabels
        fig_obj.plot_type = ['line']
        fig_obj.set_legend(["Time Zero"], visible=True, update_canvas=False)
        fig_obj.axes[0].cla()
        if self.controller.settings['2Dmap_transpose']:
            fig_obj.plot(ylimits=False, update_canvas=False)
        else:
            fig_obj.plot(ylimits=False, update_canvas=False)
        fig_obj.set_invert_yaxis(self.main_figure.invert_yaxis[0], i=0)

    def plot_fit_map(self, fig_obj=None, **plot_kwargs):
        if fig_obj is None:
            fig_obj = self.fit_figure
        fit_map = np.vstack(
            [self.traces.tr[key]['fit']
             for key in self.traces.active_traces]).T
        if len(self.xvalues) > 1:
            x = np.concatenate((self.xvalues
                                - np.abs(self.xvalues[1] - self.xvalues[0])/2,
                                [self.xvalues[-1]
                                 + np.abs(self.xvalues[1] - self.xvalues[0])]))
        else:
            x = np.array([self.xvalues[0]-0.5, self.xvalues[0]+0.5])
        y = self.traces.xdata
        try:
            plotmap = np.concatenate((fit_map, np.zeros((1, len(y)))))
        except ValueError:
            plotmap = np.concatenate(
                (np.transpose(fit_map), np.zeros((1, len(y)))))
        except Exception:
            raise
        fig_obj.plot_function = (
            lambda *args, ax=None, fig=None,
            **kwargs: pcolor(fig_obj.axes[0], x,
                             y - np.abs(y[1]-y[0])/2,
                             np.transpose(plotmap),
                             **fig_obj.color_obj[0].get_kwargs(),
                             **kwargs))
        fig_obj.plot_type = ['2D']
        fig_obj.set_for_all_axes(
            fig_obj.set_transpose,
            self.controller.settings['2Dmap_transpose'])
        fig_obj.set_legend(visible=False, update_canvas=False)
        fig_obj.ylabels = self.main_figure.ylabels
        fig_obj.xlimits = self.main_figure.xlimits
        fig_obj.ylimits = fig_obj.ylimits
        fig_obj.axes[0].cla()
        fig_obj.plot(**plot_kwargs)

    def show_single_fit(self, *args):
        self.show_trace_at.set(np.round(
            self.xvalues[self.show_trace_number], 1))
        plot_kwargs = {'include_fit': True,
                       'fill': 0.1, 'include_fit_comps': False}
        legend_entries = [self.show_fit]
        if self.show_components.get():
            leg_dict = {'gauss': 'Gaussian ',
                        'hermite': 'Comp. ',
                        'osc': 'Modulation ',
                        'const': 'Offset'}
            legend_entries.append('Fit')
            plot_kwargs['include_fit_comps'] = True
            plot_kwargs['show_fit_legend'] = True
            for key in self.traces.tr[self.show_fit]['fit_comps'].keys():
                key_split = key.split("_")
                try:
                    entry = leg_dict[key_split[0]]
                except Exception:
                    entry = key
                else:
                    try:
                        entry += key_split[1]
                    except Exception:
                        pass
                legend_entries.append(entry)

        self.trace_figure.set_plot_kwargs(**plot_kwargs)
        self.trace_figure.set_legend(entries=legend_entries,
                                     update_canvas=False)
        self.trace_figure.plot()
        if self.result_disp_mode.get().lower() == 't0 in map':
            self.plot_t0_map()
        self._disp_current_para()
        if self.report_window:
            self.show_para_for_trace(open_new=False)

    def _disp_current_para(self):
        paras = self.traces.tr[self.show_fit]['fit_paras']
        for disp in self.poly_amp_display:
            disp.grid_forget()
        self.poly_amp_display = []
        sigma = self.get_sigma(keys=[self.show_fit])[0]
        self.sigma_display.config(text=np.round(
            sigma/self.irf_factor, 2 * int(self.irf_factor == 1)))
        t0 = self.get_t0(keys=[self.show_fit])[0]
        self.t0_display.config(text=np.round(t0, 3))
        for name, para in paras.items():
            if re.search('amp', name):
                i = int(re.findall('\d+', name)[-1]) - 1
                while i > len(self.poly_amp_display) - 1:
                    self.poly_amp_display.append(
                        tk.Label(self.hermite_para_disp, text=" ",
                                 width=5))
                    self.poly_amp_display[-1].grid(
                        row=3, column=i, sticky=tk.W)
                self.poly_amp_display[i].config(text=np.round(para, 2))

    def window_mode_callback(self, *args):
        if self.window_mode.get() == 'Range':
            self.win_step_lbl.config(text="Steps:")
            xlim = self.data_obj.get_xlim()
            self.xlower.set(xlim[0])
            self.xupper.set(xlim[1])
            self.xstep.set(
                np.round((xlim()[1] - xlim()[0])/20))
            self.xupper_entry.config(state='normal')
            self.xstep_entry.config(state='normal')
            self.enter_guesses_opt.set('Enter')
            self.add_datasets.set(0)
        else:
            self.win_step_lbl.config(text="  ")
            self.xlower.set(np.round(np.mean(self.data_obj.get_xlim())))
            self.xupper.set(0)
            self.xstep.set(0)
            self.xupper_entry.config(state='disabled')
            self.xstep_entry.config(state='disabled')
            self.enter_guesses_opt.set('Enter')
            self.add_datasets.set(1)


# %%
class DynamicLineShapeFit(tk.Toplevel):
    def __init__(self, parent, controller, dat, data_obj, dim=None):
        tk.Toplevel.__init__(self, parent)
        # variables
        if dim is None:
            dim = controller.settings['figure_std_size']
        self.title("Dynamic Line Shape Analysis")
        self.controller = controller
        move_toplevel_to_default(self, controller)
        self.parent = parent
        self.data_obj = data_obj
        self.frames = {}
        self.buttons = {}
        self.entries = {}
        self.optmenus = {}
        self.checks = {}

        # Frames
        # self.frames['mainToolbar'] = tk.Frame(self)
        # self.frames['traceToolbar'] = tk.Frame(self)

        # main figure
        self.main_figure = self.controller.tk_mpl_figure_wrap(
            self, dim=dim, plot_function=dat.plot_2d, xlabels=dat.xlabel,
            ylabels=dat.ylabel, clabels=dat.clabel, xlimits=dat.xlims,
            ylimits=dat.ylims, color_obj=self.data_obj.color)

        self.main_figure.plot()
        self.main_figure.grid(row=1, column=0, padx=5, pady=5, sticky='wnse')

        fit_kw = {}
        for key, val in self.controller.settings['fit_kwargs'].items():
            fit_kw[key] = val
        fit_kw['linewidth'] = 2.0
        plot_kw = {'marker': 'o', 'fill': 0.2, 'color': "#BBBBBB",
                   'markerfacecolor': "#BBBBBB", 'markeredgecolor': "#888888",
                   'markersize': 5.0}
        self.trace_figure = TkMplFigure(
            self, dim=dim, xlabels=[dat.xlabel], plot_type='linefit',
            ylabels=[dat.clabel], xlimits=[dat.xlims], ylimits=None,
            legends=False,
            fit_kwargs=fit_kw, callbacks={
                'button_press_event': self.trace_fig_callback},
            **plot_kw)
        self.trace_figure.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

        # LS Fit frame
        self.fit_frame = CustomFrame(self, border=False)
        self.fit_frame.grid(row=3, column=0, sticky='wnse', pady=5, padx=5)
        #
        self.win_opts = GroupBox(self.fit_frame, text="Window")
        self.win_opts.grid(
            row=0, column=0, columnspan=2, padx=2, pady=2, sticky='nwse')

        tk.Label(self.win_opts, text="spec. range:").grid(
            row=0, column=0, sticky=tk.W)
        self.xlower = tk.DoubleVar(value=np.round(dat.xlims[0]))
        self.entries['xlower'] = tk.Entry(
            self.win_opts, textvariable=self.xlower, width=10)
        self.entries['xlower'].grid(row=0, column=1, sticky='we', padx=2)
        self.xupper = tk.DoubleVar(value=np.round(dat.xlims[1]))
        self.entries['xupper'] = tk.Entry(
            self.win_opts, textvariable=self.xupper, width=10)
        self.entries['xupper'].grid(row=0, column=2, sticky='we', padx=2)
        self.entries['xlower'].bind(
            "<Return>", lambda *args, c='lower': self.xwin_callback(
                *args, case=c))
        self.entries['xupper'].bind(
            "<Return>", lambda *args, c='upper': self.xwin_callback(
                *args, case=c))

        tk.Label(self.win_opts, text="t range:").grid(
            row=1, column=0, sticky=tk.W)
        self.tlower = tk.DoubleVar()
        try:
            self.tlower.set(self.parent.vars['tlower'].get())
        except Exception:
            self.tlower.set(
                0.0 if self.data_obj.time_delays[0] < 0
                else self.data_obj.time_delays[0])
        self.entries['tlower'] = tk.Entry(
            self.win_opts, textvariable=self.tlower, width=10)
        self.entries['tlower'].grid(row=1, column=1, sticky='we', padx=2)
        self.entries['tlower'].bind(
            "<Return>", lambda *args, c='lower': self.time_win_callback(
                *args, case=c))
        self.tupper = tk.DoubleVar()
        try:
            self.tupper.set(self.parent.vars['tupper'].get())
        except Exception:
            self.tupper.set(self.data_obj.time_delays[-1])
        self.entries['tupper'] = tk.Entry(
            self.win_opts, textvariable=self.tupper, width=10)
        self.entries['tupper'].grid(row=1, column=2, sticky='we', padx=2)
        self.entries['tupper'].bind(
            "<Return>", lambda *args, c='upper': self.time_win_callback(
                *args, case=c))
        self.tstep_lbl = tk.Label(
            self.win_opts,
            text="time steps (" + self.data_obj.time_unit + "):")
        self.tstep_lbl.grid(row=2, column=0, sticky=tk.W)
        try:
            self.tstep_lbl.config(
                text=("time steps ("
                      + self.data_obj.time_unit_factors[
                          self.data_obj.time_unit][1]
                      + "):"))
        except Exception:
            self.tstep_factor = 1
        else:
            self.tstep_factor = 1e3
        self.timesteps = tk.DoubleVar(value=np.round(
            np.abs(self.data_obj.time_delays[1]
                   - self.data_obj.time_delays[0])
            * self.tstep_factor))
        self.entries['tsteps'] = tk.Entry(
            self.win_opts, textvariable=self.timesteps, width=10)
        self.entries['tsteps'].grid(row=2, column=1, sticky='we', padx=2)
        self.reverse = tk.IntVar(value=1)
        tk.ttk.Checkbutton(
            self.win_opts, text="reverse\ntime", variable=self.reverse).grid(
                row=2, column=2, sticky=tk.W, padx=2)
        #
        self.model_opts = GroupBox(self.fit_frame, text="Model", dim=(2, 3))
        self.model_opts.grid(row=0, column=2, padx=2, pady=2, sticky='nwse')

        tk.Label(self.model_opts, text="Line shape:").grid(
            row=0, column=0, sticky='w')
        self.lineshape = tk.StringVar(value="Gaussian")
        tk.ttk.OptionMenu(
            self.model_opts, self.lineshape, self.lineshape.get(),
            "Gaussian", "Lorentzian", "Voigt").grid(
                row=0, column=1, sticky='we')

        tk.Label(self.model_opts, text="Components:").grid(
            row=1, column=0, sticky='nw')
        self.num_comp = tk.IntVar(value=2)
        tk.ttk.OptionMenu(
            self.model_opts, self.num_comp, 2, 1, 2, 3, 4, 5).grid(
                row=1, column=1, sticky='we')
        tk.Label(self.model_opts, text="Line width model:").grid(
            row=2, column=0, sticky='nw')
        self.linewidth_model = tk.StringVar(value='unconstrained')
        tk.ttk.OptionMenu(
            self.model_opts, self.linewidth_model, 'unconstrained',
            'unconstrained', 'linear').grid(row=2, column=1, sticky='nw')

        self.equal_space = tk.IntVar(value=0)
        self.checks['equal_space'] = tk.ttk.Checkbutton(
            self.model_opts, text="Equal Spacing", variable=self.equal_space)
        self.checks['equal_space'].grid(row=3, column=0, sticky='nw')
        self.offset_opt = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.model_opts, text="Offset", variable=self.offset_opt).grid(
                row=3, column=1, sticky='nw')

        self.run_fit_frame = GroupBox(self.fit_frame, text="Fit", dim=(2, 4))
        self.run_fit_frame.grid(row=1, column=1, columnspan=2,
                                padx=2, pady=2, sticky='nwse')
        tk.Label(self.run_fit_frame, text="Guesses:").grid(
            row=0, column=0, sticky='nw')

        self.enter_guesses = tk.IntVar(value=1)
        tk.ttk.Checkbutton(
            self.run_fit_frame, variable=self.enter_guesses,
            text="Enter Guesses").grid(row=0, column=0, sticky='w')
        self.enter_bounds = tk.IntVar(value=1)
        self.checks['enter_bounds'] = tk.ttk.Checkbutton(
            self.run_fit_frame, text="Enter Bounds",
            variable=self.enter_bounds)
        self.checks['enter_bounds'].grid(row=0, column=1)

        tk.ttk.Label(self.run_fit_frame, text='Algorithm:').grid(
            row=1, column=0)
        self.algo = tk.StringVar(value='leastsq')
        tk.ttk.OptionMenu(
            self.run_fit_frame, self.algo, self.algo.get(),
            'leastsq', 'SLSQP', 'nelder').grid(row=1, column=1)

        tk.ttk.Label(self.run_fit_frame,
                     text="Func. Tol. (log10):").grid(
                         row=2, column=0, sticky='w')
        self.fun_tol = tk.IntVar(value=-5)
        self.entries['fun_tol'] = tk.ttk.Entry(
            self.run_fit_frame, textvariable=self.fun_tol, width=5)
        self.entries['fun_tol'].grid(row=2, column=1)
        tk.ttk.Label(self.run_fit_frame, text="Init. variance (%):").grid(
            row=2, column=2, sticky='w')
        self.init_variance = tk.DoubleVar(value=1.0)
        self.entries['init_variance'] = tk.ttk.Entry(
            self.run_fit_frame, textvariable=self.init_variance, width=5)
        self.entries['init_variance'].grid(row=2, column=3)

        self.verify_fit = tk.IntVar(value=0)
        self.checks['verify_fit'] = tk.ttk.Checkbutton(
            self.run_fit_frame,
            text="Control first fit",
            variable=self.verify_fit)
        self.checks['verify_fit'].grid(row=3, column=0)

        self.live_plot_enabled = tk.IntVar(value=0)
        self.checks['live_plot_enabled'] = tk.ttk.Checkbutton(
            self.run_fit_frame,
            text="Live Plot",
            variable=self.live_plot_enabled)
        self.checks['live_plot_enabled'].grid(row=3, column=1)

        self.buttons['run_fit'] = ttk.Button(
            self.run_fit_frame, text="Run fit", command=self.start_fit)

        self.buttons['run_fit'].grid(row=4, column=0, padx=2, pady=2)

        self.cancel_button = ttk.Button(
            self.run_fit_frame, text="Cancel", command=self.cancel_fit)
        self.cancel_button.grid(row=4, column=1, padx=2, pady=2)
        self.cancel_button.config(state='disabled')

        self.disp_opts = GroupBox(
            self.fit_frame, text="Display results", dim=(2, 2))
        self.disp_opts.grid(row=1, column=0, padx=2, pady=2, sticky='nwse')
        self.show_movie_after = tk.IntVar(value=0)
        self.show_fit = tk.IntVar(value=0)
        self.show_para_curve = tk.IntVar(value=0)
        tk.ttk.Checkbutton(self.disp_opts,
                           text="Fit & Residual",
                           variable=self.show_fit,
                           command=self.show_fit_callback).grid(
                               row=0, column=0, sticky='w')
        tk.ttk.Checkbutton(
            self.disp_opts,
            text="Parameter Curves",
            variable=self.show_para_curve,
            command=self.show_para_callback).grid(
                row=1, column=0, sticky='w', columnspan=2)
        tk.ttk.Checkbutton(
            self.disp_opts,
            text="Movie",
            variable=self.show_movie_after,
            command=self.show_movie_callback).grid(row=0, column=1, sticky='w')

        self.browse_results_frame = GroupBox(self, text="Results", dim=(4, 5))
        self.frames['result_disp_commands'] = CustomFrame(
            self.browse_results_frame, dim=(2, 2), border=False)
        self.browse_results_frame.grid(
            row=3, column=1, sticky='wnse', pady=5, padx=5)

        tk.Label(self.browse_results_frame, text="Show spectrum at:").grid(
            row=1, column=0, sticky=tk.W)
        self.show_fit_at = tk.DoubleVar(value=self.tupper.get())
        self.entries['show_fit_at'] = tk.Entry(
            self.browse_results_frame, textvariable=self.show_fit_at, width=10)
        self.entries['show_fit_at'].grid(
            row=1, column=1, sticky='we', columnspan=2)
        self.entries['show_fit_at'].bind('<Return>',
                                         self.show_fit_trace_callback)

        tk.Label(self.browse_results_frame, text=self.data_obj.time_unit).grid(
            row=1, column=3, sticky=tk.W)

        self.plot_components = tk.IntVar(value=0)
        self.checks['plot_components'] = tk.ttk.Checkbutton(
            self.browse_results_frame,
            text="Show Components",
            variable=self.plot_components,
            command=self.show_fit_trace_callback)
        self.checks['plot_components'].grid(row=2, column=0, sticky='w')

        left_arrow = {'children': [('Button.leftarrow', None)]}
        right_arrow = {'children': [('Button.rightarrow', None)]}
        style = ttk.Style(self)
        style.layout('L.TButton', [('Button.focus', left_arrow)])
        style.layout('R.TButton', [('Button.focus', right_arrow)])
        self.buttons['previous_fit'] = ttk.Button(
            self.browse_results_frame,
            style='L.TButton',
            command=lambda *args: self.show_next_trace(*args, incr=-1))
        self.buttons['previous_fit'].grid(row=2, column=1, sticky='ne', padx=1)
        self.buttons['next_fit'] = ttk.Button(
            self.browse_results_frame,
            style='R.TButton',
            command=self.show_next_trace)
        self.buttons['next_fit'].grid(row=2, column=2, sticky='nw', padx=1)

        self.buttons['show_movie'] = ttk.Button(
            self.frames['result_disp_commands'],
            text="Show Movie",
            command=self.show_movie_callback)
        self.buttons['show_movie'].grid(row=0, column=0, padx=5)

        self.buttons['show_fit'] = ttk.Button(
            self.frames['result_disp_commands'],
            text="Show Fit Map",
            command=self.show_fit_and_resid)
        self.buttons['show_fit'].grid(
            row=1, column=0, padx=5)

        self.buttons['inspect_traces'] = tk.ttk.Button(
            self.frames['result_disp_commands'],
            text="Inspect Fit Traces",
            command=self.inspect_traces)
        self.buttons['inspect_traces'].grid(row=2, column=0, padx=5)
        self.frames['result_disp_commands'].grid(row=3, column=0, columnspan=4,
                                                 sticky='wnse', rowspan=2)

        # parameter fit frame
        self.para_frame = GroupBox(self, text="Parameters", dim=(2, 5))
        self.para_frame.grid(row=3, column=2, sticky='wnse', pady=5, padx=5)

        self.buttons['show_paras'] = ttk.Button(
            self.para_frame,
            text="Show Parameters",
            command=self.open_para_curve_window)
        self.buttons['show_paras'].grid(
            row=0, column=0, padx=5)

        self.buttons['open_para_fit'] = ttk.Button(
            self.para_frame,
            text="Fit",
            command=lambda *args, c='native':
                self.open_para_fit_window(case=c))
        self.buttons['open_para_fit'].grid(row=1, column=0, padx=2, pady=2)

        self.buttons['show_areas'] = ttk.Button(
            self.para_frame,
            text="Plot Areas",
            command=self.open_area_or_widths_window)
        self.buttons['show_areas'].grid(row=2, column=0, padx=5)
        self.buttons['open_area_fit'] = ttk.Button(
            self.para_frame,
            text="Fit",
            command=lambda *args, c='areas': self.open_para_fit_window(case=c))
        self.buttons['open_area_fit'].grid(row=3, column=0, padx=2, pady=2)

        ttk.Button(self.para_frame, text="Save Parameters",
                   command=self.save_paras).grid(
                       row=4, column=0, padx=2, pady=2)

        # progressbar frame
        self.progressbar_frame = tk.Frame(
            self, highlightbackground="grey", highlightthickness=1, bd=0)
        self.progressbar_frame.grid(
            row=4, column=0, columnspan=3, sticky='we', pady=5, padx=5)
        self.progress_val = tk.DoubleVar(value=0)
        self.progressbar = ttk.Progressbar(self.progressbar_frame,
                                           orient=tk.HORIZONTAL,
                                           length=400,
                                           mode='determinate',
                                           variable=self.progress_val)
        self.progressbar.grid(row=0, column=0, sticky=tk.W, padx=20)
        self.pb_len = self.progressbar.cget("length")
        self.status_lbl = tk.Label(
            self.progressbar_frame, text="Please set up fit and run.")
        self.status_lbl.grid(row=0, column=1, sticky='we', pady=10)

        self.fit_done = False
        # self.auto_bounds_factor = 1e3
        self.fit_obj = None
        self.prev_para = None
        self.previous_inits = None
        center_toplevel(self, controller)
        disable_list = [['open_para_fit', 'show_fit', 'show_paras',
                         'show_movie', 'next_fit', 'previous_fit',
                         'show_areas', 'open_area_fit'],
                        ['show_fit_at'], [],
                        # ['plotTraceVia']
                        ]
        self.widget_list = [self.buttons, self.entries, self.checks,
                            # self.optmenus
                            ]
        for i in range(len(self.widget_list)):
            for k in disable_list[i]:
                self.widget_list[i][k].config(state='disabled')
        for widget in self.entries.values():
            widget.config(justify=tk.RIGHT)

    # UI callbacks & methods
    def time_win_callback(self, *args, case='upper'):
        if case == 'upper' and self.show_fit_at.get() > self.tupper.get():
            self.show_fit_at.set(self.tupper.get())
        elif case == 'lower' and self.show_fit_at.get() < self.tlower.get():
            self.show_fit_at.set(self.tlower.get())
        self.main_figure.set_ylim([self.tlower.get(), self.tupper.get()])

    def xwin_callback(self, *args, **kwargs):
        self.main_figure.set_xlim([self.xlower.get(), self.xupper.get()])
        self.trace_figure.set_xlim([self.xlower.get(), self.xupper.get()])

    def show_fit_trace_callback(self, *args):
        try:
            self._get_time_index()
            self.plot_fit_trace()
        except Exception:
            pass

    def show_para_callback(self):
        if self.show_para_curve.get():
            try:
                self.open_para_curve_window()
            except Exception:
                pass

    def show_fit_callback(self):
        if self.show_fit.get():
            try:
                self.show_fit_and_resid()
            except Exception:
                pass

    def enable_widgets(self):
        for widget in self.widget_list:
            for k in widget.keys():
                widget[k].config(state='normal')

    def save_paras(self):
        file = save_box(
            fext=".txt",
            filetypes=(('text files', '.txt'), ('Matlab files', '.mat')))
        if file is not None:
            self.para_traces.save(file, save_fit=False,
                trace_keys=[key for key in self.para_traces.tr.keys()])

    def trace_fig_callback(self, event):
        if event.dblclick or event.button == 3:
            fig = self.controller.open_topfigure_wrap(
                self,
                fig_obj=self.trace_figure,
                controller=self,
                plot_func=lambda *args, **kwargs: self.trace_plot_func(
                    *args, plot_comp=self.plot_components.get(), **kwargs),
                editable=True)
            fig.fr.figure.plot_function = (
                lambda *args, **kwargs:
                    self.trace_plot_func(
                        *args,
                        plot_comp=self.plot_components.get(),
                        fit_comp_alpha=0,
                        **kwargs))
            fig.fr.figure.plot(update_canvas=False)
            fig.fr.opts.plot_opts.vars['fill_curve'].set(1)
            fig.fr.opts.plot_opts.vars['fill_curve_alpha'].set("0.1")
            fig.fr.opts.plot_opts.fill_curve_callback()
            fig.fr.figure.clear_fill(lines=1)

    # running line shape fit
    def start_fit(self):
        def start_thread(*args, **kwargs):
            self._fit_running = True
            self.queue = Queue()
            self.task = ThreadedTask(self.data_obj.dynamic_line_shape, *args,
                                     after_finished_func=self._after_fit,
                                     interruptible=True,
                                     queue=self.queue,
                                     **kwargs)
            self.task.start()
            self.listener = threading.Thread(
                target=self._live_plot, args=(self.queue,))
            self.listener.start()
            self.queue.join()

        self.current_line_shape = self.lineshape.get()

        self.live_plot_enabled_ = self.live_plot_enabled.get()
        self.task = None
        fit_obj = self.data_obj.init_line_shape_fit(
            num_comp=self.num_comp.get(),
            fit_obj=None, guesses=None,
            line_shape=self.current_line_shape,
            constraints={'width': self.linewidth_model.get(),
                         'spacing': 'equal' if bool(self.equal_space.get())
                         else 'unconstrained'},
            constraint_inits={'width': 500.0,
                              'spacing': 1000.0},
            offset=bool(self.offset_opt.get()),
            fun_tol=np.power(
                10.0, self.fun_tol.get()),
            method=self.algo.get(),
            num_function_eval=int(1e5))

        self.xinds = np.logical_and(
            self.data_obj.wavenumbers >= self.xlower.get(),
            self.data_obj.wavenumbers <= self.xupper.get())
        if self.reverse.get():
            time_ind = np.where(
                self.data_obj.time_delays <= self.tupper.get())[0][-1]
        else:
            time_ind = np.where(
                self.data_obj.time_delays >= self.tlower.get())[0][0]
        fit_obj.auto_guess_gaussian(x=self.data_obj.wavenumbers[self.xinds],
                                    y=self.data_obj.delA[time_ind, self.xinds])
        for c in ('upper', 'lower'):
            self.time_win_callback(case=c)
        if self.enter_guesses.get():
            if self.enter_bounds.get():
                window = FitParaOptsWindow(self, fit_obj.params,
                                           controller=self,
                                           input_values=self.previous_inits)
                self.wait_window(window)
                if window.output is None:
                    self._after_fit()
                    return
                else:
                    fit_obj.params = window.write_parameters(fit_obj.params)
                    self.previous_inits = window.output
                    guesses = None
            else:
                headers = {
                    "curve": "",
                    "comp": "Component",
                    "x": self.data_obj.spec_unit.join(["Position (", ")"]),
                    "y": "Amplitude",
                    "x2": self.data_obj.spec_unit.join(["Sigma (", ")"])}
                window = FitParaEntryWindowLmfit(
                    self, [""], fit_obj,
                    controller=self,
                    headers=headers,
                    input_values=self.previous_inits,
                    case="guess")

                self.wait_window(window)

                try:
                    guesses = window.output[""]
                except Exception:
                    self._after_fit()
                    return
        else:
            guesses = 'auto'
        fit_obj, success, report = self.data_obj.single_line_shape(
            time_index=time_ind, fit_obj=fit_obj,
            xinds=self.xinds)
        self.trace_plot = []
        if success:
            self.trace_figure.axes[0].cla()
            self.trace_figure.plot()
            self.trace_plot.extend(self.trace_figure.axes[0].plot(
                self.data_obj.wavenumbers[self.xinds],
                self.data_obj.delA[time_ind, self.xinds]))
            self.trace_plot.extend(self.trace_figure.axes[0].plot(
                self.data_obj.wavenumbers[self.xinds], fit_obj.curve,
                color=self.controller.settings['fit_color']))
            self.trace_plot.append(
                self.trace_figure.axes[0].set_title(" ".join([
                    str(np.round(self.data_obj.time_delays[time_ind],
                             self.data_obj.time_delay_precision)),
                     self.data_obj.time_unit]),
                    transform=self.trace_figure.axes[0].transAxes))
            if self.plot_components.get():
                curve, comps = fit_obj.fit_function(
                    fit_obj.result.params, return_comps=True)(fit_obj.x)
                self._plot_components(comps=comps, fit_obj=fit_obj)
            self.trace_figure.canvas.draw()
        else:
            self._after_fit()
            messagebox.showerror(message="\n\n".join([
                "Fit failed. Please adjust window and/or parameters.",
                report]), parent=self)
            return
        if self.verify_fit.get():
            window = BinaryDialog(
                self, self, prompt="First Fit plotted in trace figure."
                + " Continue Fit?")
            if not window.output:
                self._after_fit()
                return
        self.fit_obj = fit_obj
        self.areas = None
        self.prev_para = {"": {}}
        self.trace_figure.axes[0].collections.clear()
        for p in self.fit_obj.params:
            self.prev_para[""][p] = self.fit_obj.params[p].value
        self.status_lbl.config(text="Fit in progress.")
        self.current_num_comp = self.num_comp.get()
        self.widths = self.linewidth_model.get()
        self.buttons['run_fit'].config(state='disabled')
        self.cancel_button.config(state='normal')
        start_thread(
            fit_obj=self.fit_obj, xinds=self.xinds,
            yrange=[self.tlower.get(), self.tupper.get()],
            guesses=guesses,
            reverse=bool(self.reverse.get()),
            variance=self.init_variance.get()/100)

    def cancel_fit(self):
        try:
            self.task.raise_exception()
        except Exception as e:
            messagebox.showerror(message=e, parent=self)

    def _live_plot(self, in_queue):
        while self._fit_running:
            d = in_queue.get()
            fit = d['fit']
            dat = d['dat']
            ind = d['point']
            prog = d['prog']
            if self.live_plot_enabled_:
                self.trace_plot[0].set_ydata(dat)
                self.trace_plot[1].set_ydata(fit)
                self.trace_plot[2].set_text(
                    " ".join(
                        [str(np.round(self.data_obj.time_delays[ind],
                                      self.data_obj.time_delay_precision)),
                         self.data_obj.time_unit]))
                try:
                    self.after(1, lambda: self.trace_figure.canvas.draw())
                except Exception as e:
                    print(e)
            try:
                self.progress_val.set(prog)
                self.progressbar.update_idletasks()
            except Exception as e:
                print(e)
            self.update()

    def _after_fit(self):
        try:
            if self.task.output is not None:
                (self.fit_para, self.fit_matrix, self.ci,
                 self.residual, time_points) = self.task.output
                self._fit_running = False
                self.time_points = np.sort(time_points)
                self.after(100, self._post_fit_ops)
            else:
                self._fit_running = False
        except Exception:
            self._fit_running = False
        finally:
            self.buttons['run_fit'].config(state='normal')
            self.cancel_button.config(state='disabled')

    def _post_fit_ops(self):
        self.progress_val.set(self.pb_len)
        self.time_inds = np.logical_and(
            self.data_obj.time_delays >= self.time_points[0],
            self.data_obj.time_delays <= self.time_points[-1])
        # write parameter trace object
        self.para_traces = TATrace()
        self.para_traces.active_traces = [p for p in self.fit_obj.params]
        self.para_traces.xdata = self.time_points
        self._sort_para_traces()
        for key in self.para_traces.active_traces:
            self.para_traces.tr[key] = {
                'y': np.array(self.fit_para[key]['value'])}
            lb1 = self.para_traces.ylabel
        lb2 = self.main_figure.get_xlabel()
        self.para_traces.assign_axes_labels(
            {'amp': {'ylabel': lb1, 'title': 'Amplitude'},
             'x0': {'ylabel': lb2, 'title': 'Position'},
             'sig': {'ylabel': lb2, 'title': 'Sigma'},
             'gam': {'ylabel': lb2, 'title': 'Gamma'},
             'spac': {'ylabel': lb2, 'title': 'Spacing'},
             'const': {'ylabel': lb1, 'title': 'Offset'}})
        self.para_traces.xlabel = self.main_figure.get_ylabel()
        self.calculate_areas()
        self.calculate_linewidths()
        # write fit and residual matrices
        self.wav = self.data_obj.wavenumbers[self.xinds]
        # post-fit functions; update plots and GUI
        self.show_fit_trace_callback()
        self.status_lbl.config(text="Fit is done.")
        self.para_toplevel_fig = None
        self.fit_done = True
        if self.show_movie_after.get():
            self.show_movie_callback()
        if self.show_fit.get():
            self.show_fit_and_resid()
        if self.show_para_curve.get():
            self.open_para_curve_window()
        self.enable_widgets()
        self.quantity_to_calc = "Widths" if re.search(
            'voigt', self.current_line_shape, re.I) else "Areas"
        self.buttons['show_areas'].config(text="Plot " + self.quantity_to_calc)
        self.buttons['open_area_fit'].config(text="Fit "
                                             + self.quantity_to_calc)

    def _sort_para_traces(self):
        active = []
        j = 1
        found = True
        while found:
            found = False
            for name in 'amp', 'x0', 'sigma', 'gamma':
                for tr in self.para_traces.active_traces:
                    if re.search("_".join([name, str(j)]), tr, re.I):
                        active.append(tr)
                        found = True
                        break
            j += 1
        for name in 'const', 'spacing', 'factor':
            for tr in self.para_traces.active_traces:
                if re.search(name, tr, re.I):
                    active.append(tr)
                    break
        self.para_traces.active_traces = active

    # results operations
    # display fit spectrum at time point
    def _get_time_index(self):
        try:
            self.time_ind = np.where(
                self.time_points <= self.show_fit_at.get())[0][-1]
        except Exception:
            self.time_ind = 0

    def trace_plot_func(self, *args, plot_comp=False,
                        include_fit=True, **kwargs):
        def plot_comp(i, j, ax, zord, cycle, *args, fit_comp_alpha=0.25,
                      transpose=False, **kwargs):
            lines = []
            lw = 0.2
            for key, comp in self.comps.items():
                lines.append(plot(ax, self.fit_obj.x, comp, linewidth=lw,
                                  color=cycle[i], zorder=zord[::-1][i],
                                  transpose=transpose, label=key))
                if fit_comp_alpha:
                    lines.append(fill_between(ax, self.fit_obj.x, comp,
                                              alpha=fit_comp_alpha,
                                              transpose=transpose,
                                              zorder=zord[::-1][i],
                                              label='_nolegend_',
                                              color=cycle[i]))
                i += 1
            return i, lines

        if include_fit and plot_comp:
            if self.comps is None:
                self.comps = self.calculate_comps(self.time_ind)
            function_extension = plot_comp
            numlines = len(list(self.comps.keys())) + 2
        else:
            function_extension = None
            numlines = 1

        return tk_mpl_plot_function(
            self.trace_plot[0].get_xdata(),
            [self.data_obj.delA[self.time_inds, :][
                self.time_ind, self.xinds]], *args,
            fit_x=self.fit_obj.x, fit_y=[
                self.fit_matrix[self.time_ind, :]],
            min_cycle_len=numlines,
            include_fit=include_fit,
            show_fit_legend=True,
            function_extension=function_extension,
            **kwargs)

    def plot_fit_trace(self):
        if self.show_fit_at.get() < self.time_points[0]:
            self.show_fit_at.set(np.max(self.time_points))
            self._get_time_index()
        elif self.show_fit_at.get() > self.time_points[-1]:
            self.show_fit_at.set(np.min(self.time_points[0]))
            self._get_time_index()
        self.show_fit_at.set(self.time_points[self.time_ind])
        z = self.data_obj.delA[self.time_inds, :]
        z = z[self.time_ind, self.xinds]
        self.trace_plot[0].set_ydata(z)
        self.trace_plot[1].set_ydata(self.fit_matrix[self.time_ind, :])
        self.trace_plot[2].set_text("")
        if self.plot_components.get():
            self.comps = self._plot_components()
        else:
            self.comps = None
            self.trace_figure.axes[0].collections.clear()
            for i in range(len(self.trace_plot) - 1, 2, -1):
                self.trace_figure.axes[0].lines[i - 1].remove()
                self.trace_plot.pop(i)
        self.trace_figure.axes[0].relim()
        self.trace_figure.axes[0].autoscale()
        self.trace_figure.axes[0].set_title(
            " ".join([str(np.round(self.show_fit_at.get(), 3)),
                      self.data_obj.time_unit]))
        self.trace_figure.canvas.draw()

    def calculate_comps(self, time_ind):
        paras = lmfit.Parameters()
        for p in self.fit_obj.params:
            try:
                paras.add(p, value=self.para_traces.tr[p]['y'][time_ind])
            except KeyError:
                paras.add(p, value=self.fit_obj.params[p].value)
                raise
            except Exception:
                raise
        curve, comps = self.fit_obj.fit_function(
            paras, return_comps=True)(self.fit_obj.x)
        return comps

    def _plot_components(self, alpha=0.2, comps=None, fit_obj=None):
        if not comps:
            comps = self.calculate_comps(self.time_ind)
        if not fit_obj:
            fit_obj = self.fit_obj
        if alpha:
            self.trace_figure.axes[0].collections.clear()
            lw = 0.1
        else:
            lw = 0.5
        j = 3
        zord, cycle = set_line_cycle_properties(2 + len(comps.keys()))
        for comp in comps.values():
            try:
                self.trace_plot[j].set_ydata(comp)
            except Exception:
                line, = self.trace_figure.axes[0].plot(
                    fit_obj.x, comp, linewidth=lw, color=cycle[j - 1])
                self.trace_plot.append(line)
            if alpha:
                self.trace_figure.axes[0].fill_between(
                    fit_obj.x, comp, alpha=alpha, label='_nolegend_',
                    color=cycle[j - 1])
            j += 1
        return comps

    def show_next_trace(self, incr=1):
        self.time_ind = self.time_ind + incr
        try:
            self.show_fit_at.set(self.time_points[self.time_ind])
        except Exception:
            if incr > 0:
                self.time_ind = 0
            else:
                self.time_ind = len(self.time_points) - 1
            self.show_fit_at.set(self.time_points[self.time_ind])
        self.plot_fit_trace()

    def inspect_traces(self, *args):
        z = self.data_obj.delA[:, self.xinds]
        z = z[self.time_inds, :]
        dat = {'x': self.fit_obj.x,
               'y': self.time_points,
               'z': z,
               'xlabel': self.main_figure.get_xlabel(),
               'ylabel': self.main_figure.get_ylabel(),
               'zlabel': self.main_figure.get_clabel(),
               'fit': self.fit_matrix}
        TraceManagerWindow(self, self.controller, dat, self.data_obj,
                           map_figure=self.main_figure,
                           title='Dynamic Line Shape Fit Traces',
                           residual_widgets=True)

    # plot parameters curves and areas
    def calculate_areas(self):
        self.areas = TATrace(xdata=self.para_traces.xdata,
                             xlabel=self.para_traces.xlabel,
                             xunit=self.para_traces.get_xunit(),
                             ylabel="",
                             yunit="")

        if re.search('gauss', self.current_line_shape, re.I):
            amp_key = 'gauss_amp_'
            wd_key = 'gauss_sigma_'
        elif re.search('loren', self.current_line_shape, re.I):
            amp_key = 'lorentz_amp_'
            wd_key = 'lorentz_gamma_'
        else:
            return
        for i in range(self.current_num_comp):
            self.areas.tr['Component ' + str(i + 1)] = {
                'y': np.power(np.pi, 2) * np.multiply(
                    self.para_traces.tr[amp_key + str(i + 1)]['y'],
                    self.para_traces.tr[wd_key + str(i + 1)]['y'])}
            self.areas.active_traces.append('Component ' + str(i + 1))

    def calculate_linewidths(self):
        def calc_width(sigma, gamma):
            w_g = 2 * np.sqrt(2 * np.log(2)) * sigma
            w_l = 2 * gamma
            return 0.5346*w_l + np.sqrt(0.2166 * w_l**2 + w_g**2)

        if not re.search('voigt', self.current_line_shape, re.I):
            self.widths = None
            return
        self.widths = TATrace(xdata=self.para_traces.xdata,
                              xlabel=self.para_traces.xlabel,
                              xunit=self.para_traces.get_xunit(),
                              ylabel="",
                              yunit="")
        for i in range(self.current_num_comp):
            self.widths.tr['Component ' + str(i + 1)] = {
                'y': np.array([calc_width(
                    self.para_traces.tr['voigt_sigma_' + str(i + 1)]['y'][j],
                    self.para_traces.tr['voigt_gamma_' + str(i + 1)]['y'][j])
                               for j in range(len(self.para_traces.xdata))])}
            self.widths.active_traces.append('Component ' + str(i + 1))

    def open_area_or_widths_window(self):
        if re.search('are', self.quantity_to_calc, re.I):
            case = 'areas'
            if not self.areas:
                self.calculate_areas()
        elif re.search('wid', self.quantity_to_calc, re.I):
            case = 'widths'
            if not self.widths:
                self.calculate_linewidths()
        self.open_para_curve_window(case=case)

    def plot_para_curve(self, j, tr_obj=None, fig=None, ax=None, fill=0,
                        plot_ci=True, reverse_z=False, transpose=False,
                        color_order=None, color_cycle=None,
                        interpolate_colors=False,
                        # legend_kwargs=None,
                        **plot_kwargs):
        if fig is None:
            fig = plt.figure()
            plt.close()
            fig.set_tight_layout(True)
        if ax is None:
            ax = fig.add_subplot(111)
        if not tr_obj:
            tr_obj = self.para_traces
        key = tr_obj.active_traces[j]
        ax.cla()
        ax.plot(tr_obj.xdata,
                tr_obj.tr[key]['y'], **plot_kwargs)
        if np.any(fill):
            fill = get_fill_list(fill, 1)
            ax.fill_between(tr_obj.xdata,
                            tr_obj.tr[key]['y'], alpha=fill[0])
        if plot_ci:
            try:
                ax.fill_between(tr_obj.xdata,
                                [tr_obj.tr[key]['y'][k]+self.ci[k][j] for k in
                                 range(len(tr_obj.tr[key]['y']))],
                                [tr_obj.tr[key]['y'][k]-self.ci[k][j] for k in
                                 range(len(tr_obj.tr[key]['y']))],
                                color=self.controller.settings['fit_color'],
                                alpha=0.15,
                                label='_nolegend_')
            except Exception:
                pass

    def save_para_traces(self, tr_obj):
        file = save_box(filetypes=[('text files', '.txt'),
                                   ('Matlab files', '.mat')],
                        fext='.txt', parent=self)
        if file is not None:
            return tr_obj.save_traces(file, save_fit=False)

    def open_para_curve_window(self, case="native"):
        if case.lower() == 'native':
            tr_obj = self.para_traces
        elif case.lower() == 'areas':
            tr_obj = self.areas
        elif case.lower() == 'widths':
            tr_obj = self.widths
        else:
            return
        plot_func = (lambda *args, include_fit=True, **kwargs:
                     tr_obj.plot_single(
                         *args, include_fit=include_fit, **kwargs))
        num_subplots = len(tr_obj.active_traces)
        button_dict = {'Save traces': lambda: self.save_para_traces(tr_obj),
                       'Save by parameter': self.save_para_separately}
        titles = []
        legends = []
        if case.lower() == 'native':
            nrows = (self.current_num_comp
                     + int(bool(self.equal_space.get())
                           or bool(self.offset_opt.get())
                           or self.linewidth_model.get() != 'unconstrained'))
            ncols = 3
            i = 1
            for lb in tr_obj.active_traces:
                if re.search('amp', lb, re.I):
                    pname = 'Amplitude'
                elif re.search('x0', lb, re.I):
                    pname = 'Position'
                elif re.search('sig', lb, re.I):
                    pname = 'Sigma'
                elif re.search('gamma', lb, re.I):
                    pname = 'Gamma'
                else:
                    continue
                titles.append("".join(['Comp. ', str(i), ': ', pname]))
                legends.append([pname])
                if pname == 'Sigma' or pname == 'Gamma':
                    i += 1
            for lb in tr_obj.active_traces:
                if re.search('spac', lb, re.I):
                    pname = 'Spacing'
                elif re.search('factor', lb, re.I):
                    pname = 'Width Factor'
                elif re.search('const', lb, re.I):
                    pname = 'Offset'
                else:
                    continue
                titles.append(pname)
                legends.append([pname])
            ylabels = []
            for i in range(nrows):
                ylabels.extend([self.main_figure.clabels[0],
                                self.main_figure.get_xlabel(),
                                self.main_figure.get_xlabel()])
                if re.search('voigt', self.current_line_shape, re.I):
                    ylabels.append(self.main_figure.get_xlabel())
        else:
            if case.lower() == 'areas':
                lbl = "Area of "
            else:
                lbl = "Width of "
            ncols = 3 - int(num_subplots < 3) - int(num_subplots < 2)
            nrows = int(num_subplots/3) + int(np.mod(num_subplots, 3) > 0)
            for name in tr_obj.active_traces:
                legends.append([name])
                titles.append(lbl + name)
            ylabels = self.main_figure.get_xlabel()
        self.para_toplevel_fig = self.controller.open_topfigure_wrap(
            self,
            plot_func=plot_func,
            num_subplots=num_subplots,
            controller=self,
            dim=[500*3, 350*nrows],
            plot_type='linefit',
            editable=True,
            num_rows=nrows,
            num_columns=ncols,
            xlabels=self.main_figure.get_ylabel(),
            ylabels=ylabels,
            legends=legends,
            axes_titles=titles,
            button_dict=button_dict,
            canvas_callback={'button_press_event':
                             lambda event: self.para_fig_window_callback(
                                 event, nrows, plot_func)})
        self.para_toplevel_fig.fr.opts.plot_opts.add_plot_ci_option()
        self.para_toplevel_fig.fr.nrows = nrows
        self.para_toplevel_fig.fr.figure.plot_all()

    def plot_areas(self, j, fig=None, ax=None, fill=0, reverse_z=False,
                   color_order=None, color_cycle=None,
                   interpolate_colors=False, transpose=False,
                   # legend_kwargs=None,
                   **plot_kwargs):
        if fig is None:
            fig = plt.figure()
            plt.close()
            ax = None
        if ax is None:
            ax = fig.add_subplot(111)
        ax.cla()
        line = ax.plot(self.para_traces.xdata, self.areas[j], **plot_kwargs)
        if np.any(fill):
            fill = get_fill_list(fill, 1)
            ax.fill_between(self.para_traces.xdata, self.areas[j],
                            alpha=fill[0], color=line.get_color(),
                            label='_nolegend_')

    # line fit of parameters
    def open_para_fit_window(self, case='native'):
        if self.fit_done:
            to = self.areas if case.lower() == 'areas' else self.para_traces
            win = tk.Toplevel(self)
            win.fr = FitTracePage(
                win, xunit=self.data_obj.time_unit, mode='kinetic',
                trace_obj=to, figure_dim=[500, 300])
            win.fr.grid(sticky='wnse')
            win.fr.fit_figure.set_plot_kwargs_all(
                fit_kwargs={"color": "black"})
            # win.fr.frames['trace_select'].update_box()
            win.fr.auto_axis_limits()
            win.title("Dynamic Line Shape: Parameter Fit")
            center_toplevel(win, self)
            win.protocol("WM_DELETE_WINDOW", lambda *args, **kwargs:
                         self._delete_fit_window(win, to))

    def _delete_fit_window(self, window, trace_obj):
        trace_obj.set_all_active()
        window.destroy()

    # movie
    def show_movie_callback(self):
        frame_labels = [str(np.round(td, 3)) + ' ' +
                        self.data_obj.time_unit for td in self.time_points]
        x = self.data_obj.wavenumbers[self.xinds]
        z = self.data_obj.delA[:, self.xinds]
        z = z[self.time_inds, :]
        z = np.array([z])
        if self.plot_components.get():
            comps = []
            for i in range(np.shape(z)[1]):
                comps.append([v for v in self.calculate_comps(i).values()])
            comps = np.array(comps).transpose(1, 0, 2)
            z = np.concatenate((z, comps), axis=0)
            d = [x, z]
            num_comps = np.shape(comps)[0]
            plt_kw = {'linestyle': [" " for i in range(num_comps + 1)],
                      'marker': [None for i in range(num_comps + 1)]}
            plt_kw['linestyle'][0] = '-'
        else:
            d = [x, z]
        clims = self.data_obj.get_clim(z, opt='asymmetric', offset=0.05)
        mov = PlotMovie(self, self.controller, data=d,
                        fit=[x, np.array([np.array(self.fit_matrix)])],
                        xlimits=[self.trace_figure.get_xlim()],
                        ylimits=[clims], show_movie=False,
                        title='DLSA_' + self.fit_obj.model,
                        frame_labels=frame_labels,
                        plot_kwargs=plt_kw)
        # mov.show_movie(update_canvas=False)
        mov.show_movie()
        mov.stop()
        mov.fr.opts.opt_panels['lines'].vars['fill_curve'].set(1)
        mov.fr.opts.opt_panels['lines'].vars['fill_curve_alpha'].set(0.1)
        if self.plot_components.get():
            mov.fr.opts.opt_panels['lines'].fill_curve_callback(
                update_canvas=False)
            for i in range(1, num_comps + 1):
                mov.fr.figure.set_lines_properties(
                    j=i, update_canvas=False, linestyle="None")
            mov.replay()
        else:
            mov.fr.opts.opt_panels['lines'].fill_curve_callback(
                update_canvas=True)

    # fit and residual maps
    def show_fit_and_resid(self):
        z = self.data_obj.delA[self.time_inds, :]
        self.fit_plot_data = [z[:, self.xinds], self.fit_matrix, self.residual]
        self.fit_figure_win = self.controller.open_topfigure_wrap(
            self,
            plot_func=self.plot_fit_maps,
            num_subplots=3, num_rows=1,
            num_columns=3,
            plot_type='2D',
            editable=True,
            dim=[900, 300],
            controller=self,
            xlabels=self.main_figure.get_xlabel(),
            ylabels=self.main_figure.get_ylabel(),
            xlimits=self.main_figure.get_xlim(),
            ylimits=self.main_figure.get_ylim(),
            axes_titles=[
                'Data', 'Fit', 'Residual'],
            header={
                'text': 'Double-click on plots to edit and/or save',
                'fontsize': 12},
            canvas_callback={'button_press_event': self.fit_fig_callback})
        self.fit_figure_win.fr.figure.color_obj[0].update(
            clims=self.data_obj.color.clims)
        self.fit_figure_win.fr.figure.color_obj[1].update(
            clims=self.data_obj.color.clims)
        self.fit_figure_win.fr.figure.color_obj[2].update(
            clims=self.data_obj.get_clim(self.residual))
        self.fit_figure_win.fr.figure.plot_all()

    def save_fit_or_resid(self, case='fit'):
        case_dict = {'fit': [self.fit_matrix, 'lineshape_map'],
                     'resid': [self.residual, 'lineshape_residual']}
        file = save_box(fext='.mat',
                        filetypes=[('text files', '.txt'),
                                   ('Matlab files', '.mat')],
                        fname=case_dict[case][1],
                        parent=self)
        if file is not None:
            self.data_obj.save_data_matrix(
                file, matrix=case_dict[case][0],
                x=self.data_obj.wavelengths[self.xinds],
                y=self.time_points)

    def fit_fig_callback(self, event):
        if event.dblclick:
            i = int(event.inaxes.get_position().extents[0] >= 0.5)
            fig = self.fit_figure_win.fr.figure
            i = fig.current_axes_no
            button_dict = [{}, {'Save Fit': (
                lambda: self.save_fit_or_resid(case='fit'))},
                {'Save Residual': (
                    lambda: self.save_fit_or_resid(case='resid'))}]

            self.controller.open_topfigure_wrap(
                self, plot_func=(lambda *args, **kwargs:
                                 self.plot_fit_maps(i, *args, **kwargs)),
                num_subplots=1, controller=self,
                axes_titles=fig.axes_titles[i], xlabel=fig.xlabels[i],
                ylabel=fig.ylabels[i], ylimits=fig.get_ylim(i=i),
                xlimits=fig.get_xlim(i=i), color_obj=fig.color_obj[i],
                editable=True, xvals=self.data_obj.wavenumbers[self.xinds],
                yvals=self.time_points, button_dict=button_dict[i])

    def plot_fit_maps(self, i, *args, fig=None, ax=None,
                      cmap=None, **kwargs):
        if fig is None:
            fig = plt.figure()
            plt.close()
            fig.set_tight_layout(True)
        if ax is None:
            ax = fig.add_subplot(111)
        if cmap is None:
            cmap = self.data_obj.cmap
        if not kwargs:
            kwargs = self.parent.plot_kwargs
            if 'vmin' in kwargs.keys():
                kwargs['vmin'] = []
            if 'vmax' in kwargs.keys():
                kwargs['vmax'] = []
        try:
            im = pcolor(ax, self.data_obj.wavenumbers[self.xinds],
                        self.time_points,
                        self.fit_plot_data[i],
                        cmap=cmap, **kwargs)
        except Exception as e:
            messagebox.showerror("", str(e), parent=self)
            raise
            im = None
        return im

    def save_para_separately(self):
        para = {}
        para['amp'] = [TATrace(trace_type='Kinetic',
                               xdata=self.para_traces.xdata,
                               xlabel=self.main_figure.get_ylabel()),
                       'Amplitude']
        para['pos'] = [TATrace(trace_type='Kinetic',
                               xdata=self.para_traces.xdata,
                               ylabel=self.main_figure.get_xlabel(),
                               xlabel=self.main_figure.get_ylabel()),
                       'Position']
        if re.search('gauss', self.current_line_shape, re.I):
            para['sig'] = [TATrace(trace_type='Kinetic',
                                   xdata=self.para_traces.xdata,
                                   ylabel=self.main_figure.get_xlabel(),
                                   xlabel=self.main_figure.get_ylabel()),
                           'Sigma']
        else:
            para['gamma'] = [TATrace(
                trace_type='Kinetic',
                xdata=self.para_traces.xdata,
                ylabel=self.main_figure.get_xlabel(),
                xlabel=self.main_figure.get_ylabel()),
                'Gamma']
        if self.equal_space.get():
            para['cin'] = [TATrace(trace_type='Kinetic',
                                   xdata=self.para_traces.xdata,
                                   ylabel=self.main_figure.get_xlabel(),
                                   xlabel=self.main_figure.get_ylabel()),
                           'Spacing']
        if self.offset_opt.get():
            para['set'] = [TATrace(trace_type='Kinetic',
                                   xdata=self.para_traces.xdata,
                                   xlabel=self.main_figure.get_ylabel()),
                           'Offset']
        for key in self.para_traces.active_traces:
            for k, val in para.items():
                if key[3:6] == k:
                    val[0].tr[key] = {'y': self.para_traces.tr[key]['y']}
                    # val[0].ylabel = self.paraPlotDict[key].get_ylabel()
                    val[0].active_traces.append(key)
                    break
        path = filedialog.askdirectory()
        if len(path) > 0:
            for val in para.values():
                fname = path + '/lineshape_fit_parameters_' + val[1] + '.txt'
                val[0].save_traces(fname, save_fit=False,
                                   trace_keys='all', parent=self)

    def save_para_trace(self, key):
        file = save_box(filetypes=[('text files', '.txt'),
                                   ('Matlab files', '.mat')],
                        fext='.txt',
                        parent=self)
        if file is not None:
            return self.para_traces.save_traces(
                file, trace_key=[key], save_fit=self.para_traces.fit_done)
        else:
            return None
# to be tested

    def para_fig_window_callback(self, event, numrows, plot_func,
                                 include_fit=False):
        if event.dblclick:
            try:
                clickpos = event.inaxes.get_position().extents
            except Exception:
                return
            else:
                col = 1
                row = 1
                while col < 3:
                    if col/3 >= clickpos[0]:
                        break
                    else:
                        col = col + 1
                while row < numrows:
                    if clickpos[1] > (numrows-row)/numrows:
                        break
                    row = row + 1
                button_dict = {'Save data': (
                    lambda: self.save_para_trace(
                        self.para_traces.active_traces[
                            self.para_plot_ax_pos[col][row]]))}
                win = self.controller.open_topfigure_wrap(
                    self, plot_func=lambda *args, **kwargs:
                        plot_func(self.para_plot_ax_pos[col][row],
                                  **kwargs),
                    plot_type='line',
                    editable=True,
                    controller=self,
                    button_dict=button_dict,
                    include_fit=include_fit,
                    parent=self)
                win.fr.opts.plot_opts.add_plot_ci_option()


# %%
class ModifyData(tk.Toplevel):
    def __init__(self, parent, data_obj, controller=None):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent
        self.data_obj = data_obj
        self.modify_unit_panel = GroupBox(self, dim=(2, 3), text="Modify unit")
        self.modify_unit_panel.grid(row=0, column=0,
                                    sticky='wnse')
        tk.Label(self.modify_unit_panel, text=self.data_obj.get_x_mode()).grid(
            row=0, column=0, sticky='w')
        tk.Label(self.modify_unit_panel, text='time delays').grid(
            row=1, column=0, sticky='w')
        tk.Label(self.modify_unit_panel, text='del. Abs.').grid(
            row=2, column=0, sticky='w')
        self.x_unit = tk.StringVar(value=self.data_obj.spec_unit)
        self.x_unit_select = tk.ttk.OptionMenu(
            self.modify_unit_panel, self.x_unit, self.data_obj.spec_unit,
            *self.data_obj.x_unit_dict[self.data_obj.get_x_mode()].keys(),
            command=self.change_x_unit)
        self.x_unit_select.config(width=10)
        self.x_unit_select.grid(row=0, column=1, sticky='we')

        self.time_unit = tk.StringVar(value=self.data_obj.time_unit)
        self.time_unit_select = tk.ttk.OptionMenu(
            self.modify_unit_panel, self.time_unit, self.data_obj.time_unit,
            *self.data_obj.time_unit_factors.keys(),
            command=self.change_time_unit)
        self.time_unit_select.config(width=10)
        self.time_unit_select.grid(row=1, column=1, sticky='we')

        self.delA_unit = tk.StringVar(value=self.data_obj.delA_unit)
        self.delA_unit_select = tk.ttk.OptionMenu(
            self.modify_unit_panel, self.delA_unit, self.data_obj.delA_unit,
            *self.data_obj.delA_unit_dict.keys(),
            command=self.change_delA_unit)
        self.delA_unit_select.config(width=10)
        self.delA_unit_select.grid(row=2, column=1, sticky='we')

        self.scale_data_panel = GroupBox(self, dim=(2, 3), text="Scale by")
        self.scale_data_panel.grid(row=0, column=1, sticky='wnse')
        tk.Label(self.scale_data_panel, text=self.data_obj.get_x_mode()).grid(
            row=0, column=0, sticky='w')
        tk.Label(self.scale_data_panel, text='time delays').grid(
            row=1, column=0, sticky='w')
        tk.Label(self.scale_data_panel, text='del. Abs.').grid(
            row=2, column=0, sticky='w')
        self.x_scale = tk.DoubleVar(value=1)
        self.x_scale_entry = tk.ttk.Entry(
            self.scale_data_panel, textvariable=self.x_scale, width=10)
        self.x_scale_entry.bind('<Return>', self.change_x_unit)
        self.x_scale_entry.grid(row=0, column=1, sticky='we')

        self.time_scale = tk.DoubleVar(value=1)
        self.time_scale_entry = tk.ttk.Entry(
            self.scale_data_panel, textvariable=self.time_scale, width=10)
        self.time_scale_entry.bind('<Return>', self.change_time_unit)
        self.time_scale_entry.grid(row=1, column=1, sticky='we')

        self.delA_scale = tk.DoubleVar(value=1)
        self.delA_scale_entry = tk.ttk.Entry(
            self.scale_data_panel, textvariable=self.delA_scale, width=10)
        self.delA_scale_entry.bind('<Return>', self.change_delA_unit)
        self.delA_scale_entry.grid(row=2, column=1, sticky='we')

        self.cutout_region_panel = GroupBox(
            self, dim=(3, 3), text="Cut region")
        self.cut_x_lower = tk.DoubleVar()
        self.cut_x_upper = tk.DoubleVar()
        tk.ttk.Label(self.cutout_region_panel,
                     text="x window:").grid(row=0, column=0)
        self.cut_x_lower_entry = tk.ttk.Entry(
            self.cutout_region_panel, textvariable=self.cut_x_lower, width=5)
        self.cut_x_lower_entry.grid(row=0, column=1)
        self.cut_x_upper_entry = tk.ttk.Entry(
            self.cutout_region_panel, textvariable=self.cut_x_upper, width=5)
        self.cut_x_upper_entry.grid(row=0, column=2)
        self.cut_x_lower_entry.bind("<Return>", self.cut_x_region)
        self.cut_x_upper_entry.bind("<Return>", self.cut_x_region)

        self.cut_y_lower = tk.DoubleVar()
        self.cut_y_upper = tk.DoubleVar()
        tk.ttk.Label(self.cutout_region_panel,
                     text="y window:").grid(row=1, column=0, pady=5)
        self.cut_y_lower_entry = tk.ttk.Entry(
            self.cutout_region_panel, textvariable=self.cut_y_lower, width=5)
        self.cut_y_lower_entry.grid(row=1, column=1, pady=5)
        self.cut_y_upper_entry = tk.ttk.Entry(
            self.cutout_region_panel, textvariable=self.cut_y_upper, width=5)
        self.cut_y_upper_entry.grid(row=1, column=2, pady=5)
        self.cut_y_lower_entry.bind("<Return>", self.cut_y_region)
        self.cut_y_upper_entry.bind("<Return>", self.cut_y_region)

        tk.ttk.Button(self.cutout_region_panel, text="Reset",
                      command=self.reset_region_cut).grid(
                          row=0, column=3, rowspan=2, padx=5, pady=5)
        self.cutout_region_panel.grid(row=1, column=0, columnspan=2,
                                      sticky='w')

        self.protocol("WM_DELETE_WINDOW", self.close_window)

        tk.ttk.Button(self,
                      text="OK",
                      command=lambda: self.close_window(ok=True)).grid(
                          row=2, column=0, columnspan=2)
        self.x_factor = 1
        self.time_factor = 1
        self.delA_factor = 1
        if controller is not None:
            center_toplevel(self, controller)

    def close_window(self, ok=False):
        if ok:
            self.change_x_unit(update_plot=False)
            self.change_time_unit(update_plot=False)
            self.change_delA_unit(update_plot=True)
            try:
                self.parent.update_trace_spec_axis()
            except Exception:
                pass
        self.destroy()

    def cut_y_region(self, *args):
        self.data_obj.cutout_region(y=[self.cut_y_lower.get(),
                                       self.cut_y_upper.get()])
        self.parent.update_main_plot()

    def cut_x_region(self, *args):
        self.data_obj.cutout_region(x=[self.cut_x_lower.get(),
                                       self.cut_x_upper.get()])
        self.parent.update_main_plot()

    def reset_region_cut(self, *args):
        self.data_obj.reset_cutout()
        self.parent.update_main_plot()

    def change_x_unit(self, *args, update_plot=True):
        factor = self.data_obj.modify_x(
            unit=self.x_unit.get(), factor=self.x_scale.get() / self.x_factor)
        self.x_factor = self.x_scale.get()
        try:
            self.parent.vars['lambda_lim_low'].set(
                factor * self.parent.vars['lambda_lim_low'].get())
            self.parent.vars['lambda_lim_up'].set(
                factor * self.parent.vars['lambda_lim_up'].get())
            self.parent.main_figure.set_xlim(
                [self.parent.vars['lambda_lim_low'].get(),
                 self.parent.vars['lambda_lim_up'].get()])
            self.parent.main_figure.xlabels = [self.data_obj.xlabel]
            self.parent.update_main_plot()
        except Exception:
            raise

    def change_time_unit(self, *args, update_plot=True):
        factor = self.data_obj.modify_time(unit=self.time_unit.get(
        ), factor=self.time_scale.get() / self.time_factor)
        self.time_factor = self.time_scale.get()
        try:
            self.parent.vars['tupper'].set(
                factor * self.parent.vars['tupper'].get())
            self.parent.vars['tlower'].set(
                factor * self.parent.vars['tlower'].get())
            self.parent.main_figure.set_ylim(
                [self.parent.vars['tlower'].get(),
                 self.parent.vars['tupper'].get()])
            self.parent.main_figure.ylabels = [
                'time delays (' + self.data_obj.time_unit + ')']
            self.parent.update_main_plot()
        except Exception:
            raise

    def change_delA_unit(self, *args, update_plot=True):
        factor = self.data_obj.modify_delA(unit=self.delA_unit.get(
        ), factor=self.delA_scale.get() / self.delA_factor)
        self.delA_factor = self.delA_scale.get()
        try:
            self.parent.frames['plot_color'].vars['lower_clim'].set(
                factor * self.parent.frames[
                    'plot_color'].vars['lower_clim'].get())
            self.parent.frames['plot_color'].vars['upper_clim'].set(
                factor * self.parent.frames[
                    'plot_color'].vars['upper_clim'].get())
            self.parent.frames['plot_color'].clim_callback()
            self.parent.main_figure.clabels = [self.data_obj.zlabel]
            self.parent.update_main_plot()
        except Exception:
            raise


# %%
# class FitGuessWindow(tk.Toplevel):
#     def __init__(self, parent, inputDict = None, controller = None,
#                  bounds = False, title = None):
#         tk.Toplevel.__init__(self, parent)
#         self.parent = parent
#         if controller is None:
#             self.controller = parent
#         if inputDict is None:
#             self.dict = {'labelheaders': ['Trace','Component'],
#                         'parameters':['amplitude','time','offset'],
#                          'singularPara':['offset'],
#                          'para_tracesanslation':{'offset': 'Offset', 'amplitude': 'Amplitude',
#                                             'time': 'Time constant (ps)'},
#                          'curves':{'500 nm': {'Exp 1':[0,1,4],'Exp 2':[2,3]}}}
#         else:
#             self.dict = inputDict
#         row = 0
#         self.flag = False
#         self.valueDict = {}
#         self.firstEntryColumn = 0
#         self.entryWidth = 8
#         self.frame = tk.Frame(self)
#         self.placeWidgetsRecursive(self.dict['curves'], self.valueDict)
#         for column,header in enumerate(self.dict['labelheaders']):
#             tk.Label(self, text = header).grid(row = row, column = column,
#                  sticky = tk.W, padx = 5)
#         for col,para in enumerate(self.dict['parameters']):
#             tk.Label(self, text = self.dict['para_tracesanslation'][para]).grid(row = row, column = column + col + 1,
#                  sticky = tk.W, padx = 5)
#         row = row + 1
#         self.frame.grid(row = row, column = 0, columnspan = column + col + 2, sticky = 'wnse')
#         row = row + 1
#         self.button_frame = tk.Frame(self)

#         ttk.Button(self.button_frame, text = "OK", command = self.okButtonCallback).grid(
#                 row = 0, column = 0)
#         ttk.Button(self.button_frame, text = "Cancel", command = self.destroy).grid(
#                 row = 0, column = 1)
#         self.button_frame.grid(row = row,
#                   column = 0, columnspan = column + col + 1)


#     def okButtonCallback(self):
#         for key, val in self.valueDict.items():
#             singularParas = {}
#             self.dict['curves'][key] = self.getParameters(val, self.dict['curves'][key], singularParas)
#             for k, v in singularParas.items():
#                 self.dict['curves'][key][k] = v
#         self.flag = True
#         self.destroy()

#     def getParameters(self, dct, writeDct, singularParas):
#         writeDct = {}
#         for key,val in dct.items():
#             writeDct[key] = {}
#             try:
#                 val.keys()
#             except Exception:
#                 continue
#             else:
#                 flag = False
#                 for k in val.keys():
#                     if k in self.dict['singularPara']:
#                         singularParas[k] = val[k].get()
#                         flag = True
#                     elif k in self.dict['parameters']:
#                         writeDct[key][k] = val[k].get()
#                         flag = True
#                 if not flag:
#                     writeDct[key] = self.getParameters(val, writeDct[key], singularParas)
#         return writeDct


#     def placeWidgetsRecursive(self, dct, writeDct, row = 1, column = 0):
#         for key, val in dct.items():
#             tk.Label(self.frame, text = key).grid(row = row, column = column,
#                  sticky = tk.W, padx = 5)
#             writeDct[key] = {}
#             try:
#                 val.keys()
#             except Exception:
#                 self.firstEntryColumn = column
#                 for i, lb in enumerate(val):
#                     try:
#                         para = self.dict['parameters'][i]
#                     except Exception: pass
#                     else:
#                         self.placeEntry(para, lb, writeDct[key], row = row, column = column + i + 1,
#                                         sticky = tk.W, padx = 5)
#                 row += 1
#             else:
#                 row, column = self.placeWidgetsRecursive(val, writeDct[key], row, column + 1)
#         return row, column - 1


#     def placeEntry(self, para, val, writeDct, **grid_kwargs):
#         writeDct[para] = tk.DoubleVar(value = val)
#         tk.Entry(self.frame, textvariable = writeDct[para], width = self.entryWidth).grid(
#                 **grid_kwargs)


# %%         Deprecated, to be removed
# class ExpFitGuessWindow(tk.Toplevel, tk.Frame):
#     def __init__(self, parent, controller, num_comp, paraNames, paraVals,
#                  mode = 'enter', offset = True, bounds = False,
#                  popup = True, headers = None, title = None, customShape = False,
#                  timeZeroPara = False):
#         if popup:
#             tk.Toplevel.__init__(self, parent)
#             move_toplevel_to_default(self, controller)
#         else:
#             tk.Frame.__init__(self, parent)
#         if headers == None:
#             self.headers = ["Parameter","Amplitude","Lifetime (" + self.data_obj.time_unit + ")"]
#             if offset:
#                 self.headers.append("Offset")
#             if timeZeroPara:
#                 self.headers.append("Time zero")
#         else:
#             self.headers = headers
#         self.headerLabels = []
#         self.bounds = bounds
#         self.offset = offset
#         self.timeZeroPara = timeZeroPara
#         if title:
#             self.title(title)
#         if bounds:
#             for header,i in zip(self.headers, (0,*range(1,2*(2 + int(offset) + int(timeZeroPara)),2))):
#                 tk.Label(self, text = header).grid(row = 0, column = i, pady = 5, padx = 5, sticky = tk.W)
#             for i in range(2 + int(offset)):
#                 tk.Label(self, text = 'Lower').grid(row = 1, column = 2*i + 1,
#                      pady = 5, padx = 5, sticky = tk.W)
#                 tk.Label(self, text = 'Upper').grid(row = 1, column = 2*i + 2,
#                      pady = 5, padx = 5, sticky = tk.W)
#         else:
#             for header,i in zip(self.headers, range(len(self.headers))):
#                 self.headerLabels.append(tk.Label(self, text = header))
#                 self.headerLabels[-1].grid(row = 0, column = i, pady = 5, padx = 5, sticky = tk.W)
#         if mode == 'enter':
#             self.entries = {}
#             nrow = self.placeEntries(paraNames,paraVals, num_comp, offset, bounds)
#             self.paraNames = paraNames
#             self.paraVals = paraVals
#             if popup:
#                 if bounds:
#                     self.title("Enter Parameter Bounds")
#                 else:
#                     self.title("Enter Guesses")
#                 self.continueButton = ttk.Button(self, text = "Continue", command =
#                        lambda: self.continueFit(num_comp, offset, bounds, customShape))
#                 self.continueButton.grid(
#                     row = nrow, column = 0, columnspan = (2 + int(offset) + int(
#                             timeZeroPara))*(1+int(bounds)) + 1)
#         else:
#             self.placeLabels(paraNames, paraVals, num_comp, offset)
#             if popup:
#                 self.title("Fit Results")
#         if popup:
#             center_toplevel(self,controller)


#     def updateHeaderLabels(self):
#         for l in self.headerLabels:
#             l.grid_forget()
#         self.headerLabels = []
#         iterator = (0,*range(1,2*(2 + int(self.offset)),2)) if self.bounds else range(3 + int(self.offset))
#         for header, i in zip(self.headers, iterator):
#             self.headerLabels.append(tk.Label(self, text = header))
#             self.headerLabels[-1].grid(row = 0, column = i, pady = 5, padx = 5, sticky = tk.W)


#     def placeEntries(self, names, vals, num_comp, offset, bounds):
#         def placeForParameter(i, m):
#             def placeEntry(k,l,key):
#                 self.entries[key] = [tk.DoubleVar()]
#                 try:
#                     self.entries[key][0].set(vals[i][l])
#                 except Exception: pass
#                 self.entries[key].append(tk.Entry(self,
#                     textvariable = self.entries[key][0], width = 10))
#                 self.entries[key][1].grid(row = m, column = k,
#                             padx = 5, pady = 5, sticky = tk.W)
#             tk.Label(self, text = names[i]).grid(row = m, column = 0, padx = 5, sticky = tk.W)
#             l = 0
#             if bounds:
#                 if offset:
#                     placeEntry(5,-2-2*int(self.timeZeroPara), names[i] + "Offset" + "lower")
#                     placeEntry(6,-1-2*int(self.timeZeroPara), names[i] + "Offset" + "upper")
#                 if self.timeZeroPara:
#                     placeEntry(7,-2, names[i] + "TimeZero" + "lower")
#                     placeEntry(8,-1, names[i] + "TimeZero" + "upper")
#                 for j in range(num_comp):
#                     for k in (1,2):
#                         placeEntry(2*k - 1,l,names[i] + "_Exp" + str(j) + "_" + self.headers[k] + "lower")
#                         l = l + 1
#                         placeEntry(2*k,l,names[i] + "_Exp" + str(j) + "_" + self.headers[k] + "upper")
#                         l = l + 1
#                     m = m + 1
#             else:
#                 if offset:
#                     placeEntry(len(self.headers),-1-int(self.timeZeroPara), names[i] + "Offset")
#                 if self.timeZeroPara:
#                     placeEntry(len(self.headers),-1, names[i] + "TimeZero")
#                 for j in range(num_comp):
#                     for k in (1,2):
#                         try:
#                             placeEntry(k,l,names[i] + "_Exp" + str(j) + "_" + self.headers[k])
#                             l = l + 1
#                         except Exception: pass
#                     m = m + 1
#             return m

#         later = []
#         m = 2 if bounds else 1
#         for i in range(len(names)):
#             try:
#                 np.double(names[i][0])
#             except Exception:
#                 later.append(i)
#             else:
#                 m = placeForParameter(i,m)
#         for i in later:
#             m = placeForParameter(i,m)
#         return m

#     def placeLabels(self, names, vals, num_comp, offset):
#         def placeForParameter(i, m):
#             tk.Label(self, text = names[i]).grid(row = m, column = 0, padx = 5, sticky = tk.W)
#             l = 0
#             if offset:
#                 tk.Label(self, text = vals[i][-1]).grid(row = m, column = 3,
#                      padx = 5, pady = 5, sticky = tk.W)
#             for j in range(num_comp):
#                 for k in (1,2):
#                     tk.Label(self, text = vals[i][l]).grid(row = m, column = k,
#                          padx = 5, pady = 5, sticky = tk.W)
#                     l = l + 1
#                 m = m + 1
#             return m
#         later = []
#         m = 1
#         for i in range(len(names)):
#             try:
#                 np.double(names[i][0])
#             except Exception:
#                 later.append(i)
#             else:
#                 m = placeForParameter(i,m)
#         for i in later:
#             m = placeForParameter(i,m)

#     def continueFit(self, num_comp, offset, bounds, customShape):
#         self.output = []
#         if customShape:
#             self.output = {}
#             for k in self.entries.keys():
#                 self.output[k] = self.entries[k][0].get()
#         elif bounds:
#             for name in self.paraNames:
#                 out = []
#                 for i in range(num_comp):
#                     for k in (1,2):
#                         out.append(self.entries[
#                                 name + "_Exp" + str(i) + "_"
#                                 + self.headers[k] + "lower"][0].get())
#                         out.append(self.entries[
#                                 name + "_Exp" + str(i) + "_"
#                                 + self.headers[k] + "upper"][0].get())
#                 if offset:
#                     out.append(self.entries[name + "Offset" + "lower"][0].get())
#                     out.append(self.entries[name + "Offset" + "upper"][0].get())
#                 if self.timeZeroPara:
#                     out.append(self.entries[name + "TimeZero" + "lower"][0].get())
#                     out.append(self.entries[name + "TimeZero" + "upper"][0].get())
#                 self.output.append(out)
#         else:
#             for name in self.paraNames:
#                 out = []
#                 for i in range(num_comp):
#                     for k in (1,2):
#                         try:
#                             out.append(self.entries[
#                                 name + "_Exp" + str(i) + "_" + self.headers[k]][0].get())
#                         except Exception: pass
#                 if offset:
#                     out.append(self.entries[name + "Offset"][0].get())
#                 self.output.append(out)
#         self.destroy()


# %%
# class FitResultsDisplay(tk.Toplevel):
#     def __init__(self, parent, entry_dict, controller=None, headers=[],
#                  title="Fit Results"):
#         tk.Toplevel.__init__(self, parent)
#         if controller is not None:
#             move_toplevel_to_default(self, controller)
#         self.title(title)
#         tk.Label(self, text=title).grid(row=0, column=0, pady=5)
#         self.frame = tk.Frame(self)

#         for i, header in enumerate(headers):
#             tk.Label(self.frame, text=header).grid(row=0, column=i,
#                                                    sticky=tk.W, padx=5)

#         self.placeLabelsRecursive(entry_dict, row=1, column=0)

#         self.frame.grid(row=1, column=0)
#         ttk.Button(self, text="OK", command=self.destroy).grid(
#             row=2, column=0)
#         if controller is not None:
#             center_toplevel(self, controller)

#     def placeLabelsRecursive(self, entry_dict, row=1, column=0):
#         for key, val in entry_dict.items():
#             tk.Label(self.frame, text=key).grid(row=row, column=column,
#                                                 sticky=tk.W, padx=5)
#             try:
#                 val.keys()
#             except Exception:
#                 for i, lb in enumerate(val):
#                     tk.Label(self.frame, text=str(lb)).grid(
#                         row=row, column=column + i + 1, sticky=tk.W, padx=5)
#                 row += 1
#             else:
#                 row, column = self.placeLabelsRecursive(val, row, column + 1)
#         return row, column - 1


# %%
# class GlobalSettingsWindow(tk.Toplevel):
#     # work in progress, meant for interactive manipulation of the
#     # global settings file. For now direct editing of the .txt is
#     # required
#     def __init__(self, parent, controller, update_function=None):
#         return
