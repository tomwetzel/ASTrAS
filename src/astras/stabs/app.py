# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:43:26 2019

@author: bittmans
"""
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
# from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox

# from customplots import *
from ..common.tk.figures import open_topfigure, tk_mpl_plot_function
from ..common.helpers import GlobalSettings
from ..common.tk.general import (CustomFrame, CustomEntry, GroupBox,
                                 center_toplevel, save_box)
from ..common.dataobjects import TATraceCollection, TATrace, StaticAbsData
from scipy.signal import savgol_filter
# from scipy.ndimage import gaussian_filter

import os

import re

import numpy as np

import matplotlib as mpl
mpl.use("Agg")


def savgol_filter_wrap(dat, order=5, framelength=3):
    return savgol_filter(dat, order, framelength)


class AppMain(tk.Tk):
    def __init__(self, *args, parent=None, **kwargs):
        if parent is None:
            tk.Tk.__init__(self, *args, **kwargs)
        else:
            tk.Toplevel.__init__(self, parent, *args, **kwargs)
        container = CustomFrame(self, dim=(2, 2), border=False)
        container.grid(padx=20, pady=20)
        # read from config file
        self.settings = GlobalSettings(config_path='config.txt')
        plt.style.use(self.settings['plot_style'])
        for key, value in self.settings['rcParams'].items():
            mpl.rcParams[key] = value
        self.frames = {}
        self.entries = {}
        self.vars = {}
        self.buttons = {}
        self.optmenus = {}
        self.widgets = {}
        self.labels = {}
        self.data = StaticAbsData()
        self.num_dens = 1.0e21
        self.sample_length = 100

        self.title("ASTrAS - Steady-State Absorption Analysis")

        self.menubar = tk.Menu(container)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.calcmenu = tk.Menu(self.menubar, tearoff=0)
        self.savemenu = tk.Menu(self.menubar, tearoff=0)

        # filemenu.add_separator()
        self.menubar.add_cascade(label="Load spectrum", menu=self.filemenu)
        self.menubar.add_cascade(label="Calculate", menu=self.calcmenu)
        self.menubar.add_cascade(label="Save", menu=self.savemenu)

        self.filemenu.add_command(label="Absorption (single)", command=lambda:
                                  self.load_single(case='abs'))
        self.filemenu.add_command(
            label="Absorption (multiple)",
            command=lambda: self.load_multi(case='abs'))
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Intensity", command=lambda:
                                  self.load_single(case='int'))
        self.filemenu.add_command(label="Reference", command=lambda:
                                  self.load_single(case='ref'))
        self.filemenu.add_command(label="Background", command=lambda:
                                  self.load_single(case='bkg'))

        self.calcmenu.add_command(
            label='Absorption', command=self.calc_abs_spec)
        self.calcmenu.add_command(
            label="Difference Abs.", command=self.calc_diff_abs)

        self.savemenu.add_command(
            label='Active', command=lambda c='active': self.save(case=c))
        self.savemenu.add_command(
            label='All', command=lambda c='all': self.save(case=c))
        self.savemenu.add_command(label='Traces (all)', command=lambda c='all':
                                  self.save_as_trace(case=c))

        tk.Tk.config(self, menu=self.menubar)

        self.frames['toolbar'] = CustomFrame(
            container, dim=(1, 1), border=False)

        self.figure = plt.figure()
        plt.close()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, container)

        self.set_fontsize()
        self.canvas.get_tk_widget().config(width=500, height=400)
        self.toolbar = NavigationToolbar2Tk(
            self.canvas, self.frames['toolbar'])
        self.toolbar.update()
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(sticky='wnse')
        self.canvas._tkcanvas.grid(row=0, column=0, pady=5)
        self.frames['toolbar'].grid(row=1, column=0, sticky='w')
        self.canvas.callbacks.connect(
            'button_press_event', self.canvas_callback)

        self.frames['panels'] = CustomFrame(
            container, dim=(1, 2), border=False)
        self.frames['calculation'] = CustomFrame(
            self.frames['panels'], dim=(3, 8), border=True)
        self.frames['results'] = CustomFrame(
            self.frames['panels'], dim=(1, 1), border=True)
        self.frames['sample_type'] = CustomFrame(
            self.frames['panels'], dim=(2, 1), border=False)

        self.vars['sample_type'] = tk.StringVar(value='Solid')
        tk.ttk.Label(self.frames['sample_type'], text='Mode:').grid(
            row=0, column=0, sticky='w')
        self.optmenus['sample_type'] = tk.ttk.OptionMenu(
            self.frames['sample_type'], self.vars['sample_type'], 'Solid',
            'Solution', command=self.set_sample_type)
        self.optmenus['sample_type'].grid(row=0, column=1, sticky='w')

        # parameter entries crystal
        self.vars['od'] = tk.DoubleVar(value=1.0)
        self.vars['spec_point'] = tk.DoubleVar(value=400)
        self.vars['spec_unit'] = tk.StringVar(value='nm')
        self.vars['num_dens'] = tk.DoubleVar(value=self.num_dens)
        self.vars['thickness'] = tk.DoubleVar(value=self.sample_length)
        self.vars['length_unit'] = tk.StringVar(value='nm')
        self.vars['num_dens_calc_mode'] = tk.StringVar(value='Mass')
        self.vars['fluence'] = tk.DoubleVar(value=2.0)
#        self.vars['excEnergy'] = tk.DoubleVar(value = 400.0)
        self.vars['exc_spec_unit'] = tk.StringVar(value='nm')
        self.vars['baseline_value'] = tk.DoubleVar(value=0.0)
        for entrykey in ('od', 'num_dens', 'thickness', 'fluence',
                         'spec_point'):
            self.entries[entrykey] = CustomEntry(
                self.frames['calculation'], textvariable=self.vars[entrykey],
                width=13, justify=tk.RIGHT)
        self.entries['spec_point'].entry.bind(
            '<Return>', self.get_od_at_lambda)
        self.labels['spec_unit'] = tk.ttk.Label(
            self.frames['calculation'], text='wavelength:')
        self.labels['spec_unit'].grid(row=0, column=0)
        self.entries['spec_point'].grid(row=0, column=1, sticky='we')
        tk.ttk.Label(self.frames['calculation'], text='od:').grid(
            row=1, column=0)
        self.entries['od'].grid(row=1, column=1)
        self.optmenus['spec_unit'] = tk.ttk.OptionMenu(
            self.frames['calculation'], self.vars['spec_unit'], 'nm')
        self.optmenus['spec_unit'].grid(row=0, column=2, sticky='we')
        tk.ttk.Label(self.frames['calculation'],
                     text='Pathlength:').grid(row=2, column=0)
        self.entries['thickness'].grid(row=2, column=1, sticky='we')
        self.optmenus['length_unit'] = tk.ttk.OptionMenu(
            self.frames['calculation'], self.vars['length_unit'], 'nm', 'mm')
        self.optmenus['length_unit'].grid(row=2, column=2, sticky='we')
        self.labels['num_dens_label'] = tk.ttk.Label(
            self.frames['calculation'], text='Number Density:')
        self.labels['num_dens_label'].grid(row=3, column=0)
        self.entries['num_dens'].grid(row=3, column=1)
        self.labels['num_dens_unit'] = tk.ttk.Label(
            self.frames['calculation'], text="cm^-3")
        self.labels['num_dens_unit'].grid(row=3, column=2)

        self.buttons['calc_num_dens'] = tk.ttk.Button(
            self.frames['calculation'], text='Calculate from:',
            command=self.open_num_dens_calc)
        self.buttons['calc_num_dens'].grid(row=4, column=1)
        self.optmenus['num_dens_calc_mode'] = tk.ttk.OptionMenu(
            self.frames['calculation'], self.vars['num_dens_calc_mode'],
            'Mass', 'Unit Cell')
        self.optmenus['num_dens_calc_mode'].config(width=10)
        self.optmenus['num_dens_calc_mode'].grid(row=4, column=2)

        tk.ttk.Label(self.frames['calculation'],
                     text='Fluence (mJcm^-2):').grid(row=6, column=0)
        self.entries['fluence'].grid(row=6, column=1)
        self.entries['fluence'].bind('<Return>', self.calc_results)

        tk.ttk.Button(
            self.frames['calculation'], text="Calculate",
            command=self.calc_results).grid(row=7, column=1)

        tk.ttk.Button(
            self.frames['calculation'], text="Plot",
            command=self.plot_exc_frac).grid(row=7, column=2)

        self.result_text = [tk.ttk.Label(self.frames['results'],
                                         justify=tk.LEFT),
                            tk.ttk.Label(self.frames['results'],
                                         justify=tk.LEFT)]
        self.result_text[0].grid(row=0, column=0, sticky='wn')
        self.result_text[1].grid(row=0, column=1, sticky='en')

        self.frames['sample_type'].grid(row=0, column=0, sticky='nwse')

        self.frames['calculation'].grid(row=1, column=0, sticky='wsne', padx=5,
                                        pady=5)
        self.frames['results'].grid(row=2, column=0, sticky='wnse', padx=5,
                                    pady=5)
        self.frames['panels'].grid(row=0, column=1, rowspan=2, sticky='wns')

        self.frames['spec_ops'] = CustomFrame(container, border=True,
                                              dim=(5, 5))
        self.buttons['plot'] = tk.ttk.Button(self.frames['spec_ops'],
                                             text="Plot",
                                             command=self.plot)
        self.buttons['plot'].grid(row=0, column=0)
        self.vars['plot_mode'] = tk.StringVar(value='Absorption')
        self.optmenus['plot_mode'] = tk.ttk.OptionMenu(
            self.frames['spec_ops'], self.vars['plot_mode'],
            *('Absorption', 'Intensity', 'Reference', 'Background'),
            command=self.plot)
        self.optmenus['plot_mode'].config(width=10)
        self.optmenus['plot_mode'].grid(row=0, column=1, sticky='w')
        self.vars['plot_which'] = tk.StringVar(value='All spectra')
        self.optmenus['plot_which'] = tk.ttk.OptionMenu(
            self.frames['spec_ops'], self.vars['plot_which'],
            'All spectra', 'Active spectrum')
        self.optmenus['plot_which'].grid(row=0, column=2, sticky='w')
        self.vars['overlay_plot'] = tk.IntVar(value=1)
        self.widgets['overlay_plot'] = tk.ttk.Checkbutton(
            self.frames['spec_ops'], variable=self.vars['overlay_plot'],
            command=self.plot, text='Overlay plots')
        self.widgets['overlay_plot'].grid(row=0, column=3, sticky='w')
        tk.ttk.Button(
            self.frames['spec_ops'], text="Clear figure",
            command=self.clear_axes).grid(row=0, column=4, sticky='w')

        tk.ttk.Label(self.frames['spec_ops'],
                     text='Active spectrum: No.').grid(
                         row=1, column=0, sticky='w')
        self.vars['active_spec'] = tk.IntVar(value=1)
        self.optmenus['active_spec'] = tk.ttk.OptionMenu(
            self.frames['spec_ops'], self.vars['active_spec'],
            *list(range(1, 10)), command=self.set_active_spec)
        self.optmenus['active_spec'].grid(row=1, column=1, sticky='w')
        tk.ttk.Button(
            self.frames['spec_ops'], text='Subtract baseline',
            command=self.subtr_baseline).grid(row=2, column=0, sticky='w')
        self.vars['baseline_subtr_mode'] = tk.StringVar(value='Constant')
        self.optmenus['baseline_subtr_mode'] = tk.ttk.OptionMenu(
            self.frames['spec_ops'], self.vars['baseline_subtr_mode'],
            'Constant')
        self.optmenus['baseline_subtr_mode'].grid(row=2, column=1, sticky='w')
        self.entries['baseline_value'] = tk.ttk.Entry(
            self.frames['spec_ops'], textvariable=self.vars['baseline_value'],
            width=10, justify=tk.RIGHT)
        self.entries['baseline_value'].bind('<Return>', self.subtr_baseline)
        self.entries['baseline_value'].grid(row=2, column=2, sticky='w')
        tk.ttk.Label(self.frames['spec_ops'], text='Subtract from:').grid(
            row=2, column=3, sticky='w')
        self.vars['base_subtr_from'] = tk.StringVar(value='All spectra')
        self.optmenus['base_subtr_from'] = tk.ttk.OptionMenu(
            self.frames['spec_ops'], self.vars['base_subtr_from'],
            'All spectra', 'Active spectrum')
        self.optmenus['base_subtr_from'].config(width=14)
        self.optmenus['base_subtr_from'].grid(row=2, column=4, sticky='w')

        tk.ttk.Label(self.frames['spec_ops'],
                     text="Magnify (active):").grid(
                         row=3, column=0, sticky='w')
        self.vars['magnification'] = tk.DoubleVar(value=1)
        self.entries['magnification'] = tk.ttk.Entry(
            self.frames['spec_ops'],
            textvariable=self.vars['magnification'], width=5)
        self.entries['magnification'].grid(row=3, column=1, sticky='w')
        self.entries['magnification'].bind('<Return>', self.magnify_spec)

        tk.ttk.Button(self.frames['spec_ops'], text="Reset",
                      command=self.reset_magnif).grid(
                          row=3, column=2, sticky='w')

        tk.ttk.Label(self.frames['spec_ops'], text='Legend:').grid(
            row=4, column=0, sticky='w')
        self.vars['legend'] = tk.StringVar(value=[])
        self.entries['legend'] = tk.ttk.Entry(
            self.frames['spec_ops'], textvariable=self.vars['legend'],
            width=40)
        self.entries['legend'].bind('<Return>', self.set_legend)
        self.entries['legend'].grid(row=4, column=2, sticky='w', columnspan=3)
        self.vars['legend_filenames'] = tk.IntVar(value=1)
        self.widgets['legend_filenames'] = tk.ttk.Checkbutton(
            self.frames['spec_ops'],
            variable=self.vars['legend_filenames'],
            text='Use file names',
            command=self.legend_filenames_callback)
        self.widgets['legend_filenames'].grid(row=4, column=1, sticky='w')

        tk.ttk.Button(self.frames['spec_ops'], text="Filter",
                      command=self.filter).grid(row=5, column=0, sticky='w')
        tk.ttk.Label(self.frames['spec_ops'], text="Sigma:").grid(
            row=5, column=1, sticky='w')
        tk.ttk.Label(self.frames['spec_ops'], text="Order:").grid(
            row=5, column=3, sticky='w')
        self.vars['filt_sigma'] = tk.DoubleVar(value=1.0)
        self.vars['filt_order'] = tk.IntVar(value=0)
        tk.ttk.Entry(self.frames['spec_ops'],
                     textvariable=self.vars['filt_sigma'], width=5).grid(
                         row=5, column=2, sticky='w')
        tk.ttk.Entry(self.frames['spec_ops'],
                     textvariable=self.vars['filt_order'], width=5).grid(
                         row=5, column=4, sticky='w')
        tk.ttk.Button(self.frames['spec_ops'], text="Reset",
                      command=self.reset_filter).grid(row=5, column=5,
                                                      sticky='w')
        tk.ttk.Button(
            self.frames['spec_ops'], text='Clear all',
            command=self.clear_data).grid(row=6, column=0, sticky='w')

        tk.ttk.Button(
            self.frames['spec_ops'], text='Clear active spectrum',
            command=self.clear_active_spec).grid(row=6, column=1, sticky='w')

        tk.ttk.Button(self.frames['spec_ops'], text="Sort",
                      command=self.sort_spectra).grid(
                          row=7, column=0, sticky='w')
        self.frames['spec_ops'].grid(row=2, column=0, sticky='wnse',
                                     padx=5, pady=5)
        self.frames['spec_calc'] = GroupBox(
            container, text="Centroid and Integral", border=True, dim=(2, 4))
        self.frames['spec_calc_props'] = CustomFrame(
            self.frames['spec_calc'], border=False, dim=(2, 3))
        tk.ttk.Label(self.frames['spec_calc'],
                     text='Wavelength range:').grid(
                         row=0, column=0, columnspan=2)
        self.vars['spec_calc_range_lower'] = tk.DoubleVar()
        self.vars['spec_calc_range_upper'] = tk.DoubleVar()
        tk.ttk.Entry(self.frames['spec_calc'],
                     textvariable=self.vars['spec_calc_range_lower'],
                     width=5).grid(row=1, column=0, sticky='e', padx=2)
        tk.ttk.Entry(self.frames['spec_calc'],
                     textvariable=self.vars['spec_calc_range_upper'],
                     width=5).grid(row=1, column=1, sticky='w', padx=2)
        tk.ttk.Button(self.frames['spec_calc'],
                      text='Show',
                      command=self.calc_spec_props).grid(
                          row=2, column=0, columnspan=2)
        self.spec_prop_labels = {}
        for i, key in enumerate(['Centroid', 'Integral', 'Mean']):
            tk.ttk.Label(self.frames['spec_calc_props'], text=key).grid(
                row=i, column=0, sticky='w')
            self.spec_prop_labels[key] = tk.ttk.Label(
                self.frames['spec_calc_props'])
            self.spec_prop_labels[key].grid(row=i, column=1, sticky='w')

        self.frames['spec_calc_props'].grid(row=3, column=0, columnspan=2,
                                            sticky='wnse', padx=2, pady=2)
        self.frames['spec_calc'].grid(row=2, column=1, sticky='wnse',
                                      padx=5, pady=5)

        self.length_unit_dict = {'nm': [1e-9],
                                 'mm': [1e-3]}
        self.spec_ind = 0
        self.calc_results()

    # wrapper for using (custom) standard plot options for toplevel figures
    def open_figure(self, *args, **kwargs):
        if 'dim' not in kwargs.keys():
            kwargs['dim'] = self.settings['figure_std_size']
        for key in ('fit_kwargs', 'plot_style', 'xlabel_pos', 'ylabel_pos'):
            if key not in kwargs.keys():
                kwargs[key] = self.settings[key]
        return open_topfigure(*args, **kwargs)

    def save_as_trace(self, *args, case='all'):
        file = save_box(filetypes=[('text files', '.txt'),
                                   ('Matlab files', '.mat')], fext='.txt')
        try:
            fname = file.name
        except Exception:
            return
        self.data.save_abs_traces(path=fname, case=case)

    def calc_spec_props(self, *args):
        var = {}
        try:
            var['Centroid'] = self.data.calc_centroid(xrange=[self.vars[
                'spec_calc_range_lower'].get(), self.vars[
                'spec_calc_range_upper'].get()])
        except Exception:
            for lbl in self.spec_prop_labels.values():
                lbl.config(text="")
        else:
            var['Integral'], var['Mean'] = self.data.integrate_spectrum(
                xrange=[self.vars[
                    'spec_calc_range_lower'].get(), self.vars[
                    'spec_calc_range_upper'].get()])
            for key in var.keys():
                self.spec_prop_labels[key].config(text=var[key])

    def plot_exc_frac(self, *args):
        self.calc_results()
        slope = self.data.exc_frac / self.data.fluence
        x_max = 100/slope
        x = np.linspace(0, x_max, num=10)
        y = slope*x
        self.open_figure(self,
                         plot_func=lambda *args, ax=None, **kwargs:
                             ax.plot(x, y),
                         plot_type='line',
                         editable=True,
                         ylabels='Excitation Fraction (%)',
                         xlabels='Fluence (mJcm$^{-2}$)',
                         axes_grid_on=True, legends=False)

    def sort_spectra(self, *args):
        self.data.sort_spectra(mode="value")
        self.plot()

    def set_sample_type(self, row=1, column=0, sticky='wsne', padx=5,
                        pady=5):
        if self.vars['sample_type'].get() == 'Solid':
            self.buttons['calc_num_dens'].config(state='normal')
            self.optmenus['num_dens_calc_mode'].config(state='normal')
            self.labels['num_dens_label'].config(text='Number Density:')
            self.vars['length_unit'].set('nm')
            self.labels['num_dens_unit'].config(text='cm^-3')
            self.vars['num_dens'].set(1e21)
            self.vars['thickness'].set(100)

        elif self.vars['sample_type'].get() == 'Solution':
            self.buttons['calc_num_dens'].config(state='disabled')
            self.optmenus['num_dens_calc_mode'].config(state='disabled')
            self.labels['num_dens_label'].config(text='Concentration:')
            self.vars['length_unit'].set('mm')
            self.labels['num_dens_unit'].config(text='mmol/L')
            self.vars['num_dens'].set(1)
            self.vars['thickness'].set(1)

    def legend_filenames_callback(self, *args):
        if self.vars['legend_filenames'].get():
            self.entries['legend'].config(state='disabled')
            return self.set_legend(mode='file')
        else:
            self.entries['legend'].config(state='normal')
            return self.set_legend(mode='input')

    def set_active_spec(self, *args):
        self.data.set_active_spec(index=self.vars['active_spec'].get() - 1)
        self.get_od_at_lambda()
        try:
            self.calc_results()
        except Exception:
            pass
        if not self.vars['plot_which'].get().startswith('All'):
            self.axes.cla()
            self.plot()

    def set_fontsize(self, value=12):
        for item in ([self.axes.xaxis.label, self.axes.yaxis.label]
                     + self.axes.get_xticklabels() +
                     self.axes.get_yticklabels()):
            item.set_fontsize(value)
        self.canvas.draw()

    def clear_active_spec(self):
        self.data.clear_active_spec()
        self.vars['active_spec'].set(1)
        self.axes.cla()
        self.plot()

    def clear_data(self):
        self.data.clear_data()
        self.axes.cla()
        self.canvas.draw()

    def magnify_spec(self, *args):
        self.data.magnify_abs(factor=self.vars['magnification'].get(),
                              case='active')
        self.plot()

    def reset_magnif(self):
        self.data.reset_magnif()
        self.plot()

    def set_legend(self, *args, mode='input'):
        if mode == 'input':
            labels = (self.vars['legend'].get().split(","))
        elif mode == 'file':
            if self.vars['plot_which'].get().startswith('All'):
                labels = list(self.data.abs_spectra.keys())
            else:
                labels = [self.data.get_active_key()]
        else:
            return
        self.axes.legend(labels)
        self.canvas.draw()
        return labels

    def clear_axes(self):
        self.axes.cla()
        self.canvas.draw()

    def load_multi(self, case='abs', filetypes=['txt']):
        path = filedialog.askdirectory()
        try:
            files = [f for f in os.listdir(path) if re.search(filetypes[0], f)]
        except Exception:
            pass
        else:
            if case == 'abs':
                try:
                    for f in files:
                        self.data.load_abs_spectrum(path + '/' + f)
                except Exception:
                    raise
                else:
                    self.axes.cla()
                    self.plot()
                    self.vars['active_spec'].set(
                        len(list(self.data.abs_spectra.keys())))
                    self.get_od_at_lambda()

    def load_single(self, case='abs'):
        path = filedialog.askopenfilename()
        if case == 'abs':
            if not self.data.load_abs_spectrum(path):
                return
            self.axes.cla()
            self.plot()
            self.vars['active_spec'].set(
                len(list(self.data.abs_spectra.keys())))
            self.get_od_at_lambda()
            try:
                self.calc_results()
            except Exception:
                pass
        elif case == 'int':
            try:
                self.data.load_transmiss(path)
            except Exception:
                pass
            else:
                self.axes.plot(self.data.wavelengths, self.data.trans)
                self.canvas.draw()
        elif case == 'ref':
            try:
                self.data.load_ref_spec(path)
            except Exception:
                pass
            else:
                self.axes.plot(self.data.wavelengths, self.data.ref)
                self.canvas.draw()
        elif case == 'bkg':
            try:
                self.data.load_bkg_spectrum(path)
            except Exception:
                pass

    def save(self, case='active'):
        path = save_box(as_file=True)
        self.data.save(path.name, case=case)

    def canvas_callback(self, event):
        if event.dblclick or event.button == 3:
            legends = [txt.get_text()
                       for txt in self.axes.get_legend().get_texts()]
            self.open_figure(self, plot_func=self.plot_func, plot_type='line',
                             dim=[600, 400], editable=True,
                             legends=[legends],
                             ylabels=self.axes.get_ylabel(),
                             xlabels=self.axes.get_xlabel())

    def calc_diff_abs(self, *args):
        DifferenceAbsorption(self, self.data)

    def get_od_at_lambda(self, *args):
        spec_ind = self.spec_ind
        self.spec_ind = np.where(
            self.data.abs_spectra[self.data.get_active_key()][0, :]
            >= self.vars['spec_point'].get())[0][0]
        try:
            self.vars['od'].set(
                self.data.abs_spectra[
                    self.data.get_active_key()][1, :][self.spec_ind])
        except Exception:
            self.spec_ind = spec_ind
            raise
        else:
            self.calc_results()

    def open_num_dens_calc(self):
        entry_dict = {
            'Mass': [{'Mass density:': [1.0, 'g/cm^3', 'normal'],
                      'Molar weight:':[300, 'g/mol', 'normal'],
                      'Molar density:':[0.0033, 'mol/cm^3', 'disabled']}, 2],
            'Unit Cell': [{'a': [1.0, 'nm', 'normal'],
                           'b':[1.0, 'nm', 'normal'],
                           'c':[1.0, 'nm', 'normal'],
                           'Volume':[1.0, 'nm^3', 'normal'],
                           'Z':[4.0, '', 'normal']}, 3]}
        self.num_dens_calc = EntryWindow(
            self,
            entry_dict=entry_dict[self.vars['num_dens_calc_mode'].get()][0],
            numcol=entry_dict[self.vars['num_dens_calc_mode'].get()][1],
            command=self.calc_num_dens)
        if self.vars['num_dens_calc_mode'].get() == 'Unit Cell':
            for k in ('a', 'b', 'c'):
                self.num_dens_calc.vars[k].trace(
                    'w', self.calc_unit_cell_vol)

    def calc_unit_cell_vol(self, *args):
        prod = 1
        try:
            for k in ('a', 'b', 'c'):
                prod = prod*self.num_dens_calc.vars[k].get()
        except Exception:
            pass
        else:
            self.num_dens_calc.vars['Volume'].set(prod)

    def calc_num_dens(self, *args):
        if self.vars['num_dens_calc_mode'].get() == 'Mass':
            self.num_dens_calc.vars['Molar density:'].set(np.round(
                self.num_dens_calc.vars['Mass density:'].get() /
                self.num_dens_calc.vars['Molar weight:'].get(), 6))
            self.vars['num_dens'].set(np.format_float_scientific(
                self.num_dens_calc.vars['Molar density:'].get()
                * 6.022*1e23, 6))
        elif self.vars['num_dens_calc_mode'].get() == 'Unit Cell':
            self.vars['num_dens'].set(np.format_float_scientific(
                self.num_dens_calc.vars['Z'].get()
                / (self.num_dens_calc.vars['Volume'].get()*1e-27*1e6)))
        self.calc_results()

    def calc_results(self, *args):
        if self.vars['exc_spec_unit'].get() == 'nm':
            photon_energy = (6.626e-34*2.998e8
                             / (self.vars['spec_point'].get()
                                * self.length_unit_dict[
                                    self.vars['exc_spec_unit'].get()][0]))
        elif self.vars['exc_spec_unit'].get() == 'eV':
            photon_energy = self.vars['spec_point'].get()*1.6022e-16  # cm^-2
        else:
            messagebox.showerror(
                message="Invalid unit/quantity for excitation energy")
            return
        if self.vars['sample_type'].get() == 'Solid':
            self.data.calc_properties_crystal(
                od=self.vars['od'].get(),
                length=(self.vars['thickness'].get()
                        * self.length_unit_dict[
                            self.vars['length_unit'].get()][0]),
                num_dens=self.vars['num_dens'].get(),
                photon_energy=photon_energy,
                fluence=self.vars['fluence'].get())
            disp_text = ["Abs. coefficient:\nAbs. cross section:"
                         + "\nPenetration depth:\n\nPhoton energy:"
                         + "\nPhoton flux:\nExcitation fraction:",
                         str(np.format_float_scientific(
                             self.data.abs_coeff, 3))
                         + " cm^-1\n"
                         + str(np.format_float_scientific(
                             self.data.abs_cross, 6))
                         + " cm^2\n"
                         + str(np.round(self.data.penetr_depth*1e-9
                                        / self.length_unit_dict[
                                            self.vars['length_unit'].get()][0],
                                        1))
                         + " " + self.vars['length_unit'].get() + "\n\n"
                         + str(np.format_float_scientific(photon_energy, 6))
                         + "\n"
                         + str(np.format_float_scientific(
                             self.data.photon_flux, 6))
                         + " cm^-2\n"
                         + str(np.round(self.data.exc_frac, 3))
                         + " %"]
        else:
            num_dens = self.data.calc_properties_solution(
                od=self.vars['od'].get(),
                length=(self.vars['thickness'].get()
                        * self.length_unit_dict[
                            self.vars['length_unit'].get()][0]),
                conc=self.vars['num_dens'].get()/1000,
                photon_energy=photon_energy,
                fluence=self.vars['fluence'].get())
            disp_text = ["Abs. coefficient:\nMolar Abs. coeff.: "
                         + "\nAbs. cross section:\nPenetration depth:"
                         + "\nNumber Density:\n\nPhoton energy: \nPhoton flux:"
                         + "\nExcitation fraction: ",
                         str(np.format_float_scientific(
                             self.data.abs_coeff, 3))
                         + " cm^-1\n"
                         + str(np.format_float_scientific(
                             self.data.abs_coeff_mol, 3))
                         + " L*mol^-1*cm^-1\n"
                         + str(np.format_float_scientific(
                             self.data.abs_cross, 6))
                         + " cm^2\n"
                         + str(np.round(self.data.penetr_depth, 1))
                         + " mm\n"
                         + str(np.format_float_scientific(num_dens, 4))
                         + " cm^-3\n\n"
                         + str(np.format_float_scientific(photon_energy, 6))
                         + "\n"
                         + str(np.format_float_scientific(
                             self.data.photon_flux, 6))
                         + " cm^-2\n"
                         + str(np.round(self.data.exc_frac, 3))
                         + " %"]
        self.result_text[0].config(text=disp_text[0])
        self.result_text[1].config(text=disp_text[1])

    def filter(self):
        parameters = {"sigma": self.vars['filt_sigma'].get(),
                      "order": self.vars['filt_order'].get()}
        self.data.filter_abs_spectrum(parameters=parameters)
        self.plot()

    def reset_filter(self):
        self.data.clear_filter()
        self.plot()

    def plot_func(self, *args, fig=None, ax=None, zero_line=True, **kwargs):
        if fig is None:
            fig = self.figure
        if ax is None:
            ax = self.axes
        if not self.vars['overlay_plot'].get():
            ax.cla()
        x = []
        y = []
        if self.vars['plot_mode'].get().lower() == 'absorption':
            if re.search('all', self.vars['plot_which'].get(), re.I):
                keys = self.data.abs_spectra.keys()
            else:
                keys = [self.data.get_active_key()]
            for k in keys:
                x.append(self.data.abs_spectra[k][0, :])
                y.append(self.data.abs_spectra[k][1, :])
        else:
            x = self.data.wavelengths
            if self.vars['plot_mode'].get().lower() == 'intensity':
                y = self.data.trans
            if self.vars['plot_mode'].get().lower() == 'reference':
                y = self.data.ref
            if self.vars['plot_mode'].get().lower() == 'background':
                y = self.data.bkg
            else:
                return
        lines = tk_mpl_plot_function(np.array(x), np.array(y), *args,
                                     fig=fig, ax=ax, **kwargs)
        if zero_line:
            ax.plot([x[0][0], x[0][-1]], [0, 0], color='grey',
                    linestyle='-.', dashes=(5, 1))
        return lines

    def plot(self, *args, **kwargs):
        self.plot_func(*args, fig=self.figure, ax=self.axes, **kwargs)
        self.axes.set_xlabel(self.labels['spec_unit'].cget("text")[:-1] + " ("
                             + self.vars['spec_unit'].get() + ")")
        lbl_dict = {'Absorption': 'Abs.'}
        for key in ('Intensity', 'Reference', 'Background'):
            lbl_dict[key] = 'Intensity (counts)'
        self.axes.set_ylabel(lbl_dict[self.vars['plot_mode'].get()])
        self.set_fontsize()
        self.legend_filenames_callback()
        self.canvas.draw()

    def subtr_baseline(self, *args):
        if self.vars['baseline_subtr_mode'].get() == 'Constant':
            self.data.subtr_baseline(
                self.vars['baseline_value'].get(),
                from_all=self.vars['base_subtr_from'].get() == 'All spectra')
            self.axes.cla()
            self.plot()
        self.get_od_at_lambda()

    def calc_abs_spec(self, *args):
        err, key = self.data.calc_abs_spectrum()
        if err is None:
            self.vars['active_spec'].set(
                len(list(self.data.abs_spectra.keys())))
            self.vars['plot_which'].set('Active spectrum')
            self.plot()
        else:
            messagebox.showerror("Error", str(err))
        # try:
        #     self.data.calc_abs_spectrum()
        # except Exception:
        #     pass
        # else:
        #     self.vars['active_spec'].set(
        #         len(list(self.data.abs_spectra.keys())))
        #     self.vars['plot_which'].set('Active spectrum')
        #     self.plot()


class DifferenceAbsorption(tk.Toplevel, AppMain):
    def __init__(self, parent, data):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent
        self.title("Differential steady state absorption")
        self.data = data
        self.vars = {}
        self.entries = {}
        self.buttons = {}
        self.frames = {}
        self.widgets = {}

        self.frames['toolbar'] = CustomFrame(self, border=False)
        self.figure = plt.figure()
        plt.close()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.set_fontsize()
        self.canvas.get_tk_widget().config(width=500, height=400)
        self.toolbar = NavigationToolbar2Tk(
            self.canvas, self.frames['toolbar'])
        self.toolbar.update()
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(sticky='wnse')
        self.canvas._tkcanvas.grid(row=0, column=0, pady=5, rowspan=2)
        self.canvas.callbacks.connect(
            'button_press_event', self.canvas_callback)
        self.frames['toolbar'].grid(row=2, column=0, sticky='w')

        self.filelist = tk.Listbox(self, width=60)
        for k in self.data.abs_spectra.keys():
            self.filelist.insert(tk.END, k)
        self.filelist.grid(row=0, column=1, sticky='w')
        self.filelist.bind('<<ListboxSelect>>', self.filelist_callback)
        self.vars['subtr_which'] = tk.StringVar()
        self.entries['subtr_which'] = tk.ttk.Entry(
            self, textvariable=self.vars['subtr_which'], width=60)
        self.entries['subtr_which'].grid(row=3, column=0, columnspan=2,
                                         sticky='w')

        self.vars['disp_legend'] = tk.IntVar(value=1)
        self.widgets['disp_legend'] = tk.ttk.Checkbutton(
            self, text='Show Legend', variable=self.vars['disp_legend'],
            command=self.show_legend_callback)
        self.widgets['disp_legend'].grid(row=1, column=1, sticky='w')

        self.frames['commands'] = CustomFrame(self, dim=(5, 1), border=True)

        tk.ttk.Button(
            self.frames['commands'], text='Plot abs.',
            command=self.plot_abs).grid(row=1, column=0, sticky='w')

        self.vars['stack_plots'] = tk.IntVar(value=1)
        self.widgets['stack_plots'] = tk.ttk.Checkbutton(
            self.frames['commands'], variable=self.vars['stack_plots'],
            text='Stack plots')
        self.widgets['stack_plots'].grid(row=1, column=1, sticky='w')
        tk.ttk.Button(
            self.frames['commands'], text='Plot difference',
            command=self.plot_diff).grid(row=0, column=0, sticky='w')

        self.vars['stack_diff_plots'] = tk.IntVar(value=1)
        self.widgets['stack_diff_plots'] = tk.ttk.Checkbutton(
            self.frames['commands'], variable=self.vars['stack_diff_plots'],
            text='Stack plots')
        self.widgets['stack_diff_plots'].grid(row=0, column=1, sticky='w')

        tk.ttk.Button(
            self.frames['commands'], text='Clear difference',
            command=self.clear_diff).grid(row=0, column=2, sticky='w')

        tk.ttk.Label(self.frames['commands'], text="Subtraction Mode:").grid(
            row=0, column=3, sticky='w')
        self.vars['subtr_mode'] = tk.StringVar(value="Reference")
        self.widgets['subtr_mode'] = tk.ttk.OptionMenu(
            self.frames['commands'], self.vars['subtr_mode'],
            self.vars['subtr_mode'].get(), "Reference", "Individual",
            command=self.clear_diff)
        self.widgets['subtr_mode'].grid(row=0, column=4, sticky='w')

        self.frames['commands'].grid(row=4, column=0, sticky='wnse',
                                     columnspan=2)

        self.menubar = tk.Menu(self)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)

        # filemenu.add_separator()
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.filemenu.add_command(
            label="Save Diff. Spectra", command=self.save_diff_abs)
        self.filemenu.add_command(
            label="Save as traces", command=self.save_diff_abs_traces)
        tk.Tk.config(self, menu=self.menubar)

        self.abs_to_plot = {}
        self.current_plot_mode = 'abs'

    def filelist_callback(self, *args):
        if re.search("indiv", self.vars['subtr_mode'].get(), re.I):
            try:
                if (re.search(" - ", self.vars['subtr_which'].get())
                        or self.vars['subtr_which'].get() == ""):
                    self.vars['subtr_which'].set(
                        self.filelist.get(self.filelist.curselection()))
                    self.axes.cla()
                    self.canvas.draw()
                else:
                    self.vars['subtr_which'].set(" - ".join([
                        self.vars['subtr_which'].get(),
                        self.filelist.get(self.filelist.curselection())]))
                    self.plot_diff()
            except Exception:
                return
            else:
                self.filelist.selection_clear(self.filelist.curselection())
        else:
            self.vars['subtr_which'].set(
                self.filelist.get(self.filelist.curselection()))
            self.plot_diff()

    def show_legend_callback(self, *args):
        self.axes.get_legend().set_visible(self.vars['disp_legend'].get())
        self.canvas.draw()

    def calc_diff(self, *args):
        if not self.vars['stack_diff_plots'].get():
            self.data.diff_abs = {}
        if re.search("indiv", self.vars['subtr_mode'].get(), re.I):
            keys = self.vars['subtr_which'].get().split(" - ")
            spec_keys = [keys[0]]
            ref_keys = keys[1]
        else:
            ref_keys = self.vars['subtr_which'].get()
            spec_keys = [k for k in self.data.abs_spectra.keys()
                         if k != ref_keys]

        self.data.calc_diff_abs(spec_keys, ref_keys)

    def clear_diff(self, *args):
        self.vars['subtr_which'].set("")
        self.data.diff_abs = {}
        self.axes.cla()
        self.canvas.draw()

    def canvas_callback(self, event):
        if event.dblclick or event.button == 3:
            if self.current_plot_mode == 'abs':
                func = self.plot_abs
            else:
                func = self.plot_diff
            self.parent.open_figure(
                self, lambda fig, ax, **kwargs:
                    func(fig=fig, ax=ax, **kwargs),
                plot_type='line',
                dim=[600, 400], editable=True)

    def plot_diff(self, *args, fig=None, ax=None, **kwargs):
        if ax is None:
            ax = self.axes
        try:
            self.calc_diff()
        except Exception:
            raise
        else:
            ax.cla()
            for d in self.data.diff_abs.values():
                ax.plot(d[0, :], d[1, :])
            ax.legend(list(self.data.diff_abs.keys()))
            ax.set_ylabel('$\Delta$ Abs.')
            ax.set_xlabel('wavelength (nm)')
            self.set_fontsize()
            self.canvas.draw()
            self.current_plot_mode = 'diff'
            return [0, 0]

    def plot_abs(self, *args, fig=None, ax=None, **kwargs):
        if ax is None:
            ax = self.axes
        if not self.vars['stack_plots'].get():
            self.abs_to_plot = {}
        ax.cla()
        try:
            self.abs_to_plot[
                self.filelist.get(self.filelist.curselection())] = (
                    self.data.abs_spectra[
                        self.filelist.get(self.filelist.curselection())])
        except Exception:
            return
        else:
            for d in self.abs_to_plot.values():
                ax.plot(d[0], d[1])
            ax.set_ylabel('Abs.')
            ax.set_xlabel('wavelength (nm)')
            self.set_fontsize()
            self.canvas.draw()
            self.current_plot_mode = 'abs'
            return [0, 0]

    def save_diff_abs(self, *args, case='all'):
        file = save_box(filetypes=[('text files', '.txt'),
                                   ('Matlab files', '.mat')],
                        fext='.txt', parent=self)
        try:
            fname = file.name
        except Exception:
            return
        self.data.save_diff_abs(path=fname, case=case)

    def save_diff_abs_traces(self, *args, case='all'):
        file = save_box(filetypes=[('text files', '.txt'),
                                   ('Matlab files', '.mat')],
                        fext='.txt', parent=self)
        try:
            fname = file.name
        except Exception:
            return
        self.data.save_diff_abs_traces(path=fname, case=case)


class EntryWindow(tk.Toplevel):
    def __init__(self, parent, command=None, numcol=2, entry_dict={}):
        tk.Toplevel.__init__(self, parent)
        i = 0
        j = 0
        self.vars = {}
        self.entries = {}
        for k in entry_dict.keys():
            tk.ttk.Label(self, text=k + ' (' + entry_dict[k][1] + ')').grid(
                             row=i, column=2*j)
            self.vars[k] = tk.DoubleVar(value=entry_dict[k][0])
            self.entries[k] = tk.ttk.Entry(
                self, textvariable=self.vars[k], width=10,
                state=entry_dict[k][2], justify=tk.RIGHT)
            if command:
                self.entries[k].bind('<Return>', command)
            self.entries[k].grid(row=i, column=2*j + 1, sticky='w')
            j = j + 1
            if j == numcol:
                i = i + 1
                j = 0
        if command:
            tk.ttk.Button(self, text='OK',
                          command=lambda: self.execute(
                              command, close=True)).grid(
                                  row=i + 1, column=0, columnspan=2*numcol)
        else:
            tk.ttk.Button(self, text='OK', command=self.destroy).grid(
                row=i + 1, column=0, columnspan=2*numcol)
        center_toplevel(self, parent)

    def execute(self, command, close=True):
        command()
        if close:
            self.destroy()

# %%
"""
class StaticAbsData():
    def __init__(self):
        self.wavelengths = None
        self.abs = None
        self.ref = None
        self.bkg = None
        self.trans = None
        self.baseline = 0
        self.abs_spectra = {}
        self.abs_nonfilt = {}
        self.magnif_factors = {}
        self.diff_abs = {}
        self._active_key = ""

    def set_active_spec(self, index=0):
        try:
            self._active_key = list(self.abs_spectra.keys())[index]
        except IndexError:
            self._active_key = self.abs_spectra.keys()
        except Exception:
            pass
        # try:
        #     self._active_key = list(self.abs_spectra.keys())[index]
        # except Exception:
        #     pass
        else:
            self.abs = self.abs_spectra[self._active_key][1]

    def get_active_key(self):
        return self._get_abs_keys('active')[0]

    def calc_properties_crystal(self, od=1.0, length=1e-7, num_dens=1e21,
                                photon_energy=5e-19, fluence=2.0):
        # length in m
        # number density in cm^-3
        # photon energy in J
        # fluence in mJ*(cm^-2)
        # od = -log10(Transmission)
        self.abs_coeff = od/length*1e-2
        self.abs_cross = self.abs_coeff/num_dens
        self.penetr_depth = 1/(self.abs_coeff)*1e7  # nm
        self.photon_flux = fluence*1e-3/photon_energy
        photon_abs_ratio = 1 - np.power(10, -od)
        self.exc_frac = float(
            self.photon_flux*photon_abs_ratio/(length*1e2*num_dens)*100)
        self.fluence = fluence

    def calc_properties_solution(self, od=1.0, length=1e-3, conc=1e-3,
                                 photon_energy=5e-19, fluence=2.0):
        # input parameter units: length in meters,
        # conc (= concentration) in mol/L
        # photon_energy in Joules, fluence in mJcm^(-2)
        self.fluence = fluence
        self.abs_coeff = od/length*1e-2
        self.abs_coeff_mol = od/(length*conc)*1e-2
        num_dens = conc*6.022e23*1e-3  # cm^-3
        self.abs_cross = self.abs_coeff/num_dens
        self.penetr_depth = 1/(self.abs_coeff)*10  # mm
        self.photon_flux = fluence*1e-3/photon_energy
        photon_abs_ratio = 1-np.power(10, -od)
        self.exc_frac = float(
            self.photon_flux*photon_abs_ratio/(length*1e2*num_dens)*100)
        return num_dens

    def clear_active_spec(self):
        del self.abs_spectra[self._active_key]
        self.set_active_spec(index=0)
        # try:
        #     self._active_key = list(self.abs_spectra.keys())[0]
        # except IndexError:
        #     self._active_key = self.abs_spectra.keys()
        # except Exception:
        #     pass
        # try:
        #     self.abs = self.abs_spectra[self._active_key][1]
        # except Exception:
        #     self.abs = None

    def clear_data(self, *args):
        self.abs_spectra = {}
        self.abs_nonfilt = {}
        self.abs = None

    def load_abs_spectrum(self, path):
        try:
            loaded_data = self.load_spec_txt(path)
        except FileNotFoundError:
            return False
        except Exception:
            raise
        else:
            self.abs_fname = re.split('\.', re.split('\/', path)[-1])[-2]
            self.wavelengths = loaded_data[:, 0]
            self._active_key = self.abs_fname
            self.abs = loaded_data[:, 1]
            self.abs_spectra[self.abs_fname] = np.array(
                [loaded_data[:, 0], loaded_data[:, 1]])
            return True

    def filter_abs_spectrum(self, mode='gauss', parameters=None,
                            case='active'):
        if parameters is None:
            parameters = {}
        std_para = {"savgol": {"order": 5, "framelength": 3},
                    "gauss": {"sigma": 1, "order": 0}}
        func = {"savgol": savgol_filter_wrap, "gauss": gaussian_filter}
        for key, val in std_para[mode].items():
            if key not in parameters.keys():
                parameters[key] = val
        keys = self._get_abs_keys(case)
        for key in keys:
            self.abs_nonfilt[key] = np.array(
                [a for a in self.abs_spectra[key][1, :]])
            self.abs_spectra[key][1, :] = func[mode](
                self.abs_spectra[key][1, :], **parameters)

    def clear_filter(self):
        for key, val in self.abs_nonfilt.items():
            try:
                self.abs_spectra[key][1, :] = val
            except KeyError:
                pass
            except Exception:
                raise

    def _get_abs_keys(self, case):
        if re.search('active', case, re.I):
            return [self._active_key]
        elif re.search('all', case, re.I):
            return self.abs_spectra.keys()
        else:
            return [case]

    def magnify_abs(self, factor=1, case='active'):
        if factor == 0:
            return
        keys = self._get_abs_keys(case)
        for key in keys:
            if key in self.magnif_factors.keys():
                fac = factor/self.magnif_factors[key]
            else:
                fac = factor
            self.magnif_factors[key] = factor
            self.abs_spectra[key][1, :] = fac*self.abs_spectra[key][1, :]

    def reset_magnif(self):
        for key, fac in self.magnif_factors.items():
            try:
                self.abs_spectra[key][1, :] = self.abs_spectra[key][1, :]/fac
            except Exception:
                continue
            else:
                self.magnif_factors[key] = 1

    def load_ref_spec(self, path):
        try:
            loaded_data = self.load_spec_txt(path)
        except FileNotFoundError:
            pass
        except Exception:
            raise
        else:
            self.wavelengths = loaded_data[:, 0]
            self.ref = loaded_data[:, 1]

    def load_bkg_spectrum(self, path):
        try:
            loaded_data = self.load_spec_txt(path)
        except FileNotFoundError:
            pass
        except Exception:
            raise
        else:
            self.wavelengths = loaded_data[:, 0]
            self.bkg = loaded_data[:, 1]

    def load_transmiss(self, path):
        try:
            loaded_data = self.load_spec_txt(path)
        except FileNotFoundError:
            pass
        except Exception:
            raise
        else:
            self.wavelengths = loaded_data[:, 0]
            self.trans = loaded_data[:, 1]
            self.trans_fname = re.split('\/', path)[-1]

    def load_spec_txt(self, path, load_all=True):
        m = []
        with open(path) as f:
            while True:
                line = f.readline()
                if not line.startswith('#'):
                    break
            while line:
                m.append(line)
                line = f.readline()
            f.close()
        m = np.array([x.strip().split("\t") for x in m])

        try:
            return m.astype(np.float64)
        except Exception:
            try:
                m = np.array([[x.replace('n. def.', 'nan')
                             for x in i] for i in m])
                return m.astype(np.float64)
            except Exception:
                try:
                    m = np.array([x[0].split(",") for x in m])
                except Exception:
                    raise
                else:
                    try:
                        return m.astype(np.float64)
                    except Exception:
                        raise

    def save(self, path, case='active'):
        with open(path, 'w+') as f:
            np.savetxt(f, np.transpose(
                self.abs_spectra[self._active_key]), delimiter='\t')
            f.close()

    def sort_spectra(self, mode="name"):
        new_dct = {}
        for attr in ["abs_spectra", "diff_abs"]:
            new_dct[attr] = {}
            dct = getattr(self, attr)
            keys = list(dct.keys())
            if re.search("name|alph", mode, re.I):
                sorted_inds = sorted(range(len(keys)), key=keys.__getitem__)
            elif re.search("val", mode, re.I):
                vals = []
                for key in keys:
                    try:
                        vals.append(np.double(re.findall("[\d\.]+", key)[0]))
                    except Exception:
                        vals.append(np.inf)
                sorted_inds = np.argsort(vals)
            for i in sorted_inds:
                new_dct[attr][keys[i]] = dct[keys[i]]
        for key, attr in new_dct.items():
            setattr(self, key, attr)

    def save_abs_traces(self, path=None, case='active'):
        self._save_traces(path, case, case='abs')

    def _save_traces(self, path, keys, case='abs'):
        traces = TATraceCollection()
        if re.search('diff', case, re.I):
            spectra = self.diff_abs
            ylabel = "$\Delta$ Abs. (OD)"
        else:
            spectra = self.abs_spectra
            ylabel = "Abs. (OD)"
        if not type(keys) is list:
            if re.search('active', keys, re.I):
                keys = [self._active_key]
            elif re.search('all', keys, re.I):
                keys = spectra.keys()
        for key in keys:
            trace = TATrace(xdata=spectra[key][0, :],
                            trace_type='Spectral',
                            yunit='OD', ylabel=ylabel,
                            xlabel='wavelength (nm)', xunit='nm')
            trace.add_trace(spectra[key][1, :], "")

            traces.add_trace(trace, key)
        traces.save(trace_type='Spectral',
                    fname=path)

    def save_diff_abs(self, path, case='active'):
        key = list(self.diff_abs.keys())[-1]
        with open(path, 'w+') as f:
            np.savetxt(f, np.transpose(self.diff_abs[key]), delimiter='\t')
            f.close()

    def save_diff_abs_traces(self, path=None, case='active'):
        self._save_traces(path, case, case='diff')

    def calc_abs_spectrum(self):
        try:
            self.abs = np.log10(
                np.divide((self.ref-self.bkg), (self.trans-self.bkg)))
        except Exception:
            if self.ref is None:
                messagebox.showerror("Error", "Reference spectrum not loaded.")
                raise
            elif self.bkg is None:
                messagebox.showerror(
                    "Error", "Background spectrum not loaded.")
                raise
            elif self.trans is None:
                messagebox.showerror(
                    "Error", "Transmitted spectrum not loaded.")
                raise
            else:
                messagebox.showerror(
                    "Error", "Error calculating absorption spectrum")
                raise
        else:
            wl = [w for w in self.wavelengths]
            self.abs_spectra[self.trans_fname] = np.array([wl, self.abs])
            self._active_key = self.trans_fname
            return self._active_key

    def calc_diff_abs(self, spec_keys, ref_keys):
        ref = self.abs_spectra[ref_keys][1, :]
        for key in spec_keys:
            try:
                self.diff_abs[" - ".join([key, ref_keys])] = np.array(
                    [self.abs_spectra[key][0, :],
                     self.abs_spectra[key][1, :] - ref])
            except KeyError:
                pass
            except Exception:
                raise

    def subtr_baseline(self, val, case='abs', from_all=False):
        if case == 'abs':
            if from_all:
                for spec in self.abs_spectra.values():
                    spec[1, :] = spec[1, :] - val
            else:
                self.abs_spectra[self._active_key][1, :] = (
                    self.abs_spectra[self._active_key][1, :] - val)
        elif case == 'ref':
            self.ref = self.ref - val
        elif case == 'trans':
            self.trans = self.trans - val
        else:
            return
        self.baseline = self.baseline + val

    def get_abs_xind(self, key, xrange, mode='slice'):
        x = self.abs_spectra[key][0, :]
        try:
            xlower = np.where(x >= xrange[0])[0][0]
        except Exception:
            xlower = 0
        try:
            xupper = np.where(x <= xrange[1])[0][-1]
        except Exception:
            xupper = len(x)
        if re.search('slice', mode, re.I):
            return slice(xlower, xupper)
        elif re.search('ind', mode, re.I):
            return [xlower, xupper - 1]

    def calc_centroid(self, xrange=None, spectrum='active'):
        def centroid(tr, x):
            return np.sum(tr * x)/np.sum(tr)

        if re.search('active', spectrum, re.I):
            xrange = self.get_abs_xind(self._active_key, xrange, mode='slice')
            s = self.abs_spectra[self._active_key]
            return centroid(s[1, xrange], s[0, xrange])
        elif re.search('all', spectrum, re.I):
            c = {}
            for key, val in self.abs_spectra.items():
                xrange = self.get_abs_xind(key, xrange, mode='slice')
                c[key] = centroid(val[1, xrange], val[0, xrange])
            return c

    def integrate_spectrum(self, xrange=None, spectrum='current'):
        if re.search('curr', spectrum, re.I):
            xrange = self.get_abs_xind(self._active_key, xrange, mode='slice')
            return (np.sum(self.abs_spectra[self._active_key][1, xrange]),
                    np.mean(self.abs_spectra[self._active_key][1, xrange]))
        elif re.search('all', spectrum, re.I):
            integ = {}
            mean = {}
            for key, val in self.abs_spectra.items():
                xrange = self.get_abs_xind(key, xrange, mode='slice')
                integ[key] = np.sum(val[1, xrange])
                mean[key] = np.mean(val[1, xrange])
            return integ, mean

    def plot_abs_spectra(self, ax, plot_which='All', zero_line=True,
                         **kwargs):
        if plot_which == 'All':
            plot_which = self.abs_spectra.keys()
            for k in plot_which:
                ax.plot(self.abs_spectra[k][0, :],
                        self.abs_spectra[k][1, :], **kwargs)
            if zero_line:
                ax.plot([self.abs_spectra[k][0, 0],
                         self.abs_spectra[k][0, -1]],
                        [0, 0],
                        color='grey', linestyle='-.', dashes=(5, 1),
                        **kwargs)
        else:
            dat = self.abs_spectra[self._active_key]
            ax.plot(dat[0, :], dat[1, :])
            if zero_line:
                ax.plot([dat[0, 0], dat[0, -1]], [0, 0], color='grey',
                        linestyle='-.', dashes=(5, 1), **kwargs)
                """
