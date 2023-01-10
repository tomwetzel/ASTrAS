# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:05:32 2020

@author: bittmans
"""
from ..common.helpers import GlobalSettings
from ..common.dataobjects import TATraceCollection
from ..common.tk.linefit import FitTracePage
from ..common.tk.general import load_box, save_box
from ..common.tk.figures import open_figure_options
import matplotlib.font_manager
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

import numpy as np

import matplotlib as mpl
mpl.use("Agg")


class AppMain(tk.Tk, tk.Toplevel):
    def __init__(self, *args, inputobj=None, config_filepath='config.txt',
                 parent=None, **kwargs):
        if parent is None:
            tk.Tk.__init__(self, *args, **kwargs)
        else:
            tk.Toplevel.__init__(self, parent, *args, **kwargs)
        self.title("ASTrAS - Line Fit")
        self.color_settings = {'cmap_name': 'Red Blue',
                               'sym_lim': True,
                               'sym_map': True,
                               'clims': [-10, 10]}
        self.color_settings['plot_kwargs'] = {
            'vmin': self.color_settings['clims'][0],
            'vmax': self.color_settings['clims'][1],
            'cmap': plt.get_cmap('RdBu_r')}
        self.settings = GlobalSettings(config_path=config_filepath)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        # define container
        container = tk.Frame(self)
        container.grid()
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        # define frame(s). currently only one, but can be extended
        self.frames = {}
        for F in [FitTracePage]:
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.fr = self.frames[FitTracePage]
        self.fr.fit_figure.set_fit_kwargs(**self.settings['fit_kwargs'])
        self.mode = 'kinetic'
        self.show_frame(FitTracePage)
        self.fr.trace_coll.traces = {}
        self.fr.trace_coll.active_members = []
        # self.fr.fit_figure.set_callback(self.fig_callback)
        try:
            for ax, kwargs in self.settings['ticklabel_format'].items():
                self.fr.fit_figure.set_ticklabel_format(
                    i=0, axis=ax, update_canvas=False, **kwargs)
        except KeyError:
            pass
        except Exception:
            raise
        self.fr.fit_figure.canvas.draw()
        self.xunit = 'ps'
        # initialize menu bar
        self.menubar = tk.Menu(container)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.filemenu.add_command(
            label="Load trace data", command=self.load_traces)
        self.filemenu.add_command(label="Save traces",
                                  command=self.save_traces)
        self.filemenu.add_command(
            label="Save traces and fits", command=self.fr.save_fit)
        tk.Tk.config(self, menu=self.menubar)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def load_traces(self, filetypes=None, default_dir=None, multiple=False):
        if filetypes is None:
            filetypes = [('Trace files', '.txt .mat')]
        file, fext = load_box(
            title="Select trace file", filetypes=filetypes,
            default_dir=default_dir, fext=".txt")
        if file is None:
            return
        coll = TATraceCollection()
        try:
            coll.load_traces(file.name)
        except Exception:
            raise
            messagebox.showerror(
                message="Failed to load traces. Please ensure proper "
                + "file formatting.")
            return
        new_mode = False
        for member in coll.active_members:
            if coll.traces[member].type.lower() != self.mode:
                new_mode = coll.traces[member].type.lower()
        if new_mode:
            self.fr.trace_coll.remove_all_traces()
        else:
            new_mode = self.mode
        if new_mode == 'kinetic':
            setup_frame = self.setup_kinetic_fit
        elif new_mode == 'spectral':
            setup_frame = self.setup_line_fit
        else:
            messagebox.showinfo("", "Unknown trace type: " + new_mode)
            return
        self.mode = new_mode

        for member in coll.active_members:
            trace = coll.traces[member]
            self.fr.trace_coll.add_trace(trace, member)
        try:
            ylbl = self.fr.trace_coll.traces[
                self.fr.trace_coll.active_members[0]].ylabel
        except Exception:
            ylbl = ""
        else:
            for member in self.fr.trace_coll.active_members:
                if ylbl != self.fr.trace_coll.traces[member].ylabel:
                    ylbl = ""
                    break
            ylbl = ylbl.strip(" \n")
        self.fr.fit_figure.set_ylabel(ylbl)
        fit_done = False
        for member in self.fr.trace_coll.active_members:
            for k in self.fr.trace_coll.traces[member].active_traces:
                if 'fit' in self.fr.trace_coll.traces[member].tr[k].keys():
                    fit_done = True
                    try:
                        self.fr.success[member][k] = True
                    except KeyError:
                        self.fr.success[member] = {k: True}
                    except Exception:
                        raise
        self.fr.fit_done = fit_done
        setup_frame()
        self.fr.disp_results()
        self.fr.update_plot()

    def save_traces(self):
        file = save_box(fext='.txt',
                        filetypes=[('text files', '.txt'),
                                   ('Matlab files', '.mat')],
                        fname='traces',
                        parent=self)
        if file is not None:
            self.fr.trace_coll.save(file)

    def setup_line_fit(self):
        self.fr.set_mode('line_shape')
        self.setup_general_opts()

    def setup_general_opts(self):
        xmin = []
        xmax = []
        for member in self.fr.trace_coll.active_members:
            xmin.append(np.min(self.fr.trace_coll.traces[member].xdata))
            xmax.append(np.max(self.fr.trace_coll.traces[member].xdata))
        self.fr.frames['trace_select'].update_box()
        self.fr.auto_axis_limits()

    def setup_kinetic_fit(self):
        self.fr.set_mode('kinetic')
        self.setup_general_opts()
        for member in self.fr.trace_coll.active_members:
            xunit = self.fr.trace_coll.traces[member].get_xunit()
            time_unit_factors = self.fr.trace_coll.traces[
                member].time_unit_factors
        self.fr.func_opts.t0_label.config(text="Time zero (" + xunit + ")")
        self.fr.fit_range_label.config(text="Fit range (" + xunit + ")")
        if xunit in time_unit_factors.keys():
            self.fr.func_opts.irf_opt_check.config(
                text="IRF (" + time_unit_factors[xunit][1] + ")")
            self.fr.vars['irf_factor'] = 1e-3
            self.fr.vars['irf_val'].set(100)
        else:
            self.fr.func_opts.irf_opt_check.config(text="IRF (" +
                                                   xunit + ")")
            self.fr.vars['irf_factor'] = 1.0
            self.fr.vars['irf_val'].set(0.1)

    # def fig_callback(self, event):
    #     if event.button == 3:
    #         open_figure_options(
    #             self.fr, self.fr.fit_figure, default_vals=self.settings,
    #             controller=self.fr.controller)
