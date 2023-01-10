# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 23:13:16 2020

@author: bittmans
"""

import tkinter as tk
# from tkinter import ttk
import os
import re
import traceback
# import numpy as np

from queue import Queue
# import threading
from ..common.helpers import ThreadedTask

from ..common.dataobjects import TAParameterDependence
from ..common.tk.figures import TkMplFigure, open_topfigure

from ..common.tk.general import GroupBox, CustomFrame, CustomProgressbarWindow


class AppMain(tk.Tk):
    def __init__(self, *args, parent=None, **kwargs):
        if parent is None:
            tk.Tk.__init__(self, *args, **kwargs)
        else:
            tk.Toplevel.__init__(self, parent, *args, **kwargs)
        self.title("ASTrAS - Parameter Dependence")
        style = tk.ttk.Style()
        style.theme_use("vista")
        style.configure("Custom.TLabelframe.Label", foreground="black")
        style.configure("Custom.TLabelframe", foreground="grey")

        self.menubar = tk.Menu(self)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.filemenu.add_command(label='Load Scans', command=self.load_scans)
        tk.Tk.config(self, menu=self.menubar)

        self.main_frame = tk.Frame(self)

        self.fig_frame = tk.Frame(self.main_frame)
        self.data_obj = TAParameterDependence()
        self.map_fig = TkMplFigure(
            self.fig_frame, plot_function=self.data_obj.plot_ta_map,
            xlabels=[self.data_obj.xlabel],
            clabels=[self.data_obj.zlabel],
            ylabels=[
                "time delays (ps)"],
            dim=(600, 400),
            callbacks={'button_press_event': self.map_fig_callback})
        self.map_fig.grid(row=0, column=0, sticky='wnse')

        self.para_dep_fig = TkMplFigure(
            self.fig_frame, plot_function=self.data_obj.plot_function,
            plot_type='linefit', ylabels=['Singular Value (1st Comp.)'],
            dim=(600, 400),
            callbacks={'button_press_event': self.para_fig_callback})
        self.para_dep_fig.grid(row=0, column=1, sticky='wnse')

        self.fig_frame.grid(row=0, column=0, columnspan=3, sticky='wnse')

        tk.ttk.Label(self.main_frame, text="Show Data:").grid(
            row=1, column=0, sticky='w')
        self.scan_to_plot = tk.StringVar(value=" ")
        self.scan_to_plot_select = tk.ttk.OptionMenu(
            self.main_frame, self.scan_to_plot, " ", " ")
        self.scan_to_plot.trace('w', self.scan_to_plot_callback)
        self.scan_to_plot_select.grid(row=2, column=0, sticky='w')

        self.settings_groupbox = GroupBox(
            self.main_frame, dim=(2, 4), text="Options")

        tk.ttk.Label(self.settings_groupbox,
                     text="Read Parameter\nfrom:").grid(
                         row=0, column=0, sticky='w')
        self.para_read_mode = tk.StringVar(value="Filename")
        self.para_read_mode_select = tk.ttk.OptionMenu(
            self.settings_groupbox, self.para_read_mode, "Filename",
            "Filename", "Scan File", command=self.sort_scans)
        self.para_read_mode_select.grid(row=0, column=1, sticky='we')

        tk.ttk.Label(
            self.settings_groupbox, text="Quantity to plot:").grid(
                row=1, column=0, sticky='w')
        self.quantity_mode = tk.StringVar(value="Singular Value")
        self.quantity_mode_select = tk.ttk.OptionMenu(
            self.settings_groupbox, self.quantity_mode,
            "Singular Value", "Singular Value", "Integral",
            command=self.quantity_mode_callback)
        self.quantity_mode_select.grid(row=1, column=1, sticky='we')

        self.settings_groupbox.grid(row=1, column=1, rowspan=2, sticky='wnse')

#        self.status_label = tk.ttk.Label(self.main_frame, text = "Idle")
#        self.status_label.grid(row = 4, column = 0, sticky = 'w')

        self.window_opts_frame = tk.Frame(self.settings_groupbox)
        self.window_opts = GroupBox(
            self.settings_groupbox, dim=(3, 2), text='Window')
        tk.ttk.Label(self.window_opts, text='x:').grid(
            row=0, column=0, sticky='w')
        self.window_x_lower = tk.DoubleVar(value=400)
        self.window_x_lower_entry = tk.Entry(
            self.window_opts, textvariable=self.window_x_lower)
        self.window_x_lower_entry.grid(row=0, column=1, sticky='w')
        self.window_x_upper = tk.DoubleVar(value=600)
        self.window_x_upper_entry = tk.Entry(
            self.window_opts, textvariable=self.window_x_upper)
        self.window_x_upper_entry.grid(row=0, column=2, sticky='w')
        tk.ttk.Label(self.window_opts, text='y:').grid(
            row=1, column=0, sticky='w')
        self.window_y_lower = tk.DoubleVar(value=600)
        self.window_y_lower_entry = tk.Entry(
            self.window_opts, textvariable=self.window_y_lower)
        self.window_y_lower_entry.grid(row=1, column=1, sticky='w')
        self.window_y_upper = tk.DoubleVar(value=800)
        self.window_y_upper_entry = tk.Entry(
            self.window_opts, textvariable=self.window_y_upper)
        self.window_y_upper_entry.grid(row=1, column=2, sticky='w')
        for i, entry in enumerate((
                self.window_x_lower_entry, self.window_x_upper_entry,
                self.window_y_lower_entry, self.window_y_upper_entry)):
            entry.config(width=5)
            entry.bind('<Return>', lambda *args:
                       self.window_callback(*args, case='both'))
        self.window_opts.grid(row=0, column=2, sticky='wnse', padx=5, pady=5)
        self.window_opts_frame.grid(
            row=1, column=2, rowspan=1, sticky='wnse', padx=5, pady=5)

        self.svd_opts_frame = tk.Frame(self.settings_groupbox)
        self.svd_opts = GroupBox(
            self.svd_opts_frame, dim=(1, 2), text='SVD Components')
        self.no_svd_comp = tk.IntVar(value=1)
        self.no_svd_comp_select = tk.ttk.OptionMenu(
            self.svd_opts, self.no_svd_comp, 1, 1, 2, 3, 4)
        self.no_svd_comp_select.grid(row=0, column=0, sticky='we')
        self.svd_opts.grid(sticky='wnse')
        self.svd_opts_frame.grid(
            row=1, column=2, rowspan=1, sticky='wnse', padx=5, pady=5)

        tk.ttk.Button(self.main_frame, text='Calculate',
                      command=self.calculate_para_dep).grid(
                          row=3, column=0, columnspan=2)

        self.result_ops = GroupBox(self.main_frame, text="Result")

        self.curr_para_frame = CustomFrame(
            self.result_ops, dim=(4, 1), border=False)

        self.unit_conv_box = GroupBox(
            self.result_ops, dim=(3, 3), text="Conversion")
        tk.ttk.Label(self.curr_para_frame, text='Current Parameter:').grid(
            row=0, column=0, sticky='w')
        self.current_para = tk.StringVar()
        self.current_para.trace('w', self.manual_parameter_check)
        self.current_para_entry = tk.Entry(
            self.curr_para_frame, textvariable=self.current_para)
        self.current_para_entry.bind('<Return>', self.set_current_parameter)
        self.current_para_entry.grid(row=0, column=1, sticky='w')
        tk.ttk.Label(self.curr_para_frame, text='Current Unit:').grid(
            row=0, column=2, sticky='w')
        self.current_unit = tk.StringVar()
        self.current_unit.trace('w', self.manual_parameter_check)
        self.current_unit_entry = tk.Entry(
            self.curr_para_frame, textvariable=self.current_unit)
        self.current_unit_entry.bind('<Return>', self.set_current_parameter)
        self.current_unit_entry.grid(row=0, column=3)

        tk.ttk.Label(self.unit_conv_box, text="Convert to:").grid(
            row=0, column=0, sticky='w')
        tk.ttk.Label(self.unit_conv_box, text="Unit:").grid(
            row=1, column=0, sticky='w')
        self.convert_to = tk.StringVar(value="Power")
        self.convert_to_select = tk.ttk.OptionMenu(
            self.unit_conv_box, self.convert_to, self.convert_to.get(),
            "Power", "Energy", "Fluence",
            command=self.convert_to_select_callback)
        self.convert_to_select.grid(row=0, column=1, sticky='we')
        self.conversion_units = {
            "Power": ["nW", "uW", "mW", "W", "kW", "MW", "GW"],
            "Energy": ['pJ', 'nJ', 'uJ', 'mJ', 'J', 'kJ', 'MJ'],
            "Fluence": ['mJcm^-2', 'uJcm^-2', 'uJm^-2', 'Jcm^-2']}
        self.convert_to_unit = tk.StringVar(value="mW")
        self.convert_to_unit_select = tk.ttk.OptionMenu(
            self.unit_conv_box, self.convert_to_unit,
            "mW", *self.conversion_units["Power"])
        self.convert_to_unit.trace('w', self.para_conversion)
        self.convert_to_unit_select.grid(row=1, column=1, sticky='we')
        self.conversion_check_label = tk.ttk.Label(
            self.unit_conv_box, text=" ")
        self.conversion_check_label.grid(row=2, column=0, columnspan=2,
                                         sticky='w')

        self.conversion_para_box = GroupBox(
            self.unit_conv_box, dim=(1, 4), text='Conversion Parameters')
        tk.ttk.Label(self.conversion_para_box,
                     text='Rep. Rate (Hz):').grid(row=0, column=0, sticky='w')
        self.rep_rate = tk.DoubleVar(value=1000)
        self.rep_rate_entry = tk.Entry(
            self.conversion_para_box, textvariable=self.rep_rate)
        self.rep_rate_entry.grid(row=1, column=0, sticky='w')
        tk.ttk.Label(self.conversion_para_box,
                     text='Beam Diameter (um):').grid(
                         row=2, column=0, sticky='w')
        self.beam_size = tk.DoubleVar(value=100)
        self.beam_size_entry = tk.Entry(
            self.conversion_para_box, textvariable=self.beam_size)
        self.beam_size_entry.bind('<Return>', self.beam_size_callback)
        self.beam_size_entry.grid(row=3, column=0, sticky='w')

        self.conversion_para_box.grid(
            row=0, column=2, sticky='wnse', rowspan=3)

        self.fit_para_box = GroupBox(self.result_ops, dim=(3, 4), text='Fit')

        tk.ttk.Label(self.fit_para_box, text='Function:').grid(
            row=0, column=0, sticky='w')
        self.fit_func = tk.StringVar()
        self.fit_func_select = tk.ttk.OptionMenu(
            self.fit_para_box, self.fit_func, "Linear", "Linear", "Square",
            "Cubic")
        self.fit_func_select.grid(column=1, row=0, sticky='wnse', columnspan=3)

        tk.ttk.Label(self.fit_para_box, text='Fit range (x):').grid(
            row=1, column=0, sticky='w')
        self.fit_xlower = tk.DoubleVar(value=0)
        self.fit_xupper = tk.DoubleVar(value=20)
        self.fit_xlower_entry = tk.ttk.Entry(
            self.fit_para_box, textvariable=self.fit_xlower, width=5)
        self.fit_xupper_entry = tk.ttk.Entry(
            self.fit_para_box, textvariable=self.fit_xupper, width=5)
        self.fit_xlower_entry.bind(
            '<Return>', lambda *args: self.update_fit(*args, auto_range=False))
        self.fit_xupper_entry.bind(
            '<Return>', lambda *args: self.update_fit(*args, auto_range=False))
        self.fit_xlower_entry.grid(row=1, column=1, sticky='w', padx=2)
        self.fit_xupper_entry.grid(row=1, column=2, sticky='w', padx=2)

        tk.ttk.Label(self.fit_para_box, text="Plot range (x):").grid(
            row=2, column=0, sticky='w')
        self.expol_lower = tk.DoubleVar(value=0)
        self.expol_upper = tk.DoubleVar(value=20)
        self.expol_lower_entry = tk.ttk.Entry(self.fit_para_box,
                                              textvariable=self.expol_lower,
                                              width=5)
        self.expol_upper_entry = tk.ttk.Entry(self.fit_para_box,
                                              textvariable=self.expol_upper,
                                              width=5)
        self.expol_lower_entry.bind('<Return>', self.fit_para)
        self.expol_upper_entry.bind('<Return>', self.fit_para)

        self.expol_lower_entry.grid(row=2, column=1, sticky='w', padx=2)
        self.expol_upper_entry.grid(row=2, column=2, sticky='w', padx=2)

        tk.ttk.Button(self.fit_para_box, text='Fit',
                      command=self.fit_para).grid(
                          row=3, column=0, columnspan=3)
        self.curr_para_frame.grid(row=0, column=0, columnspan=2, sticky='wnse')
        self.unit_conv_box.grid(row=1, column=0, sticky='wnse')

        self.fit_para_box.grid(row=1, column=1, sticky='wnse')

        self.result_ops.grid(row=1, column=2, rowspan=2, sticky='wnse')

        self.main_frame.grid()

    def scan_to_plot_callback(self, *args):
        try:
            self.data_obj.delA = self.data_obj.sorted_scans[
                self.scan_to_plot.get()]['scan'].delA
            self.data_obj.wavelengths = self.data_obj.sorted_scans[
                self.scan_to_plot.get()]['scan'].wavelengths
            self.data_obj.time_delays = self.data_obj.sorted_scans[
                self.scan_to_plot.get()]['scan'].time_delays
        except Exception:
            traceback.print_exc()
        else:
            self.map_fig.plot()

    def fit_para(self, *args):
        model_dict = {'linear': 'linear',
                      "square": 'poly2',
                      "cubic": 'poly3'}
        fit_func = self.fit_func.get().lower()
        self.data_obj.fit(
            fit_func=model_dict[fit_func],
            xrange=[self.fit_xlower.get(), self.fit_xupper.get()],
            fit_plot_range=[self.expol_lower.get(), self.expol_upper.get()])
        self.data_obj.disp_fit_func = False
        self.para_dep_fig.set_legend(
            ["Data", self.data_obj.fit_disp, "Included"])
        self.para_dep_fig.set_legend_visibility(True, update_canvas=False)
        self.para_dep_fig.plot()

    def beam_size_callback(self, *args):
        if (self.data_obj.x_para.lower() == 'fluence'
                and self.data_obj.x_para_original.lower() != 'fluence'):
            self.data_obj.x = self.data_obj.x * \
                (self.data_obj.beam_size / (self.beam_size.get()*1e-6))**2
        elif (self.data_obj.x_para_original.lower() == 'fluence'
              and self.data_obj.x_para.lower() != 'fluence'):
            self.data_obj.x = self.data_obj.x * \
                ((self.beam_size.get()*1e-6) / self.data_obj.beam_size)**2
        self.data_obj.beam_size = self.beam_size.get()*1e-6
        self.para_dep_fig.plot()

    def convert_to_select_callback(self, *args):
        self.convert_to_unit_select['menu'].delete(0, 'end')
        for unit in self.conversion_units[self.convert_to.get()]:
            self.convert_to_unit_select['menu'].add_command(
                label=unit, command=lambda o=unit: self.convert_to_unit.set(o))
        default_units = {'power': 'mW', 'energy': 'nJ', 'fluence': 'mJcm^-2'}
        self.convert_to_unit.set(default_units[self.convert_to.get().lower()])

    def para_conversion(self, *args):
        self.beam_size_callback()
        self.data_obj.convert_para(
            convert_to=self.convert_to.get(),
            convert_to_unit=self.convert_to_unit.get(),
            rep_rate=self.rep_rate.get(),
            beam_size=self.beam_size.get()*1e-6)
        self.para_dep_fig.set_xlabel(
            label=self.data_obj.x_label, update_canvas=False)
        self.para_dep_fig.plot()
        self.current_para.set(self.data_obj.x_para)
        self.current_unit.set(self.data_obj.x_unit)
        self.update_fit()

    def set_current_parameter(self, *args):
        self.data_obj.x_para = self.current_para.get()
        self.data_obj.x_unit = self.current_unit.get()
        self.data_obj.x_label = self.data_obj.x_para + \
            ' (' + self.data_obj.x_unit + ')'
        self.para_dep_fig.set_xlabel(label=self.data_obj.x_label)

    def manual_parameter_check(self, *args):
        state = 'disabled'
        if self.current_para.get().lower() in ("power", "energy", "fluence"):
            if (self.current_unit.get() in self.conversion_units[
                    self.current_para.get().lower().title()]):
                state = 'normal'
                lbl = " "
            else:
                lbl = "Invalid unit"
        else:
            lbl = "Invalid parameter"
        self.conversion_check_label.config(text=lbl)
        self.convert_to_select.config(state=state)
        self.convert_to_unit_select.config(state=state)

    def calculate_para_dep(self, *args):
        if self.quantity_mode.get() == 'Singular Value':
            method = 'svd'
            lb = 'first '
            if self.no_svd_comp.get() > 1:
                lb += str(self.no_svd_comp.get()) + ' '
            self.para_dep_fig.ylabels = ['Singular Value (' + lb + 'comp.)']
        else:
            method = self.quantity_mode.get()
            self.para_dep_fig.ylabels = ['Integral']
        self.window_callback(case='both')
        self.data_obj.calculate_values(
            method=method,
            x_limits=[self.window_x_lower.get(), self.window_x_upper.get()],
            y_limits=[self.window_y_lower.get(), self.window_y_upper.get()])
        self.para_dep_fig.xlabels = [self.data_obj.auto_xlabel()]
        self.para_dep_fig.plot(update_canvas=False)
        self.para_dep_fig.set_legend_visibility(False, update_canvas=True)
        self.update_fit()

    def update_fit(self, *args, auto_range=True):
        if auto_range:
            self.fit_xlower.set(min(self.data_obj.x))
            self.fit_xupper.set(max(self.data_obj.x))
        self.expol_lower.set(1.1 * self.fit_xlower.get() -
                             0.1 * self.fit_xupper.get())
        self.expol_upper.set(1.1 * self.fit_xupper.get() -
                             0.1 * self.fit_xlower.get())
        self.fit_para()

    def window_callback(self, *args, case='x'):
        if case == 'x' or case == 'both':
            if self.window_x_lower.get() >= self.window_x_upper.get():
                self.window_x_lower.set(self.map_fig.xlimits[0][0])
                self.window_x_upper.set(self.map_fig.xlimits[0][1])
            else:
                self.map_fig.set_axes_lim(
                    x=[self.window_x_lower.get(), self.window_x_upper.get()])
        if case == 'y' or case == 'both':
            if self.window_y_lower.get() >= self.window_y_upper.get():
                self.window_y_lower.set(self.map_fig.ylimits[0][1])
                self.window_y_upper.set(self.map_fig.ylimits[0][0])
            else:
                self.map_fig.set_axes_lim(
                    y=[self.window_y_upper.get(), self.window_y_lower.get()])

    def quantity_mode_callback(self, *args):
        if self.quantity_mode.get().lower() == "singular value":
            self.svd_opts_frame.tkraise()
        else:
            self.window_opts_frame.tkraise()

    def scan_folder(self, path):
        ext = []
        for t in (".mat", ".scan", ".txt"):
            for f in os.scandir(path):
                if re.search(t, f.name):
                    ext.append(t)
                    break
        return ext

    def load_scans(self):
        path = tk.filedialog.askdirectory()
        try:
            self.scan_folder(path)
        except Exception:
            return
        else:
            self.progbar_win = CustomProgressbarWindow(self, controller=self)
            self.progbar_win.frame.start_timer()
            self.data_obj.load_scans(path, [".mat", ".scan", ".txt"],
                                     progressbar=self.progbar_win.frame)
            self.sort_scans()

    def map_fig_callback(self, *args):
        open_topfigure(self, fig_obj=self.map_fig, editable=True)

    def para_fig_callback(self, *args):
        open_topfigure(self, fig_obj=self.para_dep_fig, editable=True)

    def sort_scans(self, *args):
        def main_task(*args, **kwargs):
            self.data_obj.sort_scans_and_average(*args, **kwargs)


#                self.after(1, lambda: self.status_label.config(text = lb))
        if not self.data_obj.data_loaded:
            return
        self.is_running = True
        mode_dict = {'filename': 'folder',
                     'scan file': 'power'}
        self.sortmode = mode_dict[self.para_read_mode.get().lower()]
        self.queue = Queue()
        self.task = ThreadedTask(main_task,
                                 after_finished_func=self.after_finishing,
                                 interruptible=True, out_queue=self.queue,
                                 sortmode=self.sortmode)
        self.task.start()
    #     self.listener = threading.Thread(
    #         target = self.update_label, args = (self.queue,))
    #     self.listener.start()
    #     self.queue.join()
    #     print('hi4')
    #     self.data_obj.sort_scans_and_average(
    #         sortmode = mode_dict[self.para_read_mode.get().lower()])
    #     self.map_fig.plot_all()

    # def update_label(self, in_queue):
    #     while self.is_running:
    #         lb = in_queue.get()
    #         if lb is False:
    #             return
    #         self.after(10, self.status_label.config(text = lb[1][0]))

    #         self.after(1)
    #         print(lb)

    def after_finishing(self):
        self.is_running = False
        self.after(1, lambda: self.current_para.set(self.data_obj.x_para))
        self.current_unit.set(self.data_obj.x_unit)
        self.window_y_lower.set(min(self.data_obj.time_delays))
        self.window_y_upper.set(max(self.data_obj.time_delays))
        self.scan_to_plot_select['menu'].delete(0, tk.END)
        for scan in self.data_obj.sorted_scans.keys():
            self.scan_to_plot_select['menu'].add_command(
                label=scan, command=lambda o=scan: self.scan_to_plot.set(o))
        self.scan_to_plot.set(scan)
        self.progbar_win.destroy()

# app = App()
# app.mainloop()
