# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:57:53 2022

@author: bittmans
"""
from time import sleep

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import threading
import re
import ctypes


def idle_function(*args, **kwargs):
    return


def write_to_dict(**kwargs):
    return kwargs


class BlankObject():
    # customizable object to which arbitrary attributes can be added.
    def __init__(self):
        return


def split_string_into_lines(string, max_len=40):
    words = string.split(" ")
    out_string = ""
    i = 0
    while i < len(words):
        line = words[i]
        i += 1
        while len(line) < max_len and i < len(words):
            line = " ".join([line, words[i]])
            i += 1
        out_string = "\n".join([out_string, line])
    return(out_string)


class GlobalSettings(dict):
    # a class for importing global configuration from a (txt) file
    # and accessing them. It is designed predominantely for the
    # astras package, but can be used in other contexts.
    def __init__(self, config_path=None):
        # default settings
        #       only to avoid missing parameters. will be overwritten
        #       if parameter in loaded config file
        default = {'geometry': "1400x780",
                   'init_canv_geom': "174x174",
                   'scrollable': False,
                   'ui_closing_warning': True,
                   '2Dmap_invert_y': True,
                   '2Dmap_transpose': False,
                   'plot_style': 'fast',
                   'fit_kwargs': {},
                   'fit_linestyle': '-',
                   'fit_linewidth': 2,
                   'fit_marker': 'None',
                   'fit_markersize': 5,
                   'fit_color': 'black',
                   'fill_curves': 0,
                   'input_spectral_quantity': 'wavelength',
                   'input_spectral_unit': 'nm',
                   'time_unit': 'ps',
                   'input_time_conversion_factor': 1e3,
                   'time_delay_precision': 10,
                   'xlabel_default': 'wavelength (nm)',
                   'ylabel_default': 'time delay (ps)',
                   'clabel_default': '$\Delta$ Abs. (mOD)',
                   'xlim_lower': 410,
                   'xlim_upper': 690,
                   'rcParams': {},
                   'figure_std_size': [600, 400],
                   'font': "Computer Modern Roman",
                   'xtick_format_sci': False,
                   'ytick_format_sci': False,
                   }
        # fonts for plots using mpl
        self.mpl_font_dct = {'serif': ['Computer Modern Roman',
                                       'DejaVu Serif',
                                       'Bitstream Vera Serif',
                                       'New Century Schoolbook',
                                       'Century Schoolbook L',
                                       'Utopia',
                                       'ITC Bookman',
                                       'Bookman',
                                       'Nimbus Roman No9 L',
                                       'Times New Roman',
                                       'Times',
                                       'Palatino',
                                       'Charter',
                                       'serif'],
                             'sans-serif': ['DejaVu Sans',
                                            'Bitstream Vera Sans',
                                            'Computer Modern Sans Serif',
                                            'Lucida Grande',
                                            'Verdana',
                                            'Geneva',
                                            'Lucid',
                                            'Arial',
                                            'Helvetica',
                                            'Avant Garde',
                                            'sans-serif']
                             }
        for key, value in default.items():
            self[key] = value
        self.default_header = [
            'Start up configuration file for astras and related apps.',
            'Please only change if you know what you are doing.']
        if config_path is not None:
            try:
                self.read_config_file(config_path)
            except Exception:
                try:
                    self.set_mpl_settings()
                except Exception:
                    raise
        else:
            self.set_mpl_settings()

    def read_config_file(self, filepath, write_mpl_params=True):
        dct = {'rcParams': {}}
        with open(filepath, mode='r') as f:
            # skip header
            line = f.readline()
            while line[0] == "#":
                line = f.readline()
            lines = [line]
            lines.extend(f.readlines())
            f.close()
        rcParams_flag = False
        # read variables
        for line in lines:
            try:
                if line[0] == '#':
                    continue
                value = line[re.search(r'=\s*', line).span()[1]:
                             re.search(r'>', line).span()[0]]
                varname = re.findall('(?<=<).+(?==)', line)[0].strip()
            except Exception:
                if re.search('\%\%\%', line):
                    rcParams_flag = False
                    if re.search('rcParams', line):
                        rcParams_flag = True
            else:
                if rcParams_flag:
                    dct['rcParams'][varname] = value
                else:
                    dct[varname] = value
        # write to settings dict
        # process numerical lists, sorted by separator in config file
        num_list_dct = {",": ["xlabel_pos", "ylabel_pos"],
                        "x": ["figure_std_size"]}
        for sep, val in num_list_dct.items():
            for key in val:
                if key in dct.keys():
                    try:
                        dct[key] = np.double(dct[key].strip().split(sep))
                    except Exception:
                        del dct[key]
        # boolean variables
        for key, val in dct.items():
            if str(val).lower() == "true":
                self[key] = True
            elif str(val).lower() == "false":
                self[key] = False
            else:
                self[key] = val
        # misc. (specific) processing
        self['fit_kwargs'] = {}
        for key in ('linestyle', 'marker',
                    'color', 'linewidth'):
            self['fit_kwargs'][key] = self['fit_' + key]
        self['ticklabel_format'] = {}
        for ax in ('x', 'y'):
            self['ticklabel_format'][ax] = {'style': 'plain'}
            try:
                self['ticklabel_format'][ax]['scilimits'] = tuple(
                    int(s)
                    for s in self[ax + 'tick_format_sci'].strip().split(","))
            except Exception:
                if self[ax + 'tick_format_sci'] is True:
                    self['ticklabel_format'][ax] = {'scilimits': (0, 0),
                                                    'style': 'sci'}
            else:
                self['ticklabel_format'][ax]['style'] = 'sci'
        # mpl parameters
        if write_mpl_params:
            self.set_mpl_settings()

    def set_mpl_settings(self):
        plt.style.use(self['plot_style'])
        disabled = []
        for key, value in self['rcParams'].items():
            if key in disabled:
                print(
                    key.join([
                        'rc parameter ',
                        ' disabled. See config file for alternatives.']))
            else:
                mpl.rcParams[key] = value

    def write_config_file(self, filepath, header=None):
        if header is None:
            header = self.default_header
        with open(filepath, mode='w') as f:
            f.truncate(0)
            for line in header:
                f.write(line.join(["#", "\n"]))
            for name, value in self.dct.items():
                f.write("".join(["<", name, " = ", str(value), ">\n"]))
            f.close()


class ThreadedTask(threading.Thread):
    def __init__(self, target_func, *fun_args, after_finished_func=None,
                 interruptible=False, **fun_kwargs):
        threading.Thread.__init__(self)
        self.target_func = lambda: target_func(*fun_args, **fun_kwargs)
        self.output = None
        self.interruptible = interruptible
        self.after_finished_func = after_finished_func

    def run(self, *args, **kwargs):
        self.task_running = True
        if self.interruptible:
            self.task = threading.Thread(target=self.run_function)
            self.task.start()
            while self.task_running:
                sleep(0.1)
        if self.after_finished_func:
            self.after_finished_func()

    def run_function(self):
        self.output = self.target_func()
        self.task_running = False

    def get_id(self, task):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return task._thread_id
        for thread_id, thread in threading._active.items():
            if thread is task:
                return thread_id

    def raise_exception(self, func=None):
        if self.interruptible:
            thread_id = self.get_id(self.task)
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                thread_id, ctypes.py_object(SystemExit))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
                print('Exception raise failure')
            self.task_running = False
            if func is not None:
                func()
            self.handle_exception()

    def handle_exception(self):
        # overwriteable
        return

# class ThreadedTask(threading.Thread):
#     def __init__(self, target_func, *fun_args, after_finished_func=None,
#                  interruptible=False, **fun_kwargs):
#         threading.Thread.__init__(self)
#         self.fun_args = fun_args
#         self.fun_kwargs = fun_kwargs
#         self.target_func = target_func
#         # self.target_func = lambda: target_func(*fun_args, **fun_kwargs)
#         self.output = None
#         self.interruptible = interruptible
#         self.after_finished_func = after_finished_func

#     def run(self, *args, **kwargs):
#         self.task_running = True
#         if self.interruptible:
#             self.task = threading.Thread(target=self.run_function,
#                                          args=self.fun_args,
#                                          kwargs=self.fun_kwargs)
#             self.task.start()
#             while self.task_running:
#                 sleep(0.1)
#         if self.after_finished_func:
#             self.after_finished_func()

#     def run_function(self, *args, **kwargs):
#         # self.output = self.target_func(*self.fun_args, **self.fun_kwargs)
#         self.output = self.target_func(*args, **kwargs)
#         self.task_running = False

#     def get_id(self, task):
#         # returns id of the respective thread
#         if hasattr(self, '_thread_id'):
#             return task._thread_id
#         for thread_id, thread in threading._active.items():
#             if thread is task:
#                 return thread_id

#     def raise_exception(self):
#         if self.interruptible:
#             thread_id = self.get_id(self.task)
#             res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
#                 thread_id, ctypes.py_object(SystemExit))
#             if res > 1:
#                 ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
#                 print('Exception raise failure')
#             self.task_running = False