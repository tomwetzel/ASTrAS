# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:42:08 2022

@author: bittmans
"""
import tkinter as tk
import numpy as np
from tkinter import ttk
from uncertainties import ufloat
from .general import (CustomFrame, center_toplevel, ScrollFrame, GroupBox,
                      move_toplevel_to_default, MultiDisplayWindow,
                      save_box)
from ..fitmethods import FitModels, LineFitParaDicts, lmfitreport_to_dict
from ..dataobjects import TATrace, TATraceCollection
from ..helpers import ThreadedTask, BlankObject
from .figures import TkMplFigure, open_topfigure, FigureOptionWindow
from .traces import TraceCollectionSelection

import re


def get_trace_fit_guess_input(trace_obj, curve_labels=None,
                              guesses_input=None):
    clbl_dict = None
    if curve_labels is None:
        if type(trace_obj) is TATraceCollection:
            clbl_dict = {}
            curve_labels = []
            if len(trace_obj.active_members) < 2:
                for member in trace_obj.active_members:
                    clbl_dict[member] = {}
                    for key in trace_obj.traces[member].active_traces:
                        clbl_dict[member][key] = key
                        curve_labels.append(key)
            else:
                for member in trace_obj.active_members:
                    clbl_dict[member] = {}
                    for key in trace_obj.traces[member].active_traces:
                        curve_labels.append(":\n".join([member, key]))
                        clbl_dict[member][key] = curve_labels[-1]
        elif type(trace_obj) is TATrace:
            curve_labels = trace_obj.active_traces
        else:
            return
    if type(trace_obj) is TATraceCollection:
        if guesses_input is None:
            guess_in = None
        else:
            guess_in = {}
            for member, dct in guesses_input.items():
                for key, val in dct.items():
                    try:
                        guess_in[clbl_dict[member][key]] = guesses_input[
                            member][key]
                    except Exception as e:
                        print(e)
    else:
        guess_in = guesses_input
    return curve_labels, clbl_dict, guess_in


def get_trace_fit_guess_output(out, curve_label_dict=None,
                               guesses_output=None):
    if curve_label_dict is None:
        output = out
    else:
        output = {}
        for member, key_dct in curve_label_dict.items():
            output[member] = {}
            for key, val in key_dct.items():
                output[member][key] = out[val]
    if guesses_output is None:
        return output
    else:
        for key, val in output.items():
            guesses_output[key] = val
        return guesses_output


def get_trace_fit_guesses(parent, fit_obj, trace_obj=None, curve_labels=None,
                          guesses_input=None, guesses_output=None,
                          case='guess', **window_kwargs):
    curve_labels, curve_label_dict, guess_in = get_trace_fit_guess_input(
        trace_obj, curve_labels=curve_labels, guesses_input=guesses_input)
    window = FitParaEntryWindowLmfit(parent, curve_labels, fit_obj,
                                     input_values=guess_in, case=case,
                                     **window_kwargs)
    parent.wait_window(window)
    return get_trace_fit_guess_output(window.output,
                                      curve_label_dict=curve_label_dict,
                                      guesses_output=guesses_output)


# %%
class FitParaOptsWindow(tk.Toplevel):
    def __init__(self, parent, *args, controller=None, **kwargs):
        tk.Toplevel.__init__(self, parent)
        self.fr = FitParaOpts(self, *args, **kwargs)
        self.fr.grid(row=0, column=0, sticky='wnse')
        button_frame = CustomFrame(self, dim=(2, 1))
        ttk.Button(button_frame, text="OK",
                   command=lambda: self._closing_func(True)).grid(
                       row=0, column=0, padx=5, pady=5, sticky='e')
        ttk.Button(button_frame, text="Cancel",
                   command=lambda: self._closing_func(False)).grid(
                       row=0, column=1, padx=5, pady=5, sticky='w')
        button_frame.grid(row=1, column=0, sticky='wnse')
        self.write_parameters = self.fr.write_parameters
        self.get_output = self.fr.get_output
        self.protocol("WM_DELETE_WINDOW", lambda: self._closing_func(False))
        if controller is not None:
            center_toplevel(self, controller)

    def _closing_func(self, success):
        self.output = None
        if success:
            self.output = self.fr.get_output()
        self.destroy()


class FitParaOpts(tk.Frame):
    def __init__(self, parent, params, curve_labels=None, input_values=None,
                 name_dict=None, window_mode='select', exclude_para=None,
                 scrollable=False, width=320, height=150, columnwidth=8,
                 grid_kwargs=None):
        tk.Frame.__init__(self, parent)
        if name_dict is None:
            self.pname_dict, axis_dct, shared = (
                LineFitParaDicts().get_attributes())
        else:
            self.pname_dict = name_dict
        if curve_labels is None:
            self.curves = [""]
        else:
            self.curves = curve_labels
        if exclude_para is None:
            self.exclude = []
        else:
            self.exclude = exclude_para
        if grid_kwargs is None:
            self.grid_kwargs = {'sticky': 'w',
                                'padx': 5,
                                'pady': 5}
        else:
            self.grid_kwargs = grid_kwargs
        self.params = params
        self.window_mode = window_mode
        self.columnwidth = columnwidth
        self.frames = {}
        self.vars = {}
        self.widgets = {}
        self.labels = []
        self.output = None

        if scrollable:
            self.wdgt_frame = ScrollFrame(self, scrolldir='y')
        else:
            self.wdgt_frame = tk.Frame(self)

        row = self.setup_frames()
        self.setup_widgets(params, input_values=input_values)
        self.wdgt_frame.grid(row=row, column=0, sticky='wnse')

    def setup_frames(self, row=0, curves=None, window_mode=None):
        if curves is None:
            curves = self.curves
        if window_mode is None:
            window_mode = self.window_mode
        for fr in self.frames.values():
            fr.grid_forget()
            del fr
        self.frames = {}
        if re.search('select', window_mode, re.I) and len(curves) > 1:
            self.curve = tk.StringVar(value=curves[0])
            self.curve_select = ttk.OptionMenu(
                self, self.curve, self.curve.get(), *curves,
                command=self._curve_select_callback)
            self.curve_select.grid(row=row, column=0)
            row += 1
            for curve in curves:
                self.frames[curve] = CustomFrame(self.wdgt_frame, dim=(5, 1))
                self.frames[curve].grid(row=0, column=0, sticky='wnse')
            self.frames[curves[0]].tkraise()
        else:
            for i, curve in enumerate(curves):
                self.frames[curve] = CustomFrame(self.wdgt_frame)
                self.frames[curve].grid(row=i, column=0, sticky='wnse')
        self.curves = curves
        self.window_mode = window_mode
        return row

    def setup_widgets(self, params=None, input_values=None, exclude_para=None):
        if params is None:
            params = self.params
        else:
            self.params = params
        if exclude_para is None:
            exclude_para = self.exclude
        else:
            self.exclude = exclude_para
        if input_values is None:
            values = {}
            for curve in self.frames.keys():
                values[curve] = {}
        else:
            values = input_values
        for key, vars in self.vars.items():
            for k in vars.keys():
                del self.vars[key][k]
                self.widgets[key][k].grid_forget()
                del self.widgets[key][k]
        for lb in self.labels:
            lb.grid_forget()
            del lb
        self.vars = {}
        self.widgets = {}
        self.labels = []
        self.output = None

        for curve, frame in self.frames.items():
            self.labels.append(ttk.Label(frame, text=str(curve)))
            self.labels[-1].grid(row=0, column=0,
                                 columnspan=4)
            for i, header in enumerate(['Parameter', 'Initial', 'Fix', 'Min',
                                        'Max']):
                ttk.Label(frame, text=header).grid(row=0, column=i,
                                                   **self.grid_kwargs)
            row = 2
            for p in params:
                if p in exclude_para or params[p].expr:
                    continue
                if p not in values[curve].keys():
                    values[curve][p] = {'val': params[p].value,
                                        'fix': not params[p].vary,
                                        'min': params[p].min,
                                        'max': params[p].max}
                key = '_'.join([str(curve), p])
                self.vars[key] = {}
                self.widgets[key] = {}
                try:
                    pname = self.pname_dict[p]
                except Exception:
                    pname = p
                self.labels.append(ttk.Label(frame, text=pname))
                self.labels[-1].grid(row=row, column=0, **self.grid_kwargs)
                for i, k in enumerate(('val', 'fix', 'min', 'max')):
                    value = values[curve][p][k]
                    if k == 'fix':
                        self.vars[key][k] = tk.IntVar(value=value)
                        self.widgets[key][k] = ttk.Checkbutton(
                            frame, variable=self.vars[key][k])
                    else:
                        self.vars[key][k] = tk.DoubleVar(value=value)
                        self.widgets[key][k] = ttk.Entry(
                            frame, textvariable=self.vars[key][k])
                    self.widgets[key][k].config(width=self.columnwidth)
                    self.widgets[key][k].grid(
                        row=row, column=i + 1, **self.grid_kwargs)
                row += 1

    def write_parameters(self, params):
        self.get_output()
        para = self.output[self.curves[0]]
        for p in params:
            if p in para.keys():
                params[p].value = para[p]['val']
                params[p].vary = not para[p]['fix']
                params[p].min = para[p]['min']
                params[p].max = para[p]['max']
        return params

    def get_output(self):
        self.output = {}
        for curve in self.frames.keys():
            self.output[curve] = {}
            for p in self.params:
                if p in self.exclude or self.params[p].expr:
                    continue
                self.output[curve][p] = {}
                for k in ('val', 'fix', 'min', 'max'):
                    self.output[curve][p][k] = self.vars['_'.join(
                        [str(curve), p])][k].get()
        if not self.output:
            return
        else:
            return self.output

    def _curve_select_callback(self, e):
        self.frames[self.curve.get()].tkraise()


# %%
class FitParaEntryWindowLmfit(tk.Toplevel):
    def __init__(self, parent, *args, controller=None, mode='enter',
                 case='guesses', **kwargs):
        tk.Toplevel.__init__(self, parent)
        if controller is not None:
            move_toplevel_to_default(self, controller)
        if mode == "enter":
            title = "Enter "
            title += "Guesses" if re.search('guess', case, re.I) else "Bounds"
        else:
            title = "Results" if re.search('guess', case, re.I) else "Bounds"
        self.title(title)
        self.fr = FitParaEntryLmfit(
            self, *args, mode=mode, case=case, **kwargs)
        self.setup_table = self.fr.setup_table
        self.fr.grid(row=0, column=0, sticky='wnse')
        button_frame = CustomFrame(self, dim=(2, 1))
        ttk.Button(button_frame, text="OK",
                   command=lambda: self._closing_func(True)).grid(
            row=0, column=0, padx=5, pady=5, sticky='e')
        ttk.Button(button_frame, text="Cancel",
                   command=lambda: self._closing_func(False)).grid(
            row=0, column=1, padx=5, pady=5, sticky='w')
        button_frame.grid(row=1, column=0, sticky='wnse')
        self.protocol("WM_DELETE_WINDOW", lambda: self._closing_func(False))
        if controller is not None:
            center_toplevel(self, controller)

    def _closing_func(self, success):
        self.output = {}
        if success:
            self.output = self.fr.get_output()
        self.destroy()


class FitParaEntryLmfit(tk.Frame):
    def __init__(self, parent, curve_labels, fit_obj=None, params=None,
                 input_values=None, para_names=None, pos_dict=None,
                 mode='enter', case='guesses', name_dict=None,
                 scrollable=False, exclude_para=None, headers=None, title=None,
                 grid_kwargs=None, xunit='ps', width=320, height=150,
                 entry_kwargs=None, show_table=True,
                 fix_para_opt=True):
        tk.Frame.__init__(self, parent)

        names, self._para_axis_dict, self._shared_paras = (
            LineFitParaDicts().get_attributes())
        if name_dict is None:
            self.name_dict = names
        else:
            self.name_dict = name_dict
        if headers is None:
            headers = {"curve": "Curve",
                       "comp": "Component",
                       "x": "Lifetime (ps)",
                       "y": "Amplitude"}
        self.headers = headers

        self.output = {}
        self.vars = {}
        self.labels = {}
        self.entries = {}
        self.fit_obj = fit_obj
        self.curve_labels = curve_labels
        self.scrollable = scrollable

        if entry_kwargs:
            self.entry_kwargs = entry_kwargs
        else:
            self.entry_kwargs = {"width": 10,
                                 "justify": "right"}
        if not exclude_para:
            self.exclude_para = []
        else:
            self.exclude_para = exclude_para

        self.canvas = tk.Canvas(self)
        if width and height:
            self.canvas.config(width=width, height=height)
        self.table_frame = tk.Frame(self.canvas)
        if self.scrollable:
            scrollcheck = False
            if re.search('y', self.scrollable, re.I):
                scrollcheck = True
                self.scrollbary = tk.Scrollbar(self, orient="vertical",
                                               command=self.canvas.yview)
                self.canvas.configure(yscrollcommand=self.scrollbary.set)
                self.scrollbary.grid(row=0, column=1, sticky='sne')
            if re.search('x', self.scrollable, re.I):
                scrollcheck = True
                self.scrollbarx = tk.Scrollbar(self, orient="horizontal",
                                               command=self.canvas.xview)
                self.canvas.configure(xscrollcommand=self.scrollbarx.set)
                self.scrollbarx.grid(row=1, column=0, sticky='sew')
            if scrollcheck:
                self.table_frame.bind(
                    "<Configure>", lambda *args: self.canvas.config(
                        scrollregion=self.canvas.bbox("all")))
                self.canvas.bind("<Configure>", self._config_canvas)
                self.frame_win = self.canvas.create_window(
                    (1, 1), window=self.table_frame, anchor='nw',
                    tags="self.table_frame")
            else:
                self.table_frame.grid(sticky='wnse')
        else:
            self.table_frame.grid(sticky='wnse')
        self.set_mode(mode=mode, case=case)
        if show_table:
            self.setup_table(params=params, input_values=input_values,
                             para_names=para_names, headers=headers,
                             grid_kwargs=grid_kwargs)
        self.canvas.grid(sticky='wnse', row=0, column=0, padx=10, pady=10)

    def _config_canvas(self, e):
        self.canvas.itemconfig(
            self.frame_win, height=e.height, width=e.width)

    def set_mode(self, mode="enter", case="guess"):
        self.mode = mode
        self.case = case
        if mode == "enter":
            if case == "bounds":
                self._widget_func = self._place_double_entry
            else:
                self.case = "guess"
                self._widget_func = self._place_entry
        else:
            if case == "bounds":
                self._widget_func = self._place_double_label
            else:
                self.case = "guess"
                self._widget_func = self._place_label

    def set_values(self, input_dict, overwrite=True):
        for curve, dct in input_dict.items():
            if curve not in self.vars.keys():
                continue
            for key, val in dct.items():
                if ((not overwrite and key in self.vars[curve].keys())
                        or key in self.exclude_para):
                    continue
                try:
                    value = val.value
                except Exception:
                    value = val
                try:
                    self.vars[curve][key].set(value)
                except Exception:
                    pass

    def setup_table(self, curve_labels=None, params=None, fit_obj=None,
                    input_values=None, para_names=None, headers=None,
                    grid_kwargs=None, exclude_para=None):
        def set_table_entry(table, text, name, pname, comp_no, value):
            for i, key in enumerate(headers.keys()):
                if key == self._para_axis_dict[pname]:
                    break
            if text not in table.keys():
                table[text] = {}
            if comp_no not in table[text].keys():
                table[text][comp_no] = {}
            table[text][comp_no][name] = [value, i - self.column_offset]
            return table

        if exclude_para is None:
            exclude_para = self.exclude_para
        elif not exclude_para:
            exclude_para = []
        for val in self.entries.values():
            for entry in val.values():
                entry.grid_forget()
                del entry
        self.entries = {}
        for val in self.labels.values():
            for label in val:
                label.grid_forget()
                del label
        self.labels = {'headers': []}

        if params is None:
            if not fit_obj:
                fit_obj = self.fit_obj
            else:
                self.fit_obj = fit_obj
            params = fit_obj.params

        if not curve_labels:
            curve_labels = self.curve_labels
        else:
            self.curve_labels = curve_labels
        if grid_kwargs is None:
            grid_kwargs = {'sticky': 'w', 'padx': 5, 'pady': 5}
        if headers is None:
            headers = self.headers
        else:
            hd = {}
            if "curve" not in headers.keys():
                hd["curve"] = "Curve"
            if "comp" not in headers.keys():
                hd["comp"] = "Component"
            for key, val in headers.items():
                hd[key] = val
            headers = hd
            self.headers = headers

        for first_column, h in enumerate(headers.keys()):
            if h[0] in ('x', 'y', 'z'):
                break
        self.column_offset = first_column
        if input_values is None:
            input_values = {}
        if not para_names:
            para_names = [p for p in params
                          if p not in exclude_para and not params[p].expr]
        else:
            pnames = []
            for name in para_names:
                if name not in exclude_para:
                    for p in params:
                        if re.search(name, p, re.I) and not params[p].expr:
                            pnames.append(p)
            para_names = pnames
        if re.search("bound", self.case, re.I):
            for i, k in enumerate(headers.keys()):
                cspan = 2 if k not in ["curve", "comp"] else 1
                self.labels['headers'].append(
                    ttk.Label(self.table_frame, text=headers[k]))
                if i < self.column_offset:
                    self.labels['headers'][-1].grid(
                        row=0, column=i, columnspan=cspan, **grid_kwargs)
                else:
                    self.labels['headers'][-1].grid(
                        row=0, column=2 * i - self.column_offset,
                        columnspan=cspan, **grid_kwargs)
        else:
            for i, k in enumerate(headers.keys()):
                self.labels['headers'].append(
                    ttk.Label(self.table_frame, text=headers[k]))
                self.labels['headers'][-1].grid(row=0,
                                                column=i, **grid_kwargs)
        row = 1
        for curve in curve_labels:
            values = {}
            self.labels[curve] = []
            if re.search('enter', self.mode):
                if curve not in self.vars.keys():
                    self.vars[curve] = {}
                if curve not in self.entries.keys():
                    self.entries[curve] = {}
            shared_params = set()
            table = {}
            # get entry values from input or lmfit parameters
            if curve in input_values.keys():
                for p in para_names:
                    if p in input_values[curve].keys():
                        values[p] = input_values[curve][p]
                    else:
                        if re.search('guess', self.case, re.I):
                            values[p] = params[p].value
                        else:
                            values[p] = [params[p].min, params[p].max]
            else:
                for p in para_names:
                    if re.search('guess', self.case, re.I):
                        values[p] = params[p].value
                    else:
                        values[p] = [params[p].min, params[p].max]
            # curve label
            self.labels[curve].append(ttk.Label(self.table_frame, text=curve))
            self.labels[curve][-1].grid(row=row,
                                        column=0, rowspan=2, sticky='nw')

            for name in para_names:
                try:
                    if re.search('_\d', name):
                        func = name[:re.search("_", name).span()[0]]
                    else:
                        func = name
                    if func in self._shared_paras:
                        shared_params.add(name)
                        continue
                    comp_no = re.findall("(\d+)", name)[-1]
                    text = self.name_dict[func]
                    pname = name[re.search("_", name).span()[
                        1]:re.search("_(\d+)", name).span()[0]]
                except Exception:
                    continue
                value = values[name]
                table = set_table_entry(
                    table, text, name, pname, comp_no, value)
            for name in shared_params:
                try:
                    pname = name[:re.search("_", name).span()[0]]
                except Exception:
                    pname = name
                text = self.name_dict[pname]
                value = values[name]
                table = set_table_entry(table, text, name, pname, '1', value)
            for key, dct in table.items():
                if len(list(dct.keys())) > 1:
                    def text(i): return " ".join([key, i])
                else:
                    def text(i): return key
                for comp, values in dct.items():
                    try:
                        self.labels[curve].append(
                            ttk.Label(self.table_frame, text=text(comp)))
                        self.labels[curve][-1].grid(row=row, column=1,
                                                    **grid_kwargs)
                    except Exception:
                        pass
                    for pname, val in values.items():
                        value = val[0]
                        if curve in self.vars.keys():
                            if pname in self.vars[curve].keys():
                                value = self.vars[curve][pname].get()
                        self._widget_func(curve, pname, val[0], row=row,
                                          column=val[1], **grid_kwargs)
                    row += 1
        if self.scrollable:
            self.update()
            self.frame_win = self.canvas.create_window(
                (1, 1), window=self.table_frame, anchor='nw',
                tags="self.table_frame")

    def set_axis_assignment(self, para, axis, **update_kwargs):
        try:
            self._para_axis_dict[para] = axis
        except Exception:
            return
        else:
            self.setup_table(**update_kwargs)

    def get_output(self):
        self.output = {}
        for curve, var in self.vars.items():
            self.output[curve] = {}
            for key, val in var.items():
                if type(val) is list:
                    self.output[curve][key] = []
                    for value in val:
                        try:
                            self.output[curve][key].append(
                                np.double(value.get()))
                        except ValueError:
                            self.output[curve][key].append(0)
                            val.set(0)
                        except Exception:
                            raise
                else:
                    try:
                        self.output[curve][key] = np.double(val.get())
                    except ValueError:
                        self.output[curve][key] = 0
                        val.set(0)
                    except Exception:
                        raise
        return self.output

    def _place_entry(self, curve, name, value, column=0, **grid_kwargs):
        self.vars[curve][name] = tk.StringVar(value=str(value))
        self.entries[curve][name] = ttk.Entry(
            self.table_frame, textvariable=self.vars[curve][name],
            **self.entry_kwargs)
        self.entries[curve][name].grid(
            column=column + self.column_offset, **grid_kwargs)

    def _place_double_entry(self, curve, name, value, column=0, **grid_kwargs):
        self.vars[curve][name] = [tk.StringVar(
            value=str(value[0])), tk.StringVar(value=str(value[1]))]
        self.entries[curve][name] = [
            ttk.Entry(self.table_frame, textvariable=self.vars[curve][name][0],
                      **self.entry_kwargs),
            ttk.Entry(self.table_frame, textvariable=self.vars[curve][name][1],
                      **self.entry_kwargs)]
        self.entries[curve][name][0].grid(
            column=2*column + self.column_offset, **grid_kwargs)
        self.entries[curve][name][1].grid(
            column=2*column + self.column_offset + 1, **grid_kwargs)

    def _place_label(self, key, name, value, column=0, **grid_kwargs):
        ttk.Label(self.table_frame, text=str(value)).grid(
            column=column + self.column_offset, **grid_kwargs)

    def _place_double_label(self, key, name, value, column=0, **grid_kwargs):
        ttk.Label(self.table_frame, text=str(value[0])).grid(
            column=2*column + self.column_offset, **grid_kwargs)
        ttk.Label(self.table_frame, text=str(value[1])).grid(
            column=2*column + self.column_offset + 1, **grid_kwargs)


# %%
class FitResultsDisplayLmfit(ttk.Treeview):
    def __init__(self, parent, column_dict=None, display_error=True,
                 precision=6, name_dict=None, scrollable=False,
                 **treeview_kwargs):
        ttk.Treeview.__init__(self, parent, **treeview_kwargs)
        names, self._para_axis_dict, self._shared_paras = (
            LineFitParaDicts().get_attributes())
        if column_dict is None:
            self.column_dict = {"#0": {"heading": {"text": "Curve",
                                                   "anchor": "w"},
                                       "column": {"width": 220,
                                                  "stretch": False},
                                       "axis": None},
                                "#1": {"heading": {"text": "Lifetime (ps)",
                                                   "anchor": "w"},
                                       "column": {"width": 100,
                                                  "stretch": False},
                                       "axis": "x"},
                                "#2": {"heading": {"text": "Amplitude",
                                                   "anchor": "w"},
                                       "column": {"width": 100,
                                                  "stretch": True},
                                       "axis": "y"}}
        else:
            self.column_dict = column_dict
        if name_dict is None:
            self.name_dict = names
        else:
            self.name_dict = name_dict

        self.setup_columns(self.column_dict)
        self.display_error = display_error
        self.precision = precision

    def _config_canvas(self, e):
        self.canvas.itemconfig(
            self.frame_win, height=e.height, width=e.width)

    def setup_columns(self, col_dct):
        self["columns"] = list(col_dct.keys())[1:]
        for key, val in col_dct.items():
            self.heading(key, **val["heading"])
            self.column(key, **val["column"])
        self.column_dict = col_dct

    def update_columns(self):
        return

    def show_results(self, results, columns=None, display_error=None):
        def display_para(curve_key, para_name, disp_error=True):
            vary = results[curve_key]['params'][para_name].vary
            if vary and disp_error:
                return ufloat(results[curve_key]['params'][para_name],
                              results[curve_key]['stderrors'][para_name])
            elif vary:
                return np.round(results[curve_key]['params'][para_name],
                                self.precision)
            else:
                return str(np.round(results[curve_key]['params'][para_name],
                                    self.precision)) + " (fixed)"

        def set_table_entry(table, text, pname, comp_no, value):
            for i, key in enumerate(self.column_dict.keys()):
                if (self.column_dict[key]["axis"] ==
                        self._para_axis_dict[pname]):
                    break
            if text not in table.keys():
                table[text] = {}
            if comp_no not in table[text].keys():
                table[text][comp_no] = [
                    "-" for k in self.column_dict.keys()][:-1]
            table[text][comp_no][i - 1] = value
            return table

        self.delete(*self.get_children())
        if columns is None:
            columns = self.column_dict
        if columns:
            self.setup_columns(columns)
        if display_error is None:
            display_error = self.display_error
        else:
            self.display_error = display_error
        for curve, res in results.items():
            if not res['success']:
                continue
            if res['stderrors'] is None:
                disp_error = False
            else:
                disp_error = display_error

            def disp_para(c, p): return display_para(
                c, p, disp_error=disp_error)
            tree_master = self.insert("", "end", text=curve, open=True)
            shared_params = set()
            table = {}
            for para in res['params']:
                if re.search("_div_|_minus_", para, re.I):
                    continue
                try:
                    func = para[:re.search("_", para).span()[0]]
                    if func in self._shared_paras:
                        shared_params.add(para)
                        continue
                    comp_no = re.findall("(\d+)", para)[-1]
                    text = self.name_dict[func]
                    pname = para[re.search("_", para).span()[
                        1]:re.search("_(\d+)", para).span()[0]]
                except Exception:
                    raise
                value = disp_para(curve, para)
                table = set_table_entry(table, text, pname, comp_no, value)

            for para in shared_params:
                value = disp_para(curve, para)
                pname = para[:re.search("_", para).span()[0]]
                text = self.name_dict[pname]
                table = set_table_entry(table, text, pname, 1, value)

            for key, dct in table.items():
                if len(list(dct.keys())) > 1:
                    def text(func, i): return " ".join([func, i])
                else:
                    def text(func, i): return func
                for j in dct.keys():
                    self.insert(tree_master, "end",
                                text=text(key, j), values=dct[j])


# %%
class SpectralFitOptions(CustomFrame):
    def __init__(self, parent, controller, xunit=None, **kwargs):
        CustomFrame.__init__(self, parent, **kwargs)
        self.controller = controller
        self.frames = {}
        self.vars = {}
        self.checks = {}
        self.labels = {}
        self.optmenus = {}
        if xunit:
            self.xunit = xunit
        else:
            self.xunit = self.settings['input_spectral_unit']

        self.model_opts = CustomFrame(self, border=True)
        self.plot_opts = CustomFrame(self, border=True)

        tk.Label(self.model_opts, text="Model").grid(
            row=0, column=0, sticky='nw')
        tk.ttk.Label(self.model_opts, text="Line Shape").grid(
            row=1, column=0, sticky='nw')
        self.vars['line_shape'] = tk.StringVar(value="Gaussian")
        self.optmenus['line_shape'] = tk.ttk.OptionMenu(
            self.model_opts, self.vars['line_shape'],
            'Gaussian', 'Gaussian', 'Lorentzian',
            command=self.update_fit_parameters)
        self.optmenus['line_shape'].grid(row=1, column=1, sticky='wne')
        tk.Label(self.model_opts, text="No. of components").grid(
            row=2, column=0, sticky='nw')
        self.vars['num_comp'] = tk.IntVar(value=2)
        tk.ttk.OptionMenu(
            self.model_opts, self.vars['num_comp'], 2, 1, 2, 3, 4,
            command=self.update_fit_parameters).grid(
                row=2, column=1, sticky='nw')

        tk.Label(self.model_opts, text="Line width model").grid(
            row=3, column=0, sticky='nw')
        self.vars['width_constr'] = tk.StringVar(value='unconstrained')
        tk.ttk.OptionMenu(
            self.model_opts, self.vars['width_constr'],
            self.vars['width_constr'].get(), 'unconstrained', 'linear',
            command=self.linewidth_model_callback).grid(
                row=3, column=1, sticky='nw')
        self.linewidth_model_opts = tk.Frame(self.model_opts)
        tk.ttk.Label(self.linewidth_model_opts, text="Slope:").grid(
            row=0, column=0, sticky='w')
        self.vars['wdfactor'] = tk.DoubleVar(value=500)
        tk.ttk.Entry(
            self.linewidth_model_opts, textvariable=self.vars['wdfactor'],
            width=8).grid(row=0, column=1, sticky='w')
        self.vars['fix_wd_factor'] = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.linewidth_model_opts, text="Fix",
            variable=self.vars['fix_wd_factor']).grid(
                row=0, column=3, sticky='w')
        self.linewidth_model_opts.grid(
            row=4, column=0, columnspan=2, sticky='wnse')

        self.linewidth_model_opts.grid_remove()
        tk.ttk.Label(self.model_opts).grid(
            row=4, column=0, columnspan=2, sticky='wnse')

        self.vars['equal_space'] = tk.IntVar(value=0)
        self.equal_space_check = tk.ttk.Checkbutton(
            self.model_opts, text="Equal Spacing",
            variable=self.vars['equal_space'],
            command=self.spacing_callback)
        self.equal_space_check.grid(row=5, column=0, sticky='nw')
        self.vars['add_offset'] = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.model_opts, text="Offset",
            variable=self.vars['add_offset'],
            command=self.update_fit_parameters).grid(
                row=5, column=1, sticky='nw')
        self.spacing_opts = tk.Frame(self.model_opts)
        tk.ttk.Label(self.spacing_opts, text="Spacing:").grid(
            row=0, column=0, sticky='w')
        self.vars['spacing'] = tk.DoubleVar(value=1000)
        tk.ttk.Entry(
            self.spacing_opts, textvariable=self.vars['spacing'],
            width=8).grid(row=0, column=1, sticky='w')
        self.vars['fix_spacing'] = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.spacing_opts, text="Fix", variable=self.vars['fix_spacing'],
            command=self.update_fit_parameters).grid(
                row=0, column=2, sticky='w')
        self.spacing_opts.grid(row=6, column=0, columnspan=2, sticky='wnse')
        self.spacing_opts.grid_remove()
        tk.ttk.Label(self.model_opts).grid(
            row=6, column=0, columnspan=2, sticky='wnse')

        self.model_opts.grid(row=0, column=0, sticky='wnse', padx=5, pady=5)

        tk.ttk.Label(self.plot_opts, text="X axis (plot):").grid(
            row=0, column=0, sticky='w', padx=5, pady=5)
        self.vars['xmode'] = tk.StringVar(value="wavenumber")
        self.optmenus['xmode'] = tk.ttk.OptionMenu(
            self.plot_opts, self.vars['xmode'], "wavenumber", "wavenumber",
            "wavelength")
        self.vars['xmode'].trace('w', self.xaxis_select_callback)
        self.optmenus['xmode'].grid(
            row=0, column=1, sticky='we', padx=5, pady=5)
        self.plot_opts.grid(row=1, column=0, sticky='wnse', padx=5, pady=5)

    def selection_callback(self, fitpage, trace_obj, *args, **kwargs):
        fitpage.fit_init = trace_obj.init_line_shape_fit
        try:
            fitpage.plot_resid_fft.set(0)
            fitpage.plot_resid_fft_check.config(state='disabled')
        except Exception:
            pass
        for key, var in self.vars.items():
            fitpage.vars[key] = var
        self.set_result_disp_headers()
        self.xaxis_select_callback()

    def spacing_callback(self, *args):
        if self.vars['equal_space'].get():
            self.spacing_opts.grid()
            self.spacing_opts.tkraise()
        else:
            try:
                self.spacing_opts.grid_remove()
            except Exception:
                pass
        self.update_fit_parameters()

    def linewidth_model_callback(self, *args):
        if self.vars['width_constr'].get() == 'linear':
            self.linewidth_model_opts.grid()
            self.linewidth_model_opts.tkraise()
        else:
            try:
                self.linewidth_model_opts.grid_remove()
            except Exception:
                pass
        self.update_fit_parameters()

    def set_figure_labels(self, figure, xlabel=None, update_canvas=False):
        figure.set_axes_label(x=xlabel, update_canvas=update_canvas)

    def get_init_table_kwargs(self, **kwargs):
        headers = {"curve": "Curve",
                   "comp": "Component",
                   "x": self.xunit.join(["Position (", ")"]),
                   "y": "Amplitude"}
        if re.search('gauss', self.vars['line_shape'].get(), re.I):
            headers['x2'] = self.xunit.join(["Sigma (", ")"])
        else:
            headers['x2'] = self.xunit.join(["Gamma (", ")"])
        return {"headers": headers, "exclude_para": ['spacing', 'wdfactor']}

    def set_result_disp_headers(self, **kwargs):
        columns = {"#0": {"heading": {"text": "Curve", "anchor": "w"},
                          "column": {"width": 250, "stretch": False},
                          "axis": None},
                   "#1": {"heading":
                          {"text": self.xunit.join(
                              ["Position (", ")"]), "anchor": "w"},
                          "column": {"width": 75, "stretch": False},
                          "axis": "x"},
                   "#2": {"heading": {"text": "Amplitude", "anchor": "w"},
                          "column": {"width": 75, "stretch": True},
                          "axis": "y"},
                   "#3": {"heading":
                          {"text": self.xunit.join(
                              ["Sigma (", ")"]), "anchor": "w"},
                          "column": {"width": 75, "stretch": True},
                          "axis": "y"}}
        self.controller.update_results_disp_header(col_dct=columns)

    def get_specific_inits(self):
        return {}

    def set_result_table(self, *args, **kwargs):
        return False

    def get_fit_kwargs(self):
        constraints = {'width': self.vars['width_constr'].get(),
                       'spacing': 'equal' if self.vars['equal_space'].get()
                       else ""}
        constraint_inits = {'width': self.vars['wdfactor'].get(),
                            'spacing': self.vars['spacing'].get()}
        fixed_para = {}
        for key in ['spacing']:
            if self.vars['fix_' + key].get():
                fixed_para[key] = self.vars[key].get()
        kwargs = {'num_comp': self.vars['num_comp'].get(),
                  'line_shape': self.vars['line_shape'].get().lower(),
                  'constraints': constraints,
                  'offset': bool(self.vars['add_offset'].get()),
                  'constraint_inits': constraint_inits,
                  'fixed_para': fixed_para}
        return kwargs

    def update_fit_parameters(self, *args, update_table=True):
        fit_obj_kwargs = self.get_fit_kwargs()
        self.controller.update_fit_parameters(
            update_table=update_table, fit_obj_kwargs=fit_obj_kwargs)

    def xaxis_select_callback(self, *args):
        current_xunit = self.xunit
        active = {}
        coll = self.controller.trace_coll
        for member in coll.active_members:
            active[member] = [tr for tr in coll.traces[
                member].active_traces]
        coll.active_members = coll.traces.keys()
        for m in coll.active_members:
            coll.traces[m].active_traces = coll.traces[m].tr.keys()
        try:
            xunit = self.controller.trace_coll.set_x_mode(
                mode=self.vars['xmode'].get())
        except Exception:
            self.vars['xmode'].set('wavenumber')
            xunit = self.controller.trace_coll.set_x_mode('wavenumber')
        self.xunit = xunit.replace('$', '')
        self.controller.fit_range_label.config(
            text=self.xunit.join(['Fit range (', ')']))
        xlabel = None
        if current_xunit != self.xunit:
            for member in coll.active_members:
                for trace in coll.traces[member].active_traces:
                    try:
                        xlabel = trace.xlabel
                    except Exception:
                        pass
                    else:
                        break
            if not xlabel:
                xlabel = "".join([self.vars['xmode'].get(), " (", xunit, ")"])
        coll.active_members = []
        for member, traces in active.items():
            coll.active_members.append(member)
            coll.traces[member].active_traces = []
            for tr in traces:
                coll.traces[member].active_traces.append(tr)
        if current_xunit != self.xunit:
            self.set_figure_labels(self.controller.fit_figure, xlabel=xlabel)
            self.set_result_disp_headers()
            self.controller.auto_axis_limits()
        try:
            self.controller.update_plot()
        except Exception:
            raise

    def auto_axis_limits(self, *args, xlim=None, **kwargs):
        if xlim is None:
            xmin = []
            xmax = []
            for member in self.controller.trace_coll.active_members:
                tr_collect = self.controller.trace_coll.traces[member]
                x = tr_collect.xdata
                for trace in tr_collect.active_traces:
                    valid = ~np.isnan(tr_collect.tr[trace]['y'])
                    xmin.append(x[valid][0])
                    xmax.append(x[valid][-1])
            xlim = [np.min(xmin), np.max(xmax)]
        self.controller.xlower.set(min(xlim))
        self.controller.xupper.set(max(xlim))


# %%
class KineticFitOptions(CustomFrame):
    def __init__(self, parent, controller, xunit=None,
                 ylabel="$\Delta$ Abs. (mOD)", kin_model_input=None,
                 **kwargs):
        CustomFrame.__init__(self, parent, **kwargs)

        self.controller = controller
        self.ylabel = ylabel
        self.func_str = None
        self.frames = {}
        self.vars = {}
        self.checks = {}
        self.labels = {}
        self.optmenus = {}
        self.kin_model_input = kin_model_input
        if xunit:
            self.xunit = xunit
        else:
            self.xunit = "ps"

        self.frames['exp_opts'] = CustomFrame(
            self, dim=(3, 2), border=True)
        self.frames['modul_opts'] = CustomFrame(self, dim=(3, 2), border=True)

        tk.Label(self.frames['exp_opts'],
                 text='No. of exponentials:').grid(row=0, column=0,
                                                   sticky=tk.W, columnspan=2)
        self.num_exp = 1
        self.vars['num_exp_total'] = tk.IntVar(value=self.num_exp)
        self.num_exp_select = tk.ttk.OptionMenu(
            self.frames['exp_opts'], self.vars['num_exp_total'],
            self.num_exp, 1, 2, 3, 4, 5, 6, 7,
            command=self.update_fit_parameters)
        self.num_exp_select.grid(row=0, column=2, columnspan=2, sticky=tk.W)

        self.t0_label = tk.Label(
            self.frames['exp_opts'], text="time zero (ps):")
        self.t0_label.grid(row=1, column=0, sticky=tk.W)
        self.vars['time_zero'] = tk.DoubleVar(value=0.0)
        self.t0_entry = tk.Entry(
            self.frames['exp_opts'],
            textvariable=self.vars['time_zero'], width=8)
        self.t0_entry.grid(row=1, column=1, sticky=tk.W)

        self.vars['time_zero_fixed'] = tk.IntVar(value=1)
        self.checks['time_zero_fixed'] = tk.ttk.Checkbutton(
            self.frames['exp_opts'], text='Fix time zero',
            variable=self.vars['time_zero_fixed'], state='disabled')
        self.checks['time_zero_fixed'].grid(row=1, column=2, sticky=tk.W)

        self.vars['add_offset'] = tk.IntVar(value=1)
        tk.ttk.Checkbutton(
            self.frames['exp_opts'],
            text="Offset",
            variable=self.vars['add_offset'],
            command=self.update_fit_parameters).grid(
                row=2, column=0, sticky='w')

        self.vars['include_irf'] = tk.IntVar(value=0)
        self.irf_opt_check = tk.ttk.Checkbutton(
            self.frames['exp_opts'],
            text="IRF (fs):",
            variable=self.vars['include_irf'],
            command=self.irf_opt_check_callback)
        self.vars['irf_factor'] = 1e-3
        self.irf_opt_check.grid(row=2, column=1)
        self.vars['irf_val'] = tk.DoubleVar(value=0.1)
        self.irf_val_entry = tk.Entry(
            self.frames['exp_opts'],
            textvariable=self.vars['irf_val'],
            width=7)
        self.irf_val_entry.grid(row=2, column=2)
        self.vars['fix_irf'] = tk.IntVar(value=1)
        self.fix_irf_check = tk.ttk.Checkbutton(
            self.frames['exp_opts'],
            text="Fix",
            variable=self.vars['fix_irf'])
        self.fix_irf_check.grid(row=2, column=3)

        tk.Label(self.frames['exp_opts'],
                 text='Correlation:').grid(row=3, column=0, sticky=tk.W)
        self.vars['para_correl'] = tk.StringVar(value="None")
        para_correl_select = tk.ttk.OptionMenu(
            self.frames['exp_opts'], self.vars['para_correl'],
            self.vars['para_correl'].get(), "None", "Kinetic Model",
            command=self.para_correl_callback)
        para_correl_select.config(width=20)
        para_correl_select.grid(row=3, column=1, columnspan=2, sticky=tk.W)

        self.model_disp = tk.ttk.Label(self.frames['exp_opts'], text="")
        self.model_disp.grid(row=4, column=0, columnspan=3, sticky='e')

        # oscillations

        self.vars['add_modulation'] = tk.IntVar(value=0)
        self.checks['cosine'] = tk.ttk.Checkbutton(
            self.frames['modul_opts'],
            text="Cosine Modulation",
            variable=self.vars['add_modulation'],
            command=self.modul_opt_callback)
        self.checks['cosine'].grid(row=0, column=0, sticky=tk.W)
        self.labels['num_cos'] = tk.Label(
            self.frames['modul_opts'], text="Frequency\nComponents:")
        self.labels['num_cos'].grid(row=0, column=1, sticky=tk.W)
        self.vars['num_modul_comp'] = tk.IntVar(value=1)
        self.optmenus['num_cos'] = tk.ttk.OptionMenu(
            self.frames['modul_opts'], self.vars['num_modul_comp'], 1, 1, 2, 3)
        self.optmenus['num_cos'].grid(row=0, column=2)

        self.vars['modul_fixed_para'] = tk.IntVar(value=0)
        self.checks['fix_cos_freq'] = tk.ttk.Checkbutton(
            self.frames['modul_opts'],
            text="Fixed frequencies",
            variable=self.vars['modul_fixed_para'])
        self.checks['fix_cos_freq'].grid(row=1, column=0)
        self.labels['num_damp_comp'] = tk.Label(
            self.frames['modul_opts'], text="Damping\nComponents:")
        self.labels['num_damp_comp'].grid(row=1, column=1)
        self.vars['num_damp_comp'] = tk.IntVar(value=0)
        self.optmenus['num_damp_comp'] = tk.ttk.OptionMenu(
            self.frames['modul_opts'], self.vars['num_damp_comp'], 0, 0, 1, 2,
            command=self.num_damp_comp_callback)
        self.optmenus['num_damp_comp'].grid(row=1, column=2)

        for widget in (self.optmenus['num_cos'],
                       self.optmenus['num_damp_comp'],
                       self.checks['fix_cos_freq'],
                       self.labels['num_damp_comp'],
                       self.labels['num_cos']):
            widget.config(state='disabled')

        self.frames['exp_opts'].grid(row=0, column=0, padx=1, pady=(1, 0),
                                     sticky='we')
        self.frames['modul_opts'].grid(row=3, column=0, padx=1, pady=1,
                                       sticky='we')

    def selection_callback(self, fitpage, trace_obj, *args):
        fitpage.fit_init = trace_obj.init_exp_fit
        try:
            fitpage.plot_resid_fft_check.config(state='normal')
        except Exception:
            pass
        for key, var in self.vars.items():
            fitpage.vars[key] = var

    def set_figure_labels(self, figure, update_canvas=False, **kwargs):
        figure.set_axes_label(x=self.xunit.join(['time delay (', ')']),
                              y=self.ylabel, update_canvas=update_canvas)

    def get_init_table_kwargs(self, xunit=None, **kwargs):
        if not xunit:
            xunit = self.xunit
        headers = {"curve": "Curve",
                   "comp": "Component",
                   "x": xunit.join(["Lifetime (", ")"]),
                   "y": "Amplitude"}
        if self.vars['add_modulation'].get():
            headers['x2'] = "Phase"
        return {"headers": headers, "exclude_para": ['x0_1', 'sigma_1']}

    def get_specific_inits(self):
        specific_inits = {}
        if self.vars['include_irf'].get():
            specific_inits['sigma_1'] = self.vars['irf_val'].get() * \
                self.vars['irf_factor']
        specific_inits['x0_1'] = self.vars['time_zero'].get()
        return specific_inits

    def set_result_table(self, table, xunit=None, **kwargs):
        if not xunit:
            xunit = self.xunit
        if self.vars['add_modulation'].get():
            table.column_dict["#1"]["heading"]["text"] = (
                "Lifetime (" + xunit + ") / Osc. Freq. (1/" + xunit + ")")
            table.column_dict["#2"]["heading"]["text"] = "Amplitude"
            table.column_dict["#2"]["column"]["stretch"] = False
            table.column_dict['#3'] = {"heading":
                                       {"text": "Phase", "anchor": "w"},
                                       "column":
                                       {"width": 75, "stretch": True},
                                       "axis": "x2"}
        else:
            table.column_dict["#1"]["heading"]["text"] = (
                "Lifetime (" + xunit + ")")
            table.column_dict["#2"]["heading"]["text"] = "Amplitude"
            table.column_dict["#2"]["column"]["stretch"] = True
            if "#3" in table.column_dict.keys():
                del table.column_dict["#3"]
        return True

    def get_fit_kwargs(self, func_str=None):
        if func_str is None:
            func_str = self.func_str
        else:
            self.func_str = func_str
        kwargs = {'num_exp': self.num_exp,
                  'offset': bool(self.vars['add_offset'].get()),
                  'irf': (
                          self.vars['irf_val'].get()*self.vars['irf_factor']
                          if self.vars['include_irf'].get() else None),
                  'irf_fixed': bool(self.vars['fix_irf'].get()),
                  't0': self.vars['time_zero'].get(),
                  'cos_modul': (self.vars['num_modul_comp'].get()
                                if self.vars['add_modulation'].get()
                                else None),
                  'fix_modul_paras': self.vars['modul_fixed_para'].get(),
                  'damp_comp': self.vars['num_damp_comp'].get(),
                  'fix_t0': bool(self.vars['time_zero_fixed'].get())}
        if self.vars['para_correl'].get().lower() == "none":
            kwargs['func_type'] = 'exp'
        elif func_str:
            kwargs['func_type'] = 'kineticexp' + func_str
        else:
            kwargs['func_type'] = self.vars['para_correl'].get()[-1].join(
                ['kineticexp', 'select', ''])

        return kwargs

    def update_fit_parameters(self, *args, update_table=True, func_str=None):
        if self.vars['num_exp_total'].get() < self.vars['num_damp_comp'].get():
            self.vars['num_damp_comp'].set(self.vars['num_exp_total'].get())
        self.num_exp = self.vars['num_exp_total'].get(
        ) - self.vars['num_damp_comp'].get()
        fit_obj_kwargs = self.get_fit_kwargs(func_str=func_str)
        self.controller.update_fit_parameters(update_table=True,
                                              fit_obj_kwargs=fit_obj_kwargs)

    # widget callbacks
    def irf_opt_check_callback(self, *args):
        if (self.vars['include_irf'].get()
                or self.vars['para_correl'].get().lower() != "none"):
            self.vars['time_zero_fixed'].set(0)
            self.checks['time_zero_fixed'].config(state='normal')
        else:
            self.vars['time_zero_fixed'].set(1)
            self.checks['time_zero_fixed'].config(state='disabled')

    def auto_axis_limits(self, xmin, *args, **kwargs):
        self.vars['time_zero'].set(0.0 if xmin < 0 else xmin)

    def para_correl_callback(self, *args):
        func_str = None
        self.func_str = None
        if not self.vars['para_correl'].get() == "None":
            win = KineticModelOptionsWindow(
                self, self.controller, input_vars_dict=self.kin_model_input)
            self.wait_window(win)
            if win.output is None:
                self.vars['para_correl'].set("None")
            else:
                func_str = win.output[1]
                self.num_exp = int(func_str[0]) + \
                    self.vars['num_damp_comp'].get()
                self.vars['num_exp_total'].set(self.num_exp)
                self.num_exp_select.config(state='disabled')
                self.checks['cosine'].config(state='disabled')
                self.model_disp.config(text=win.output[0])
                self.vars['add_modulation'].set(0)
                self.modul_opt_callback(update=False)
                self.kin_model_input = win.output[2]
        if self.vars['para_correl'].get() == "None":
            self.num_exp_select.config(state='normal')
            self.checks['cosine'].config(state='normal')
            self.model_disp.config(text="")
        self.irf_opt_check_callback()
        self.update_fit_parameters(func_str=func_str)

    def modul_opt_callback(self, *args, update=True):
        if not self.vars['add_modulation'].get():
            self.vars['num_exp_total'].set(
                self.num_exp if self.num_exp > 0 else 1)
            self.vars['num_damp_comp'].set(0)
        else:
            self.vars['num_exp_total'].set(self.num_exp)
        for widget in (
                self.optmenus['num_cos'], self.optmenus['num_damp_comp'],
                self.checks['fix_cos_freq'], self.labels['num_damp_comp'],
                self.labels['num_cos']):
            widget.config(
                state=('normal' if self.vars['add_modulation'].get()
                       else 'disabled'))
        if update:
            self.update_fit_parameters()

    def num_damp_comp_callback(self, *args):
        if self.vars['num_exp_total'].get() < self.vars['num_damp_comp'].get():
            self.vars['num_damp_comp'].set(self.vars['num_exp_total'].get())
        if self.vars['add_modulation'].get():
            self.vars['num_exp_total'].set(
                self.vars['num_damp_comp'].get() + self.num_exp)
            if self.vars['num_exp_total'].get() >= 5:
                self.vars['num_exp_total'].set(4)
            elif self.vars['num_exp_total'].get() == 0:
                self.vars['num_exp_total'].set(1)
        self.update_fit_parameters()


# %%
class FitTracePage(tk.Frame):
    def __init__(self, parent, controller=None,
                 main_page=None,
                 xlabel="time delay (ps)", figure_settings=None,
                 ylabel="$\Delta$ Abs. (mOD)", mode='kinetic',
                 trace_obj=None, **kwargs):
        tk.Frame.__init__(self, parent)
        try:
            xunit = re.findall(r"(?<=\().+(?=\))", xlabel)[0]
        except IndexError:
            tk.messagebox.showerror(message="Unable to read x unit. "
                                    + "Using default.")
            xunit = "ps"
            xlabel = "time delay (ps)"
        if figure_settings is None:
            self.figure_settings = {}
        else:
            self.figure_settings = figure_settings
        if main_page is None:
            self._main_page = BlankObject()
        else:
            self._main_page = main_page
        if type(trace_obj) is TATrace:
            self.trace_coll = TATraceCollection()
            self.trace_coll.add_trace(trace_obj, "")
        elif type(trace_obj) is TATraceCollection:
            self.trace_coll = trace_obj
        else:
            self.trace_coll = TATraceCollection(init_trace=True)
        self.fit_obj = None
        self.fit_done = False
        self.bounds = {}
        self.prev_inits = {}
        self.ind = [0]
        self.baseline_subtract_dict = {}
        self.controller = controller
        self.frames = {}
        self.checks = {}
        self.optmenus = {}
        self.labels = {}
        self.vars = {}

        for i in range(6):
            self.columnconfigure(i, weight=1)
            self.rowconfigure(i, weight=1)

        # frames
        self.frames['fit_opts'] = GroupBox(self, dim=(1, 6), border=True,
                                           text="Fit Options")
        self.frames['time_win'] = CustomFrame(
            self.frames['fit_opts'], dim=(3, 2))
        self.frames['kin_fit_opts'] = KineticFitOptions(
            self.frames['fit_opts'], self, xunit=xunit)
        self.frames['spec_fit_opts'] = SpectralFitOptions(
            self.frames['fit_opts'], self, xunit=xunit)
        self.frames['algo_options'] = CustomFrame(
            self.frames['fit_opts'], dim=(2, 7))
        self.frames['fit_opt_buttons'] = CustomFrame(
            self.frames['fit_opts'], dim=(2, 1))
        self.frames['numfeval'] = CustomFrame(
            self.frames['algo_options'], dim=(3, 2))
        self.frames['guesses'] = GroupBox(self, dim=(2, 4), border=True,
                                          text="Initial Guesses")

        self.fit_figure = TkMplFigure(
            self, callbacks={'button_press_event':
                             self.fit_fig_callback},
            plot_function=self.trace_coll.plot,
            xlabels=xlabel,
            ylabels=ylabel,
            plot_type='linefit',
            **self.figure_settings)

        self.fit_figure.grid(row=0, column=0, columnspan=3, padx=5,
                             sticky='wnse')

        self.frames['trace_select'] = TraceCollectionSelection(
            self, self.trace_coll,
            update_func=self.trace_sel_update,
            header='Select traces to fit',
            fig_obj=self.fit_figure)

        # Fit options
        self.frames['fit_opts'].grid(row=0, column=3, columnspan=1,
                                     sticky='nswe', padx=(20, 0))
        tk.Label(self.frames['fit_opts'], text="Fit options").grid(
            row=0, column=0, sticky='nw')

        self.frames['time_win'].grid(row=1, column=0, sticky='nswe')

        self.fit_range_label = tk.Label(
            self.frames['time_win'], text="Fit range (ps):")
        self.fit_range_label.grid(row=0, column=0, sticky=tk.W)
        self.xlower = tk.DoubleVar(value=0.0)
        self.xlower_entry = tk.Entry(
            self.frames['time_win'], textvariable=self.xlower, width=8)
        self.xlower_entry.grid(row=0, column=1, sticky=tk.W)
        self.xlower_entry.bind('<Return>', self.set_axlim)

        self.xupper = tk.DoubleVar(value=15)
        self.xupper_entry = tk.Entry(
            self.frames['time_win'], textvariable=self.xupper, width=8)
        self.xupper_entry.grid(row=0, column=2, pady=2, sticky=tk.W)
        self.xupper_entry.bind('<Return>', self.set_axlim)

        # Fit function specific frames:
        self.frames['spec_fit_opts'].grid(
            row=2, rowspan=2, column=0, sticky='wnse')
        self.frames['kin_fit_opts'].grid(
            row=2, rowspan=2, column=0, sticky='wnse')

        # algorithm options, running and saving
        self.frames['algo_options'].grid(row=4, column=0, sticky='nswe')

        tk.Label(self.frames['algo_options'],
                 text="Parameter bounds:").grid(row=0, column=0, sticky=tk.W)
        self.bounds_mode = tk.StringVar(value="None")
        bounds_mode_select = tk.ttk.OptionMenu(self.frames['algo_options'],
                                               self.bounds_mode,
                                               self.bounds_mode.get(),
                                               "None", "Enter")
        bounds_mode_select.config(width=7)
        bounds_mode_select.grid(row=0, column=1, sticky=tk.E, padx=8)

        tk.ttk.Label(self.frames['algo_options'],
                     text="Solver:").grid(row=1, column=0, sticky='w')
        self.vars['algo'] = tk.StringVar(value='leastsq')
        self.optmenus['algo'] = tk.ttk.OptionMenu(
            self.frames['algo_options'], self.vars['algo'], 'leastsq',
            'leastsq', 'nelder', 'ampgo', 'SLSQP')
        self.optmenus['algo'].grid(row=1, column=1, sticky='e')

        tk.Label(self.frames['algo_options'],
                 text='Max. function evaluations:').grid(
                     row=2, column=0, sticky=tk.W)
        tk.Label(self.frames['algo_options'],
                 text='Function tolerance:').grid(row=3, column=0, sticky='w')
        self.max_feval_prefac = tk.IntVar(value=1)
        self.max_feval_exp = tk.IntVar(value=5)
        self.ftol_prefac = tk.IntVar(value=1)
        self.ftol_exp = tk.IntVar(value=-1)
        tk.Entry(self.frames['numfeval'],
                 textvariable=self.max_feval_prefac,
                 width=2).grid(row=0, column=0, sticky=tk.E)
        tk.Label(self.frames['numfeval'], text="E").grid(row=0, column=1,
                                                         sticky='we', padx=5)
        tk.Entry(self.frames['numfeval'], textvariable=self.max_feval_exp,
                 width=2).grid(row=0, column=2, sticky=tk.W)
        tk.Entry(self.frames['numfeval'],
                 textvariable=self.ftol_prefac,
                 width=2).grid(row=1, column=0, sticky=tk.E)
        tk.Label(self.frames['numfeval'], text="E").grid(
            row=1, column=1, sticky='we', padx=5)
        tk.Entry(self.frames['numfeval'],
                 textvariable=self.ftol_exp,
                 width=2).grid(row=1, column=2, sticky=tk.W)

        self.frames['numfeval'].grid(row=2, column=1, sticky=tk.E, padx=10,
                                     rowspan=2)

        self.reject_outliers = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.frames['algo_options'],
            variable=self.reject_outliers,
            text="Outlier rejection").grid(
                row=4, column=0, sticky='w')
        tk.ttk.Label(self.frames['algo_options'],
                     text="Threshold (rel.)").grid(row=4, column=1, sticky='w')
        self.outlier_reject_thresh = tk.DoubleVar(value=4)
        tk.ttk.Entry(
            self.frames['algo_options'],
            textvariable=self.outlier_reject_thresh,
            width=5).grid(row=4, column=2)

        self.disp_error = tk.IntVar(value=1)
        tk.ttk.Checkbutton(
            self.frames['algo_options'],
            variable=self.disp_error,
            text='Display Errors',
            command=self.disp_results).grid(
                row=5, column=0, sticky=tk.W)
        self.enable_resid_plot = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.frames['algo_options'],
            variable=self.enable_resid_plot,
            text='Show residual',
            command=self.open_resid_figure).grid(
                row=5, column=1, columnspan=2, sticky=tk.W)
        self.plot_fit_error = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.frames['algo_options'],
            variable=self.plot_fit_error,
            text='Plot Error',
            command=self.update_plot).grid(row=6, column=0, sticky=tk.W)

        self.plot_resid_fft = tk.IntVar(value=0)
        self.plot_resid_fft_check = tk.ttk.Checkbutton(
            self.frames['algo_options'],
            variable=self.plot_resid_fft,
            text='Show Resid. FFT',
            command=self.open_resid_fft_fig)
        self.plot_resid_fft_check.grid(row=6, column=1, sticky='w')

        self.plot_components = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.frames['algo_options'],
            variable=self.plot_components,
            text='Plot Components').grid(row=7, column=0, sticky='w')
        self.plot_components.trace('w', self.plot_comps_check_callback)

        self.subtract_baseline = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.frames['algo_options'],
            variable=self.subtract_baseline,
            text='Subtract Offset',
            command=self.subtract_baseline_callback).grid(
                row=7, column=1, sticky='w')

        self.start_button = tk.ttk.Button(
            self.frames['fit_opt_buttons'], text="Fit",
            command=self.fit_traces)
        self.start_button.grid(
            row=0, column=0, pady=5, padx=5)

        self.cancel_button = ttk.Button(self.frames['fit_opt_buttons'],
                                        text="Cancel",
                                        command=self.cancel,
                                        state='disabled')
        self.cancel_button.grid(row=0, column=1, padx=5, pady=5)

        self.save_button = ttk.Button(self.frames['fit_opt_buttons'],
                                      text="Save results",
                                      command=self.save_fit)
        self.save_button.grid(row=0, column=2, pady=5, padx=5)

        self.frames['fit_opt_buttons'].grid(row=6, column=0, sticky='wnse')

        self.set_mode(mode=mode)

        # result table
        tk.Label(self, text="Fit Results").grid(
            row=2, column=0, sticky=tk.W, pady=(20, 0))

        self.result_frame = ScrollFrame(self, widget=FitResultsDisplayLmfit,
                                        scroll_dir='x')
        self.result_table = self.result_frame.widget

        self.result_table.name_dict['x0_1'] = 'Time Zero'
        self.result_table.name_dict['sigma_1'] = 'Sigma IRF'
        self.result_frame.grid(row=3, column=0, columnspan=3, sticky='wens')
        self.result_frame.update_window()

        # initials frame
        self.frames['guesses'].grid(row=1, column=3, columnspan=2, rowspan=4,
                                    sticky='wne', padx=(20, 5), pady=5)
        tk.Label(self.frames['guesses'], text='Guess mode:').grid(
            row=0, column=0, sticky=tk.E)
        self.guess_mode = tk.StringVar(value='Auto')
        tk.ttk.OptionMenu(
            self.frames['guesses'], self.guess_mode, self.guess_mode.get(),
            'Auto', 'Enter').grid(row=0, column=1, sticky=tk.W)
        self.guess_mode.trace('w', self.guess_mode_callback)
        self.results_as_init = tk.IntVar(value=0)
        tk.ttk.Checkbutton(
            self.frames['guesses'],
            text="Set entries to previous result",
            variable=self.results_as_init,
            command=self.results_as_init_callback).grid(
                row=1, column=0, columnspan=2)
        # initial table
        curve_labels, curve_label_dict, guess_in = get_trace_fit_guess_input(
            self.trace_coll)
        self.init_entry = FitParaEntryLmfit(self.frames['guesses'],
                                            curve_labels,
                                            self.fit_obj,
                                            scrollable='yx',
                                            exclude_para=['x0_1', 'sigma_1'])

        self.init_entry.grid(row=2, column=0, sticky='nsew', columnspan=2,
                             padx=5, pady=5)

        self.guess_mode_callback()
        self.success = {}

        # trace selection
        self.frames['trace_select'].grid(row=0, column=4, sticky='wne', padx=5)
        try:
            self.frames['trace_select'].update_box()
            self.auto_axis_limits()
        except Exception:
            pass

    def write_app_settings(self):
        for key, val in self.vars.items():
            try:
                self.controller.app_settings[FitTracePage][key] = val.get()
            except Exception as e:
                print(e)

    def load_app_settings(self):
        for key in self.vars.keys():
            try:
                self.vars[key].set(
                    self.controller.app_settings[FitTracePage][key])
            except Exception as e:
                print(e)

    def set_mode(self, mode='kinetic'):
        if re.search('kinetic|exp', mode, re.I):
            self.func_opts = self.frames['kin_fit_opts']
        elif re.search('spectral|line.*shape', mode, re.I):
            self.func_opts = self.frames['spec_fit_opts']
        else:
            return
        self.func_opts.selection_callback(self, self.trace_coll)
        try:
            self.func_opts.update_fit_parameters()
        except Exception:
            pass
        self.func_opts.tkraise()
        self.mode = mode

    # UI Callbacks
    def plot_comps_check_callback(self, *args):
        self.update_plot()

    def subtract_baseline_callback(self, *args):
        if self.subtract_baseline.get():
            for member in self.trace_coll.active_members:
                if member not in self.baseline_subtract_dict.keys():
                    self.baseline_subtract_dict[member] = {}
                for key in self.trace_coll.traces[member].active_traces:
                    trace = self.trace_coll.traces[member].tr[key]
                    try:
                        offset = trace['fit_paras']['const_amp_1'].value
                    except Exception:
                        continue
                    if key in self.baseline_subtract_dict[member].keys():
                        offset -= self.baseline_subtract_dict[member][key]
                    trace['y'] = trace['y'] - offset
                    trace['fit'] = trace['fit'] - offset
                    self.baseline_subtract_dict[member][key] = offset
        else:
            for member in self.trace_coll.traces.keys():
                if member not in self.baseline_subtract_dict.keys():
                    self.baseline_subtract_dict[member] = {}
                for key in self.trace_coll.traces[member].active_traces:
                    if key in self.baseline_subtract_dict[member].keys():
                        trace = self.trace_coll.traces[member].tr[key]
                        offset = self.baseline_subtract_dict[member][key]
                        trace['y'] = trace['y'] + offset
                        trace['fit'] = trace['fit'] + offset
                        self.baseline_subtract_dict[member][key] = 0
        self.update_plot()

    def trace_sel_update(self, update_plot=True):
        if update_plot:
            self.update_plot(
                color_order={'ind': self.frames['trace_select'].ind})
        self.update_init_table()
        if self.fit_done:
            self.disp_results()

    def auto_axis_limits(self, ylim=None, **kwargs):
        try:
            xmin = np.min([np.min(self.trace_coll.traces[member].xdata)
                           for member in self.trace_coll.active_members])
        except Exception:
            self.set_axlim(ylim=ylim, **kwargs)
            return
        self.xlower.set(np.round(xmin, 2))
        self.xupper.set(np.max([np.max(self.trace_coll.traces[member].xdata)
                        for member in self.trace_coll.active_members]))
        try:
            self.func_opts.auto_axis_limits(xmin)
        except Exception as e:
            print(e)
        self.set_axlim(ylim=ylim, **kwargs)

    def guess_mode_callback(self, *args):
        if self.guess_mode.get().lower() == "auto":
            state = 'disabled'
        else:
            state = 'normal'
        for val in self.init_entry.entries.values():
            for entry in val.values():
                entry.config(state=state)

    def results_as_init_callback(self, *args):
        if self.guess_mode.get().lower() == 'auto':
            self.results_as_init.set(0)

    def set_axlim(self, *args, ylim=None, xlim=None, **kwargs):
        if xlim is not None:
            self.xlower.set(xlim[0])
            self.xupper.set(xlim[1])
        r = self.xupper.get() - self.xlower.get()
        self.fit_figure.set_axes_lim(x=[self.xlower.get() - r * 0.05,
                                        self.xupper.get() + r * 0.05],
                                     y=ylim)

    def fit_fig_callback(self, event):
        try:
            if event.dblclick:
                open_topfigure(self, plot_func=self.trace_coll.plot,
                               plot_type='linefit'
                               if self.fit_done else 'line',
                               editable=True, fig_obj=self.fit_figure,
                               controller=self.controller,
                               **self.figure_settings)
            elif event.button == 3:
                FigureOptionWindow(self, self.fit_figure,
                                   controller=self.controller,
                                   **self.figure_settings)
        except Exception:
            return

    def update_fit_parameters(self, *args, update_table=True,
                              fit_obj_kwargs=None):
        if fit_obj_kwargs is None:
            fit_obj_kwargs = self.func_opts.get_fit_kwargs()
        self.fit_obj = self.fit_init(
            fit_obj=self.fit_obj,
            method=self.vars['algo'].get(),
            num_function_eval=int(
                self.max_feval_prefac.get()*10**self.max_feval_exp.get()),
            fun_tol=self.ftol_prefac.get()*10**self.ftol_exp.get(),
            **fit_obj_kwargs)
        if update_table:
            self.update_init_table()

    # guesses and bounds inputs
    def update_init_table(self, *args, show=True):
        curve_labels, curve_label_dict, inits = get_trace_fit_guess_input(
            self.trace_coll, guesses_input=self.prev_inits)
        table_kwargs = self.func_opts.get_init_table_kwargs()

        self.init_entry.setup_table(curve_labels=curve_labels,
                                    fit_obj=self.fit_obj,
                                    **table_kwargs)
        self.write_inits_to_table(curve_labels=curve_labels,
                                  curve_label_dict=curve_label_dict,
                                  inits=inits, overwrite=True)
        if self.guess_mode.get().lower() == "auto":
            self.guess_mode_callback()

    def write_inits_to_table(self, inits=None, curve_labels=None,
                             curve_label_dict=None, overwrite=None):
        if inits is None:
            inits = self.prev_inits
        if not curve_labels or not curve_label_dict:
            curve_labels, curve_label_dict, inits = get_trace_fit_guess_input(
                self.trace_coll, guesses_input=inits)
        if overwrite is None:
            overwrite = (self.guess_mode.get().lower() != 'Enter'
                         or self.results_as_init.get())
        self.init_entry.set_values(inits, overwrite=overwrite)

    def get_inits(self):
        if self.guess_mode.get() == 'Enter':
            curve_labels, curve_label_dict, guess_in = (
                get_trace_fit_guess_input(self.trace_coll))
            inits = get_trace_fit_guess_output(
                self.init_entry.get_output(),
                curve_label_dict=curve_label_dict)
        else:
            inits = {}
            for key in self.trace_coll.active_members:
                inits[key] = "auto"
        specific_inits = self.func_opts.get_specific_inits()
        for key, val in specific_inits.items():
            self.fit_obj.params[key].value = val
        return inits

    def set_bounds(self):
        self.fit_obj.set_std_bounds()
        self.bounds = get_trace_fit_guesses(
            self, self.fit_obj,
            trace_obj=self.trace_coll,
            guesses_input=self.bounds,
            case="bounds",
            controller=self.controller,
            **self.func_opts.get_init_table_kwargs())
        return self.bounds

    # running fit
    def fit_traces(self):
        def start_thread(*args, **kwargs):
            self.task = ThreadedTask(self._fit_task, *args,
                                     after_finished_func=self._after_fit,
                                     interruptible=True, **kwargs)
            self.task.start()

        self._toggle_widget_state_during_fit(case='start')
        self.update_fit_parameters(update_table=False)
        self.fit_obj.set_attributes(
            reject_outliers=self.reject_outliers.get(),
            outlier_threshold=self.outlier_reject_thresh.get(),
            outliers=[[], []])
        self.fit_figure.set_plot_kwargs(
            plot_outliers=self.reject_outliers.get())
        # get guesses and bounds
        guesses = self.get_inits()
        if self.bounds_mode.get() == "Enter":
            try:
                bounds = self.set_bounds()
            except Exception:
                self._toggle_widget_state_during_fit(case='end')
                return
        else:
            bounds = {}
            for key in self.trace_coll.active_members:
                bounds[key] = None

        # get fit range (indices)
        self.range_ind = {}
        for key in self.trace_coll.active_members:
            trace = self.trace_coll.traces[key]
            try:
                lower = np.where(np.asarray(trace.xdata) >=
                                 self.xlower.get())[0][0]
            except Exception:
                lower = 0
            try:
                upper = np.where(np.asarray(trace.xdata) <=
                                 self.xupper.get())[0][-1]
            except Exception:
                upper = len(trace.xdata) - 1
            self.range_ind[key] = [lower, upper]
        # run fit thread
        start_thread(guesses=guesses, bounds=bounds)

    def _fit_task(self, guesses=None, bounds=None):
        fit_para = {}
        prev_inits = {}
        reports = {}
        success = {}
        for key in self.trace_coll.active_members:
            kwargs = {}
            if guesses is not None:
                kwargs['inits'] = guesses[key]
            if bounds is not None:
                kwargs['bounds'] = bounds[key]
            fit_para[key], prev_inits[key], reports[key], success[key] = (
                self.trace_coll.traces[key].run_fit(
                    fit_obj=self.fit_obj,
                    fit_range=self.range_ind[key],
                    **kwargs))
        return fit_para, prev_inits, reports, success

    def _after_fit(self):
        try:
            if self.task.output is not None:
                self.fit_para, self.prev_inits, self.fitreports, success = (
                    self.task.output)
                for key, val in success.items():
                    if key not in self.success.keys():
                        self.success[key] = {}
                    for k, v in val.items():
                        self.success[key][k] = v
                self.after(100, self._post_fit_opts)
        except Exception:
            raise
        finally:
            self._toggle_widget_state_during_fit(case='end')

    def _post_fit_opts(self):
        if self.guess_mode.get() == 'Auto':
            self.write_inits_to_table(inits=self.prev_inits)
        elif self.results_as_init.get():
            self.write_inits_to_table(inits=self.fit_para)
        self.fit_done = True
        self.update_plot()
        self.open_resid_figure()
        self.open_resid_fft_fig()
        self.show_fit_report()
        try:
            self.disp_results()
        except Exception:
            pass

    def _toggle_widget_state_during_fit(self, case='start'):
        if case == 'start':
            state_group_1 = 'disabled'
            state_group_2 = 'normal'
        elif case == 'end':
            state_group_1 = 'normal'
            state_group_2 = 'disabled'
        else:
            return
        for widget in (self.start_button, self.save_button):
            widget.config(state=state_group_1)
        self.cancel_button.config(state=state_group_2)

    def show_fit_report(self):
        report_dict = {}
        for member, rep in self.fitreports.items():
            for key, val in rep.items():
                report_dict[": ".join([str(member), key])
                            ] = lmfitreport_to_dict(val)
        self.fit_report = MultiDisplayWindow(self,
                                             controller=self.controller,
                                             title="Fit Reports",
                                             header="", input_dict=report_dict,
                                             close_button="OK",
                                             mode='expandable',
                                             orient='horizontal')

    def cancel(self):
        try:
            self.task.raise_exception()
        except Exception as e:
            tk.messagebox.showerror(message=e, parent=self)

    def save_fit(self):
        file = save_box(filetypes=[('text files', '.txt'),
                                   ('Matlab files', '.mat')],
                        fext='.mat', parent=self)
        if file is not None:
            self.trace_coll.xlabel = self.fit_figure.get_xlabel()
            errors = self.trace_coll.save_traces(file, trace_type=self.mode)
            if errors:
                message = "Error(s) occurred saving fit data.\n"
                no_fit = []
                unknown = []
                for key, val in errors.items():
                    if val == 'no_fit':
                        no_fit.append(key)
                    else:
                        unknown.append([key, val])
                if len(no_fit) > 0:
                    message += ("No fit data for one or more traces. " +
                                "Traces without fit have not been saved.\n")
                if len(unknown) > 0:
                    for err in unknown:
                        message += ": ".join([str(e) for e in err]) + '\n'
                tk.messagebox.showerror(message=message, parent=self)

    def open_resid_fft_fig(self, *args):
        if self.fit_done and self.plot_resid_fft.get():
            self.trace_coll.residual_fft()
            legends = self.trace_coll.get_legend()
            open_topfigure(self, plot_func=self.trace_coll.plot_resid_fft,
                           plot_type='line',
                           controller=self.controller,
                           editable=True, legends=[legends],
                           xlabels="wavenumber (cm$^{-1}$)",
                           ylabels=self.fit_figure.get_ylabel(),
                           axes_titles='FFT of Residual',
                           **self.figure_settings)

    def open_resid_figure(self, *args):
        if self.fit_done and self.enable_resid_plot.get():
            legends = self.trace_coll.get_legend()
            open_topfigure(self, plot_func=lambda *args, **kwargs:
                           self.trace_coll.plot_residual(*args, **kwargs),
                           plot_type='line',
                           controller=self.controller,
                           editable=True, legends=[legends],
                           xlabels=self.fit_figure.get_xlabel(),
                           ylabels=self.fit_figure.get_ylabel(),
                           axes_titles=["Residual"],
                           xlimits=self.fit_figure.get_xlim(),
                           **self.figure_settings)

    def update_plot(self, *args, **kwargs):
        self.fit_figure.set_plot_kwargs(
            include_fit=self.fit_done,
            plot_fit_error=self.plot_fit_error.get(),
            include_fit_comps=self.plot_components.get())
        self.fit_figure.plot(**kwargs)

    def disp_results(self, *args):
        set_columns = self.func_opts.set_result_table(self.result_table)
        results = {}
        for member in self.trace_coll.active_members:
            if len(self.trace_coll.active_members) < 2:
                master_lbl = ""
            else:
                master_lbl = member + ": "
            for key in self.trace_coll.traces[member].active_traces:
                trace = self.trace_coll.traces[member].tr[key]
                lbl = master_lbl + key
                try:
                    results[lbl] = {
                        'success': self.success[member][key],
                        'params': trace['fit_paras'],
                        'stderrors': trace['fit_para_errors'],
                        'fit_subfunctions': trace['fit_subfunctions']}
                except Exception:
                    pass
        self.result_table.display_error = self.disp_error.get()
        columns = None if set_columns else False
        self.result_table.show_results(results, columns=columns)
        self.result_frame.update_window()

    def update_results_disp_header(self, col_dct=None):
        if col_dct is None:
            col_dct = self.result_table.column_dict
        curr_wd = 0
        for key, col in col_dct.items():
            curr_wd += col["column"]["width"]
        new_wd = self.fit_figure.canvas.get_width_height()[1]
        for key, col in col_dct.items():
            col["column"]["width"] = int(
                col["column"]["width"] / curr_wd * new_wd)
        self.result_table.setup_columns(col_dct=col_dct)

    # page Navigation functions, only work in conjunction with TA app
    def _leave_page(self):
        mp = self._main_page
        try:
            if mp.trace_opts.trace_mode in ('Spectral', 'Right SV'):
                mp.trace_opts.current_traces.set_x_mode(
                    mp.data_obj.get_x_mode())
        except Exception:
            return False
        else:
            if self.fit_done:
                mp.trace_figure.set_plot_kwargs(
                    include_fit=True)
            mp.trace_figure.plot()
            return True

    def _enter_page(self):
        if self._update_content(reset=False):
            self.frames['trace_select'].update_box()
            self.start_button.config(state='normal')
            self.cancel_button.config(state='disabled')
            self.update_results_disp_header()
            return True
        else:
            tk.messagebox.showerror("Error", "No traces to fit.", parent=self)
            return False

    def _update_content(self, reset=True):
        mp = self._main_page
        try:
            traces = mp.trace_opts.current_traces
        except Exception:
            return False
        else:
            if len(traces.active_traces) == 0:
                return False
            self.fit_done = reset
            self.trace_coll.remove_all_traces()
            self.trace_coll.add_trace(traces, "")
            if mp.trace_opts.trace_mode in ('Spectral', 'Right SV'):
                mode = 'spectral'
                traces.set_x_mode(mp.data_obj.get_x_mode())
            else:
                mode = 'kinetic'
                traces._xmode = 'time'
            self.set_mode(mode)
            xlim = mp.trace_figure.get_xlim()
            if mode == 'kinetic':
                self.func_opts.vars['time_zero'].set(
                    self.xlower.get())
                if (mp.data_obj.time_unit
                        in mp.data_obj.time_unit_factors.keys()):
                    self.func_opts.irf_opt_check.config(
                        text=("IRF ("
                              + mp.data_obj.time_unit_factors[
                                  mp.data_obj.time_unit][1]
                              + ")"))
                    self.func_opts.vars['irf_factor'] = 1e-3
                    self.func_opts.vars['irf_val'].set(100)
                else:
                    self.func_opts.irf_opt_check.config(
                        text=mp.data_obj.time_unit.join(["IRF (", ")"]))
                    self.func_opts.vars['irf_factor'] = 1.0
                    self.func_opts.vars['irf_val'].set(0.1)
            else:
                self.func_opts.xaxis_select_callback()
                xlim = np.sort(traces.convert_spec_values(
                    xlim, x_in=mp.data_obj.get_x_mode(), x_out='wavenumber'))
            self.set_axlim(xlim=xlim, ylim=mp.trace_figure.get_ylim())
            return True


# %%
class KineticModelOptionsWindow(tk.Toplevel):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Toplevel.__init__(self, parent)
        self.fr = KineticModelOptions(self, *args, **kwargs)
        self.fr.grid(row=0, column=0, sticky='wnse')
        tk.ttk.Button(self, text='OK', command=self._ok).grid(
            row=1, column=0, columnspan=2, pady=5)
        self.output = None
        center_toplevel(self, controller)

    def _ok(self, *args):
        self.output = self.fr.get_output()
        self.destroy()


class KineticModelOptions(CustomFrame):
    def __init__(self, parent, fit_obj=None, max_comp=8,
                  model_classes=None, dim=None,
                  input_vars_dict=None, **frame_kwargs):
        if dim is None:
            dim = (2, 2)
        CustomFrame.__init__(self, parent, dim=dim, **frame_kwargs)
        self.fit_models = FitModels()
        self.fit_obj = fit_obj
        self.output = None
        if input_vars_dict is None:
            input_vars_dict = {}
        self._models_categorized = {'Sequential': {'Chain': [],
                                                   'Branched Chain': []},
                                    'Parallel': {" ": []}}
        for key, val in self.fit_models.model_dict.items():
            if val['category'].lower() == 'parallel':
                self._models_categorized['Parallel'][" "].append(key)
                continue
            for cat, subcat in self._models_categorized.items():
                if (val['category'].lower()
                        in [k.lower() for k in subcat.keys()]):
                    subcat_key = (val['category']
                                  if val['category'] in subcat.keys()
                                  else val['category'].lower())
                    self._models_categorized[cat][subcat_key].append(key)
                    break

        if model_classes is None:
            model_classes = list(self._models_categorized.keys())
        # default or input values
        vars_dict = {'class': 'Sequential', 'type': 'Chain',
                     'model': 'ABC', 'inf': 1,
                     'comps': {}}
        for key, val in input_vars_dict.items():
            vars_dict[key] = val
        self.model_class = tk.StringVar(value=vars_dict['class'])
        self.model_type = tk.StringVar(value=vars_dict['type'])
        self.model = tk.StringVar(value=vars_dict['model'])
        self.inf_comp = tk.IntVar(value=vars_dict['inf'])
        # setup widgets
        self.model_select_frame = GroupBox(self, dim=(2, 4),
                                           text="Select Model")
        tk.ttk.OptionMenu(self.model_select_frame,
                          self.model_class,
                          model_classes[0],
                          *model_classes).grid(row=0, column=1, sticky='w')

        self.modeltype_select = tk.ttk.OptionMenu(
            self.model_select_frame, self.model_type, self.model_type.get(),
            *self._models_categorized[self.model_class.get()].keys())

        self.model_select = tk.ttk.OptionMenu(
            self.model_select_frame, self.model, self.model.get(),
            *self._models_categorized[
                self.model_class.get()][
                    self.model_type.get()])

        tk.ttk.Checkbutton(
            self.model_select_frame,
            variable=self.inf_comp,
            text="Infinite Component",
            command=self.inf_comp_callback).grid(
                row=3, column=0, columnspan=2, sticky='w')
        tk.ttk.Label(self.model_select_frame, text="Class:").grid(
            row=0, column=0, sticky='w')
        tk.ttk.Label(self.model_select_frame, text="Type:").grid(
            row=1, column=0, sticky='w')
        self.modeltype_select.grid(row=1, column=1, sticky='w')
        tk.ttk.Label(self.model_select_frame, text="Model:").grid(
            row=2, column=0, sticky='w')
        self.model_select.grid(row=2, column=1, sticky='w')

        self.model_select_frame.grid(row=0, column=0, sticky='wnse',
                                     padx=5, pady=5)

        self.model_display_frame = CustomFrame(self, dim=(1, 1), height=100)
        self.model_display = tk.ttk.Label(self.model_display_frame)
        self.model_display.grid(sticky='wnse')

        self.model_display_frame.grid(row=1, column=0, padx=5, pady=5)

        self.comp_select_frame = GroupBox(self, text="Select Comp.")
        self.components = {}
        self.comp_checks = {}
        self.comp_list = tk.Frame(self.comp_select_frame)
        self.comp_list.grid(sticky='wnse')
        self.update_components()
        for key, val in vars_dict['comps'].items():
            self.components[key].set(val)

        self.comp_select_frame.grid(row=0, column=1, rowspan=2, sticky='wnse',
                                    padx=5, pady=5)
        self.model_class.trace('w', self.model_class_callback)
        self.model_type.trace('w', self.model_type_callback)
        self.model.trace('w', self.update_components)

    def callback(*args, **kwargs):
        # for overwriting
        return

    def inf_comp_callback(self, *args):
        self.callback()

    def get_output(self):
        num_comp = self.fit_models.model_dict[self.model.get()][
            'number_of_decays']
        vars_dict = {'class': self.model_class.get(),
                     'type': self.model_type.get(),
                     'model': self.model.get(),
                     'inf': self.inf_comp.get(),
                     'comps': {}}
        # read selected components
        if re.search("paral", self.model_class.get(), re.I):
            selected_comps = [(str(i + 1), i + 1) for i in range(num_comp)]
            for c in self.components.keys():
                vars_dict['comps'][c] = self.components[c].get()
        else:
            selected_comps = []
            for i, c in enumerate(self.components.keys()):
                vars_dict['comps'][c] = self.components[c].get()
                if vars_dict['comps'][c]:
                    selected_comps.append((c, i + 1))
        if len(selected_comps) == 0:
            self.output = None
            return self.output
        # write model name
        model_name = ' '.join([self.model_class.get(), self.model_type.get(),
                               self.model.get() + ':'])
        if len(selected_comps) == 1:
            model_name += ' Comp. ' + selected_comps[0][0]
        else:
            model_name += " Sum of "
            if len(list(self.components.keys())) == len(selected_comps) - 1:
                model_name += '-'.join([selected_comps[0]
                                        [0], selected_comps[-1][0]])
            else:
                model_name += '+'.join([c[0] for c in selected_comps])
        # write model string for fit object
        fit_object_str = self.fit_models.write_model_string(
            self.model.get(), num_comp=num_comp, inf_comp=self.inf_comp.get(),
            selected_comps=selected_comps)
        self.output = [model_name, fit_object_str, vars_dict]
        return self.output

    def model_class_callback(self, *args):
        self.modeltype_select['menu'].delete(0, 'end')
        for t in self._models_categorized[self.model_class.get()].keys():
            self.modeltype_select['menu'].add_command(
                label=t, command=lambda ty=t: self.model_type.set(ty))
        self.model_type.set(
            list(self._models_categorized[self.model_class.get()].keys())[0])

    def model_type_callback(self, *args):
        self.model_select['menu'].delete(0, 'end')
        models = self._models_categorized[
            self.model_class.get()][self.model_type.get()]
        self.model.set(models[0])
        for model in models:
            self.model_select['menu'].add_command(
                label=model, command=lambda m=model: self.model.set(m))

    def update_components(self, *args):
        curr_keys = list(self.components.keys())
        for key in curr_keys:
            del self.components[key]
            self.comp_checks[key].grid_forget()
            del self.comp_checks[key]
        self.components = {}
        self.comp_checks = {}
        if re.search('para', self.model_class.get(), re.I):
            comps = ["All"]
        else:
            comps = [char for char in self.model.get()
                     if re.search('[A-Za-z]', char)]
            comps.sort()
        r = 0
        col = 0
        for c in comps:
            self.components[c] = tk.IntVar(value=1)
        for c in self.components.keys():
            self.comp_checks[c] = tk.ttk.Checkbutton(
                self.comp_list, variable=self.components[c],
                text=c,
                command=self.callback)
            self.comp_checks[c].grid(row=r, column=col, sticky='w')
            r += 1
            if r > 5:
                r = 0
                col += 1

        if re.search('para', self.model_class.get(), re.I):
            self.comp_checks["All"].config(state='disabled')
        self.update_model_disp()
        self.callback()

    def update_model_disp(self, *args):
        lbl = self.fit_models.model_dict[
            self.model.get()]['mechanism']
        lbl = ";\n".join([lb.strip() for lb in lbl.split(";")])
        self.model_display.config(text=lbl)


class FitResultsDisplay(tk.Toplevel):
    def __init__(self, parent, entry_dict, controller=None, headers=[],
                 title="Fit Results"):
        tk.Toplevel.__init__(self, parent)
        if controller is not None:
            move_toplevel_to_default(self, controller)
        self.title(title)
        tk.Label(self, text=title).grid(row=0, column=0, pady=5)
        self.frame = tk.Frame(self)

        for i, header in enumerate(headers):
            tk.Label(self.frame, text=header).grid(row=0, column=i,
                                                   sticky=tk.W, padx=5)

        self.place_labels_recursive(entry_dict, row=1, column=0)

        self.frame.grid(row=1, column=0)
        ttk.Button(self, text="OK", command=self.destroy).grid(
            row=2, column=0)
        if controller is not None:
            center_toplevel(self, controller)

    def place_labels_recursive(self, entry_dict, row=1, column=0):
        for key, val in entry_dict.items():
            tk.Label(self.frame, text=key).grid(row=row, column=column,
                                                sticky=tk.W, padx=5)
            try:
                val.keys()
            except Exception:
                for i, lb in enumerate(val):
                    tk.Label(self.frame, text=str(lb)).grid(
                        row=row, column=column + i + 1, sticky=tk.W, padx=5)
                row += 1
            else:
                row, column = self.place_labels_recursive(val, row, column + 1)
        return row, column - 1
