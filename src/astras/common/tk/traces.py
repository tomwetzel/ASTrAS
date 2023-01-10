# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 09:59:00 2022

@author: bittmans
"""
from .general import GroupBox, EntryDialog
from ..helpers import idle_function

from tkinter import ttk
import tkinter as tk


class TraceSelection(GroupBox):
    def __init__(self, parent, traces,
                 update_func=None, header="Select traces",
                 layout='vertical', fig_obj=None, delete_button=False,
                 none_option=False):
        GroupBox.__init__(self, parent,
                          dim=(1, 5) if layout == 'vertical' else (2, 4),
                          text=header)
        self.parent = parent
        self.traces = traces
        self.none_option = none_option
        self.buttons = {}
        self.update_kwargs = {}
        self.fig_obj = fig_obj
        if update_func is None:
            self.update_func = idle_function
        else:
            self.update_func = update_func

        rows = [1, 2, 3, 4, 5, 6] if layout == 'vertical' else [
            0, 1, 2, 3, 4, 5]
        column = 0 if layout == 'vertical' else 1
        span = 1 if layout == 'vertical' else 4
        sticky = 'n' if layout == 'vertical' else 'nw'

        select_frame = tk.Frame(self)
        self.select = tk.Listbox(select_frame, selectmode=tk.EXTENDED)
        self.select.grid(row=0, column=0, sticky='wn')

        self.select.bind(
            "<<ListboxSelect>>", self.select_click_callback)

        self.select_yscroll = tk.Scrollbar(select_frame)

        self.select.config(yscrollcommand=self.select_yscroll.set)
        self.select_yscroll.config(command=self.select.yview)
        self.select_yscroll.grid(row=0, column=1, sticky='nse')

        self.select_xscroll = tk.Scrollbar(select_frame, orient=tk.HORIZONTAL)

        self.select.config(xscrollcommand=self.select_xscroll.set)
        self.select_xscroll.config(command=self.select.xview)
        self.select_xscroll.grid(row=1, column=0, sticky='n')

        select_frame.grid(row=0, column=0, sticky='wne',
                          rowspan=span, padx=5, pady=5)

        self.buttons['move_up'] = ttk.Button(self, text='Move Up',
                                             command=lambda:
                                                 self.move_list_item('up'))
        self.buttons['move_up'].grid(row=rows[0],
                                     column=column, pady=1, padx=2,
                                     sticky=sticky)
        self.buttons['move_down'] = ttk.Button(
            self, text='Move Down',
            command=lambda:
                self.move_list_item('down'))
        self.buttons['move_down'].grid(row=rows[1],
                                       column=column, pady=1, padx=2,
                                       sticky=sticky)
        self.buttons['sort'] = ttk.Button(
            self, text="Sort", command=self.sort_traces)
        self.buttons['sort'].grid(
            row=rows[2], column=column, pady=1, padx=2,
            sticky=sticky)
        self.buttons['rename'] = ttk.Button(self, text='Rename',
                                            command=self.rename_trace)
        self.buttons['rename'].grid(row=rows[3],
                                    column=column, pady=1, padx=2,
                                    sticky=sticky)
        if delete_button:
            self.buttons['delete'] = ttk.Button(self, text='Delete',
                                                command=self.delete_trace)
            self.buttons['delete'].grid(row=rows[4],
                                        column=column, pady=1, padx=2,
                                        sticky=sticky)

        if self.fig_obj:
            default = ("Multi Axes" if self.fig_obj.get_num_subplots() > 1
                       else "Single Axes")
            self.plot_mode = tk.StringVar(value=default)
            self.plot_mode_selection = tk.ttk.OptionMenu(
                self, self.plot_mode, default, "Single Axes", "Multi Axes")
            self.plot_mode.trace('w', self.plot_mode_callback)
            self.ax_lim = {default: {'x': list(self.fig_obj.xlimits),
                                     'y': list(self.fig_obj.ylimits)}}
            self.current_ax_mode = default
            self.plot_mode_selection.grid(
                row=rows[5], column=column, pady=1, padx=2, sticky=sticky)

    def plot_mode_callback(self, *args):
        self.fig_obj.current_axes_no = 0
        if self.current_ax_mode != self.plot_mode.get():
            self.ax_lim[self.current_ax_mode] = {'x': self.fig_obj.xlimits,
                                                 'y': self.fig_obj.ylimits}
            self.current_ax_mode = self.plot_mode.get()
        keys = ['plot_type', 'plot_kwargs', 'grid_kwargs', 'legend_kwargs',
                'axes_grid_on', 'axes_frame_on', 'color_obj', 'xlimits',
                'ylimits', 'xlabels', 'ylabels', 'clabels', 'fontsize',
                'axes_titles', 'texts', 'text_kwargs', 'transpose',
                'invert_yaxis', 'fit_kwargs']
        kwargs = {}
        for key in keys:
            try:
                kwargs[key] = getattr(self.fig_obj, key)[0]
            except AttributeError:
                pass
            except Exception as e:
                print(e)
        legends = self.get_legend_entries()
        if self.plot_mode.get() == "Single Axes":
            self.fig_obj.set_num_subplots(1, legends=legends, **kwargs)
            self.update_func()
            self.fig_obj.plot_function = self.traces.plot
        else:
            self.fig_obj.set_num_subplots(
                self.get_number_of_active_traces(), legends=legends, **kwargs)
            self.update_func()
            self.fig_obj.plot_function = self.traces.plot_single
        for i in range(self.fig_obj.get_num_subplots()):
            try:
                self.fig_obj.xlimits[i] = self.ax_lim[self.plot_mode.get(
                )]['x'][i]
                self.fig_obj.ylimits[i] = self.ax_lim[self.plot_mode.get(
                )]['y'][i]
            except Exception:
                self.fig_obj.set_axes_lim(
                    i=i, x=False, y=False, update_canvas=False)
        self.fig_obj.plot_all()

    def get_legend_entries(self):
        if self.current_ax_mode == "Single Axes":
            return [[tr for tr in self.traces.active_traces]]
        else:
            return [[tr] for tr in self.traces.active_traces]

    def get_number_of_active_traces(self):
        return len(self.traces.active_traces)

    def select_click_callback(self, *args, **kwargs):
        items = self.select.curselection()
        if len(items) > 0:
            if 0 in items:
                self.select.selection_clear(1, tk.END)
                self.ind = [i for i, key in enumerate(
                    self.select_labels)]
            elif self.none_option and 1 in items:
                self.select.selection_clear(2, tk.END)
                self.ind = []
            else:
                self.ind = [i - (1 + int(self.none_option))
                            for i in self.select.curselection()]
            self.update_active_traces()
            if self.fig_obj:
                if self.plot_mode.get() == "Multi Axes":
                    self.plot_mode_callback()
                    return
                else:
                    self.fig_obj.set_legend(
                        entries=self.get_legend_entries()[0])
            self.update_func(update_plot=True, **self.update_kwargs)
            self.update_kwargs = {}

    def update_active_traces(self):
        self.traces.active_traces = [
            self.select_labels[i] for i in self.ind]

    def _update_box(self, items=None):
        if items is None:
            for lb in self.traces.tr.keys():
                self.select.insert(tk.END, lb)
                self.select_labels.append(lb)
        else:
            for item in items:
                if item in self.traces.tr.keys():
                    self.select.insert(tk.END, item)
                    self.select_labels.append(item)

    def update_box(self, items=None, select_item=0, generate_event=True,
                   **update_kwargs):
        self.select.delete(0, tk.END)
        self.select_labels = []
        self.select.insert(tk.END, "All")
        if self.none_option:
            self.select.insert(tk.END, "None")
        self._update_box(items=items)
        self.update_kwargs = update_kwargs
        if len(self.select_labels) == 0:
            select_item = 0
        self.select.select_set(select_item)
        if generate_event:
            self.select.event_generate("<<ListboxSelect>>")

    def delete_trace(self, *args, **kwargs):
        for i in self.ind:
            self.traces.delete_trace(self.select_labels[i])
        sel = min(self.ind)
        if sel < 1:
            sel = 1
        self.update_box(select_item=sel)

    def item_insert(self, *args, **kwargs):
        # only to be overwritten by child class
        return

    def re_order_traces(self):
        self.traces.re_order_traces(self.select_labels)

    def move_list_item(self, case='up'):
        index_offset = 1 + int(self.none_option)
        try:
            oldindex = self.select.curselection()[0] - index_offset
        except Exception:
            return
        else:
            if (oldindex == 0 + int(case != 'up')
                    * (len(self.select_labels) - 1)):
                return
            newindex = oldindex + 1 - 2*int(case == 'up')
            self.select_labels.insert(
                newindex, self.select_labels.pop(oldindex))
            self.item_insert(newindex, oldindex)
            self.update_box(items=self.select_labels,
                            select_item=newindex + index_offset,
                            generate_event=False)
            self.re_order_traces()

    def sort_traces(self, *args, **kwargs):
        sorted_keys = self.traces.sort_by_value()
        if sorted_keys is not None:
            self.traces.active_traces = sorted_keys[0]
            self.update_box(items=self.traces.active_traces)
            self.update_func(**kwargs)

    def rename_trace(self, *args, **kwargs):
        try:
            key = self.select_labels[self.ind[0]]
        except Exception:
            return
        else:
            win = EntryDialog(self, prompt="Enter Name", title="Rename trace",
                              input_values=[key])
            self.wait_window(win)
            name = win.output
            if name is not None:
                self.traces.rename_trace(key, name[0])
                self.select_labels[self.ind[0]] = name[0]
                self.update_box(items=self.select_labels)
                self.traces.active_traces = []
                for key in self.traces.tr.keys():
                    self.traces.active_traces.append(key)
                self.update_func(**kwargs)


class TraceCollectionSelection(TraceSelection):
    def __init__(self, parent, traces, **kwargs):
        TraceSelection.__init__(self, parent, traces, **kwargs)
        self.buttons['sort'].config(state='disabled')

    # redefined methods of parent class TraceSelection
    def get_legend_entries(self):
        legend_entries = [tr for member in self.traces.active_members
                          for tr in self.traces.traces[member].active_traces]
        if self.plot_mode.get() == "Single Axes":
            return [[tr for tr in legend_entries]]
        else:
            return [[tr] for tr in legend_entries]

    def get_number_of_active_traces(self):
        return self.traces.get_number_of_active_traces()

    def update_active_traces(self):
        self.traces.active_members = []
        for tr in self.traces.traces.values():
            tr.active_traces = []
        for i in self.ind:
            self.traces.traces[
                self.select_items[i][0]].active_traces.append(
                    self.select_items[i][1])
            if self.select_items[i][0] not in self.traces.active_members:
                self.traces.active_members.append(
                    self.select_items[i][0])

    def _update_box(self, items=None):
        self.select_items = []
        if items is None:
            for key, tr in self.traces.traces.items():
                for lb in tr.tr.keys():
                    self.select.insert(tk.END, lb)
                    self.select_labels.append(lb)
                    self.select_items.append([key, lb])
        else:
            for item in items:
                for key, tr in self.traces.traces.items():
                    if item in tr.tr.keys():
                        self.select.insert(tk.END, item)
                        self.select_labels.append(item)
                        self.select_items.append([key, item])

    def delete_trace(self):
        self.traces.remove_active_traces()
        sel = min(self.ind)
        if sel < 1:
            sel = 1
        self.update_box(select_item=sel)

    def re_order_traces(self, *args, **kwargs):
        return

    def item_insert(self, newindex, oldindex):
        self.select_items.insert(
            newindex, self.select_items.pop(oldindex))

    def rename_trace(self, *args):
        try:
            key = self.select_labels[self.ind[0]]
        except Exception:
            return
        else:
            win = EntryDialog(self, prompt="Enter Name", title="Rename trace",
                              input_values=[key])
            self.wait_window(win)
            name = win.output
            member = self.select_items[self.ind[0]][0]
            if name is not None:
                self.traces.traces[member].rename_trace(key, name[0])
                self.select_labels[self.ind[0]] = name[0]
                self.select_items[self.ind[0]][1] = name[0]
                self.update_box(items=self.select_labels)
                self.update_active_traces()
                if self.update_func is not None:
                    self.update_func()
