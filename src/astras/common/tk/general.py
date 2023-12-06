# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:48:12 2022

@author: bittmans
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import re
import time
import numpy as np
import threading
# from uncertainties import ufloat


def save_box(title=None, fname=None, dir_name=None,
             fext=".txt", filetypes=None, as_file=False,
             return_ext=False, parent=None):
    if filetypes is None:
        filetypes = [('all files', '.*'), ('text files', '.txt')]
    options = {}
    options['defaultextension'] = fext
    options['filetypes'] = filetypes
    options['initialdir'] = dir_name
    options['initialfile'] = fname
    options['title'] = title
    if as_file:
        file = filedialog.asksaveasfile(mode='w', parent=parent, **options)
    else:
        file = filedialog.asksaveasfile(parent=parent, **options)
    try:
        file.name
    except Exception:
        if return_ext:
            return None, None
        else:
            return None
    else:
        if return_ext:
            try:
                ext = re.findall("\....", file.name)[-1]
            except Exception:
                ext = None
            return file, ext
        else:
            return file


def load_box(title=None, filetypes=None, default_dir=None,
             fext=".txt", as_file=False, parent=None):
    if filetypes is None:
        filetypes = [('all files', '.*')]
    options = {'filetypes': filetypes,
               'initialfile': default_dir,
               'defaultextension': fext}

    if as_file:
        file = filedialog.askopenfile(mode='w', parent=parent, **options)
    else:
        file = filedialog.askopenfile(parent=parent, **options)
    try:
        ext = file.name[-re.search("\.", file.name[::-1]).end():]
    except Exception:
        ext = None
    return file, ext


def move_toplevel_to_default(window, controller):
    window.geometry("+%d+%d" % (controller.winfo_x(), controller.winfo_y()))
    window.update()


def center_toplevel(window, controller, offset=None):
    if offset is None:
        offset = controller.winfo_width(), controller.winfo_height()
    window.update()
    window.geometry("+%d+%d" % (controller.winfo_x() + 0.5*offset[0] -
                                0.5*window.winfo_width(),
                                controller.winfo_y() + 0.5*offset[1] -
                                0.5*window.winfo_height()))
    window.update()


def test_numeric(argin):
    out = False
    try:
        float(argin)
    except ValueError:
        messagebox.showerror(message="Please enter numeric value")
    else:
        out = True
    return out


def general_error(*args):
    messagebox.showerror("Error", str(
        args[0]) + ' (' + repr(type(args[0])) + ')')


def no_data_error(*args, **kwargs):
    messagebox.showerror("No data loaded", "Please load data via file menu.")


def unknown_error(*args):
    if args:
        messagebox.showerror(
            "Error", "Unknown Error in source code line:" + str(args))
    else:
        messagebox.showerror("Error", "Unknown Error.")


def enable_disable_child_widgets(parent, case='enable'):
    if case == 'enable':
        case = 'normal'
    elif case not in ('disabled','normal'):
        case = 'disabled'
    for child in parent.winfo_children():
        child.configure(state=case)


# classes
class TkVarMimic():
    def __init__(self, value=None):
        self._value = value

    def get(self):
        return self._value

    def set(self, value=None):
        self._value = value


class GroupBox(ttk.LabelFrame):
    def __init__(self, parent, dim=(1, 1), weight=1, text=None,
                 labelstyle="Custom.TLabelframe.Label",
                 style="Custom.TLabelframe", **kwargs):

        lb = ttk.Label(parent, text=text)
        ttk.LabelFrame.__init__(self, parent, labelwidget=lb,
                                **kwargs)
        try:
            lb.configure(style=labelstyle)
        except Exception:
            pass
        try:
            self.configure(style=style)
        except Exception:
            pass
        for i in range(dim[0]):
            self.columnconfigure(i, weight=weight)
        for i in range(dim[1]):
            self.rowconfigure(i, weight=weight)


class CustomFrame(tk.Frame):
    def __init__(self, parent, dim=(1, 1), border=False, weight=1,
                 highlightbackground="light gray", highlightcolor="light gray",
                 highlightthickness=1, bd=0, **frame_kwargs):
        if border:
            tk.Frame.__init__(self, parent,
                              highlightbackground=highlightbackground,
                              highlightthickness=highlightthickness,
                              highlightcolor=highlightcolor, bd=bd,
                              **frame_kwargs)
        else:
            tk.Frame.__init__(self, parent, **frame_kwargs)
        for i in range(dim[0]):
            self.columnconfigure(i, weight=weight)
        for i in range(dim[1]):
            self.rowconfigure(i, weight=weight)


class CustomEntry(tk.Frame):
    def __init__(self, parent, framekwargs={'borderwidth': 5,
                                            'relief': tk.SUNKEN},
                 **entrykwargs):
        tk.Frame.__init__(self, parent, **framekwargs)
        self.entry = tk.Entry(self, relief=tk.FLAT, **entrykwargs)
        self.entry.grid()


class BinaryDialog(tk.Toplevel):
    def __init__(self, parent, controller=None, title=None, prompt="Continue?",
                 yes_button_text="Continue", no_button_text="Cancel"):
        tk.Toplevel.__init__(self, parent)
        if title is not None:
            self.title(title)
        tk.Label(self, text=prompt).grid(row=0, column=0, columnspan=2,
                                         sticky='wnse', padx=10, pady=10)
        ttk.Button(self, text=yes_button_text, command=self.continue_).grid(
            row=1, column=0, padx=5, pady=5)
        ttk.Button(self, text=no_button_text, command=self.destroy).grid(
            row=1, column=1, padx=5, pady=5)
        self.output = False
        if controller is not None:
            center_toplevel(self, controller)
        parent.wait_window(self)

    def continue_(self):
        self.output = True
        self.destroy()


class MultipleOptionsWindow(tk.Toplevel):
    def __init__(self, parent, controller, opts, text='Select options',
                 buttontext='Continue', check_values=None, wait=True,
                 **grid_kwargs):
        tk.Toplevel.__init__(self, parent)
        frame = tk.Frame(self)
        self.boxes = {}
        self.opts = opts
        self.opts.append('all')
        if check_values is None:
            check_values = []
        if len(check_values) != len(opts):
            check_values = []
            for i in range(len(opts)):
                check_values.append(0)
        tk.Label(frame, text=text).grid(
            row=0, column=0, sticky=tk.W+tk.E, **grid_kwargs)
        for i in range(len(opts)):
            self.boxes[opts[i]] = [tk.IntVar(value=check_values[i]), []]
            self.boxes[opts[i]][1] = ttk.Checkbutton(
                frame, text=opts[i], variable=self.boxes[opts[i]][0])
            self.boxes[opts[i]][1].grid(
                row=i + 1, column=0, sticky=tk.W, **grid_kwargs)
        self.boxes['all'][1].config(command=self.check_all_callback)
        self.protocol("WM_DELETE_WINDOW", self._delete_window)
        ttk.Button(frame, text=buttontext, command=self._delete_window).grid(
            row=len(opts) + 1, column=0, sticky=tk.W+tk.E, **grid_kwargs)
        frame.grid(sticky='wnse', padx=5, pady=5)

        center_toplevel(self, controller)
        if wait:
            parent.wait_window(self)

    def _delete_window(self):
        self.output = []
        for o in self.opts:
            if self.boxes[o][0].get():
                self.output.append(o)
        self.destroy()

    def check_all_callback(self, *args):
        if self.boxes['all'][0].get():
            for o in self.opts:
                self.boxes[o][0].set(1)
        else:
            for o in self.opts:
                self.boxes[o][0].set(0)


class EntryDialog(tk.Toplevel):
    def __init__(self, parent, controller=None, prompt=None,
                 entry_num=1, entrylabels=None, entry_type="string",
                 input_values=None, title=None):
        tk.Toplevel.__init__(self, parent)
        if prompt is None:
            prompt = "Please enter"
        if controller is not None:
            self.controller = controller
        if title is not None:
            self.title(title)
        self.parent = parent
        self.entries = []
        self.vars = []
        self.output = None

        for i in range(entry_num):
            column = 0
            try:
                tk.Label(self, text=entrylabels[i]).grid(
                    row=i, column=column, sticky='w', padx=5, pady=5)
                column = column + 1
            except Exception:
                pass
            if entry_type == "string":
                self.vars.append(tk.StringVar())
            elif entry_type == 'double':
                self.vars.append(tk.DoubleVar())
            else:
                self.vars.append(tk.StringVar())
            self.entries.append(tk.Entry(self, textvariable=self.vars[-1]))
            self.entries[-1].grid(row=i, column=column, padx=5, pady=5)
            if input_values is not None:
                try:
                    self.vars[-1].set(input_values[i])
                except Exception:
                    pass
        self.button_frame = CustomFrame(self, dim=(2, 1))
        ttk.Button(self.button_frame, text="OK",
                   command=self.quit_window).grid(
                       row=0, column=0, padx=5, pady=5)
        ttk.Button(self.button_frame, text="Cancel",
                   command=self.destroy).grid(
                       row=0, column=1, padx=5, pady=5)
        self.button_frame.grid(row=entry_num, column=0, padx=10, pady=10,
                               columnspan=2 - int(entrylabels is None),
                               sticky='wnse')

    def quit_window(self):
        self.output = []
        for var in self.vars:
            self.output.append(var.get())
        self.destroy()


class CustomTimer(tk.Label):
    def __init__(self, parent, text="Running Task.", start=False, digits=2):
        tk.Label.__init__(self, parent, text=text)
        self.text = text
        self.digits = digits
        self.paused = False
        self.elapsed = 0
        self.previous_time = 0
        if start:
            self.start()

    def start(self):
        self.running = True
        self.starting_time = time.time()
        self.previous_time = 0
        self.elapsed = 0
        threading.Thread(target=self.runTimer).start()

    def stop(self):
        self.running = False

    def pause(self):
        self.paused = True
        self.previous_time = self.elapsed

    def cont(self):
        self.starting_time = time.time()
        self.paused = False

    def runTimer(self):
        while self.running:
            time.sleep(0.2)
            currentTime = time.time()
            self.elapsed = self.previous_time + \
                np.round(currentTime-self.starting_time, self.digits)
            self.config(text=self.text + " Elapsed time: " +
                        str(self.elapsed) + " s.")
            while self.paused:
                time.sleep(0.2)


class CustomProgressbarWindow(tk.Toplevel):
    def __init__(self, parent, controller=None, title=None,
                 cancel_button=False, **kwargs):
        tk.Toplevel.__init__(self, parent)
        if title:
            self.title(title)
        self.frame = CustomProgressbar(self, **kwargs)
        self.frame.grid(row=0, column=0, sticky='nwse')
        self.val = self.frame.val
        self.bar = self.frame.bar
        self.start_timer = self.frame.start_timer
        self.update_timer = self.frame.update_timer
        self.update_label = self.frame.update_label
        self.increase_value = self.frame.increase_value
        self.update_value = self.frame.update_value
        self.set_max = self.frame.set_max
        self.reset_max = self.frame.reset_max
        self.get_timer_started = self.frame.get_timer_started
        self.get_cancelled = self.frame.get_cancelled
        self.set_val_ratio = self.frame.set_val_ratio

        if cancel_button:
            ttk.Button(self, text="Cancel", command=self.cancel).grid(
                row=1, column=0, padx=5, pady=5)
        if controller is not None:
            center_toplevel(self, controller)

    def cancel(self):
        self.cancelled = True
        self.frame.cancelled = True
        self.frame.cancel_callback()
        self.destroy()


class CustomProgressbar(tk.Frame):
    def __init__(self, parent, n=300, text="Running operation. Please Wait.",
                 num_digits=3, start_timer=False):
        tk.Frame.__init__(self, parent)
        self.cancelled = False
        self.timer_started = False
        self.num_digits = num_digits
        self.toplabel = tk.Label(self, text=text)
        self.toplabel.grid(row=0, column=0)
        self.val = tk.DoubleVar(value=1)
        self.bar = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=n,
                                   mode='determinate', variable=self.val)
        self.bar.grid(row=1, column=0, padx=5, pady=5)

        self.bar["value"] = 1
        self.bar["maximum"] = n
        if start_timer:
            self.start_timer()

    def start_timer(self):
        self.starting_time = time.time()
        self.timer_started = True
        self.time_label = tk.Label(self, text='Elapsed time: 0 s')
        self.time_label.grid(row=2, column=0, padx=5, pady=5)

    def update_timer(self):
        try:
            self.time_label.config(
                text='Elapsed time: ' + str(np.round((
                    time.time() - self.starting_time), self.num_digits)) + 's')
        except Exception:
            return
        else:
            self.update()

    def get_timer_started(self):
        return self.timer_started

    def get_cancelled(self):
        return self.cancelled

    def update_label(self, string):
        self.toplabel.config(text=string)
        self.update()

    def increase_value(self):
        self.val.set(self.val.get() + 1)
        self.bar.update_idletasks()
        self.update()

    def update_value(self, val):
        self.val.set(val)
        self.bar.update_idletasks()
        self.update()

    def set_val_ratio(self, val):
        val = val * self.bar["maximum"]
        self.update_value(val)

    def set_max(self, val):
        self.bar["maximum"] = val
    
    def cancel_callback(self, *args, **kwargs):
        return

    def reset_max(self, val):
        self.set_max(val)
        self.update_value(0)


# %%
class MultiDisplayWindow(tk.Toplevel):
    def __init__(self, parent, *frame_args, controller=None, title=None,
                 header=None, close_button=None, **frame_kwargs):
        tk.Toplevel.__init__(self, parent)
        row = 0
        if title is not None:
            self.title(title)
            if header is None:
                ttk.Label(self, text=title).grid(row=row, column=0)
                row += 1
        self.fr = MultiDisplay(
            self, *frame_args, header=header, **frame_kwargs)
        self.fr.grid(row=row, column=0, sticky='wnse')
        if close_button is not None:
            ttk.Button(self, text=close_button, command=self.destroy).grid(
                row=row + 1, column=0)
        if controller is not None:
            center_toplevel(self, controller)


class MultiDisplay(tk.Frame):
    def __init__(self, parent, header=None,
                 input_dict=None, mode='simple',
                 orient='vertical',
                 parent_update_func=None):
        tk.Frame.__init__(self, parent)
        self.frames = {}
        self.sub_frames = {}
        self.row = 0
        self.mode = mode
        self.vert_orient = orient.lower() == 'vertical'
        self.update_parent = parent_update_func

        if header is not None:
            ttk.Label(self, text=header).grid(row=self.row, column=0)
            self.row += 1
        if input_dict is None:
            input_dict = {"Test 1": {"type": "label",
                                     "content": "Message 1",
                                     "grid_kwargs": {}},
                          "Test 2": {"type": "label",
                                     "content": "Message 2",
                                     "grid_kwargs": {}}}
            self.mode = 'simple'
        self.selected_frame = tk.StringVar()
        try:
            self.selected_frame.set(list(input_dict.keys())[0])
        except Exception:
            pass
        self.frame_select = ttk.OptionMenu(
            self, self.selected_frame, self.selected_frame.get(),
            *input_dict.keys())
        if len(list(input_dict.keys())) > 1:
            self.frame_select.grid(row=self.row, column=0)
            self.row += 1

        if input_dict:
            self.add_frames(input_dict)
        self.selected_frame.trace('w', self.show_frame)

    def add_frames(self, input_dict):
        if self.mode == 'simple':
            for key, val in input_dict.items():
                self.frames[key] = tk.Frame(self)
                if val["type"] == "label":
                    ttk.Label(self.frames[key], text=val["content"]).grid(
                        **val["grid_kwargs"])
                else:
                    print("Widget \""
                          + val["type"]
                          + "\" not available yet for class MultiDisplay."
                          + " Feel free to add.")
                self.frames[key].grid(row=self.row,
                                      column=0, sticky="wnse", padx=5, pady=5)
        elif self.mode == 'expandable':
            for key, val in input_dict.items():
                sub_row = 0
                sub_col = 0
                self.frames[key] = tk.Frame(self)
                self.sub_frames[key] = {}
                for section, content in val["content"].items():
                    header = ttk.Label(
                        self.frames[key], text=section.join(["[[", "]]"]))
                    header.grid(row=sub_row, column=sub_col,
                                **val["grid_kwargs"])
                    self.sub_frames[key][section] = {
                        'label': ttk.Label(self.frames[key],
                                           text=content['text']),
                        'visible': not content['visible']}
                    self.sub_frames[key][section]["label"].grid(
                        row=sub_row + 1, column=sub_col, **val["grid_kwargs"])
                    self.sub_frames[key][section]["label"].grid_remove()
                    header.bind('<Button-1>', (
                        lambda *args, key=section:
                            self.toggle_sub_frame(*args, key=key)))
                    sub_row += 2*int(self.vert_orient)
                    sub_col += int(not self.vert_orient)
                self.frames[key].grid(
                    row=self.row, column=0, sticky="wnse", padx=5, pady=5)
                for k in self.sub_frames[key].keys():
                    self.toggle_sub_frame(frame=key, key=k)
        menu = self.frame_select['menu']
        menu.delete(0, 'end')
        for key in self.frames.keys():
            menu.add_command(
                label=key, command=lambda k=key: self.selected_frame.set(k))
        self.select_frame()

    def select_frame(self, *args, frame=None):
        if frame is None:
            self.show_frame()
        else:
            self.selected_frame.set(frame)

    def show_frame(self, *args):
        self.frames[self.selected_frame.get()].tkraise()

    def toggle_sub_frame(self, *args, frame=None, key=None):
        if key is None:
            return
        if frame is None:
            frame = self.selected_frame.get()
        if self.sub_frames[frame][key]['visible']:
            self.sub_frames[frame][key]["label"].grid_remove()
            self.sub_frames[frame][key]['visible'] = False
        else:
            self.sub_frames[frame][key]["label"].grid()
            self.sub_frames[frame][key]['visible'] = True
        if self.update_parent:
            self.update_parent()


# %%
class BusyWindow(tk.Toplevel, tk.Frame):
    def __init__(self, parent, controller, task_name='calculations',
                 message='', case='Toplevel'):
        if case.lower() == 'frame':
            tk.Frame.__init__(self, parent)
        else:
            tk.Toplevel.__init__(self, parent, takefocus=True)
        tk.Label(self, text='Running ' + task_name + '...\n' + message
                 + '\nPlease wait.').grid(padx=10, pady=10)
        center_toplevel(self, controller)


class LoadScreen(tk.Frame, tk.Toplevel):
    def __init__(self, parent, controller=None, message=None,
                 geometry="1000x650", offset=None,
                 progressbar_options={'orient': 'horizontal',
                                      'length': 500,
                                      'mode': 'determinate'},
                 progressbar_steps=5):
        tk.Toplevel.__init__(self, parent, takefocus=True)
        self.lift()
        self.overrideredirect(1)
        self.geometry(geometry)
        if message:
            tk.Label(self, text=message).grid(
                row=0, column=0, sticky=tk.W+tk.E,
                pady=(self.winfo_height()/2, 5))
        if not controller:
            controller = parent
        center_toplevel(self, controller, offset=offset)
        self.progressbar = ttk.Progressbar(self, **progressbar_options)
        self.progressbar.update()
        self.progressbar["value"] = 0
        self.progressbar["maximum"] = progressbar_steps
        self.progressbar.grid(row=1, column=0, sticky='wnse',
                              padx=self.winfo_width()/2
                              - progressbar_options["length"]/2,
                              pady=(5, self.winfo_height()/2))

    def progbar_advance(self, incr=1):
        self.progressbar["value"] = self.progressbar["value"] + incr
        self.progressbar.update()
        self.update()


# %%
class ScrollFrame(CustomFrame):
    def __init__(self, parent, *widget_args, widget=None, scroll_dir=None,
                 frame_kwargs=None, width=None, height=None, **widget_kwargs):
        if not frame_kwargs:
            frame_kwargs = {}
        if not scroll_dir:
            scroll_dir = 'y'
        CustomFrame.__init__(self, parent, **frame_kwargs)
        self.canvas = tk.Canvas(self)

        if widget:
            self.widget = widget(self.canvas, *widget_args, **widget_kwargs)
            self.widget.grid(sticky='nswe')
            self.widget.update()
            if not width:
                width = self.widget.winfo_width()
            if not height:
                height = self.widget.winfo_height()
        else:
            self.widget = tk.Frame(self.canvas)
        if width:
            self.canvas.config(width=width)
        if height:
            self.canvas.config(height=height)
        self.update()
        if re.search('y', scroll_dir, re.I):
            self.scrollbary = tk.Scrollbar(self, orient="vertical",
                                           command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.scrollbary.set)
            self.scrollbary.grid(row=0, column=1, sticky='sne')
        if re.search('x', scroll_dir, re.I):
            self.scrollbarx = tk.Scrollbar(self, orient="horizontal",
                                           command=self.canvas.xview)
            self.canvas.configure(xscrollcommand=self.scrollbarx.set)
            self.scrollbarx.grid(row=1, column=0, sticky='sew')
        self.widget.bind(
            "<Configure>", lambda *args:
                self.canvas.config(scrollregion=self.canvas.bbox("all")))
        self.frame_window = self.canvas.create_window(
            (1, 1), window=self.widget, anchor='nw', tags="self.widget")
        self.canvas.grid(row=0, column=0, sticky='wnse')

    def _config_canvas(self, e):
        self.canvas.itemconfig(
            self.frame_window, height=e.height, width=e.width)

    def update_window(self):
        self.update()
        self.frame_window = self.canvas.create_window(
            (1, 1), window=self.widget, anchor='nw', tags="self.widget")


class GlobalSettingsWindow(tk.Toplevel):
    # work in progress, meant for interactive manipulation of the
    # global settings file. For now direct editing of the .txt is
    # required
    def __init__(self, parent, controller, update_function=None):
        return
