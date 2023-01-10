# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 13:11:32 2022

@author: bittmans
Launcher for the different applications contained in the ASTrAS software
package. Apps can also be run directly using scripts (see examples)
"""
import tkinter as tk
from astras.ta.app import AppMain as TA
from astras.linefit.app import AppMain as Line
from astras.stabs.app import AppMain as Stabs
from astras.paradep.app import AppMain as ParaDep
from astras.common.tk.figures import FigureEditor
from astras.common.tk.general import CustomFrame
# from astras.ta.app import AppMain


def launch(*args, window_pad=40, **kwargs):
    app = Launcher(*args, window_pad=window_pad, **kwargs)
    app.mainloop()
    return app

class Launcher(tk.Tk):
    def __init__(self, *args, window_pad=40,
                 config_path='config.txt', **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        # self.geometry(geometry)
        frame = CustomFrame(self, dim=(1,6))
        frame.grid(sticky='wnse', padx=window_pad, pady=window_pad)
        self.title("ASTrAS Launcher")
        self.config_path = config_path
        common_widget_kwargs = {'column': 0,
                                'padx': 10,
                                'pady': 10,
                                'sticky': 'we'}
        tk.ttk.Label(frame, text="Launch App").grid(
            row=0, column=0)
        tk.ttk.Button(frame, text="Time-resolved Spectroscopy",
                      command=self.launch_ta_app).grid(
                          row=1, **common_widget_kwargs)
        tk.ttk.Button(frame, text="Line Fit App",
                      command=self.launch_line_app).grid(
                          row=2, **common_widget_kwargs)
        tk.ttk.Button(frame, text="Static Absorption Spectroscopy",
                      command=self.launch_stabs_app).grid(
                          row=3, **common_widget_kwargs)
        tk.ttk.Button(frame, text="Parameter Dependence",
                      command=self.launch_paradep_app).grid(
                          row=4, **common_widget_kwargs)
        tk.ttk.Button(frame, text="Figure Editor",
                      command=self.launch_figure_edit).grid(
                          row=5, **common_widget_kwargs)

    def launch_ta_app(self):
        win = TA(config_filepath=self.config_path, parent=self)
        self.wait_window(win)
        
    def launch_line_app(self):
        win = Line(config_filepath=self.config_path, parent=self)
        self.wait_window(win)

    def launch_stabs_app(self):
        win = Stabs(parent=self)
        self.wait_window(win)

    def launch_paradep_app(self):
        win = ParaDep(parent=self)
        self.wait_window(win)

    def launch_figure_edit(self):
        win = FigureEditor(parent=self, config_filepath=self.config_path)
        self.wait_window(win)
        
# launch()
