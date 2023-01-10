# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 13:11:32 2022

@author: bittmans
Launcher for the different applications contained in the ASTrAS software
package. Apps can also be run directly using scripts (see examples)
"""

from astras import launcher

launcher.launch(config_path='config.txt', window_pad=40)
