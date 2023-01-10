# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:18:07 2020

@author: bittmans
"""

from astras.ta.app import AppMain

# app = AppMain(geometry = "1400x780", init_canv_geom = "174x174",
#               scrollable = False)
# app = AppMain(geometry = "950x600", init_canv_geom = "174x174",
#               scrollable = True)
app = AppMain(config_filepath='config')
app.mainloop()
