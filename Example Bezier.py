# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 23:04:57 2020

@author: Raphael Loo
"""

import pandas as pd
import numpy as np


import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from bezier_splines import Bezier

#%%
# Data processing
# Data from https://fred.stlouisfed.org/series/AHETPI
# We will plot the month-on-month % change (y-axis) in US average hourly earnings
# vs the year-on-year equivalent (x-axis)

df = pd.read_csv('AHETPI.csv', index_col=0).tail(36)
df.index = pd.to_datetime(df.index)

mom = df.pct_change() * 100
mom.columns = ['% MoM']
yoy = df.pct_change(12) * 100
yoy.columns=  ['% YoY']

data = yoy.join(mom).dropna()
#%%
# Plot the graph

swirlogram = Bezier(data)
swirlogram.plot(arrow_width=0.0175)

swirlogram.ax.scatter(data['% YoY'], data['% MoM'], s=16, c='r')

swirlogram.ax.set_xlim((2.4,3.9))
swirlogram.ax.set_ylim((0,0.55))
swirlogram.ax.set_xlabel('% chg YoY')
swirlogram.ax.set_ylabel('% chg MoM')

swirlogram.fig.suptitle('Swirlogram: Last 2Y Wage Growth')
swirlogram.fig.subplots_adjust(
top=0.902,
bottom=0.122,
left=0.104,
right=0.953,
hspace=0.2,
wspace=0.2
)