# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:59:48 2018

@author: Jimmy-New_Windows
"""
import sys
import csv
import math
import time
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# just load the data
fname_3pt = 'threes.csv'
data_3pt = pd.read_csv(fname_3pt, sep=',', header=0)
location_x = data_3pt['X_shot']
location_y = data_3pt['Y_shot']
w = data_3pt['Home']

# plot first heatmap
plt.figure()
#plt.figure(figsize=(15, 11.5))
court = plt.imread("fullcourt.png")
edges_x = range(0,96,2)
edges_y = range(0,52,2)
#H, xedges, yedges = np.histogram2d(location_x, location_y, bins=95, weights=w)
H, xedges, yedges = np.histogram2d(location_y, location_x, bins=(edges_y,edges_x))
plt.imshow(H,extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]])

plt.set_cmap('gray_r')
plt.colorbar()
plt.xlabel("x coordinate (ft)", fontsize=20)
plt.ylabel("y coordinate (ft)", fontsize=20)
#plt.imshow(court, zorder=0, extent=[0,94,0,50])
plt.imshow(court, extent=[0,94,0,50])
#plt.xlim(0,95)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# plot second heatmap
plt.figure()
#plt.figure(figsize=(15, 11.5))
court = plt.imread("fullcourt.png")
edges_x = range(0,96,2)
edges_y = range(0,52,2)
#H, xedges, yedges = np.histogram2d(location_x, location_y, bins=95, weights=w)
H_2, xedges_2, yedges_2 = np.histogram2d(location_y, location_x, bins=(edges_y,edges_x), weights=w)

fraction_in = H_2/H
fraction_in[H <= 5] = 0
fraction_in[fraction_in == 0] = np.nan
#plt.imshow(fraction_in,extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]])
plt.imshow(fraction_in,extent=[0,94,0,50])
plt.set_cmap('gray_r')
plt.colorbar()
plt.xlabel("x coordinate (ft)", fontsize=20)
plt.ylabel("y coordinate (ft)", fontsize=20)
#plt.imshow(court, zorder=0, extent=[0,94,50,0])
#plt.imshow(court, extent=[0,94,50,0])
plt.imshow(court, extent=[0,94,50,0])
#plt.xlim(0,95)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# plot third heatmap
plt.figure()
#plt.figure(figsize=(15, 11.5))
court = plt.imread("fullcourt.png")
edges_x = range(0,96,2)
edges_y = range(0,52,2)
#H, xedges, yedges = np.histogram2d(location_x, location_y, bins=95, weights=w)
H_2, xedges_2, yedges_2 = np.histogram2d(location_y, location_x, bins=(edges_y,edges_x), weights=w)

fraction_in = H_2/H
H_2[H <= 5] = 0
fraction_in[fraction_in == 0] = np.nan
#plt.imshow(fraction_in,extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]])
plt.imshow(H_2,extent=[0,94,0,50])
plt.set_cmap('gray_r')
plt.colorbar()
plt.xlabel("x coordinate (ft)", fontsize=20)
plt.ylabel("y coordinate (ft)", fontsize=20)
#plt.imshow(court, zorder=0, extent=[0,94,50,0])
#plt.imshow(court, extent=[0,94,50,0])
plt.imshow(court, extent=[0,94,50,0])
#plt.xlim(0,95)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()