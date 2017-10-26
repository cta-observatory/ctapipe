#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple plot of reconstructed extinction profile

@author: Johan Bregeon
"""

from traitlets import Unicode
from ctapipe.utils import get_dataset

from matplotlib import pyplot as plt

from ctapipe.atmosphere.lidar_processor import pLidarRun

# JB - doesn't understand traitlets...
# set CTAPIPE_SVC_PATH to a path containing the file below
#lidar_file_path = Unicode(get_dataset('hess_elastic_lidar_data.txt'), allow_none=True,
#                          help='Path to the atmospheric profile file, e.g.'
#                               'hess_elastic_lidar_data.txt').tag(config=True)
lidar_file_path = get_dataset('hess_elastic_lidar_data.txt')

r=pLidarRun()
r.readFile(lidar_file_path)
alpha_wl1, alpha_wl2=r.process()
altitudes=r.BinsAltCenter[:-1]*1000.+1650
print('Date Time: %s'%r.DateTime)
print('Background: %.5f %.5f'%(r.BkgWL1, r.BkgWL2))
print('Tau4: %.2f %.2f'%r.calcTau4())
print('Prob: %.2f %.2f'%r.calcTransmission(h=4))

# plot both wavelength on same plot
fig = plt.figure(figsize=(6, 8))
ax = fig.add_subplot(111)
wl355, = ax.plot(alpha_wl1/1000., altitudes, color='blue')
wl532, = ax.plot(alpha_wl2/1000., altitudes, color='green')
legend_handles = [wl355, wl532]
legend_labels = ["355 nm", "532 nm"]

plt.title('Lidar Atmospheric Extinction')
ax.set_xlabel("Extinction (m^-1)")
#ax.set_xbound(0, 1e-1)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.set_ylabel("Altitude")
ax.set_ybound(0, 12000)
ax.legend(legend_handles, legend_labels,
          bbox_to_anchor=(1.02, 1.), loc=1,
          borderaxespad=0., fontsize=10)
plt.show()
