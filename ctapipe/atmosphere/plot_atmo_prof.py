#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot Atmospheric profile table

@author: Johan Bregeon
"""

# @TODO 
#

from traitlets import Unicode
from ctapipe.utils import get_dataset

from matplotlib import pyplot as plt
from ctapipe.atmosphere.atmo_prof import read_atmo_prof

# set CTAPIPE_SVC_PATH to a path containing the file below
input_path=get_dataset('atmprof26.dat')

# can't make this work...
#input_path = Unicode(get_dataset('atmprof26.dat'), allow_none=True,
#                     help='Path to the atmospheric profile file, e.g. '
#                          'atmprof26.dat').tag(config=True)    

altitudes, rho, thick, index, temperature, pressure, pw_p = \
       read_atmo_prof(input_path)
       
fig = plt.figure(figsize=(6, 8))
ax  = fig.add_subplot(111)
#ax_l = self.fig.add_subplot(121)
#ax_r = self.fig.add_subplot(122)

# plot 2 reference wave lengths

pprof, = ax.plot(pressure, altitudes, color='blue')
tprof, = ax.plot(temperature, altitudes, color='green')
legend_handles = [pprof, tprof]
legend_labels = ["Pressure", "Temperature"]
    

plt.title('Atmospheric Profile')
ax.set_xlabel("Pressure (mb), Temperature (K)")
ax.set_xbound(0, 1100)
ax.set_ylabel("Altitudes")
ax.set_ybound(0, 120000)

ax.legend(legend_handles, legend_labels,
          bbox_to_anchor=(1.0, 1.), loc=0,
          borderaxespad=0., fontsize=10)
plt.show()
