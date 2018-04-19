#!/usr/bin/env python3

import matplotlib.pylab as plt
from astropy.table import Table
from numpy import ones_like
from ctapipe.utils import datasets
from ctapipe.visualization import ArrayDisplay
from ctapipe.io import event_source

if __name__ == '__main__':

    #plt.style.use("ggplot")
    plt.figure(figsize=(9.5, 8.5))

    source = event_source(datasets.get_dataset_path("gamma_test.simtel.gz"),
                          max_events=1)
    for event in source:
        pass

    subarray = event.inst.subarray

    # display the array, and set the color value to 50
    ad = ArrayDisplay(subarray)

    # label them
#    for tel in tels:
#        name = "CT{tid}".format(tid=tel['TelID'])
#        plt.text(tel['TelX'], tel['TelY'], name, fontsize=8)

    #ad.axes.set_xlim(-1000, 1000)
    #ad.axes.set_ylim(-1000, 1000)
    plt.tight_layout()
    plt.show()
