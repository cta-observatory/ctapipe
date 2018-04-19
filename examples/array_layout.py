#!/usr/bin/env python3

import matplotlib.pylab as plt
from astropy.table import Table
from numpy import ones_like

from ctapipe.utils import datasets
from ctapipe.visualization import ArrayDisplay

if __name__ == '__main__':

    plt.style.use("ggplot")
    plt.figure(figsize=(9.5, 8.5))

    # load up an example table that has the telescope positions and
    # mirror areas in it:
    arrayfile = datasets.get_dataset_path("PROD2_telconfig.fits.gz")
    tels = Table.read(arrayfile, hdu="TELESCOPE_LEVEL0")

    X = tels['TelX']
    Y = tels['TelY']
    A = tels['MirrorArea'] * 2  # exaggerate scale a bit

    # display the array, and set the color value to 50
    ad = ArrayDisplay(X, Y, A, title="Prod 2 Full Array")
    ad.values = ones_like(X) * 50

    # label them
    for tel in tels:
        name = "CT{tid}".format(tid=tel['TelID'])
        plt.text(tel['TelX'], tel['TelY'], name, fontsize=8)

    ad.axes.set_xlim(-1000, 1000)
    ad.axes.set_ylim(-1000, 1000)
    plt.tight_layout()
    plt.show()
