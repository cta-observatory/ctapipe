#!/usr/bin/env python3

import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt

from ctapipe.utils import datasets
from ctapipe.visualization import ArrayDisplay

if __name__ == '__main__':

    plt.style.use("ggplot")
    plt.figure(figsize=(10, 8))

    arrayfile = datasets.get_dataset_path("PROD2_telconfig.fits.gz")
    tels = Table.read(arrayfile, hdu="TELESCOPE_LEVEL0")

    adisp = ArrayDisplay(
        tels['TelX'],
        tels['TelY'],
        tels['MirrorArea'] * 2,
        title='PROD2 telescopes',
        autoupdate=True
    )
    plt.tight_layout()

    values = np.zeros(len(tels))

    # do a small animation to show various trigger patterns:

    for ii in range(20):

        # generate a random trigger pattern and integrated intensity:
        ntrig = np.random.poisson(10)
        trigmask = np.random.random_integers(len(tels) - 1, size=ntrig)
        values[:] = 0
        values[trigmask] = np.random.uniform(0, 100, size=ntrig)

        # update the display:
        adisp.values = values
        plt.pause(0.5)
