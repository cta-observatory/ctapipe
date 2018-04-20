#!/usr/bin/env python3

import matplotlib.pylab as plt
import numpy as np
from ctapipe.utils import datasets
from ctapipe.visualization import ArrayDisplay
from ctapipe.io import event_source
from astropy import units as u

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

    plt.pause(5.0)
    plt.tight_layout()

    for angle in np.linspace(0, 360, 60) * u.deg:
        print(angle, np.sin(angle))
        ad.set_r_phi(np.sin(angle), angle)
        plt.pause(0.01)

    ad.set_r_phi(0,0*u.deg)
    plt.pause(0.01)


