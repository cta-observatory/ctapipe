#!/usr/bin/env python3

import matplotlib.pylab as plt
import numpy as np
from astropy import units as u

from ctapipe.io import event_source
from ctapipe.utils import datasets
from ctapipe.visualization import ArrayDisplay

if __name__ == '__main__':

    plt.figure(figsize=(9.5, 8.5))

    # load up a single event, so we can get the subarray info:
    source = event_source(
        datasets.get_dataset_path("gamma_test_large.simtel.gz"),
        max_events=1,
    )

    event = next(iter(source))

    # display the array
    subarray = event.inst.subarray
    ad = ArrayDisplay(subarray, tel_scale=3.0)

    print("Now setting vectors")
    plt.pause(1.0)
    plt.tight_layout()

    for phi in np.linspace(0, 360, 30) * u.deg:
        r = np.cos(phi / 2)
        ad.set_vector_rho_phi(r, phi)
        plt.pause(0.01)

    ad.set_vector_rho_phi(0, 0 * u.deg)
    plt.pause(1.0)

    print("Now setting values")
    ad.telescopes.set_linewidth(0)
    for ii in range(50):
        vals = np.random.uniform(100.0, size=subarray.num_tels)
        ad.values = vals
        plt.pause(0.01)

    print("Setting labels")
    for ii in range(3):
        ad.add_labels()
        plt.pause(0.5)
        ad.remove_labels()
        plt.pause(0.5)
