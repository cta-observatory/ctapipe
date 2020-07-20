#!/usr/bin/env python3
"""
Example of drawing and updating a Camera using a toymodel shower images.

the animation should remain interactive, so try zooming in when it is
running.
"""

import matplotlib.pylab as plt
import numpy as np
from astropy import units as u
from matplotlib.animation import FuncAnimation

from ctapipe.image import toymodel
from ctapipe.instrument import TelescopeDescription
from ctapipe.visualization import CameraDisplay

if __name__ == "__main__":

    plt.style.use("ggplot")
    fig, ax = plt.subplots()

    # load the camera
    tel = TelescopeDescription.from_name("SST-1M", "DigiCam")
    geom = tel.camera.geometry

    fov = 0.3
    maxwid = 0.05
    maxlen = 0.1

    disp = CameraDisplay(geom, ax=ax)
    disp.cmap = "inferno"
    disp.add_colorbar(ax=ax)

    def update(frame):
        x, y = np.random.uniform(-fov, fov, size=2)
        width = np.random.uniform(0.01, maxwid)
        length = np.random.uniform(width, maxlen)
        angle = np.random.uniform(0, 180)
        intens = width * length * (5e4 + 1e5 * np.random.exponential(2))

        model = toymodel.Gaussian(
            x=x * u.m,
            y=y * u.m,
            width=width * u.m,
            length=length * u.m,
            psi=angle * u.deg,
        )
        image, _, _ = model.generate_image(geom, intensity=intens, nsb_level_pe=5,)
        disp.image = image

    anim = FuncAnimation(fig, update, interval=500)
    plt.show()
