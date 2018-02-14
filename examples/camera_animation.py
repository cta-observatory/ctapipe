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

if __name__ == '__main__':

    plt.style.use("ggplot")
    fig, ax = plt.subplots()

    # load the camera
    tel = TelescopeDescription.from_name("SST-1M", "DigiCam")
    print(tel, tel.optics.equivalent_focal_length)
    geom = tel.camera

    # poor-man's coordinate transform from telscope to camera frame (it's
    # better to use ctapipe.coordiantes when they are stable)
    scale = tel.optics.equivalent_focal_length.to(geom.pix_x.unit).value
    fov = np.deg2rad(4.0)
    maxwid = np.deg2rad(0.01)
    maxlen = np.deg2rad(0.03)

    disp = CameraDisplay(geom, ax=ax)
    disp.cmap = plt.cm.terrain
    disp.add_colorbar(ax=ax)

    def update(frame):
        centroid = np.random.uniform(-fov, fov, size=2) * scale
        width = np.random.uniform(0, maxwid) * scale
        length = np.random.uniform(0, maxlen) * scale + width
        angle = np.random.uniform(0, 360)
        intens = np.random.exponential(2) * 50
        model = toymodel.generate_2d_shower_model(
            centroid=centroid,
            width=width,
            length=length,
            psi=angle * u.deg,
        )
        image, sig, bg = toymodel.make_toymodel_shower_image(
            geom,
            model.pdf,
            intensity=intens,
            nsb_level_pe=5000,
        )
        image /= image.max()
        disp.image = image

    anim = FuncAnimation(fig, update, interval=250)
    plt.show()
