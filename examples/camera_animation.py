"""
Example of drawing and updating a Camera using a mock shower images.

the animation should remain interactive, so try zooming in when it is
running.
"""

import matplotlib.pylab as plt
from ctapipe import io, visualization
from ctapipe.reco import mock
from matplotlib.animation import FuncAnimation
import numpy as np
from astropy import units as u


if __name__ == '__main__':

    plt.style.use("ggplot")
    fig, ax = plt.subplots()

    # load the camera
    geom = io.CameraGeometry.from_name("hess", 1)
    disp = visualization.CameraDisplay(geom, ax=ax)
    disp.cmap = plt.cm.terrain
    disp.add_colorbar(ax=ax)

    def update(frame):
        centroid = np.random.uniform(-0.5, 0.5, size=2)
        width = np.random.uniform(0, 0.01)
        length = np.random.uniform(0, 0.03) + width
        angle = np.random.uniform(0, 360)
        intens = np.random.exponential(2) * 50
        model = mock.generate_2d_shower_model(
            centroid=centroid,
            width=width,
            length=length,
            psi=angle * u.deg,
        )
        image, sig, bg = mock.make_mock_shower_image(
            geom, model.pdf,
            intensity=intens,
            nsb_level_pe=5000,
        )
        image /= image.max()
        disp.image = image


    anim = FuncAnimation(fig, update, interval=250)
    plt.show()
