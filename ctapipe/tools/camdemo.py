"""
Example tool, displaying fake events in a camera.

the animation should remain interactive, so try zooming in when it is
running.
"""

import matplotlib.pylab as plt
from ctapipe import io, visualization
from ctapipe.reco import mock
from ctapipe import reco
from matplotlib.animation import FuncAnimation
import numpy as np
from astropy import units as u

from ctapipe.core import Tool


class CameraDemo(Tool):

    name = u"ctapipe-cam-demo"
    description = "Display fake events in a demo camera"

    def __init__(self):
        super().__init__()
        self._counter = 0

    def start(self):
        self.log.info("Starting Camera Display")
        self._display_cam_animation()

    def _display_cam_animation(self):
        plt.style.use("ggplot")
        fig = plt.figure(num="ctapipe Camera Demo", figsize=(7, 7))
        ax = plt.subplot(111)

        # load the camera
        geom = io.CameraGeometry.from_name("hess", 1)
        disp = visualization.CameraDisplay(geom, ax=ax)
        disp.cmap = plt.cm.terrain

        def update(frame):

            centroid = np.random.uniform(-0.5, 0.5, size=2)
            width = np.random.uniform(0, 0.01)
            length = np.random.uniform(0, 0.03) + width
            angle = np.random.uniform(0, 360)
            intens = np.random.exponential(2) * 50
            model = mock.generate_2d_shower_model(centroid=centroid,
                                                  width=width,
                                                  length=length,
                                                  psi=angle * u.deg)
            image, sig, bg = mock.make_mock_shower_image(geom, model.pdf,
                                                         intensity=intens,
                                                         nsb_level_pe=5000)

            # alternate between cleaned and raw images
            if self._counter > 20:
                plt.suptitle("Image Cleaning ON")
                cleanmask = reco.cleaning.tailcuts_clean(
                    geom, image, pedvars=80)
                for ii in range(3):
                    reco.cleaning.dilate(geom, cleanmask)
                image[cleanmask == 0] = 0  # zero noise pixels
            if self._counter >= 40:
                plt.suptitle("Image Cleaning OFF")
                self._counter = 0

            disp.image = image
            disp.set_limits_percent(100)
            self._counter += 1

        anim = FuncAnimation(fig, update, interval=100)
        plt.show()


def main(args=None):

    app = CameraDemo()
    app.run()


if __name__ == '__main__':
    main()
