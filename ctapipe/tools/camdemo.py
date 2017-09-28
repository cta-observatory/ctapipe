"""
Example tool, displaying fake events in a camera.

the animation should remain interactive, so try zooming in when it is
running.
"""

import matplotlib.pylab as plt
import numpy as np
from astropy import units as u
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import TelescopeDescription, CameraGeometry, OpticsDescription
from ctapipe.core import Tool, traits
from ctapipe.image import toymodel, tailcuts_clean, dilate
from matplotlib.animation import FuncAnimation


class CameraDemo(Tool):

    name = u"ctapipe-camdemo"
    description = "Display fake events in a demo camera"

    delay = traits.Int(50, help="Frame delay in ms", min=20).tag(config=True)
    cleanframes = traits.Int(100, help="Number of frames between turning on "
                             "cleaning", min=0).tag(config=True)
    autoscale = traits.Bool(False, help='scale each frame to max if '
                                        'True').tag(config=True)
    blit = traits.Bool(False, help='use blit operation to draw on screen ('
                                   'much faster but may cause some draw '
                                   'artifacts)').tag(config=True)
    camera = traits.CaselessStrEnum(
        CameraGeometry.get_known_camera_names(),
        default_value='NectarCam',
        help='Name of camera to display').tag(config=True)

    optics =traits.CaselessStrEnum(
        OpticsDescription.get_known_optics_names(),
        default_value='MST',
        help='Telescope optics description name'
    ).tag(config=True)

    aliases = traits.Dict({
        'delay': 'CameraDemo.delay',
        'cleanframes': 'CameraDemo.cleanframes',
        'autoscale': 'CameraDemo.autoscale',
        'blit': 'CameraDemo.blit',
        'camera': 'CameraDemo.camera',
        'optics' : 'CameraDemo.optics',
    })


    def __init__(self):
        super().__init__()
        self._counter = 0
        self.imclean = False

    def start(self):
        self.log.info("Starting CameraDisplay for {}".format(self.camera))
        self._display_camera_animation()

    def _display_camera_animation(self):
        # plt.style.use("ggplot")
        fig = plt.figure(num="ctapipe Camera Demo", figsize=(7, 7))
        ax = plt.subplot(111)

        # load the camera
        tel = TelescopeDescription.from_name(optics_name=self.optics,
                                             camera_name=self.camera)
        geom = tel.camera

        # poor-man's coordinate transform from telscope to camera frame (it's
        # better to use ctapipe.coordiantes when they are stable)
        scale = tel.optics.effective_focal_length.to(geom.pix_x.unit).value
        fov = np.deg2rad(4.0)
        maxwid = np.deg2rad(0.01)
        maxlen = np.deg2rad(0.03)

        disp = CameraDisplay(geom, ax=ax, autoupdate=True,
                             title="{}, f={}".format(tel,
                             tel.optics.effective_focal_length))
        disp.cmap = plt.cm.terrain

        def update(frame):

            centroid = np.random.uniform(-fov, fov, size=2) * scale
            width = np.random.uniform(0, maxwid) * scale
            length = np.random.uniform(0, maxlen) * scale + width
            angle = np.random.uniform(0, 360)
            intens = np.random.exponential(2) * 50
            model = toymodel.generate_2d_shower_model(centroid=centroid,
                                                      width=width,
                                                      length=length,
                                                      psi=angle * u.deg)
            image, sig, bg = toymodel.make_toymodel_shower_image(geom, model.pdf,
                                                                 intensity=intens,
                                                                 nsb_level_pe=5000)

            # alternate between cleaned and raw images
            if self._counter == self.cleanframes:
                plt.suptitle("Image Cleaning ON")
                self.imclean = True
            if self._counter == self.cleanframes * 2:
                plt.suptitle("Image Cleaning OFF")
                self.imclean = False
                self._counter = 0

            if self.imclean:
                cleanmask = tailcuts_clean(geom, image / 80.0)
                for ii in range(3):
                    dilate(geom, cleanmask)
                image[cleanmask == 0] = 0  # zero noise pixels

            self.log.debug("count = {}, image sum={} max={}"
                           .format(self._counter, image.sum(), image.max()))
            disp.image = image

            if self.autoscale:
                disp.set_limits_percent(95)
            else:
                disp.set_limits_minmax(-100, 4000)

            disp.axes.figure.canvas.draw()
            self._counter += 1
            return [ax, ]

        self.anim = FuncAnimation(fig, update, interval=self.delay,
                                  blit=self.blit)
        plt.show()


def main(args=None):

    app = CameraDemo()
    app.run()


if __name__ == '__main__':
    main()

