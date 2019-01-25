"""
Example tool, displaying fake events in a camera.

the animation should remain interactive, so try zooming in when it is
running.
"""

import matplotlib.pylab as plt
import numpy as np
from astropy import units as u
from matplotlib.animation import FuncAnimation

from ctapipe.core import Tool, traits
from ctapipe.image import toymodel, tailcuts_clean, dilate, \
    hillas_parameters, HillasParameterizationError
from ctapipe.instrument import TelescopeDescription, CameraGeometry, \
    OpticsDescription
from ctapipe.visualization import CameraDisplay


class CameraDemo(Tool):
    name = u"ctapipe-camdemo"
    description = "Display fake events in a demo camera"

    delay = traits.Int(50, help="Frame delay in ms", min=20).tag(config=True)
    cleanframes = traits.Int(20, help="Number of frames between turning on "
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

    optics = traits.CaselessStrEnum(
        OpticsDescription.get_known_optics_names(),
        default_value='MST',
        help='Telescope optics description name'
    ).tag(config=True)

    num_events = traits.Int(0, help='events to show before exiting (0 for '
                                    'unlimited)').tag(config=True)

    display = traits.Bool(True, "enable or disable display (for "
                                "testing)").tag(config=True)

    aliases = traits.Dict({
        'delay': 'CameraDemo.delay',
        'cleanframes': 'CameraDemo.cleanframes',
        'autoscale': 'CameraDemo.autoscale',
        'blit': 'CameraDemo.blit',
        'camera': 'CameraDemo.camera',
        'optics': 'CameraDemo.optics',
        'num-events': 'CameraDemo.num_events'
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
        foclen = tel.optics.equivalent_focal_length.to(geom.pix_x.unit).value
        fov = np.deg2rad(4.0)
        scale = foclen
        minwid = np.deg2rad(0.1)
        maxwid = np.deg2rad(0.3)
        maxlen = np.deg2rad(0.5)

        self.log.debug("scale={} m, wid=({}-{})".format(scale, minwid, maxwid))

        disp = CameraDisplay(
            geom, ax=ax, autoupdate=True,
            title="{}, f={}".format(tel, tel.optics.equivalent_focal_length)
        )
        disp.cmap = plt.cm.terrain

        def update(frame):


            centroid = np.random.uniform(-fov, fov, size=2) * scale
            width = np.random.uniform(0, maxwid-minwid) * scale + minwid
            length = np.random.uniform(0, maxlen) * scale + width
            angle = np.random.uniform(0, 360)
            intens = np.random.exponential(2) * 500
            model = toymodel.generate_2d_shower_model(centroid=centroid,
                                                      width=width,
                                                      length=length,
                                                      psi=angle * u.deg)
            self.log.debug(
                "Frame=%d width=%03f length=%03f intens=%03d",
                frame, width, length, intens
            )

            image, sig, bg = toymodel.make_toymodel_shower_image(
                geom,
                model.pdf,
                intensity=intens,
                nsb_level_pe=3,
            )

            # alternate between cleaned and raw images
            if self._counter == self.cleanframes:
                plt.suptitle("Image Cleaning ON")
                self.imclean = True
            if self._counter == self.cleanframes * 2:
                plt.suptitle("Image Cleaning OFF")
                self.imclean = False
                self._counter = 0
                disp.clear_overlays()

            if self.imclean:
                cleanmask = tailcuts_clean(geom, image,
                                           picture_thresh=10.0,
                                           boundary_thresh=5.0)
                for ii in range(2):
                    dilate(geom, cleanmask)
                image[cleanmask == 0] = 0  # zero noise pixels
                try:
                    hillas = hillas_parameters(geom, image)
                    disp.overlay_moments(hillas, with_label=False,
                                         color='red', alpha=0.7,
                                         linewidth=2, linestyle='dashed')
                except HillasParameterizationError:
                    disp.clear_overlays()
                    pass

            self.log.debug("Frame=%d  image_sum=%.3f max=%.3f",
                           self._counter, image.sum(), image.max())
            disp.image = image

            if self.autoscale:
                disp.set_limits_percent(95)
            else:
                disp.set_limits_minmax(-5, 200)

            disp.axes.figure.canvas.draw()
            self._counter += 1
            return [ax, ]

        frames = None if self.num_events == 0 else self.num_events
        repeat = True if self.num_events == 0 else False

        self.log.info("Running for {} frames".format(frames))
        self.anim = FuncAnimation(fig, update,
                                  interval=self.delay,
                                  frames=frames,
                                  repeat=repeat,
                                  blit=self.blit)

        if self.display:
            plt.show()


def main():
    app = CameraDemo()
    app.run()


if __name__ == '__main__':
    main()
