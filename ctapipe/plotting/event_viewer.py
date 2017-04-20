import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from ctapipe import visualization, io
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import HessioR1Calibrator
from ctapipe.instrument import CameraGeometry
from ctapipe.plotting.array import ArrayPlotter, NominalPlotter
from numpy import ceil, sqrt
from ctapipe.core import Component
from ctapipe.core.traits import Float, Bool


class EventViewer(Component):

    name = 'EventViewer'
    test = Bool(True, help='').tag(config=True)

    def __init__(self, draw_hillas_planes=False):

        self.array_view = None
        self.nominal_view = None

        self.geom = dict()
        self.cam_display = dict()
        self.draw_hillas_planes = draw_hillas_planes

    def draw_source(self, source):

        for event in source:
            self.draw_event(event)

        return

    def draw_event(self, event, hillas_parameters=None):

        tel_list = event.r0.tels_with_data
        images = event.dl1

        plt.close()
        ntels = len(tel_list)
        nn = int(ceil(sqrt(ntels)))

        fig = plt.figure(figsize=(20, 20 * 0.66))

        if self.draw_hillas_planes:
            y_axis_split = 1
        else:
            y_axis_split = 2

        outer_grid = gridspec.GridSpec(1, y_axis_split, width_ratios=[y_axis_split, 1])
        camera_grid = gridspec.GridSpecFromSubplotSpec(nn, nn, subplot_spec=outer_grid[0])

        ii = 0
        for ii, tel_id in zip(range(len(tel_list)), tel_list):
            if tel_id not in self.geom:
                self.geom[tel_id] = CameraGeometry.guess(
                    event.inst.pixel_pos[tel_id][0],
                    event.inst.pixel_pos[tel_id][1],
                    event.inst.optical_foclen[tel_id])

            ax = plt.subplot(camera_grid[ii])
            self.get_camera_view(tel_id, images.tel[tel_id].image[0], ax)

        reco_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1])

        array = ArrayPlotter(telescopes=tel_list, instrument=event.inst,# system=tilted_system,
                             ax=plt.subplot(reco_grid[0]))

        array.draw_position(event.mc.core_x, event.mc.core_y, use_centre=True)
        array.draw_array(((-300,300),(-300,300)))

        if hillas_parameters is not None:
            array.overlay_hillas(hillas_parameters)

            nominal =  NominalPlotter(hillas_parameters=hillas_parameters, draw_axes=True, ax=plt.subplot(reco_grid[1]))
            nominal.draw_array()


        plt.show()

        return

        return 0

    def get_camera_view(self, tel_id, image, axis):
        """

        Parameters
        ----------
        images

        Returns
        -------

        """
        #if tel_id not in self.cam_display:
        # Argh this is annoying, for some reason we cannot cahe the displays
        self.cam_display[tel_id] = visualization.CameraDisplay(self.geom[tel_id], title="CT{0}".format(tel_id))
        self.cam_display[tel_id].add_colorbar()
        self.cam_display[tel_id].pixels.set_antialiaseds(False)
        self.cam_display[tel_id].autoupdate = True
        self.cam_display[tel_id].cmap = "viridis"

        self.cam_display[tel_id].ax = axis
        self.cam_display[tel_id].image = image

        self.cam_display[tel_id].set_limits_percent(95)

        return self.cam_display[tel_id]