import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from numpy import ceil, sqrt

from ctapipe import visualization
from ctapipe.core import Component
from ctapipe.core.traits import Bool
from ctapipe.plotting.array import NominalPlotter


class EventViewer(Component):
    """
    Event viewer class built on top of the other plotters to allow a single
    view of both the camera images and the projected Hillas parameters for a
    single event. Can be further modified to show the reconstructed shower
    direction and core position if needed. Plus further info
    """
    test = Bool(True, help='').tag(config=True)

    def __init__(self, draw_hillas_planes=False):
        """

        Parameters
        ----------
        draw_hillas_planes: bool
            Determines whether a projection of the Hillas parameters in the nominal and tilted systems should be drawn
        """
        self.array_view = None
        self.nominal_view = None

        self.cam_display = dict()
        self.draw_hillas_planes = draw_hillas_planes

    def draw_source(self, source):
        """
        Loop over events and draw each

        Parameters
        ----------
        source: ctapipe source object

        Returns
        -------
            None
        """
        for event in source:
            self.draw_event(event)

        return

    def draw_event(self, event, hillas_parameters=None):
        """
        Draw display for a given event

        Parameters
        ----------
        event: ctapipe event object
        hillas_parameters: dict
            Dictionary of Hillas parameters (in nominal system)

        Returns
        -------
            None
        """
        tel_list = event.r0.tels_with_data
        images = event.dl1

        # First close any plots that already exist
        plt.close()
        ntels = len(tel_list)

        plt.figure(figsize=(20, 20 * 0.66))

        # If we want to draw the Hillas parameters in different planes we need to split our figure
        if self.draw_hillas_planes:
            y_axis_split = 2
        else:
            y_axis_split = 1

        outer_grid = gridspec.GridSpec(1, y_axis_split,
                                       width_ratios=[y_axis_split, 1])
        # Create a square grid for camera images
        nn = int(ceil(sqrt(ntels)))
        nx = nn
        ny = nn

        while nx * ny >= ntels:
            ny -= 1
        ny += 1
        while nx * ny >= ntels:
            nx -= 1
        nx += 1

        camera_grid = gridspec.GridSpecFromSubplotSpec(ny, nx,
                                                       subplot_spec=outer_grid[
                                                           0])

        # Loop over camera images of all telescopes and create plots
        for ii, tel_id in zip(range(ntels), tel_list):
            ax = plt.subplot(camera_grid[ii])
            geom = event.inst.subarray.tel[tel_id].camera
            self.get_camera_view(tel_id,
                                 image=images.tel[tel_id].image,
                                 axis=ax,
                                 geom=geom)

        # If we want to draw the Hillas parameters in different planes we need to make a couple more viewers
        if self.draw_hillas_planes:
            # Split the second sub figure into two further figures
            reco_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=
            outer_grid[1])
            # Create plot of telescope positions at ground level

            # Draw MC position (this should change later)
            array.draw_position(event.mc.core_x, event.mc.core_y,
                                use_centre=True)
            array.draw_array(((-300, 300), (-300, 300)))

            # If we have valid Hillas parameters we should draw them in the Nominal system
            if hillas_parameters is not None:
                array.overlay_hillas(hillas_parameters, draw_axes=True)

                nominal = NominalPlotter(hillas_parameters=hillas_parameters,
                                         draw_axes=True,
                                         ax=plt.subplot(reco_grid[1]))
                nominal.draw_array()
        self.nominal_view = nominal
        self.array_view = array

        plt.show()

        return

    def get_camera_view(self, tel_id, image, axis, geom):
        """
        Create camera viewer for a given camera image

        Parameters
        ----------
        tel_id: int
            Telescope ID number
        image: ndarray
            Array of calibrated pixel intensities
        axis: matplotlib axis
            Axis on which to draw plot
        geom: ctapipe.instrument.CameraGeometry
            Camera geometry for this tel_id

        Returns
        -------
            Camera display
        """
        # if tel_id not in self.cam_display:
        # Argh this is annoying, for some reason we cannot cahe the displays
        self.cam_display[tel_id] = visualization.CameraDisplay(geom, title="CT{"
                                                                           "0}".format(
            tel_id))
        self.cam_display[tel_id].add_colorbar()
        self.cam_display[tel_id].pixels.set_antialiaseds(False)
        self.cam_display[tel_id].autoupdate = True
        self.cam_display[tel_id].cmap = "viridis"

        self.cam_display[tel_id].ax = axis
        self.cam_display[tel_id].image = image

        self.cam_display[tel_id].set_limits_percent(95)

        return self.cam_display[tel_id]
