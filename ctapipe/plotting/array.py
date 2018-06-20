from matplotlib import pyplot as plt
import numpy as np
from ctapipe.visualization.mpl_array import ArrayDisplay
import astropy.units as u
from ctapipe.coordinates import GroundFrame


class NominalPlotter:
    """
    Simple plotter for drawing camera level items in the nominal system
    """

    def __init__(self, hillas_parameters, draw_axes=False, ax=None, **kwargs):
        """

        Parameters
        ----------
        instrument: dictionary
            intrument containers for this event
        telescopes: list
            List of telescopes included
        system: Coordinate system
            Coordinate system to transform coordinates into
        """
        self.axes = ax if ax is not None else plt.gca()

        self.cen_x = [i.x.to(u.deg).value for i in hillas_parameters.values()]
        self.cen_y = [i.y.to(u.deg).value for i in hillas_parameters.values()]

        self.centre = (0, 0)
        self.array = ArrayDisplay(telx=np.asarray(self.cen_x),
                                  tely=np.asarray(self.cen_y),
                                  tel_type=np.ones(len(self.cen_y)),
                                  axes=self.axes)

        self.hillas = hillas_parameters
        scale_fac = 57.3 * 2

        self.array.overlay_moments(hillas_parameters, (self.cen_x, self.cen_y), scale_fac,
                                   cmap="Greys", alpha=0.5, **kwargs)

        if draw_axes:
            self.array.overlay_axis(hillas_parameters, (self.cen_x, self.cen_y))

    def background_contour(self, x, y, background, **kwargs):
        """
        Draw image contours in background of the display, useful when likelihood fitting

        Parameters
        ----------
        x: ndarray
            array of image X coordinates
        y: ndarray
            array of image Y coordinates
        background: ndarray
            Array of image to use in background
        kwargs: key=value
            any style keywords to pass to matplotlib

        Returns
        -------
        None
        """

        self.axes.contour(x, y, background, **kwargs)

        # Annoyingly we need to redraw everything
        self.array = ArrayDisplay(telx=np.asarray(self.cen_x),
                                  tely=np.asarray(self.cen_y),
                                  tel_type=np.ones(len(self.cen_y)),
                                  axes=self.axes)

    def draw_array(self, coord_range=((-4, 4), (-4, 4))):
        """
        Draw the array plotter (including any overlayed elements)

        Parameters
        ----------
        coord_range: tuple
            XY range in which to draw plotter

        Returns
        -------
        None
        """

        self.array.axes.set_xlim((self.centre[0] + coord_range[0][0],
                                  coord_range[0][1] + self.centre[0]))
        self.array.axes.set_ylim((self.centre[1] + coord_range[1][0],
                                  coord_range[1][1] + self.centre[1]))

        # self.axes.tight_layout()
        # self.axes.show()

    def draw_position(self, source_x, source_y, use_centre=False, **kwargs):
        """
        Draw a marker at a position in the array plotter (for marking reconstructed
        positions etc)

        Parameters
        ----------
        source_x: float
            X position of point
        source_y: float
            Y position of point
        use_centre: bool
            Centre the plotter on this position
        kwargs: key=value
            any style keywords to pass to matplotlib

        Returns
        -------
        None
        """
        self.array.add_polygon(centroid=(source_x.to(u.deg).value,
                                         source_y.to(u.deg).value),
                               radius=0.1, nsides=3, **kwargs)
        if use_centre:
            self.centre = (source_x.value, source_y.value)
