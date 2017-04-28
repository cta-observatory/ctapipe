from matplotlib import pyplot as plt
import numpy as np
from ctapipe.visualization import ArrayDisplay
import astropy.units as u
from ctapipe.coordinates import TiltedGroundFrame, GroundFrame


class ArrayPlotter:
    """
    Simple plotter for drawing array level items
    """

    def __init__(self, instrument, telescopes=None, system=None, ax=None):
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

        self.instrument = instrument
        self.system = system
        if telescopes is None:
            self.telescopes = instrument.telescope_ids
        else:
            self.telescopes = telescopes
        type_dict = {28.0: 1, 16.0: 2,
                     2.1500000953674316: 3,
                     2.2829999923706055: 4,
                     5.599999904632568: 5}

        tel_x = [self.instrument.tel_pos[i][0].to(u.m).value for i in self.telescopes]
        tel_y = [self.instrument.tel_pos[i][1].to(u.m).value for i in self.telescopes]
        tel_z = [self.instrument.tel_pos[i][2].to(u.m).value for i in self.telescopes]

        self.axes = ax if ax is not None else plt.gca()

        tel_type = np.asarray([type_dict[self.instrument.optical_foclen[i].to(u.m).value] for i in self.telescopes])
        self.tel_type = tel_type

        if system is not None:
            ground = GroundFrame(x=np.asarray(tel_x)*u.m, y=np.asarray(tel_y)*u.m, z=np.asarray(tel_z)*u.m)
            new_sys = ground.transform_to(system)
            self.tel_x = new_sys.x
            self.tel_y = new_sys.y
        else:
            self.tel_x = tel_x*u.m
            self.tel_y = tel_y*u.m

        self.centre = (0,0)
        self.array = ArrayDisplay(telx=np.asarray(self.tel_x), tely=np.asarray(self.tel_y), tel_type=tel_type,
                                  axes=self.axes)

        self.hillas = None

    def overlay_hillas(self, hillas, scale_fac=20000, draw_axes=False, **kwargs):
        """
        Overlay hillas parameters on top of the array map

        Parameters
        ----------
        hillas: dictionary
            Hillas moments objects to overlay
        scale_fac: float
            Scaling factor to array to hillas width and length when drawing
        kwargs: key=value
            any style keywords to pass to matplotlib

        Returns
        -------
        None
        """
        tel_x = [self.instrument.tel_pos[i][0].to(u.m).value for i in hillas]
        tel_y = [self.instrument.tel_pos[i][1].to(u.m).value for i in hillas]
        tel_z = [self.instrument.tel_pos[i][2].to(u.m).value for i in hillas]
        if self.system is not None:
            ground = GroundFrame(x=np.asarray(tel_x)*u.m, y=np.asarray(tel_y)*u.m, z=np.asarray(tel_z)*u.m)
            new_sys = ground.transform_to(self.system)
            self.array.overlay_moments(hillas, (new_sys.x, new_sys.y), scale_fac,
                                       cmap="Viridis", alpha=0.5, **kwargs)
            if draw_axes:
                self.array.overlay_axis(hillas, (new_sys.x, new_sys.y))
        else:
            self.array.overlay_moments(hillas, (tel_x, tel_y), scale_fac, alpha=0.5, cmap="viridis", **kwargs)

        self.hillas = hillas

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
        self.array = ArrayDisplay(telx=np.asarray(self.tel_x), tely=np.asarray(self.tel_y), tel_type=self.tel_type)

        if self.hillas is not None:
            self.overlay_hillas(self.hillas)

    def draw_array(self, range=((-2000,2000),(-2000,2000))):
        """
        Draw the array plotter (including any overlayed elements)

        Parameters
        ----------
        range: tuple
            XY range in which to draw plotter

        Returns
        -------
        None
        """

        self.array.axes.set_xlim((self.centre[0]+range[0][0], range[0][1]+self.centre[0]))
        self.array.axes.set_ylim((self.centre[1]+range[1][0], range[1][1]+self.centre[1]))

        #self.axes.tight_layout()
        #self.axes.show()

    def draw_position(self, core_x, core_y, use_centre=False, **kwargs):
        """
        Draw a marker at a position in the array plotter (for marking reconstructed
        positions etc)

        Parameters
        ----------
        core_x: float
            X position of point
        core_y: float
            Y position of point
        use_centre: bool
            Centre the plotter on this position
        kwargs: key=value
            any style keywords to pass to matplotlib

        Returns
        -------
        None
        """
        ground = GroundFrame(x=np.asarray(core_x) * u.m, y=np.asarray(core_y) * u.m, z=np.asarray(0) * u.m)

        if self.system is not None:
            new_sys = ground.transform_to(self.system)
        else:
            new_sys = ground

        self.array.add_polygon(centroid=(new_sys.x.value,new_sys.y.value), radius=10, nsides=3, **kwargs)
        if use_centre:
            self.centre = (new_sys.x.value,new_sys.y.value)


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

        self.cen_x = [i.cen_x.to(u.deg).value for i in hillas_parameters.values()]
        self.cen_y = [i.cen_y.to(u.deg).value for i in hillas_parameters.values()]

        self.centre = (0,0)
        self.array = ArrayDisplay(telx=np.asarray(self.cen_x), tely=np.asarray(self.cen_y),
                                  tel_type=np.ones(len(self.cen_y)), axes=self.axes)

        self.hillas = hillas_parameters
        scale_fac = 57.3 * 5
        self.array.overlay_moments(hillas_parameters, (self.cen_x, self.cen_y), scale_fac,
                                   cmap="viridis", alpha=0.5, **kwargs)

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
        self.array = ArrayDisplay(telx=np.asarray(self.tel_x), tely=np.asarray(self.tel_y))

    def draw_array(self, range=((-4,4),(-4,4))):
        """
        Draw the array plotter (including any overlayed elements)

        Parameters
        ----------
        range: tuple
            XY range in which to draw plotter

        Returns
        -------
        None
        """

        self.array.axes.set_xlim((self.centre[0]+range[0][0], range[0][1]+self.centre[0]))
        self.array.axes.set_ylim((self.centre[1]+range[1][0], range[1][1]+self.centre[1]))

        #self.axes.tight_layout()
        #self.axes.show()

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
        self.array.add_polygon(centroid=(source_x.to(u.deg).value,source_y.to(u.deg).value), radius=0.1, nsides=3, **kwargs)
        if use_centre:
            self.centre = (source_x.value,source_y.value)
