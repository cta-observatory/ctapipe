from matplotlib import pyplot as plt
import numpy as np
from ctapipe.visualization import ArrayDisplay
import astropy.units as u
from ctapipe.coordinates import TiltedGroundFrame, GroundFrame

class ArrayPlotter:
    """
    Simple plotter for drawing array level items
    """

    def __init__(self, instrument, telescopes=None, system=None):
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

        tel_type = np.asarray([type_dict[self.instrument.optical_foclen[i].to(u.m).value] for i in self.telescopes])

        if system is not None:
            ground = GroundFrame(x=np.asarray(tel_x)*u.m, y=np.asarray(tel_y)*u.m, z=np.asarray(tel_z)*u.m)
            new_sys = ground.transform_to(system)
            self.tel_x = new_sys.x
            self.tel_y = new_sys.y
        else:
            self.tel_x = tel_x*u.m
            self.tel_y = tel_y*u.m

        self.centre = (0,0)
        self.array = ArrayDisplay(telx=np.asarray(self.tel_x), tely=np.asarray(self.tel_y), tel_type=tel_type)

        self.hillas = None

    def overlay_hillas(self, hillas, scale_fac=20000, **kwargs):
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
                                       cmap="Greys", alpha=0.5, **kwargs)
        else:
            self.array.overlay_moments(hillas, (tel_x, tel_y), scale_fac, cmap="viridis", **kwargs)

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

        plt.contour(x, y, background, **kwargs)

        # Annoyingly we need to redraw everything
        self.array = ArrayDisplay(telx=np.asarray(self.tel_x), tely=np.asarray(self.tel_y))

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

        plt.tight_layout()
        plt.show()

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
        if self.system is not None:
            ground = GroundFrame(x=np.asarray(core_x)*u.m, y=np.asarray(core_y)*u.m, z=np.asarray(0)*u.m)
            new_sys = ground.transform_to(self.system)

        self.array.add_polygon(centroid=(new_sys.x.value,new_sys.y.value), radius=20, nsides=3, **kwargs)
        if use_centre:
            self.centre = (new_sys.x.value,new_sys.y.value)
