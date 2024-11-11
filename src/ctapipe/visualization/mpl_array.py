from itertools import cycle

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord

from ..coordinates import GroundFrame
from ..exceptions import OptionalDependencyMissing
from .mpl_camera import polar_to_cart


class ArrayDisplay:
    """
    Display a top-town view of a telescope array.

    This can be used in two ways: by default, you get a display of all
    telescopes in the subarray, colored by telescope type, however you can
    also color the telescopes by a value (like trigger pattern, or some other
    scalar per-telescope parameter). To set the color value, simply set the
    ``value`` attribute, and the fill color will be updated with the value. You
    might want to set the border color to zero to avoid confusion between the
    telescope type color and the value color (
    ``array_disp.telescope.set_linewidth(0)``)

    To display a vector field over the telescope positions, e.g. for
    reconstruction, call `set_vector_uv()` to set cartesian vectors,
    or `set_vector_rho_phi()` to set polar coordinate vectors.
    These both take an array of length N_tels, or a single value.


    Parameters
    ----------
    subarray: ctapipe.instrument.SubarrayDescription
        the array layout to display
    axes: matplotlib.axes.Axes
        matplotlib axes to plot on, or None to use current one
    title: str
        title of array plot
    tel_scale: float
        scaling between telescope mirror radius in m to displayed size
    autoupdate: bool
        redraw when the input changes
    radius: Union[float, list, None]
        set telescope radius to value, list/array of values. If None, radius
        is taken from the telescope's mirror size.
    """

    def __init__(
        self,
        subarray,
        axes=None,
        autoupdate=True,
        tel_scale=2.0,
        alpha=0.7,
        title=None,
        radius=None,
        frame=GroundFrame(),
    ):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import PatchCollection
            from matplotlib.lines import Line2D
            from matplotlib.patches import Circle
        except ModuleNotFoundError:
            raise OptionalDependencyMissing("matplotlib") from None

        self.frame = frame
        self.subarray = subarray
        self.axes = axes or plt.gca()

        # get the telescope positions. If a new frame is set, this will
        # transform to the new frame.
        self.tel_coords = subarray.tel_coords.transform_to(frame).cartesian
        self.unit = self.tel_coords.x.unit
        self.frame = frame

        # set up colors per telescope type
        tel_types = [str(tel) for tel in subarray.tels.values()]
        if radius is None:
            # set radius to the mirror radius (so big tels appear big)
            radius = [
                np.sqrt(tel.optics.mirror_area.to("m2").value) * tel_scale
                for tel in subarray.tel.values()
            ]

            self.radii = radius
        else:
            self.radii = np.ones(len(tel_types)) * radius

        if title is None:
            title = subarray.name

        # get default matplotlib color cycle (depends on the current style)
        color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        # map a color to each telescope type:
        tel_type_to_color = {}
        for tel_type in list(set(tel_types)):
            tel_type_to_color[tel_type] = next(color_cycle)

        tel_color = [tel_type_to_color[ttype] for ttype in tel_types]

        patches = []
        for x, y, r, c in zip(
            list(self.tel_coords.x.to_value("m")),
            list(self.tel_coords.y.to_value("m")),
            list(radius),
            tel_color,
        ):
            patches.append(Circle(xy=(x, y), radius=r, fill=True, color=c, alpha=alpha))

        # build the legend:
        legend_elements = []
        for ttype in list(set(tel_types)):
            color = tel_type_to_color[ttype]
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=color,
                    label=ttype,
                    markersize=10,
                    alpha=alpha,
                    linewidth=0,
                )
            )
        self.axes.legend(handles=legend_elements)

        self.add_radial_grid()

        # create the plot
        self.tel_colors = tel_color
        self.autoupdate = autoupdate
        self.telescopes = PatchCollection(patches, match_original=True)
        self.telescopes.set_linewidth(2.0)

        self.axes.add_collection(self.telescopes)
        self.axes.set_aspect(1.0)
        self.axes.set_title(title)
        xunit = self.tel_coords.x.unit.to_string("latex")
        yunit = self.tel_coords.y.unit.to_string("latex")
        xname, yname, _ = frame.get_representation_component_names().keys()
        self.axes.set_xlabel(f"{xname} [{xunit}] $\\rightarrow$")
        self.axes.set_ylabel(f"{yname} [{yunit}] $\\rightarrow$")
        self._labels = []
        self._quiver = None
        self.axes.autoscale_view()

    @property
    def values(self):
        """An array containing a value per telescope"""
        return self.telescopes.get_array()

    @values.setter
    def values(self, values):
        """set the telescope colors to display"""
        self.telescopes.set_array(np.ma.masked_invalid(values))
        self._update()

    def add_radial_grid(self, spacing=100 * u.m):
        """add some dotted rings for distance estimation. The number of rings
        is estimated automatically from the spacing and the array footprint.

        Parameters
        ----------
        spacing: Quantity
            spacing between rings

        """
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Circle

        n_circles = np.round(
            (np.sqrt(self.subarray.footprint / np.pi) / spacing).to_value(""),
            0,
        )
        circle_radii = np.arange(1, n_circles + 2, 1) * spacing.to_value(self.unit)
        circle_patches = PatchCollection(
            [
                Circle(
                    xy=(0, 0),
                    radius=r,
                    fill=False,
                    fc="none",
                    linestyle="dotted",
                    color="gray",
                    alpha=0.1,
                    lw=1,
                )
                for r in circle_radii
            ],
            color="#eeeeee",
            ls="dotted",
            fc="none",
            lw=3,
        )

        self.axes.add_collection(circle_patches)

    def set_vector_uv(self, uu, vv, c=None, **kwargs):
        """sets the vector field U,V and color for all telescopes

        Parameters
        ----------
        uu: array[n_tels]
            x-component of direction vector
        vv: array[n_tels]
            y-component of direction vector
        c: color or list of colors
            vector color for each telescope (or one for all)
        kwargs:
            extra args passed to plt.quiver(), ignored on subsequent updates
        """
        coords = self.tel_coords
        uu = u.Quantity(uu).to_value("m")
        vv = u.Quantity(vv).to_value("m")
        N = len(coords.x)

        # matplotlib since 3.2 does not allow scalars anymore
        # if quiver was already created with a certain number of arrows
        if np.isscalar(uu):
            uu = np.full(N, uu)
        if np.isscalar(vv):
            vv = np.full(N, vv)

        # passing in None for C does not work, we need to provide
        # a variadic number of arguments
        args = [coords.x.to_value("m"), coords.y.to_value("m"), uu, vv]

        if c is None:
            # use colors by telescope type if the user did not provide any
            kwargs["color"] = kwargs.get("color", self.tel_colors)
        else:
            # same as above, enable use of scalar to set all values at once
            if np.isscalar(c):
                c = np.full(N, c)
            args.append(c)

        if self._quiver is None:
            self._quiver = self.axes.quiver(
                *args, scale_units="xy", angles="xy", scale=1, **kwargs
            )
        else:
            self._quiver.set_UVC(uu, vv, c)

    def set_vector_rho_phi(self, rho, phi, c=None, **kwargs):
        """sets the vector field using R, Phi for each telescope

        Parameters
        ----------
        rho: float or array[float]
            vector magnitude for each telescope
        phi: array[Angle]
            vector angle for each telescope
        c: color or list of colors
            vector color for each telescope (or one for all)
        """
        phi = Angle(phi).rad
        uu, vv = polar_to_cart(rho, phi)
        self.set_vector_uv(uu, vv, c=c, **kwargs)

    def set_vector_hillas(
        self, hillas_dict, core_dict, length, time_gradient, angle_offset
    ):
        """
        Function to set the vector angle and length from a set of Hillas parameters.

        In order to proper use the arrow on the ground, also a dictionary with the time
        gradients for the different telescopes is needed. If the gradient is 0 the arrow
        is not plotted on the ground, whereas if the value of the gradient is negative,
        the arrow is rotated by 180 degrees (Angle(angle_offset) not added).

        This plotting behaviour has been tested with the timing_parameters function
        in ctapipe/image.

        Parameters
        ----------
        hillas_dict: Dict[int, HillasParametersContainer]
            mapping of tel_id to Hillas parameters
        core_dict : Dict[int, CoreParameters]
            mapping of tel_id to CoreParametersContainer
        length: Float
            length of the arrow (in meters)
        time_gradient: Dict[int, value of time gradient (no units)]
            dictionary for value of the time gradient for each telescope
        angle_offset: Float
            This should be the ``event.pointing.array_azimuth`` parameter

        """

        # rot_angle_ellipse is psi parameter in HillasParametersContainer
        rho = np.zeros(self.subarray.n_tels) * u.m
        rot_angle_ellipse = np.zeros(self.subarray.n_tels) * u.deg

        for tel_id, params in hillas_dict.items():
            idx = self.subarray.tel_indices[tel_id]
            rho[idx] = u.Quantity(length, u.m)

            psi = core_dict[tel_id]

            if time_gradient[tel_id] > 0.01:
                angle_offset = Angle(angle_offset)
                rot_angle_ellipse[idx] = psi + angle_offset + 180 * u.deg
            elif time_gradient[tel_id] < -0.01:
                rot_angle_ellipse[idx] = psi + angle_offset
            else:
                rho[idx] = 0 * u.m

        self.set_vector_rho_phi(rho=rho, phi=rot_angle_ellipse)

    def set_line_hillas(self, hillas_dict, core_dict, range, **kwargs):
        """
        Plot the telescope-wise direction of the shower as a segment.

        Each segment will be centered with a point on the telescope position
        and will be 2*range long.

        Parameters
        ----------
        hillas_dict: Dict[int, HillasParametersContainer]
            mapping of tel_id to Hillas parameters
        core_dict : Dict[int, CoreParameters]
            mapping of tel_id to CoreParametersContainer
        range: float
            half of the length of the segments to be plotted (in meters)
        """

        # transform to GroundFrame
        positions_in_frame = SkyCoord(self.tel_coords, frame=self.frame)
        coords = positions_in_frame.transform_to(GroundFrame())
        c = self.tel_colors

        r = np.array([-range, range])

        for tel_id, params in hillas_dict.items():
            idx = self.subarray.tel_indices[tel_id]
            x_0 = coords[idx].x.to_value(u.m)
            y_0 = coords[idx].y.to_value(u.m)

            psi = core_dict[tel_id]

            x = x_0 + np.cos(psi).value * r
            y = y_0 + np.sin(psi).value * r

            # transform back to desired frame
            line = (
                SkyCoord(x, y, z=0, unit="m", frame=GroundFrame())
                .transform_to(self.frame)
                .cartesian
            )

            self.axes.plot(line.x, line.y, color=c[idx], **kwargs)
            self.axes.scatter(
                positions_in_frame[idx].cartesian.x.to_value(u.m),
                positions_in_frame[idx].cartesian.y.to_value(u.m),
                color=c[idx],
            )

    def add_labels(self):
        px = self.tel_coords.x.to_value("m")
        py = self.tel_coords.y.to_value("m")
        for tel, x, y, r in zip(self.subarray.tels, px, py, self.radii):
            name = str(tel)
            lab = self.axes.text(
                x,
                y - r * 1.8,
                name,
                fontsize=8,
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="top",
            )
            self._labels.append(lab)

    def remove_labels(self):
        for lab in self._labels:
            lab.remove()
        self._labels = []

    def _update(self):
        """signal a redraw if necessary"""
        import matplotlib.pyplot as plt

        if self.autoupdate:
            plt.draw()

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
        """

        # use zorder to ensure the contours appear under the telescopes.
        self.axes.contour(x, y, background, zorder=0, **kwargs)
