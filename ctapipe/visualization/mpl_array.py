from itertools import cycle

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from ctapipe.coordinates import GroundFrame
from ctapipe.visualization.mpl_camera import polar_to_cart


class ArrayDisplay:
    """
    Display a top-town view of a telescope array.

    This can be used in two ways: by default, you get a display of all
    telescopes in the subarray, colored by telescope type, however you can
    also color the telescopes by a value (like trigger pattern, or some other
    scalar per-telescope parameter). To set the color value, simply set the
    `value` attribute, and the fill color will be updated with the value. You
    might want to set the border color to zero to avoid confusion between the
    telescope type color and the value color (
    `array_disp.telescope.set_linewidth(0)`)

    To display a vector field over the telescope positions, e.g. for
    reconstruction, call `set_uv()` to set cartesian vectors, or `set_r_phi()`
    to set polar coordinate vectors.  These both take an array of length
    N_tels, or a single value.


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

    def __init__(self, subarray, axes=None, autoupdate=True,
                 tel_scale=2.0, alpha=0.7, title=None,
                 radius=None, frame=GroundFrame()):

        self.frame = frame
        self.subarray = subarray

        # get the telescope positions. If a new frame is set, this will
        # transform to the new frame.
        self.tel_coords = subarray.tel_coords.transform_to(frame)

        # set up colors per telescope type
        tel_types = [str(tel) for tel in subarray.tels.values()]
        if radius is None:
            # set radius to the mirror radius (so big tels appear big)
            radius = [np.sqrt(tel.optics.mirror_area.to("m2").value) * tel_scale
                      for tel in subarray.tel.values()]

        if title is None:
            title = subarray.name

        # get default matplotlib color cycle (depends on the current style)
        color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        # map a color to each telescope type:
        tel_type_to_color = {}
        for tel_type in list(set(tel_types)):
            tel_type_to_color[tel_type] = next(color_cycle)

        tel_color = [tel_type_to_color[ttype] for ttype in tel_types]

        patches = []
        for x, y, r, c in zip(list(self.tel_coords.x.value),
                              list(self.tel_coords.y.value),
                              list(radius),
                              tel_color):
            patches.append(
                Circle(
                    xy=(x, y),
                    radius=r,
                    fill=True,
                    color=c,
                    alpha=alpha,
                )
            )

        # build the legend:
        legend_elements = []
        for ttype in list(set(tel_types)):
            color = tel_type_to_color[ttype]
            legend_elements.append(
                Line2D([0], [0], marker='o', color=color,
                       label=ttype, markersize=10, alpha=alpha,
                       linewidth=0)
            )
        plt.legend(handles=legend_elements)

        self.tel_colors = tel_color
        self.autoupdate = autoupdate
        self.telescopes = PatchCollection(patches, match_original=True)
        self.telescopes.set_linewidth(2.0)

        self.axes = axes or plt.gca()
        self.axes.add_collection(self.telescopes)
        self.axes.set_aspect(1.0)
        self.axes.set_title(title)
        self._labels = []
        self._quiver = None
        self.axes.autoscale_view()

    @property
    def values(self):
        """An array containing a value per telescope"""
        return self.telescopes.get_array()

    @values.setter
    def values(self, values):
        """ set the telescope colors to display  """
        self.telescopes.set_array(np.ma.masked_invalid(values))
        self._update()

    def set_vector_uv(self, u, v, c=None, **kwargs):
        """ sets the vector field U,V and color for all telescopes

        Parameters
        ----------
        u: array[num_tels]
            x-component of direction vector
        v: array[num_tels]
            y-component of direction vector
        c: color or list of colors
            vector color for each telescope (or one for all)
        kwargs:
            extra args passed to plt.quiver(), ignored on subsequent updates
        """
        if c is None:
            c = self.tel_colors

        if self._quiver is None:
            coords = self.tel_coords
            self._quiver = self.axes.quiver(
                coords.x, coords.y,
                u, v,
                color=c,
                scale_units='xy',
                angles='xy',
                scale=1,
                **kwargs
            )
        else:
            self._quiver.set_UVC(u, v)

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
        u, v = polar_to_cart(rho, phi)
        self.set_vector_uv(u, v, c=c, **kwargs)

    def set_vector_hillas(self, hillas_dict, length, time_gradient, angle_offset):
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
        length: Float
            length of the arrow (in meters)
        time_gradient: Dict[int, value of time gradient (no units)]
            dictionary for value of the time gradient for each telescope
        angle_offset: Float
            This should be the event.mcheader.run_array_direction[0] parameter

        """

        # rot_angle_ellipse is psi parameter in HillasParametersContainer
        rho = np.zeros(self.subarray.num_tels) * u.m
        rot_angle_ellipse = np.zeros(self.subarray.num_tels) * u.deg

        for tel_id, params in hillas_dict.items():
            idx = self.subarray.tel_indices[tel_id]
            rho[idx] = length * u.m

            if time_gradient[tel_id] > 0.01:
                params.psi = Angle(params.psi)
                angle_offset = Angle(angle_offset)
                rot_angle_ellipse[idx] = params.psi + angle_offset + 180 * u.deg
            elif time_gradient[tel_id] < -0.01:
                rot_angle_ellipse[idx] = params.psi + angle_offset
            else:
                rho[idx] = 0 * u.m

        self.set_vector_rho_phi(rho=rho, phi=rot_angle_ellipse)

    def set_line_hillas(self, hillas_dict, range, **kwargs):
        """
        Function to plot a segment of length 2*range for each telescope from a set of Hillas parameters.
        The segment is centered on the telescope position.
        A point is added at each telescope position for better visualization.

        Parameters
        ----------
        hillas_dict: Dict[int, HillasParametersContainer]
            mapping of tel_id to Hillas parameters
        range: float
            half of the length of the segments to be plotted (in meters)
        """

        coords = self.tel_coords
        c = self.tel_colors

        r = np.array([-range, range])
        for tel_id, params in hillas_dict.items():
            idx = self.subarray.tel_indices[tel_id]
            x_0 = coords[idx].x.to_value(u.m)
            y_0 = coords[idx].y.to_value(u.m)
            x = x_0 + np.cos(params.psi) * r
            y = y_0 + np.sin(params.psi) * r
            self.axes.plot(x, y, color=c[idx], **kwargs)
            self.axes.scatter(x_0, y_0, color=c[idx])

    def add_labels(self):
        px = self.tel_coords.x.value
        py = self.tel_coords.y.value
        for tel, x, y in zip(self.subarray.tels, px, py):
            name = str(tel)
            lab = self.axes.text(x, y, name, fontsize=8, clip_on=True)
            self._labels.append(lab)

    def remove_labels(self):
        for lab in self._labels:
            lab.remove()
        self._labels = []

    def _update(self):
        """ signal a redraw if necessary """
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
