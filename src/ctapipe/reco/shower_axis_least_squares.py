from functools import partial

import astropy.units as u
import numba
import numpy as np
from astropy.coordinates import AltAz, Latitude, Longitude
from iminuit import Minuit

from ..containers import ReconstructedGeometryContainer
from ..coordinates import TelescopeFrame, get_point_on_shower_axis
from ..core import QualityQuery
from ..core.traits import Unicode
from .reconstructor import Reconstructor

_INVALID = ReconstructedGeometryContainer(
    telescopes=[], prefix="ShowerAxisLeastSquares"
)
_SHOWER_POINT_DISTANCE = u.Quantity(10, u.km)


class ShowerAxisLeastSquares(Reconstructor):
    """Fit direction and impact using least squares of the pixels to the shower axis.

    The total squared distance of all pixel that survived cleaning to the shower axis
    is numerically minimized.
    The shower axis is defined by the primary origin and a second point on the shower
    axis further down into the atmosphere (`get_point_on_shower_axis`).

    These two point are transformed into the `TelescopeFrame`, which is used
    for the distance computation from pixel coordinates to a line defined by these
    two points.

    A `QualityQuery` is used to decide which images are used.
    The `~ctapipe.containers.DL1CameraContainer` is passed as ``dl1`` into the query
    evaluation. E.g. to include only images with at least 10 pixels and 50 photons, use

    .. code-block::

        config = traitlets.Config()
        config.ShowerAxisLeastSquares.QualityQuery.quality_criteria = [
            ("n_pixels >= 10", "dl1.parameters.morphology.n_pixels >= 10"),
            ("intensity >= 50", "dl1.parameters.hillas.intensity >= 50"),
        ]


    This is algorithm 6 of :cite:p:`hofmann-1999-comparison`.
    """

    initial_guess = Unicode(
        None,
        allow_none=True,
        help="Prefix of initial guess, e.g. 'HillasReconstructor'",
    ).tag(config=True)

    def __init__(self, subarray, **kwargs):
        super().__init__(subarray=subarray, **kwargs)
        self.quality_query = QualityQuery(parent=self)
        self._geometries = {
            tel_id: tel.camera.geometry.transform_to(TelescopeFrame())
            for tel_id, tel in self.subarray.tel.items()
        }

    def __call__(self, event):
        dl1_tel = {
            tel_id: dl1
            for tel_id, dl1 in event.dl1.tel.items()
            if all(self.quality_query(dl1=dl1))
        }

        # need at least two telescopes, otherwise impact is degenerate
        if len(dl1_tel) < 2:
            event.dl2.stereo.geometry[self.__class__.__name__] = _INVALID
            return

        idx = self.subarray.tel_ids_to_indices(list(dl1_tel.keys()))
        tel_positions = self.subarray.tel_coords[idx]

        pointing = AltAz(
            alt=[event.pointing.tel[tel_id].altitude for tel_id in dl1_tel],
            az=[event.pointing.tel[tel_id].azimuth for tel_id in dl1_tel],
        )
        frame = TelescopeFrame(telescope_pointing=pointing)

        images = [dl1.image[dl1.image_mask] for dl1 in dl1_tel.values()]
        pix_fov_lon = [
            self._geometries[tel_id].pix_x.to_value(u.deg)[dl1.image_mask]
            for tel_id, dl1 in dl1_tel.items()
        ]
        pix_fov_lat = [
            self._geometries[tel_id].pix_y.to_value(u.deg)[dl1.image_mask]
            for tel_id, dl1 in dl1_tel.items()
        ]

        geometry = event.dl2.stereo.geometry
        if (
            self.initial_guess is not None
            and (dl2 := geometry[self.initial_guess]).is_valid
        ):
            core_x = dl2.core_x.to_value(u.m)
            core_y = dl2.core_y.to_value(u.m)
            alt = dl2.alt.to_value(u.rad)
            az = dl2.az.to_value(u.rad)
        else:
            core_x = 0.0
            core_y = 0.0
            # average tel pointing as initial guess
            alt = pointing.alt.rad.mean()
            az = pointing.az.rad.mean()

        cost = partial(
            self._cost,
            pix_fov_lon=pix_fov_lon,
            pix_fov_lat=pix_fov_lat,
            images=images,
            telescope_position=tel_positions,
            frame=frame,
        )
        m = Minuit(
            cost,
            core_x=core_x,
            core_y=core_y,
            alt=alt,
            az=az,
            name=("core_x", "core_y", "alt", "az"),
        )
        m.errors["core_x"] = 10
        m.errors["core_y"] = 10
        m.limits["alt"] = (0, np.pi / 2)
        m.limits["az"] = (-np.pi, np.pi)
        m.migrad()

        prefix = self.__class__.__name__
        geometry[prefix] = ReconstructedGeometryContainer(
            prefix=prefix,
            telescopes=list(dl1_tel.keys()),
            alt=u.Quantity(m.values["alt"], u.rad).to(u.deg),
            az=u.Quantity(m.values["az"], u.rad).to(u.deg),
            core_x=u.Quantity(m.values["core_x"], u.m),
            core_y=u.Quantity(m.values["core_y"], u.m),
            is_valid=True,
            alt_uncert=u.Quantity(m.errors["alt"], u.rad).to(u.deg),
            az_uncert=u.Quantity(m.errors["az"], u.rad).to(u.deg),
            core_uncert_x=u.Quantity(m.errors["core_x"], u.m),
            core_uncert_y=u.Quantity(m.errors["core_x"], u.m),
            average_intensity=np.mean(
                [dl1.parameters.hillas.intensity for dl1 in dl1_tel.values()]
            ),
        )
        self.last_minuit = m

    @staticmethod
    def _cost(
        core_x,
        core_y,
        alt,
        az,
        pix_fov_lon,
        pix_fov_lat,
        images,
        telescope_position,
        frame,
    ):
        # get point on shower axis to define lines in telescope frame
        alt = Latitude(alt, u.rad)
        az = Longitude(az, u.rad)

        axis_point = get_point_on_shower_axis(
            core_x=u.Quantity(core_x, u.m),
            core_y=u.Quantity(core_y, u.m),
            alt=alt,
            az=az,
            telescope_position=telescope_position,
            distance=_SHOWER_POINT_DISTANCE,
        ).transform_to(frame)

        source = AltAz(alt=alt, az=az).transform_to(frame)
        p1_fov_lon = source.fov_lon.to_value(u.deg)
        p1_fov_lat = source.fov_lat.to_value(u.deg)
        p2_fov_lon = axis_point.fov_lon.to_value(u.deg)
        p2_fov_lat = axis_point.fov_lat.to_value(u.deg)

        loss = 0.0
        for i in range(len(pix_fov_lon)):
            dist = _point_to_line_distance(
                pix_fov_lon[i],
                pix_fov_lat[i],
                p1_fov_lon[i],
                p1_fov_lat[i],
                p2_fov_lon[i],
                p2_fov_lat[i],
            )
            loss += np.average(dist**2, weights=images[i])
        return loss


@numba.vectorize(cache=True)
def _point_to_line_distance(x0, y0, x1, y1, x2, y2):
    """
    Distance from a point to a line defined by two points.

    See https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
    """
    num = np.abs((x2 - x1) * (y0 - y1) - (x0 - x1) * (y2 - y1))
    denom = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return num / denom
