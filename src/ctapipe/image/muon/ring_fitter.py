import numpy as np
import traitlets as traits

from ctapipe.containers import MuonRingContainer
from ctapipe.core import Component

from .fitting import kundu_chaudhuri_circle_fit, taubin_circle_fit

# the fit methods do not expose the same interface, so we
# force the same interface onto them, here.
# we also modify their names slightly, since the names are
# exposed to the user via the string traitlet `fit_method`


def kundu_chaudhuri(fov_lon, fov_lat, weights, mask):
    """kundu_chaudhuri_circle_fit with fov_lon, fov_lat, weights, mask interface"""
    return kundu_chaudhuri_circle_fit(fov_lon[mask], fov_lat[mask], weights[mask])


def taubin(fov_lon, fov_lat, weights, mask):
    """taubin_circle_fit with fov_lon, fov_lat, weights, mask interface"""
    return taubin_circle_fit(fov_lon, fov_lat, mask)


def kundu_chaudhuri_taubin(fov_lon, fov_lat, weights, mask):
    """taubin_circle_fit with fov_lon, fov_lat, weights, mask interface
    with initial parameters provided by kundu_chaudhuri"""
    taubin_r_initial, xc_initial, yc_initial, _, _, _ = kundu_chaudhuri_circle_fit(
        fov_lon[mask], fov_lat[mask], weights[mask], nan_errors_flag=True
    )
    return taubin_circle_fit(
        fov_lon,
        fov_lat,
        mask,
        weights,
        taubin_r_initial,
        xc_initial,
        yc_initial,
    )


FIT_METHOD_BY_NAME = {
    m.__name__: m for m in [kundu_chaudhuri, taubin, kundu_chaudhuri_taubin]
}

__all__ = ["MuonRingFitter"]


class MuonRingFitter(Component):
    """Different ring fit algorithms for muon rings"""

    fit_method = traits.CaselessStrEnum(
        list(FIT_METHOD_BY_NAME.keys()),
        default_value="kundu_chaudhuri_taubin",
    ).tag(config=True)

    def __call__(self, geom, image, clean_mask):
        """
        Perform a circle fit to ``image`` with the chosen ``fit_method``.

        Parameters
        ----------
        geom: CameraGeometry
            Defines the pixel coordinates.
            Must be in the `ctapipe.coordinates.TelescopeFrame`
        image: np.ndarray[np.float32]
            Image intensity values
        clean_mask: np.array([],dtype=bool)
            Boolean mask of clean pixels, where True means the pixel contains signal.
            This can be generated using a `ctapipe.image.ImageCleaner`.

        Returns
        -------
        MuonRingContainer:
            Results of the ring fit.
        """

        fit_function = FIT_METHOD_BY_NAME[self.fit_method]
        (
            radius,
            center_fov_lon,
            center_fov_lat,
            radius_err,
            center_fov_lon_err,
            center_fov_lat_err,
        ) = fit_function(geom.pix_x, geom.pix_y, image, clean_mask)

        return MuonRingContainer(
            center_fov_lon=center_fov_lon,
            center_fov_lat=center_fov_lat,
            radius=radius,
            center_fov_lon_err=center_fov_lon_err,
            center_fov_lat_err=center_fov_lat_err,
            radius_err=radius_err,
            center_phi=np.arctan2(center_fov_lat, center_fov_lon),
            center_distance=np.sqrt(center_fov_lon**2 + center_fov_lat**2),
        )
