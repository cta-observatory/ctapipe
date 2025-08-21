"""
Class description to be added.

"""

import astropy.units as u
import numpy as np

from ...containers import MuonEfficiencyContainer
from ...coordinates import TelescopeFrame
from ...core import TelescopeComponent
from ...core.traits import FloatTelescopeParameter
from ...exceptions import OptionalDependencyMissing
from ...utils.quantities import all_to_value

try:
    from iminuit import Minuit
except ModuleNotFoundError:
    Minuit = None

__all__ = [
    "MuonImpactpointIntensityFitter",
]


def chord_length_loss_function(radius, rho, phi):
    """
    Function for integrating the length of a chord across a circle (effective chord length).

    A circular mirror is used for signal, and a circular camera is used for shadowing.

    Parameters
    ----------
    radius: float or ndarray
        radius of circle
    rho: float or ndarray
        fractional distance of impact point from circle center
    phi: float or ndarray in radians
        rotation angles to calculate length

    Returns
    -------
    float or ndarray:
        effective chord length

    References
    ----------
    See :cite:p:`vacanti19941`.
    Equation 6: for effective chord length calculations inside/outside the ring.
    Equation 7: for filtering out non-physical solutions.


    """
    discriminant_norm = 1 - (rho**2 * np.sin(phi) ** 2)
    valid = discriminant_norm >= 0

    if not valid:
        return 0

    if rho <= 1.0:
        # muon has hit the mirror
        effective_chord_length = radius * (
            np.sqrt(discriminant_norm) + rho * np.cos(phi)
        )
    else:
        # muon did not hit the mirror
        # Filtering out non-physical solutions for phi
        if np.abs(phi) < np.arcsin(1.0 / rho):
            effective_chord_length = 2 * radius * np.sqrt(discriminant_norm)
        else:
            return 0

    return effective_chord_length


def muon_ring_phi_distribution_fit(
    x, y, mask, image, amplitude_initial=None, rho_initial=None, phi0_initial=None
):
    """
    muon_ring_phi_distribution_fit.


    Parameters
    ----------
    x : array-like or astropy.units.Quantity
        x-coordinates of the points.
    y : array-like or astropy.units.Quantity
        y-coordinates of the points.
    mask : array-like of bool
        Boolean mask indicating which pixels survive the cleaning process.
    weights : array-like of float
        Weights for the points. If not provided, all points are assigned equal weights (1).
    amplitude_initial : unitless float, optional
    rho_initial : astropy.units.Quantity, optional
    phi0_initial : astropy.units.Quantity, optional

    Returns
    -------
    amplitude : astropy.units.Quantity
        Fitted amplitude.
    rho : astropy.units.Quantity
        Fitted rho.
    phi0 : astropy.units.Quantity
        Fitted phi0.
    amplitude_err : astropy.units.Quantity
        Fitted radius of the circle error.
    rho_err : astropy.units.Quantity
        Fitted x-coordinate of the circle center error.
    phi0_err : astropy.units.Quantity
        Fitted y-coordinate of the circle center error.

    Raises
    ------
    OptionalDependencyMissing
        If the iminuit package is not installed.

    Notes
    -----
    The Taubin circle fit minimizes a specific loss function that balances the
    squared residuals of the points from the circle with the weights. This method
    is particularly useful for fitting circles to noisy data.

    References
    ----------
    - Barcelona_Muons_TPA_final.pdf (slide 6)
    """

    if Minuit is None:
        raise OptionalDependencyMissing("iminuit")

    original_unit = x.unit
    x, y = all_to_value(x, y, unit=original_unit)

    # x_masked = x[mask]
    # y_masked = y[mask]
    # image_masked = image[mask]

    # minimization method
    # fit = Minuit(
    #    make_loss_function(x_masked, y_masked, weights_masked),
    #    xc=xc_initial.to_value(original_unit),
    #    yc=yc_initial.to_value(original_unit),
    #    r=r_initial.to_value(original_unit),
    # )
    # fit.errordef = Minuit.LEAST_SQUARES

    # set initial parameters uncertainty to a big value
    # taubin_error = max_fov * 0.1
    # fit.errors["xc"] = taubin_error
    # fit.errors["yc"] = taubin_error
    # fit.errors["r"] = taubin_error

    # set wide rage for the minimisation
    # fit.limits["xc"] = (-max_fov, max_fov)
    # fit.limits["yc"] = (-max_fov, max_fov)
    # fit.limits["r"] = (0, max_fov)

    # fit.migrad()

    # radius = Quantity(fit.values["r"], original_unit)
    # center_x = Quantity(fit.values["xc"], original_unit)
    # center_y = Quantity(fit.values["yc"], original_unit)
    # radius_err = Quantity(fit.errors["r"], original_unit)
    # center_x_err = Quantity(fit.errors["xc"], original_unit)
    # center_y_err = Quantity(fit.errors["yc"], original_unit)

    amplitude = np.nan
    rho = np.nan * original_unit
    phi0 = np.nan * u.deg
    amplitude_err = np.nan
    rho_err = np.nan * original_unit
    phi0_err = np.nan * u.deg

    return amplitude, rho, phi0, amplitude_err, rho_err, phi0_err


class MuonImpactpointIntensityFitter(TelescopeComponent):
    """
    Fit muon ring images with a theoretical model to estimate optical efficiency.

    """

    min_lambda_m = FloatTelescopeParameter(
        help="Minimum wavelength for Cherenkov light in m", default_value=300e-9
    ).tag(config=True)

    max_lambda_m = FloatTelescopeParameter(
        help="Minimum wavelength for Cherenkov light in m", default_value=600e-9
    ).tag(config=True)

    hole_radius_m = FloatTelescopeParameter(
        help="The radius of the hole in the center of the primary mirror dish in meters."
        "The hole is not circular in shape; however, it can be well approximated as a circle with the same area."
        "It is defined with the flat-to-flat distance (LST: 1.51 m, MST: 1.2 m, SST: 0.78 m)."
        "We approximate the hexagonal hole with a circle that has the same surface area.",
        default_value=[
            ("type", "LST_*", 0.74),
            ("type", "MST_*", 0.59),
            ("type", "SST_1M_*", 0.38),
        ],
    ).tag(config=True)

    def __init__(self, subarray, **kwargs):
        if Minuit is None:
            raise OptionalDependencyMissing("iminuit") from None

        super().__init__(subarray=subarray, **kwargs)
        self._geometries_tel_frame = {
            tel_id: tel.camera.geometry.transform_to(TelescopeFrame())
            for tel_id, tel in subarray.tel.items()
        }

    def __call__(self, tel_id, center_x, center_y, radius, image, pedestal, mask=None):
        """

        Parameters
        ----------
        tel_id: int
            the telescope id
        center_x: Angle quantity
            Initial guess for muon ring center in telescope frame
        center_y: Angle quantity
            Initial guess for muon ring center in telescope frame
        radius: Angle quantity
            Initial guess for muon ring radius in telescope frame
        image: ndarray
            Amplitude of image pixels
        pedestal: ndarray
            Pedestal standard deviation in each pixel
        mask: ndarray
            mask marking the pixels to be used in the likelihood fit

        Returns
        -------
        MuonEfficiencyContainer
        """
        telescope = self.subarray.tel[tel_id]
        if telescope.optics.n_mirrors != 1:
            raise NotImplementedError(
                "Currently only single mirror telescopes"
                f" are supported in {self.__class__.__name__}"
            )

        geometry = telescope.camera.geometry.transform_to(TelescopeFrame())

        # results_phi_dist = muon_ring_phi_distribution_fit(
        muon_ring_phi_distribution_fit(
            geometry.pix_x,
            geometry.pix_y,
            mask,
            image,
            amplitude_initial=None,
            rho_initial=None,
            phi0_initial=None,
        )

        return MuonEfficiencyContainer()
