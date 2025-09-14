"""
Class description to be added.

"""


import astropy.units as u
import numpy as np

# dev to be removed
import pandas as pd
from astropy.units import Quantity
from scipy.ndimage import correlate1d

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


def chord_length(radius, rho, phi0, phi):
    """
    Function for integrating the length of a chord across a circle (effective chord length).

    A circular mirror is used for signal, and a circular camera is used for shadowing.

    Parameters
    ----------
    radius: float
        radius of circle
    rho: float
        distance of impact point from circle center
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

    if phi0 == 0:
        if radius <= 0:
            return np.zero(len(phi))

        phi_modulo = ((phi + np.pi) % (2 * np.pi) - np.pi) * np.where(phi < -1, -1, 1)

        rho_R = rho / radius
        discriminant_norm = 1 - (rho_R**2 * np.sin(phi_modulo) ** 2)
        discriminant_norm[discriminant_norm < 0] = 0.0

        if rho_R <= 1.0:
            # muon has hit the mirror
            effective_chord_length = radius * (
                np.sqrt(discriminant_norm) + rho_R * np.cos(phi_modulo)
            )
        else:
            # muon did not hit the mirror
            effective_chord_length = 2 * radius * np.sqrt(discriminant_norm)
            # Filtering out non-physical solutions for phi
            effective_chord_length *= np.where(
                (np.abs(phi_modulo) < np.arcsin(1.0 / rho_R)), 1, 0
            )

        return effective_chord_length

    return chord_length(radius, rho, 0, phi - phi0)


def save_histogram_to_csv(hist):
    df = pd.DataFrame(
        {
            "x": ((np.roll(hist[1], 1) + hist[1]) / 2.0)[1:],
            "y": hist[0],
        }
    )
    csvName = str("save_histogram_to_csv" + str(np.random.randint(0, 10000)) + ".csv")
    df.to_csv(csvName, sep=" ", index=False)

    return


def compute_muon_ring_width(
    x,
    y,
    mask,
    image,
    ring_center_x,
    ring_center_y,
):
    """
    compute_muon_ring_width


    Parameters
    ----------

    Returns
    -------


    """

    radius_bin_resolution = 0.05 * u.deg

    camera_unit = x.unit
    x, y, ring_center_x, ring_center_y, radius_bin_resolution = all_to_value(
        x, y, ring_center_x, ring_center_y, radius_bin_resolution, unit=camera_unit
    )

    max_fov = np.abs(2 * x.max())
    n_ring_radius_bins = int(2 * max_fov / radius_bin_resolution)

    ring_radius = np.sqrt(
        (y[mask] - ring_center_y) ** 2 + (x[mask] - ring_center_x) ** 2
    )
    weights = image[mask]
    save_histogram_to_csv(
        np.histogram(
            ring_radius,
            weights=weights,
            bins=np.linspace(-max_fov, max_fov, n_ring_radius_bins + 1),
        )
    )

    if weights.sum() == 0.0:
        return 0.0 * u.deg

    weighted_mean = np.average(
        ring_radius,
        weights=weights,
    )
    weighted_var = np.average((ring_radius - weighted_mean) ** 2, weights=weights)
    print(np.sqrt(weighted_var) * camera_unit)

    return np.sqrt(weighted_var) * camera_unit


def comput_absolute_optical_efficiency_from_muon_ring():
    """
    comput_absolute_optical_efficiency_from_muon_ring


    Parameters
    ----------

    Returns
    -------


    """

    return 0.2


def fit_muon_ring_phi_distribution(
    x,
    y,
    mask,
    image,
    ring_center_x,
    ring_center_y,
    optics,
    shadow_radius,
):
    """
    fit_muon_ring_phi_distribution


    Parameters
    ----------
    x : array-like or astropy.units.Quantity
        x-coordinates of the points.
    y : array-like or astropy.units.Quantity
        y-coordinates of the points.
    mask : array-like of bool
        Boolean mask indicating which pixels survive the cleaning process.
    image : array-like of float

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

    """

    if Minuit is None:
        raise OptionalDependencyMissing("iminuit")

    camera_unit = x.unit
    size_impact_point_unit = u.m
    x, y, ring_center_x, ring_center_y = all_to_value(
        x, y, ring_center_x, ring_center_y, unit=camera_unit
    )

    n_phi_bins = 12  # to be added to configure file as input perameters
    n_of_smoothing_points = 1  # 1 --> no smoothing

    hist_phi = np.histogram(
        np.arctan2(
            y[mask] - ring_center_y,
            x[mask] - ring_center_x,
        ),
        weights=image[mask],
        bins=np.linspace(-np.pi, np.pi, n_phi_bins + 1),
    )

    phi_y = (
        correlate1d(hist_phi[0], np.ones(n_of_smoothing_points), mode="wrap", axis=0)
        / n_of_smoothing_points,
    )[0]
    phi_y_err = np.sqrt(np.abs(phi_y))

    weights = (phi_y > 0).astype(int)

    phi_x = ((np.roll(hist_phi[1], 1) + hist_phi[1]) / 2.0)[1:]
    phi_x_err = np.ones(len(phi_x)) * np.pi / n_phi_bins

    phi_err = np.sqrt(phi_x_err**2 + phi_y_err**2)

    total_integral = np.sum(phi_y)

    if total_integral <= 0.0:
        return MuonEfficiencyContainer()

    amplitude_initial = np.nan
    rho_initial = np.nan
    phi0_initial = np.nan

    amplitude_initial = total_integral / 110.0
    rho_initial = 2 * (np.max(phi_y) - np.min(phi_y)) / total_integral * 110.0
    phi0_initial = phi_x[np.argmax(phi_y)]

    amplitude_initial = 12 if np.isnan(amplitude_initial) else amplitude_initial
    rho_initial = 8 if np.isnan(rho_initial) is np.nan else rho_initial
    phi0_initial = 0 if np.isnan(phi0_initial) is np.nan else phi0_initial

    # minimization method
    fit = Minuit(
        phi_dist_loss_function(phi_x, phi_y, phi_err, weights),
        amplitude=amplitude_initial,
        R_mirror=np.sqrt(optics.mirror_area.to_value(u.m**2) / np.pi),
        R_shadow=shadow_radius.to_value(u.m),
        rho=rho_initial,
        phi0=phi0_initial,
    )
    fit.errordef = Minuit.LEAST_SQUARES

    fit.fixed["R_mirror"] = True
    fit.fixed["R_shadow"] = True
    #
    # set initial parameters uncertainty to a big value
    # taubin_error = max_fov * 0.1
    fit.errors["amplitude"] = 10
    fit.errors["R_mirror"] = 0.0001
    fit.errors["R_shadow"] = 0.0001
    fit.errors["rho"] = 10.0
    fit.errors["phi0"] = np.pi

    fit.migrad()

    rho = Quantity(fit.values["rho"], size_impact_point_unit)

    return MuonEfficiencyContainer(
        impact=rho,
        impact_x=rho * np.cos(Quantity(fit.values["phi0"], camera_unit)),
        impact_y=rho * np.sin(Quantity(fit.values["phi0"], camera_unit)),
        is_valid=fit.valid,
        parameters_at_limit=fit.fmin.has_parameters_at_limit,
        likelihood_value=fit.fval,
    )


def phi_dist_loss_function(x, y, err, w):
    """dist_loss_function

    x, y, err: positions of pixels surviving the cleaning
        should not be quantities
    w : array-like of float, weights for the points

    """

    def loss_function(amplitude, R_mirror, R_shadow, rho, phi0):
        signal = amplitude * chord_length(R_mirror, rho, phi0, x)
        shadow = amplitude * chord_length(R_shadow, rho, phi0, x)
        diff_squared = ((signal - shadow - y) * w / err) ** 2

        return diff_squared.sum()

    return loss_function


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

    def __call__(
        self,
        tel_id,
        center_x,
        center_y,
        radius,
        image,
        pedestal,
        mask=None,
    ):
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

        mu_eff_container = fit_muon_ring_phi_distribution(
            geometry.pix_x,
            geometry.pix_y,
            mask,
            image,
            center_x,
            center_y,
            optics=telescope.optics,
            shadow_radius=self.hole_radius_m.tel[tel_id] * u.m,
        )

        mu_eff_container.width = compute_muon_ring_width(
            geometry.pix_x,
            geometry.pix_y,
            mask,
            image,
            center_x,
            center_y,
        )
        mu_eff_container.optical_efficiency = (
            comput_absolute_optical_efficiency_from_muon_ring()
        )

        return mu_eff_container
