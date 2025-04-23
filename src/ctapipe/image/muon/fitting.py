import numpy as np
from astropy.units import Quantity

from ...exceptions import OptionalDependencyMissing
from ...utils.quantities import all_to_value

__all__ = ["kundu_chaudhuri_circle_fit", "taubin_circle_fit"]

try:
    from iminuit import Minuit
except ModuleNotFoundError:
    Minuit = None


def kundu_chaudhuri_circle_fit(x, y, weights):
    """
    Fast and reliable analytical circle fitting method previously used
    in the H.E.S.S. experiment for muon identification

    Implementation based on :cite:p:`chaudhuri93`

    Parameters
    ----------
    x: array-like or astropy quantity
        x coordinates of the points
    y: array-like or astropy quantity
        y coordinates of the points
    weights: array-like
        weights of the points

    """

    weights_sum = np.sum(weights)
    mean_x = np.sum(x * weights) / weights_sum
    mean_y = np.sum(y * weights) / weights_sum

    a1 = np.sum(weights * (x - mean_x) * x)
    a2 = np.sum(weights * (y - mean_y) * x)

    b1 = np.sum(weights * (x - mean_x) * y)
    b2 = np.sum(weights * (y - mean_y) * y)

    c1 = 0.5 * np.sum(weights * (x - mean_x) * (x**2 + y**2))
    c2 = 0.5 * np.sum(weights * (y - mean_y) * (x**2 + y**2))

    center_x = (b2 * c1 - b1 * c2) / (a1 * b2 - a2 * b1)
    center_y = (a2 * c1 - a1 * c2) / (a2 * b1 - a1 * b2)

    radius = np.sqrt(
        np.sum(weights * ((center_x - x) ** 2 + (center_y - y) ** 2)) / weights_sum
    )

    return radius, center_x, center_y


def taubin_circle_fit(
    x, y, mask, weights=None, taubin_r_initial=None, xc_initial=None, yc_initial=None
):
    """
    reference : Barcelona_Muons_TPA_final.pdf (slide 6)
    updated with weight

    Parameters
    ----------
    x: array-like or astropy quantity
        x coordinates of the points
    y: array-like or astropy quantity
        y coordinates of the points
    mask: array-like boolean
        true for pixels surviving the cleaning
    weights: array-like float
    taubin_r_initial: astropy quantity - initial r
    xc_initial: astropy quantity - initial xc (center of the ring)
    yc_initial: astropy quantity - initial yc (center of the ring)
    """
    if Minuit is None:
        raise OptionalDependencyMissing("iminuit")

    original_unit = x.unit
    x, y = all_to_value(x, y, unit=original_unit)

    x_masked = x[mask]
    y_masked = y[mask]

    if weights is None:
        weights_masked = np.ones(len(x_masked))
    else:
        weights_masked = weights[mask]

    R = x.max()  # x.max() just happens to be identical with R in many cases.
    if taubin_r_initial is None or xc_initial is None or yc_initial is None:
        taubin_r_initial = Quantity(R / 2, original_unit)
        xc_initial = Quantity(0, original_unit)
        yc_initial = Quantity(0, original_unit)

    taubin_error = R * 0.1

    # minimization method
    fit = Minuit(
        make_taubin_loss_function(x_masked, y_masked, weights_masked),
        xc=xc_initial.to_value(original_unit),
        yc=yc_initial.to_value(original_unit),
        r=taubin_r_initial.to_value(original_unit),
    )
    fit.errordef = Minuit.LEAST_SQUARES

    fit.errors["xc"] = taubin_error
    fit.errors["yc"] = taubin_error
    fit.errors["r"] = taubin_error

    fit.limits["xc"] = (-2 * R, 2 * R)
    fit.limits["yc"] = (-2 * R, 2 * R)
    fit.limits["r"] = (0, R)

    fit.migrad()

    radius = Quantity(fit.values["r"], original_unit)
    center_x = Quantity(fit.values["xc"], original_unit)
    center_y = Quantity(fit.values["yc"], original_unit)

    return radius, center_x, center_y


def make_taubin_loss_function(x, y, w):
    """closure around taubin_loss_function to make
    surviving pixel positions availaboe inside.

    x, y: positions of pixels surviving the cleaning
        should not be quantities
    """

    def taubin_loss_function(xc, yc, r):
        """taubin fit formula
        reference : Barcelona_Muons_TPA_final.pdf (slide 6)
        """
        upper_term = ((w * ((x - xc) ** 2 + (y - yc) ** 2 - r**2)) ** 2).sum()

        lower_term = (w * ((x - xc) ** 2 + (y - yc) ** 2)).sum()

        return np.abs(upper_term) / np.abs(lower_term)

    return taubin_loss_function
