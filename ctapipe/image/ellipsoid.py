# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: UTF-8 -*-
"""
Shower image parametrization based on image fitting.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from iminuit import Minuit

from ctapipe.image.cleaning import dilate
from ctapipe.image.hillas import hillas_parameters
from ctapipe.image.leakage import leakage_parameters
from ctapipe.image.pixel_likelihood import neg_log_likelihood_approx
from ctapipe.image.toymodel import Gaussian, SkewedCauchy, SkewedGaussian

from ..containers import CameraImageFitParametersContainer, ImageFitParametersContainer

__all__ = ["image_fit_parameters", "ImageFitParameterizationError"]


def create_initial_guess(geometry, image, size):
    """
    This function computes the seeds of the fit with Hillas parameters
    Parameters
    ----------
    geometry : ctapipe.instrument.CameraGeometry
        Camera geometry, the cleaning mask should be applied to improve performance
    image : ndarray
        Charge in each pixel, the cleaning mask should already be applied to
        improve performance.
    size : float/int
        Total charge after cleaning and dilation
    Returns
    -------
    initial_guess : seed
    """
    unit = geometry.pix_x.unit
    hillas = hillas_parameters(geometry, image)

    initial_guess = {}

    if unit.is_equivalent(u.m):
        initial_guess["x"] = hillas.x
        initial_guess["y"] = hillas.y
    else:
        initial_guess["x"] = hillas.fov_lon
        initial_guess["y"] = hillas.fov_lat

    initial_guess["length"] = hillas.length
    initial_guess["width"] = hillas.width
    initial_guess["psi"] = hillas.psi
    initial_guess["skewness"] = hillas.skewness

    if (hillas.width.to_value(unit) == 0) or (hillas.length.to_value(unit) == 0):
        raise ImageFitParameterizationError("Hillas width and/or length is zero")

    initial_guess["amplitude"] = size

    return initial_guess


def extra_rows(n, cleaned_mask, geometry):
    """
    This function adds n extra rows around the cleaned pixels
    Parameters
    ----------
    n : int
       number of extra rows to add after cleaning
    cleaned_mask : boolean
       The cleaning mask applied for Hillas parametrization
    geometry : ctapipe.instrument.CameraGeometry
        Camera geometry
    """
    mask = cleaned_mask.copy()
    for row in range(n):
        mask = dilate(geometry, mask)

    mask = np.array((mask.astype(int) + cleaned_mask.astype(int)), dtype=bool)

    return mask


def sensible_boundaries(geometry, cleaned_image, pdf):
    """
    Computes boundaries of the fit based on the deviation from Hillas parameters
    Parameters
    ----------
    geometry: ctapipe.instrument.CameraGeometry
        Camera geometry
    cleaned_image: ndarray
        Charge for each pixel, cleaning mask should be applied
    pdf: str
        name of the PDF used, options = "Gaussian", "Cauchy", "Skewed"
    Returns
    -------
    list of boundaries
    """
    hillas = hillas_parameters(geometry, cleaned_image)

    unit = geometry.pix_x.unit
    camera_radius = geometry.guess_radius()

    cogx_min, cogx_max = np.sign(hillas.x) * min(
        np.abs(hillas.x - u.Quantity(0.2, unit)), camera_radius
    ), np.sign(hillas.x) * min(np.abs(hillas.x + u.Quantity(0.2, unit)), camera_radius)
    cogy_min, cogy_max = np.sign(hillas.y) * min(
        np.abs(hillas.y - u.Quantity(0.2, unit)), camera_radius
    ), np.sign(hillas.y) * min(np.abs(hillas.y + u.Quantity(0.2, unit)), camera_radius)

    psi_min, psi_max = -np.pi / 2, np.pi / 2
    length_min, length_max = hillas.length, hillas.length + u.Quantity(0.3, unit)
    width_min, width_max = hillas.width, hillas.width + u.Quantity(0.1, unit)
    skew_min, skew_max = -0.99, 0.99
    ampl_min, ampl_max = 0, np.inf

    if pdf != "Gaussian":
        return [
            (cogx_min.to_value(unit), cogx_max.to_value(unit)),
            (cogy_min.to_value(unit), cogy_max.to_value(unit)),
            (psi_min, psi_max),
            (length_min.to_value(unit), length_max.to_value(unit)),
            (width_min.to_value(unit), width_max.to_value(unit)),
            (skew_min, skew_max),
            (ampl_min, ampl_max),
        ]

    return [
        (cogx_min.to_value(unit), cogx_max.to_value(unit)),
        (cogy_min.to_value(unit), cogy_max.to_value(unit)),
        (psi_min, psi_max),
        (length_min.to_value(unit), length_max.to_value(unit)),
        (width_min.to_value(unit), width_max.to_value(unit)),
        (ampl_min, ampl_max),
    ]


def boundaries(geometry, image, dilated_mask, x0, pdf):
    """
    Computes the boundaries of the fit.
    Parameters
    ----------
    geometry : ctapipe.instrument.CameraGeometry
        Camera geometry
    image : ndarray
        Charge in each pixel, no cleaning mask should be applied
    dilated_mask : boolean
        mask after image cleaning and dilation
    x0 : dict
       seeds of the fit
    pdf: str
       name of the PDF, options = "Gaussian", "Cauchy", "Skewed"
    Returns
    -------
    list of boundaries
    """
    x = geometry.pix_x.value
    y = geometry.pix_y.value
    unit = geometry.pix_x.unit
    camera_radius = geometry.guess_radius().to_value(unit)
    leakage = leakage_parameters(geometry, image, dilated_mask)

    # Dilated image
    row_image = image.copy()
    row_image[~dilated_mask] = 0.0
    row_image[row_image < 0] = 0.0

    max_x = np.max(x[dilated_mask])
    min_x = np.min(x[dilated_mask])
    max_y = np.max(y[dilated_mask])
    min_y = np.min(y[dilated_mask])

    cogx_min, cogx_max = np.sign(min_x) * min(np.abs(min_x), camera_radius), np.sign(
        max_x
    ) * min(np.abs(max_x), camera_radius)

    cogy_min, cogy_max = np.sign(min_y) * min(np.abs(min_y), camera_radius), np.sign(
        max_y
    ) * min(np.abs(max_y), camera_radius)

    if (leakage.intensity_width_1 > 0.2) & (
        leakage.intensity_width_2 > 0.2
    ):  # truncated
        if (x0["x"] > 0) & (x0["y"] > 0):
            max_x = 2 * max_x
            max_y = 2 * max_y
        if (x0["x"] < 0) & (x0["y"] > 0):
            min_x = 2 * min_x
            max_y = 2 * max_y
        if (x0["x"] < 0) & (x0["y"] < 0):
            min_x = 2 * min_x
            min_y = 2 * min_y
        if (x0["x"] > 0) & (x0["y"] < 0):
            max_x = 2 * max_x
            min_y = 2 * min_y

    long_dis = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

    width_unc = u.Quantity(0.05, unit)
    length_min, length_max = x0["length"].value, long_dis
    width_min, width_max = x0["width"].value, x0["width"].value + width_unc.to_value(
        unit
    )

    scale = length_min / np.sqrt(1 - 2 / np.pi)
    skew_min, skew_max = -0.99, 0.99

    if pdf == "Gaussian":
        amplitude = np.sum(row_image) / (2 * np.pi * width_min * length_min)

        return [
            (cogx_min, cogx_max),
            (cogy_min, cogy_max),
            (-np.pi / 2, np.pi / 2),
            (length_min, length_max),
            (width_min, width_max),
            (0, amplitude),
        ]
    if pdf == "Skewed":
        amplitude = np.sum(row_image) / scale * 1 / (2 * np.pi * width_min)

        return [
            (cogx_min, cogx_max),
            (cogy_min, cogy_max),
            (-np.pi / 2, np.pi / 2),
            (length_min, length_max),
            (width_min, width_max),
            (skew_min, skew_max),
            (0, amplitude),
        ]
    if pdf == "Cauchy":
        amplitude = (
            np.sum(row_image) / scale * 1 / (np.sqrt(2 * np.pi) * np.pi * width_min / 2)
        )

        return [
            (cogx_min, cogx_max),
            (cogy_min, cogy_max),
            (-np.pi / 2, np.pi / 2),
            (length_min, length_max),
            (width_min, width_max),
            (skew_min, skew_max),
            (0, amplitude),
        ]


class ImageFitParameterizationError(RuntimeError):
    pass


def image_fit_parameters(
    geom,
    image,
    n,
    cleaned_mask,
    pdf="Cauchy",
    bounds=None,
):
    """
    Computes image parameters for a given shower image.
    Implementation based on https://arxiv.org/pdf/1211.0254.pdf
    Parameters
    ----------
    geom : ctapipe.instrument.CameraGeometry
        Camera geometry
    image : ndarray
        Charge in each pixel, no cleaning mask should be applied
    bounds : default format [(low_limx, high_limx), (low_limy, high_limy), ...]
        Boundary conditions. If bounds == None, boundaries function is applied as a default.
    n : int
      number of extra rows to add after cleaning
    cleaned_mask : boolean
       The cleaning mask to apply to find Hillas parameters
    pdf : str
        name of the prob distrib to use for the fit, options = "Gaussian", "Cauchy", "Skewed"
    Returns
    -------
    ImageFitParametersContainer:
        container of image-fitting parameters
    """
    # For likelihood calculation we need the with of the
    # pedestal distribution for each pixel
    # currently this is not available from the calibration,
    # so for now lets hard code it in a dict

    ped_table = {
        "LSTCam": 2.8,
        "NectarCam": 2.3,
        "FlashCam": 2.3,
        "CHEC": 0.5,
        "DUMMY": 0,
        "testcam": 0,
    }
    spe_width = 0.5
    pedestal = ped_table[geom.name]
    pdf_dict = {
        "Gaussian": Gaussian,
        "Skewed": SkewedGaussian,
        "Cauchy": SkewedCauchy,
    }

    unit = geom.pix_x.unit
    pix_x = geom.pix_x
    pix_y = geom.pix_y
    image = np.asanyarray(image, dtype=np.float64)

    if np.sum(image) == 0.0:
        raise ImageFitParameterizationError("size=0, cannot calculate HillasParameters")

    if isinstance(image, np.ma.masked_array):
        image = np.ma.filled(image, 0)

    if not (pix_x.shape == pix_y.shape == image.shape == cleaned_mask.shape):
        raise ValueError("Image length and number of pixels do not match")

    cleaned_image = image.copy()
    cleaned_image[~cleaned_mask] = 0.0
    cleaned_image[cleaned_image < 0] = 0.0

    dilated_mask = extra_rows(n, cleaned_mask, geom)
    dilated_image = image.copy()
    dilated_image[~dilated_mask] = 0.0
    dilated_image[dilated_image < 0] = 0.0
    size = np.sum(dilated_image)

    x0 = create_initial_guess(geom, cleaned_image, size)

    if np.count_nonzero(image) <= len(x0):
        raise ImageFitParameterizationError(
            "The number of free parameters is higher than the number of pixels to fit, cannot perform fit"
        )

    def fit(cog_x, cog_y, psi, length, width, skewness, amplitude):
        prediction = pdf_dict[pdf](
            cog_x * unit,
            cog_y * unit,
            length * unit,
            width * unit,
            psi * u.rad,
            skewness,
            amplitude,
        ).pdf(geom.pix_x, geom.pix_y)
        prediction[np.isnan(prediction)] = 1e9
        like = neg_log_likelihood_approx(dilated_image, prediction, spe_width, pedestal)
        if np.isnan(like):
            like = 1e9
        return like

    def fit_gauss(cog_x, cog_y, psi, length, width, amplitude):
        prediction = pdf_dict[pdf](
            cog_x * unit,
            cog_y * unit,
            length * unit,
            width * unit,
            psi * u.rad,
            amplitude,
        ).pdf(geom.pix_x, geom.pix_y)
        prediction[np.isnan(prediction)] = 1e9
        like = neg_log_likelihood_approx(dilated_image, prediction, spe_width, pedestal)
        if np.isnan(like):
            like = 1e9
        return like

    if pdf != "Gaussian":
        m = Minuit(
            fit,
            cog_x=x0["x"].to_value(unit),
            cog_y=x0["y"].to_value(unit),
            length=x0["length"].to_value(unit),
            width=x0["width"].to_value(unit),
            psi=x0["psi"].value,
            skewness=x0["skewness"],
            amplitude=x0["amplitude"],
        )
    else:
        m = Minuit(
            fit_gauss,
            cog_x=x0["x"].to_value(unit),
            cog_y=x0["y"].to_value(unit),
            length=x0["length"].to_value(unit),
            width=x0["width"].to_value(unit),
            psi=x0["psi"].value,
            amplitude=x0["amplitude"],
        )

    if bounds is None:
        bounds = boundaries(geom, image, dilated_mask, x0, pdf)
        m.limits = bounds
    if bounds is not None:
        m.limits = bounds

    m.errordef = 1  # neg log likelihood
    m.simplex().migrad()
    m.hesse()

    likelihood = m.fval
    pars = m.values
    errors = m.errors

    fit_rcog = np.linalg.norm([pars[0], pars[1]])
    fit_phi = np.arctan2(pars[1], pars[0])

    b = pars[1] ** 2 + pars[0] ** 2
    A = (-pars[1] / (b)) ** 2
    B = (pars[0] / (b)) ** 2
    fit_phi_err = np.sqrt(A * errors[0] ** 2 + B * errors[1] ** 2)
    fit_rcog_err = np.sqrt(
        pars[0] ** 2 / b * errors[0] ** 2 + pars[1] ** 2 / b * errors[1] ** 2
    )

    delta_x = geom.pix_x.value - pars[0]
    delta_y = geom.pix_y.value - pars[1]

    longitudinal = delta_x * np.cos(pars[2]) + delta_y * np.sin(pars[2])

    m4_long = np.average(longitudinal**4, weights=dilated_image)
    kurtosis_long = m4_long / pars[3] ** 4

    if pdf != "Gaussian":
        skewness_long = pars[5]
        skewness_uncertainty = errors[5]
        amplitude = pars[6]
        amplitude_uncertainty = errors[6]
    else:
        m3_long = np.average(longitudinal**3, weights=dilated_image)
        skewness_long = m3_long / pars[3] ** 3
        skewness_uncertainty = np.nan
        amplitude = pars[5]
        amplitude_uncertainty = errors[5]

    if unit.is_equivalent(u.m):
        return CameraImageFitParametersContainer(
            x=u.Quantity(pars[0], unit),
            x_uncertainty=u.Quantity(errors[0], unit),
            y=u.Quantity(pars[1], unit),
            y_uncertainty=u.Quantity(errors[1], unit),
            r=u.Quantity(fit_rcog, unit),
            r_uncertainty=u.Quantity(fit_rcog_err, unit),
            phi=Angle(fit_phi, unit=u.rad),
            phi_uncertainty=Angle(fit_phi_err, unit=u.rad),
            intensity=size,
            amplitude=amplitude,
            amplitude_uncertainty=amplitude_uncertainty,
            length=u.Quantity(pars[3], unit),
            length_uncertainty=u.Quantity(errors[3], unit),
            width=u.Quantity(pars[4], unit),
            width_uncertainty=u.Quantity(errors[4], unit),
            psi=Angle(pars[2], unit=u.rad),
            psi_uncertainty=Angle(errors[2], unit=u.rad),
            skewness=skewness_long,
            skewness_uncertainty=skewness_uncertainty,
            kurtosis=kurtosis_long,
            likelihood=likelihood,
            n_pix_fit=np.count_nonzero(cleaned_image),
            n_free_par=len(x0),
            is_valid=m.valid,
            is_accurate=m.accurate,
        )
    return ImageFitParametersContainer(
        fov_lon=u.Quantity(pars[0], unit),
        fov_lon_uncertainty=u.Quantity(errors[0], unit),
        fov_lat=u.Quantity(pars[1], unit),
        fov_lat_uncertainty=u.Quantity(errors[1], unit),
        r=u.Quantity(fit_rcog, unit),
        r_uncertainty=u.Quantity(fit_rcog_err, unit),
        phi=Angle(fit_phi, unit=u.rad),
        phi_uncertainty=Angle(fit_phi_err, unit=u.rad),
        intensity=size,
        amplitude=amplitude,
        amplitude_uncertainty=amplitude_uncertainty,
        length=u.Quantity(pars[3], unit),
        length_uncertainty=u.Quantity(errors[3], unit),
        width=u.Quantity(pars[4], unit),
        width_uncertainty=u.Quantity(errors[4], unit),
        psi=Angle(pars[2], unit=u.rad),
        psi_uncertainty=Angle(errors[2], unit=u.rad),
        skewness=skewness_long,
        skewness_uncertainty=skewness_uncertainty,
        kurtosis=kurtosis_long,
        likelihood=likelihood,
        n_pix_fit=np.count_nonzero(cleaned_image),
        n_free_par=len(x0),
        is_valid=m.valid,
        is_accurate=m.accurate,
    )
