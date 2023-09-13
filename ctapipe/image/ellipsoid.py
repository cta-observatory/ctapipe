# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: UTF-8 -*-
"""
Shower image parametrization based on image fitting.
"""

from enum import Enum

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from iminuit import Minuit

from ctapipe.image.cleaning import dilate
from ctapipe.image.hillas import hillas_parameters
from ctapipe.image.pixel_likelihood import (
    mean_poisson_likelihood_gaussian,
    neg_log_likelihood_approx,
)
from ctapipe.image.toymodel import SkewedGaussian, SkewedGaussianLaplace

from ..containers import CameraImageFitParametersContainer, ImageFitParametersContainer

PED_TABLE = {
    "LSTCam": 2.8,
    "NectarCam": 2.3,
    "FlashCam": 2.3,
    "SSTCam": 0.5,
    "CHEC": 0.5,
    "DUMMY": 0,
    "testcam": 0,
}
SPE_WIDTH = 0.5

__all__ = [
    "image_fit_parameters",
    "ImageFitParameterizationError",
    "PDFType",
]


class PDFType(Enum):
    gaussian = "gaussian"
    laplace = "laplace"
    skewed = "skewed"


def create_initial_guess(geometry, hillas, size):
    """

    This function computes the seeds of the fit with the Hillas parameters

    Parameters
    ----------
    geometry: ctapipe.instrument.CameraGeometry
        Camera geometry, the cleaning mask should be applied to improve performance
    image: ndarray
        Charge in each pixel, the cleaning mask should already be applied to
        improve performance.
    size: float/int
        Total charge after cleaning and dilation

    Returns
    -------
    initial_guess: dict
    Seed parameters of the fit.
    """
    unit = geometry.pix_x.unit
    initial_guess = {}

    if unit.is_equivalent(u.m):
        initial_guess["cog_x"] = hillas.x.to_value(unit)
        initial_guess["cog_y"] = hillas.y.to_value(unit)
    else:
        initial_guess["cog_x"] = hillas.fov_lon.to_value(unit)
        initial_guess["cog_y"] = hillas.fov_lat.to_value(unit)

    initial_guess["length"] = hillas.length.to_value(unit)
    initial_guess["width"] = hillas.width.to_value(unit)
    initial_guess["psi"] = hillas.psi.value

    initial_guess["skewness"] = hillas.skewness

    if (hillas.width.to_value(unit) == 0) or (hillas.length.to_value(unit) == 0):
        raise ImageFitParameterizationError("Hillas width and/or length is zero")

    initial_guess["amplitude"] = size
    return initial_guess


def boundaries(geometry, image, dilated_mask, hillas, pdf):
    """

    Computes the boundaries of the fit.

    Parameters
    ----------
    geometry : ctapipe.instrument.CameraGeometry
        Camera geometry
    image : ndarray
        Charge in each pixel, no cleaning mask should be applied
    dilated_mask : ndarray
        mask (array of booleans) after image cleaning and dilation
    x0 : dict
       seeds of the fit
    pdf: PDFType instance
        e.g. PDFType("gaussian")

    Returns
    -------
    list of boundaries
    """
    x = geometry.pix_x.value
    y = geometry.pix_y.value
    unit = geometry.pix_x.unit
    camera_radius = geometry.radius.to_value(unit)

    # Using dilated image
    row_image = image.copy()
    row_image[~dilated_mask] = 0.0
    row_image[row_image < 0] = 0.0

    max_x = np.max(x[dilated_mask])
    min_x = np.min(x[dilated_mask])
    max_y = np.max(y[dilated_mask])
    min_y = np.min(y[dilated_mask])

    ang_unit = hillas.psi.unit
    psi_unc = 10 * u.deg
    psi_min, psi_max = max(
        hillas.psi.value - psi_unc.to_value(ang_unit), -np.pi / 2
    ), min(hillas.psi.value + psi_unc.to_value(ang_unit), np.pi / 2)

    cogx_min, cogx_max = np.sign(min_x) * min(np.abs(min_x), camera_radius), np.sign(
        max_x
    ) * min(np.abs(max_x), camera_radius)

    cogy_min, cogy_max = np.sign(min_y) * min(np.abs(min_y), camera_radius), np.sign(
        max_y
    ) * min(np.abs(max_y), camera_radius)

    long_dis = 2 * np.sqrt(
        (max_x - min_x) ** 2 + (max_y - min_y) ** 2
    )  # maximum distance of the shower in the longitudinal direction

    if unit.is_equivalent(u.m):
        width_unc = 0.05 * u.m
    else:
        width_unc = 0.1 * u.deg
    length_min, length_max = hillas.length.to_value(unit), long_dis
    width_min, width_max = hillas.width.to_value(unit), hillas.width.to_value(
        unit
    ) + width_unc.to_value(unit)

    scale = length_min / np.sqrt(1 - 2 / np.pi)
    skew_min, skew_max = min(max(-0.99, hillas.skewness - 0.3), 0.99), max(
        -0.99, min(0.99, hillas.skewness + 0.3)
    )  # Guess from Hillas unit tests

    bounds = [
        (cogx_min, cogx_max),
        (cogy_min, cogy_max),
        (psi_min, psi_max),
        (length_min, length_max),
        (width_min, width_max),
        (skew_min, skew_max),
    ]

    if pdf == PDFType.gaussian:
        amplitude = np.sum(row_image) / (2 * np.pi * width_min * length_min)
    elif pdf == PDFType.skewed:
        amplitude = np.sum(row_image) / scale / (2 * np.pi * width_min)
    else:
        amplitude = (
            np.sum(row_image) / scale / (np.sqrt(2 * np.pi) * np.sqrt(2) * width_min)
        )

    bounds.append((0, amplitude))
    return bounds


class ImageFitParameterizationError(RuntimeError):
    pass


def image_fit_parameters(
    geom,
    image,
    n_row,
    cleaned_mask,
    pdf=PDFType("skewed"),
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
    bounds : list
        default format: [(low_x, high_x), (low_y, high_y), ...]
        Boundary conditions. If bounds == None, boundaries function is applied as a default.
    n_row : int
      number of extra rows of neighbors added to the cleaning mask
    cleaned_mask : ndarray
       Cleaning mask (array of booleans) after cleaning
    pdf: PDFType instance
        e.g. PDFType("gaussian")

    Returns
    -------
    ImageFitParametersContainer:
        container of image parameters after fitting
    """
    # For likelihood calculation we need the width of the
    # pedestal distribution for each pixel
    # currently this is not available from the calibration,
    # so for now lets hard code it in a dict

    pedestal = PED_TABLE[geom.name]
    pdf = PDFType(pdf)
    pdf_dict = {
        PDFType.gaussian: SkewedGaussian,
        PDFType.skewed: SkewedGaussian,
        PDFType.laplace: SkewedGaussianLaplace,
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

    dilated_mask = cleaned_mask.copy()
    for row in range(n_row):
        dilated_mask = dilate(geom, dilated_mask)

    dilated_image = image.copy()
    dilated_image[~dilated_mask] = 0.0
    dilated_image[dilated_image < 0] = 0.0
    size = np.sum(dilated_image)

    hillas = hillas_parameters(geom, cleaned_image)

    x0 = create_initial_guess(geom, hillas, size)  # seeds

    if np.count_nonzero(image) <= len(x0):
        raise ImageFitParameterizationError(
            "The number of free parameters is higher than the number of pixels to fit, cannot perform fit"
        )

    def likelihood(cog_x, cog_y, psi, length, width, skewness, amplitude):
        parameters = [
            cog_x * unit,
            cog_y * unit,
            length * unit,
            width * unit,
            psi * u.rad,
            skewness,
            amplitude,
        ]

        prediction = pdf_dict[pdf](*parameters).pdf(geom.pix_x, geom.pix_y)

        prediction[np.isnan(prediction)] = 1e9
        like = neg_log_likelihood_approx(dilated_image, prediction, SPE_WIDTH, pedestal)
        if np.isnan(like):
            like = 1e9
        return like

    if pdf == PDFType.gaussian:
        x0["skewness"] = 0

    m = Minuit(
        likelihood,
        **x0,
    )

    if pdf == PDFType.gaussian:
        m.fixed = np.zeros(len(x0.values()))
        m.fixed[-2] = True

    if bounds is None:
        bounds = boundaries(geom, image, dilated_mask, hillas, pdf)
        m.limits = bounds
    else:
        m.limits = bounds

    m.errordef = 1  # neg log likelihood
    m.simplex().migrad()
    m.hesse()

    likelihood = m.fval
    pars = m.values
    errors = m.errors

    like_array = likelihood
    like_array *= dilated_mask
    goodness_of_fit = np.sum(
        like_array[like_array > 0]
        - mean_poisson_likelihood_gaussian(
            pdf_dict[pdf](
                pars[0] * unit,
                pars[1] * unit,
                pars[3] * unit,
                pars[4] * unit,
                pars[2] * u.rad,
                pars[5],
                pars[6],
            ).pdf(geom.pix_x, geom.pix_y),
            SPE_WIDTH,
            pedestal,
        )
    )

    fit_rcog = np.linalg.norm([pars[0], pars[1]])
    fit_phi = np.arctan2(pars[1], pars[0])

    b = pars[1] ** 2 + pars[0] ** 2
    A = (-pars[1] / (b)) ** 2
    B = (pars[0] / (b)) ** 2
    fit_phi_err = np.sqrt(A * errors[0] ** 2 + B * errors[1] ** 2)
    fit_rcog_err = np.sqrt(
        pars[0] ** 2 / b * errors[0] ** 2 + pars[1] ** 2 / b * errors[1] ** 2
    )

    delta_x = pix_x.value - pars[0]
    delta_y = pix_y.value - pars[1]

    longitudinal = delta_x * np.cos(pars[2]) + delta_y * np.sin(pars[2])

    m4_long = np.average(longitudinal**4, weights=dilated_image)
    kurtosis_long = m4_long / pars[3] ** 4

    skewness_long = pars[5]
    skewness_uncertainty = errors[5]
    amplitude = pars[6]
    amplitude_uncertainty = errors[6]

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
            goodness_of_fit=goodness_of_fit,
            n_pixels=np.count_nonzero(cleaned_image),
            free_parameters=m.nfit,
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
        goodness_of_fit=goodness_of_fit,
        n_pixels=np.count_nonzero(cleaned_image),
        free_parameters=m.nfit,
        is_valid=m.valid,
        is_accurate=m.accurate,
    )
