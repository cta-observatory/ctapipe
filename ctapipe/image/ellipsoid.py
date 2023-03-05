# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: UTF-8 -*-
"""
Image fitting based shower image parametrization.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from ctapipe.image.cleaning import dilate
import scipy.optimize as opt
from iminuit import Minuit
from ctapipe.image.pixel_likelihood import chi_squared, neg_log_likelihood_approx, neg_log_likelihood_numeric
from ctapipe.image.hillas import hillas_parameters, camera_to_shower_coordinates
from ctapipe.image.toymodel import SkewedCauchy, SkewedGaussian, Gaussian
from ..containers import CameraImageFitParametersContainer, ImageFitParametersContainer
from itertools import combinations
from ctapipe.image.leakage import leakage_parameters
from ctapipe.image.concentration import concentration_parameters

__all__ = ["image_fit_parameters", "ImageFitParameterizationError"]

def create_initial_guess(geometry, image, pdf):
    """
    This function computes the initial guess for the image fit using the Hillas parameters
    
    Parameters
    ----------
    geometry : ctapipe.instrument.CameraGeometry
        Camera geometry, the cleaning mask should be applied to improve performance
    image : array_like
        Charge in each pixel, the cleaning mask should already be applied to
        improve performance.
    pdf : str
        Name of the PDF function to use

    Returns
    -------
    initial_guess : Hillas parameters 
    """
    hillas = hillas_parameters(geometry, image)
    
    initial_guess = {}
    initial_guess["x"] = hillas.x
    initial_guess["y"] = hillas.y
    initial_guess["length"] = hillas.length
    initial_guess["width"] = hillas.width
    initial_guess["psi"] = hillas.psi
    initial_guess["skewness"] = hillas.skewness
    skew23 = np.abs(hillas.skewness) ** (2 / 3)
    delta = np.sign(hillas.skewness) * np.sqrt((np.pi / 2 * skew23) / (skew23 + (0.5 * (4 - np.pi)) ** (2 / 3)))
    scale_skew = hillas.length.to_value(u.m) / np.sqrt(1 - 2 * delta ** 2 / np.pi)
    
    if pdf == "Gaussian":
        initial_guess["amplitude"] = hillas.intensity/(np.sqrt(2*np.pi)*hillas.length.value*hillas.width.value)
    if pdf == "Cauchy":
        initial_guess["amplitude"] = hillas.intensity/(np.pi*scale_skew*hillas.width.value)
    if pdf == "Skewed":
        initial_guess["amplitude"] = hillas.intensity/(np.sqrt(2*np.pi)*scale_skew*hillas.width.value)

    return initial_guess

def extra_rows(n, cleaned_mask, geometry):
    """
    This function adds n extra rows to the cleaning mask for a better fit of the shower tail
    
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
    for ii in range(n):
        mask = dilate(geometry, mask)

    mask = np.array((mask.astype(int) + cleaned_mask.astype(int)), dtype=bool)

    return mask

def boundaries(geometry, image, cleaning_mask, clean_row_mask, x0, pdf):
    """
    Computes the boundaries of the fit.

    Parameters
    ----------
    geometry : ctapipe.instrument.CameraGeometry
        Camera geometry
    image : array-like
        Charge in each pixel, no cleaning mask should be applied
    cleaning_mask : boolean
        mask after image cleaning
    clean_row_mask : boolean
        mask after image cleaning and dilation
    x0 : dict
       seeds of the fit
    pdf: str
       PDF name
    
    Returns
    -------
    Limits of the fit for each free parameter
    """
    row_image = image.copy()
    row_image[~clean_row_mask] = 0.0
    row_image[row_image < 0] = 0.0

    cleaned_image= image.copy()
    cleaned_image[~cleaning_mask] = 0.0

    pix_area = geometry.pix_area.value[0]
    area = pix_area * np.count_nonzero(row_image)

    leakage = leakage_parameters(geometry, image, clean_row_mask)
    fract_pix_border = leakage.pixels_width_2
    fract_int_border = leakage.intensity_width_2

    delta_x = geometry.pix_x.value - x0["x"].value
    delta_y = geometry.pix_y.value - x0["y"].value
    longitudinal = delta_x * np.cos(x0["psi"].value) + delta_y * np.sin(x0["psi"].value)
    transverse = delta_x * -np.sin(x0["psi"].value) + delta_y * np.cos(x0["psi"].value)

    cogx_min, cogx_max = np.min(geometry.pix_x.value[cleaned_image>0]), np.max(geometry.pix_x.value[cleaned_image>0])
    cogy_min, cogy_max = np.min(geometry.pix_y.value[cleaned_image>0]), np.max(geometry.pix_y.value[cleaned_image>0])

    x_dis = np.max(longitudinal[row_image>0]) - np.min(longitudinal[row_image>0])
    y_dis = np.max(transverse[row_image>0]) - np.min(transverse[row_image>0])
    length_min, length_max = np.sqrt(pix_area), x_dis/(1 - fract_pix_border)

    width_min, width_max = np.sqrt(pix_area), y_dis
    scale = length_min/ np.sqrt(1 - 2 / np.pi) 
    skew_min, skew_max = -0.99, 0.99

    if pdf == "Gaussian":
        return [(cogx_min, cogx_max), (cogy_min, cogy_max), (-np.pi/2, np.pi/2), (length_min, length_max), (width_min, width_max), (0, np.sum(row_image)*1/(2*np.pi*width_min*length_min))]
    if pdf == "Skewed":
        return [(cogx_min, cogx_max), (cogy_min, cogy_max), (-np.pi/2, np.pi/2), (length_min, length_max), (width_min, width_max), (skew_min, skew_max), (0, np.sum(row_image) * 1/scale * 1/(np.sqrt(2*np.pi)*width_min))]
    if pdf == "Cauchy":
        return [(cogx_min, cogx_max), (cogy_min, cogy_max), (-np.pi/2, np.pi/2), (length_min, length_max), (width_min, width_max), (skew_min, skew_max), (0, np.sum(row_image) * 1/scale * 1/(np.pi*width_min))]

def image_fit_parameters(geom, image, n, cleaned_mask, spe_width, pedestal, pdf, bounds=None):
    """
    Computes image parameters for a given shower image.

    Implementation similar to https://arxiv.org/pdf/1211.0254.pdf

    Parameters
    ----------
    geom : ctapipe.instrument.CameraGeometry
        Camera geometry
    image : array_like
        Charge in each pixel
    bounds : default format [(low_limx, high_limx), (low_limy, high_limy), ...]
        Parameters boundary condition. If bounds == None, boundaries function is applied as a default
    n : int
      number of extra rows after cleaning
    cleaned_mask : boolean
       The cleaning mask applied for Hillas parametrization
    spe_width: ndarray
        Width of single p.e. peak (:math:`σ_γ`).
    pedestal: ndarray
        Width of pedestal (:math:`σ_p`).
    pdf : str
        name of the prob distrib to use for the fit

    Returns
    -------
    ImageFitParametersContainer:
        container of image-fitting parameters
    """
    unit = geom.pix_x.unit
    pix_x = geom.pix_x
    pix_y = geom.pix_y
    image = np.asanyarray(image, dtype=np.float64)

    pdf_dict = {"Gaussian": Gaussian,
            "Skewed": SkewedGaussian,
            "Cauchy": SkewedCauchy,
            }

    if isinstance(image, np.ma.masked_array):
        image = np.ma.filled(image, 0)

    if not (pix_x.shape == pix_y.shape == image.shape):
        raise ValueError("Image and pixel shape do not match")

    prev_image = image.copy()
    prev_image[~cleaned_mask] = 0.0
    x0 = create_initial_guess(geom, prev_image, pdf)

    if np.count_nonzero(image) <= len(x0):
        raise ImageFitParameterizationError("The number of free parameters is higher than the number of pixels to fit, cannot perform fit")

    mask = extra_rows(n, cleaned_mask, geom)
    cleaned_image = image.copy()  
    cleaned_image[~mask] = 0.0
    cleaned_image[cleaned_image<0] = 0.0
    size = np.sum(cleaned_image)

    def fit(cog_x, cog_y, psi, length, width, skewness, amplitude):
        prediction = pdf_dict[pdf](cog_x*unit, cog_y*unit, length*unit, width*unit, psi*u.rad, skewness, amplitude).pdf(geom.pix_x, geom.pix_y)
        return neg_log_likelihood_approx(cleaned_image, prediction, spe_width, pedestal)

    def fit_gauss(cog_x, cog_y, psi, length, width, amplitude):
        prediction = pdf_dict[pdf](cog_x*unit, cog_y*unit, length*unit, width*unit, psi*u.rad, amplitude).pdf(geom.pix_x, geom.pix_y)
        return neg_log_likelihood_approx(cleaned_image, prediction, spe_width, pedestal)

    if pdf != "Gaussian":
        m = Minuit(fit, cog_x=x0['x'].value, cog_y=x0['y'].value, psi=x0['psi'].value, length=x0['length'].value, width=x0['width'].value, skewness=x0["skewness"], amplitude=x0["amplitude"])
    else:
        m = Minuit(fit_gauss, cog_x=x0['x'].value, cog_y=x0['y'].value, psi=x0['psi'].value, length=x0['length'].value, width=x0['width'].value, amplitude=x0["amplitude"])

    bounds = boundaries(geom, image, cleaned_mask, mask, x0, pdf)

    if bounds != None:
        m.limits = bounds
    
    m.errordef=1  #neg log likelihood
    m.migrad()
    m.hesse()

    likelihood = m.fval
    pars = m.values
    errors = m.errors

    fit_rcog = np.linalg.norm([pars[0], pars[1]])
    fit_phi = np.arctan2(pars[1], pars[0])

    b = pars[1]**2 + pars[0]**2
    A = (-pars[1]/(b))**2
    B = (pars[0]/(b))**2
    fit_phi_err = np.sqrt(A*errors[0]**2 + B*errors[1]**2)
    fit_rcog_err = np.sqrt(pars[0]**2/b*errors[0]**2 + pars[1]**2/b*errors[1]**2)

    delta_x = geom.pix_x.value - pars[0]
    delta_y = geom.pix_y.value - pars[1]

    longitudinal = delta_x * np.cos(pars[2]) + delta_y * np.sin(pars[2])

    m4_long = np.average(longitudinal**4, weights=cleaned_image)
    kurtosis_long = m4_long / pars[3]**4

    if pdf != "Gaussian":
        skewness_long = pars[5]
        amplitude=pars[6],
        amplitude_uncertainty=errors[6]
    else:
        m3_long = np.average(longitudinal**3, weights=image)
        skewness_long = m3_long / pars[3]**3 
        amplitude=pars[5],
        amplitude_uncertainty=errors[5]

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
        kurtosis=kurtosis_long,
        likelihood=likelihood,
        n_pix_fit=np.count_nonzero(cleaned_image),
        n_free_par=len(x0),
        is_valid=m.valid,
        is_accurate=m.accurate,
        )


