# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: UTF-8 -*-
"""
Ellipsoid-style image fitting based shower image parametrization.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from ctapipe.image.cleaning import dilate
import scipy.optimize as opt
from iminuit import Minuit
from ctapipe.image.pixel_likelihood import chi_squared, neg_log_likelihood_approx
from ctapipe.image.hillas import hillas_parameters, camera_to_shower_coordinates
from ctapipe.image.toymodel import SkewedCauchy, SkewedGaussian
from ..containers import CameraImageFitParametersContainer, ImageFitParametersContainer

__all__ = ["image_fit_parameters", "ImageFitParameterizationError"]

def create_initial_guess(geometry, image):
    """
    This function computes the initial guess for the fit using the Hillas parameters
    
    Parameters
    ----------
    geometry : ctapipe.instrument.CameraGeometry
        Camera geometry, the cleaning mask should be applied to improve performance
    image : array_like
        Charge in each pixel, the cleaning mask should already be applied to
        improve performance.

    Returns
    -------
    initial_guess : initial Hillas parameters 
    """
    hillas = hillas_parameters(geometry, image)

    initial_guess = {}
    initial_guess["x"] = hillas.x
    initial_guess["y"] = hillas.y
    initial_guess["length"] = hillas.length
    initial_guess["width"] = hillas.width
    initial_guess["psi"] = hillas.psi
    initial_guess["skewness"] = hillas.skewness

    return initial_guess

def extra_rows(n, cleaned_mask, geometry):
    """
    Parameters
    ----------
    n : int
       number of extra rows to add after cleaning
    cleaned_mask : boolean
       The cleaning mask applied for Hillas parametrization
    geometry : ctapipe.instrument.CameraGeometry
        Camera geometry, the cleaning mask should be applied to improve performance

    """
    mask = cleaned_mask.copy()
    for ii in range(n):
        mask = dilate(geometry, mask)

    mask = np.array((mask.astype(int) + cleaned_mask.astype(int)), dtype=bool)

    return mask

#def boundaries():


def image_fit_parameters(geom, image, bounds, n, cleaned_mask, spe_width, pedestal):
    """
    Computes image parameters for a given shower image.

    Implementation analogous to https://arxiv.org/pdf/1211.0254.pdf

    Parameters
    ----------
    geom : ctapipe.instrument.CameraGeometry
        Camera geometry, the cleaning mask should be applied to improve performance
    image : array_like
        Charge in each pixel, the cleaning mask should already be applied to
        improve performance.
    bounds : default format [(low_limx, high_limx), (low_limy, high_limy), ...]
        Parameters boundary condition
    n : int
      number of extra rows after cleaning
    cleaned_mask : boolean
       The cleaning mask applied for Hillas parametrization
    spe_width: ndarray
        Width of single p.e. peak (:math:`σ_γ`).
    pedestal: ndarray
        Width of pedestal (:math:`σ_p`).

    Returns
    -------
    ImageFitParametersContainer:
        container of image-fitting parameters
    """
    unit = geom.pix_x.unit
    pix_x = geom.pix_x
    pix_y = geom.pix_y
    image = np.asanyarray(image, dtype=np.float64)

    if isinstance(image, np.ma.masked_array):
        image = np.ma.filled(image, 0)

    if not (pix_x.shape == pix_y.shape == image.shape):
        raise ValueError("Image and pixel shape do not match")

    size = np.sum(image)

    x0 = create_initial_guess(geom, image)

    if size == 0:
        raise ImageFitParameterizationError("size=0, cannot calculate ImageFitParameters")

    if np.count_nonzero(image) <= len(x0):
        raise ImageFitParameterizationError("The number of free parameters is higher than the number of pixels to fit, cannot perform fit")

    mask = extra_rows(n, cleaned_mask, geom)
    cleaned_image = image.copy()  
    cleaned_image[~mask] = 0.0
    cleaned_image[cleaned_image<0] = 0.0
    size = np.sum(cleaned_image)

    def fit(cog_x, cog_y, psi, length, width, skewness, amplitude):
        prediction = size * SkewedCauchy(cog_x*unit, cog_y*unit, length*unit, width*unit, psi*u.rad, skewness).pdf(geom.pix_x, geom.pix_y)
        return neg_log_likelihood_approx(cleaned_image, prediction, spe_width, pedestal)

    m = Minuit(fit, cog_x=x0['x'].value, cog_y=x0['y'].value, psi=x0['psi'].value, length=x0['length'].value, width=x0['width'].value, skewness=x0['skewness'], amplitude=size)

    if bounds != None:
        m.limits = bounds
    
    m.errordef=1  #neg log likelihood
    m.simplex().migrad()
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

    cov = np.cov(delta_x, delta_y, aweights=cleaned_image, ddof=0)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    longitudinal = delta_x * np.cos(pars[2]) + delta_y * np.sin(pars[2])

    m4_long = np.average(longitudinal**4, weights=cleaned_image)
    kurtosis_long = m4_long / pars[3]**4
    skewness_long = pars[5]

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
            length=u.Quantity(pars[3], unit),
            length_uncertainty=u.Quantity(errors[3], unit),
            width=u.Quantity(pars[4], unit),
            width_uncertainty=u.Quantity(errors[4], unit),
            psi=Angle(pars[2], unit=u.rad),
            psi_uncertainty=Angle(errors[2], unit=u.rad),
            skewness=skewness_long,
            skewness_uncertainty=errors[5],
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
        length=u.Quantity(pars[3], unit),
        length_uncertainty=u.Quantity(errors[3], unit),
        width=u.Quantity(pars[4], unit),
        width_uncertainty=u.Quantity(errors[4], unit),
        psi=Angle(pars[2], unit=u.rad),
        psi_uncertainty=Angle(errors[2], unit=u.rad),
        skewness=skewness_long,
        skewness_uncertainty=errors[5],
        kurtosis=kurtosis_long,
        likelihood=likelihood,
        n_pix_fit=np.count_nonzero(cleaned_image),
        n_free_par=len(x0),
        is_valid=m.valid,
        is_accurate=m.accurate,
        )


