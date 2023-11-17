import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose

from ctapipe.containers import CameraHillasParametersContainer


def test_psi_0(prod5_lst):
    from ctapipe.image import timing_parameters

    """
    Simple test that gradient fitting gives expected answers for perfect
    gradient
    """
    grad = 2.0
    intercept = 1.0
    deviation = 0.1

    geom = prod5_lst.camera.geometry
    hillas = CameraHillasParametersContainer(x=0 * u.m, y=0 * u.m, psi=0 * u.deg)

    random = np.random.default_rng(0)
    peak_time = intercept + grad * geom.pix_x.value
    peak_time += random.normal(0, deviation, geom.n_pixels)

    timing = timing_parameters(
        geom,
        image=np.ones(geom.n_pixels),
        peak_time=peak_time,
        hillas_parameters=hillas,
        cleaning_mask=np.ones(geom.n_pixels, dtype=bool),
    )

    # Test we get the values we put in back out again
    assert_allclose(timing.slope, grad / geom.pix_x.unit, rtol=1e-2)
    assert_allclose(timing.intercept, intercept, rtol=1e-2)
    assert_allclose(timing.deviation, deviation, rtol=1e-2)


def test_psi_20(prod5_lst):
    from ctapipe.image import timing_parameters

    # Then try a different rotation angle
    grad = 2
    intercept = 1
    deviation = 0.1

    geom = prod5_lst.camera.geometry
    psi = 20 * u.deg
    hillas = CameraHillasParametersContainer(x=0 * u.m, y=0 * u.m, psi=psi)

    random = np.random.default_rng(0)
    peak_time = intercept + grad * (
        np.cos(psi) * geom.pix_x.value + np.sin(psi) * geom.pix_y.value
    )
    peak_time += random.normal(0, deviation, geom.n_pixels)

    timing = timing_parameters(
        geom,
        image=np.ones(geom.n_pixels),
        peak_time=peak_time,
        hillas_parameters=hillas,
        cleaning_mask=np.ones(geom.n_pixels, dtype=bool),
    )

    # Test we get the values we put in back out again
    assert_allclose(timing.slope, grad / geom.pix_x.unit, rtol=1e-2)
    assert_allclose(timing.intercept, intercept, rtol=1e-2)
    assert_allclose(timing.deviation, deviation, rtol=1e-2)


def test_ignore_negative(prod5_lst):
    from ctapipe.image import timing_parameters

    grad = 2.0
    intercept = 1.0
    deviation = 0.1

    geom = prod5_lst.camera.geometry
    hillas = CameraHillasParametersContainer(x=0 * u.m, y=0 * u.m, psi=0 * u.deg)

    random = np.random.default_rng(0)
    peak_time = intercept + grad * geom.pix_x.value
    peak_time += random.normal(0, deviation, geom.n_pixels)

    image = np.ones(geom.n_pixels)
    image[5:10] = -1.0

    cleaning_mask = image >= 0

    timing = timing_parameters(
        geom,
        image,
        peak_time=peak_time,
        hillas_parameters=hillas,
        cleaning_mask=cleaning_mask,
    )

    # Test we get the values we put in back out again
    assert_allclose(timing.slope, grad / geom.pix_x.unit, rtol=1e-2)
    assert_allclose(timing.intercept, intercept, rtol=1e-2)
    assert_allclose(timing.deviation, deviation, rtol=1e-2)
