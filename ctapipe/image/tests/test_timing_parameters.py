from ctapipe.image.timing_parameters import timing_parameters
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose
from ctapipe.instrument.camera import CameraGeometry
from ctapipe.io.containers import HillasParametersContainer


def test_psi_0():
    """
    Simple test that gradient fitting gives expected answers for perfect
    gradient
    """
    grad = 2.0
    intercept = 1.0

    geom = CameraGeometry.from_name("LSTCam")
    hillas = HillasParametersContainer(x=0 * u.m, y=0 * u.m, psi=0 * u.deg)

    timing = timing_parameters(
        geom,
        image=np.ones(geom.n_pixels),
        peakpos=intercept + grad * geom.pix_x.value,
        hillas_parameters=hillas,
    )

    # Test we get the values we put in back out again
    assert_allclose(timing.slope, grad / geom.pix_x.unit)
    assert_allclose(timing.intercept, intercept)


def test_psi_20():

    # Then try a different rotation angle
    grad = 2
    intercept = 1

    geom = CameraGeometry.from_name("LSTCam")
    psi = 20 * u.deg
    hillas = HillasParametersContainer(x=0 * u.m, y=0 * u.m, psi=psi)

    timing = timing_parameters(
        geom,
        image=np.ones(geom.n_pixels),
        peakpos=intercept + grad * (np.cos(psi) * geom.pix_x.value
                                    + np.sin(psi) * geom.pix_y.value),
        hillas_parameters=hillas,
    )

    # Test we get the values we put in back out again
    assert_allclose(timing.slope, grad / geom.pix_x.unit)
    assert_allclose(timing.intercept, intercept)


def test_ignore_negative():
    grad = 2.0
    intercept = 1.0

    geom = CameraGeometry.from_name("LSTCam")
    hillas = HillasParametersContainer(x=0 * u.m, y=0 * u.m, psi=0 * u.deg)

    image = np.ones(geom.n_pixels)
    image[5:10] = -1.0

    timing = timing_parameters(
        geom,
        image,
        peakpos=intercept + grad * geom.pix_x.value,
        hillas_parameters=hillas,
    )

    # Test we get the values we put in back out again
    assert_allclose(timing.slope, grad / geom.pix_x.unit)
    assert_allclose(timing.intercept, intercept)
