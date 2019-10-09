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
    deviation = 0.1

    geom = CameraGeometry.from_name("LSTCam")
    hillas = HillasParametersContainer(x=0 * u.m, y=0 * u.m, psi=0 * u.deg)

    random = np.random.RandomState(1)
    pulse_time = intercept + grad * geom.pix_x.value
    pulse_time += random.normal(0, deviation, geom.n_pixels)

    timing = timing_parameters(
        geom,
        image=np.ones(geom.n_pixels),
        pulse_time=pulse_time,
        hillas_parameters=hillas,
        cleaning_mask=np.ones(geom.n_pixels, dtype=bool)
    )

    # Test we get the values we put in back out again
    assert_allclose(timing.slope, grad / geom.pix_x.unit, rtol=1e-2)
    assert_allclose(timing.intercept, intercept, rtol=1e-2)
    assert_allclose(timing.deviation, deviation, rtol=1e-2)


def test_psi_20():

    # Then try a different rotation angle
    grad = 2
    intercept = 1
    deviation = 0.1

    geom = CameraGeometry.from_name("LSTCam")
    psi = 20 * u.deg
    hillas = HillasParametersContainer(x=0 * u.m, y=0 * u.m, psi=psi)

    random = np.random.RandomState(1)
    pulse_time = intercept + grad * (np.cos(psi) * geom.pix_x.value
                                     + np.sin(psi) * geom.pix_y.value)
    pulse_time += random.normal(0, deviation, geom.n_pixels)

    timing = timing_parameters(
        geom,
        image=np.ones(geom.n_pixels),
        pulse_time=pulse_time,
        hillas_parameters=hillas,
        cleaning_mask=np.ones(geom.n_pixels, dtype=bool)
    )

    # Test we get the values we put in back out again
    assert_allclose(timing.slope, grad / geom.pix_x.unit, rtol=1e-2)
    assert_allclose(timing.intercept, intercept, rtol=1e-2)
    assert_allclose(timing.deviation, deviation, rtol=1e-2)


def test_ignore_negative():
    grad = 2.0
    intercept = 1.0
    deviation = 0.1

    geom = CameraGeometry.from_name("LSTCam")
    hillas = HillasParametersContainer(x=0 * u.m, y=0 * u.m, psi=0 * u.deg)

    random = np.random.RandomState(1)
    pulse_time = intercept + grad * geom.pix_x.value
    pulse_time += random.normal(0, deviation, geom.n_pixels)

    image = np.ones(geom.n_pixels)
    image[5:10] = -1.0

    cleaning_mask = image >= 0

    timing = timing_parameters(
        geom,
        image,
        pulse_time=pulse_time,
        hillas_parameters=hillas,
        cleaning_mask=cleaning_mask,
    )

    # Test we get the values we put in back out again
    assert_allclose(timing.slope, grad / geom.pix_x.unit, rtol=1e-2)
    assert_allclose(timing.intercept, intercept, rtol=1e-2)
    assert_allclose(timing.deviation, deviation, rtol=1e-2)
