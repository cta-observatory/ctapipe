import numpy as np
from ctapipe.instrument import CameraGeometry, CameraReadout
from ctapipe.image.toymodel import obtain_time_image, WaveformModel
from pytest import approx
from scipy.stats import poisson, skewnorm, norm
import astropy.units as u


def test_intensity():
    from ctapipe.image.toymodel import Gaussian

    np.random.seed(0)
    geom = CameraGeometry.from_name("LSTCam")

    x, y = u.Quantity([0.2, 0.3], u.m)
    width = 0.05 * u.m
    length = 0.15 * u.m
    intensity = 50
    psi = "30d"

    # make a toymodel shower model
    model = Gaussian(x=x, y=y, width=width, length=length, psi=psi)

    _, signal, _ = model.generate_image(geom, intensity=intensity, nsb_level_pe=5,)

    # test if signal reproduces given cog values
    assert np.average(geom.pix_x.to_value(u.m), weights=signal) == approx(0.2, rel=0.15)
    assert np.average(geom.pix_y.to_value(u.m), weights=signal) == approx(0.3, rel=0.15)

    # test if signal reproduces given width/length values
    cov = np.cov(geom.pix_x.value, geom.pix_y.value, aweights=signal)
    eigvals, _ = np.linalg.eigh(cov)

    assert np.sqrt(eigvals[0]) == approx(width.to_value(u.m), rel=0.15)
    assert np.sqrt(eigvals[1]) == approx(length.to_value(u.m), rel=0.15)

    # test if total intensity is inside in 99 percent confidence interval
    assert poisson(intensity).ppf(0.05) <= signal.sum() <= poisson(intensity).ppf(0.95)


def test_skewed():
    from ctapipe.image.toymodel import SkewedGaussian

    # test if the parameters we calculated for the skew normal
    # distribution produce the correct moments
    np.random.seed(0)
    geom = CameraGeometry.from_name("LSTCam")

    x, y = u.Quantity([0.2, 0.3], u.m)
    width = 0.05 * u.m
    length = 0.15 * u.m
    intensity = 50
    psi = "30d"
    skewness = 0.3

    model = SkewedGaussian(
        x=x, y=y, width=width, length=length, psi=psi, skewness=skewness
    )
    model.generate_image(
        geom, intensity=intensity, nsb_level_pe=5,
    )

    a, loc, scale = model._moments_to_parameters()
    mean, var, skew = skewnorm(a=a, loc=loc, scale=scale).stats(moments="mvs")

    assert np.isclose(mean, 0)
    assert np.isclose(var, length.to_value(u.m) ** 2)
    assert np.isclose(skew, skewness)


def test_compare():
    from ctapipe.image.toymodel import SkewedGaussian, Gaussian

    geom = CameraGeometry.from_name("LSTCam")

    x, y = u.Quantity([0.2, 0.3], u.m)
    width = 0.05 * u.m
    length = 0.15 * u.m
    intensity = 50
    psi = "30d"

    skewed = SkewedGaussian(x=x, y=y, width=width, length=length, psi=psi, skewness=0)
    normal = Gaussian(x=x, y=y, width=width, length=length, psi=psi)

    signal_skewed = skewed.expected_signal(geom, intensity=intensity)
    signal_normal = normal.expected_signal(geom, intensity=intensity)

    assert np.isclose(signal_skewed, signal_normal).all()


def test_obtain_time_image():
    geom = CameraGeometry.from_name("CHEC")
    centroid_x = u.Quantity(0.05, u.m)
    centroid_y = u.Quantity(0.05, u.m)
    psi = u.Quantity(70, u.deg)

    time_gradient = u.Quantity(0, u.ns / u.m)
    time_intercept = u.Quantity(0, u.ns)
    time = obtain_time_image(
        geom.pix_x,
        geom.pix_y,
        centroid_x,
        centroid_y,
        psi,
        time_gradient,
        time_intercept,
    )
    np.testing.assert_allclose(time, 0)

    time_gradient = u.Quantity(0, u.ns / u.m)
    time_intercept = u.Quantity(40, u.ns)
    time = obtain_time_image(
        geom.pix_x,
        geom.pix_y,
        centroid_x,
        centroid_y,
        psi,
        time_gradient,
        time_intercept,
    )
    np.testing.assert_allclose(time, 40)

    time_gradient = u.Quantity(20, u.ns / u.m)
    time_intercept = u.Quantity(40, u.ns)
    time = obtain_time_image(
        geom.pix_x,
        geom.pix_y,
        centroid_x,
        centroid_y,
        psi,
        time_gradient,
        time_intercept,
    )
    np.testing.assert_allclose(time.std(), 1.710435)

    time_gradient = u.Quantity(20, u.ns / u.m)
    time_intercept = u.Quantity(40, u.ns)
    time = obtain_time_image(
        centroid_x,
        centroid_y,
        centroid_x,
        centroid_y,
        psi,
        time_gradient,
        time_intercept,
    )
    np.testing.assert_allclose(time, 40)


def test_waveform_model():
    from ctapipe.image.toymodel import Gaussian

    geom = CameraGeometry.from_name("CHEC")
    readout = CameraReadout.from_name("CHEC")

    ref_duration = 67
    n_ref_samples = 100
    pulse_sigma = 3
    ref_x_norm = np.linspace(0, ref_duration, n_ref_samples)
    ref_y_norm = norm.pdf(ref_x_norm, ref_duration / 2, pulse_sigma)

    readout.reference_pulse_shape = ref_y_norm[np.newaxis, :]
    readout.reference_pulse_sample_width = u.Quantity(
        ref_x_norm[1] - ref_x_norm[0], u.ns
    )
    readout.sampling_rate = u.Quantity(2, u.GHz)

    centroid_x = u.Quantity(0.05, u.m)
    centroid_y = u.Quantity(0.05, u.m)
    length = u.Quantity(0.03, u.m)
    width = u.Quantity(0.008, u.m)
    psi = u.Quantity(70, u.deg)
    time_gradient = u.Quantity(50, u.ns / u.m)
    time_intercept = u.Quantity(20, u.ns)

    _, charge, _ = Gaussian(
        x=centroid_x, y=centroid_y, width=width, length=length, psi=psi
    ).generate_image(geom, 10000)
    time = obtain_time_image(
        geom.pix_x,
        geom.pix_y,
        centroid_x,
        centroid_y,
        psi,
        time_gradient,
        time_intercept,
    )
    time[charge == 0] = 0

    waveform_model = WaveformModel.from_camera_readout(readout)
    waveform = waveform_model.get_waveform(charge, time, 96)
    np.testing.assert_allclose(waveform.sum(axis=1), charge, rtol=1e-3)
    np.testing.assert_allclose(
        waveform.argmax(axis=1) / readout.sampling_rate.to_value(u.GHz), time, rtol=1e-1
    )

    time_2 = time + 1
    time_2[charge == 0] = 0
    waveform_2 = waveform_model.get_waveform(charge, time_2, 96)
    np.testing.assert_allclose(waveform_2.sum(axis=1), charge, rtol=1e-3)
    np.testing.assert_allclose(
        waveform_2.argmax(axis=1) / readout.sampling_rate.to_value(u.GHz),
        time_2,
        rtol=1e-1,
    )
    assert (
        waveform_2.argmax(axis=1)[charge != 0] > waveform.argmax(axis=1)[charge != 0]
    ).all()
