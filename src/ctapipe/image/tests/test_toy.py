from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
from pytest import approx
from scipy.stats import norm, poisson, skewnorm

from ctapipe.coordinates.telescope_frame import TelescopeFrame
from ctapipe.image.toymodel import WaveformModel, obtain_time_image


@pytest.mark.parametrize("frame", ["telescope", "camera"])
@pytest.mark.parametrize("seed", [None, 0])
def test_intensity(seed, frame, monkeypatch, prod5_lst):
    """
    Test generation of the toymodel roughly follows the given intensity.

    Tests once with passing a custom rng instance, once with relying on the
    modules rng.
    """
    from ctapipe.image import toymodel

    if frame == "camera":
        geom = prod5_lst.camera.geometry
        unit = u.m
    else:
        geom = prod5_lst.camera.geometry.transform_to(TelescopeFrame())
        unit = u.deg

    x, y = u.Quantity([0.2, 0.3], unit)
    width = 0.05 * unit
    length = 0.15 * unit

    intensity = 200
    psi = 30 * u.deg

    # make sure we set a fixed seed for this test even when testing the
    # API without giving the rng
    monkeypatch.setattr(toymodel, "TOYMODEL_RNG", np.random.default_rng(0))

    # make a toymodel shower model
    model = toymodel.Gaussian(x=x, y=y, width=width, length=length, psi=psi)

    if seed is None:
        _, signal, _ = model.generate_image(geom, intensity=intensity, nsb_level_pe=5)
    else:
        rng = np.random.default_rng(seed)
        _, signal, _ = model.generate_image(
            geom, intensity=intensity, nsb_level_pe=5, rng=rng
        )

    # test if signal reproduces given cog values
    assert np.average(geom.pix_x.to_value(unit), weights=signal) == approx(
        0.2, rel=0.15
    )
    assert np.average(geom.pix_y.to_value(unit), weights=signal) == approx(
        0.3, rel=0.15
    )

    # test if signal reproduces given width/length values
    cov = np.cov(geom.pix_x.value, geom.pix_y.value, aweights=signal)
    eigvals, _ = np.linalg.eigh(cov)

    assert np.sqrt(eigvals[0]) == approx(width.to_value(unit), rel=0.15)
    assert np.sqrt(eigvals[1]) == approx(length.to_value(unit), rel=0.15)

    # test if total intensity is inside in 99 percent confidence interval
    assert poisson(intensity).ppf(0.05) <= signal.sum() <= poisson(intensity).ppf(0.95)


@pytest.mark.parametrize("frame", ["telescope", "camera"])
def test_skewed(frame, prod5_lst):
    from ctapipe.image.toymodel import SkewedGaussian

    # test if the parameters we calculated for the skew normal
    # distribution produce the correct moments
    rng = np.random.default_rng(0)
    if frame == "camera":
        geom = prod5_lst.camera.geometry
        unit = u.m
    else:
        geom = prod5_lst.camera.geometry.transform_to(TelescopeFrame())
        unit = u.deg

    x, y = u.Quantity([0.2, 0.3], unit)
    width = 0.05 * unit
    length = 0.15 * unit
    intensity = 50
    psi = 30 * u.deg
    skewness = 0.3

    model = SkewedGaussian(
        x=x, y=y, width=width, length=length, psi=psi, skewness=skewness
    )
    model.generate_image(geom, intensity=intensity, nsb_level_pe=5, rng=rng)

    a, loc, scale = model._moments_to_parameters()
    mean, var, skew = skewnorm(a=a, loc=loc, scale=scale).stats(moments="mvs")

    assert np.isclose(mean, 0)
    assert np.isclose(var, length.to_value(unit) ** 2)
    assert np.isclose(skew, skewness)


@pytest.mark.parametrize("frame", ["telescope", "camera"])
def test_compare(frame, prod5_lst):
    from ctapipe.image.toymodel import Gaussian, SkewedGaussian

    if frame == "camera":
        geom = prod5_lst.camera.geometry
        unit = u.m
    else:
        geom = prod5_lst.camera.geometry.transform_to(TelescopeFrame())
        unit = u.deg

    x, y = u.Quantity([0.2, 0.3], unit)
    width = 0.05 * unit
    length = 0.15 * unit
    intensity = 50
    psi = 30 * u.deg

    skewed = SkewedGaussian(x=x, y=y, width=width, length=length, psi=psi, skewness=0)
    normal = Gaussian(x=x, y=y, width=width, length=length, psi=psi)

    signal_skewed = skewed.expected_signal(geom, intensity=intensity)
    signal_normal = normal.expected_signal(geom, intensity=intensity)

    assert np.isclose(signal_skewed, signal_normal).all()


@pytest.mark.parametrize("frame", ["telescope", "camera"])
def test_obtain_time_image(frame, prod5_sst):
    geom = prod5_sst.camera.geometry

    if frame == "camera":
        geom = prod5_sst.camera.geometry
        unit = u.m
        scale = 1.0
    else:
        geom_cam_frame = prod5_sst.camera.geometry
        geom = geom_cam_frame.transform_to(TelescopeFrame())
        unit = u.deg
        # further down we test the std deviation, but that scales
        # with the size of our shower in the camera
        r_tel_frame = geom.guess_radius().to_value(u.deg)
        r_cam_frame = geom_cam_frame.guess_radius().to_value(u.m)
        scale = r_tel_frame / r_cam_frame

    centroid_x = u.Quantity(0.05, unit)
    centroid_y = u.Quantity(0.05, unit)
    psi = u.Quantity(70, u.deg)

    time_gradient = u.Quantity(0, u.ns / unit)
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

    time_gradient = u.Quantity(0, u.ns / unit)
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

    time_gradient = u.Quantity(20 / scale, u.ns / unit)
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
    np.testing.assert_allclose(time.std(), 1.710435, rtol=0.1)

    time_gradient = u.Quantity(20, u.ns / unit)
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


@pytest.mark.parametrize("frame", ["telescope", "camera"])
def test_waveform_model(frame, prod5_sst):
    from ctapipe.image.toymodel import Gaussian

    prod5_sst = deepcopy(prod5_sst)
    readout = prod5_sst.camera.readout

    if frame == "camera":
        geom = prod5_sst.camera.geometry
        unit = u.m
    else:
        geom = prod5_sst.camera.geometry.transform_to(TelescopeFrame())
        unit = u.deg

    ref_duration = 67
    n_ref_samples = 100
    pulse_sigma = 3
    ref_x_norm = np.linspace(0, ref_duration, n_ref_samples)
    ref_y_norm = norm.pdf(ref_x_norm, ref_duration / 2, pulse_sigma)

    readout.reference_pulse_shape = np.array([ref_y_norm, ref_y_norm])
    readout.reference_pulse_sample_width = u.Quantity(
        ref_x_norm[1] - ref_x_norm[0], u.ns
    )
    readout.sampling_rate = u.Quantity(2, u.GHz)

    centroid_x = u.Quantity(0.05, unit)
    centroid_y = u.Quantity(0.05, unit)
    length = u.Quantity(0.03, unit)
    width = u.Quantity(0.008, unit)
    psi = u.Quantity(70, u.deg)
    time_gradient = u.Quantity(50, u.ns / unit)
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
    assert waveform.shape[-3] == 2
    np.testing.assert_allclose(waveform.sum(axis=-1)[0], charge, rtol=1e-3)
    np.testing.assert_allclose(
        waveform.argmax(axis=-1)[0] / readout.sampling_rate.to_value(u.GHz),
        time,
        rtol=1e-1,
    )

    time_2 = time + 1
    time_2[charge == 0] = 0
    waveform_2 = waveform_model.get_waveform(charge, time_2, 96)
    np.testing.assert_allclose(waveform_2.sum(axis=-1)[0], charge, rtol=1e-3)
    np.testing.assert_allclose(
        waveform_2.argmax(axis=-1)[0] / readout.sampling_rate.to_value(u.GHz),
        time_2,
        rtol=1e-1,
    )
    assert (
        waveform_2.argmax(axis=-1)[0, charge != 0]
        > waveform.argmax(axis=-1)[0, charge != 0]
    ).all()

    waveform_model = WaveformModel.from_camera_readout(readout, gain_channel="HIGH")
    waveform = waveform_model.get_waveform(charge, time, 96)
    assert waveform.shape[-3] == 1

    with pytest.raises(ValueError):
        WaveformModel.from_camera_readout(readout, gain_channel=0)
