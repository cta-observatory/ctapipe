import numpy as np
import astropy.units as u
from ctapipe.image import toymodel,tailcuts_clean
from ctapipe.instrument import CameraGeometry

from ctapipe.image.muon import kundu_chaudhuri_circle_fit
from ctapipe.image.muon.muon_ring_finder import TaubinFitter

np.random.seed(0)


def test_kundu_chaudhuri():

    num_tests = 10
    center_xs = np.random.uniform(-1000, 1000, num_tests)
    center_ys = np.random.uniform(-1000, 1000, num_tests)
    radii = np.random.uniform(10, 1000, num_tests)

    for center_x, center_y, radius in zip(center_xs, center_ys, radii):

        phi = np.random.uniform(0, 2 * np.pi, 100)
        x = center_x + radius * np.cos(phi)
        y = center_y + radius * np.sin(phi)

        weights = np.ones_like(x)

        fit_radius, fit_x, fit_y = kundu_chaudhuri_circle_fit(x, y, weights)

        assert np.isclose(fit_x, center_x)
        assert np.isclose(fit_y, center_y)
        assert np.isclose(fit_radius, radius)


def test_kundu_chaudhuri_with_units():

    center_x = 0.5 * u.meter
    center_y = 0.5 * u.meter
    radius = 1 * u.meter

    phi = np.random.uniform(0, 2 * np.pi, 100)
    x = center_x + radius * np.cos(phi)
    y = center_y + radius * np.sin(phi)

    weights = np.ones_like(x)

    fit_radius, fit_x, fit_y = kundu_chaudhuri_circle_fit(x, y, weights)

    assert fit_x.unit == center_x.unit
    assert fit_y.unit == center_y.unit
    assert fit_radius.unit == radius.unit


def test_taubin_with_units():
    """
    flashCam example
    for this test, values are selectively chosen knowing that they converge
    """
    center_xs = 0.3 * u.m
    center_ys = 0.6 * u.m
    ring_radius = 0.3 * u.m
    ring_width = 0.05 * u.m
    muon_model = toymodel.RingGaussian(
        x=center_xs,
        y=center_ys,
        radius=ring_radius,
        sigma=ring_width,
    )

    geom = CameraGeometry.from_name("FlashCam")
    flashcam_focal_length = u.Quantity(16, u.m)
    image, _, _ = muon_model.generate_image(
        geom, intensity=1000, nsb_level_pe=5,
    )
    mask = tailcuts_clean(geom, image, 10, 12)
    x = (geom.pix_x / flashcam_focal_length) * u.rad
    y = (geom.pix_y / flashcam_focal_length) * u.rad

    muon_ring_parameters = TaubinFitter.fit(
        pixx=x[mask],
        pixy=y[mask],
        radius=0.03,
        error=0.03,
        limit=(-0.0625, 0.0625),
    )
    xc_fit = muon_ring_parameters.ring_center_x
    yc_fit = muon_ring_parameters.ring_center_y
    r_fit = muon_ring_parameters.ring_radius

    assert np.isclose(xc_fit * flashcam_focal_length / u.m, center_xs / u.m, 1e-1)
    assert np.isclose(yc_fit * flashcam_focal_length / u.m, center_ys / u.m, 1e-1)
    assert np.isclose(r_fit * flashcam_focal_length / u.m, ring_radius / u.m, 1e-1)

