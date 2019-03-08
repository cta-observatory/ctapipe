from ctapipe.image.muon import muon_ring_finder
import numpy as np
import astropy.units as u
from ctapipe.instrument import CameraGeometry
from ctapipe.image import tailcuts_clean
from ctapipe.image.toymodel import RingGaussian


def test_ChaudhuriKunduRingFitter_old():

    fitter = muon_ring_finder.ChaudhuriKunduRingFitter()

    points = np.linspace(-100, 100, 200)

    x, y = np.meshgrid(points, points) * u.deg
    weight = np.zeros(x.shape)

    c_x = 50 * u.deg
    c_y = 20 * u.deg

    r = np.sqrt((x - c_x)**2 + (y - c_y)**2)

    min_r = 10 * u.deg
    max_r = 20 * u.deg

    weight[(r > min_r) & (r < max_r)] = 1
    output = fitter.fit(x, y, weight)

    lim_p = 0.05 * u.deg
    lim_r = 1 * u.deg
    rad_a = 0.5 * (max_r + min_r)

    assert abs(output.ring_center_x - c_x) < lim_p
    assert abs(output.ring_center_y - c_y) < lim_p
    assert abs(output.ring_radius - rad_a) < lim_r


def test_ChaudhuriKunduRingFitterHline():

    fitter = muon_ring_finder.ChaudhuriKunduRingFitter()

    x = np.linspace(20, 30, 10) * u.deg   # Make linear array in x
    y = np.full_like(x, 15)               # Fill y array of same size with y
    weight = np.ones(x.shape)           # Fill intensity array with value

    output = fitter.fit(x, y, weight)

    # TODO in muon_ring_fitter decide what to do if unreconstructable
    # ... add Status Flag?
    assert output.ring_radius is not np.NaN
    assert output.ring_phi is not np.NaN
    assert output.ring_inclination is not np.NaN


def test_ChaudhuriKunduRingFitter():

    geom = CameraGeometry.from_name('LSTCam')
    focal_length = u.Quantity(28, u.m)

    ring_radius = u.Quantity(0.4, u.m)  # make sure this is in camera coordinates
    ring_width = u.Quantity(0.03, u.m)
    center_x = u.Quantity(-0.2, u.m)
    center_y = u.Quantity(-0.3, u.m)

    muon_model = RingGaussian(
        x=center_x, y=center_y,
        sigma=ring_width, radius=ring_radius,
    )

    image, _, _ = muon_model.generate_image(
        geom, intensity=1000, nsb_level_pe=5,
    )

    clean_mask = tailcuts_clean(
        geom, image, boundary_thresh=5, picture_thresh=10
    )

    fitter = muon_ring_finder.ChaudhuriKunduRingFitter()

    x = geom.pix_x / focal_length * u.rad
    y = geom.pix_y / focal_length * u.rad

    # fit 3 times, first iteration use cleaning, after that use
    # distance to previous fit result
    result = None
    for _ in range(3):
        if result is None:
            mask = clean_mask
        else:
            dist = np.sqrt((x - result.ring_center_x)**2 + (y - result.ring_center_y)**2)
            ring_dist = np.abs(dist - result.ring_radius)
            mask = ring_dist < (result.ring_radius * 0.4)

        result = fitter.fit(x[mask], y[mask], image[mask])

    assert np.isclose(
        result.ring_radius.to_value(u.rad), ring_radius / focal_length,
        rtol=0.05,
    )
    assert np.isclose(
        result.ring_center_x.to_value(u.rad), center_x / focal_length,
        rtol=0.05
    )
    assert np.isclose(
        result.ring_center_y.to_value(u.rad), center_y / focal_length,
        rtol=0.05,
    )


if __name__ == '__main__':
    test_ChaudhuriKunduRingFitter_old()
    test_ChaudhuriKunduRingFitterHline()
    test_ChaudhuriKunduRingFitter()
