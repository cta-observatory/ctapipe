from ctapipe.image.muon import muon_ring_finder
import numpy as np
import astropy.units as u
from ctapipe.instrument import CameraGeometry
from functools import partial
from ctapipe.image import toymodel, tailcuts_clean


def test_ChaudhuriKunduRingFitter_old():

    fitter = muon_ring_finder.ChaudhuriKunduRingFitter(parent=None)

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

    fitter = muon_ring_finder.ChaudhuriKunduRingFitter(parent=None)

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

    geom = CameraGeometry.from_name('HESS-I')

    ring_rad = np.deg2rad(1. * u.deg) * 15.  # make sure this is in camera coordinates
    ring_width = np.deg2rad(0.05 * u.deg) * 15.
    geom_pixall = np.empty(geom.pix_x.shape + (2,))
    geom_pixall[..., 0] = geom.pix_x.value
    geom_pixall[..., 1] = geom.pix_y.value

    # image = generate_muon_model(geom_pixall, ring_rad, ring_width, 0.3, 0.2)
    muon_model = partial(toymodel.generate_muon_model, radius=ring_rad.value,
                         width=ring_width.value, centre_x=-0.2, centre_y=-0.3)

    toymodel_image, toy_signal, toy_noise = \
        toymodel.make_toymodel_shower_image(geom, muon_model)

    clean_toy_mask = tailcuts_clean(geom, toymodel_image,
                                    boundary_thresh=5, picture_thresh=10)

    muonring = muon_ring_finder.ChaudhuriKunduRingFitter(None)

    x = np.rad2deg((geom.pix_x.value / 15.) * u.rad)  # .value
    y = np.rad2deg((geom.pix_y.value / 15.) * u.rad)  # .value

    muonringparam = muonring.fit(x, y, toymodel_image * clean_toy_mask)

    dist = np.sqrt(np.power(x - muonringparam.ring_center_x, 2)
                   + np.power(y - muonringparam.ring_center_y, 2))
    ring_dist = np.abs(dist - muonringparam.ring_radius)
    muonringparam = muonring.fit(x, y, toymodel_image * (ring_dist <
                                                         muonringparam.ring_radius * 0.4))

    dist = np.sqrt(np.power(x - muonringparam.ring_center_x, 2) +
                   np.power(y - muonringparam.ring_center_y, 2))
    ring_dist = np.abs(dist - muonringparam.ring_radius)
    muonringparam = muonring.fit(x, y, toymodel_image * (ring_dist <
                                                         muonringparam.ring_radius * 0.4))

    print('Fitted ring radius', muonringparam.ring_radius, 'c.f.', ring_rad)
    print('Fitted ring centre', muonringparam.ring_center_x, muonringparam.ring_center_y)

    assert muonringparam.ring_radius is not ring_rad  # .value
    assert muonringparam.ring_center_x is not -0.2
    assert muonringparam.ring_center_y is not -0.3

if __name__ == '__main__':
    test_ChaudhuriKunduRingFitter_old()
    test_ChaudhuriKunduRingFitterHline()
    test_ChaudhuriKunduRingFitter()
