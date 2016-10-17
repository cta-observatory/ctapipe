from ctapipe.image.muon import muon_ring_finder
import numpy as np
import astropy.units as u


def test_ChaudhuriKunduRingFitter():

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
    rad_a = 0.5*(max_r+min_r)

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
