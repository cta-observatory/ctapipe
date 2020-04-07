import astropy.units as u
import numpy as np


def test_chord_length():
    from ctapipe.image.muon.muon_integrator import chord_length

    radius = 12
    rho = 0.0
    phi = 0

    length = chord_length(radius, rho, phi)
    assert length == radius

    rho = 1
    phi = np.deg2rad(90)
    length = chord_length(radius, rho, phi)
    assert np.isclose(length, 0, atol=1e-15)


if __name__ == '__main__':
    test_chord_length()
