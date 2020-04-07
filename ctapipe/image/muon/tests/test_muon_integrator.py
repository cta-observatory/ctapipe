import astropy.units as u


def test_chord_length():
    from ctapipe.image.muon.muon_integrator import chord_length

    radius = 12 * u.m
    rho = 0.0
    phi = 0 * u.deg

    length = chord_length(radius, rho, phi)
    assert length == radius

    rho = 1
    phi = 90 * u.deg
    length = chord_length(radius, rho, phi)
    assert u.isclose(length, 0 * u.m, atol=1e-15 * u.m)


if __name__ == '__main__':
    test_chord_length()
