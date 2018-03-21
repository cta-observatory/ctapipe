import numpy as np
import astropy.units as u

from ctapipe.image.muon import muon_integrator


def test_chord_length():
    radius = 12 * u.m
    rho = 0.1
    phi = np.asarray([0, 10, 20, 30]) * u.degree
    chord_length = muon_integrator.MuonLineIntegrate.chord_length(radius, rho, phi)
    assert(chord_length is not np.nan)

if __name__ == '__main__':
    test_chord_length()
