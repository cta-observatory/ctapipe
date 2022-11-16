import astropy.units as u
import numpy as np
from pytest import approx
from scipy.integrate import quad


def test_emission():
    from ctapipe.image import GaussianShowermodel

    # This is a shower straight from (45deg,45deg)
    total_photons = 15000
    x = 0 * u.meter
    y = 0 * u.meter
    azimuth = 45 * u.deg
    altitude = 45 * u.deg
    h_max = 17000 * u.meter
    width = 10 * u.meter
    length = 3000 * u.meter

    model = GaussianShowermodel(
        total_photons=total_photons,
        x=x.to_value(u.m),
        y=y.to_value(u.m),
        azimuth=azimuth.to_value(u.rad),
        altitude=altitude.to_value(u.rad),
        h_max=h_max.to_value(u.m),
        width=width.to_value(u.m),
        length=length.to_value(u.m),
    )

    assert (
        approx(
            quad(
                lambda x: 2 * np.pi * model.emission_probability(np.array([x])) * x,
                0,
                np.pi,
            )[0],
            1e-2,
        )
        == 1
    )
