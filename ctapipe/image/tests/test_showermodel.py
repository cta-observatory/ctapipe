import astropy.units as u
import numpy as np
from pytest import approx
from scipy.integrate import dblquad, quad


def test_gaussian():
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
        x=x,
        y=y,
        azimuth=azimuth,
        altitude=altitude,
        h_max=h_max,
        width=width,
        length=length,
    )

    # integration over x and y axis
    zenith = 90 * u.deg - altitude
    trigs = np.cos(azimuth.to_value(u.rad)) * np.sin(
        zenith.to_value(u.rad)
    )  # only calculate trigonometric functions once since angle is 45 deg
    proj_h_max = h_max / np.cos(
        zenith.to_value(u.rad)
    )  # calculate radius on sphere where height/z equals h_max

    def integral(z):
        return dblquad(
            model.density,
            (proj_h_max * trigs).value - width.value,
            (proj_h_max * trigs).value + width.value,
            lambda x: 0,
            lambda x: 1,
            args=(z,),
        )

    zs = np.linspace(
        proj_h_max * np.cos(zenith.to_value(u.rad)) - 20 * u.meter,
        proj_h_max * np.cos(zenith.to_value(u.rad)) + 20 * u.meter,
        41,
    )

    # one dimensional distriubtion along z axis
    dist = np.array([integral(z.value)[0] for z in zs])
    assert zs[np.argmax(dist)].value == approx(
        proj_h_max.value * np.cos(zenith.to_value(u.rad)),
        rel=0.49,
    )
    assert model.barycenter.value == approx(
        np.array(
            [
                proj_h_max.value * trigs,
                proj_h_max.value * trigs,
                proj_h_max.value * np.cos(zenith.to_value(u.rad)),
            ]
        )
    )


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
        x=x,
        y=y,
        azimuth=azimuth,
        altitude=altitude,
        h_max=h_max,
        width=width,
        length=length,
    )

    assert (
        approx(
            quad(
                lambda x: 2
                * np.pi
                * model.emission_probability(np.array([x]) * u.rad).to_value(u.sr**-1)
                * x,
                0,
                np.pi,
            )[0],
            1e-2,
        )
        == 1
    )
