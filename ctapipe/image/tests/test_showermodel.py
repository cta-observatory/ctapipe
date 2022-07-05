import astropy.units as u
from scipy.integrate import dblquad
import numpy as np
from pytest import approx


def test_gaussian():
    from ctapipe.image import showermodel

    # This is a shower straight from (45deg,45deg)
    total_photons = 15000
    x = 0 * u.meter
    y = 0 * u.meter
    phi = 45 * u.deg
    theta = 45 * u.deg
    first_interaction = 20000 * u.meter
    width = 10 * u.meter
    length = 3000 * u.meter

    model = showermodel.Gaussian(
        total_photons=total_photons,
        x=x,
        y=y,
        phi=phi,
        theta=theta,
        first_interaction=first_interaction,
        width=width,
        length=length,
    )

    # integration over x and y axis
    trigs = np.cos(phi.to_value(u.rad)) * np.sin(
        theta.to_value(u.rad)
    )  # only calculate trigonometric functions once since angle is 45 deg
    proj_first_inter = first_interaction / np.cos(
        theta.to_value(u.rad)
    )  # calculate radius on sphere where height/z equals first_interaction

    def integral(z):
        return dblquad(
            model.density,
            ((proj_first_inter - length / 2) * trigs).value - width.value,
            ((proj_first_inter - length / 2) * trigs).value + width.value,
            lambda x: 0,
            lambda x: 1,
            args=(z,),
        )

    zs = np.linspace(
        (proj_first_inter - length / 2) * np.cos(theta.to_value(u.rad)) - 20 * u.meter,
        (proj_first_inter - length / 2) * np.cos(theta.to_value(u.rad)) + 20 * u.meter,
        41,
    )

    # one dimensional distriubtion along z axis
    dist = np.array([integral(z.value)[0] for z in zs])
    assert zs[np.argmax(dist)].value == approx(
        (proj_first_inter.value - length.value / 2) * np.cos(theta.to_value(u.rad)),
        rel=0.49,
    )
    assert model.barycenter.value == approx(
        np.array(
            [
                (proj_first_inter.value - length.value / 2) * trigs,
                (proj_first_inter.value - length.value / 2) * trigs,
                (proj_first_inter.value - length.value / 2)
                * np.cos(theta.to_value(u.rad)),
            ]
        )
    )
