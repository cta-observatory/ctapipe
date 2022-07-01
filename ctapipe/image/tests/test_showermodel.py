import astropy.units as u
from scipy.integrate import dblquad
import numpy as np
from pytest import approx


def test_gaussian():
    from ctapipe.image import showermodel

    # This is a shower straight from above
    Nc = 15000  # cherenkov photons
    x = 0 * u.meter  # position of intersection on ground
    y = 0 * u.meter  # position of intersection on ground
    phi = 0 * u.deg  # phi orientation spherical coords
    theta = 0 * u.deg  # theta orientation spherical coords
    h_bary = 20000 * u.meter  # height of the barycenter
    width = 10 * u.meter  # width of shower
    length = 3000 * u.meter  # length of shower

    model = showermodel.Gaussian(
        Nc=Nc, x=x, y=y, phi=phi, theta=theta, h_bary=h_bary, width=width, length=length
    )

    # integration over x and z axis
    def integral(z):
        return dblquad(
            model.density,
            -width.value,
            width.value,
            lambda x: 0,
            lambda x: 1,
            args=(z,),
        )

    zs = np.linspace(h_bary - 20 * u.meter, h_bary + 20 * u.meter, 41)
    print(zs)
    # one dimensional distriubtion along z axis
    dist = np.array([integral(z.value)[0] for z in zs])
    assert zs[np.argmax(dist)].value == approx(h_bary.value)
