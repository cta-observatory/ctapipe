from ctapipe.calib.array.muon_integrator import MuonLineIntegrate
from ctapipe.calib.array.muon_ring_finder import chaudhuri_kundu_circle_fit
import numpy as np
import astropy.units as u


def test_circle_fit_chauduri():
    """
    Simple test function for circle fit algorithm
    """

    centre_x,centre_y,radius =chaudhuri_kundu_circle_fit([0,1,2,1],[0,1,0,-1],np.ones(4))

    assert centre_x == 1
    assert centre_y == 0
    assert radius == 1


def test_muon_integration():
    """
    Test some integration functions of line class
    """
    # test if we put muon at centre we get telescope radius
    radius = 10*u.m
    test_tel = MuonLineIntegrate(mirror_radius=radius,hole_radius=0.*u.m,pixel_width=0.2*u.deg)
    ang,profile = test_tel.plot_pos(0*u.m,1.2*u.deg,0*u.rad)

    assert np.all(profile == 10 )

    ang,profile = test_tel.plot_pos(radius,1.2*u.deg,0*u.rad)
    # check max radius is that of the telescope (give a bit of leeway as the angle picked may not be perfect)
    assert profile.max() > radius.value * 0.99
    # check minimum chord is 0
    assert profile.min() == 0

    xp=list()
    yp=list()
    for x in np.arange(-5,5,0.2):
        for y in np.arange(-5,5,0.2):
            xp.append(x)
            yp.append(y)
    xp = np.asarray(xp) * u.deg
    yp = np.asarray(yp) * u.deg

    # test normalisation of prediction is correct
    image = test_tel.image_prediction(0*u.m,centre_x=0*u.deg, centre_y=0*u.deg, radius=1.2*u.deg,width=0.04*u.deg,
                              pixel_x=xp,pixel_y=yp,phi=0*u.rad)

    # if radius changes this must be adapted
    assert np.abs(image.max() - 846.815116837) < 0.1

#test_circle_fit_chauduri()
#test_muon_integration()