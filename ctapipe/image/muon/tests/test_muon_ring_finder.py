import numpy as np
import astropy.units as u
from ctapipe.instrument import CameraGeometry
from ctapipe.image.muon import MuonRingFitter
from ctapipe.image import tailcuts_clean, toymodel

def test_chaudhuri_kundu_fitter():
    center_xs = 0.3 * u.m
    center_ys = 0.6 * u.m
    ring_radius = 0.3 * u.m
    ring_width = 0.05 * u.m
    muon_model = toymodel.RingGaussian(
        x=center_xs,
        y=center_ys,
        radius=ring_radius,
        sigma=ring_width,
    )

    geom = CameraGeometry.from_name("FlashCam")
    flashcam_focal_length = u.Quantity(16, u.m)
    image, _, _ = muon_model.generate_image(
        geom, intensity=1000, nsb_level_pe=5,
    )
    mask = tailcuts_clean(geom, image, 10, 12)
    x = (geom.pix_x / flashcam_focal_length) * u.rad
    y = (geom.pix_y / flashcam_focal_length) * u.rad
    img = image * mask

    #call specific method with fit_method, teldes needed for Taubin fit
    muonfit = MuonRingFitter(teldes = None, fit_method="chaudhuri_kundu")
    muon_ring_parameters = muonfit.fit(x, y, img)
    xc_fit = muon_ring_parameters.ring_center_x
    yc_fit = muon_ring_parameters.ring_center_y
    r_fit = muon_ring_parameters.ring_radius

    assert np.isclose(xc_fit * flashcam_focal_length / u.m / u.rad, center_xs / u.m, 1e-1)
    assert np.isclose(yc_fit * flashcam_focal_length / u.m / u.rad, center_ys / u.m, 1e-1)
    assert np.isclose(r_fit * flashcam_focal_length / u.m / u.rad, ring_radius / u.m, 1e-1)


if __name__ == '__main__':
    test_chaudhuri_kundu_fitter()
