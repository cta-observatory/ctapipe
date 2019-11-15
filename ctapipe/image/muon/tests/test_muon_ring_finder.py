import astropy.units as u
from ctapipe.instrument import CameraGeometry, TelescopeDescription
from ctapipe.image.muon import MuonRingFitter
from ctapipe.image import tailcuts_clean, toymodel

def test_chaudhuri_kundu_fitter():
    # flashCam example
    center_xs = 0.3
    center_ys = 0.6
    ring_radius = 0.3
    ring_width = 0.05
    muon_model = toymodel.RingGaussian(
        x=center_xs * u.m,
        y=center_ys * u.m,
        radius=ring_radius * u.m,
        sigma=ring_width * u.m,
    )

    #testing with flashcam
    geom = CameraGeometry.from_name("FlashCam")
    image, _, _ = muon_model.generate_image(
        geom, intensity=1000, nsb_level_pe=5,
    )
    mask = tailcuts_clean(geom, image, 10, 12)
    x = geom.pix_x.to_value(u.m)
    y = geom.pix_y.to_value(u.m)
    img = image * mask

    #call specific method with fit_method
    muonfit = MuonRingFitter(teldes = None, fit_method="chaudhuri_kundu")
    muon_ring_parameters = muonfit.fit(x, y, img)
    xc_fit = muon_ring_parameters.ring_center_x
    yc_fit = muon_ring_parameters.ring_center_y
    r_fit = muon_ring_parameters.ring_radius

    assert u.isclose(xc_fit, center_xs, 1e-1)
    assert u.isclose(yc_fit, center_ys, 1e-1)
    assert u.isclose(r_fit, ring_radius, 1e-1)


def test_taubin_fitter():
    center_xs = 0.3
    center_ys = 0.6
    ring_radius = 0.3
    ring_width = 0.05
    muon_model = toymodel.RingGaussian(
        x=center_xs* u.m,
        y=center_ys* u.m,
        radius=ring_radius* u.m,
        sigma=ring_width* u.m,
    )
    geom = CameraGeometry.from_name("FlashCam")
    # teldes needed for Taubin fit
    teldes = TelescopeDescription.from_name('MST', 'FlashCam')
    focal_length = teldes.optics.equivalent_focal_length.to_value(u.m)
    image, _, _ = muon_model.generate_image(
        geom, intensity=1000, nsb_level_pe=5,
    )
    mask = tailcuts_clean(geom, image, 10, 12)
    x = geom.pix_x.to_value(u.m)/focal_length
    y = geom.pix_y.to_value(u.m)/focal_length
    img = image * mask

    muonfit = MuonRingFitter(teldes=teldes, fit_method="taubin")
    muon_ring_parameters = muonfit.fit(x[mask], y[mask], img)
    xc_fit = muon_ring_parameters.ring_center_x * focal_length
    yc_fit = muon_ring_parameters.ring_center_y * focal_length
    r_fit = muon_ring_parameters.ring_radius * focal_length


    assert u.isclose(xc_fit, center_xs, 1e-1)
    assert u.isclose(yc_fit, center_ys, 1e-1)
    assert u.isclose(r_fit, ring_radius, 1e-1)

if __name__ == '__main__':
    test_chaudhuri_kundu_fitter(),
    test_taubin_fitter()
