import astropy.units as u
from ctapipe.instrument import CameraGeometry, TelescopeDescription
from ctapipe.image.muon import MuonRingFitter
from ctapipe.image import tailcuts_clean, toymodel


def my_tailcuts_clean(geom, charge, *args, **kwargs):
    survivors = tailcuts_clean(geom, charge, *args, **kwargs)
    return geom, charge, survivors


def test_chaudhuri_kundu_fitter():
    # flashCam example
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

    #testing with flashcam
    geom = CameraGeometry.from_name("FlashCam")
    charge, _, _ = muon_model.generate_image(
        geom, intensity=1000, nsb_level_pe=5,
    )
    cleaned_image = my_tailcuts_clean(geom, charge, 10, 12)

    muonfit = MuonRingFitter(fit_method="chaudhuri_kundu")
    fit_result = muonfit(*cleaned_image)

    print(fit_result)
    print(center_xs, center_ys, ring_radius)

    assert u.isclose(fit_result.ring_center_x, center_xs, 5e-2)
    assert u.isclose(fit_result.ring_center_y, center_ys, 5e-2)
    assert u.isclose(fit_result.ring_radius, ring_radius, 5e-2)


def test_taubin_fitter():
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
    # teldes needed for Taubin fit
    geom = CameraGeometry.from_name("FlashCam")
    charge, _, _ = muon_model.generate_image(
        geom, intensity=1000, nsb_level_pe=5,
    )
    cleaned_image = my_tailcuts_clean(geom, charge, 10, 12)


    muonfit = MuonRingFitter(fit_method="taubin")
    fit_result = muonfit(*cleaned_image)

    print(fit_result)
    print(center_xs, center_ys, ring_radius)

    assert u.isclose(fit_result.ring_center_x, center_xs, 5e-2)
    assert u.isclose(fit_result.ring_center_y, center_ys, 5e-2)
    assert u.isclose(fit_result.ring_radius, ring_radius, 5e-2)


if __name__ == '__main__':
    test_chaudhuri_kundu_fitter()
    test_taubin_fitter()
