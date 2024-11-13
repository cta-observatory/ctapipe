import astropy.units as u
import pytest

from ctapipe.image import tailcuts_clean, toymodel
from ctapipe.image.muon import MuonRingFitter


def test_MuonRingFitter_has_methods():
    # just to make sure, the test below is running for at least 2 methods
    # basically making sure, we do not test no method at all and always pass
    assert len(MuonRingFitter.fit_method.values) >= 2


@pytest.mark.parametrize("method", MuonRingFitter.fit_method.values)
def test_MuonRingFitter(method, prod5_mst_flashcam):
    """test MuonRingFitter"""
    pytest.importorskip("iminuit")

    # flashCam example
    center_xs = 0.3 * u.m
    center_ys = 0.6 * u.m
    radius = 0.3 * u.m
    width = 0.05 * u.m

    muon_model = toymodel.RingGaussian(
        x=center_xs,
        y=center_ys,
        radius=radius,
        sigma=width,
    )

    # testing with flashcam
    geom = prod5_mst_flashcam.camera.geometry
    charge, _, _ = muon_model.generate_image(
        geom,
        intensity=1000,
        nsb_level_pe=5,
    )
    survivors = tailcuts_clean(geom, charge, 10, 12)

    muonfit = MuonRingFitter(fit_method=method)
    fit_result = muonfit(geom.pix_x, geom.pix_y, charge, survivors)

    print(fit_result)
    print(center_xs, center_ys, radius)

    assert u.isclose(fit_result.center_fov_lon, center_xs, 5e-2)
    assert u.isclose(fit_result.center_fov_lat, center_ys, 5e-2)
    assert u.isclose(fit_result.radius, radius, 5e-2)
