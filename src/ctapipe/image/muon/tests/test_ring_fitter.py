import astropy.units as u
import numpy as np
import pytest

from ctapipe.image import tailcuts_clean, toymodel
from ctapipe.image.muon import MuonRingFitter


def test_MuonRingFitter_has_methods():
    # just to make sure, the test below is running for at least 2 methods
    # basically making sure, we do not test no method at all and always pass
    assert len(MuonRingFitter.fit_method.values) >= 2


@pytest.mark.parametrize(
    "geom_optics_name, method, center_x_deg, center_y_deg, ring_asymmetry_magnitude, ring_asymmetry_orientation_angle_deg",
    [
        (
            "LSTCam",
            MuonRingFitter.fit_method.values[0],
            -0.3 * u.deg,
            0.4 * u.deg,
            1.1,
            45 * u.deg,
        ),
        (
            "LSTCam",
            MuonRingFitter.fit_method.values[1],
            -0.3 * u.deg,
            0.4 * u.deg,
            1.4,
            45 * u.deg,
        ),
        (
            "LSTCam",
            MuonRingFitter.fit_method.values[2],
            -0.3 * u.deg,
            0.4 * u.deg,
            1.4,
            45 * u.deg,
        ),
        (
            "FlashCam",
            MuonRingFitter.fit_method.values[0],
            -1.3 * u.deg,
            1.4 * u.deg,
            1.1,
            45 * u.deg,
        ),
        (
            "FlashCam",
            MuonRingFitter.fit_method.values[1],
            -1.3 * u.deg,
            1.4 * u.deg,
            1.4,
            45 * u.deg,
        ),
        (
            "FlashCam",
            MuonRingFitter.fit_method.values[2],
            -1.3 * u.deg,
            1.4 * u.deg,
            1.4,
            45 * u.deg,
        ),
        (
            "NectarCam",
            MuonRingFitter.fit_method.values[0],
            -1.3 * u.deg,
            1.4 * u.deg,
            1.1,
            45 * u.deg,
        ),
        (
            "NectarCam",
            MuonRingFitter.fit_method.values[1],
            -1.3 * u.deg,
            1.4 * u.deg,
            1.4,
            45 * u.deg,
        ),
        (
            "NectarCam",
            MuonRingFitter.fit_method.values[2],
            -1.3 * u.deg,
            1.4 * u.deg,
            1.4,
            45 * u.deg,
        ),
        (
            "CHEC",
            MuonRingFitter.fit_method.values[0],
            -2.5 * u.deg,
            1.3 * u.deg,
            1.1,
            45 * u.deg,
        ),
        (
            "CHEC",
            MuonRingFitter.fit_method.values[1],
            -2.5 * u.deg,
            1.3 * u.deg,
            1.1,
            45 * u.deg,
        ),
        (
            "CHEC",
            MuonRingFitter.fit_method.values[2],
            -2.5 * u.deg,
            1.3 * u.deg,
            1.1,
            45 * u.deg,
        ),
    ],
)
def test_MuonRingFitter(
    geom_optics_name,
    method,
    center_x_deg,
    center_y_deg,
    ring_asymmetry_magnitude,
    ring_asymmetry_orientation_angle_deg,
    prod5_lst,
    prod5_mst_flashcam,
    prod5_mst_nectarcam,
    prod5_sst,
):
    """test MuonRingFitter"""

    pytest.importorskip("iminuit")

    intensity = 750
    nsb_level_pe = 3
    picture_thresh = 7
    boundary_thresh = 5

    if geom_optics_name == "LSTCam":
        geom = prod5_lst.camera.geometry
        optics = prod5_lst.optics
        intensity = 750
        nsb_level_pe = 3
        picture_thresh = 7
        boundary_thresh = 5
    elif geom_optics_name == "FlashCam":
        geom = prod5_mst_flashcam.camera.geometry
        optics = prod5_mst_flashcam.optics
        intensity = 400
        nsb_level_pe = 2
        picture_thresh = 7
        boundary_thresh = 5
    elif geom_optics_name == "NectarCam":
        geom = prod5_mst_nectarcam.camera.geometry
        optics = prod5_mst_nectarcam.optics
        intensity = 400
        nsb_level_pe = 2
        picture_thresh = 7
        boundary_thresh = 5
    elif geom_optics_name == "CHEC":
        geom = prod5_sst.camera.geometry
        optics = prod5_sst.optics
        intensity = 500
        nsb_level_pe = 0
        picture_thresh = 2
        boundary_thresh = 1
    else:
        geom = prod5_lst.camera.geometry
        optics = prod5_lst.optics
        intensity = 750
        nsb_level_pe = 3
        picture_thresh = 7
        boundary_thresh = 5

    center_xs = optics.effective_focal_length * np.tan(center_x_deg)
    center_ys = optics.effective_focal_length * np.tan(center_y_deg)
    radius = optics.effective_focal_length * np.tan((1.1 * u.deg))
    width = 0.07 * radius
    min_error = 0.05 * radius

    muon_model = toymodel.RingGaussian(
        x=center_xs,
        y=center_ys,
        radius=radius,
        sigma=width,
        asymmetry_magnitude=ring_asymmetry_magnitude,
        asymmetry_orientation_angle_deg=ring_asymmetry_orientation_angle_deg,
    )

    charge, _, _ = muon_model.generate_image(
        geom,
        intensity=intensity,
        nsb_level_pe=nsb_level_pe,
    )
    survivors = tailcuts_clean(geom, charge, picture_thresh, boundary_thresh)

    muonfit = MuonRingFitter(fit_method=method)
    fit_result = muonfit(geom.pix_x, geom.pix_y, charge, survivors)

    print(geom_optics_name)
    print(fit_result)
    print(center_xs, center_ys, radius)

    assert u.isclose(
        fit_result.center_fov_lon,
        center_xs,
        atol=(max(fit_result.center_fov_lon_err, min_error)),
    )
    assert u.isclose(
        fit_result.center_fov_lat,
        center_ys,
        atol=(max(fit_result.center_fov_lat_err, min_error)),
    )
    assert u.isclose(
        fit_result.radius, radius, atol=(max(fit_result.radius_err, min_error))
    )
