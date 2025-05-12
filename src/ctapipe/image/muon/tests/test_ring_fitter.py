from collections import namedtuple

import astropy.units as u
import numpy as np
import pytest

from ctapipe.image import tailcuts_clean, toymodel
from ctapipe.image.muon import MuonRingFitter


def test_muon_ring_fitter_has_methods():
    # just to make sure, the test below is running for at least 2 methods
    # basically making sure, we do not test no method at all and always pass
    assert len(MuonRingFitter.fit_method.values) >= 2


parameter_names = [
    "tel_fixture_name",
    "method",
    "center_x",
    "center_y",
    "impact_rho",
    "impact_angle",
    "intensity",
    "nsb_level_pe",
    "picture_thresh",
    "boundary_thresh",
]
Parameters = namedtuple("MuonTestParams", parameter_names)


@pytest.mark.parametrize(
    parameter_names,
    [
        Parameters(
            tel_fixture_name="prod5_lst",
            method=MuonRingFitter.fit_method.values[0],
            center_x=-0.3 * u.deg,
            center_y=0.4 * u.deg,
            impact_rho=0.5,
            impact_angle=45 * u.deg,
            intensity=750,
            nsb_level_pe=3,
            picture_thresh=7,
            boundary_thresh=5,
        ),
        Parameters(
            tel_fixture_name="prod5_lst",
            method=MuonRingFitter.fit_method.values[1],
            center_x=-0.3 * u.deg,
            center_y=0.4 * u.deg,
            impact_rho=0.8,
            impact_angle=45 * u.deg,
            intensity=750,
            nsb_level_pe=3,
            picture_thresh=7,
            boundary_thresh=5,
        ),
        Parameters(
            tel_fixture_name="prod5_lst",
            method=MuonRingFitter.fit_method.values[2],
            center_x=-0.3 * u.deg,
            center_y=0.4 * u.deg,
            impact_rho=0.8,
            impact_angle=45 * u.deg,
            intensity=750,
            nsb_level_pe=3,
            picture_thresh=7,
            boundary_thresh=5,
        ),
        Parameters(
            tel_fixture_name="prod5_mst_flashcam",
            method=MuonRingFitter.fit_method.values[0],
            center_x=-1.3 * u.deg,
            center_y=1.4 * u.deg,
            impact_rho=0.5,
            impact_angle=45 * u.deg,
            intensity=400,
            nsb_level_pe=2,
            picture_thresh=7,
            boundary_thresh=5,
        ),
        Parameters(
            tel_fixture_name="prod5_mst_flashcam",
            method=MuonRingFitter.fit_method.values[1],
            center_x=-1.3 * u.deg,
            center_y=1.4 * u.deg,
            impact_rho=0.8,
            impact_angle=45 * u.deg,
            intensity=400,
            nsb_level_pe=2,
            picture_thresh=7,
            boundary_thresh=5,
        ),
        Parameters(
            tel_fixture_name="prod5_mst_flashcam",
            method=MuonRingFitter.fit_method.values[2],
            center_x=-1.3 * u.deg,
            center_y=1.4 * u.deg,
            impact_rho=0.8,
            impact_angle=45 * u.deg,
            intensity=400,
            nsb_level_pe=2,
            picture_thresh=7,
            boundary_thresh=5,
        ),
        Parameters(
            tel_fixture_name="prod5_mst_nectarcam",
            method=MuonRingFitter.fit_method.values[0],
            center_x=-1.3 * u.deg,
            center_y=1.4 * u.deg,
            impact_rho=0.5,
            impact_angle=45 * u.deg,
            intensity=400,
            nsb_level_pe=2,
            picture_thresh=7,
            boundary_thresh=5,
        ),
        Parameters(
            tel_fixture_name="prod5_mst_nectarcam",
            method=MuonRingFitter.fit_method.values[1],
            center_x=-1.3 * u.deg,
            center_y=1.4 * u.deg,
            impact_rho=0.8,
            impact_angle=45 * u.deg,
            intensity=400,
            nsb_level_pe=2,
            picture_thresh=7,
            boundary_thresh=5,
        ),
        Parameters(
            tel_fixture_name="prod5_mst_nectarcam",
            method=MuonRingFitter.fit_method.values[2],
            center_x=-1.3 * u.deg,
            center_y=1.4 * u.deg,
            impact_rho=0.8,
            impact_angle=45 * u.deg,
            intensity=400,
            nsb_level_pe=2,
            picture_thresh=7,
            boundary_thresh=5,
        ),
        Parameters(
            tel_fixture_name="prod5_sst",
            method=MuonRingFitter.fit_method.values[0],
            center_x=-2.5 * u.deg,
            center_y=1.3 * u.deg,
            impact_rho=0.5,
            impact_angle=45 * u.deg,
            intensity=500,
            nsb_level_pe=0,
            picture_thresh=2,
            boundary_thresh=1,
        ),
        Parameters(
            tel_fixture_name="prod5_sst",
            method=MuonRingFitter.fit_method.values[1],
            center_x=-2.5 * u.deg,
            center_y=1.3 * u.deg,
            impact_rho=0.5,
            impact_angle=45 * u.deg,
            intensity=500,
            nsb_level_pe=0,
            picture_thresh=2,
            boundary_thresh=1,
        ),
        Parameters(
            tel_fixture_name="prod5_sst",
            method=MuonRingFitter.fit_method.values[2],
            center_x=-2.5 * u.deg,
            center_y=1.3 * u.deg,
            impact_rho=0.5,
            impact_angle=45 * u.deg,
            intensity=500,
            nsb_level_pe=0,
            picture_thresh=2,
            boundary_thresh=1,
        ),
    ],
)
def test_muon_ring_fitter(
    request,
    tel_fixture_name,
    method,
    center_x,
    center_y,
    impact_rho,
    impact_angle,
    intensity,
    nsb_level_pe,
    picture_thresh,
    boundary_thresh,
):
    """test MuonRingFitter"""

    pytest.importorskip("iminuit")

    # Dynamically retrieve the fixture for the specified camera
    tel = request.getfixturevalue(tel_fixture_name)
    geom = tel.camera.geometry
    optics = tel.optics

    center_xs = optics.effective_focal_length * np.tan(center_x)
    center_ys = optics.effective_focal_length * np.tan(center_y)
    radius = optics.effective_focal_length * np.tan((1.1 * u.deg))
    width = 0.07 * radius
    min_error = 0.05 * radius

    muon_model = toymodel.RingGaussian(
        x=center_xs,
        y=center_ys,
        radius=radius,
        sigma=width,
        rho=impact_rho,
        phi0=impact_angle,
    )

    charge, _, _ = muon_model.generate_image(
        geom,
        intensity=intensity,
        nsb_level_pe=nsb_level_pe,
    )
    survivors = tailcuts_clean(geom, charge, picture_thresh, boundary_thresh)

    muonfit = MuonRingFitter(fit_method=method)
    fit_result = muonfit(geom.pix_x, geom.pix_y, charge, survivors)

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
