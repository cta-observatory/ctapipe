import astropy.units as u
import numpy as np
import pytest


def test_dummy(prod5_lst, reference_location):
    from ctapipe.image.muon.impactpoint_intensity_fitter import (
        MuonImpactpointIntensityFitter,
        compute_absolute_optical_efficiency_from_muon_ring,
    )
    from ctapipe.instrument import SubarrayDescription

    pytest.importorskip("iminuit")

    tel_id = 1
    telescope = prod5_lst
    subarray = SubarrayDescription(
        name="LST",
        tel_positions={tel_id: [0, 0, 0] * u.m},
        tel_descriptions={tel_id: telescope},
        reference_location=reference_location,
    )

    fitter = MuonImpactpointIntensityFitter(subarray=subarray)

    fitter(
        tel_id=tel_id,
        center_x=0 * u.deg,
        center_y=2 * u.deg,
        radius=1.3 * u.deg,
        image=np.zeros(telescope.camera.geometry.n_pixels),
        pedestal=np.zeros(telescope.camera.geometry.n_pixels),
    )

    res0 = compute_absolute_optical_efficiency_from_muon_ring(
        measured_number_pe=1.0,
        radius=1.1 * u.deg,
        min_lambda_m=300e-9 * u.m,
        max_lambda_m=600e-9 * u.m,
        hole_radius_m=0.0 * u.m,
        optics=telescope.optics,
        rho=0.0 * u.m,
    )

    print(" ")
    print("res: ", 0.0, " ", 1.0 / res0)

    assert 1
