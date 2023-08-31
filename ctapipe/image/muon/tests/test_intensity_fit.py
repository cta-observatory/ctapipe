import astropy.units as u
import numpy as np
import pytest


def test_chord_length():
    from ctapipe.image.muon.intensity_fitter import chord_length

    radius = 12
    rho = 0.0
    phi = 0

    length = chord_length(radius, rho, phi)
    assert length == radius

    rho = 1
    phi = np.deg2rad(90)
    length = chord_length(radius, rho, phi)
    assert np.isclose(length, 0, atol=1e-15)


def test_muon_efficiency_fit(prod5_lst):
    from ctapipe.coordinates import CameraFrame, TelescopeFrame
    from ctapipe.image.muon.intensity_fitter import (
        MuonIntensityFitter,
        image_prediction,
    )
    from ctapipe.instrument import SubarrayDescription

    telescope = prod5_lst
    subarray = SubarrayDescription(
        "LSTMono",
        {0: [0, 0, 0] * u.m},
        {0: telescope},
    )

    center_x = 0.8 * u.deg
    center_y = 0.4 * u.deg
    radius = 1.2 * u.deg
    ring_width = 0.05 * u.deg
    impact_parameter = 5 * u.m
    phi = 0 * u.rad
    efficiency = 0.5

    focal_length = telescope.optics.equivalent_focal_length
    geom = telescope.camera.geometry
    mirror_radius = np.sqrt(telescope.optics.mirror_area / np.pi)
    pixel_diameter = (
        2
        * u.rad
        * (np.sqrt(geom.pix_area / np.pi) / focal_length).to_value(
            u.dimensionless_unscaled
        )
    )

    tel = CameraFrame(
        x=geom.pix_x,
        y=geom.pix_y,
        focal_length=focal_length,
        rotation=geom.cam_rotation,
    ).transform_to(TelescopeFrame())
    x = tel.fov_lon
    y = tel.fov_lat

    image = image_prediction(
        mirror_radius,
        hole_radius=0 * u.m,
        impact_parameter=impact_parameter,
        phi=phi,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        ring_width=ring_width,
        pixel_x=x,
        pixel_y=y,
        pixel_diameter=pixel_diameter[0],
    )

    fitter = MuonIntensityFitter(subarray=subarray)
    result = fitter(
        tel_id=0,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        image=image * efficiency,
        pedestal=np.full_like(image, 1.1),
    )

    assert u.isclose(result.impact, impact_parameter, rtol=0.05)
    assert u.isclose(result.width, ring_width, rtol=0.05)
    assert u.isclose(result.optical_efficiency, efficiency, rtol=0.05)
    assert result.is_valid
    assert not result.parameters_at_limit
    assert np.isfinite(result.likelihood_value)


def test_scts(prod5_sst):
    from ctapipe.image.muon.intensity_fitter import MuonIntensityFitter
    from ctapipe.instrument import SubarrayDescription

    telescope = prod5_sst
    subarray = SubarrayDescription(
        "ssts",
        {0: [0, 0, 0] * u.m},
        {0: telescope},
    )

    fitter = MuonIntensityFitter(subarray=subarray)
    with pytest.raises(NotImplementedError):
        fitter(
            tel_id=0,
            center_x=0 * u.deg,
            center_y=2 * u.deg,
            radius=1.3 * u.deg,
            image=np.zeros(telescope.camera.geometry.n_pixels),
            pedestal=np.zeros(telescope.camera.geometry.n_pixels),
        )
