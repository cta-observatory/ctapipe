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


def test_muon_efficiency_fit(prod5_lst, reference_location):
    from ctapipe.coordinates import TelescopeFrame
    from ctapipe.image.muon.intensity_fitter import (
        MuonIntensityFitter,
        image_prediction,
    )
    from ctapipe.instrument import SubarrayDescription

    pytest.importorskip("iminuit")

    tel_id = 1
    telescope = prod5_lst
    subarray = SubarrayDescription(
        name="LSTMono",
        tel_positions={tel_id: [0, 0, 0] * u.m},
        tel_descriptions={tel_id: telescope},
        reference_location=reference_location,
    )

    center_x = 0.8 * u.deg
    center_y = 0.4 * u.deg
    radius = 1.1 * u.deg
    ring_width = 0.05 * u.deg
    impact_parameter = 5 * u.m
    phi = 0 * u.rad
    efficiency = 0.5

    geom = telescope.camera.geometry.transform_to(TelescopeFrame())
    mirror_radius = np.sqrt(telescope.optics.mirror_area / np.pi)

    pixel_diameter = geom.pixel_width[0]
    x = geom.pix_x
    y = geom.pix_y

    fitter = MuonIntensityFitter(subarray=subarray)

    image = image_prediction(
        mirror_radius,
        hole_radius=fitter.hole_radius_m.tel[tel_id] * u.m,
        impact_parameter=impact_parameter,
        phi=phi,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        ring_width=ring_width,
        pixel_x=x,
        pixel_y=y,
        pixel_diameter=pixel_diameter,
    )

    result = fitter(
        tel_id=tel_id,
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


def test_scts(prod5_sst, reference_location):
    from ctapipe.image.muon.intensity_fitter import MuonIntensityFitter
    from ctapipe.instrument import SubarrayDescription

    pytest.importorskip("iminuit")

    telescope = prod5_sst
    subarray = SubarrayDescription(
        name="ssts",
        tel_positions={0: [0, 0, 0] * u.m},
        tel_descriptions={0: telescope},
        reference_location=reference_location,
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
