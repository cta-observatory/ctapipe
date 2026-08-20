import time
from collections import namedtuple

import astropy.units as u
import numpy as np
import pytest
from scipy.constants import alpha


@pytest.mark.parametrize(
    "ver_initial, mu_x, mu_y, n_photon_phi, n_trials, allowed_time_factor",
    [
        (
            np.array(
                [
                    [86.6025, 0.0],
                    [43.3013, 75.0],
                    [-43.3013, 75.0],
                    [-86.6025, 0.0],
                    [-43.3013, -75.0],
                    [43.3013, -75.0],
                ]
            ),
            0.0,
            0.0,
            100,
            100,
            30,
        ),
    ],
)
def test_polygon_chord_performances(
    ver_initial, mu_x, mu_y, n_photon_phi, n_trials, allowed_time_factor
):
    from ctapipe.image.muon.intensity_fitter import PolygonChord, chord_length

    photon_phi = np.linspace(0, np.pi, n_photon_phi) * u.rad

    times_convex = []
    times_pol = []
    ref_times = []
    for _ in np.arange(n_trials):
        start = time.perf_counter()
        __ = PolygonChord(
            photon_phi,
            np.stack([ver_initial]),
        ).convex_multipolygon_chord(mu_x, mu_y)
        times_convex.append(time.perf_counter() - start)
        #
        start = time.perf_counter()
        __ = PolygonChord(
            photon_phi,
            ver_initial,
        ).polygon_chord(mu_x, mu_y)
        times_pol.append(time.perf_counter() - start)
        #
        start = time.perf_counter()
        __ = chord_length(radius=12, rho=0.5, phi=photon_phi.to_value(u.rad), phi0=0)
        ref_times.append(time.perf_counter() - start)

    times_convex_mean = np.mean(np.array(times_convex))
    ref_times_mean = np.mean(np.array(ref_times))
    times_pol_mean = np.mean(np.array(times_pol))

    assert times_convex_mean / allowed_time_factor < ref_times_mean
    assert times_pol_mean / allowed_time_factor < times_convex_mean


@pytest.mark.parametrize(
    "muon_x, muon_y, photon_phi, expected_chord_length",
    [
        (0.0, 200.0, 90.0 * u.deg, 75.0),
        (0.0, -200.0, 150.0 * u.deg, 75.0),
        (0.0, 0.0, 90.0 * u.deg, 225.0),
        (0.0, -400.0, 90.0 * u.deg, 450.0),
        (0.0, 0.0, 30.0 * u.deg, 75.0),
        (0.0, -200.0, 0.0 * u.deg, 86.6025 * 3),
        (200.0, 200.0, 0.0 * u.deg, 0.0),
    ],
)
def test_multi_polygon_chord(muon_x, muon_y, photon_phi, expected_chord_length):
    from ctapipe.image.muon.intensity_fitter import PolygonChord

    ver_a = np.array(
        [
            [86.6025, 0.0],
            [43.3013, 75.0],
            [-43.3013, 75.0],
            [-86.6025, 0.0],
            [-43.3013, -75.0],
            [43.3013, -75.0],
        ]
    )
    ver_b = ver_a + [[0.0, 200.0]]
    ver_c = ver_a + [[0.0, -200.0]]
    ver_d = ver_a + [[200.0, -200.0]]

    vertices_i = np.stack(
        [
            ver_a,
            ver_b,
            ver_c,
            ver_d,
        ]
    )

    assert np.isclose(
        PolygonChord(
            [photon_phi.to_value(u.rad)], vertices_i
        ).convex_multipolygon_chord(muon_x, muon_y)[0],
        expected_chord_length,
        atol=0.001,
    )

    res = np.stack(
        [
            PolygonChord([photon_phi.to_value(u.rad)], ver).polygon_chord(
                muon_x, muon_y
            )
            for ver in vertices_i
        ]
    )

    assert np.isclose(
        np.sum(res, axis=0)[0],
        expected_chord_length,
        atol=0.001,
    )


@pytest.mark.parametrize(
    "ver_initial, muon_x, muon_y, photon_phi, expected_chord_length",
    [
        (
            np.array(
                [
                    [86.6025, 0.0],
                    [43.3013, 75.0],
                    [-43.3013, 75.0],
                    [-86.6025, 0.0],
                    [-43.3013, -75.0],
                    [43.3013, -75.0],
                ]
            ),
            0.0,
            0.0,
            90.0 * u.deg,
            75.0,
        ),
        (
            np.array(
                [
                    [86.6025, 0.0],
                    [43.3013, 75.0],
                    [-43.3013, 75.0],
                    [-86.6025, 0.0],
                    [-43.3013, -75.0],
                    [43.3013, -75.0],
                ]
            ),
            0.0,
            -200.0,
            90.0 * u.deg,
            150.0,
        ),
        (
            np.array(
                [
                    [86.6025, 0.0],
                    [43.3013, 75.0],
                    [-43.3013, 75.0],
                    [-86.6025, 0.0],
                    [-43.3013, -75.0],
                    [43.3013, -75.0],
                ]
            ),
            0.0,
            0.0,
            30.0 * u.deg,
            75.0,
        ),
        (
            np.array(
                [
                    [86.6025, 0.0],
                    [43.3013, 75.0],
                    [-43.3013, 75.0],
                    [-86.6025, 0.0],
                    [-43.3013, -75.0],
                    [43.3013, -75.0],
                ]
            ),
            200.0,
            200.0,
            30.0 * u.deg,
            0.0,
        ),
    ],
)
def test_polygon_chord(ver_initial, muon_x, muon_y, photon_phi, expected_chord_length):
    from ctapipe.image.muon.intensity_fitter import PolygonChord

    assert np.isclose(
        PolygonChord.from_vertices(ver_initial)._polygon_chord(
            muon_x, muon_y, photon_phi.to_value(u.rad)
        ),
        expected_chord_length,
        atol=0.001,
    )

    assert np.isclose(
        PolygonChord([photon_phi.to_value(u.rad)], ver_initial).polygon_chord(
            muon_x,
            muon_y,
        )[0],
        expected_chord_length,
        atol=0.001,
    )


parameter_names = [
    "radius",
    "rho",
    "phi",
    "expected_length",
]
Parameters = namedtuple("MuonTestParams", parameter_names)


@pytest.mark.parametrize(
    parameter_names,
    [
        Parameters(
            radius=12,
            rho=0.0,
            phi=0.0 * u.deg,
            expected_length=12,
        ),
        Parameters(
            radius=12,
            rho=1,
            phi=90.0 * u.deg,
            expected_length=0,
        ),
        Parameters(
            radius=12,
            rho=1.1,
            phi=180.0 * u.deg,
            expected_length=0,
        ),
        Parameters(
            radius=12,
            rho=2,
            phi=0.0 * u.deg,
            expected_length=24,
        ),
    ],
)
def test_chord_length(
    radius,
    rho,
    phi,
    expected_length,
):
    from ctapipe.image.muon.intensity_fitter import chord_length

    length = chord_length(radius, rho, phi.to_value(u.rad))
    assert np.isclose(length, expected_length, atol=1e-15)


@pytest.mark.parametrize("rho", [0.5, 1.2])
def test_chord_length_periodicity(rho):
    from ctapipe.image.muon.intensity_fitter import chord_length

    radius = 10.0

    phi1 = np.linspace(0, 2 * np.pi, 1000)
    reference_length = chord_length(radius, rho, phi1)

    for offset in (-2 * np.pi, 2 * np.pi):
        phi2 = phi1 + offset
        offset_length = chord_length(radius, rho, phi2)

        np.testing.assert_array_almost_equal(reference_length, offset_length)


@pytest.mark.parametrize("phi0", [45.0 * u.deg, 90.0 * u.deg])
def test_chord_length_phi0_par(phi0):
    from ctapipe.image.muon.intensity_fitter import chord_length

    radius = 12.0
    rho = 0.5

    phi = np.linspace(0, 2 * np.pi, 1000)
    reference_length = chord_length(radius, rho, phi, phi0.to_value(u.rad))

    assert np.isclose(phi[np.argmax(reference_length)], phi0.to_value(u.rad), atol=1e-2)


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
        pix_type=telescope.camera.geometry.pix_type,
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


def test_normalisation_factor(prod5_lst, reference_location):
    """Test of the absolute normalization factor."""
    from ctapipe.coordinates import TelescopeFrame
    from ctapipe.image.muon.intensity_fitter import (
        image_prediction,
    )

    pytest.importorskip("iminuit")

    telescope = prod5_lst

    geom = telescope.camera.geometry.transform_to(TelescopeFrame())
    mirror_radius = np.sqrt(telescope.optics.mirror_area / np.pi)

    pixel_diameter = geom.pixel_width[0]
    x = geom.pix_x
    y = geom.pix_y

    image = image_prediction(
        mirror_radius,
        hole_radius=0 * u.m,
        impact_parameter=0 * u.m,
        phi=0 * u.rad,
        center_x=0.0 * u.deg,
        center_y=0.0 * u.deg,
        radius=1.1 * u.deg,
        ring_width=0.05 * u.deg,
        pixel_x=x,
        pixel_y=y,
        pixel_diameter=pixel_diameter,
        oversampling=3,
        min_lambda=300 * u.nm,
        max_lambda=600 * u.nm,
        pix_type=telescope.camera.geometry.pix_type,
    )

    measured = np.sum(image)
    expected = expected_nphot(
        r_mirror=mirror_radius,
        theta_cher=1.1 * u.deg,
        lambda_min=300 * u.nm,
        lambda_max=600 * u.nm,
    )

    assert u.isclose(measured, expected, rtol=0.02)


def expected_nphot(r_mirror, theta_cher, lambda_min, lambda_max):
    """
    The trivial solution for the number of photons incident on the telescope mirror.

    It is a trivial case, since we assume a muon impact at the center of the dish,
    with no shadowing and a constant Cherenkov angle.
    We neglect the light yield attenuation due to atmospheric absorption.

    Parameters
    ----------
    Rmirror: quantity[length]
        mirror radius
    theta_cher: quantity[angle]
        Cherenkov angle
    lambda_min: quantity[length]
        photon wavelength
    lambda_max: quantity[length]
        photon wavelength

    Returns
    -------
    float: number of Cherenkov photons

    """

    return (
        np.pi
        * alpha
        * r_mirror.to_value(u.m)
        * np.sin(2 * theta_cher)
        * (lambda_min.to_value(u.m) ** -1 - lambda_max.to_value(u.m) ** -1)
    )
