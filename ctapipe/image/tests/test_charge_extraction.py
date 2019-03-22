import pytest
import numpy as np
from scipy.stats import norm
from numpy.testing import assert_allclose
from ctapipe.instrument import CameraGeometry
from ctapipe.image.charge_extractors import (
    ChargeExtractor,
    FullIntegrator,
    SimpleIntegrator,
    GlobalPeakIntegrator,
    LocalPeakIntegrator,
    NeighbourPeakIntegrator,
    AverageWfPeakIntegrator,
)


@pytest.fixture(scope='module')
def camera_waveforms():
    camera = CameraGeometry.from_name("CHEC")

    n_pixels = camera.n_pixels
    n_samples = 96
    mid = n_samples // 2
    pulse_sigma = 6
    r_hi = np.random.RandomState(1)
    r_lo = np.random.RandomState(2)

    x = np.arange(n_samples)

    # Randomize times
    t_pulse_hi = r_hi.uniform(mid - 10, mid + 10, n_pixels)[:, None]
    t_pulse_lo = r_lo.uniform(mid + 10, mid + 20, n_pixels)[:, None]

    # Create pulses
    y_hi = norm.pdf(x, t_pulse_hi, pulse_sigma)
    y_lo = norm.pdf(x, t_pulse_lo, pulse_sigma)

    # Randomize amplitudes
    y_hi *= r_hi.uniform(100, 1000, n_pixels)[:, None]
    y_lo *= r_lo.uniform(100, 1000, n_pixels)[:, None]

    y = np.stack([y_hi, y_lo])

    return y, camera


def test_full_integration(camera_waveforms):
    waveforms, camera = camera_waveforms
    integrator = FullIntegrator()
    integration, peakpos, window = integrator.extract_charge(waveforms)

    assert_allclose(integration[0][0], 545.945, rtol=1e-3)
    assert_allclose(integration[1][0], 970.025, rtol=1e-3)


def test_simple_integration(camera_waveforms):
    waveforms, camera = camera_waveforms
    integrator = SimpleIntegrator(window_start=45)
    integration, peakpos, window = integrator.extract_charge(waveforms)

    assert_allclose(integration[0][0], 232.559, rtol=1e-3)
    assert_allclose(integration[1][0], 32.539, rtol=1e-3)


def test_global_peak_integration(camera_waveforms):
    waveforms, camera = camera_waveforms
    integrator = GlobalPeakIntegrator()
    integration, peakpos, window = integrator.extract_charge(waveforms)

    assert_allclose(integration[0][0], 232.559, rtol=1e-3)
    assert_allclose(integration[1][0], 425.406, rtol=1e-3)


def test_local_peak_integration(camera_waveforms):
    waveforms, camera = camera_waveforms
    integrator = LocalPeakIntegrator()
    integration, peakpos, window = integrator.extract_charge(waveforms)

    assert_allclose(integration[0][0], 240.3, rtol=1e-3)
    assert_allclose(integration[1][0], 427.158, rtol=1e-3)


def test_nb_peak_integration(camera_waveforms):
    waveforms, camera = camera_waveforms
    nei = camera.neighbor_matrix_where
    integrator = NeighbourPeakIntegrator()
    integrator.neighbours = nei
    integration, peakpos, window = integrator.extract_charge(waveforms)

    assert_allclose(integration[0][0], 94.671, rtol=1e-3)
    assert_allclose(integration[1][0], 426.887, rtol=1e-3)


def test_averagewf_peak_integration(camera_waveforms):
    waveforms, camera = camera_waveforms
    integrator = AverageWfPeakIntegrator()
    integration, peakpos, window = integrator.extract_charge(waveforms)

    assert_allclose(integration[0][0], 232.559, rtol=1e-3)
    assert_allclose(integration[1][0], 425.406, rtol=1e-3)


def test_charge_extractor_factory(camera_waveforms):
    waveforms, camera = camera_waveforms
    extractor = ChargeExtractor.from_name('LocalPeakIntegrator')
    extractor.extract_charge(waveforms)


def test_charge_extractor_factory_args():
    '''config is supposed to be created by a `Tool`
    '''
    from traitlets.config.loader import Config
    config = Config(
        {
            'ChargeExtractor': {
                'window_width': 20,
                'window_shift': 3,
            }
        }
    )

    local_peak_integrator = ChargeExtractor.from_name(
        'LocalPeakIntegrator',
        config=config,
    )
    assert local_peak_integrator.window_width == 20
    assert local_peak_integrator.window_shift == 3

    ChargeExtractor.from_name(
        'FullIntegrator',
        config=config,
    )
