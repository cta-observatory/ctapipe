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
    pulse_sigma = 6
    r = np.random.RandomState(1)
    r.uniform(0, 10, 5)

    x = np.arange(n_samples)
    y = norm.pdf(x, r.uniform(n_samples // 2 - 10, n_samples // 2 + 10, n_pixels)[:, None], pulse_sigma)
    y *= r.uniform(100, 1000, n_pixels)[:, None]

    # 2 Channels
    y = np.stack([y, y * r.uniform(0, 0.5, n_pixels)[:, None]])

    return y, camera


def test_full_integration(camera_waveforms):
    waveforms, camera = camera_waveforms
    integrator = FullIntegrator()
    integration, peakpos, window = integrator.extract_charge(waveforms)

    assert_allclose(integration[0][0], 267.843, rtol=1e-3)
    assert_allclose(integration[1][0], 6.194, rtol=1e-3)


def test_simple_integration(camera_waveforms):
    waveforms, camera = camera_waveforms
    integrator = SimpleIntegrator()
    integration, peakpos, window = integrator.extract_charge(waveforms)

    assert_allclose(integration[0][0], 3.524e-06, rtol=1e-3)
    assert_allclose(integration[1][0], 8.15e-08, rtol=1e-3)


def test_global_peak_integration(camera_waveforms):
    waveforms, camera = camera_waveforms
    integrator = GlobalPeakIntegrator()
    integration, peakpos, window = integrator.extract_charge(waveforms)

    assert_allclose(integration[0][0], 51.647, rtol=1e-3)
    assert_allclose(integration[1][0], 1.194, rtol=1e-3)


def test_local_peak_integration(camera_waveforms):
    waveforms, camera = camera_waveforms
    integrator = LocalPeakIntegrator()
    integration, peakpos, window = integrator.extract_charge(waveforms)

    assert_allclose(integration[0][0], 118.027, rtol=1e-3)
    assert_allclose(integration[1][0], 2.729, rtol=1e-3)


def test_nb_peak_integration(camera_waveforms):
    waveforms, camera = camera_waveforms
    nei = camera.neighbor_matrix_where
    integrator = NeighbourPeakIntegrator()
    integrator.neighbours = nei
    integration, peakpos, window = integrator.extract_charge(waveforms)

    assert_allclose(integration[0][0], 95.3, rtol=1e-3)
    assert_allclose(integration[1][0], 0.2237, rtol=1e-3)


def test_averagewf_peak_integration(camera_waveforms):
    waveforms, camera = camera_waveforms
    integrator = AverageWfPeakIntegrator()
    integration, peakpos, window = integrator.extract_charge(waveforms)

    assert_allclose(integration[0][0], 51.647, rtol=1e-3)
    assert_allclose(integration[1][0], 1.194, rtol=1e-3)


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
