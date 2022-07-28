"""
Test of sliding window extractor for LST camera pulse shape with
the correction for the integration window completeness
"""

import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose
from traitlets.config.loader import Config

from ctapipe.containers import DL1CameraContainer
from ctapipe.image.extractor import ImageExtractor, SlidingWindowMaxSum
from ctapipe.image.toymodel import WaveformModel
from ctapipe.instrument import SubarrayDescription


def test_sw_pulse_lst(prod5_lst):
    """
    Test function of sliding window extractor for LST camera pulse shape with
    the correction for the integration window completeness
    """

    # prepare array with 1 LST
    subarray = SubarrayDescription(
        "LST1",
        tel_positions={1: np.zeros(3) * u.m},
        tel_descriptions={1: prod5_lst},
    )

    tel_id = list(subarray.tel.keys())[0]

    n_pixels = subarray.tel[tel_id].camera.geometry.n_pixels
    n_samples = 40
    readout = subarray.tel[tel_id].camera.readout

    random = np.random.RandomState(1)
    min_charge = 100
    max_charge = 1000
    charge_true = random.uniform(min_charge, max_charge, n_pixels)
    time_true = random.uniform(
        n_samples // 2 - 1, n_samples // 2 + 1, n_pixels
    ) / readout.sampling_rate.to_value(u.GHz)

    waveform_model = WaveformModel.from_camera_readout(readout)
    waveform = waveform_model.get_waveform(charge_true, time_true, n_samples)
    selected_gain_channel = np.zeros(charge_true.size, dtype=np.int8)

    # define extractor
    config = Config({"SlidingWindowMaxSum": {"window_width": 8}})
    extractor = SlidingWindowMaxSum(subarray=subarray)
    extractor = ImageExtractor.from_name(
        "SlidingWindowMaxSum", subarray=subarray, config=config
    )

    broken_pixels = np.zeros(n_pixels, dtype=bool)
    dl1: DL1CameraContainer = extractor(
        waveform, tel_id, selected_gain_channel, broken_pixels
    )
    print(dl1.image / charge_true)
    assert_allclose(dl1.image, charge_true, rtol=0.02)
    assert dl1.is_valid
