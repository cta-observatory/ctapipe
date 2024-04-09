from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.time import Time
from traitlets.config import Config

from ctapipe.calib.camera.flatfield import FlasherFlatFieldCalculator
from ctapipe.containers import ArrayEventContainer
from ctapipe.instrument import SubarrayDescription


def test_flasherflatfieldcalculator(prod5_sst, reference_location):
    """test of flasherFlatFieldCalculator"""
    tel_id = 0
    n_gain = 2
    n_events = 10
    n_pixels = 1855
    ff_level = 10000

    subarray = SubarrayDescription(
        "test array",
        tel_positions={0: np.zeros(3) * u.m},
        tel_descriptions={0: deepcopy(prod5_sst)},
        reference_location=reference_location,
    )
    subarray.tel[0].camera.readout.reference_pulse_shape = np.ones((1, 2))
    subarray.tel[0].camera.readout.reference_pulse_sample_width = u.Quantity(1, u.ns)

    config = Config(
        {"FixedWindowSum": {"peak_index": 15, "window_shift": 0, "window_width": 10}}
    )
    ff_calculator = FlasherFlatFieldCalculator(
        subarray=subarray,
        charge_product="FixedWindowSum",
        sample_size=n_events,
        tel_id=tel_id,
        config=config,
    )
    # create one event
    data = ArrayEventContainer()
    data.meta["origin"] = "test"
    data.trigger.time = Time.now()

    # initialize mon and r1 data
    data.mon.tel[tel_id].pixel_status.hardware_failing_pixels = np.zeros(
        (n_gain, n_pixels), dtype=bool
    )
    data.mon.tel[tel_id].pixel_status.pedestal_failing_pixels = np.zeros(
        (n_gain, n_pixels), dtype=bool
    )
    data.mon.tel[tel_id].pixel_status.flatfield_failing_pixels = np.zeros(
        (n_gain, n_pixels), dtype=bool
    )
    data.r1.tel[tel_id].waveform = np.zeros((n_gain, n_pixels, 40))

    # flat-field signal put == delta function of height ff_level at sample 20
    data.r1.tel[tel_id].waveform[:, :, 20] = ff_level
    print(data.r1.tel[tel_id].waveform[0, 0, 20])

    # First test: good event
    while ff_calculator.n_events_seen < n_events:
        if ff_calculator.calculate_relative_gain(data):
            assert data.mon.tel[tel_id].flatfield

            print(data.mon.tel[tel_id].flatfield)
            assert np.mean(data.mon.tel[tel_id].flatfield.charge_median) == ff_level
            assert np.mean(data.mon.tel[tel_id].flatfield.relative_gain_median) == 1
            assert np.mean(data.mon.tel[tel_id].flatfield.relative_gain_std) == 0

    # Second test: introduce some failing pixels
    failing_pixels_id = np.array([10, 20, 30, 40])
    data.r1.tel[tel_id].waveform[:, failing_pixels_id, :] = 0
    data.mon.tel[tel_id].pixel_status.pedestal_failing_pixels[
        :, failing_pixels_id
    ] = True

    while ff_calculator.n_events_seen < n_events:
        if ff_calculator.calculate_relative_gain(data):
            # working pixel have good gain
            assert data.mon.tel[tel_id].flatfield.relative_gain_median[0, 0] == 1

            # bad pixels do non influence the gain
            assert np.mean(data.mon.tel[tel_id].flatfield.relative_gain_std) == 0
