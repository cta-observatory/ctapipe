import numpy as np
from ctapipe.calib.camera.flatfield import *
from ctapipe.io.containers import EventAndMonDataContainer
from traitlets.config.loader import Config
import astropy.units as u
from ctapipe.instrument import SubarrayDescription, TelescopeDescription


def test_flasherflatfieldcalculator():
    """test of flasherFlatFieldCalculator"""
    tel_id = 0
    n_gain = 2
    n_events = 10
    n_pixels = 1855
    ff_level = 10000

    subarray = SubarrayDescription(
        "test array",
        tel_positions={0: np.zeros(3) * u.m},
        tel_descriptions={
            0: TelescopeDescription.from_name(
                optics_name="SST-ASTRI", camera_name="CHEC"
            ),
        }
    )

    config = Config({
        "FixedWindowSum": {
            "window_start": 15,
            "window_width": 10
        }
    })
    ff_calculator = FlasherFlatFieldCalculator(
        subarray=subarray,
        charge_product="FixedWindowSum",
        sample_size=n_events,
        tel_id=tel_id,
        config=config
    )
    # create one event
    data = EventAndMonDataContainer()
    data.meta['origin'] = 'test'

    # initialize mon and r1 data
    data.mon.tel[tel_id].pixel_status.hardware_failing_pixels = np.zeros((n_gain, n_pixels), dtype=bool)
    data.mon.tel[tel_id].pixel_status.pedestal_failing_pixels = np.zeros((n_gain, n_pixels), dtype=bool)
    data.mon.tel[tel_id].pixel_status.flatfield_failing_pixels = np.zeros((n_gain, n_pixels), dtype=bool)
    data.r1.tel[tel_id].waveform = np.zeros((n_gain, n_pixels, 40))
    data.r1.tel[tel_id].trigger_time = 1000
    
    # flat-field signal put == delta function of height ff_level at sample 20
    data.r1.tel[tel_id].waveform[:, :, 20] = ff_level
    print(data.r1.tel[tel_id].waveform[0, 0, 20])

    # First test: good event
    while ff_calculator.num_events_seen < n_events:
        if ff_calculator.calculate_relative_gain(data):
            assert data.mon.tel[tel_id].flatfield

            print(data.mon.tel[tel_id].flatfield)
            assert np.mean(data.mon.tel[tel_id].flatfield.charge_median) == ff_level
            assert np.mean(data.mon.tel[tel_id].flatfield.relative_gain_median) == 1
            assert np.mean(data.mon.tel[tel_id].flatfield.relative_gain_std) == 0

    # Second test: introduce some failing pixels
    failing_pixels_id = np.array([10, 20, 30, 40])
    data.r1.tel[tel_id].waveform[:, failing_pixels_id, :] = 0
    data.mon.tel[tel_id].pixel_status.pedestal_failing_pixels[:,failing_pixels_id] = True

    while ff_calculator.num_events_seen < n_events:
        if ff_calculator.calculate_relative_gain(data):

            # working pixel have good gain
            assert (data.mon.tel[tel_id].flatfield.relative_gain_median[0, 0] == 1)

            # bad pixels do non influence the gain
            assert np.mean(data.mon.tel[tel_id].flatfield.relative_gain_std) == 0

