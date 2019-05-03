import numpy as np
from ctapipe.calib.camera.flatfield import *
from ctapipe.io.containers import EventAndMonDataContainer


def test_flasherflatfieldcalculator():
    tel_id = 0
    n_events = 10
    n_pixels = 1855
    ff_level = 10000

    ff_calculator = FlasherFlatFieldCalculator(charge_product="LocalPeakWindowSum",
                                               sample_size=n_events,
                                               tel_id=tel_id)
    # create one event
    data = EventAndMonDataContainer()

    # fill the values necessary for the pedestal calculation
    data.mon.tel[tel_id].pixel_status.hardware_mask = np.zeros(n_pixels, dtype=bool)
    data.mon.tel[tel_id].pixel_status.pedestal_mask = np.zeros(n_pixels, dtype=bool)
    data.r1.tel[tel_id].waveform = np.zeros((2, n_pixels, 40))

    # flat-field signal put == delta function of height ff_level at sample 20
    data.r1.tel[tel_id].waveform[:, :, 20] = ff_level

    # First test: good event
    for counts in np.arange(n_events):
        if ff_calculator.calculate_relative_gain(data):
            assert data.mon.tel[tel_id].flatfield
            assert np.mean(data.mon.tel[tel_id].flatfield.charge_median) == ff_level
            assert np.mean(data.mon.tel[tel_id].flatfield.relative_gain_median) == 1
            assert np.mean(data.mon.tel[tel_id].flatfield.relative_gain_std) == 0

    # Second test: introduce some failing pixels
    failing_pixels_id = np.array([10, 20, 30, 40])
    data.r1.tel[tel_id].waveform[:, failing_pixels_id, :] = 0
    data.mon.tel[tel_id].pixel_status.pedestal_mask[failing_pixels_id] = True

    for counts in np.arange(n_events):
        if ff_calculator.calculate_relative_gain(data):
            # test that bad pixels are in the outliers
            assert (np.mean(data.mon.tel[tel_id].flatfield.charge_median_outliers[:, failing_pixels_id]) == True)

            # working pixel have good gain
            assert (data.mon.tel[tel_id].flatfield.relative_gain_median[0, 0] == 1)

            # bad pixels do non influence the gain
            assert np.mean(data.mon.tel[tel_id].flatfield.relative_gain_std) == 0

