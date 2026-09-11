"""Test assuring combining several steps keeps data consistent"""

import numpy as np


def test_r1_simtel_broken_pixels(prod5_gamma_simtel_path, tmp_path):
    """Test that broken pixel information is preserved for simtel r1"""
    from ctapipe.io import DataWriter, EventSource

    # write r1 waveforms for simtel file
    r1_path = tmp_path / "events.r1.h5"

    with EventSource(prod5_gamma_simtel_path) as source:
        with DataWriter(source, output_path=r1_path, write_r1_waveforms=True) as writer:
            for event in source:
                writer(event)

    with EventSource(r1_path) as source:
        n_checked = 0
        for event in source:
            for tel_id, mon in event.monitoring.tel.items():
                # not filled at the moment
                # assert mon.camera.coefficients.time_shift is not None
                readout = source.subarray.tel[tel_id].camera.readout

                # filled from pixel status
                outlier_mask = mon.camera.coefficients.outlier_mask
                assert outlier_mask is not None
                assert outlier_mask.shape == (readout.n_channels, readout.n_pixels)

                # flashcam has 6 disabled pixels, only one gain
                if readout.name == "FlashCam":
                    assert np.count_nonzero(outlier_mask[0]) == 6
                # we expect no other cameras to have disabled pixels here
                else:
                    pixel_index = np.arange(readout.n_pixels)
                    selected_gain_channel = event.r1.tel[tel_id].selected_gain_channel
                    active_mask = outlier_mask[selected_gain_channel, pixel_index]
                    assert np.count_nonzero(active_mask) == 0

                n_checked += 1

        assert n_checked > 0
