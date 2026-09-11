"""Test assuring combining several steps keeps data consistent"""

import numpy as np
import pytest

from ctapipe.image import ImageExtractor


@pytest.fixture(scope="session")
def r1_file(tmp_path_factory, prod5_gamma_simtel_path):
    from ctapipe.io import DataWriter, EventSource

    outdir = tmp_path_factory.mktemp("r1_")
    # write r1 waveforms for simtel file
    #
    r1_path = outdir / "events.r1.h5"

    with EventSource(prod5_gamma_simtel_path) as source:
        with DataWriter(source, output_path=r1_path, write_r1_waveforms=True) as writer:
            for event in source:
                writer(event)

    return r1_path


@pytest.mark.parametrize("extractor", ImageExtractor.non_abstract_subclasses().keys())
def test_r1_simtel_broken_pixels(r1_file, tmp_path, extractor):
    """Test that broken pixel information is preserved for simtel r1"""
    from ctapipe.calib.camera import CameraCalibrator
    from ctapipe.io import EventSource

    with EventSource(r1_file) as source:
        calibrator = CameraCalibrator(source.subarray, image_extractor_type=extractor)

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

                # check to dl1
                calibrator(event)
                assert event.dl1.tel[tel_id].image is not None

        assert n_checked > 0
