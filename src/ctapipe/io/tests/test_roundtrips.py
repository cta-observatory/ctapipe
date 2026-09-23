"""Test assuring combining several steps keeps data consistent"""

import numpy as np
import pytest

from ctapipe.containers import PixelStatus
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
def test_r1_simtel_broken_pixels(r1_file, extractor):
    """Test that broken pixel information is preserved for simtel r1"""
    from ctapipe.calib.camera import CameraCalibrator
    from ctapipe.io import EventSource

    with EventSource(r1_file) as source:
        calibrator = CameraCalibrator(source.subarray, image_extractor_type=extractor)

        n_checked = 0
        for event in source:
            for tel_id, r1 in event.r1.tel.items():
                readout = source.subarray.tel[tel_id].camera.readout

                assert r1.pixel_status is not None
                assert r1.pixel_status.shape == (readout.n_pixels,)

                # flashcam has 6 disabled pixels, only one gain
                if readout.name == "FlashCam":
                    invalid = PixelStatus.is_invalid(r1.pixel_status)
                    assert np.count_nonzero(invalid) == 6
                # we expect no other cameras to have disabled pixels here
                else:
                    selected_gain_channel = r1.selected_gain_channel
                    gain_bits = np.where(
                        selected_gain_channel == 0,
                        np.uint8(PixelStatus.HIGH_GAIN_STORED),
                        np.uint8(PixelStatus.LOW_GAIN_STORED),
                    )
                    active_gain_stored = (r1.pixel_status & gain_bits) != 0
                    assert np.all(active_gain_stored)

                n_checked += 1

                # check to dl1
                calibrator(event)
                assert event.dl1.tel[tel_id].image is not None

        assert n_checked > 0
