"""
Tests for ImageProcessor functionality
"""
from numpy import isfinite

from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageProcessor
from ctapipe.image.cleaning import MARSImageCleaner


def test_image_processor(example_event, example_subarray):
    """ ensure we get parameters out when we input an event with images """

    calibrate = CameraCalibrator(subarray=example_subarray)
    process_images = ImageProcessor(
        subarray=example_subarray,
        is_simulation=True,
        image_cleaner_type="MARSImageCleaner",
    )

    assert isinstance(process_images.clean, MARSImageCleaner)

    calibrate(example_event)
    process_images(example_event)

    for dl1tel in example_event.dl1.tel.values():
        assert isfinite(dl1tel.image_mask.sum())
        assert isfinite(dl1tel.parameters.hillas.length.value)
        assert isfinite(dl1tel.parameters.timing.slope.value)
        assert isfinite(dl1tel.parameters.leakage.pixels_width_1)
        assert isfinite(dl1tel.parameters.concentration.cog)
        assert isfinite(dl1tel.parameters.morphology.num_pixels)
        assert isfinite(dl1tel.parameters.intensity_statistics.max)
        assert isfinite(dl1tel.parameters.peak_time_statistics.max)

    process_images.check_image.to_table()
