"""
Tests for ImageProcessor functionality
"""

import numpy as np
import pytest
from numpy import isfinite

from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageCleaner, ImageProcessor


@pytest.mark.parametrize("cleaner", ImageCleaner.non_abstract_subclasses().values())
def test_image_processor(cleaner, example_event, example_subarray):
    """ensure we get parameters out when we input an event with images"""

    calibrate = CameraCalibrator(subarray=example_subarray)
    process_images = ImageProcessor(
        subarray=example_subarray, image_cleaner_type=cleaner.__name__
    )

    assert isinstance(process_images.clean, cleaner)

    calibrate(example_event)
    process_images(example_event)

    for dl1tel in example_event.dl1.tel.values():
        n_survived_pixels = dl1tel.image_mask.sum()
        assert isfinite(n_survived_pixels)
        if n_survived_pixels > 1:
            assert isfinite(dl1tel.parameters.hillas.length.value)
            dl1tel.parameters.hillas.length.to("deg")
            assert isfinite(dl1tel.parameters.timing.slope.value)
            assert isfinite(dl1tel.parameters.leakage.pixels_width_1)
            assert isfinite(dl1tel.parameters.concentration.cog)
            assert isfinite(dl1tel.parameters.morphology.n_pixels)
            assert isfinite(dl1tel.parameters.intensity_statistics.max)
            assert isfinite(dl1tel.parameters.peak_time_statistics.max)
        else:
            assert np.isnan(dl1tel.parameters.hillas.length.value)

    process_images.check_image.to_table()
