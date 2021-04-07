"""
Tests for ShowerProcessor functionalities.
"""
from numpy import isfinite

from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageProcessor
from ctapipe.reco import ShowerProcessor


def test_shower_processor_geometry(example_event, example_subarray):
    """Ensure we get shower geometry when we input an event with parametrized images."""

    calibrate = CameraCalibrator(subarray=example_subarray)
    process_images = ImageProcessor(
        subarray=example_subarray,
        is_simulation=True,
        image_cleaner_type="MARSImageCleaner",
    )
    process_shower = ShowerProcessor(
        subarray=example_subarray,
        is_simulation=True,
        geometry=True,
        energy=False,
        classification=False)

    calibrate(example_event)
    process_images(example_event)

    # config = traitlets.config({
    #
    #     "ShowerQualityQuery" : "quality_criteria": [
    # ["enough_pixels", "lambda im: np.count_nonzero(im) > 3"]
    # ]
    #
    # })

    process_shower(example_event)

    for dl2 in example_event.dl2.shower.values():
        assert isfinite(event.dl2.shower['HillasReconstructor'].alt)
        assert isfinite(event.dl2.shower['HillasReconstructor'].az)
        assert isfinite(event.dl2.shower['HillasReconstructor'].core_x)
        assert isfinite(event.dl2.shower['HillasReconstructor'].core_x)
        assert isfinite(event.dl2.shower['HillasReconstructor'].core_y)
        assert event.dl2.shower['HillasReconstructor'].is_valid
        assert isfinite(event.dl2.shower['HillasReconstructor'].average_intensity)
