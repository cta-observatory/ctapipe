import numpy as np
import pytest
from traitlets.config import Config

from ctapipe.image import ImageCleaner
from ctapipe.instrument import TelescopeDescription, SubarrayDescription


@pytest.mark.parametrize("method", ImageCleaner.non_abstract_subclasses().keys())
def test_image_cleaner(method):
    """ Test that we can construct and use a component-based ImageCleaner"""

    config = Config(
        {
            "TailcutsImageCleaner": {
                "boundary_threshold_pe": 5.0,
                "picture_threshold_pe": 10.0,
            },
            "MARSImageCleaner": {
                "boundary_threshold_pe": 5.0,
                "picture_threshold_pe": 10.0,
            },
            "FACTImageCleaner": {
                "boundary_threshold_pe": 5.0,
                "picture_threshold_pe": 10.0,
                "time_limit_ns": 6.0,
            },
        }
    )

    tel = TelescopeDescription.from_name("MST", "NectarCam")
    subarray = SubarrayDescription(
        name="test", tel_positions={1: None}, tel_descriptions={1: tel}
    )

    clean = ImageCleaner.from_name(method, config=config, subarray=subarray)

    image = np.zeros_like(tel.camera.geometry.pix_x.value, dtype=np.float)
    image[10:30] = 20.0
    image[31:40] = 8.0
    times = np.linspace(-5, 10, image.shape[0])

    mask = clean(tel_id=1, image=image, arrival_times=times)

    # we're not testing the algorithm here, just that it does something (for the
    # algorithm tests, see test_cleaning.py
    assert np.count_nonzero(mask) > 0


@pytest.mark.parametrize("method", ImageCleaner.non_abstract_subclasses().keys())
def test_image_cleaner_no_subarray(method):
    with pytest.raises(TypeError):
        ImageCleaner.from_name(method)
