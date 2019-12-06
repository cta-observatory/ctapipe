from ctapipe.image import ImageCleaner
from ctapipe.instrument import TelescopeDescription, SubarrayDescription
from traitlets.config import Config
import numpy as np
import pytest

@pytest.mark.parametrize("method", ['TailcutsImageCleaner',])
def test_image_cleaner(method):
    """ Test that we can construct and use a component-based ImageCleaner"""

    config = Config({
        "TailcutsImageCleaner": {"boundary_threshold": 5.0, "picture_threshold": 10.0},
    })

    tel = TelescopeDescription.from_name("MST", "NectarCam")
    subarray = SubarrayDescription(
        name="test", tel_positions={1: None}, tel_descriptions={1: tel}
    )


    clean = ImageCleaner.from_name(method, config)

    image = np.zeros_like(tel.camera.pix_x.value, dtype=np.float)
    image[10:30] = 20.0
    image[31:40] = 8.0

    mask = clean(tel_id=1, subarray=subarray, image=image)

    assert np.count_nonzero(mask) == 22
