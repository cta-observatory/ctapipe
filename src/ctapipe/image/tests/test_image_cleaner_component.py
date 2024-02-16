import numpy as np
import pytest
from traitlets.config import Config

from ctapipe.image import ImageCleaner
from ctapipe.instrument import SubarrayDescription


@pytest.mark.parametrize("method", ImageCleaner.non_abstract_subclasses().keys())
def test_image_cleaner(method, prod5_mst_nectarcam, example_event, reference_location):
    """Test that we can construct and use a component-based ImageCleaner"""

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

    subarray = SubarrayDescription(
        name="test",
        tel_positions={1: None},
        tel_descriptions={1: prod5_mst_nectarcam},
        reference_location=reference_location,
    )

    clean = ImageCleaner.from_name(method, config=config, subarray=subarray)

    image = np.zeros_like(
        prod5_mst_nectarcam.camera.geometry.pix_x.value,
        dtype=np.float64,
    )
    image[10:30] = 20.0
    image[31:40] = 8.0
    times = np.linspace(-5, 10, image.shape[0])

    example_event.dl1.tel[1].image = image
    example_event.dl1.tel[1].peak_time = times

    mask = clean(tel_id=1, event=example_event)

    # we're not testing the algorithm here, just that it does something (for the
    # algorithm tests, see test_cleaning.py
    assert np.count_nonzero(mask) > 0


@pytest.mark.parametrize("method", ImageCleaner.non_abstract_subclasses().keys())
def test_image_cleaner_no_subarray(method):
    with pytest.raises(TypeError):
        ImageCleaner.from_name(method)
