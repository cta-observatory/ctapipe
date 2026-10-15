import numpy as np
import pytest
from traitlets.config import Config

from ctapipe.containers import CameraMonitoringContainer, PixelStatus
from ctapipe.image import ImageCleaner
from ctapipe.instrument import SubarrayDescription


@pytest.mark.parametrize("method", ImageCleaner.non_abstract_subclasses().keys())
def test_image_cleaner(method, prod5_mst_nectarcam, reference_location):
    """Test that we can construct and use a component-based ImageCleaner"""

    config = Config(
        {
            "TailcutsImageCleaner": {
                "boundary_threshold_pe": 5.0,
                "picture_threshold_pe": 10.0,
            },
            "NSBImageCleaner": {
                "boundary_threshold_pe": 5.0,
                "picture_threshold_pe": 10.0,
                "time_limit": None,
                "bright_cleaning_threshold": None,
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

    monitoring = CameraMonitoringContainer()
    monitoring.pixel_statistics.pedestal_image.std = np.ones(
        prod5_mst_nectarcam.camera.geometry.n_pixels
    )

    clean = ImageCleaner.from_name(method, config=config, subarray=subarray)

    image = np.zeros_like(
        prod5_mst_nectarcam.camera.geometry.pix_x.value,
        dtype=np.float64,
    )
    image[10:30] = 20.0
    image[31:40] = 8.0
    times = np.linspace(-5, 10, image.shape[0])

    mask = clean(
        tel_id=1,
        image=image,
        arrival_times=times,
        monitoring=monitoring,
    )

    # we're not testing the algorithm here, just that it does something (for the
    # algorithm tests, see test_cleaning.py
    assert np.count_nonzero(mask) > 0


@pytest.mark.parametrize("method", ImageCleaner.non_abstract_subclasses().keys())
def test_image_cleaner_no_subarray(method):
    with pytest.raises(TypeError):
        ImageCleaner.from_name(method)


@pytest.mark.parametrize(
    ("camera_fixture", "pedestal_case"),
    [
        ("prod5_mst_flashcam", "single-gain"),
        ("prod5_mst_nectarcam", "gain-selected"),
        ("prod5_mst_nectarcam", "missing"),
        ("prod5_mst_nectarcam", "1d"),
        ("prod5_mst_nectarcam", "singleton"),
    ],
)
def test_nsb_image_cleaner_pedestal_shape(
    camera_fixture, pedestal_case, request, reference_location
):
    telescope = request.getfixturevalue(camera_fixture)
    subarray = SubarrayDescription(
        name="test",
        tel_positions={1: None},
        tel_descriptions={1: telescope},
        reference_location=reference_location,
    )
    clean = ImageCleaner.from_name(
        "NSBImageCleaner",
        subarray=subarray,
        picture_threshold_pe=10.0,
        boundary_threshold_pe=30.0,
        keep_isolated_pixels=True,
        time_limit=None,
        bright_cleaning_threshold=None,
        pedestal_factor=2.5,
    )
    n_pixels = telescope.camera.geometry.n_pixels
    image = np.full(n_pixels, 20.0)
    monitoring = CameraMonitoringContainer()
    expected = np.ones(n_pixels, dtype=bool)
    pixel_status = np.full(n_pixels, PixelStatus.HIGH_GAIN_STORED, dtype=np.uint8)
    if pedestal_case != "missing":
        pedestal_std = np.full(n_pixels, 4.0)
        pedestal_std[::2] = 12.0
        expected[::2] = False
        if pedestal_case in {"single-gain", "singleton"}:
            pedestal_std = pedestal_std[np.newaxis, :]
        elif pedestal_case == "gain-selected":
            # Different thresholds in the two channels reveal incorrect selection.
            pedestal_std = np.stack([pedestal_std, 16.0 - pedestal_std])
            expected = np.stack([expected, ~expected])
            low_gain = np.arange(n_pixels) % 3 == 0
            pixel_status[low_gain] = np.uint8(PixelStatus.LOW_GAIN_STORED)
            expected = expected[low_gain.astype(int), np.arange(n_pixels)]
        monitoring.pixel_statistics.pedestal_image.std = pedestal_std

    mask = clean(
        tel_id=1,
        image=image,
        monitoring=monitoring,
        pixel_status=pixel_status,
    )

    np.testing.assert_array_equal(mask, expected)
