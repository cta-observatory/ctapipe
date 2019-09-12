import numpy as np
from numpy.testing import assert_array_equal
from ctapipe.image.reducer import NullDataVolumeReducer
from ctapipe.image.reducer import TailCutsDataVolumeReducer
from ctapipe.instrument import CameraGeometry


def test_null_data_volume_reducer():
    waveforms = np.random.uniform(0, 1, (2048, 96))
    reducer = NullDataVolumeReducer()
    reduced_waveforms = reducer(waveforms)
    assert_array_equal(waveforms, reduced_waveforms)


def test_tailcuts_data_volume_reducer():
    # create set of waveforms and expected masks
    geom = CameraGeometry.from_name("LSTCam")
    image = np.zeros_like(geom.pix_id, dtype=np.float)
    mask = np.zeros_like(geom.pix_id, dtype=np.bool)
    expected_masks = np.empty([geom.pix_id.shape[0], 0], dtype=bool)
    created_waveforms = np.empty([geom.pix_id.shape[0], 0], dtype=float)
    # created set of pixels are connected in one line, not in a blob
    image[9] = 10.0
    # Should be selected as core-pixel from Step 1) tailcuts_clean
    image[[10, 8, 6, 5]] = 4.0
    mask[[10, 9, 8, 6, 5]] = True
    # 10 and 8 as boundary-pixel from Step 1) tailcuts_clean
    # 6 and 5 as iteration-pixel in Step 2)
    mask[geom.neighbor_matrix_sparse.dot(mask)] = True
    # pixels from dilate at the end in Step 3)

    for i in range(42):  # more dim
        expected_masks = np.column_stack((expected_masks, mask))
        created_waveforms = np.column_stack((created_waveforms, image))

    expected_return = np.ma.masked_array(created_waveforms,
                                         mask=expected_masks)
    reducer = TailCutsDataVolumeReducer()
    reduced_waveforms = reducer(
        geom=geom,
        waveforms=created_waveforms,
        picture_thresh=7,
        boundary_thresh=3,
        iteration_thresh=3,
        end_dilates=1
    )
    assert_array_equal(expected_return, reduced_waveforms)
