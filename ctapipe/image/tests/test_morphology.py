import numpy as np
from numpy.testing import assert_allclose
from ctapipe.instrument import CameraGeometry


def test_number_of_islands():
    from ctapipe.image import number_of_islands

    # test with LST geometry (1855 pixels)
    geom = CameraGeometry.from_name("LSTCam")

    # create 18 triggered pixels grouped to 5 clusters
    mask = np.zeros(geom.n_pixels).astype("bool")
    triggered_pixels = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 37, 38, 111, 222]
    )
    mask[triggered_pixels] = True

    island_labels_true = np.zeros(geom.n_pixels, dtype=np.int16)
    island_labels_true[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]] = 1
    island_labels_true[14] = 2
    island_labels_true[[37, 38]] = 3
    island_labels_true[111] = 4
    island_labels_true[222] = 5

    n_islands, island_labels = number_of_islands(geom, mask)
    n_islands_true = 5
    # non cleaning pixels should be zero
    assert np.all(island_labels[~mask] == 0)
    # all other should have some label
    assert np.all(island_labels[mask] > 0)

    assert n_islands == n_islands_true
    assert_allclose(island_labels, island_labels_true)
    assert island_labels.dtype == np.int16


def test_number_of_island_sizes():
    from ctapipe.image import number_of_island_sizes

    island_labels = np.array(
        100 * [0]
        + 2 * [1]
        + 2 * [2]
        + 3 * [3]
        + 49 * [4]
        + 51 * [5]
        + 3 * [6]
        + 100 * [7]
        + [8]
        + 2 * [9]
        + [12]
    )

    n_small, n_medium, n_large = number_of_island_sizes(island_labels)
    assert n_small == 5
    assert n_medium == 3
    assert n_large == 2


def test_largest_island():
    """Test selection of largest island in imagea with given cleaning masks."""
    from ctapipe.image import number_of_islands, largest_island

    # Create a simple rectangular camera made of 17 pixels
    camera = CameraGeometry.make_rectangular(17, 1)

    # Assume a simple image (flattened array) made of 0, 1 or 2 photoelectrons
    # [2, 1, 1, 1, 1, 2, 2, 1, 0, 2, 1, 1, 1, 0, 2, 2, 2]
    # Assume a virtual tailcut cleaning that requires:
    # - picture_threshold = 2 photoelectrons,
    # - boundary_threshold = 1 photoelectron,
    # - min_number_picture_neighbors = 0
    # There will be 4 islands left after cleaning:
    clean_mask = np.zeros(camera.n_pixels).astype("bool")  # initialization
    clean_mask[[0, 1]] = 1
    clean_mask[[4, 5, 6, 7]] = 2  # this is the biggest
    clean_mask[[9, 10]] = 3
    clean_mask[[14, 15, 16]] = 4
    # Label islands (their number is not important here)
    _, islands_labels = number_of_islands(camera, clean_mask)
    # Create the true mask which takes into account only the biggest island
    # Pixels with no signal are labelled with a 0
    true_mask_largest = np.zeros(camera.n_pixels).astype("bool")
    true_mask_largest[[4, 5, 6, 7]] = 1
    # Apply the function to test
    mask_largest = largest_island(islands_labels)

    # Now the degenerate case of only one island surviving
    # Same process as before
    clean_mask_one = np.zeros(camera.n_pixels).astype("bool")
    clean_mask_one[[0, 1]] = 1
    _, islands_labels_one = number_of_islands(camera, clean_mask_one)
    true_mask_largest_one = np.zeros(camera.n_pixels).astype("bool")
    true_mask_largest_one[[0, 1]] = 1
    mask_largest_one = largest_island(islands_labels_one)

    # Last the case of no islands surviving
    clean_mask_0 = np.zeros(camera.n_pixels).astype("bool")
    _, islands_labels_0 = number_of_islands(camera, clean_mask_0)
    true_mask_largest_0 = np.zeros(camera.n_pixels).astype("bool")
    mask_largest_0 = largest_island(islands_labels_0)

    # Check if the function recovers the ground truth in all of the three cases
    assert (mask_largest_one == true_mask_largest_one).all()
    assert (mask_largest_0 == true_mask_largest_0).all()
    assert_allclose(mask_largest, true_mask_largest)
