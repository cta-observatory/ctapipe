import numpy as np
from numpy.testing import assert_allclose
from ctapipe.image import cleaning
from ctapipe.instrument import CameraGeometry


def test_tailcuts_clean_simple():
    geom = CameraGeometry.from_name("LSTCam")
    image = np.zeros_like(geom.pix_id, dtype=np.float)

    num_pix = 40
    some_neighs = geom.neighbors[num_pix][0:3]  # pick 3 neighbors
    image[num_pix] = 5.0  # set a single image pixel
    image[some_neighs] = 3.0  # make some boundaries that are neighbors
    image[10] = 3.0  # a boundary that is not a neighbor

    mask = cleaning.tailcuts_clean(geom, image, picture_thresh=4.5, boundary_thresh=2.5)

    assert 10 not in geom.pix_id[mask]
    assert set(some_neighs).union({num_pix}) == set(geom.pix_id[mask])
    assert np.count_nonzero(mask) == 4


def test_dilate():
    geom = CameraGeometry.from_name("LSTCam")
    mask = np.zeros_like(geom.pix_id, dtype=bool)

    mask[100] = True  # a single pixel far from a border is true.
    assert np.count_nonzero(mask) == 1

    # dilate a single row
    dmask = cleaning.dilate(geom, mask)
    assert np.count_nonzero(dmask) == 1 + 6

    # dilate a second row
    dmask = cleaning.dilate(geom, dmask)
    assert np.count_nonzero(dmask) == 1 + 6 + 12

    # dilate a third row
    dmask = cleaning.dilate(geom, dmask)
    assert np.count_nonzero(dmask) == 1 + 6 + 12 + 18


def test_tailcuts_clean():
    # start with simple 3-pixel camera
    geom = CameraGeometry.make_rectangular(3, 1, (-1, 1))

    p = 15  # picture value
    b = 7  # boundary value

    # for no isolated pixels:
    testcases = {
        (p, p, 0): [True, True, False],
        (p, 0, p): [False, False, False],
        (p, b, p): [True, True, True],
        (p, b, 0): [True, True, False],
        (b, b, 0): [False, False, False],
        (0, p, 0): [False, False, False],
    }

    for image, mask in testcases.items():
        result = cleaning.tailcuts_clean(
            geom,
            np.array(image),
            picture_thresh=15,
            boundary_thresh=5,
            keep_isolated_pixels=False,
        )
        assert (result == mask).all()


def test_mars_cleaning_1st_pass():
    """Test the MARS-like cleaning 1st pass."""

    # start with simple 3-pixel camera
    geom = CameraGeometry.make_rectangular(3, 1, (-1, 1))

    p = 10  # picture value
    b1 = 7  # 1st boundary value
    b2 = 7  # 2nd boundary value

    # The following test-cases are a superset of those used to test
    # 'tailcuts_clean': since 'mars_image_cleaning' uses it internally, so it
    # has to be backwards-compatible.

    # for no isolated pixels and min_number_picture_neighbors = 0 :
    testcases = {
        # from 'tailcuts_clean', for backwards-compatibility
        (p, p, p): [True, True, True],
        (p, p, 0): [True, True, False],
        (p, 0, p): [False, False, False],
        (0, p, 0): [False, False, False],
        # as in 'tailcuts_clean', but specifying 1st boundary threshold
        (p, b1, p): [True, True, True],
        (p, b1, 0): [True, True, False],
        (b1, b1, 0): [False, False, False],
        (b1, b1, b1): [False, False, False],
        (0, b1, 0): [False, False, False],  # to put in test_tailcuts_clean!
        # specific for 'mars_image_cleaning'
        (p, b2, p): [True, True, True],
        (p, b2, 0): [True, True, False],
        (b1, b2, 0): [False, False, False],
        (b2, b2, 0): [False, False, False],
        (0, b2, 0): [False, False, False],
        (p, b1, b2): [True, True, True],
        (p, b2, b1): [True, True, True],
        (p, b1, b1): [True, True, True],
        (p, b2, b2): [True, True, True],
        (p, b1, b2 - 1): [True, True, False],
    }

    for image, mask in testcases.items():
        result = cleaning.mars_cleaning_1st_pass(
            geom,
            np.array(image),
            picture_thresh=10,
            boundary_thresh=7,
            keep_isolated_pixels=False,
        )
        assert (result == mask).all()


def test_tailcuts_clean_min_neighbors_1():
    """
    requiring that picture pixels have at least one neighbor above picture_thres:
    """

    # start with simple 3-pixel camera
    geom = CameraGeometry.make_rectangular(3, 1, (-1, 1))

    p = 15  # picture value
    b = 7  # boundary value

    testcases = {
        (p, p, 0): [True, True, False],
        (p, 0, p): [False, False, False],
        (p, b, p): [False, False, False],
        (p, b, 0): [False, False, False],
        (b, b, 0): [False, False, False],
        (0, p, 0): [False, False, False],
        (p, p, p): [True, True, True],
    }

    for image, mask in testcases.items():
        result = cleaning.tailcuts_clean(
            geom,
            np.array(image),
            picture_thresh=15,
            boundary_thresh=5,
            min_number_picture_neighbors=1,
            keep_isolated_pixels=False,
        )
        assert (result == mask).all()


def test_tailcuts_clean_min_neighbors_2():
    """ requiring that picture pixels have at least two neighbors above
    picture_thresh"""

    # start with simple 3-pixel camera
    geom = CameraGeometry.make_rectangular(3, 1, (-1, 1))

    p = 15  # picture value
    b = 7  # boundary value

    testcases = {
        (p, p, 0): [False, False, False],
        (p, 0, p): [False, False, False],
        (p, b, p): [False, False, False],
        (p, b, 0): [False, False, False],
        (b, b, 0): [False, False, False],
        (p, p, p): [True, True, True],
    }

    for image, mask in testcases.items():
        result = cleaning.tailcuts_clean(
            geom,
            np.array(image),
            picture_thresh=15,
            boundary_thresh=5,
            min_number_picture_neighbors=2,
            keep_isolated_pixels=False,
        )
        assert (result == mask).all()


def test_tailcuts_clean_with_isolated_pixels():
    # start with simple 3-pixel camera
    geom = CameraGeometry.make_rectangular(3, 1, (-1, 1))

    p = 15  # picture value
    b = 7  # boundary value

    testcases = {
        (p, p, 0): [True, True, False],
        (p, 0, p): [True, False, True],
        (p, b, p): [True, True, True],
        (p, b, 0): [True, True, False],
        (0, p, 0): [False, True, False],
        (b, b, 0): [False, False, False],
    }

    for image, mask in testcases.items():
        result = cleaning.tailcuts_clean(
            geom,
            np.array(image),
            picture_thresh=15,
            boundary_thresh=5,
            keep_isolated_pixels=True,
        )
        assert (result == mask).all()


def test_number_of_islands():
    # test with LST geometry (1855 pixels)
    geom = CameraGeometry.from_name("LSTCam")

    # create 18 triggered pixels grouped to 5 clusters
    island_mask_true = np.zeros(geom.n_pixels)
    mask = np.zeros(geom.n_pixels).astype("bool")
    triggered_pixels = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 37, 38, 111, 222]
    )
    island_mask_true[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]] = 1
    island_mask_true[14] = 2
    island_mask_true[[37, 38]] = 3
    island_mask_true[111] = 4
    island_mask_true[222] = 5
    mask[triggered_pixels] = 1

    n_islands, island_mask = cleaning.number_of_islands(geom, mask)
    n_islands_true = 5
    assert n_islands == n_islands_true
    assert_allclose(island_mask, island_mask_true)
    assert island_mask.dtype == np.int32


def test_fact_image_cleaning():
    # use LST pixel geometry
    geom = CameraGeometry.from_name("LSTCam")
    # create some signal pixels
    values = np.zeros(len(geom))
    timing = np.zeros(len(geom))
    signal_pixels = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 37, 38, 111, 222]
    )
    values[signal_pixels] = 5
    timing[signal_pixels] = 10
    # manipulate some of those
    values[[1, 2]] = 3
    values[7] = 1
    timing[[5, 6, 13, 111]] = 20

    mask = cleaning.fact_image_cleaning(
        geom,
        values,
        timing,
        boundary_threshold=2,
        picture_threshold=4,
        min_number_neighbors=2,
        time_limit=5,
    )

    expected_pixels = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11])
    expected_mask = np.zeros(len(geom)).astype(bool)
    expected_mask[expected_pixels] = 1
    assert_allclose(mask, expected_mask)


def test_apply_time_delta_cleaning():
    geom = CameraGeometry.from_name("LSTCam")
    pulse_time = np.zeros(geom.n_pixels, dtype=np.float)

    pixel = 40
    neighbours = geom.neighbors[pixel]
    pulse_time[neighbours] = 32.0
    pulse_time[pixel] = 30.0
    mask = pulse_time > 0

    # Test unchanged
    td_mask = cleaning.apply_time_delta_cleaning(
        geom,
        mask,
        pulse_time,
        min_number_neighbors=1,
        time_limit=5,
    )
    test_mask = mask.copy()
    assert (test_mask == td_mask).all()

    # Test time_limit
    noise_neighbour = neighbours[0]
    pulse_time[noise_neighbour] += 10
    td_mask = cleaning.apply_time_delta_cleaning(
        geom,
        mask,
        pulse_time,
        min_number_neighbors=1,
        time_limit=5,
    )
    test_mask = mask.copy()
    test_mask[noise_neighbour] = 0
    assert (test_mask == td_mask).all()

    # Test min_number_neighbours
    td_mask = cleaning.apply_time_delta_cleaning(
        geom,
        mask,
        pulse_time,
        min_number_neighbors=4,
        time_limit=5,
    )
    test_mask = mask.copy()
    test_mask[neighbours] = 0
    assert (test_mask == td_mask).all()
