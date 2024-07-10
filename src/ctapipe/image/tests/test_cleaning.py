import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose

from ctapipe.image import cleaning
from ctapipe.instrument import CameraGeometry


def test_tailcuts_clean_simple(prod5_lst):
    geom = prod5_lst.camera.geometry
    image = np.zeros_like(geom.pix_id, dtype=np.float64)

    n_pix = 40
    some_neighs = geom.neighbors[n_pix][0:3]  # pick 3 neighbors
    image[n_pix] = 5.0  # set a single image pixel
    image[some_neighs] = 3.0  # make some boundaries that are neighbors
    image[10] = 3.0  # a boundary that is not a neighbor

    mask = cleaning.tailcuts_clean(geom, image, picture_thresh=4.5, boundary_thresh=2.5)

    assert 10 not in geom.pix_id[mask]
    assert set(some_neighs).union({n_pix}) == set(geom.pix_id[mask])
    assert np.count_nonzero(mask) == 4


def test_dilate(prod5_lst):
    geom = prod5_lst.camera.geometry
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


def test_tailcuts_clean_threshold_array(prod5_lst):
    """Tests that tailcuts can also work with individual thresholds per pixel"""
    rng = np.random.default_rng(1337)
    geom = prod5_lst.camera.geometry

    # artificial event having a "shower" and a "star" at these locations
    star_x = 0.5 * u.m
    star_y = 0.5 * u.m
    shower_x = -0.5 * u.m
    shower_y = -0.5 * u.m

    star_pixels = (
        np.sqrt((geom.pix_x - star_x) ** 2 + (geom.pix_y - star_y) ** 2) < 0.1 * u.m
    )
    shower = (
        np.sqrt((geom.pix_x - shower_x) ** 2 + (geom.pix_y - shower_y) ** 2) < 0.2 * u.m
    )

    # noise level at the star cluster is much higher than normal camera
    noise = rng.normal(3, 0.2, len(geom))
    noise[star_pixels] = rng.normal(10, 1, np.count_nonzero(star_pixels))

    # large signal at the signal location
    image = rng.poisson(noise).astype(float)
    signal = rng.normal(20, 2, np.count_nonzero(shower))
    image[shower] += signal

    picture_threshold = 3 * noise
    boundary_threshold = 1.5 * noise

    # test that normal cleaning also contains star cluster
    # and that cleaning with pixel wise values removes star cluster
    normal_cleaning = cleaning.tailcuts_clean(
        geom, image, picture_threshold.mean(), boundary_threshold.mean()
    )
    pixel_cleaning = cleaning.tailcuts_clean(
        geom, image, picture_threshold, boundary_threshold
    )

    assert np.count_nonzero(normal_cleaning & star_pixels) > 0
    assert np.count_nonzero(pixel_cleaning & star_pixels) == 0


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
    """requiring that picture pixels have at least two neighbors above
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


def test_fact_image_cleaning(prod3_lst):
    # use LST pixel geometry
    geom = prod3_lst.camera.geometry
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

    expected_pixels = np.array([0, 1, 2, 3, 4, 8, 9])
    expected_mask = np.zeros(len(geom)).astype(bool)
    expected_mask[expected_pixels] = 1
    assert_allclose(mask, expected_mask)


def test_apply_time_delta_cleaning(prod3_lst):
    geom = prod3_lst.camera.geometry
    peak_time = np.zeros(geom.n_pixels, dtype=np.float64)

    pixel = 40
    neighbors = geom.neighbors[pixel]
    peak_time[neighbors] = 32.0
    peak_time[pixel] = 30.0
    mask = peak_time > 0

    # Test unchanged
    td_mask = cleaning.apply_time_delta_cleaning(
        geom, mask, peak_time, min_number_neighbors=1, time_limit=5
    )
    test_mask = mask.copy()
    assert (test_mask == td_mask).all()

    # Test time_limit
    noise_neighbor = neighbors[0]
    peak_time[noise_neighbor] += 10
    td_mask = cleaning.apply_time_delta_cleaning(
        geom, mask, peak_time, min_number_neighbors=1, time_limit=5
    )
    test_mask = mask.copy()
    test_mask[noise_neighbor] = 0
    assert (test_mask == td_mask).all()

    # Test min_number_neighbors
    td_mask = cleaning.apply_time_delta_cleaning(
        geom, mask, peak_time, min_number_neighbors=4, time_limit=5
    )
    test_mask = mask.copy()
    test_mask[neighbors] = 0
    assert (test_mask == td_mask).all()

    # Test unselected neighbors
    mask[156] = 0
    peak_time[noise_neighbor] -= 10
    td_mask = cleaning.apply_time_delta_cleaning(
        geom, mask, peak_time, min_number_neighbors=3, time_limit=5
    )
    test_mask = mask.copy()
    test_mask[[41, 157]] = 0
    assert (test_mask == td_mask).all()


def test_time_constrained_clean(prod5_lst):
    geom = prod5_lst.camera.geometry
    charge = np.zeros(geom.n_pixels, dtype=np.float64)
    peak_time = np.zeros(geom.n_pixels, dtype=np.float64)

    # define signal pixels and their charges/timings (1 core pixel + 6 neighboring core pixels + 12 neighboring boundary pixels)
    core_pixel = 100
    core_neighbors = geom.neighbors[core_pixel]
    boundary_pixels = np.setdiff1d(
        np.array([geom.neighbors[core_neighbor] for core_neighbor in core_neighbors]),
        np.append(core_neighbors, core_pixel),
    )
    charge[core_pixel], charge[core_neighbors], charge[boundary_pixels] = 15, 10, 6
    peak_time[core_pixel], peak_time[core_neighbors], peak_time[boundary_pixels] = (
        18,
        20,
        21,
    )

    # define initial cleaning parameters
    picture_thresh, boundary_thresh = 8, 4
    time_limit_core, time_limit_boundary = 4.5, 1.5
    min_number_picture_neighbors = 1

    mask_signal = charge > 0

    # 1. basic test
    mask_reco = cleaning.time_constrained_clean(
        geom,
        charge,
        peak_time,
        picture_thresh=picture_thresh,
        boundary_thresh=boundary_thresh,
        time_limit_core=time_limit_core,
        time_limit_boundary=time_limit_boundary,
        min_number_picture_neighbors=min_number_picture_neighbors,
    )
    test_mask = mask_signal.copy()
    assert (test_mask == mask_reco).all()

    # 2. increased min_number_picture_neighbors test (here 3)
    min_number_picture_neighbors = 3
    mask_reco = cleaning.time_constrained_clean(
        geom,
        charge,
        peak_time,
        picture_thresh=picture_thresh,
        boundary_thresh=boundary_thresh,
        time_limit_core=time_limit_core,
        time_limit_boundary=time_limit_boundary,
        min_number_picture_neighbors=min_number_picture_neighbors,
    )
    # removed pixels : boundary pixels
    test_mask = mask_signal.copy()
    test_mask[boundary_pixels] = 0
    assert (test_mask == mask_reco).all()

    # 3. strict time_limit_boundary test (here 0.5)
    min_number_picture_neighbors = 1
    time_limit_boundary = 0.5
    mask_reco = cleaning.time_constrained_clean(
        geom,
        charge,
        peak_time,
        picture_thresh=picture_thresh,
        boundary_thresh=boundary_thresh,
        time_limit_core=time_limit_core,
        time_limit_boundary=time_limit_boundary,
        min_number_picture_neighbors=min_number_picture_neighbors,
    )
    # removed pixels : boundary pixels
    test_mask = mask_signal.copy()
    test_mask[boundary_pixels] = 0
    assert (test_mask == mask_reco).all()

    # 4. time_limit_core test (one of core_neighbors have peak time >5 slice away from the average)
    time_limit_boundary = 1.5
    noise_core_neighbor = core_neighbors[0]
    peak_time[noise_core_neighbor] = 25
    mask_reco = cleaning.time_constrained_clean(
        geom,
        charge,
        peak_time,
        picture_thresh=picture_thresh,
        boundary_thresh=boundary_thresh,
        time_limit_core=time_limit_core,
        time_limit_boundary=time_limit_boundary,
        min_number_picture_neighbors=min_number_picture_neighbors,
    )
    # removed pixels : the noise core neighbor pixel + one neighboring boundary
    test_mask = mask_signal.copy()
    test_mask[noise_core_neighbor] = 0
    noise_boundary = np.setdiff1d(
        geom.neighbors[noise_core_neighbor],
        np.array(
            [geom.neighbors[core_neighbor] for core_neighbor in core_neighbors[1:]]
        ),
    )
    test_mask[noise_boundary] = 0
    assert (test_mask == mask_reco).all()

    # 5. time_limit_core test for brighter pixels (one of core_neighbors have peak time >5 slice away from the average)
    charge[core_pixel], charge[core_neighbors] = 30, 20
    mask_reco = cleaning.time_constrained_clean(
        geom,
        charge,
        peak_time,
        picture_thresh=picture_thresh,
        boundary_thresh=boundary_thresh,
        time_limit_core=time_limit_core,
        time_limit_boundary=time_limit_boundary,
        min_number_picture_neighbors=min_number_picture_neighbors,
    )
    # removed pixels : one neighboring boundary to the noise core pixel
    test_mask = mask_signal.copy()
    noise_boundary = np.setdiff1d(
        geom.neighbors[noise_core_neighbor],
        np.array(
            [geom.neighbors[core_neighbor] for core_neighbor in core_neighbors[1:]]
        ),
    )
    test_mask[noise_boundary] = 0
    assert (test_mask == mask_reco).all()


def test_bright_cleaning():
    n_pixels = 1855
    fraction = 0.5
    image = np.zeros(n_pixels)
    # set 3 bright pixels to 100 so mean of them is 100 as well
    image[:3] = 100
    # 10 pixels above fraction * mean
    image[10:20] = 60
    # 15 pixels below fraction * mean
    image[50:65] = 30

    threshold = 90
    mask = cleaning.bright_cleaning(image, threshold, fraction, n_pixels=3)
    assert np.count_nonzero(mask) == 3 + 10
    # test that it doesn't select any pixels if mean of the 3 brightest pixels
    # is below threshold
    threshold = 110
    mask = cleaning.bright_cleaning(image, threshold, fraction, n_pixels=3)
    assert np.count_nonzero(~mask) == 0


def test_nsb_image_cleaning(prod5_lst):
    geom = prod5_lst.camera.geometry
    charge = np.zeros(geom.n_pixels, dtype=np.float32)
    peak_time = np.zeros(geom.n_pixels, dtype=np.float32)

    core_pixel_1 = 100
    neighbors_1 = geom.neighbors[core_pixel_1]
    core_pixel_2 = 1100
    neighbors_2 = geom.neighbors[core_pixel_2]

    # Set core pixel to 50 and first row of neighbors to 29.
    # These two islands does not overlap.
    charge[neighbors_1] = 29
    charge[core_pixel_1] = 50
    charge[neighbors_2] = 29
    charge[core_pixel_2] = 50

    args = {
        "picture_thresh_min": 45,
        "boundary_thresh": 20,
        "keep_isolated_pixels": True,
        "time_limit": None,
        "bright_cleaning_n_pixels": 3,
        "bright_cleaning_fraction": 0.9,
        "bright_cleaning_threshold": None,
        "largest_island_only": False,
        "pedestal_factor": 2,
        "pedestal_std": None,
    }

    # return normal tailcuts cleaning result if every other step is None/False:
    mask = cleaning.nsb_image_cleaning(geom, charge, peak_time, **args)
    # 2 * (1 core and 6 boundary_pixels) should be selected
    assert np.count_nonzero(mask) == 2 * (1 + 6)

    # Test that tailcuts core threshold is adapted correctly with the pedestal
    # charge std.
    pedestal_std = np.ones(geom.n_pixels)
    pedestal_std[core_pixel_1] = 30
    args["pedestal_std"] = pedestal_std

    mask = cleaning.nsb_image_cleaning(geom, charge, peak_time, **args)
    # only core_pixel_2 with its boundaries should be selected since
    # pedestal_std[core_pixel_1] * pedestal_factor > charge[core_pixel_1]
    assert np.count_nonzero(mask) == 1 + 6

    # if no pixel survives tailcuts cleaning it should not select any pixel
    pedestal_std[core_pixel_2] = 30
    args["pedestal_std"] = pedestal_std

    mask = cleaning.nsb_image_cleaning(geom, charge, peak_time, **args)
    assert np.count_nonzero(mask) == 0

    # deselect core_pixel_1 with time_delta_cleaning by setting all its neighbors
    # peak_time to 10
    peak_time[neighbors_1] = 10
    args["pedestal_std"] = None
    args["time_limit"] = 3

    mask = cleaning.nsb_image_cleaning(geom, charge, peak_time, **args)
    assert np.count_nonzero(mask) == 1 + 6 + 6

    # 3 brightest pixels are [50, 50, 29], so mean is 43. With a fraction of 0.9
    # all boundaries should be deselected
    args["time_limit"] = None
    args["bright_cleaning_threshold"] = 40

    mask = cleaning.nsb_image_cleaning(geom, charge, peak_time, **args)
    assert np.count_nonzero(mask) == 1 + 1

    # Set neighbors of core_pixel_2 to 0 so largest_island_only should select only
    # core_pixel_1 with its neighbors
    charge[neighbors_2] = 0
    args["bright_cleaning_threshold"] = None
    args["largest_island_only"] = True

    mask = cleaning.nsb_image_cleaning(geom, charge, peak_time, **args)
    assert np.count_nonzero(mask) == 1 + 6
