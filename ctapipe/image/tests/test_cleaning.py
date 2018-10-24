import numpy as np
from numpy.testing import assert_allclose
from ctapipe.image import cleaning
from ctapipe.instrument import CameraGeometry


def test_tailcuts_clean_simple():
    geom = CameraGeometry.from_name("LSTCam")
    image = np.zeros_like(geom.pix_id, dtype=np.float)

    num_pix = 40
    some_neighs = geom.neighbors[num_pix][0:3]  # pick 4 neighbors
    image[num_pix] = 5.0  # set a single image pixel
    image[some_neighs] = 3.0  # make some boundaries that are neighbors
    image[10] = 3.0  # a boundary that is not a neighbor

    mask = cleaning.tailcuts_clean(geom, image, picture_thresh=4.5,
                                   boundary_thresh=2.5)

    print((mask > 0).sum(), "clean pixels")
    print(geom.pix_id[mask])

    assert 10 not in geom.pix_id[mask]
    assert set(some_neighs).union({num_pix}) == set(geom.pix_id[mask])
    assert (mask > 0).sum() == 4


def test_dilate():
    geom = CameraGeometry.from_name("LSTCam")
    mask = np.zeros_like(geom.pix_id, dtype=bool)

    mask[100] = True  # a single pixel far from a border is true.
    assert mask.sum() == 1

    # dilate a single row
    dmask = cleaning.dilate(geom, mask)
    assert dmask.sum() == 1 + 6

    # dilate a second row
    dmask = cleaning.dilate(geom, dmask)
    assert dmask.sum() == 1 + 6 + 12

    # dilate a third row
    dmask = cleaning.dilate(geom, dmask)
    assert dmask.sum() == 1 + 6 + 12 + 18


def test_tailcuts_clean():
    # start with simple 3-pixel camera
    geom = CameraGeometry.make_rectangular(3, 1, (-1, 1))

    p = 15  # picture value
    b = 7  # boundary value

    # for no isolated pixels:
    testcases = {(p, p, 0): [True, True, False],
                 (p, 0, p): [False, False, False],
                 (p, b, p): [True, True, True],
                 (p, b, 0): [True, True, False],
                 (b, b, 0): [False, False, False],
                 (0, p, 0): [False, False, False]}

    for image, mask in testcases.items():
        result = cleaning.tailcuts_clean(geom, np.array(image),
                                         picture_thresh=15,
                                         boundary_thresh=5,
                                         keep_isolated_pixels=False)
        assert (result == mask).all()


def test_tailcuts_clean_min_neighbors_1():
    """
    requiring that picture pixels have at least one neighbor above picture_thres:
    """

    # start with simple 3-pixel camera
    geom = CameraGeometry.make_rectangular(3, 1, (-1, 1))

    p = 15  # picture value
    b = 7  # boundary value

    testcases = {(p, p, 0): [True, True, False],
                 (p, 0, p): [False, False, False],
                 (p, b, p): [False, False, False],
                 (p, b, 0): [False, False, False],
                 (b, b, 0): [False, False, False],
                 (0, p, 0): [False, False, False],
                 (p, p, p): [True, True, True]}

    for image, mask in testcases.items():
        result = cleaning.tailcuts_clean(geom, np.array(image),
                                         picture_thresh=15,
                                         boundary_thresh=5,
                                         min_number_picture_neighbors=1,
                                         keep_isolated_pixels=False)
        assert (result == mask).all()


def test_tailcuts_clean_min_neighbors_2():
    """ requiring that picture pixels have at least two neighbors above 
    picture_thresh"""

    # start with simple 3-pixel camera
    geom = CameraGeometry.make_rectangular(3, 1, (-1, 1))

    p = 15  # picture value
    b = 7  # boundary value

    testcases = {(p, p, 0): [False, False, False],
                 (p, 0, p): [False, False, False],
                 (p, b, p): [False, False, False],
                 (p, b, 0): [False, False, False],
                 (b, b, 0): [False, False, False],
                 (p, p, p): [True, True, True]}

    for image, mask in testcases.items():
        result = cleaning.tailcuts_clean(geom, np.array(image),
                                         picture_thresh=15,
                                         boundary_thresh=5,
                                         min_number_picture_neighbors=2,
                                         keep_isolated_pixels=False)
        assert (result == mask).all()


def test_tailcuts_clean_with_isolated_pixels():
    # start with simple 3-pixel camera
    geom = CameraGeometry.make_rectangular(3, 1, (-1, 1))

    p = 15  # picture value
    b = 7  # boundary value

    testcases = {(p, p, 0): [True, True, False],
                 (p, 0, p): [True, False, True],
                 (p, b, p): [True, True, True],
                 (p, b, 0): [True, True, False],
                 (0, p, 0): [False, True, False],
                 (b, b, 0): [False, False, False]}

    for image, mask in testcases.items():
        result = cleaning.tailcuts_clean(geom, np.array(image),
                                         picture_thresh=15,
                                         boundary_thresh=5,
                                         keep_isolated_pixels=True)
        assert (result == mask).all()


def test_number_of_islands():
    # test with LST geometry (1855 pixels)
    geom = CameraGeometry.from_name("LSTCam")

    # create 18 triggered pixels grouped to 5 clusters
    island_mask_true = np.zeros(geom.n_pixels)
    mask = np.zeros(geom.n_pixels).astype('bool')
    triggered_pixels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                 14,
                                 37, 38,
                                 111,
                                 222])
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
