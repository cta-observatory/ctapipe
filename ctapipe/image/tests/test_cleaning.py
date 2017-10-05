import numpy as np
from ctapipe.image import cleaning

from ctapipe.instrument import CameraGeometry


def test_tailcuts_clean():

    geom = CameraGeometry.from_name("LSTCam")
    image = np.zeros_like(geom.pix_id, dtype=np.float)
    pedvar = np.ones_like(geom.pix_id, dtype=np.float)

    N = 40
    some_neighs = geom.neighbors[N][0:3]  # pick 4 neighbors
    image[N] = 5.0              # set a single image pixel
    image[some_neighs] = 3.0    # make some boundaries that are neighbors
    image[10] = 3.0             # a boundary that is not a neighbor

    mask = cleaning.tailcuts_clean(geom, image, picture_thresh=4.5,
                                   boundary_thresh=2.5)

    print((mask > 0).sum(), "clean pixels")
    print(geom.pix_id[mask])

    assert 10 not in geom.pix_id[mask]
    assert set(some_neighs).union({N}) == set(geom.pix_id[mask])
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

    p = 15  #picture value
    b = 7   # boundary value

    # for no isolated pixels:
    testcases = {(p, p, 0): [True, True, False],
                 (p, 0, p): [False, False, False],
                 (p, b, p): [True, True, True],
                 (p, b, 0): [True, True, False],
                 (b, b, 0): [False, False, False],
                 (0, p ,0): [False, False, False]}

    for image, mask in testcases.items():
        result = cleaning.tailcuts_clean(geom, np.array(image),
                                         picture_thresh=15,
                                         boundary_thresh=5,
                                         keep_isolated_pixels=False)
        assert (result == mask).all()

# requiring that picture pixels have at least one neighbor above picture_thres:
    testcases = {(p, p, 0): [True,  True,  False],
                 (p, 0, p): [False, False, False],
                 (p, b, p): [False, False, False],
                 (p, b, 0): [False, False, False],
                 (b, b, 0): [False, False, False],
                 (0, p ,0): [False, False, False],
                 (p, p, p): [True,  True,  True]}

    for image, mask in testcases.items():
        result = cleaning.tailcuts_clean(geom, np.array(image),
                                         picture_thresh=15,
                                         boundary_thresh=5,
                                         min_number_picture_neighbors=1,
                                         keep_isolated_pixels=False)
        assert (result == mask).all()

# requiring that picture pixels have at least two neighbors above picture_thres:
    testcases = {(p, p, 0): [False, False, False],
                 (p, 0, p): [False, False, False],
                 (p, b, p): [False, False, False],
                 (p, b, 0): [False, False, False],
                 (b, b, 0): [False, False, False],
                 (p, p ,p): [True, True, True]}

    for image, mask in testcases.items():
        result = cleaning.tailcuts_clean(geom, np.array(image),
                                         picture_thresh=15,
                                         boundary_thresh=5,
                                         min_number_picture_neighbors=2,
                                         keep_isolated_pixels=False)
        assert (result == mask).all()


    # allowing isolated pixels
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
