import numpy as np
from ctapipe.image import cleaning

from ... import io


def test_tailcuts_clean():

    geom = io.CameraGeometry.from_name("HESS", 1)
    image = np.zeros_like(geom.pix_id, dtype=np.float)
    pedvar = np.ones_like(geom.pix_id, dtype=np.float)

    N = 40
    some_neighs = geom.neighbors[N][0:3]  # pick 4 neighbors
    image[N] = 5.0              # set a single image pixel
    image[some_neighs] = 3.0    # make some boundaries that are neighbors
    image[10] = 3.0             # a boundary that is not a neighbor

    mask = cleaning.tailcuts_clean(geom, image, pedvar,
                                   picture_thresh=4.5,
                                   boundary_thresh=2.5)

    print((mask > 0).sum(), "clean pixels")
    print(geom.pix_id[mask])

    assert 10 not in geom.pix_id[mask]
    assert set(some_neighs).union({N}) == set(geom.pix_id[mask])
    assert (mask > 0).sum() == 4


def test_dilate():

    geom = io.CameraGeometry.from_name("HESS", 1)
    mask = np.zeros_like(geom.pix_id, dtype=bool)

    mask[100] = True  # a single pixel far from a border is true.
    assert mask.sum() == 1

    # dilate a single row
    cleaning.dilate(geom, mask)
    assert mask.sum() == 1 + 6

    # dilate a second row
    cleaning.dilate(geom, mask)
    assert mask.sum() == 1 + 6 + 12

    # dilate a third row
    cleaning.dilate(geom, mask)
    assert mask.sum() == 1 + 6 + 12 + 18
