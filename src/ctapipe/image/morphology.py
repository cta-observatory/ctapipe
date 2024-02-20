import numpy as np
from numba import njit

from ..containers import MorphologyContainer


@njit(cache=True)
def _n_islands_sparse_indices(indices, indptr, mask):
    # non-signal pixel get label == 0, we marke the cleaning
    # pixels with -1, so we only have to check labels and not labels and mask
    # from now on.
    labels = np.zeros(len(mask), dtype=np.int16)
    labels[mask] = -1

    cleaning_pixels = np.where(mask)[0]
    n_cleaning_pixels = len(cleaning_pixels)
    current_island = 0

    to_check = []
    for i in range(n_cleaning_pixels):
        idx = cleaning_pixels[i]

        # we already visited this pixel
        if labels[idx] != -1:
            continue

        # start a new island
        current_island += 1
        labels[idx] = current_island

        # check neighbors recursively
        neighbors = indices[indptr[idx] : indptr[idx + 1]]
        for n in range(len(neighbors)):
            neighbor = neighbors[n]
            if labels[neighbor] == -1:
                to_check.append(neighbor)

        while len(to_check) > 0:
            idx = to_check.pop()
            labels[idx] = current_island

            neighbors = indices[indptr[idx] : indptr[idx + 1]]
            for n in range(len(neighbors)):
                neighbor = neighbors[n]

                if labels[neighbor] == -1:
                    to_check.append(neighbor)

    return current_island, labels


def number_of_islands(geom, mask):
    """
    Search a given pixel mask for connected clusters.
    This can be used to separate between gamma and hadronic showers.

    Parameters
    ----------
    geom : `~ctapipe.instrument.CameraGeometry`
        Camera geometry information, needs to be the full camrea geometry,
        not already masked with ``mask``
    mask : ndarray
        input mask (array of booleans) of pixels surviving the cleaning

    Returns
    -------
    n_islands : int
        Total number of clusters
    island_labels : ndarray
        Contains cluster membership of each pixel.
        Dimension equals input geometry.
        Entries range from 0 (not in the pixel mask) to n_islands.
    """
    if geom.n_pixels != len(mask):
        raise ValueError(
            "CameraGeometry has less pixels than mask"
            ", number_of_islands needs the full CameraGeometry"
        )
    neighbors = geom.neighbor_matrix_sparse
    n_islands, island_labels = _n_islands_sparse_indices(
        neighbors.indices, neighbors.indptr, mask
    )
    return n_islands, island_labels


def number_of_island_sizes(island_labels):
    """
    Return number of small, medium and large islands

    Parameters
    ----------
    island_labels: array[int]
        Array with island labels, (second return value of ``number_of_islands``)

    Returns
    -------
    n_small: int
        number of islands with less than 3 pixels
    n_medium: int
        number of islands with 3 <= n_pixels <= 50
    n_large: int
        number of islands with more than 50 pixels
    """

    # count number of pixels in each island, remove 0 = no island
    island_sizes = np.bincount(island_labels)[1:]

    # remove islands of size 0 (if labels are not consecutive)
    # should not happen, but easy to check
    island_sizes = island_sizes[island_sizes > 0]

    small = island_sizes <= 2
    large = island_sizes > 50
    n_medium = np.count_nonzero(~(small | large))

    return np.count_nonzero(small), n_medium, np.count_nonzero(large)


def largest_island(island_labels):
    """Find the biggest island and filter it from the image.

    This function takes a list of islands in an image and isolates the largest one
    for later parametrization.

    Parameters
    ----------
    island_labels : array
        Flattened array containing a list of labelled islands from a cleaned image.
        Second value returned by the function 'number_of_islands'.

    Returns
    -------
    mask : array
        A boolean mask created from the input labels and filtered for the largest island.
        If no islands survived the cleaning the array is all False.

    """
    if np.count_nonzero(island_labels) == 0:
        return np.zeros(island_labels.shape, dtype="bool")
    return island_labels == np.argmax(np.bincount(island_labels[island_labels > 0]))


def brightest_island(n_islands, island_labels, image):
    """Find the brightest island and filter it from the image.

    This function takes a list of islands in an image and the image itself
    and isolates brightest island for later parametrization.

    Parameters
    ----------
    n_islands: int
        Total number of islands, first return value of `number_of_islands`
    island_labels : array
        Flattened array containing a list of labelled islands from a cleaned image.
        Second value returned by the function 'number_of_islands'.

    image: array
        The image array

    Returns
    -------
    mask : array
        A boolean mask created from the input labels and filtered for the brightest island.
        If no islands survived the cleaning the array is all False.
    """

    if n_islands == 0:
        return np.zeros(image.shape, dtype="bool")

    # only look at the actual islands, 0 means pixel did not survive cleaning
    mask = island_labels > 0

    # calculate the sum of image for each island via numpy ufunc magic
    # basically, each pixel is added to `island_brightness` at its island_label
    # +1 because 0 means "no island" and we have up to "n_islands" values.
    island_brightness = np.zeros(n_islands + 1)
    np.add.at(island_brightness, island_labels[mask], image[mask])

    brightest = np.argmax(island_brightness)
    return island_labels == brightest


def morphology_parameters(geom, image_mask) -> MorphologyContainer:
    """
    Compute image morphology parameters

    Parameters
    ----------
    geom: ctapipe.instrument.camera.CameraGeometry
        camera description
    image_mask: np.ndarray(bool)
       image of pixels surviving cleaning (True=survives)

    Returns
    -------
    MorphologyContainer: parameters related to the morphology
    """

    n_islands, island_labels = number_of_islands(geom=geom, mask=image_mask)

    n_small, n_medium, n_large = number_of_island_sizes(island_labels)

    return MorphologyContainer(
        n_pixels=np.count_nonzero(image_mask),
        n_islands=n_islands,
        n_small_islands=n_small,
        n_medium_islands=n_medium,
        n_large_islands=n_large,
    )
