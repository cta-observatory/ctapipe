import numpy as np
from scipy.sparse.csgraph import connected_components

from ..containers import MorphologyContainer


def number_of_islands(geom, mask):
    """
    Search a given pixel mask for connected clusters.
    This can be used to seperate between gamma and hadronic showers.

    Parameters
    ----------
    geom: `~ctapipe.instrument.CameraGeometry`
        Camera geometry information
    mask: ndarray
        input mask (array of booleans)

    Returns
    -------
    num_islands: int
        Total number of clusters
    island_labels: ndarray
        Contains cluster membership of each pixel.
        Dimension equals input geometry.
        Entries range from 0 (not in the pixel mask) to num_islands.
    """
    # compress sparse neighbor matrix
    neighbor_matrix_compressed = geom.neighbor_matrix_sparse[mask][:, mask]
    # pixels in no cluster have label == 0
    island_labels = np.zeros(geom.n_pixels, dtype="int32")

    num_islands, island_labels_compressed = connected_components(
        neighbor_matrix_compressed, directed=False
    )

    # count clusters from 1 onwards
    island_labels[mask] = island_labels_compressed + 1

    return num_islands, island_labels


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


def largest_island(islands_labels):
    """Find the biggest island and filter it from the image.

    This function takes a list of islands in an image and isolates the largest one
    for later parametrization.

    Parameters
    ----------
    islands_labels : array
        Flattened array containing a list of labelled islands from a cleaned image.
        Second value returned by the function 'number_of_islands'.

    Returns
    -------
    islands_labels : array
        A boolean mask created from the input labels and filtered for the largest island.
        If no islands survived the cleaning the array is all False.

    """
    if np.count_nonzero(islands_labels) == 0:
        return np.zeros(islands_labels.shape, dtype="bool")
    return islands_labels == np.argmax(np.bincount(islands_labels[islands_labels > 0]))


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

    num_islands, island_labels = number_of_islands(geom=geom, mask=image_mask)

    n_small, n_medium, n_large = number_of_island_sizes(island_labels)

    return MorphologyContainer(
        num_pixels=np.count_nonzero(image_mask),
        num_islands=num_islands,
        num_small_islands=n_small,
        num_medium_islands=n_medium,
        num_large_islands=n_large,
    )
