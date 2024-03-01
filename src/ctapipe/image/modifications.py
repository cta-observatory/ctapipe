import numpy as np
from numba import njit

from ..core import TelescopeComponent
from ..core.traits import BoolTelescopeParameter, FloatTelescopeParameter, Int
from ..instrument import SubarrayDescription

__all__ = [
    "ImageModifier",
]


def _add_noise(image, noise_level, rng=None, correct_bias=True):
    """
    Create a new image with added poissonian noise
    """
    if not rng:
        rng = np.random.default_rng()

    noisy_image = image.copy()
    noise = rng.poisson(noise_level, size=image.shape)
    noisy_image += noise

    if correct_bias:
        noisy_image -= noise_level

    return noisy_image


@njit(cache=True)
def _smear_psf_randomly(
    image, fraction, indices, indptr, smear_probabilities, seed=None
):
    """
    Create a new image with values smeared across the
    neighbor pixels specified by `indices` and `indptr`.
    These are what defines the sparse neighbor matrix
    and are available as attributes from the neighbor matrix.
    The amount of charge that is distributed away from a given
    pixel is drawn from a poissonian distribution.
    The distribution of this charge among the neighboring
    pixels follows a multinomial.
    Pixels at the camera edge lose charge this way.
    No geometry is available in this function due to the numba optimization,
    so the indices, indptr and smear_probabilities have to match
    to get sensible results.

    Parameters:
    -----------
    image: ndarray
        1d array of the pixel charge values
    fraction: float
        fraction of charge that will be distributed among neighbors (modulo poissonian)
    indices: ndarray[int]
        CSR format index array of the neighbor matrix
    indptr: ndarray[int]
        CSR format index pointer array of the neighbor matrix
    smear_probabilities: ndarray[float]
        shape: (n_neighbors, )
        A priori distribution of the charge amongst neighbors.
        In most cases probably of the form np.full(n_neighbors, 1/n_neighbors)
    seed: int
        Random seed for the numpy rng.
        Because this is numba optimized, a rng instance can not be used here

    Returns:
    --------
    new_image: ndarray
        1d array with smeared values
    """
    new_image = image.copy()
    if seed is not None:
        np.random.seed(seed)

    for pixel in range(len(image)):
        if image[pixel] <= 0:
            continue

        to_smear = np.random.poisson(image[pixel] * fraction)
        if to_smear == 0:
            continue

        # remove light from current pixel
        new_image[pixel] -= to_smear

        # add light to neighbor pixels
        neighbors = indices[indptr[pixel] : indptr[pixel + 1]]
        n_neighbors = len(neighbors)

        # we always distribute the charge as if the maximum number
        # of neighbors of a geometry is present, so that charge
        # on the edges of the camera is lost
        neighbor_charges = np.random.multinomial(to_smear, smear_probabilities)

        for n in range(n_neighbors):
            neighbor = neighbors[n]
            new_image[neighbor] += neighbor_charges[n]

    return new_image


class ImageModifier(TelescopeComponent):
    """
    Component to tune simulated background to
    overserved NSB values.
    A differentiation between bright and dim pixels is taking place
    because this happens at DL1a level and in general the integration window
    would change for peak-searching extraction algorithms with different background levels
    introducing a bias to the charge in dim pixels.

    The performed steps include:
    - Smearing of the image (simulating a worse PSF)
    - Adding poissonian noise (different for bright and dim pixels)
    - Adding a bias to dim pixel charges (see above)
    """

    psf_smear_factor = FloatTelescopeParameter(
        default_value=0.0, help="Fraction of light to move to each neighbor"
    ).tag(config=True)
    noise_transition_charge = FloatTelescopeParameter(
        default_value=0.0, help="separation between dim and bright pixels"
    ).tag(config=True)
    noise_bias_dim_pixels = FloatTelescopeParameter(
        default_value=0.0, help="extra bias to add in dim pixels"
    ).tag(config=True)
    noise_level_dim_pixels = FloatTelescopeParameter(
        default_value=0.0, help="expected extra noise in dim pixels"
    ).tag(config=True)
    noise_level_bright_pixels = FloatTelescopeParameter(
        default_value=0.0, help="expected extra noise in bright pixels"
    ).tag(config=True)
    noise_correct_bias = BoolTelescopeParameter(
        default_value=True, help="If True subtract the expected noise from the image."
    ).tag(config=True)
    rng_seed = Int(default_value=1337, help="Seed for the random number generator").tag(
        config=True
    )

    def __init__(
        self, subarray: SubarrayDescription, config=None, parent=None, **kwargs
    ):
        """
        Parameters
        ----------
        subarray: SubarrayDescription
            Description of the subarray. Provides information about the
            camera which are useful in calibration. Also required for
            configuring the TelescopeParameter traitlets.
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This is mutually exclusive with passing a ``parent``.
        parent: ctapipe.core.Component or ctapipe.core.Tool
            Parent of this component in the configuration hierarchy,
            this is mutually exclusive with passing ``config``
        """

        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)
        self.rng = np.random.default_rng(self.rng_seed)

    def __call__(self, tel_id, image, rng=None):
        dtype = image.dtype

        if self.psf_smear_factor.tel[tel_id] > 0:
            geom = self.subarray.tel[tel_id].camera.geometry
            image = _smear_psf_randomly(
                image=image,
                fraction=self.psf_smear_factor.tel[tel_id],
                indices=geom.neighbor_matrix_sparse.indices,
                indptr=geom.neighbor_matrix_sparse.indptr,
                smear_probabilities=np.full(geom.max_neighbors, 1 / geom.max_neighbors),
                seed=self.rng.integers(0, np.iinfo(np.int64).max),
            )

        if (
            self.noise_level_dim_pixels.tel[tel_id] > 0
            or self.noise_level_bright_pixels.tel[tel_id] > 0
        ):
            bright_pixels = image > self.noise_transition_charge.tel[tel_id]
            noise = np.where(
                bright_pixels,
                self.noise_level_bright_pixels.tel[tel_id],
                self.noise_level_dim_pixels.tel[tel_id],
            )
            image = _add_noise(
                image,
                noise,
                rng=self.rng,
                correct_bias=self.noise_correct_bias.tel[tel_id],
            )

            image[~bright_pixels] += self.noise_bias_dim_pixels.tel[tel_id]

        return image.astype(dtype, copy=False)
