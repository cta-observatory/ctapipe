from collections import defaultdict

import numpy as np
from numba import njit
from traitlets import default

from ..containers import EventType
from ..core import TelescopeComponent
from ..core.env import CTAPIPE_DISABLE_NUMBA_CACHE
from ..core.traits import (
    Bool,
    BoolTelescopeParameter,
    FloatTelescopeParameter,
    Int,
    Path,
)
from ..instrument import SubarrayDescription
from ..io import EventSource
from ..utils import EventTypeFilter

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


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
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


class NoiseEventTypeFilter(EventTypeFilter):
    """
    Event filter to select noise events for MC tuning
    By default it selects SKY_PEDESTAL events
    """

    @default("allowed_types")
    def allowed_types_default(self):
        return {EventType.SKY_PEDESTAL}


@njit
def build_wf_noise_pixelwise(
    waveforms, n_noise_realizations, nsb_level, rng, shuffle_full_cameras
):
    """
    Combine "elemental noise waveforms" into total noise waveforms by
    combining a given number of them, chosen randomly

    Parameters
    ----------
    waveforms: array (nevents, ngains, npixels, nsamples), the elemental noise
    waveforms

    n_noise_realizations: int
    the number of total noise waveforms we want to generate

    nsb_level: int
    the number of elemental noise waveforms we combine to produce each total
    noise waveform

    rng: random number generator

    shuffle_full_cameras: bool
    if True, the waveform for each pixel in any given noise realization comes
    from the same combination of the input elemental noise events. If False,
    each pixel uses a different combination of the events

    Returns
    -------
    array (n_noise, ngains, npixels, nsamples), total noise waveforms

    """
    n_events, n_gains, n_pixels, n_samples = waveforms.shape
    noise = np.zeros(
        (n_noise_realizations, n_gains, n_pixels, n_samples), dtype=np.float32
    )

    for i in range(n_noise_realizations):
        if shuffle_full_cameras:
            chosen = rng.permutation(n_events)[:nsb_level]
            for event in chosen:
                noise[i] += waveforms[event]
        else:
            for pixel in range(n_pixels):
                chosen = rng.permutation(n_events)[:nsb_level]
                # The line above is slower (especially for n_events much
                # larger than nsb_level) than rng.choice(n_events, nsb_level)
                # Unfortunately rng.choice does not currently work with numba.

                for event in chosen:
                    noise[i, :, pixel] += waveforms[event, :, pixel]
    return noise


class WaveformModifier(TelescopeComponent):
    """
    Component to add NSB noise to R1 waveforms.

    This component in principle to be applied on MC shower simulations, to make
    them closer to real data in terms of noise level.

    There are two possibilities:
        1. The "showers MC" file has dark-NSB settings and electronic noise (
        waveform baseline fluctuations), and the input NSB file is a
        dedicated sim_telarray file, which must be produced with the same
        telescope array configuration (and other simulation settings) as the
        showers MC to which the noise is to be added, but containing only NSB
        noise (electronic fluctuations of the baseline should be switched off)

        2. The showers MC file is produced with no noise (baseline
        fluctuations) of any kind (electronic or NSB), just the Cherenkov
        signal (with the appropriate single-p.e.-response fluctuations),
        whereas the nsb file is a real data DL0 file from which only the
        interleaved pedestals are used (all gain channels must be present
        for all pixels). In that case, nsb_level must be =1 (to
        match the MC to the data) and shuffle_full_cameras=True (we do not
        want e.g. to duplicate stars in the FoV).


    In case (1), the number of available noise events per telescope in the NSB
    file must be at least twice the number of waveforms ("nsb_level") from
    that file that we want to add up. If the NSB file is produced with a
    level of 25% of dark NSB, and we want to simulate 10x dark NSB,
    then nsb_level=40 (=10/0.25) and the file must contain at least 80
    events. This is to guarantee that the different noise waveforms are not
    too correlated


    """

    nsb_file = Path(
        default_value=None,
        help="Path to a dedicated NSB-only file (e.g. from sim_telarray)",
    ).tag(config=True)

    nsb_level = Int(
        default_value=1,
        help=(
            "Number of random instances of the NSB waveforms from the "
            "NSB file to be added up to the waveform"
        ),
    ).tag(config=True)

    n_noise_realizations = Int(
        default_value=100,
        help=(
            "Number of different realizations of the total NSB waveform to "
            "be created per pixel"
        ),
    ).tag(config=True)

    shuffle_full_cameras = Bool(
        default_value=False,
        help=(
            "If True, full cameras are "
            "combined to generate noise "
            "waveforms for all pixels. Else, "
            "each pixel uses a different random "
            "combination of the input noise events"
        ),
    ).tag(config=True)

    rng_seed = Int(default_value=1, help="Seed for the random number generator").tag(
        config=True
    )

    def __init__(
        self,
        subarray: SubarrayDescription,
        config=None,
        parent=None,
        **kwargs,
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

        self.event_type_filter = NoiseEventTypeFilter(parent=self)

        self.total_noise = dict()
        # One key per tel_id, and each of them is an array of shape
        # [n_noise_realizations, ngains, npixels, nsamples]

        # Read in the waveforms in the NSB-only file. Store in a dictionary
        # with one key per telescope, containing an array [n_events, n_gains,
        # n_pixels, n_samples]
        nsb_database = self.read_nsb_database()

        # Check if noise statistics is sufficient:
        stats_ok = self.check_noise_statistics(nsb_database)
        if not stats_ok:
            raise ValueError("Please use an input nsb_file with more events!")

        # Now shift the waveforms so that they have mean=0 and do not introduce
        # any bias (just fluctuations)
        self.zero_baseline(nsb_database)

        # Add up waveforms selected at random to obtain different
        # realizations of the total noise that will be added
        for tel_id in nsb_database:
            self.total_noise[tel_id] = build_wf_noise_pixelwise(
                nsb_database[tel_id],
                self.n_noise_realizations,
                self.nsb_level,
                self.rng,
                self.shuffle_full_cameras,
            )

    def read_nsb_database(self):
        """
        Reads in R1 noise waveforms from an input file self.nsb_file

        Returns
        -------
        nsb_database : dict
        Dictionary with one key per telescope, containing an array [n_events,
        n_gains, n_pixels, n_samples] (noise waveforms)

        """
        nsb_database = defaultdict(list)
        with EventSource(
            input_url=self.nsb_file, skip_calibration_events=False
        ) as source:
            for event in source:
                if not self.event_type_filter(event):
                    continue
                for tel_id, r1 in event.r1.tel.items():
                    nsb_database[tel_id].append(r1.waveform)

        nsb_database = {
            tel_id: np.stack(waveforms) for tel_id, waveforms in nsb_database.items()
        }

        return nsb_database

    def check_noise_statistics(self, nsb_database):
        """
        Check that we have enough NSB-only events for all telescopes. We
        require that the number of NSB events for any telescope is at
        least two times the number of waveforms (=nsb_level) that we will add
        up. This is to avoid excessive correlation among the waveforms.

        Parameters
        ----------
        nsb_database: dict
        Dictionary with one key per telescope, containing an array [n_events,
        n_gains, n_pixels, n_samples] (noise waveforms)

        Returns
        -------
        stats_ok : bool
        True if statistics of noise events is considered sufficient
        """

        stats_ok = True
        for tel_id in nsb_database:
            nevents = nsb_database[tel_id].shape[0]
            if nevents >= 2 * self.nsb_level:
                continue
            self.log.error(
                f"Not enough NSB events available for tel_"
                f"id {tel_id}. "
                f"For nsb_level = {self.nsb_level}, at least "
                f"{2*self.nsb_level} events are needed ({nevents} "
                f"were found)."
            )
            stats_ok = False

        return stats_ok

    def zero_baseline(self, nsb_database):
        """
        For each telescope and gain we average the waveform values for all
        pixels, and subtract those averages from the waveforms.
        In this way we make sure we do not introduce any net average charge,
        but only increase the fluctuations.

        Parameters
        ----------
        nsb_database: dict
        Dictionary with one key per telescope, containing an array [n_events,
        n_gains, n_pixels, n_samples] (noise waveforms)

        Returns
        -------
        nsb_database: dict
        Dictionary, same as above but after baseline zeroing
        """

        for tel_id in nsb_database:
            for channel in range(nsb_database[tel_id].shape[1]):
                mean = np.mean(nsb_database[tel_id][:, channel, :, :])
                nsb_database[tel_id][:, channel, :, :] -= mean

    def __call__(self, tel_id, waveforms, selected_gain_channel=None):
        """
        Parameters
        ----------
        tel_id
        waveforms: ndarray [ngains, npixels, nsamples] (ngains=1 if
        gain-selected)
        selected_gain_channel: ndarray[npixels] or None if no gain selection

        Returns
        -------
        ndarray, same shape as waveforms: original waveforms plus added NSB

        """

        # Note: MC waveforms passed to this function should contain data in all
        # pixels (not DVR'ed - obviously DVR depends on noise level, it does not
        # make sense to add the noise after DVR was applied)
        noise = self.total_noise[tel_id][self.rng.integers(self.n_noise_realizations)]
        if selected_gain_channel is not None:
            noise = noise[selected_gain_channel, np.arange(waveforms.shape[1])]

        return waveforms + noise


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
