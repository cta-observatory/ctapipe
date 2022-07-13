import numpy as np

from .cleaning import ImageCleaner
from ..core.traits import (
    FloatTelescopeParameter,
    BoolTelescopeParameter,
)
from numba import njit
from scipy.interpolate import interp1d
from ..containers import EventType
from functools import lru_cache
from tqdm.auto import tqdm

import astropy.units as u


def _get_neighbors_of(pixel_id, indptr, indices):
    first = indptr[pixel_id]
    last = indptr[pixel_id + 1]
    return indices[first:last]


def _get_groups_2nn(indptr, indices):
    n_pixels = len(indptr) - 1
    combinations = []
    for pixel_id in range(n_pixels):
        neighbors = _get_neighbors_of(pixel_id, indptr, indices)
        for neighbor in neighbors:
            if neighbor > pixel_id:
                combinations.append((pixel_id, neighbor))

    return np.array(combinations, dtype=np.uint16)


def _get_groups_3nn(groups_2nn, indptr, indices):
    combinations = []
    for group in groups_2nn:
        p1, p2 = group
        neighbors_p1 = _get_neighbors_of(p1, indptr, indices)
        neighbors_p2 = _get_neighbors_of(p2, indptr, indices)
        for neighbor in np.intersect1d(neighbors_p1, neighbors_p2):
            if neighbor > p1 and neighbor > p2:
                combinations.append((p1, p2, np.uint16(neighbor)))

    return np.array(combinations, dtype=np.uint16)


def _get_groups_4nn_from_2nn(groups_2nn, indptr, indices):
    combinations = []
    for group in groups_2nn:
        p1, p2 = group
        neighbors_p1 = _get_neighbors_of(p1, indptr, indices)
        neighbors_p2 = _get_neighbors_of(p2, indptr, indices)
        common_neighbors = np.intersect1d(neighbors_p1, neighbors_p2).tolist()
        if len(common_neighbors) == 2:
            n_1, n_2 = common_neighbors
            combinations.append((p1, p2, np.uint16(n_1), np.uint16(n_2)))

    return np.array(combinations, dtype=np.uint16)


def _get_groups_2_1nn(groups_2nn, indptr, indices):
    combinations = []
    for group in groups_2nn:
        p1, p2 = group
        exclusion = set(_get_neighbors_of(p1, indptr, indices).tolist())
        exclusion |= set(_get_neighbors_of(p2, indptr, indices).tolist())
        candidates = set(exclusion)
        for core_pixel in exclusion:
            candidates |= set(_get_neighbors_of(core_pixel, indptr, indices).tolist())
        candidates = list(candidates - exclusion)
        for candidate in candidates:
            combinations.append((p1, p2, np.uint16(candidate)))

    return np.array(combinations, dtype=np.uint16)


def _get_pixel_groups(indptr, indices):
    groups_2nn = _get_groups_2nn(indptr, indices)
    groups_3nn = _get_groups_3nn(groups_2nn, indptr, indices)
    groups_4nn = _get_groups_4nn_from_2nn(groups_2nn, indptr, indices)
    # groups_21nn = _get_groups_2_1nn(groups_2nn, indptr, indices)
    groups = {
        "2nn": groups_2nn,
        "3nn": groups_3nn,
        "4nn": groups_4nn,
        # "2+1": groups_21nn,
    }
    return groups


@lru_cache(maxsize=None)
def get_pixel_groups(geometry):
    return _get_pixel_groups(
        geometry.neighbor_matrix_sparse.indptr, geometry.neighbor_matrix_sparse.indices
    )


@njit
def _is_group_in_mask(group, mask):
    for member in group:
        if not mask[member]:
            return False
    return True


@njit
def _flag_groups_where_mask(groups, mask):
    valid_indices = np.zeros(len(groups), dtype=np.bool_)
    for index in np.arange(len(groups)):
        if _is_group_in_mask(groups[index], mask):
            valid_indices[index] = True
    return valid_indices


def get_valid_groups(all_groups, mask):
    filtered_groups = {}
    for nfold, groups in all_groups.items():
        valid_indices = _flag_groups_where_mask(groups, mask)
        filtered_groups[nfold] = groups[valid_indices]
    return filtered_groups


def dilate_matrix(mask, matrix):
    return np.dot(matrix, mask)


MULTIPLICITY = {"2nn": 2, "3nn": 3, "4nn": 4, "2+1": 3}


class TimeNextNeighborCleaning(ImageCleaner):
    """
    Class to perform the time next-neighbor cleaning defined in
    https://arxiv.org/pdf/1307.4939.pdf . During the cleaning procedure, it is
    searched for groups of pixels with a given charge and time-coincidence.

    It requires only one parameter which is the fake image proability. The
    cuts in the Q-dT parameterspace then are determined from the individual
    pixel rate which gives the rate in which the pixels have a charge above
    a given threshold. Those graphs are calculated from calibration data
    (i.e. images without Cherenkov signal but pure night sky background).
    """

    fake_probability = FloatTelescopeParameter(
        default_value=0.01, help="False image error rate"
    ).tag(config=True)

    apply_pre_threshold = BoolTelescopeParameter(
        default_value=False, help="If False, skip runtime optimisation."
    ).tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.IPR_dict = {}
        self._ipr_interpolators = {}

    def __call__(
        self, tel_id: int, image: np.ndarray, arrival_times: np.ndarray = None
    ) -> np.ndarray:
        r"""
        Main function of time next neighbor cleaning developed by M. Shayduk.
        See https://arxiv.org/pdf/1307.4939.pdf.
        Search for 2nn, 3nn, 4nn and 2+1 neighbors group with a minimum charge
        and small difference in arrival time. For each potential combination of
        pixels, the minimum charge and the maximum time difference are
        considered for checking the validity with the cut in the charge-time
        space.

        Additionally a pre-cut on the charge can be applied. The strength of
        this cut is specified by the factor. The cut is selected by evaluating
        the Q-dT cut at time difference of :math:`factor \times sampling time`

        Parameters
        ----------
        tel_id : int
            telescope_id
        image : numpy.ndarray
            pixel values
        arrival_times : numpy.ndarray
            arrival time in number of samples
        fake_prob : float
            false positive rate to return
        waveform_duration : astropy.units.Quantity
            total sampling duration

        Returns
        -------
        survived_pixels : numpy.ndarray
            mask with pixels that survived the image cleaning
        """
        camera = self.subarray.tel[tel_id].camera
        geometry = camera.geometry
        survived_pixels = np.zeros(geometry.n_pixels, bool)
        n_samples = self.IPR_dict[tel_id]["n_samples"]
        sample_time = 1 / camera.readout.sampling_rate
        waveform_duration = n_samples * sample_time

        for nfold in ["2nn", "3nn", "4nn"]:

            # determine the accidental rate from fake proability
            accidental_rate = self.fake_probability.tel[tel_id] / (waveform_duration * MULTIPLICITY[nfold])

            if self.apply_pre_threshold.tel[tel_id] == False:
                candidates = None
            else:
                pre_threshold = self.pre_threshold_from_sample_time(
                    tel_id, nfold, accidental_rate, sample_time
                )
                candidates = image > pre_threshold
                if not candidates.any():
                    continue

            # Calculate the possible combinations form the candidates that
            # passed the pre-threshold.
            combinations = self.get_combinations(geometry, nfold, candidates)
            if len(combinations) == 0:
                continue
            # It can be that all pixels of a found combination already passed
            # the cuts fo a different neighbor group. Those don't have to be
            # double checked and are therefore removed. This improves a little
            # bit the performance for large images.

            # Get the charges and times for all combinations. For the charge
            # the minimum value of this group is considered while the time
            # time difference is the maximum in this group. With this
            # definitions the cuts applied should be more conservative.
            combination_image = image[combinations]
            combination_times = arrival_times[combinations]

            min_charge = np.min(combination_image, axis=1)
            min_time = np.min(combination_times, axis=1)
            max_time = np.max(combination_times, axis=1)
            time_diff = max_time - min_time

            # Apply the cut in the time and charge parameter space to check
            # which group of pixels are valid. The pixels can appear in
            # multiple groups.
            valid_groups = self.cut_time_charge(
                tel_id=tel_id,
                min_charge=min_charge,
                time_difference=time_diff,
                accidental_rate=accidental_rate,
                nfold=nfold,
            )
            valid_pixels = np.unique(combinations[valid_groups])
            survived_pixels[valid_pixels] = True

        return survived_pixels

    def fill_individual_pixel_rate(self, tel_id, image, n_samples, charge_steps):
        """
        Count the number of pixels in images about a given charge threshold
        and fill dictionary with individual pixel noise rate (IPR).

        Parameters
        ----------
        tel_id: int
            Telescope indexing number specific per source
        image: numpy.ndarray
            pixel values
        trace_length: astropy.units.Quantity
            total length of the trace, e.g. number of samples times sample_time
        charge_steps: numpy.ndarray
            array with the charge steps used to calculate the IPR graphs.
        """

        self.IPR_dict = self.IPR_dict if self.IPR_dict else {}

        if tel_id not in self.IPR_dict:
            self.IPR_dict[tel_id] = {
                "counter": 0,
                "charge": charge_steps,
                "npix": np.zeros(charge_steps.shape[0]),
                "rate": np.zeros(charge_steps.shape[0]),
                "rate_err": np.zeros(charge_steps.shape[0]),
                "n_samples": n_samples,
            }

        # count number of pixels
        charge_steps = np.append(charge_steps, np.inf)
        hist, _ = np.histogram(image, charge_steps)
        self.IPR_dict[tel_id]["counter"] += hist.sum()
        self.IPR_dict[tel_id]["charge"] = charge_steps[:-1]
        self.IPR_dict[tel_id]["npix"] += np.cumsum(hist[::-1])[::-1]

        # convert count of number of pixels above threshold into rate
        sampling_rate = self.subarray.tel[tel_id].camera.readout.sampling_rate
        waveform_duration = n_samples / sampling_rate
        sample_to_hz = (1 / (waveform_duration * self.IPR_dict[tel_id]["counter"])).to("Hz")

        self.IPR_dict[tel_id]["rate"] = self.IPR_dict[tel_id]["npix"] * sample_to_hz
        self.IPR_dict[tel_id]["rate_err"] = (
            np.sqrt(self.IPR_dict[tel_id]["npix"]) * sample_to_hz
        )

    def calculate_ipr_from_source(self, source, calibrator, charge_steps):
        """
        Fill the IPR graph from a file or a list of files. The files should
        contain pure night sky background calibration data. Therefore, the
        calibration of the events in the source are performed. The calibrator
        for this case should be equal to the one in the analysis.

        Parameters
        ----------
        source:
            EventSource type
        calibrator:
            CameraCalibrator type, LocalPeakWindowSum returns good results
        charge_steps: numpy.ndarray
            array storing the steps in charge used to create the IPR graph

        """
        for event in tqdm(source):
            if event.trigger.event_type != EventType.SKY_PEDESTAL:
                continue

            calibrator(event)
            for tel_id, dl1 in event.dl1.tel.items():
                image = dl1.image

                n_samples = event.r1.tel[tel_id].waveform.shape[-1]
                self.fill_individual_pixel_rate(tel_id, image, n_samples, charge_steps)

    @lru_cache()
    def combfactor_from_geometry(self, geometry, nfold=None, calculate=False):
        """
        Get the combinatorial factors from the geometry of the camera by
        counting the number of possible combinations for a given group of
        pixels.
        If `calculate == True`, the number of combinations will be calculated
        for the specified camera and neighbor group. Otherwise the values are
        read from a lookup table which was filled for complete cameras, e.g.
        assuming no broken pixels.

        Parameters
        ----------
        geometry: array
            Camera geometry information
        nfold: string, None
            Name of neighbor group. Select from 4nn, 3nn, 2+1 and 2nn
        calculate: bool
            If true, the value will be calculated from the geometry of the
            camera. Otherwise the values from a pre filled lookup table are
            returned

        Returns‚
        -------
        comb_factor: int
            number of possible combinations for camera type and neighbor group

        """
        # Allow to specify camera and neighbor group by hand. If not set, use
        # currently selected values.
        comb_factor = self.get_combinations(geometry, nfold)
        return len(comb_factor)

    def get_time_coincidence(self, tel_id, min_charge, nfold, accidental_rate):
        """
        Get the time coincidence for a given charge (equation 2 of
        https://arxiv.org/pdf/1307.4939.pdf). The cut on the time coincidence
        depends on the charge, the _multiplicity, combinatorial factor and the
        accepted accidental rate.

        Parameters
        ----------
        charge: numpy.ndarray
            Pixel intensity
        accidental_rate: astropy.Quantity
            False positive rate to return
        nfold: string
            Name of neighbor group. Select from 4nn, 3nn, 2+1 and 2nn

        Returns
        -------
        time_coincidence: ndarray
            array of same shape of `charge` filled with the corresponding cut
            in time
        """
        if tel_id not in self.IPR_dict:
            raise RuntimeError(f"No IPR for tel_id {tel_id}")

        if tel_id not in self._ipr_interpolators:
            self._ipr_interpolators[tel_id] = interp1d(
                self.IPR_dict[tel_id]["charge"],
                self.IPR_dict[tel_id]["rate"].to_value(u.Hz),
                "linear",
                fill_value="extrapolate",
            )

        ipr = self._ipr_interpolators[tel_id](min_charge)

        multiplicity = MULTIPLICITY[nfold]
        cfactor = self.combfactor_from_geometry(
            geometry=self.subarray.tel[tel_id].camera.geometry,
            nfold=nfold,
        )
        time_coincidence = np.exp(
            (1 / (multiplicity - 1))
            * np.log(accidental_rate.to_value(u.Hz) / (cfactor * ipr ** multiplicity))
        )
        return time_coincidence * 1e9

    def cut_time_charge(
        self, tel_id, min_charge, time_difference, accidental_rate, nfold
    ):
        """
        Apply cut in time difference and charge parameter space.

        Parameters
        ----------
        charge: ndarry
            Pixel intensity
        time_difference: astropy.units.Quantity
            Duration between first and last record
        accidental_rate:
            False positive rate to return
        nfold: string
            Name of neighbor group. Select from 4nn, 3nn, 2+1 and 2nn
        Returns
        -------
        valid_pixels: ndarray
            mask with pixels survived the cut

        """

        time_coincidence = self.get_time_coincidence(
            tel_id, min_charge, nfold, accidental_rate
        )
        return time_difference < time_coincidence

    @lru_cache(maxsize=1000)
    def pre_threshold_from_sample_time(
        self, tel_id, nfold, accidental_rate, sample_time
    ):
        """
        Get the pre threshold from the sampling time. The charge of all pixels
        should be high enough, so that the cut in the in the charge-dT space is
        above this sample time multiplied by an factor.

        Parameters
        ----------
        accidental_rate: float
            False positive rate to return
        sample_time: astropy.units.Quantity
            Sampling time of the camera
        nfold: string, None
            Group of neigboring pixels to consider
        factor: float
            Optional scaling for pre cleaning threshold

        Returns
        -------
        pre_threshold : float
            Charge of pixels with rate below threshold

        """
        min_charge = np.arange(0, 20, 0.25)
        dt = self.get_time_coincidence(
            tel_id=tel_id,
            min_charge=min_charge,
            accidental_rate=accidental_rate,
            nfold=nfold,
        )
        return min_charge[dt > sample_time.to_value(u.ns)][0]

    def get_combinations(self, geometry, nfold, mask=None):
        """
        Get a list a all possible combinations of pixels for a given geometry.
        The implementation is based on finding each pixel's neighbors via lookup in the sparse neighbor matrix.
        "2nn" pairs are straight fowrward and do not contain backward duplicates.
        "3nn" and "4nn" groups are derived from "2nn" pairs with their additional neighboring pixels thereby forming a compact group.
        "2+3" are "2nn" pairs with a separate nonneighbor pixel one pixel away from the group.
        The mask is only considered when assigning pixels to groups therefore no group is cut off and the singular pixel in "2+1" groups can skip gaps in the threshold mask.

        Parameters
        ----------
        mask: numpy.ndarray
            mask of pixels above threshold
        nfold: string
            neighbor group geometry type
        Returns
        -------
        combs: numpy.ndarray
            array of all possible combinations for given neighbor group
        """
        groups = get_pixel_groups(geometry)
        if mask is not None:
            valid_groups = get_valid_groups(groups, mask)[nfold]
            return valid_groups
        return groups[nfold]

    def boundary_search(
        self,
        tel_id,
        geometry,
        mask,
        image,
        arrival_times,
        sample_time,
        waveform_duration,
        nfold,
        fake_prob=0.001,
    ):
        """
        Find boundary pixels around core pixels.
        These pass a different lower threshold than core pixels and recieve their own mask.

        Parameters
        ----------
        source:
            EventSource type
        calibrator:
            CameraCalibrator type, LocalPeakWindowSum returns good results
        charge_steps: numpy.ndarray
            array storing the steps in charge used to create the IPR graph

        """

        bound = np.zeros(geometry.n_pixels, bool)
        if mask.any():
            second_matrix = geometry.get_n_order_neighbor_matrix(2)
            candidates = dilate_matrix(mask, second_matrix)
            candidates[image <= 0] = 0

            # Calculate the possible combinations form the candidates that
            # passed the pre-threshold
            # for nfold in ["2nn", "3nn", "4nn", "2+1"]:
            for nfold in ["2nn", "3nn", "4nn"]:
                accidental_rate = fake_prob / (waveform_duration * MULTIPLICITY[nfold])

                combinations = self.get_combinations(geometry, nfold, candidates)
                if len(combinations) < 1:
                    continue

                combination_image = image[combinations]
                combination_times = arrival_times[combinations]

                max_time = np.max(combination_times, axis=1)
                min_time = np.min(combination_times, axis=1)
                time_diff = max_time - min_time
                min_charge = np.min(combination_image, axis=1)
                valid_groups = self.cut_time_charge(
                    tel_id=tel_id,
                    min_charge=min_charge,
                    time_difference=time_diff,
                    accidental_rate=accidental_rate,
                    nfold=nfold,
                )

                valid_pixels = np.unique(combinations[valid_groups])
                bound[valid_pixels] = True

        return bound
