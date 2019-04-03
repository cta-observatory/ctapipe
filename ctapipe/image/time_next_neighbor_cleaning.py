import numpy as np
from scipy.spatial import cKDTree as KDTree
from ctapipe.instrument.camera import _get_min_pixel_seperation
from scipy.interpolate import interp1d

import astropy.units as u
import warnings
import json


def get_n_order_neighbor_matrix(geometry, order):
    """
    Get a matrix of the first and second order neighbors for each pixel.

    Parameters
    ----------
    geometry

    Returns
    -------
    Boolean matrix of dimension (n_pix, n_pix) with the first and second
    neighbors for all pixels.
    """
    neighbors = geometry.neighbor_matrix_sparse
    for i in range(order - 1):
        neighbors = neighbors.dot(geometry.neighbor_matrix_sparse)
    neighbors = neighbors.toarray()
    np.fill_diagonal(neighbors, 0)
    return neighbors


def dilate_matrix(mask, matrix):
    return np.dot(matrix, mask)


def check_number_of_neighbors(mask, geo, min_neighbors=3, order=2,
                              keep_neighbors=True):
    neighbor_matrix = get_n_order_neighbor_matrix(geo, order)
    n_neighbors = np.sum(mask * neighbor_matrix, axis=1) - 1

    valid_pixels = (n_neighbors >= min_neighbors)
    if keep_neighbors:
        valid_pixels = dilate_matrix(valid_pixels, neighbor_matrix)

    return valid_pixels & mask


class TimeNextNeighborCleaning:
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

    def __init__(self):
        self.IPR_dict = None
        self._multiplicity_dict = {'4nn': 4, '3nn': 3, '2+1': 3, '2nn': 2}
        self.__geometry = None
        self._cam_id = None
        self._IPR = None
        self.__neighbor_group = None
        self._multiplicity = None

    @property
    def _geometry(self):
        """
        Get or set the current `_geometry. Setting it to the new value also
        sets `_cam_id` and selects the corresponding `_IPR` from `IPR_dict`.
        """
        return self.__geometry

    @_geometry.setter
    def _geometry(self, geometry):
        self.__geometry = geometry
        self._cam_id = geometry.cam_id
        try:
            self._IPR = self.IPR_dict[geometry.cam_id]
        except KeyError as e:
            raise KeyError(f"No IPR graph for {geometry.cam_id} in"
                           f" self.IPR_dict.") from e

    @property
    def _neighbor_group(self):
        """
        Get or set the current _neighbor_group. Setting the _neighbor_group to
        a new value also changes `_multiplicity' to the corresponding value.
        """
        return self.__neighbor_group

    @_neighbor_group.setter
    def _neighbor_group(self, nfold):
        self.__neighbor_group = nfold
        self._multiplicity = self._multiplicity_dict[nfold]

    def load_individual_pixel_rates(self, file):
        """
        Load the IPR graphs from a json file. The dictionaries should contain
        one dictionary for each camera_id with the pixel rates and charges.

        Parameters
        ----------
        file: string
            json file in which the IPR graphs is stored
        """
        with open(file, "r") as f:
            self.IPR_dict = json.load(f)

    def dump_individual_pixel_rates(self, file):
        """
        Dump the IPR graph to file using json.

        Parameters
        ----------
        file: string

        """
        with open(file, "w") as f:
            json.dump(self.IPR_dict, f, sort_keys=True, indent=4)

    def fill_individual_pixel_rate(self, geo, image, trace_length,
                                   charge_steps):
        """
        Count the number of pixels in images about a given charge threshold
        and fill dictionary with individual pixel noise rate (IPR).

        Parameters
        ----------
        geo: ctapipe.instrument.CameraGeometry
        image: numpy.ndarray
        trace_length: astropy.units.Quantity
            total length of the trace, e.g. number of samples times sample_time
        charge_steps: numpy.ndarray
            array with the charge steps used to calculate the IPR graphs.
        """

        self.IPR_dict = self.IPR_dict if self.IPR_dict else {}

        if geo.cam_id not in self.IPR_dict:
            self.IPR_dict[geo.cam_id] = {"counter": 0,
                                         "charge": charge_steps,
                                         "npix": np.zeros(
                                             charge_steps.shape[0]),
                                         "rate": np.zeros(
                                             charge_steps.shape[0]),
                                         "rate_err": np.zeros(
                                             charge_steps.shape[0])
                                         }

        # count number of pixels
        self.IPR_dict[geo.cam_id]["counter"] += np.sum(image >= 0)
        for thbin, charge in enumerate(self.IPR_dict[geo.cam_id]["charge"]):
            self.IPR_dict[geo.cam_id]["npix"][thbin] += np.sum(image > charge)

        # convert count of number of pixels above threshold into rate
        sample_to_hz = (1 / (
                trace_length * self.IPR_dict[geo.cam_id]["counter"])).to(
            "Hz")
        self.IPR_dict[geo.cam_id]["rate"] = self.IPR_dict[geo.cam_id][
                                                "npix"] * sample_to_hz
        self.IPR_dict[geo.cam_id]["rate_err"] = np.sqrt(
            self.IPR_dict[geo.cam_id]["npix"]) * sample_to_hz

    def calculate_ipr_from_source(self, source, calibrator, charge_steps):
        """
        Fill the IPR graph from a file or a list of files. The files should
        contain pure night sky background calibration data. Therefore, the
        calibration of the events in the source are performed. The calibrator
        for this case should be equal to the one in the analysis.

        Parameters
        ----------
        source:
        calibrator:
        charge_steps: numpy.ndarray
            array storing the steps in charge used to create the IPR graph

        """
        for event in source:
            calibrator.calibrate(event)
            for tel_id in event.r0.tels_with_data:
                geometry = event.inst.subarray.tel[tel_id].camera
                image = event.dl1.tel[tel_id].image[0]

                n_samples = event.r0.tel[tel_id].num_samples
                sample_time = u.Quantity(event.mc.tel[tel_id].time_slice, u.ns)
                sum_time = sample_time * n_samples

                self.fill_individual_pixel_rate(geometry, image, sum_time,
                                                charge_steps)

    def scaled_combfactor_event_display(self):
        """
        Hardcoded values from EventDisplay scaled from 2400 pixels to the
        number of pixels. Nobody really understands where these numbers come
        from. Should use `combfactor_from_geometry` instead.

        Returns
        -------
        comb_factor: ndarray
            scales combinatorial factor
        """
        comb_factor = {'4nn': 12e5, '2+1': 95e4, '3nn': 13e4,
                       '2nn': 6e3, 'bound': 2}  # for 2400 pixels

        comb_factor.update(
            (x, y * self._geometry.n_pixels / 2400) for (x, y) in comb_factor.items())

        return comb_factor[self._neighbor_group]

    def combfactor_from_geometry(self, neighbor_group=None, cam_id=None,
                                 calculate=False):
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
        cam_id: string, None
        neighbor_group: string, None
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
        cam_id = cam_id if cam_id else self._cam_id
        neighbor_group = neighbor_group if neighbor_group else self._neighbor_group

        if not calculate:
            lookup = dict(LSTCam={'4nn': 75302, '3nn': 19269,
                                  '2nn': 5394, '2+1': 71712},
                          ASTRICam={'4nn': 41953, '3nn': 13548,
                                    '2nn': 4624, '2+1': 70904},
                          DigiCam={'4nn': 51843, '3nn': 13327,
                                   '2nn': 3744, '2+1': 49320},
                          FlashCam={'4nn': 71559, '3nn': 18319,
                                    '2nn': 5124, '2+1': 68112},
                          NectarCam={'4nn': 75302, '3nn': 19269,
                                     '2nn': 5394, '2+1': 71712},
                          CHEC={'4nn': 36289, '3nn': 11720,
                                '2nn': 4000, '2+1': 61432})
            comb_factor = lookup[cam_id][neighbor_group]

        else:
            from ctapipe.instrument import CameraGeometry

            geometry = CameraGeometry.from_name(cam_id)
            comb_factor = len(self.get_combinations(nfold=neighbor_group,
                                                    mask=geometry,
                                                    d2=2.4, d1=1.4))
            if neighbor_group == "2+1":
                # current implementation returns 2+1 and 3nn combinations for
                # 2+1 group search. Need to correct to get correct number.
                comb_3nn = len(self.get_combinations(nfold="3nn",
                                                     mask=geometry,
                                                     d2=2.4, d1=1.4))
                comb_factor -= comb_3nn

        return comb_factor

    def get_time_coincidence(self, charge, accidental_rate,
                             neighbor_group=None):
        """
        Get the time coincidence for a given charge (equation 2 of
        https://arxiv.org/pdf/1307.4939.pdf). The cut on the time coincidence
        depends on the charge, the multiplicity, combinatorial factor and the
        accepted accidental rate.

        Parameters
        ----------
        charge: numpy.ndarray
        accidental_rate: astropy.Quantity
            Allowed accidental rate for thi
        neighbor_group: string
            name of neighbor group. Select from 4nn, 3nn, 2+1 and 2nn

        Returns
        -------
        time_coincidence: ndarray
            array of same shape of `charge` filled with the corresponding cut
            in time
        """
        if neighbor_group is not None:
            self._neighbor_group = neighbor_group

        ipr_graph = interp1d(self._IPR["charge"], self._IPR["rate"], "linear")

        try:
            value_ipr = ipr_graph(charge)
        except ValueError as e:
            warnings.warn(f"Interpolation raised exception: '{e}' Will try "
                          f"solution for charges above of IPR graph.")
            # too large charge replaced by minimum rate
            mask = charge > max(ipr_graph.x)
            value_ipr = np.zeros_like(charge)
            value_ipr[~mask] = ipr_graph(charge[~mask])
            value_ipr[mask] = min(ipr_graph.y)

        nfold = self._multiplicity
        cfactor = self.combfactor_from_geometry(neighbor_group=None)

        time_coincidence = np.exp((1 / (nfold - 1)) *
                                  np.log(accidental_rate.to("Hz").value /
                                         (cfactor * value_ipr ** nfold)))

        return u.Quantity(time_coincidence * 1e9, "ns")

    def cut_time_charge(self, charge, time_difference, accidental_rate,
                        neighbor_group):
        """
        Apply cut in time difference and charge parameter space.

        Parameters
        ----------
        charge: ndarry
        time_difference: astropy.units.Quantity
        accidental_rate:
        neighbor_group: string

        Returns
        -------
        valid_pixels: ndarray
            mask with pixels survived the cut

        """

        time_coincidence = self.get_time_coincidence(charge, accidental_rate,
                                                     neighbor_group)
        valid_pixels = time_difference < time_coincidence

        return valid_pixels

    def pre_threshold_event_display(self):
        """
        The pre threshold that where hardcoded in EventDisplay. Not sure how
        they where selected.

        Returns
        -------
        pre_threshold : float
            Charge of pixels with rate below threshold

        """
        thresh_freqency = {"4nn": u.Quantity(8.5e6, "Hz"),
                           "2+1": u.Quantity(2.4e6, "Hz"),
                           "3nn": u.Quantity(4.2e6, "Hz"),
                           "2nn": u.Quantity(1.5e5, "Hz")
                           }

        bins_above_thresh = np.sum(self._IPR["rate"] >=
                                   thresh_freqency[self._neighbor_group])

        return self._IPR["charge"][bins_above_thresh - 1]

    def pre_threshold_from_sample_time(self, accidental_rate, sample_time,
                                       neighbor_group=None, factor=0.2):
        """
        Get the pre threshold from the sampling time. The charge of all pixels
        should be high enough, so that the cut in the in the charge-dT space is
        above this sample time multiplied by an factor.

        Parameters
        ----------
        accidental_rate: float
        sample_time: astropy.units.Quantity
            Sampling time of the camera
        neighbor_group: string, None
            Group of pixels to consider
        factor: float

        Returns
        -------
        pre_threshold : float
            Charge of pixels with rate below threshold

        """
        charges = np.linspace(0, 20, 1000)
        dt = self.get_time_coincidence(charges, accidental_rate,
                                       neighbor_group=neighbor_group)

        return np.min(charges[dt > factor * sample_time])

    def get_kdtree(self, mask=None):
        """
        Initialize a KDtree for this camera considering only pixels that are
        not masked.

        Parameters
        ----------
        mask: list
            boolean list specifying pixels to consider in the tree

        Returns
        -------
        kdtree: scipy.spatial.cKDTree
        points: numpy.ndarray
            points used in the tree
        """
        if mask is None:
            mask = np.ones_like(self._geometry.pix_id, bool)

        points = np.array([self._geometry.pix_x[mask],
                           self._geometry.pix_y[mask]]).T
        kdtree = KDTree(points)

        return kdtree, points

    @staticmethod
    def add_neighbors_to_combinations(combinations, neighbors):
        """
        For each combination in the combinations list, the neighbors specified
        are added resulting in all possible combinations with one pixel more
        than the input combinations.

        Parameters
        ----------
        combinations: list, tuple, numpy.ndarray
            input of N pairs of combinations of M pixels
        neighbors: list
            2d list with pixel IDs considered as neighbors of each pixel

        Returns
        -------
        result: numpy.array
            array with all possible combinations of M+1 pixels
        """
        # Convert ndarray to list as adding values is faster
        # TODO: Is this still needed?
        if type(combinations) == np.ndarray:
            combinations = combinations.tolist()

        result = []
        for comb in combinations:
            neigh = []
            # Found that it's faster to do this in two passes: first get
            # the list of all neighbors to comb. This can have the same pixel
            # multiple times so later only unique are selected. This is faster
            # than checking if this value already is in the list each time.
            for c in comb:
                neigh += neighbors[c]
            neigh = np.unique(neigh)

            # Add the combinations of initial combination plus the neighbors
            # of it to the result. If the neighbor is already in the
            # combination this one will not be considered.
            for n in neigh:
                if n in comb:
                    continue
                if type(comb) == tuple:
                    result.append(comb + (n,))
                elif type(comb) == list:
                    result.append(comb + [n])

        # As it might happen, that a combination is added multiple times, those
        # duplicates are removed.
        if len(result) > 0:
            result = np.sort(result, axis=1)
            result = np.unique(result, axis=0)
        else:
            result = np.array(result)

        return result

    def get_combinations(self, nfold, mask=None, d2=2.4, d1=1.4):
        """
        Get a list a all possible combinations of pixels for a given geometry.
        The implementation is based on constructing a KDtree for the pixels not
        masked. For considering two pixels as neighbors, the minimum distance
        between two pixels is considered. Direct neighbors are closer than d1
        times the min distance while the second nearest neighbors for the 2+1
        group takes d2 instead.
        First the 2nn pairs are calculated and the candidate neighbors are
        added to these pairs (iteratively) to construct the 3nn, 4nn and 2+1
        groups.

        Parameters
        ----------
        mask: numpy.ndarray
            boolean mask of pixels to consider for search
        d1: float
            Search for first neighbors in `d1` times the minimum distance
            between pixels.
        d2: float
            Search for second neighbors in `d2` times the minimum distance
            between pixels.
        nfold: string

        Returns
        -------
        combs: numpy.ndarray
            array of all possible combinations for given neighbor group
        """

        if mask is None:
            mask = np.ones(self._geometry.n_pixels, bool)

        # construct an kdtree that with only the points that are not masked.
        # TODO: Check the updates on the calculation of neighbors in ctapipe.
        kdtree, points = self.get_kdtree(mask)
        dist = _get_min_pixel_seperation(self._geometry.pix_x,
                                         self._geometry.pix_y)

        # kdtree implementation to get the possible combinations of pairs
        # within a given distance.
        combs = kdtree.query_pairs(r=d1 * dist.value)
        if nfold == "2nn":
            combs = list(combs)
        elif nfold == "2+1":
            # Returns all possible 2+1 one AND 3nn combinations. As 3nn cut
            # is looser than 2+1 cut anyway, this will not influence the result
            # but will imply some unnescessary testing of combinations.
            neighbors2 = [kdtree.query_ball_point(p, d2 * dist.value)
                          for p in points]
            combs = self.add_neighbors_to_combinations(combs, neighbors2)

        elif nfold in ("3nn", "4nn"):
            # add one neighbor to 2nn combinations
            neighbors = [kdtree.query_ball_point(p, d1 * dist.value)
                         for p in points]
            combs = self.add_neighbors_to_combinations(combs, neighbors)

            if nfold == "4nn":
                # add one more neighbor to the 3nn combinations to get possible
                # 4nn combinations.
                combs = self.add_neighbors_to_combinations(combs, neighbors)
        else:
            NotImplementedError(f'Search for {nfold}-group not implemented.')

        if len(combs) > 0:
            combs = np.array([self._geometry.pix_id[mask]])[0, combs]
        else:
            combs = np.array([])

        return combs

    @staticmethod
    def check_combinations(combinations, mask):
        """
        Check if all pixels of the combination already passed the cut. If so,
        those don't need to be double checked for other neighbor groups
        Faster than using numpy.all by factor ~3.

        Parameters
        ----------
        combinations: numpy.ndarray
            array storing the indices of the combinations.
        mask: boolean ndarray
        """
        passed_pixel = mask[combinations]

        all_pixels = np.zeros(len(combinations), bool)
        for i in range(passed_pixel.shape[1]):
            all_pixels += ~passed_pixel[:, i]

        return all_pixels

    def clean(self, geometry, image, arrival_times, sample_time, sum_time,
              fake_prob=0.001, factor=1):
        """
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
        geometry: ctapipe.instrument.CameraGeometry
        sample_time: astropy.units.Quantity
            timing information for each sample
        image: numpy.ndarray
        arrival_times: numpy.ndarray
            arrival time in number of samples
        fake_prob: float
        sum_time: astropy.units.Quantity
        factor: float

        Returns
        -------
        survived_pixels: numpy.ndarray
            mask with pixels that survived the image cleaning
        """
        self._geometry = geometry

        survived_pixels = np.zeros(geometry.n_pixels, bool)
        # for nfold in ["4nn", "3nn", "2+1", "2nn"]:
        for nfold in ["2nn", "3nn", "4nn", "2+1"]:
            self._neighbor_group = nfold

            # determine the accidental rate from fake proability
            rate_acc = fake_prob / (sum_time * self._multiplicity)
            pre_threshold = self.pre_threshold_from_sample_time(rate_acc,
                                                                sample_time,
                                                                factor=factor)
            candidates = image > pre_threshold
            if sum(candidates) == 0:
                continue

            # Calculate the possible combinations form the candidates that
            # passed the pre-threshold.
            combinations = self.get_combinations(nfold, candidates)
            if len(combinations) < 1:
                continue
            # It can be that all pixels of a found combination already passed
            # the cuts fo a different neighbor group. Those don't have to be
            # double checked and are therefore removed. This improves a little
            # bit the performance for large images.
            checked_combinations = self.check_combinations(combinations,
                                                           survived_pixels)
            combinations = combinations[checked_combinations]

            # Get the charges and times for all combinations. For the charge
            # the minimum value of this group is considered while the time
            # time difference is the maximum in this group. With this
            # definitions the cuts applied should be more conservative.
            combination_image = np.array([image])[0, combinations]
            combination_times = np.array([arrival_times])[0, combinations]
            min_charge = np.min(combination_image, axis=1)
            time_diff = (np.max(combination_times, axis=1) -
                         np.min(combination_times, axis=1))
            time_diff = time_diff * sample_time

            # Apply the cut in the time and charge parameter space to check
            # which group of pixels are valid. The pixels can appear in
            # multiple groups.
            valid_groups = self.cut_time_charge(min_charge, time_diff,
                                                rate_acc.to("Hz"), nfold)
            valid_pixels = np.unique(combinations[valid_groups])

            survived_pixels[valid_pixels] = True

        survived_pixels = check_number_of_neighbors(survived_pixels, self._geometry, 4, 2)
        bound = self.boundary_search(survived_pixels, image, arrival_times, sample_time, sum_time,
                                     fake_prob=0.001)

        return survived_pixels, bound

    def boundary_search(self, mask, image, arrival_times, sample_time, sum_time,
                        fake_prob=0.001):

        bound = np.zeros(self._geometry.n_pixels, bool)
        if any(mask):
            second_matrix = get_n_order_neighbor_matrix(self._geometry, 2)
            candidates = dilate_matrix(mask, second_matrix)
            candidates[image <= 0] = 0

            # Calculate the possible combinations form the candidates that
            # passed the pre-threshold.
            for nfold in ["2nn", "3nn", "4nn", "2+1"]:
                rate_acc = fake_prob * 50 / (sum_time * self._multiplicity)

                combinations = self.get_combinations(nfold, candidates)
                if len(combinations) < 1:
                    continue
                #checked_combinations = self.check_combinations(combinations, mask)
                #combinations = combinations[checked_combinations]

                combination_image = np.array([image])[0, combinations]
                combination_times = np.array([arrival_times])[0, combinations]
                time_diff = (np.max(combination_times, axis=1) -
                             np.min(combination_times, axis=1))
                time_diff = time_diff * sample_time

                min_charge = np.min(combination_image, axis=1)
                valid_groups = self.cut_time_charge(min_charge, time_diff,
                                                    rate_acc.to("Hz"), nfold)

                valid_pixels = np.unique(combinations[valid_groups])
                bound[valid_pixels] = True

        return bound
