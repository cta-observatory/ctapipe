import numpy as np
from scipy.spatial import cKDTree as KDTree
from ctapipe.instrument.camera import _get_min_pixel_seperation
from scipy.interpolate import interp1d

import astropy.units as u
import warnings

multiplicity = {'4nn': 4,
                '3nn': 3,
                '2+1': 3,
                '2nn': 2}

def fill_IPR_from_calibration_file(geo, image, trace_length, IPR, mincharge=0,
                                   maxcharge=25, steps=50):
    """
    Count the number of pixels in images about a given charge threshold
    and fill dictionary with individual pixel noise rate (IPR).

    Parameters
    ----------
    geo: ctapipe.instrument.CameraGeometry
    image: numpy.ndarray
    trace_length: astropy.units.Quantity
        total length of the trace, e.g. number of samples times sample_time
    IPR: dictionary
        dictionary which should be filled with IPR graph for each camera.
    mincharge: float, integer
    maxcharge: float, integer
    steps: integer
    """

    # RATE = np.zeros(IPRdim + 1)
    if geo.cam_id not in IPR:
        IPR[geo.cam_id] = {"counter": 0,  # number of pixels considered
                           "charge": np.linspace(mincharge, maxcharge,
                                                 steps + 1),  # threshold values
                           "npix": np.zeros(steps + 1),  # pixels about threshold
                           "rate": np.zeros(steps + 1),  # individual pixel rate
                           "rate_err": np.zeros(steps + 1)  # error rate
                           }

    # count number of pixels
    IPR[geo.cam_id]["counter"] += np.sum(image >= 0)

    # loop over all thresholds
    for thbin, val in enumerate(IPR[geo.cam_id]["charge"]):
        IPR[geo.cam_id]["npix"][thbin] += np.sum(image > val)

    SampleToHz = (1 / (trace_length * IPR[geo.cam_id]["counter"])).to("Hz")
    IPR[geo.cam_id]["rate"] = IPR[geo.cam_id]["npix"] * SampleToHz
    IPR[geo.cam_id]["rate_err"] = np.sqrt(IPR[geo.cam_id]["npix"]) * SampleToHz

def scaled_combfactor_from_name(npix, neighbor_group):
    """
    Hardcoded values from EventDisplay scaled from 2400 pixels to the correct
    number of pixels.

    TODO: Where do those magic numbers come from ???

    Parameters
    ----------
    npix: int
        number of pixels in camera
    neighbor_group: string
        group of neighboring pixels

    Returns
    -------
    comb_factor: ndarray
        scales combinatorial factor
    """
    comb_factor = {'4nn': 12e5, '2+1': 95e4, '3nn': 13e4,
                   '2nn': 6e3, 'bound': 2}  # for 2400 pixels

    comb_factor.update((x, y * npix / 2400) for (x, y) in comb_factor.items())

    return comb_factor[neighbor_group]


def combfactor_from_geometry(cam_id=None, neighbor_group=None,
                             calculate=False):
    """
    Get the accurate combinatorial factors from the geometry of the camera by
    counting the number of possible combinations for a given group of pixels.
    If `calculate` is `True`, the number of combinations will be calculated
    for the specified camera and neighbor group.
    Otherwise the values are read from a lookup table which was filled for
    complete cameras, e.g. assuming no broken pixels.

    Parameters
    ----------
    cam_id: string
    neighbor_group: string
    calculate: bool
        If true, the value will be calculated from the geometry of the
        camera. Otherwise the values from a prefilled lookup table are
        returned

    Returns
    -------
    comb_factor: int
        number of possible combinations for camera type and neighbor group

    """
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

    elif calculate:
        from ctapipe.instrument import CameraGeometry

        geometry = CameraGeometry.from_name(cam_id)
        mask = np.ones(geometry.n_pixels, bool)

        comb_factor = len(get_combinations(geometry, mask=mask, nfold=neighbor_group, d2=2.4, d1=1.4))
        if neighbor_group == "2+1":
            # current implementation returns 2+1 and 3nn combinations for
            # 2+1 group search. Need to correct to get correct number.
            comb_3nn = len(get_combinations(geometry, mask=mask, nfold="3nn", d2=2.4, d1=1.4))
            comb_factor -= comb_3nn

    return comb_factor


def get_time_coincidence(geometry, IPR, charge, accidental_rate,
                         neighbor_group):
    """
    Get the time coincidence for a given charge (equation 2 of
    https://arxiv.org/pdf/1307.4939.pdf). The cut on the time coincidence
    depends on the charge, the multiplicity, combinatorial factor and the
    accepted accidental rate.

    Parameters
    ----------
    charge: numpy.ndarray
    IPR : dictionary
    geometry: ctapipe.instrument.CameraGeometry
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
    IPR_graph = interp1d(IPR["charge"], IPR["rate"], "linear")

    try:
        value_IPR = IPR_graph(charge)
    except ValueError as e:
        warnings.warn(f"Interpolation raised exception: '{e}' Will try "
                      f"solution for charges above of IPR graph.")
        # too large charge replaced by minimum rate
        mask = charge > max(IPR_graph.x)
        value_IPR = np.zeros_like(charge)
        value_IPR[~mask] = IPR_graph(charge[~mask])
        value_IPR[mask] = min(IPR_graph.y)

    nfold = multiplicity[neighbor_group]
    cfactor = combfactor_from_geometry(geometry.cam_id, neighbor_group)

    time_coincidence = np.exp((1 / (nfold - 1)) *
                              np.log(accidental_rate.to("Hz").value /
                                     (cfactor * value_IPR ** nfold)))

    return u.Quantity(time_coincidence * 1e9, "ns")


def cut_time_charge(IPR, charge, time_difference, geometry,
                    accidental_rate, neighbor_group):
    """
    Apply cut in time difference and charge parameter space.

    Parameters
    ----------
    IPR: dictionary
    charge: ndarry
    time_difference: astropy.units.Quantity
    geometry: ctapipe.instrument.CameraGeometry
    accidental_rate:
    neighbor_group: string

    Returns
    -------
    valid_pixels: ndarray
        mask with pixels survived the cut

    """

    time_coincidence = get_time_coincidence(geometry, IPR, charge, accidental_rate, neighbor_group)

    valid_pixels = time_difference < time_coincidence

    return valid_pixels


def pre_threshold_ED(IPR, neighbor_group):
    """
    Get some values for the pre threshold based on the individual pixel rate.
    This pre cut es required to throw away pixels with low charge.

    Parameters
    ----------
    IPR: dictionary
    neighbor_group: string

    Returns
    -------
    pre_threshold : float
        Charge of pixels with rate below threshold

    """
    threshfreq = {"4nn": u.Quantity(8.5e6, "Hz"),
                  "2+1": u.Quantity(2.4e6, "Hz"),
                  "3nn": u.Quantity(4.2e6, "Hz"),
                  "2nn": u.Quantity(1.5e5, "Hz"),
                  "bound": u.Quantity(1.1e7, "Hz"),
                  "refbound": u.Quantity(4.0e5, "Hz")
                  }

    bins_above_thresh = np.sum(IPR["rate"] >= threshfreq[neighbor_group])

    return IPR["charge"][bins_above_thresh - 1]


def pre_threshold_from_sample(geometry, IPR, neighbor_group, accidental_rate,
                              sample_time, factor=0.2):
    """
    Get the pre threshold from the sampling time. The charge of all pixels
    should be high enough, so that the cut in the in the charge-dT space is
    above this sample time multiplied by an factor.

    Parameters
    ----------
    IPR: dictionary
    neighbor_group: string
        Group of pixels to consider
    geometry: ctapipe.instrument.CameraGeometry
    sample_time: astropy.units.Quantity

    Returns
    -------
    pre_threshold : float
        Charge of pixels with rate below threshold

    """

    charges = np.linspace(0, 20, 1000)
    dT = get_time_coincidence(geometry, IPR, charges, accidental_rate, neighbor_group)

    return np.min(charges[dT > factor * sample_time])


def get_kdtree(geometry, mask=None):
    """
    Initialize a KDtree for this camera considering only pixels that are not
    masked.

    Parameters
    ----------
    geometry: ctapipe.instrument.CameraGeometry
    mask: list
        boolean list specifying pixels to consider in the tree

    Returns
    -------
    kdtree: scipy.spatial.cKDTree
    points: numpy.ndarray
        points used in the tree
    """
    if mask is None:
        mask = np.ones_like(geometry.pix_id, bool)

    points = np.array([geometry.pix_x[mask], geometry.pix_y[mask]]).T
    kdtree = KDTree(points)

    return kdtree, points


def add_neighbors_to_combinations(combinations, neighbors):
    """
    For each combination in the combinations list, the neighbors specified are
    added resulting in all possible combinations with one pixel more than the
    input combinations.

    Parameters
    ----------
    combinations: list, tuple
        input of N pairs of combinations of M pixels
    neighbors: list
        2d list with pixel IDs that are considered as neighbors of each pixel

    Returns
    -------
    result: numpy.array
        array with all possible combinations of M+1 pixels
    """
    if type(combinations) == np.ndarray:
        combinations = combinations.tolist()

    result = []
    for comb in combinations:
        neigh = []
        for c in comb:
            neigh += neighbors[c]
        neigh = np.unique(neigh)
        for c in comb:
            neigh = neigh[neigh != c]

        for n in neigh:
            if type(comb) == tuple:
                result.append(comb + (n,))
            elif type(comb) == list:
                result.append(comb + [n])

    if len(result) > 0:
        # remove duplicate combinations
        result = np.sort(result, axis=1)
        result = np.unique(result, axis=0)
    else:
        result = np.array(result)

    return result


def get_combinations(geometry, mask=None, nfold="3nn", d2=2.4, d1=1.4):
    """
    Get a list a all possible combinations of pixels for a given geometry. The
    implementation is based on constructing a KDtree for the pixels not
    masked. For considering two pixels as neighbors, the minimum distance
    between two pixels is considered. Direct neighbors are closer than d1
    times the min distance while the second nearest neighbors for the 2+1
    group takes d2 instead.
    First the 2nn pairs are calculated and the candidate neighbors are added
    to these pairs (iteratively) to construct the 3nn, 4nn and 2+1 groups.

    Parameters
    ----------
    geometry: ctapipe.instrument.CameraGeometry
    mask: list
        boolean mask of pixels to consider for search
    d1: float
        Search for first neighbors in `d1` times the minimum distance between
        pixels.
    d2: float
        Search for second neighbors in `d2` times the minimum distance between
        pixels.
    nfold: string


    Returns
    -------
    combs: numpy.ndarray
        array of all possible combinations for given neighbor group
    """
    if mask is None:
        mask = np.ones(geometry.n_pixels, bool)

    kdtree, points = get_kdtree(geometry, mask)
    dist = _get_min_pixel_seperation(geometry.pix_x, geometry.pix_y)

    # 2nn pairs of neighbors
    combs = kdtree.query_pairs(r=d1 * dist.value)
    if nfold == "2nn":
        combs = list(combs)
    elif nfold == "2+1":
        # Returns all possible 2+1 one AND 3nn combinations. As 3nn cut
        # is looser than 2+1 cut anyway, this will not influence the result.
        neighbors2 = [kdtree.query_ball_point(p, d2 * dist.value) for p in points]
        combs = add_neighbors_to_combinations(combs, neighbors2)

    elif nfold in ("3nn", "4nn"):
        # add one neighbor to 2nn combinations
        neighbors = [kdtree.query_ball_point(p, d1 * dist.value) for p in points]
        combs = add_neighbors_to_combinations(combs, neighbors)

        if nfold == "4nn":
            # add one more neighbor to the pairs
            combs = add_neighbors_to_combinations(combs, neighbors)
    else:
        NotImplementedError(f'Search for {nfold} pixel group not implemented.')

    if len(combs) > 0:
        combs = np.array([geometry.pix_id[mask]])[0, combs]
    else:
        combs = np.array([])

    return combs


def time_next_neighbor_cleaning(IPR, sample_time, geometry, image,
                                arrival_times, fake_prob, sum_time,
                                factor=0.4):
    """
    Main function of time next neighbor cleaning developed by M. Shayduk.
    See https://arxiv.org/pdf/1307.4939.pdf.
    Search for 2nn, 3nn, 4nn and 2+1 neighbors group with a minimum charge and
    small difference in arrival time. For each potential combination of
    pixels, the minimum charge and the maximum time difference are considered
    for checking the validity with the cut in the charge-time space.

    It's very likely to have neighboring low charge pixels with exactly the
    same sample time. However, as the charge-time cut will greater 0 for
    all charges, those combinations would potentially pass this cut. To avoid
    this, a pre cut on the charge is crucial.

    Parameters
    ----------
    IPR: dictionary
    sample_time: astropy.units.Quantity
        timimng informatin for each sample
    geometry: ctapipe.instrument.CameraGeometry
    image: numpy.ndarray
    arrival_times: numpy.ndarray
        arrival time in number of samples

    Returns
    -------
    survived: numpy.ndarray
        mask with pixels that survived the image cleaning
    """
    nn = {"4nn": 4,
          "3nn": 3,
          "2+1": 3,
          "2nn": 2}

    survived = np.zeros_like(geometry.pix_id, bool)
    for neighbor_group in ["4nn", "3nn", "2+1", "2nn"]:
        accidental_rate = fake_prob / (sum_time * nn[neighbor_group])

        pre_threshold = pre_threshold_from_sample(geometry, IPR, neighbor_group, accidental_rate,
                                                  sample_time, factor=factor)

        candidates = image > pre_threshold

        if sum(candidates) == 0:
            continue

        combinations = get_combinations(geometry, candidates, nfold=neighbor_group)
        if len(combinations) < 1:
            continue

        combination_image = np.array([image])[0, combinations]
        combination_times = np.array([arrival_times])[0, combinations]

        min_charge = np.min(combination_image, axis=1)
        time_diff = np.max(combination_times, axis=1) -\
                    np.min(combination_times, axis=1)
        time_diff = time_diff * sample_time

        valid_groups = cut_time_charge(IPR, min_charge, time_diff, geometry,
                                       accidental_rate.to("Hz"), neighbor_group)
        valid_pixels = np.unique(combinations[valid_groups])

        survived[valid_pixels] = True

    return survived
