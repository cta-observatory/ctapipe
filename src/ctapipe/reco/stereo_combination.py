from abc import abstractmethod

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, CartesianRepresentation, SphericalRepresentation
from astropy.table import Table
from traitlets import UseEnum

from ctapipe.core import Component, Container
from ctapipe.core.traits import Bool, CaselessStrEnum, Float, Unicode
from ctapipe.reco.reconstructor import ReconstructionProperty

from ..compat import COPY_IF_NEEDED
from ..containers import (
    ArrayEventContainer,
    ParticleClassificationContainer,
    ReconstructedEnergyContainer,
    ReconstructedGeometryContainer,
)
from .preprocessing import telescope_to_horizontal
from .telescope_event_handling import (
    calc_combs_min_distances_event,
    calc_combs_min_distances_table,
    calc_fov_lon_lat,
    create_combs_array,
    get_combinations,
    get_index_combs,
    get_subarray_index,
    weighted_mean_std_ufunc,
)
from .utils import add_defaults_and_meta

_containers = {
    ReconstructionProperty.ENERGY: ReconstructedEnergyContainer,
    ReconstructionProperty.PARTICLE_TYPE: ParticleClassificationContainer,
    ReconstructionProperty.GEOMETRY: ReconstructedGeometryContainer,
}

__all__ = [
    "StereoCombiner",
    "StereoMeanCombiner",
    "StereoDispCombiner",
]


class StereoCombiner(Component):
    """
    Base class for algorithms that combine telescope-wise (mono) predictions
    into a single array-level (stereo) reconstruction.

    A StereoCombiner defines the interface for transforming per-telescope
    DL2 predictions (energy, direction, particle type) into a unified
    stereo estimate. Subclasses implement the actual combination strategy.

    Two usage modes are supported:

    1. **Event-wise combination** via :meth:`__call__`
       Operates directly on an :class:`~ctapipe.containers.ArrayEventContainer`
       containing DL1 and DL2 mono predictions for a single event, and writes
       the stereo-level result into ``event.dl2.stereo``.

    2. **Table-wise combination** via :meth:`predict_table`
       Takes a table of DL2 mono predictions (one row per telescope-event)
       and constructs an output table with one row per array event containing
       the stereo predictions.

    Subclasses must implement both methods, as well as any additional logic
    required for combining the chosen :class:`~ctapipe.reco.ReconstructionProperty`.
    """

    prefix = Unicode(
        default_value="",
        help="Prefix to be added to the output container / column names.",
    ).tag(config=True)

    property = UseEnum(
        ReconstructionProperty,
        help=(
            "Reconstruction property to be combined (e.g. ENERGY, GEOMETRY, "
            "PARTICLE_TYPE). Subclasses may support only a subset."
        ),
    ).tag(config=True)

    @abstractmethod
    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Compute the stereo prediction for a single array event.

        Parameters
        ----------
        event : ArrayEventContainer
            Event containing DL1 parameters and per-telescope DL2 predictions.
            Implementations must write the combined stereo prediction into
            ``event.dl2.stereo`` under the configured ``prefix``.

        Notes
        -----
        - This method modifies the event container *in place*.
        """

    @abstractmethod
    def predict_table(self, mono_predictions: Table) -> Table:
        """
        Construct stereo predictions from a table of mono DL2 predictions.

        Parameters
        ----------
        mono_predictions : astropy.table.Table
            Table containing one row per telescope-event and the
            DL2 output corresponding to the configured reconstruction property.

        Returns
        -------
        stereo_table : astropy.table.Table
            A table with one row per array event, containing the combined
            stereo predictions.
        """


class StereoMeanCombiner(StereoCombiner):
    """
    Combine telescope-wise mono reconstructions using a (weighted) mean.

    This implementation supports combining energy, geometry and
    particle-type predictions. Different weighting schemes can be chosen
    via the ``weights`` configuration trait. Supported weighting options are:

    - ``none``: all telescopes contribute equally
    - ``intensity``: weight proportional to image intensity
    - ``konrad``: intensity × (length/width)

    If ``log_target=True`` and ``ENERGY`` is combined, the combiner computes
    the geometric mean by averaging the logarithm of the energies.

    See :class:`StereoCombiner` for a description of the general interface.
    """

    weights = CaselessStrEnum(
        ["none", "intensity", "aspect-weighted-intensity"],
        default_value="none",
        help=(
            "What kind of weights to use. Options: ``none``, ``intensity``, ``aspect-weighted-intensity``."
        ),
    ).tag(config=True)

    log_target = Bool(
        False,
        help="If true, calculate exp(mean(log(values))).",
    ).tag(config=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.supported = {
            ReconstructionProperty.ENERGY,
            ReconstructionProperty.GEOMETRY,
            ReconstructionProperty.PARTICLE_TYPE,
        }
        if self.property not in self.supported:
            raise NotImplementedError(
                f"Combination of {self.property} not implemented in {self.__class__.__name__}"
            )

    def _calculate_weights(self, data):
        if isinstance(data, Container):
            if self.weights == "intensity":
                return data.hillas.intensity

            if self.weights == "aspect-weighted-intensity":
                return data.hillas.intensity * data.hillas.length / data.hillas.width

            return 1

        if isinstance(data, Table):
            if self.weights == "intensity":
                return data["hillas_intensity"]

            if self.weights == "aspect-weighted-intensity":
                return (
                    data["hillas_intensity"]
                    * data["hillas_length"]
                    / data["hillas_width"]
                )

            return np.ones(len(data))

        raise TypeError(
            "Dl1 data needs to be provided in the form of a container or astropy.table.Table"
        )

    def _combine_energy(self, event):
        ids = []
        values = []
        weights = []

        for tel_id, dl2 in event.dl2.tel.items():
            mono = dl2.energy[self.prefix]
            if mono.is_valid:
                values.append(mono.energy.to_value(u.TeV))
                if tel_id not in event.dl1.tel:
                    raise ValueError("No parameters for weighting available")
                weights.append(
                    self._calculate_weights(event.dl1.tel[tel_id].parameters)
                )
                ids.append(tel_id)

        if len(values) > 0:
            weights = np.array(weights, dtype=np.float64)
            weights /= weights.max()

            if self.log_target:
                values = np.log(values)

            mean = np.average(values, weights=weights)
            std = np.sqrt(np.cov(values, aweights=weights))

            if self.log_target:
                mean = np.exp(mean)
                std = np.exp(std)

            valid = True
        else:
            mean = std = np.nan
            valid = False

        event.dl2.stereo.energy[self.prefix] = ReconstructedEnergyContainer(
            energy=u.Quantity(mean, u.TeV, copy=COPY_IF_NEEDED),
            energy_uncert=u.Quantity(std, u.TeV, copy=COPY_IF_NEEDED),
            telescopes=ids,
            is_valid=valid,
            prefix=self.prefix,
        )

    def _combine_classification(self, event):
        ids = []
        values = []
        weights = []

        for tel_id, dl2 in event.dl2.tel.items():
            mono = dl2.particle_type[self.prefix]
            if mono.is_valid:
                values.append(mono.prediction)
                dl1 = event.dl1.tel[tel_id].parameters
                weights.append(self._calculate_weights(dl1) if dl1 else 1)
                ids.append(tel_id)

        if len(values) > 0:
            mean = np.average(values, weights=weights)
            valid = True
        else:
            mean = np.nan
            valid = False

        container = ParticleClassificationContainer(
            prediction=mean, telescopes=ids, is_valid=valid, prefix=self.prefix
        )
        event.dl2.stereo.particle_type[self.prefix] = container

    def _combine_altaz(self, event):
        ids = []
        alt_values = []
        az_values = []
        weights = []

        for tel_id, dl2 in event.dl2.tel.items():
            mono = dl2.geometry[self.prefix]
            if mono.is_valid:
                alt_values.append(mono.alt)
                az_values.append(mono.az)
                dl1 = event.dl1.tel[tel_id].parameters
                weights.append(self._calculate_weights(dl1) if dl1 else 1)
                ids.append(tel_id)

        if len(alt_values) > 0:  # by construction len(alt_values) == len(az_values)
            coord = AltAz(alt=alt_values, az=az_values)
            # https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution#Mean_direction
            mono_x, mono_y, mono_z = coord.cartesian.get_xyz()
            stereo_x = np.average(mono_x, weights=weights)
            stereo_y = np.average(mono_y, weights=weights)
            stereo_z = np.average(mono_z, weights=weights)

            mean_cartesian = CartesianRepresentation(x=stereo_x, y=stereo_y, z=stereo_z)
            mean_spherical = mean_cartesian.represent_as(SphericalRepresentation)
            mean_altaz = AltAz(mean_spherical)

            # https://en.wikipedia.org/wiki/Directional_statistics#Measures_of_location_and_spread
            r = mean_spherical.distance.to_value()
            std = np.sqrt(-2 * np.log(r))

            valid = True
        else:
            mean_altaz = AltAz(
                alt=u.Quantity(np.nan, u.deg, copy=COPY_IF_NEEDED),
                az=u.Quantity(np.nan, u.deg, copy=COPY_IF_NEEDED),
            )
            std = np.nan
            valid = False

        event.dl2.stereo.geometry[self.prefix] = ReconstructedGeometryContainer(
            alt=mean_altaz.alt,
            az=mean_altaz.az,
            ang_distance_uncert=u.Quantity(np.rad2deg(std), u.deg, copy=COPY_IF_NEEDED),
            telescopes=ids,
            is_valid=valid,
            prefix=self.prefix,
        )

    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Implement :meth:`StereoCombiner.__call__` using a weighted mean
        combination of the configured reconstruction property.
        """

        properties = [
            self.property & itm
            for itm in self.supported
            if self.property & itm in ReconstructionProperty
        ]
        for prop in properties:
            if prop is ReconstructionProperty.ENERGY:
                self._combine_energy(event)

            elif prop is ReconstructionProperty.PARTICLE_TYPE:
                self._combine_classification(event)

            elif prop is ReconstructionProperty.GEOMETRY:
                self._combine_altaz(event)

    def predict_table(self, mono_predictions: Table) -> Table:
        """
        Calculates the (array-)event-wise mean.

        Telescope events, that are nan, get discarded. This means
        you might end up with less events if all telescope predictions
        of a shower are invalid.

        See :meth:`StereoCombiner.predict_table` for the general
        input/output conventions.
        """

        prefix = f"{self.prefix}_tel"
        # TODO: Integrate table quality query once its done
        valid = mono_predictions[f"{prefix}_is_valid"]

        obs_ids, event_ids, multiplicity, tel_to_array_indices = get_subarray_index(
            mono_predictions
        )
        n_array_events = len(obs_ids)
        stereo_table = Table({"obs_id": obs_ids, "event_id": event_ids})
        # copy metadata
        for colname in ("obs_id", "event_id"):
            stereo_table[colname].description = mono_predictions[colname].description

        weights = self._calculate_weights(mono_predictions[valid])

        if self.property is ReconstructionProperty.PARTICLE_TYPE:
            if np.count_nonzero(valid) > 0:
                stereo_predictions, _ = weighted_mean_std_ufunc(
                    mono_predictions[f"{prefix}_prediction"],
                    valid,
                    tel_to_array_indices,
                    multiplicity,
                    weights=weights,
                )
            else:
                stereo_predictions = np.full(n_array_events, np.nan)

            stereo_table[f"{self.prefix}_prediction"] = stereo_predictions
            stereo_table[f"{self.prefix}_is_valid"] = np.isfinite(stereo_predictions)
            stereo_table[f"{self.prefix}_goodness_of_fit"] = np.nan

        elif self.property is ReconstructionProperty.ENERGY:
            if np.count_nonzero(valid) > 0:
                mono_energies = mono_predictions[f"{prefix}_energy"].quantity.to_value(
                    u.TeV
                )
                if self.log_target:
                    mono_energies = np.log(mono_energies)

                stereo_energy, std = weighted_mean_std_ufunc(
                    mono_energies,
                    valid,
                    tel_to_array_indices,
                    multiplicity,
                    weights=weights,
                )
                if self.log_target:
                    stereo_energy = np.exp(stereo_energy)
                    std = np.exp(std)
            else:
                stereo_energy = np.full(n_array_events, np.nan)
                std = np.full(n_array_events, np.nan)

            stereo_table[f"{self.prefix}_energy"] = u.Quantity(
                stereo_energy, u.TeV, copy=COPY_IF_NEEDED
            )

            stereo_table[f"{self.prefix}_energy_uncert"] = u.Quantity(
                std, u.TeV, copy=COPY_IF_NEEDED
            )
            stereo_table[f"{self.prefix}_is_valid"] = np.isfinite(stereo_energy)
            stereo_table[f"{self.prefix}_goodness_of_fit"] = np.nan

        elif self.property is ReconstructionProperty.GEOMETRY:
            if np.count_nonzero(valid) > 0:
                coord = AltAz(
                    alt=mono_predictions[f"{prefix}_alt"],
                    az=mono_predictions[f"{prefix}_az"],
                )
                # https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution#Mean_direction
                mono_x, mono_y, mono_z = coord.cartesian.get_xyz()

                stereo_x, _ = weighted_mean_std_ufunc(
                    mono_x,
                    valid,
                    tel_to_array_indices,
                    multiplicity,
                    weights=weights,
                )
                stereo_y, _ = weighted_mean_std_ufunc(
                    mono_y,
                    valid,
                    tel_to_array_indices,
                    multiplicity,
                    weights=weights,
                )
                stereo_z, _ = weighted_mean_std_ufunc(
                    mono_z,
                    valid,
                    tel_to_array_indices,
                    multiplicity,
                    weights=weights,
                )

                mean_cartesian = CartesianRepresentation(
                    x=stereo_x, y=stereo_y, z=stereo_z
                )
                mean_spherical = mean_cartesian.represent_as(SphericalRepresentation)
                mean_altaz = AltAz(mean_spherical)

                # https://en.wikipedia.org/wiki/Directional_statistics#Measures_of_location_and_spread
                r = mean_spherical.distance.to_value()
                std = np.sqrt(-2 * np.log(r))
            else:
                mean_altaz = AltAz(
                    alt=u.Quantity(
                        np.full(n_array_events, np.nan), u.deg, copy=COPY_IF_NEEDED
                    ),
                    az=u.Quantity(
                        np.full(n_array_events, np.nan), u.deg, copy=COPY_IF_NEEDED
                    ),
                )
                std = np.full(n_array_events, np.nan)

            stereo_table[f"{self.prefix}_alt"] = mean_altaz.alt.to(u.deg)
            stereo_table[f"{self.prefix}_az"] = mean_altaz.az.to(u.deg)

            stereo_table[f"{self.prefix}_ang_distance_uncert"] = u.Quantity(
                np.rad2deg(std), u.deg, copy=COPY_IF_NEEDED
            )

            stereo_table[f"{self.prefix}_is_valid"] = np.logical_and(
                np.isfinite(stereo_table[f"{self.prefix}_alt"]),
                np.isfinite(stereo_table[f"{self.prefix}_az"]),
            )
            stereo_table[f"{self.prefix}_goodness_of_fit"] = np.nan

        else:
            raise NotImplementedError()

        tel_ids = [[] for _ in range(n_array_events)]

        for index, tel_id in zip(
            tel_to_array_indices[valid], mono_predictions["tel_id"][valid]
        ):
            tel_ids[index].append(tel_id)

        stereo_table[f"{self.prefix}_telescopes"] = tel_ids
        add_defaults_and_meta(stereo_table, _containers[self.property], self.prefix)
        return stereo_table


class StereoDispCombiner(StereoCombiner):
    """
    Stereo combination algorithm for DISP-based direction reconstruction.

    This combiner is essentially an implementation of Algorithm 3
    of :cite:p:`hofmann-1999-comparison` and quite similar to the EventDisplay
    implementation where each telescope predicts two possible directions (SIGN = ±1).
    Especially at low energies, the DISP sign reconstruction can be quite uncertain.
    To solve this head-tail ambiguity this algorithm does the following for all valid
    telescopes of an array event:

    1. Two possible FoV positions (lon/lat) are computed from the Hillas centroid,
       the DISP parameter, and the Hillas orientation angle (Hillas psi).

    2. For every telescope pair, all four SIGN combinations are evaluated.
       The SIGN pair that minimizes the angular distance between the two predicted
       positions is selected, optionally weighted by the telescope-wise DISP
       sign score. A weighted mean for the minimum distance per telescope pair
       is calculated.

    3. A second weighted mean FoV direction of all telescope-pair minima is computed
       afterwards and transformed into horizontal (Alt/Az) coordinates.

    See :class:`StereoCombiner` for a description of the general interface.

    Notes
    -----
    - Only geometry (:class:`~ctapipe.reco.ReconstructionProperty.GEOMETRY`)
      is supported.
    - Weighting options follow the same convention as in :class:`StereoMeanCombiner`
      (none, intensity, konrad).
    - The DISP sign-score can optionally be used to prefer SIGN combinations
      with higher reliability when resolving the DISP head–tail ambiguity.
    """

    weights = CaselessStrEnum(
        ["none", "intensity", "konrad"],
        default_value="none",
        help=(
            "What kind of weights to use. Options: ``none``, ``intensity``, ``konrad``."
        ),
    ).tag(config=True)

    sign_score_limit = Float(
        default_value=0.85,
        min=0,
        max=1.0,
        allow_none=True,
        help=(
            "Lower-limit for the telescope-wise sign scores to consider in the weighting "
            "of the distances per telescope combination. The value must be between 0 and "
            "and 1. Telescope events with a sign score above this limit are taken "
            "preferably into account for calculating the minimum distance per telescope "
            "combination. Set to None to disable this feature."
        ),
    ).tag(config=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.supported = ReconstructionProperty.GEOMETRY
        if self.property not in self.supported:
            raise NotImplementedError(
                f"Combination of {self.property} not implemented in {self.__class__.__name__}"
            )

        self.n_tel_combinations = 2

    def _calculate_weights(self, data):
        if isinstance(data, Container):
            if self.weights == "intensity":
                return data.hillas.intensity

            if self.weights == "konrad":
                return data.hillas.intensity * data.hillas.length / data.hillas.width

            return 1

        if isinstance(data, Table):
            if self.weights == "intensity":
                return data["hillas_intensity"]

            if self.weights == "konrad":
                return (
                    data["hillas_intensity"]
                    * data["hillas_length"]
                    / data["hillas_width"]
                )

            return np.ones(len(data))

        raise TypeError(
            "Dl1 data needs to be provided in the form of a container or astropy.table.Table"
        )

    def _combine_altaz(self, event):
        ids = []
        fov_lon_values = []
        fov_lat_values = []
        weights = []
        dist_weights = []

        signs = np.array([-1, 1])

        for tel_id, dl2 in event.dl2.tel.items():
            if dl2.geometry[self.prefix].is_valid:
                dl1 = event.dl1.tel[tel_id].parameters
                hillas_fov_lon = dl1.hillas.fov_lon.to_value(u.deg)
                hillas_fov_lat = dl1.hillas.fov_lat.to_value(u.deg)
                hillas_psi = dl1.hillas.psi
                disp = dl2.disp[self.prefix].parameter.to_value(u.deg)
                if np.isnan(disp):
                    raise RuntimeError(
                        f"No valid DISP reconstruction parameter found for "
                        f"prefix='{self.prefix}'). "
                        f"Make sure to apply the DispReconstructor before using the "
                        f"StereoDispCombiner or adapt the prefix accordingly."
                    )

                dist_weight = np.ones(2)
                if self.sign_score_limit is not None:
                    sign_score = dl2.disp[self.prefix].sign_score
                    if sign_score >= self.sign_score_limit:
                        dist_weight[np.sign(disp) == signs] = 1 / (1 + sign_score)
                dist_weights.append(dist_weight)

                abs_disp = np.abs(disp)
                fov_lons = hillas_fov_lon + signs * abs_disp * np.cos(hillas_psi)
                fov_lats = hillas_fov_lat + signs * abs_disp * np.sin(hillas_psi)
                fov_lon_values.append(fov_lons)
                fov_lat_values.append(fov_lats)
                weights.append(self._calculate_weights(dl1) if dl1 else 1)
                ids.append(tel_id)

        if len(fov_lon_values) > 1:
            index_tel_combs = get_combinations(len(ids), self.n_tel_combinations)
            fov_lons, fov_lats, comb_weights = calc_combs_min_distances_event(
                index_tel_combs,
                np.array(fov_lon_values),
                np.array(fov_lat_values),
                np.array(weights),
                np.array(dist_weights),
            )
            fov_lon_weighted_average = np.average(fov_lons, weights=comb_weights)
            fov_lat_weighted_average = np.average(fov_lats, weights=comb_weights)
            valid = True

        elif len(fov_lon_values) == 1:
            # single tel events
            fov_lon_weighted_average = hillas_fov_lon + disp * np.cos(hillas_psi)
            fov_lat_weighted_average = hillas_fov_lat + disp * np.sin(hillas_psi)
            valid = True

        else:
            alt = az = u.Quantity(np.nan, u.deg, copy=COPY_IF_NEEDED)
            valid = False

        if valid:
            alt, az = telescope_to_horizontal(
                lon=fov_lon_weighted_average * u.deg,
                lat=fov_lat_weighted_average * u.deg,
                pointing_alt=event.monitoring.pointing.array_altitude,
                pointing_az=event.monitoring.pointing.array_azimuth,
            )

        event.dl2.stereo.geometry[self.prefix] = ReconstructedGeometryContainer(
            alt=alt,
            az=az,
            telescopes=ids,
            is_valid=valid,
            prefix=self.prefix,
        )

    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Perform DISP-based stereo direction reconstruction for a single event.

        This is the entry point used for event-wise processing.
        Calls :meth:`_combine_altaz` and stores the resulting stereo geometry
        inside ``event.dl2.stereo`` under the configured prefix.
        """

        self._combine_altaz(event)

    def predict_table(self, mono_predictions: Table) -> Table:
        """
        Compute stereo DISP-based direction reconstruction for a full table
        of mono predictions.

        This is the table-wise / batch version of the DISP stereo combination.
        Each row in ``mono_predictions`` corresponds to a telescope-event.
        The function groups rows by array event (obs_id, event_id), performs
        DISP-based direction reconstruction for each array event, and returns
        one row per array event.

        See :meth:`StereoCombiner.predict_table` for the general
        input/output conventions.
        """

        prefix = f"{self.prefix}_tel"
        valid = mono_predictions[f"{prefix}_is_valid"]

        disp_col = f"{prefix}_parameter"
        if disp_col not in mono_predictions.colnames:
            raise KeyError(
                f"Required DISP column '{disp_col}' not found in mono prediction table. "
                f"Make sure the mono events were reconstructed with the corresponding "
                f"DispReconstructor (prefix='{self.prefix}') before running StereoDispCombiner."
            )

        obs_ids, event_ids, _, tel_to_array_indices = get_subarray_index(
            mono_predictions
        )
        _, _, valid_multiplicity, _ = get_subarray_index(mono_predictions[valid])

        n_array_events = len(obs_ids)
        stereo_table = Table({"obs_id": obs_ids, "event_id": event_ids})
        # copy metadata
        for colname in ("obs_id", "event_id"):
            stereo_table[colname].description = mono_predictions[colname].description

        weights = self._calculate_weights(mono_predictions[valid])

        if np.count_nonzero(valid) > 0:
            fov_lon_values, fov_lat_values, dist_weights = calc_fov_lon_lat(
                mono_predictions[valid],
                self.sign_score_limit,
                prefix,
            )
            combs_array, combs_to_multi_indices = create_combs_array(
                valid_multiplicity.max(), self.n_tel_combinations
            )
            index_tel_combs, n_combs = get_index_combs(
                valid_multiplicity,
                combs_array,
                combs_to_multi_indices,
                self.n_tel_combinations,
            )
            combs_to_array_indices = np.repeat(
                np.arange(len(valid_multiplicity)), n_combs
            )

            (
                comb_fov_lons,
                comb_fov_lats,
                comb_weights,
            ) = calc_combs_min_distances_table(
                index_tel_combs,
                fov_lon_values,
                fov_lat_values,
                weights,
                dist_weights,
            )

            # All calculated telescope combinations are valid here.
            all_valid = np.ones(len(comb_weights), dtype=bool)
            fov_lon_combs_mean, _ = weighted_mean_std_ufunc(
                comb_fov_lons,
                all_valid,
                combs_to_array_indices,
                n_combs,
                weights=comb_weights,
            )
            fov_lat_combs_mean, _ = weighted_mean_std_ufunc(
                comb_fov_lats,
                all_valid,
                combs_to_array_indices,
                n_combs,
                weights=comb_weights,
            )

            valid_tel_to_array_indices = tel_to_array_indices[valid]
            valid_array_indices = np.unique(valid_tel_to_array_indices)

            fov_lon_mean = np.full(n_array_events, np.nan)
            fov_lat_mean = np.full(n_array_events, np.nan)
            fov_lon_mean[valid_array_indices] = fov_lon_combs_mean
            fov_lat_mean[valid_array_indices] = fov_lat_combs_mean

            _, indices_first_tel_in_array = np.unique(
                tel_to_array_indices, return_index=True
            )
            alt, az = telescope_to_horizontal(
                lon=u.Quantity(fov_lon_mean, u.deg, copy=COPY_IF_NEEDED),
                lat=u.Quantity(fov_lat_mean, u.deg, copy=COPY_IF_NEEDED),
                pointing_alt=u.Quantity(
                    mono_predictions["subarray_pointing_lat"][
                        indices_first_tel_in_array
                    ],
                    u.deg,
                    copy=False,
                ),
                pointing_az=u.Quantity(
                    mono_predictions["subarray_pointing_lon"][
                        indices_first_tel_in_array
                    ],
                    u.deg,
                    copy=False,
                ),
            )

            # Fill single telescope events from mono_predictions
            index_single_tel_events = valid_array_indices[valid_multiplicity == 1]
            mask_single_tel_events = np.isin(
                valid_tel_to_array_indices, index_single_tel_events
            )
            alt[index_single_tel_events] = mono_predictions[f"{prefix}_alt"][valid][
                mask_single_tel_events
            ]
            az[index_single_tel_events] = mono_predictions[f"{prefix}_az"][valid][
                mask_single_tel_events
            ]

        else:
            alt = az = u.Quantity(
                np.full(n_array_events, np.nan), u.deg, copy=COPY_IF_NEEDED
            )
        stereo_table[f"{self.prefix}_alt"] = alt
        stereo_table[f"{self.prefix}_az"] = az
        stereo_table[f"{self.prefix}_is_valid"] = np.logical_and(
            np.isfinite(stereo_table[f"{self.prefix}_alt"]),
            np.isfinite(stereo_table[f"{self.prefix}_az"]),
        )
        stereo_table[f"{self.prefix}_goodness_of_fit"] = np.nan

        tel_ids = [[] for _ in range(n_array_events)]

        for index, tel_id in zip(
            tel_to_array_indices[valid], mono_predictions["tel_id"][valid]
        ):
            tel_ids[index].append(tel_id)

        stereo_table[f"{self.prefix}_telescopes"] = tel_ids
        add_defaults_and_meta(stereo_table, _containers[self.property], self.prefix)
        return stereo_table
