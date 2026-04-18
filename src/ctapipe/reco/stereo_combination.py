from abc import abstractmethod

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, CartesianRepresentation, SphericalRepresentation
from astropy.table import Table
from traitlets import UseEnum

from ctapipe.containers import ImageParametersContainer
from ctapipe.core import Component
from ctapipe.core.traits import (
    Bool,
    CaselessStrEnum,
    Float,
    Integer,
    TraitError,
    Unicode,
)
from ctapipe.image.statistics import arg_n_largest
from ctapipe.reco.reconstructor import ReconstructionProperty

from ..compat import COPY_IF_NEEDED
from ..containers import (
    ArrayEventContainer,
    ParticleClassificationContainer,
    ReconstructedEnergyContainer,
    ReconstructedGeometryContainer,
)
from .preprocessing import telescope_to_horizontal
from .reconstructor import StereoQualityQuery
from .telescope_event_handling import (
    calc_combs_min_distances,
    calc_fov_lon_lat,
    check_ang_diff,
    create_combs_array,
    fill_lower_multiplicities,
    get_combinations,
    get_index_combs,
    get_subarray_index,
    valid_tels_of_multi,
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
    Base class for algorithms combining telescope-wise predictions
    to common predictions.

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
       and constructs an output table with one row per subarray event containing
       the stereo predictions.

    Subclasses must implement both methods, as well as any additional logic
    required for combining the chosen :class:`~ctapipe.reco.ReconstructionProperty`.
    """

    classes = [StereoQualityQuery]

    weights = CaselessStrEnum(
        ["none", "intensity", "aspect-weighted-intensity"],
        default_value="none",
        help=(
            "Weighting scheme used to combine telescope-wise mono predictions:\n\n"
            "- ``none``: All telescopes contribute equally (unweighted mean).\n"
            "- ``intensity``: Contributions are weighted by the image intensity, "
            "giving higher weight to brighter images.\n"
            "- ``aspect-weighted-intensity``: Contributions are weighted by the "
            "image intensity multiplied by the image aspect ratio (length/width), "
            "favoring bright and well-elongated images.\n\n"
            "The selected weighting is applied consistently to all supported "
            "reconstruction properties (e.g. geometry, energy, particle type)."
        ),
    ).tag(config=True)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.quality_query = StereoQualityQuery(parent=self)

    def _calculate_weights(self, data):
        if isinstance(data, ImageParametersContainer):
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
            "Dl1 data needs to be provided in the form of an ImageParametersContainer "
            "or astropy.table.Table."
        )

    @abstractmethod
    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Compute the stereo prediction for a single subarray event.

        Parameters
        ----------
        event : ArrayEventContainer
            Subarray event containing DL1 parameters and per-telescope DL2 predictions.
            Implementations must write the combined stereo prediction into
            ``event.dl2.stereo`` under the configured ``prefix``.

        Notes
        -----
        - This method modifies the subarray event container *in place*.
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
            A table with one row per subarray event, containing the combined
            stereo predictions.
        """


class StereoMeanCombiner(StereoCombiner):
    """
    Combine telescope-wise mono reconstructions using a (weighted) mean.

    This implementation supports combining energy, geometry, and
    particle-type predictions. The weighting scheme applied to the
    telescope-wise inputs is controlled via the ``weights`` configuration
    trait.

    If ``log_target=True`` and ``ENERGY`` is combined, the combiner computes
    the geometric mean by averaging the logarithm of the energies.

    See :class:`StereoCombiner` for a description of the general interface.
    """

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
                f"Combination of {self.property} not implemented in "
                "{self.__class__.__name__}."
            )

    def _combine_energy(self, event):
        ids = []
        values = []
        weights = []

        for tel_id, dl2 in event.dl2.tel.items():
            mono = dl2.energy[self.prefix]
            dl1_params = event.dl1.tel[tel_id].parameters
            quality_checks = self.quality_query(parameters=dl1_params)
            if not mono.is_valid:
                continue
            if not all(quality_checks):
                continue

            values.append(mono.energy.to_value(u.TeV))
            if tel_id not in event.dl1.tel:
                raise ValueError("No parameters for weighting available")
            weights.append(self._calculate_weights(event.dl1.tel[tel_id].parameters))
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
            dl1_params = event.dl1.tel[tel_id].parameters
            quality_checks = self.quality_query(parameters=dl1_params)
            if not mono.is_valid:
                continue
            if not all(quality_checks):
                continue

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
            dl1_params = event.dl1.tel[tel_id].parameters
            quality_checks = self.quality_query(parameters=dl1_params)
            if not mono.is_valid:
                continue
            if not all(quality_checks):
                continue

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
        Calculates the (subarray-)event-wise mean.

        Telescope events, that are nan, get discarded. This means
        you might end up with less events if all telescope predictions
        of a shower are invalid.

        See :meth:`StereoCombiner.predict_table` for the general
        input/output conventions.
        """

        prefix = f"{self.prefix}_tel"
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

    This combiner implements a generalized DISP stereo reconstruction based on
    Algorithm 3 of :cite:p:`hofmann-1999-comparison` and the EventDisplay
    approach, where each telescope yields two possible directions
    (DISP sign = ±1). The algorithm resolves the head–tail ambiguity by
    combining multiple telescope combinations and applying a weighted mean.

    **Algorithm overview**

    0. **Pre-selection of valid telescopes (per subarray event)**
       The set of telescopes participating in the stereo reconstruction is
       optionally restricted before any combination step:

       - If ``n_best_tels`` is set, only the ``n_best_tels`` telescopes with
         the highest weights are kept per subarray event; all other telescopes
         are excluded from further processing.
       - If ``min_ang_diff`` is set, subarray events of multiplicity = 2 that
         have a nearly parallel main shower axis (angular difference below
         ``min_ang_diff``) are rejected entirely.

       These selections reduce the effective telescope multiplicity used in
       the subsequent steps.

    1. For every remaining valid telescope, compute the two possible FoV
       positions (longitude and latitude) from the Hillas centroid, the
       reconstructed DISP distance, and the Hillas orientation angle
       (``psi``), corresponding to DISP sign = ±1.

    2. For each telescope combination of size ``n_tel_combinations``, evaluate
       all possible DISP sign assignments and select the sign combination that
       minimizes the Sum of Squared Errors (SSE) between the participating
       telescopes.

    3. Combine the resulting per-combination FoV positions using the selected
       telescope weights and compute a weighted mean FoV direction for each
       subarray event.

    4. Convert the final FoV direction to horizontal coordinates (Alt/Az).

    **Handling of lower multiplicities**

    If an subarray event has an effective telescope multiplicity smaller than
    ``n_tel_combinations`` (after applying ``n_best_tels`` and any angular
    difference cuts), but at least two telescopes remain, the reconstruction
    is performed using all available telescopes of that subevent. In this case,
    just one combination is formed with a size equal to the event multiplicity
    and the optimal DISP sign assignment is determined accordingly. Single-
    telescope events are handled separately using the mono reconstruction.

    **Traits**

    - ``n_tel_combinations``: Size of each telescope combination (minimum 2).
      Larger values increase computation time. For events with lower effective
      multiplicity, all available telescopes are used.
    - ``n_best_tels``: If set, restricts the reconstruction to the
      ``n_best_tels`` highest-weight telescopes per event.
    - ``min_ang_diff``: For multiplicity-2 events only, reject events with
      nearly parallel main shower axes (difference below this angle).
    - ``weights``: Telescope weights used in the per-combination and final
      mean (``none``, ``intensity``, ``aspect-weighted-intensity``).

    Notes
    -----
    - Only geometry (:class:`~ctapipe.reco.ReconstructionProperty.GEOMETRY`)
      is supported.
    - To reproduce the behavior of EventDisplay, set both
      ``n_tel_combinations`` and ``n_best_tels`` to 5.
    - Choosing large values of ``n_tel_combinations`` together with an
      unrestricted ``n_best_tels`` can lead to a rapid increase in the number
      of telescope combinations and thus to a significantly increased
      processing time.
    """

    n_tel_combinations = Integer(
        default_value=2,
        min=2,
        help=(
            "Number of telescopes used per combination (minimum 2). Values "
            "above 5 lead to significantly increased computation time."
        ),
    ).tag(config=True)

    n_best_tels = Integer(
        default_value=None,
        min=2,
        allow_none=True,
        help=(
            "Select the best n_best_tels telescopes by weight for the combination. "
            "Set to None to use all valid telescopes."
        ),
    ).tag(config=True)

    min_ang_diff = Float(
        default_value=None,
        min=0.0,
        allow_none=True,
        help=(
            "Minimum angular separation (in degrees) between the reconstructed main "
            "shower axes of two telescopes. This cut is applied only to subarray events "
            "with exactly two participating telescopes (multiplicity = 2). Events for "
            "which the angular difference between the two shower axes is smaller than "
            "this value are rejected, as nearly parallel axes lead to problems solving "
            "the head-tail ambiguity. Set to None to disable this cut."
        ),
    ).tag(config=True)

    property = UseEnum(
        ReconstructionProperty,
        default_value=ReconstructionProperty.GEOMETRY,
        help=(
            "Reconstruction property produced by this combiner (not configurable - "
            "fixed to GEOMETRY)."
        ),
    ).tag(config=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.n_best_tels is not None and self.n_tel_combinations > self.n_best_tels:
            raise TraitError(
                "n_tel_combinations must be less than or equal to n_best_tels."
            )
        self.supported = ReconstructionProperty.GEOMETRY
        if self.property not in self.supported:
            raise NotImplementedError(
                f"Combination of {self.property} not implemented in {self.__class__.__name__}"
            )

    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Perform DISP-based stereo direction reconstruction for a single subarray
        event.

        This is the entry point used for event-wise processing and stores the
        resulting stereo geometry inside ``event.dl2.stereo`` under the
        configured prefix.
        """

        ids, hillas_psis, fov_lon_values, fov_lat_values, weights = (
            self._collect_valid_tel_data(event)
        )

        multiplicity = len(ids)
        alt = az = u.Quantity(np.nan, u.deg, copy=COPY_IF_NEEDED)

        stereo_fov_lon, stereo_fov_lat, valid = self._compute_stereo_fov(
            multiplicity=multiplicity,
            hillas_psis=hillas_psis,
            fov_lon_values=fov_lon_values,
            fov_lat_values=fov_lat_values,
            weights=weights,
        )

        if valid:
            alt, az = telescope_to_horizontal(
                lon=stereo_fov_lon * u.deg,
                lat=stereo_fov_lat * u.deg,
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

    def _collect_valid_tel_data(self, event: ArrayEventContainer):
        ids = []
        hillas_psis = []
        fov_lon_values = []
        fov_lat_values = []
        weights = []

        signs = np.array([-1, 1])

        for tel_id, dl2 in event.dl2.tel.items():
            if not dl2.geometry[self.prefix].is_valid:
                continue

            disp_reco = dl2.disp.get(self.prefix)
            if disp_reco is None:
                raise RuntimeError(
                    "No valid DISP reconstruction parameter found for "
                    f"prefix='{self.prefix}'). "
                    "Make sure to apply the DispReconstructor before using the "
                    "StereoDispCombiner or adapt the prefix accordingly."
                )

            dl1 = event.dl1.tel[tel_id].parameters
            if not all(self.quality_query(parameters=dl1)):
                continue

            hillas_fov_lon = dl1.hillas.fov_lon.to_value(u.deg)
            hillas_fov_lat = dl1.hillas.fov_lat.to_value(u.deg)
            hillas_psi = dl1.hillas.psi
            disp = disp_reco.parameter.to_value(u.deg)

            fov_lons = hillas_fov_lon + signs * disp * np.cos(hillas_psi)
            fov_lats = hillas_fov_lat + signs * disp * np.sin(hillas_psi)

            fov_lon_values.append(fov_lons)
            fov_lat_values.append(fov_lats)
            weights.append(self._calculate_weights(dl1) if dl1 else 1)
            ids.append(tel_id)
            hillas_psis.append(hillas_psi)

        return ids, hillas_psis, fov_lon_values, fov_lat_values, weights

    def _compute_stereo_fov(
        self,
        multiplicity: int,
        hillas_psis,
        fov_lon_values,
        fov_lat_values,
        weights,
    ):
        if multiplicity == 0:
            return np.nan, np.nan, False

        if multiplicity == 1:
            stereo_fov_lon = fov_lon_values[0][1]
            stereo_fov_lat = fov_lat_values[0][1]
            return stereo_fov_lon, stereo_fov_lat, True

        if (
            multiplicity == 2
            and self.min_ang_diff is not None
            and not check_ang_diff(self.min_ang_diff, hillas_psis[0], hillas_psis[1])
        ):
            return np.nan, np.nan, False

        n_tel_combs = min(self.n_tel_combinations, multiplicity)

        best_tels_idx = np.arange(multiplicity)
        if self.n_best_tels is not None and multiplicity > self.n_best_tels:
            best_tels_idx = arg_n_largest(self.n_best_tels, np.array(weights))

        index_tel_combs = get_combinations(len(best_tels_idx), n_tel_combs)
        fov_lons, fov_lats, comb_weights = calc_combs_min_distances(
            index_tel_combs,
            np.array(fov_lon_values)[best_tels_idx],
            np.array(fov_lat_values)[best_tels_idx],
            np.array(weights)[best_tels_idx],
        )

        stereo_fov_lon = np.average(fov_lons, weights=comb_weights)
        stereo_fov_lat = np.average(fov_lats, weights=comb_weights)
        return stereo_fov_lon, stereo_fov_lat, True

    def predict_table(self, mono_predictions: Table) -> Table:
        """
        Compute stereo DISP-based direction reconstruction for a full table
        of mono predictions.

        This is the table-wise / batch version of the DISP stereo combination.
        Each row in ``mono_predictions`` corresponds to a telescope-event.
        The function groups rows by subarray event (obs_id, event_id), performs
        DISP-based direction reconstruction for each subarray event, and returns
        one row per subarray event.

        See :meth:`StereoCombiner.predict_table` for the general
        input/output conventions.
        """
        prefix_tel = f"{self.prefix}_tel"

        valid = mono_predictions[f"{prefix_tel}_is_valid"].copy()
        self._require_disp_column(mono_predictions, prefix_tel)

        obs_ids, event_ids, _, tel_to_array_indices = get_subarray_index(
            mono_predictions
        )
        n_array_events = len(obs_ids)

        valid = self._apply_min_ang_diff_cut(
            mono_predictions=mono_predictions,
            valid=valid,
            tel_to_array_indices=tel_to_array_indices,
        )
        valid = self._apply_n_best_tels_cut(
            mono_predictions=mono_predictions,
            valid=valid,
            tel_to_array_indices=tel_to_array_indices,
        )

        stereo_table = self._init_stereo_table(mono_predictions, obs_ids, event_ids)
        alt, az = self._compute_altaz_for_valid(
            mono_predictions=mono_predictions,
            valid=valid,
            tel_to_array_indices=tel_to_array_indices,
            n_array_events=n_array_events,
            prefix_tel=prefix_tel,
        )

        stereo_table[f"{self.prefix}_alt"] = alt
        stereo_table[f"{self.prefix}_az"] = az
        stereo_table[f"{self.prefix}_is_valid"] = np.logical_and(
            np.isfinite(stereo_table[f"{self.prefix}_alt"]),
            np.isfinite(stereo_table[f"{self.prefix}_az"]),
        )
        stereo_table[f"{self.prefix}_goodness_of_fit"] = np.full(
            len(stereo_table), np.nan
        )

        stereo_table[f"{self.prefix}_telescopes"] = self._collect_telescopes_per_event(
            mono_predictions=mono_predictions,
            valid=valid,
            tel_to_array_indices=tel_to_array_indices,
            n_array_events=n_array_events,
        )

        add_defaults_and_meta(stereo_table, _containers[self.property], self.prefix)
        return stereo_table

    def _require_disp_column(self, mono_predictions: Table, prefix_tel: str) -> None:
        disp_col = f"{prefix_tel}_parameter"
        if disp_col not in mono_predictions.colnames:
            raise KeyError(
                f"Required DISP column '{disp_col}' not found in mono prediction table. "
                f"Make sure the mono events were reconstructed with the corresponding "
                f"DispReconstructor (prefix='{self.prefix}') before running "
                f"StereoDispCombiner."
            )

    def _apply_min_ang_diff_cut(
        self,
        mono_predictions: Table,
        valid: np.ndarray,
        tel_to_array_indices: np.ndarray,
    ) -> np.ndarray:
        if self.min_ang_diff is None:
            return valid

        # Reject events with multiplicity 2 and nearly parallel main shower axes
        mask_multi2_tels = valid_tels_of_multi(2, tel_to_array_indices[valid])
        if not np.any(mask_multi2_tels):
            return valid

        valid_idx = np.flatnonzero(valid)
        pairs_in_valid = np.flatnonzero(mask_multi2_tels).reshape(-1, 2)
        valid_psis = mono_predictions["hillas_psi"][valid]

        keep_pairs = check_ang_diff(
            self.min_ang_diff,
            valid_psis[pairs_in_valid[:, 0]],
            valid_psis[pairs_in_valid[:, 1]],
        )
        if np.all(keep_pairs):
            return valid

        tels_to_invalidate = valid_idx[pairs_in_valid[~keep_pairs].ravel()]
        valid[tels_to_invalidate] = False
        return valid

    def _apply_n_best_tels_cut(
        self,
        *,
        mono_predictions: Table,
        valid: np.ndarray,
        tel_to_array_indices: np.ndarray,
    ) -> np.ndarray:
        if self.n_best_tels is None:
            return valid

        weights = self._calculate_weights(mono_predictions[valid])
        valid_idx = np.flatnonzero(valid)
        valid_tel_to_array_indices = tel_to_array_indices[valid]

        if valid_tel_to_array_indices.size == 0:
            return valid

        order = np.lexsort((-np.asarray(weights), valid_tel_to_array_indices))

        starts = np.empty(valid_tel_to_array_indices.size, dtype=bool)
        starts[0] = True
        starts[1:] = valid_tel_to_array_indices[1:] != valid_tel_to_array_indices[:-1]

        start_pos = np.where(starts, np.arange(valid_tel_to_array_indices.size), 0)
        group_start = np.maximum.accumulate(start_pos)
        rank = np.arange(valid_tel_to_array_indices.size) - group_start

        drop_n = rank >= self.n_best_tels
        if not np.any(drop_n):
            return valid

        valid[valid_idx[order[drop_n]]] = False
        return valid

    def _init_stereo_table(
        self, mono_predictions: Table, obs_ids: np.ndarray, event_ids: np.ndarray
    ) -> Table:
        stereo_table = Table({"obs_id": obs_ids, "event_id": event_ids})
        for colname in ("obs_id", "event_id"):
            stereo_table[colname].description = mono_predictions[colname].description
        return stereo_table

    def _compute_altaz_for_valid(
        self,
        mono_predictions: Table,
        valid: np.ndarray,
        tel_to_array_indices: np.ndarray,
        n_array_events: int,
        prefix_tel: str,
    ):
        if np.count_nonzero(valid) == 0:
            nan = u.Quantity(
                np.full(n_array_events, np.nan), u.deg, copy=COPY_IF_NEEDED
            )
            return nan, nan

        weights = self._calculate_weights(mono_predictions[valid])
        _, _, valid_multiplicity, _ = get_subarray_index(mono_predictions[valid])

        fov_lon_values, fov_lat_values = calc_fov_lon_lat(
            mono_predictions[valid], prefix_tel
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
        combs_to_array_indices = np.repeat(np.arange(len(valid_multiplicity)), n_combs)

        comb_fov_lons, comb_fov_lats, comb_weights = calc_combs_min_distances(
            index_tel_combs,
            fov_lon_values,
            fov_lat_values,
            weights,
        )

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

        if self.n_tel_combinations >= 3:
            fill_lower_multiplicities(
                fov_lon_combs_mean,
                fov_lat_combs_mean,
                self.n_tel_combinations,
                valid_tel_to_array_indices,
                valid_multiplicity,
                fov_lon_values,
                fov_lat_values,
                weights,
            )

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
                mono_predictions["subarray_pointing_lat"][indices_first_tel_in_array],
                u.deg,
                copy=COPY_IF_NEEDED,
            ),
            pointing_az=u.Quantity(
                mono_predictions["subarray_pointing_lon"][indices_first_tel_in_array],
                u.deg,
                copy=COPY_IF_NEEDED,
            ),
        )

        # Fill single telescope events from mono_predictions
        index_single_tel_events = valid_array_indices[valid_multiplicity == 1]
        if index_single_tel_events.size > 0:
            mask_single_tel_events = valid_tels_of_multi(1, valid_tel_to_array_indices)
            alt[index_single_tel_events] = mono_predictions[f"{prefix_tel}_alt"][valid][
                mask_single_tel_events
            ]
            az[index_single_tel_events] = mono_predictions[f"{prefix_tel}_az"][valid][
                mask_single_tel_events
            ]

        return alt, az

    def _collect_telescopes_per_event(
        self,
        mono_predictions: Table,
        valid: np.ndarray,
        tel_to_array_indices: np.ndarray,
        n_array_events: int,
    ):
        tel_ids = [[] for _ in range(n_array_events)]
        for index, tel_id in zip(
            tel_to_array_indices[valid], mono_predictions["tel_id"][valid]
        ):
            tel_ids[index].append(tel_id)
        return tel_ids
