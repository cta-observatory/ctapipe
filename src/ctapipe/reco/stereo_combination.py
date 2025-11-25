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
    Base Class for algorithms combining telescope-wise predictions to common prediction.
    """

    prefix = Unicode(
        default_value="",
        help="Prefix to be added to the output container / column names.",
    ).tag(config=True)

    property = UseEnum(
        ReconstructionProperty,
        help="Which property is being combined.",
    ).tag(config=True)

    @abstractmethod
    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Fill event container with stereo predictions.
        """

    @abstractmethod
    def predict_table(self, mono_predictions: Table) -> Table:
        """
        Constructs stereo predictions from a table of
        telescope events.
        """


class StereoMeanCombiner(StereoCombiner):
    """
    Calculate array-event prediction as (weighted) mean of telescope-wise predictions.
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
        Calculate the mean prediction for a single array event.
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
        Telescope events, that are nan, get discarded.
        This means you might end up with less events if
        all telescope predictions of a shower are invalid.
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
    """ """

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
                disp = dl2.disp[self.prefix].parameter.value

                dist_weight = np.ones(2)
                if self.sign_score_limit is not None:
                    sign_score = dl2.disp[self.prefix].sign_score
                    if sign_score >= self.sign_score_limit:
                        dist_weight[np.sign(disp) == signs] = 1 / (1 + sign_score)
                dist_weights.append(dist_weight)

                fov_lons = hillas_fov_lon + signs * np.abs(disp) * np.cos(hillas_psi)
                fov_lats = hillas_fov_lat + signs * np.abs(disp) * np.sin(hillas_psi)
                fov_lon_values.append(fov_lons)
                fov_lat_values.append(fov_lats)
                weights.append(self._calculate_weights(dl1) if dl1 else 1)
                ids.append(tel_id)

        if len(fov_lon_values) > 0:
            index_tel_combs = get_combinations(range(len(ids)), self.n_tel_combinations)
            fov_lons, fov_lats, comb_weights = calc_combs_min_distances_event(
                index_tel_combs,
                np.array(fov_lon_values),
                np.array(fov_lat_values),
                np.array(weights),
                np.array(dist_weights),
            )
            fov_lon_weighted_average = np.average(fov_lons, weights=comb_weights)
            fov_lat_weighted_average = np.average(fov_lats, weights=comb_weights)
            alt, az = telescope_to_horizontal(
                lon=fov_lon_weighted_average * u.deg,
                lat=fov_lat_weighted_average * u.deg,
                pointing_alt=event.monitoring.pointing.array_altitude,
                pointing_az=event.monitoring.pointing.array_azimuth,
            )
            valid = True
        else:
            alt = az = u.Quantity(np.nan, u.deg, copy=COPY_IF_NEEDED)
            valid = False

        event.dl2.stereo.geometry[self.prefix] = ReconstructedGeometryContainer(
            alt=alt,
            az=az,
            telescopes=ids,
            is_valid=valid,
            prefix=self.prefix,
        )

    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Calculate the mean prediction for a single array event.
        """

        self._combine_altaz(event)

    def predict_table(self, mono_predictions: Table) -> Table:
        """ """
        prefix = f"{self.prefix}_tel"
        valid = mono_predictions[f"{prefix}_is_valid"]

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
