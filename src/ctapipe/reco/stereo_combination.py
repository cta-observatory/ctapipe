from abc import abstractmethod

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, CartesianRepresentation, SphericalRepresentation
from astropy.table import Table
from traitlets import UseEnum

from ctapipe.core import Component, Container
from ctapipe.core.traits import Bool, CaselessStrEnum, Unicode
from ctapipe.reco.reconstructor import ReconstructionProperty

from ..compat import COPY_IF_NEEDED
from ..containers import (
    ArrayEventContainer,
    ParticleClassificationContainer,
    ReconstructedEnergyContainer,
    ReconstructedGeometryContainer,
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
]


def _grouped_add(tel_data, n_array_events, indices):
    """
    Calculate the group-wise sum for each array event over the
    corresponding telescope events. ``indices`` is an array
    that gives the index of the subarray event for each telescope event,
    returned by
    ``np.unique(tel_events[["obs_id", "event_id"]], return_inverse=True)``
    """
    combined_values = np.zeros(n_array_events)
    np.add.at(combined_values, indices, tel_data)
    return combined_values


def _weighted_mean_ufunc(tel_values, weights, n_array_events, indices):
    # avoid numerical problems by very large or small weights
    weights = weights / weights.max()
    sum_prediction = _grouped_add(
        tel_values * weights,
        n_array_events,
        indices,
    )
    sum_of_weights = _grouped_add(
        weights,
        n_array_events,
        indices,
    )
    mean = np.full(n_array_events, np.nan)
    valid = sum_of_weights > 0
    mean[valid] = sum_prediction[valid] / sum_of_weights[valid]
    return mean


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
        ["none", "intensity", "konrad"],
        default_value="none",
        help=(
            "What kind of weights to use."
            " Options: ``none``, ``intensity``, ``konrad``."
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
            mono = dl2.classification[self.prefix]
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
        event.dl2.stereo.classification[self.prefix] = container

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
        valid_predictions = mono_predictions[valid]

        array_events, indices, multiplicity = np.unique(
            mono_predictions[["obs_id", "event_id"]],
            return_inverse=True,
            return_counts=True,
        )
        stereo_table = Table(array_events)
        # copy metadata
        for colname in ("obs_id", "event_id"):
            stereo_table[colname].description = mono_predictions[colname].description

        n_array_events = len(array_events)
        weights = self._calculate_weights(valid_predictions)

        if self.property is ReconstructionProperty.PARTICLE_TYPE:
            if len(valid_predictions) > 0:
                mono_predictions = valid_predictions[f"{prefix}_prediction"]
                stereo_predictions = _weighted_mean_ufunc(
                    mono_predictions, weights, n_array_events, indices[valid]
                )
            else:
                stereo_predictions = np.full(n_array_events, np.nan)

            stereo_table[f"{self.prefix}_prediction"] = stereo_predictions
            stereo_table[f"{self.prefix}_is_valid"] = np.isfinite(stereo_predictions)
            stereo_table[f"{self.prefix}_goodness_of_fit"] = np.nan

        elif self.property is ReconstructionProperty.ENERGY:
            if len(valid_predictions) > 0:
                mono_energies = valid_predictions[f"{prefix}_energy"].quantity.to_value(
                    u.TeV
                )

                if self.log_target:
                    mono_energies = np.log(mono_energies)

                stereo_energy = _weighted_mean_ufunc(
                    mono_energies,
                    weights,
                    n_array_events,
                    indices[valid],
                )
                variance = _weighted_mean_ufunc(
                    (mono_energies - np.repeat(stereo_energy, multiplicity)[valid])
                    ** 2,
                    weights,
                    n_array_events,
                    indices[valid],
                )
                std = np.sqrt(variance)

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
            if len(valid_predictions) > 0:
                coord = AltAz(
                    alt=valid_predictions[f"{prefix}_alt"],
                    az=valid_predictions[f"{prefix}_az"],
                )
                # https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution#Mean_direction
                mono_x, mono_y, mono_z = coord.cartesian.get_xyz()

                stereo_x = _weighted_mean_ufunc(
                    mono_x, weights, n_array_events, indices[valid]
                )
                stereo_y = _weighted_mean_ufunc(
                    mono_y, weights, n_array_events, indices[valid]
                )
                stereo_z = _weighted_mean_ufunc(
                    mono_z, weights, n_array_events, indices[valid]
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

        for index, tel_id in zip(indices[valid], valid_predictions["tel_id"]):
            tel_ids[index].append(tel_id)

        stereo_table[f"{self.prefix}_telescopes"] = tel_ids
        add_defaults_and_meta(stereo_table, _containers[self.property], self.prefix)
        return stereo_table
