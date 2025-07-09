"""Module containing classes related to event loading and preprocessing"""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.table import Column, QTable, Table, vstack
from pyirf.simulations import SimulatedEventsInfo
from pyirf.spectral import (
    DIFFUSE_FLUX_UNIT,
    POINT_SOURCE_FLUX_UNIT,
    PowerLaw,
    calculate_event_weights,
)
from pyirf.utils import calculate_source_fov_offset, calculate_theta
from tables import NoSuchNodeError
from traitlets import default

from ..compat import COPY_IF_NEEDED
from ..containers import CoordinateFrameType
from ..coordinates import NominalFrame
from ..core import Component, QualityQuery
from ..core.traits import Bool, Dict, List, Tuple, Unicode
from ..io import TableLoader
from .spectra import SPECTRA, Spectra

__all__ = ["EventLoader", "EventPreprocessor", "EventQualityQuery"]


class EventQualityQuery(QualityQuery):
    """
    Event pre-selection quality criteria for IRF computation with different defaults.
    """

    quality_criteria = List(
        Tuple(Unicode(), Unicode()),
        default_value=[
            (
                "multiplicity 4",
                "np.count_nonzero(HillasReconstructor_telescopes,axis=1) >= 4",
            ),
            ("valid classifier", "RandomForestClassifier_is_valid"),
            ("valid geom reco", "HillasReconstructor_is_valid"),
            ("valid energy reco", "RandomForestRegressor_is_valid"),
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)


class EventPreprocessor(Component):
    """Defines pre-selection cuts and the necessary renaming of columns."""

    classes = [EventQualityQuery]

    energy_reconstructor = Unicode(
        default_value="RandomForestRegressor",
        help="Prefix of the reco `_energy` column",
    ).tag(config=True)

    geometry_reconstructor = Unicode(
        default_value="HillasReconstructor",
        help="Prefix of the `_alt` and `_az` reco geometry columns",
    ).tag(config=True)

    gammaness_classifier = Unicode(
        default_value="RandomForestClassifier",
        help="Prefix of the classifier `_prediction` column",
    ).tag(config=True)

    apply_derived_columns = Bool(
        default_value=True, help="Whether to compute derived columns"
    ).tag(config=True)

    apply_check_pointing = Bool(
        default_value=True, help="Accept only pointing in altaz and not divergent."
    ).tag(config=True)

    fixed_columns = List(
        Unicode(),
        default_value=["obs_id", "event_id", "true_energy", "true_az", "true_alt"],
        help="Columns to keep from the input table without renaming.",
    ).tag(config=True)

    columns_to_rename = Dict(
        key_trait=Unicode(),
        value_trait=Unicode(),
        help=(
            "Dictionary of columns to rename. "
            "Leave unset to apply default renaming. "
            "Set to an empty dictionary to disable renaming entirely. "
            "Set to a partial dictionary to override only some names."
        ),
    ).tag(config=True)

    output_table_schema = List(
        default_value=[
            Column(name="obs_id", dtype=np.uint64, description="Observation Block ID"),
            Column(name="event_id", dtype=np.uint64, description="Array event ID"),
            Column(name="true_energy", unit=u.TeV, description="Simulated energy"),
            Column(name="true_az", unit=u.deg, description="Simulated azimuth"),
            Column(name="true_alt", unit=u.deg, description="Simulated altitude"),
            Column(name="reco_energy", unit=u.TeV, description="Reconstructed energy"),
            Column(name="reco_az", unit=u.deg, description="Reconstructed azimuth"),
            Column(name="reco_alt", unit=u.deg, description="Reconstructed altitude"),
            Column(
                name="reco_fov_lat", unit=u.deg, description="Reconstructed FOV lat"
            ),
            Column(
                name="reco_fov_lon", unit=u.deg, description="Reconstructed FOV lon"
            ),
            Column(name="pointing_az", unit=u.deg, description="Pointing azimuth"),
            Column(name="pointing_alt", unit=u.deg, description="Pointing altitude"),
            Column(name="theta", unit=u.deg, description="Angular offset from source"),
            Column(
                name="true_source_fov_offset",
                unit=u.deg,
                description="Simulated angular offset from pointing direction",
            ),
            Column(
                name="reco_source_fov_offset",
                unit=u.deg,
                description="Reconstructed angular offset from pointing direction",
            ),
            Column(
                name="gh_score",
                unit=u.dimensionless_unscaled,
                description="prediction of the classifier, defined between [0,1],"
                " where values close to 1 mean that the positive class"
                " (e.g. gamma in gamma-ray analysis) is more likely",
            ),
            Column(
                name="weight", unit=u.dimensionless_unscaled, description="Event weight"
            ),
        ],
        help="Schema definition for output event QTable",
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.quality_query = EventQualityQuery(parent=self)

    @default("columns_to_rename")
    def _default_columns_to_rename(self):
        return {
            f"{self.energy_reconstructor}_energy": "reco_energy",
            f"{self.geometry_reconstructor}_az": "reco_az",
            f"{self.geometry_reconstructor}_alt": "reco_alt",
            f"{self.gammaness_classifier}_prediction": "gh_score",
            "subarray_pointing_lat": "pointing_alt",
            "subarray_pointing_lon": "pointing_az",
        }

    def normalise_column_names(self, events: QTable) -> QTable:
        """
        Rename column names according to configuration.

        Parameters
        ----------
        events : QTable
            Input event table.

        Returns
        -------
        QTable
            Table with selected and renamed columns.

        Raises
        ------
        NotImplementedError
            If pointing is not AltAz or varies too much.
        ValueError
            If required columns are missing.
        """
        if self.apply_check_pointing:
            if events["subarray_pointing_lat"].std() > 1e-3:
                raise NotImplementedError(
                    "No support for making irfs from varying pointings yet"
                )
            if any(
                events["subarray_pointing_frame"] != CoordinateFrameType.ALTAZ.value
            ):
                raise NotImplementedError(
                    "At the moment only pointing in altaz is supported."
                )

        rename_dict = self.columns_to_rename
        rename_from = list(rename_dict.keys())
        rename_to = list(rename_dict.values())

        keep_columns = self.fixed_columns + rename_from
        for col in keep_columns:
            if col not in events.colnames:
                raise ValueError(
                    f"Input files must conform to the ctapipe DL2 data model. "
                    f"Required column {col} is missing."
                )

        events = QTable(events[keep_columns], copy=COPY_IF_NEEDED)
        if rename_from and rename_to:
            events.rename_columns(rename_from, rename_to)
        return events

    def make_empty_table(self) -> QTable:
        """
        Create an empty event table based on the configured output schema.

        Returns
        -------
        QTable
            Empty event table with configured schema.
        """

        return QTable(self.output_table_schema)


class EventLoader(Component):
    """
    Component for loading events and simulation metadata, applying preselection and optional derived column logic.
    """

    classes = [EventPreprocessor]

    # User-selectable event reading function and kwargs
    event_reader_function = Unicode(
        default_value="read_subarray_events_chunked",
        help=(
            "Function of TableLoader used to read event chunks. "
            "E.g., 'read_subarray_events_chunked' or 'read_telescope_events_chunked'."
        ),
    ).tag(config=True)

    event_reader_kwargs = Dict(
        default_value={},
        help="Extra keyword arguments passed to the event reading function, e.g., {'path': '/dl2/event/telescope/Reconstructor'}",
    ).tag(config=True)

    def __init__(self, file: Path, target_spectrum: Spectra, **kwargs):
        super().__init__(**kwargs)
        self.epp = EventPreprocessor(parent=self)
        self.target_spectrum = SPECTRA[target_spectrum]
        self.file = file

    def load_preselected_events(
        self, chunk_size: int, obs_time: u.Quantity
    ) -> tuple[QTable, int, dict]:
        """
        Load and filter events from the file.

        Parameters
        ----------
        chunk_size : int
            Size of chunks to read from the file.
        obs_time : Quantity
            Observation time to scale weights.

        Returns
        -------
        table : QTable
            Filtered and processed event table.
        n_raw_events : int
            Number of events before selection.
        meta : dict
            Metadata dictionary with simulation info and input spectrum.
        """

        opts = dict(dl2=True, simulated=True, observation_info=True)

        with TableLoader(self.file, parent=self, **opts) as load:
            header = self.epp.make_empty_table()
            sim_info, spectrum = self.get_simulation_information(load, obs_time)
            meta = {"sim_info": sim_info, "spectrum": spectrum}
            bits = [header]
            n_raw_events = 0
            reader_func = getattr(load, self.event_reader_function)
            table_reader = reader_func(chunk_size, **opts, **self.event_reader_kwargs)
            for _, _, events in table_reader:
                selected = events[self.epp.quality_query.get_table_mask(events)]
                selected = self.epp.normalise_column_names(selected)
                if self.epp.apply_derived_columns:
                    selected = self.make_derived_columns(selected)
                bits.append(selected)
                n_raw_events += len(events)

            bits.append(header)  # Putting it last ensures the correct metadata is used
            table = vstack(bits, join_type="exact", metadata_conflicts="silent")
            return table, n_raw_events, meta

    def get_simulation_information(
        self, loader: TableLoader, obs_time: u.Quantity
    ) -> tuple[SimulatedEventsInfo, PowerLaw]:
        """
        Extract simulation information from the input file.

        Parameters
        ----------
        loader : TableLoader
            Loader object for reading from the input file.
        obs_time : Quantity
            Total observation time.

        Returns
        -------
        sim_info : SimulatedEventsInfo
            Metadata about the simulated events.
        spectrum : PowerLaw
            Power-law model derived from simulation configuration.

        Raises
        ------
        NotImplementedError
            If simulation parameters vary across runs.
        """
        sim = loader.read_simulation_configuration()
        try:
            show = loader.read_shower_distribution()
        except NoSuchNodeError:
            show = Table([sim["n_showers"]], names=["n_entries"], dtype=[np.int64])

        for itm in ["spectral_index", "energy_range_min", "energy_range_max"]:
            if len(np.unique(sim[itm])) > 1:
                raise NotImplementedError(
                    f"Unsupported: '{itm}' differs across simulation runs"
                )

        sim_info = SimulatedEventsInfo(
            n_showers=show["n_entries"].sum(),
            energy_min=sim["energy_range_min"].quantity[0],
            energy_max=sim["energy_range_max"].quantity[0],
            max_impact=sim["max_scatter_range"].quantity[0],
            spectral_index=sim["spectral_index"][0],
            viewcone_max=sim["max_viewcone_radius"].quantity[0],
            viewcone_min=sim["min_viewcone_radius"].quantity[0],
        )

        return sim_info, PowerLaw.from_simulation(sim_info, obstime=obs_time)

    def make_derived_columns(self, events: QTable) -> QTable:
        """
        Add derived quantities (e.g., theta, FOV offsets) to the table.

        Parameters
        ----------
        events : QTable
            Table containing normalized events.

        Returns
        -------
        QTable
            Table with added derived columns.
        """
        events["weight"] = (
            1.0 * u.dimensionless_unscaled
        )  # defer calculation of proper weights to later
        events["gh_score"].unit = u.dimensionless_unscaled
        events["theta"] = calculate_theta(
            events,
            assumed_source_az=events["true_az"],
            assumed_source_alt=events["true_alt"],
        )
        events["true_source_fov_offset"] = calculate_source_fov_offset(
            events, prefix="true"
        )
        events["reco_source_fov_offset"] = calculate_source_fov_offset(
            events, prefix="reco"
        )

        pointing = SkyCoord(
            alt=events["pointing_alt"], az=events["pointing_az"], frame=AltAz()
        )
        reco = SkyCoord(alt=events["reco_alt"], az=events["reco_az"], frame=AltAz())
        nominal = NominalFrame(origin=pointing)
        reco_nominal = reco.transform_to(nominal)
        events["reco_fov_lon"] = u.Quantity(-reco_nominal.fov_lon)  # minus for GADF
        events["reco_fov_lat"] = u.Quantity(reco_nominal.fov_lat)
        return events

    def make_event_weights(
        self,
        events: QTable,
        spectrum: PowerLaw,
        kind: str,
        fov_offset_bins: u.Quantity | None = None,
    ) -> QTable:
        """
        Compute event weights to match the target spectrum.

        Parameters
        ----------
        events : QTable
            Input events.
        spectrum : PowerLaw
            Spectrum from simulation.
        kind : str
            Type of events ("gammas", etc.).
        fov_offset_bins : Quantity, optional
            Offset bins for integrating the diffuse flux into point source bins.

        Returns
        -------
        QTable
            Table with updated weights.

        Raises
        ------
        ValueError
            If ``fov_offset_bins`` is required but not provided.
        """
        if (
            kind == "gammas"
            and self.target_spectrum.normalization.unit.is_equivalent(
                POINT_SOURCE_FLUX_UNIT
            )
            and spectrum.normalization.unit.is_equivalent(DIFFUSE_FLUX_UNIT)
        ):
            if fov_offset_bins is None:
                raise ValueError(
                    "gamma_target_spectrum is point-like, but no fov offset bins "
                    "for the integration of the simulated diffuse spectrum were given."
                )

            for low, high in zip(fov_offset_bins[:-1], fov_offset_bins[1:]):
                fov_mask = events["true_source_fov_offset"] >= low
                fov_mask &= events["true_source_fov_offset"] < high

                events["weight"][fov_mask] = calculate_event_weights(
                    events[fov_mask]["true_energy"],
                    target_spectrum=self.target_spectrum,
                    simulated_spectrum=spectrum.integrate_cone(low, high),
                )
        else:
            events["weight"] = calculate_event_weights(
                events["true_energy"],
                target_spectrum=self.target_spectrum,
                simulated_spectrum=spectrum,
            )

        return events
