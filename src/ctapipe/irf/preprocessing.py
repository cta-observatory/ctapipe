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

from ..compat import COPY_IF_NEEDED
from ..containers import CoordinateFrameType
from ..coordinates import NominalFrame
from ..core import Component, QualityQuery
from ..core.traits import Dict, List, Tuple, Unicode
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

    energy_reconstructor = Unicode("RandomForestRegressor").tag(config=True)
    geometry_reconstructor = Unicode("HillasReconstructor").tag(config=True)
    gammaness_classifier = Unicode("RandomForestClassifier").tag(config=True)

    fixed_columns = List(
        Unicode(),
        default_value=["obs_id", "event_id", "true_energy", "true_az", "true_alt"],
        help="Columns to always keep from the original input",
    ).tag(config=True)

    # Optional user override
    columns_to_rename_override = Dict(
        key_trait=Unicode(),
        value_trait=Unicode(),
        default_value={},
        help="Override of columns to rename. If empty, they are generated dynamically.",
    ).tag(config=True)

    output_table_schema = List(
        default_value=[
            Column(name="obs_id", dtype=np.uint64, description="Observation block ID"),
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
            Column(name="true_source_fov_offset", unit=u.deg),
            Column(name="reco_source_fov_offset", unit=u.deg),
            Column(name="gh_score", unit=u.dimensionless_unscaled),
            Column(name="weight", unit=u.dimensionless_unscaled),
        ],
        help="Schema definition for output event QTable",
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.quality_query = EventQualityQuery(parent=self)

    @property
    def columns_to_rename(self) -> dict:
        if self.columns_to_rename_override:
            return self.columns_to_rename_override

        return {
            f"{self.energy_reconstructor}_energy": "reco_energy",
            f"{self.geometry_reconstructor}_az": "reco_az",
            f"{self.geometry_reconstructor}_alt": "reco_alt",
            f"{self.gammaness_classifier}_prediction": "gh_score",
            "subarray_pointing_lat": "pointing_alt",
            "subarray_pointing_lon": "pointing_az",
        }

    def normalise_column_names(self, events: QTable) -> QTable:
        if events["subarray_pointing_lat"].std() > 1e-3:
            raise NotImplementedError(
                "No support for making irfs from varying pointings yet"
            )
        if any(events["subarray_pointing_frame"] != CoordinateFrameType.ALTAZ.value):
            raise NotImplementedError(
                "At the moment only pointing in altaz is supported."
            )

        rename_from = list(self.columns_to_rename.keys())
        rename_to = list(self.columns_to_rename.values())

        keep_columns = self.fixed_columns + rename_from
        for col in keep_columns:
            if col not in events.colnames:
                raise ValueError(
                    f"Input files must conform to the ctapipe DL2 data model. "
                    f"Required column {col} is missing."
                )

        events = QTable(events[keep_columns], copy=COPY_IF_NEEDED)
        events.rename_columns(rename_from, rename_to)
        return events

    def make_empty_table(self) -> QTable:
        return QTable(self.output_table_schema)


class EventLoader(Component):
    """
    Contains functions to load events and simulation information from a file
    and derive some additional columns needed for irf calculation.
    """

    classes = [EventPreprocessor]

    def __init__(self, file: Path, target_spectrum: Spectra, **kwargs):
        super().__init__(**kwargs)

        self.epp = EventPreprocessor(parent=self)
        self.target_spectrum = SPECTRA[target_spectrum]
        self.file = file

    def load_preselected_events(
        self, chunk_size: int, obs_time: u.Quantity
    ) -> tuple[QTable, int, dict]:
        opts = dict(dl2=True, simulated=True, observation_info=True)
        with TableLoader(self.file, parent=self, **opts) as load:
            header = self.epp.make_empty_table()
            print("!!!!!HEADER!!!!!!", header)
            sim_info, spectrum = self.get_simulation_information(load, obs_time)
            meta = {"sim_info": sim_info, "spectrum": spectrum}
            bits = [header]
            n_raw_events = 0
            for _, _, events in load.read_subarray_events_chunked(chunk_size, **opts):
                selected = events[self.epp.quality_query.get_table_mask(events)]
                # print("selected", selected)
                selected = self.epp.normalise_column_names(selected)
                # print("selected", selected)
                selected = self.make_derived_columns(selected)
                # print("selected", selected)
                bits.append(selected)
                n_raw_events += len(events)

            bits.append(header)  # Putting it last ensures the correct metadata is used
            table = vstack(bits, join_type="exact", metadata_conflicts="silent")
            return table, n_raw_events, meta

    def get_simulation_information(
        self, loader: TableLoader, obs_time: u.Quantity
    ) -> tuple[SimulatedEventsInfo, PowerLaw]:
        sim = loader.read_simulation_configuration()
        try:
            show = loader.read_shower_distribution()
        except NoSuchNodeError:
            # Fall back to using the run header
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

        altaz = AltAz()
        pointing = SkyCoord(
            alt=events["pointing_alt"], az=events["pointing_az"], frame=altaz
        )
        reco = SkyCoord(
            alt=events["reco_alt"],
            az=events["reco_az"],
            frame=altaz,
        )
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
