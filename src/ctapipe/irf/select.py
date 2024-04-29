"""Module containing classes related to eveent preprocessing and selection"""
from enum import Enum

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.table import QTable, vstack
from pyirf.cuts import calculate_percentile_cut
from pyirf.simulations import SimulatedEventsInfo
from pyirf.spectral import (
    CRAB_HEGRA,
    IRFDOC_ELECTRON_SPECTRUM,
    IRFDOC_PROTON_SPECTRUM,
    PowerLaw,
    calculate_event_weights,
)
from pyirf.utils import calculate_source_fov_offset, calculate_theta

from ..coordinates import NominalFrame
from ..core import Component, QualityQuery
from ..core.traits import Float, Integer, List, Unicode
from ..io import TableLoader
from ..irf import FovOffsetBinning


class Spectra(Enum):
    CRAB_HEGRA = 1
    IRFDOC_ELECTRON_SPECTRUM = 2
    IRFDOC_PROTON_SPECTRUM = 3


SPECTRA = {
    Spectra.CRAB_HEGRA: CRAB_HEGRA,
    Spectra.IRFDOC_ELECTRON_SPECTRUM: IRFDOC_ELECTRON_SPECTRUM,
    Spectra.IRFDOC_PROTON_SPECTRUM: IRFDOC_PROTON_SPECTRUM,
}


class EventPreProcessor(QualityQuery):
    """Defines preselection cuts and the necessary renaming of columns"""

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

    quality_criteria = List(
        default_value=[
            ("multiplicity 4", "np.count_nonzero(tels_with_trigger,axis=1) >= 4"),
            ("valid classifier", "RandomForestClassifier_is_valid"),
            ("valid geom reco", "HillasReconstructor_is_valid"),
            ("valid energy reco", "RandomForestRegressor_is_valid"),
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)

    rename_columns = List(
        help="List containing translation pairs new and old column names"
        "used when processing input with names differing from the CTA prod5b format"
        "Ex: [('valid_geom','HillasReconstructor_is_valid')]",
        default_value=[],
    ).tag(config=True)

    def normalise_column_names(self, events):
        keep_columns = [
            "obs_id",
            "event_id",
            "true_energy",
            "true_az",
            "true_alt",
        ]
        rename_from = [
            f"{self.energy_reconstructor}_energy",
            f"{self.geometry_reconstructor}_az",
            f"{self.geometry_reconstructor}_alt",
            f"{self.gammaness_classifier}_prediction",
        ]
        rename_to = ["reco_energy", "reco_az", "reco_alt", "gh_score"]

        # We never enter the loop if rename_columns is empty
        for new, old in self.rename_columns:
            rename_from.append(old)
            rename_to.append(new)

        keep_columns.extend(rename_from)
        events = QTable(events[keep_columns], copy=False)
        events.rename_columns(rename_from, rename_to)
        return events

    def make_empty_table(self):
        """This function defines the columns later functions expect to be present in the event table"""
        columns = [
            "obs_id",
            "event_id",
            "true_energy",
            "true_az",
            "true_alt",
            "reco_energy",
            "reco_az",
            "reco_alt",
            "reco_fov_lat",
            "reco_fov_lon",
            "gh_score",
            "pointing_az",
            "pointing_alt",
            "theta",
            "true_source_fov_offset",
            "reco_source_fov_offset",
            "weight",
        ]
        units = {
            "true_energy": u.TeV,
            "true_az": u.deg,
            "true_alt": u.deg,
            "reco_energy": u.TeV,
            "reco_az": u.deg,
            "reco_alt": u.deg,
            "reco_fov_lat": u.deg,
            "reco_fov_lon": u.deg,
            "pointing_az": u.deg,
            "pointing_alt": u.deg,
            "theta": u.deg,
            "true_source_fov_offset": u.deg,
            "reco_source_fov_offset": u.deg,
        }
        descriptions = {
            "obs_id": "Observation Block ID",
            "event_id": "Array Event ID",
            "true_energy": "Simulated Energy",
            "true_az": "Simulated azimuth",
            "true_alt": "Simulated altitude",
            "reco_energy": "Reconstructed energy",
            "reco_az": "Reconstructed azimuth",
            "reco_alt": "Reconstructed altitude",
            "reco_fov_lat": "Reconstructed field of view lat",
            "reco_fov_lon": "Reconstructed field of view lon",
            "pointing_az": "Pointing azimuth",
            "pointing_alt": "Pointing altitude",
            "theta": "Reconstructed angular offset from source position",
            "true_source_fov_offset": "Simulated angular offset from pointing direction",
            "reco_source_fov_offset": "Reconstructed angular offset from pointing direction",
            "gh_score": "prediction of the classifier, defined between [0,1], where values close to 1 mean that the positive class (e.g. gamma in gamma-ray analysis) is more likely",
        }

        return QTable(names=columns, units=units, descriptions=descriptions)


class EventsLoader(Component):
    classes = [EventPreProcessor]

    def __init__(self, kind, file, target_spectrum, **kwargs):
        super().__init__(**kwargs)

        self.epp = EventPreProcessor(parent=self)
        self.target_spectrum = target_spectrum
        self.kind = kind
        self.file = file

    def load_preselected_events(self, chunk_size, obs_time, fov_bins):
        opts = dict(dl2=True, simulated=True)
        with TableLoader(self.file, parent=self, **opts) as load:
            header = self.epp.make_empty_table()
            sim_info, spectrum, obs_conf = self.get_metadata(load, obs_time)
            if self.kind == "gammas":
                meta = {"sim_info": sim_info, "spectrum": spectrum}
            else:
                meta = None
            bits = [header]
            n_raw_events = 0
            for _, _, events in load.read_subarray_events_chunked(chunk_size, **opts):
                selected = events[self.epp.get_table_mask(events)]
                selected = self.epp.normalise_column_names(selected)
                selected = self.make_derived_columns(
                    selected, spectrum, obs_conf, fov_bins
                )
                bits.append(selected)
                n_raw_events += len(events)

            bits.append(header)  # Putting it last ensures the correct metadata is used
            table = vstack(bits, join_type="exact", metadata_conflicts="silent")
            return table, n_raw_events, meta

    def get_metadata(self, loader, obs_time):
        obs = loader.read_observation_information()
        sim = loader.read_simulation_configuration()
        show = loader.read_shower_distribution()

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

        return (
            sim_info,
            PowerLaw.from_simulation(sim_info, obstime=obs_time),
            obs,
        )

    def make_derived_columns(self, events, spectrum, obs_conf, fov_bins):
        if obs_conf["subarray_pointing_lat"].std() < 1e-3:
            assert all(obs_conf["subarray_pointing_frame"] == 0)
            # Lets suppose 0 means ALTAZ
            events["pointing_alt"] = obs_conf["subarray_pointing_lat"][0] * u.deg
            events["pointing_az"] = obs_conf["subarray_pointing_lon"][0] * u.deg
        else:
            raise NotImplementedError(
                "No support for making irfs from varying pointings yet"
            )

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
        events["reco_fov_lon"] = -reco_nominal.fov_lon  # minus for GADF
        events["reco_fov_lat"] = reco_nominal.fov_lat

        if (
            self.kind == "gammas"
            and self.target_spectrum.normalization.unit.is_equivalent(
                spectrum.normalization.unit * u.sr
            )
        ):
            if isinstance(fov_bins, FovOffsetBinning):
                spectrum = spectrum.integrate_cone(
                    fov_bins.fov_offset_min * u.deg, fov_bins.fov_offset_max * u.deg
                )
            else:
                spectrum = spectrum.integrate_cone(fov_bins[0], fov_bins[-1])

        events["weight"] = calculate_event_weights(
            events["true_energy"],
            target_spectrum=self.target_spectrum,
            simulated_spectrum=spectrum,
        )

        return events


class ThetaCutsCalculator(Component):
    theta_min_angle = Float(
        default_value=-1, help="Smallest angular cut value allowed (-1 means no cut)"
    ).tag(config=True)

    theta_max_angle = Float(
        default_value=0.32, help="Largest angular cut value allowed"
    ).tag(config=True)

    theta_min_counts = Integer(
        default_value=10,
        help="Minimum number of events in a bin to attempt to find a cut value",
    ).tag(config=True)

    theta_fill_value = Float(
        default_value=0.32, help="Angular cut value used for bins with too few events"
    ).tag(config=True)

    theta_smoothing = Float(
        default_value=None,
        allow_none=True,
        help="When given, the width (in units of bins) of gaussian smoothing applied (None)",
    ).tag(config=True)

    target_percentile = Float(
        default_value=68,
        help="Percent of events in each energy bin to keep after the theta cut",
    ).tag(config=True)

    def calculate_theta_cuts(self, theta, reco_energy, energy_bins):
        theta_min_angle = (
            None if self.theta_min_angle < 0 else self.theta_min_angle * u.deg
        )
        theta_max_angle = (
            None if self.theta_max_angle < 0 else self.theta_max_angle * u.deg
        )
        if self.theta_smoothing:
            theta_smoothing = None if self.theta_smoothing < 0 else self.theta_smoothing
        else:
            theta_smoothing = self.theta_smoothing

        return calculate_percentile_cut(
            theta,
            reco_energy,
            energy_bins,
            min_value=theta_min_angle,
            max_value=theta_max_angle,
            smoothing=theta_smoothing,
            percentile=self.target_percentile,
            fill_value=self.theta_fill_value * u.deg,
            min_events=self.theta_min_counts,
        )
