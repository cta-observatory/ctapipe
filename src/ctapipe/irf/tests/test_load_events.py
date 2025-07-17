import astropy.units as u
import numpy as np
import pytest
from astropy.table import Column
from pyirf.simulations import SimulatedEventsInfo
from pyirf.spectral import PowerLaw
from traitlets.config import Config

from ctapipe.io.dl2_tables_preprocessing import DL2EventLoader
from ctapipe.irf import Spectra


@pytest.fixture(scope="function")
def test_config():
    return {
        "DL2EventLoader": {"event_reader_function": "read_telescope_events_chunked"},
        "DL2EventPreprocessor": {
            "energy_reconstructor": "ExtraTreesRegressor",
            "gammaness_classifier": "ExtraTreesClassifier",
            "columns_to_rename": {},
            "output_table_schema": [
                Column(
                    name="obs_id", dtype=np.uint64, description="Observation Block ID"
                ),
                Column(name="event_id", dtype=np.uint64, description="Array event ID"),
                Column(name="tel_id", dtype=np.uint64, description="Telescope ID"),
                Column(
                    name="ExtraTreesRegressor_tel_energy",
                    unit=u.TeV,
                    description="Reconstructed energy",
                ),
                Column(
                    name="ExtraTreesRegressor_tel_energy_uncert",
                    unit=u.TeV,
                    description="Reconstructed energy uncertainty",
                ),
            ],
            "apply_derived_columns": False,
            # "disable_column_renaming": True,
            "apply_check_pointing": False,
        },
        "DL2EventQualityQuery": {
            "quality_criteria": [
                ("valid reco", "ExtraTreesRegressor_tel_is_valid"),
            ]
        },
    }


def test_event_loader(gamma_diffuse_full_reco_file, irf_event_loader_test_config):
    loader = DL2EventLoader(
        config=irf_event_loader_test_config,
        file=gamma_diffuse_full_reco_file,
        target_spectrum=Spectra.CRAB_HEGRA,
    )
    events, count, meta = loader.load_preselected_events(
        chunk_size=10000,
        obs_time=u.Quantity(50, u.h),
    )

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

    assert sorted(columns) == sorted(events.colnames)
    assert isinstance(count, int)
    assert isinstance(meta["sim_info"], SimulatedEventsInfo)
    assert isinstance(meta["spectrum"], PowerLaw)

    events = loader.make_event_weights(
        events, meta["spectrum"], "gammas", (0 * u.deg, 1 * u.deg)
    )

    assert "weight" in events.colnames


def test_loader_tel_table(gamma_diffuse_full_reco_file, test_config):
    from ctapipe.io.dl2_tables_preprocessing import DL2EventLoader
    from ctapipe.irf import Spectra

    test_config["DL2EventLoader"][
        "event_reader_function"
    ] = "read_telescope_events_chunked"
    test_config["DL2EventQualityQuery"]["quality_criteria"].append(
        ("telescope ID", "(tel_id == 35.0) | (tel_id == 19.0)"),
    )

    loader = DL2EventLoader(
        config=Config(test_config),
        file=gamma_diffuse_full_reco_file,
        target_spectrum=Spectra.CRAB_HEGRA,
    )
    events, count, meta = loader.load_preselected_events(
        chunk_size=10000,
        obs_time=u.Quantity(50, u.h),
    )

    columns = [
        "obs_id",
        "event_id",
        "tel_id",
        "ExtraTreesRegressor_tel_energy",
        "ExtraTreesRegressor_tel_energy_uncert",
    ]

    assert sorted(columns) == sorted(events.colnames)
    assert np.all(events["ExtraTreesRegressor_tel_energy"] > 0 * u.TeV)
    assert np.all(np.isin(events["tel_id"], [19, 35]))
