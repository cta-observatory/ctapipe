import astropy.units as u
import numpy as np
import pytest
from astropy.table import Column, QTable, Table
from pyirf.simulations import SimulatedEventsInfo
from pyirf.spectral import PowerLaw
from traitlets.config import Config


@pytest.fixture(scope="module")
def dummy_table():
    """Dummy table to test column renaming."""
    return Table(
        {
            "obs_id": [1, 1, 1, 2, 3, 3],
            "event_id": [1, 2, 3, 1, 1, 2],
            "true_energy": [0.99, 10, 0.37, 2.1, 73.4, 1] * u.TeV,
            "dummy_energy": [1, 10, 0.4, 2.5, 73, 1] * u.TeV,
            "classifier_prediction": [1, 0.3, 0.87, 0.93, 0, 0.1],
            "true_alt": [60, 60, 60, 60, 60, 60] * u.deg,
            "geom_alt": [58.5, 61.2, 59, 71.6, 60, 62] * u.deg,
            "true_az": [13, 13, 13, 13, 13, 13] * u.deg,
            "geom_az": [12.5, 13, 11.8, 15.1, 14.7, 12.8] * u.deg,
            "subarray_pointing_frame": np.zeros(6),
            "subarray_pointing_lat": np.full(6, 20) * u.deg,
            "subarray_pointing_lon": np.full(6, 0) * u.deg,
        }
    )


@pytest.fixture()
def test_config():
    return {
        "EventLoader": {"event_reader_function": "read_telescope_events_chunked"},
        "EventPreprocessor": {
            "energy_reconstructor": "ExtraTreesRegressor",
            "gammaness_classifier": "ExtraTreesClassifier",
            "fixed_columns": [
                "obs_id",
                "event_id",
                "tel_id",
                "ExtraTreesRegressor_tel_energy",
                "ExtraTreesRegressor_tel_energy_uncert",
            ],
            "columns_to_rename_override": {},
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
        "EventQualityQuery": {
            "quality_criteria": [
                ("valid reco", "ExtraTreesRegressor_tel_is_valid"),
            ]
        },
    }


def test_normalise_column_names(dummy_table):
    from ctapipe.irf import EventPreprocessor

    epp = EventPreprocessor(
        energy_reconstructor="dummy",
        geometry_reconstructor="geom",
        gammaness_classifier="classifier",
    )
    norm_table = epp.normalise_column_names(dummy_table)

    needed_cols = [
        "obs_id",
        "event_id",
        "true_energy",
        "true_alt",
        "true_az",
        "reco_energy",
        "reco_alt",
        "reco_az",
        "gh_score",
        "pointing_alt",
        "pointing_az",
    ]
    for c in needed_cols:
        assert c in norm_table.colnames

    with pytest.raises(ValueError, match="Required column geom_alt is missing."):
        dummy_table.rename_column("geom_alt", "alt_geom")
        epp = EventPreprocessor(
            energy_reconstructor="dummy",
            geometry_reconstructor="geom",
            gammaness_classifier="classifier",
        )
        _ = epp.normalise_column_names(dummy_table)


def test_event_loader(gamma_diffuse_full_reco_file, irf_event_loader_test_config):
    from ctapipe.irf import EventLoader, Spectra

    loader = EventLoader(
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


def test_preprocessor_tel_table_with_custom_reconstructor(tmp_path, test_config):
    from ctapipe.irf import EventPreprocessor

    # Create a test table with required columns
    table = QTable(
        {
            "obs_id": [1, 1, 2],
            "event_id": [100, 101, 102],
            "tel_id": [1, 1, 1],
            "ExtraTreesRegressor_tel_energy": [1.0, 2.0, 3.0] * u.TeV,
            "ExtraTreesRegressor_tel_is_valid": [True, False, True],
            "ExtraTreesRegressor_tel_energy_uncert": [0.1, 0.2, 0.1],
            "ExtraTreesRegressor_tel_goodness_of_fit": [0.9, 0.8, 0.95],
            "subarray_pointing_lat": [80.0, 80.0, 80.0] * u.deg,
            "subarray_pointing_lon": [0.0, 0.0, 0.0] * u.deg,
            "true_energy": [1.1, 2.1, 3.1] * u.TeV,
            "true_az": [42.0, 43.0, 44.0] * u.deg,
            "true_alt": [70.0, 71.0, 72.0] * u.deg,
        }
    )

    # Set up config
    config = test_config

    # Create preprocessor with config
    preprocessor = EventPreprocessor(config=Config(config))

    # Apply quality query and preprocessing
    mask = preprocessor.quality_query.get_table_mask(table)
    filtered = table[mask]

    # Apply renaming and derived column generation
    processed = preprocessor.normalise_column_names(filtered)

    # Check expected column names after renaming
    assert "ExtraTreesRegressor_tel_energy" in processed.colnames
    assert "obs_id" in processed.colnames  # might exist depending on classifier config
    assert "tel_id" in processed.colnames

    # Check the number of surviving rows (only valid events)
    assert len(processed) == 2
    assert np.all(processed["ExtraTreesRegressor_tel_energy"] > 0 * u.TeV)


def test_loader_tel_table(gamma_diffuse_full_reco_file, test_config):
    from ctapipe.irf import EventLoader, Spectra

    test_config["EventLoader"][
        "event_reader_function"
    ] = "read_telescope_events_chunked"
    test_config["EventQualityQuery"]["quality_criteria"].append(
        ("telescope ID", "(tel_id == 35.0) | (tel_id == 19.0)"),
    )

    loader = EventLoader(
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


def test_name_overriding(dummy_table):
    from ctapipe.irf import EventPreprocessor

    epp = EventPreprocessor(
        energy_reconstructor="dummy",
        geometry_reconstructor="geom",
        gammaness_classifier="classifier",
        fixed_columns=["obs_id", "event_id", "true_az", "true_alt"],
        columns_to_rename_override={"true_energy": "false_energy"},
    )
    norm_table = epp.normalise_column_names(dummy_table)
    columns = [
        "obs_id",
        "event_id",
        "false_energy",
        "true_az",
        "true_alt",
    ]
    assert sorted(columns) == sorted(norm_table.colnames)
