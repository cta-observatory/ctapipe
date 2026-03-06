import astropy.units as u
import numpy as np
import pytest
from astropy.table import Column, QTable, Table
from traitlets.config import Config


@pytest.fixture(scope="function")
def dummy_table():
    """Dummy table to test column renaming."""
    return Table(
        {
            "obs_id": [1, 1, 1, 2, 3, 3],
            "event_id": [1, 2, 3, 1, 1, 2],
            "true_energy": [0.99, 10, 0.37, 2.1, 73.4, 1] * u.TeV,
            "dummy_energy": [1, 10, 0.4, 2.5, 73, 1] * u.TeV,
            "dummy_telescopes": [
                [True, True, True, False, False, True],
                [True, True, True, True, False, True],
                [False, True, True, True, True, False],
                [False, False, False, False, True, True],
                [True, True, True, True, True, True],
                [True, False, True, False, True, True],
            ],
            "classifier_prediction": [1, 0.3, 0.87, 0.93, 0, 0.1],
            "classifier_telescopes": [
                [True, True, True, False, False, True],
                [True, True, True, True, False, True],
                [False, True, True, True, True, False],
                [False, False, False, False, True, True],
                [True, True, True, True, True, True],
                [True, False, True, False, True, True],
            ],
            "true_alt": [60, 60, 60, 60, 60, 60] * u.deg,
            "geom_alt": [58.5, 61.2, 59, 71.6, 60, 62] * u.deg,
            "true_az": [13, 13, 13, 13, 13, 13] * u.deg,
            "geom_az": [12.5, 13, 11.8, 15.1, 14.7, 12.8] * u.deg,
            "geom_telescopes": [
                [True, True, True, False, False, True],
                [True, True, True, True, False, True],
                [False, True, True, True, True, False],
                [False, False, False, False, True, True],
                [True, True, True, True, True, True],
                [True, False, True, False, True, True],
            ],
            "subarray_pointing_frame": np.zeros(6),
            "subarray_pointing_lat": np.full(6, 20) * u.deg,
            "subarray_pointing_lon": np.full(6, 0) * u.deg,
        }
    )


def test_normalise_column_names(dummy_table):
    from ctapipe.io.dl2_tables_preprocessing import DL2EventPreprocessor

    output_table_schema = [
        Column(name="obs_id", dtype=np.uint64, description="Observation Block ID"),
        Column(name="event_id", dtype=np.uint64, description="Array event ID"),
        Column(name="true_energy", unit=u.TeV, description="Simulated energy"),
        Column(name="true_az", unit=u.deg, description="Simulated azimuth"),
        Column(name="true_alt", unit=u.deg, description="Simulated altitude"),
        Column(name="reco_energy", unit=u.TeV, description="Reconstructed energy"),
        Column(name="reco_az", unit=u.deg, description="Reconstructed azimuth"),
        Column(name="reco_alt", unit=u.deg, description="Reconstructed altitude"),
        Column(name="pointing_alt", unit=u.deg, description="Pointing latitude"),
        Column(name="pointing_az", unit=u.deg, description="Pointing longitude"),
        Column(
            name="gh_score",
            dtype=np.float64,
            description="prediction of the classifier, defined between [0,1],"
            " where values close to 1 mean that the positive class"
            " (e.g. gamma in gamma-ray analysis) is more likely",
        ),
    ]
    epp = DL2EventPreprocessor(
        energy_reconstructor="dummy",
        geometry_reconstructor="geom",
        gammaness_classifier="classifier",
        output_table_schema=output_table_schema,
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
        epp = DL2EventPreprocessor(
            energy_reconstructor="dummy",
            geometry_reconstructor="geom",
            gammaness_classifier="classifier",
            output_table_schema=output_table_schema,
        )
        _ = epp.normalise_column_names(dummy_table)


def test_event_loader(gamma_diffuse_full_reco_file, irf_event_loader_test_config):
    pytest.importorskip("pyirf", reason="pyirf is an optional dependency")
    from pyirf.simulations import SimulatedEventsInfo
    from pyirf.spectral import PowerLaw

    from ctapipe.io.dl2_tables_preprocessing import DL2EventLoader
    from ctapipe.irf import Spectra

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
        "multiplicity",
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
    from ctapipe.io.dl2_tables_preprocessing import DL2EventPreprocessor

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
    preprocessor = DL2EventPreprocessor(config=Config(config))

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


def test_name_overriding(dummy_table):
    from ctapipe.io.dl2_tables_preprocessing import DL2EventPreprocessor

    epp = DL2EventPreprocessor(
        energy_reconstructor="dummy",
        geometry_reconstructor="geom",
        gammaness_classifier="classifier",
        columns_to_rename={"true_energy": "false_energy"},
        output_table_schema=[
            Column(name="obs_id", dtype=np.uint64, description="Observation Block ID"),
            Column(name="event_id", dtype=np.uint64, description="Array event ID"),
            Column(name="false_energy", unit=u.TeV, description="Simulated energy"),
            Column(name="true_az", unit=u.deg, description="Simulated azimuth"),
            Column(name="true_alt", unit=u.deg, description="Simulated altitude"),
            Column(name="dummy_energy", unit=u.TeV, description="Reconstructed energy"),
            Column(name="geom_az", unit=u.deg, description="Reconstructed azimuth"),
            Column(name="geom_alt", unit=u.deg, description="Reconstructed altitude"),
            Column(
                name="subarray_pointing_frame",
                unit=u.dimensionless_unscaled,
                description="Pointing frame",
            ),
            Column(
                name="subarray_pointing_lat",
                unit=u.deg,
                description="Pointing latitude",
            ),
            Column(
                name="subarray_pointing_lon",
                unit=u.deg,
                description="Pointing longitude",
            ),
            Column(
                name="classifier_prediction",
                unit=u.dimensionless_unscaled,
                description="prediction of the classifier, defined between [0,1],"
                " where values close to 1 mean that the positive class"
                " (e.g. gamma in gamma-ray analysis) is more likely",
            ),
        ],
    )
    norm_table = epp.normalise_column_names(dummy_table)
    columns = [
        "obs_id",
        "event_id",
        "false_energy",
        "true_az",
        "true_alt",
        "dummy_energy",
        "dummy_telescopes",
        "classifier_prediction",
        "classifier_telescopes",
        "geom_alt",
        "geom_az",
        "geom_telescopes",
        "subarray_pointing_frame",
        "subarray_pointing_lat",
        "subarray_pointing_lon",
    ]
    assert sorted(columns) == sorted(norm_table.colnames)
