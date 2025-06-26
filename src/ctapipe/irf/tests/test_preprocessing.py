import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from pyirf.simulations import SimulatedEventsInfo
from pyirf.spectral import PowerLaw


@pytest.fixture(scope="module")
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
    ]
    assert columns.sort() == events.colnames.sort()

    assert isinstance(count, int)
    assert isinstance(meta["sim_info"], SimulatedEventsInfo)
    assert isinstance(meta["spectrum"], PowerLaw)

    events = loader.make_event_weights(
        events, meta["spectrum"], "gammas", (0 * u.deg, 1 * u.deg)
    )
    assert "weight" in events.colnames
