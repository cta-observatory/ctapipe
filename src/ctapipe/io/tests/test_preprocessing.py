import astropy.units as u
import pytest


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
    ]

    assert sorted(columns) == sorted(events.colnames)
    assert isinstance(count, int)
    assert isinstance(meta["sim_info"], SimulatedEventsInfo)
    assert isinstance(meta["spectrum"], PowerLaw)

    events = loader.make_event_weights(
        events, meta["spectrum"], "gammas", (0 * u.deg, 1 * u.deg)
    )

    assert "weight" in events.colnames
