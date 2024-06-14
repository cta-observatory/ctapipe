import astropy.units as u
import pytest
from astropy.table import Table
from pyirf.simulations import SimulatedEventsInfo
from pyirf.spectral import PowerLaw
from traitlets.config import Config

from ctapipe.core.tool import run_tool


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
            "alt_geom": [58.5, 61.2, 59, 71.6, 60, 62] * u.deg,
            "true_az": [13, 13, 13, 13, 13, 13] * u.deg,
            "az_geom": [12.5, 13, 11.8, 15.1, 14.7, 12.8] * u.deg,
        }
    )


@pytest.fixture(scope="module")
def gamma_diffuse_full_reco_file(
    gamma_train_clf,
    particle_classifier_path,
    model_tmp_path,
):
    """
    Energy reconstruction and geometric origin reconstruction have already been done.
    """
    from ctapipe.tools.apply_models import ApplyModels

    output_path = model_tmp_path / "gamma_diffuse_full_reco.dl2.h5"
    run_tool(
        ApplyModels(),
        argv=[
            f"--input={gamma_train_clf}",
            f"--output={output_path}",
            f"--reconstructor={particle_classifier_path}",
            "--no-dl1-parameters",
            "--StereoMeanCombiner.weights=konrad",
        ],
        raises=True,
    )
    return output_path


@pytest.fixture(scope="module")
def proton_full_reco_file(
    proton_train_clf,
    particle_classifier_path,
    model_tmp_path,
):
    """
    Energy reconstruction and geometric origin reconstruction have already been done.
    """
    from ctapipe.tools.apply_models import ApplyModels

    output_path = model_tmp_path / "proton_full_reco.dl2.h5"
    run_tool(
        ApplyModels(),
        argv=[
            f"--input={proton_train_clf}",
            f"--output={output_path}",
            f"--reconstructor={particle_classifier_path}",
            "--no-dl1-parameters",
            "--StereoMeanCombiner.weights=konrad",
        ],
        raises=True,
    )
    return output_path


def test_normalise_column_names(dummy_table):
    from ctapipe.irf import EventPreProcessor

    epp = EventPreProcessor(
        energy_reconstructor="dummy",
        geometry_reconstructor="geom",
        gammaness_classifier="classifier",
        rename_columns=[("alt_geom", "reco_alt"), ("az_geom", "reco_az")],
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
    ]
    for c in needed_cols:
        assert c in norm_table.colnames

    # error if reco_{alt,az} is missing because of no-standard name
    with pytest.raises(ValueError, match="No column corresponding"):
        epp = EventPreProcessor(
            energy_reconstructor="dummy",
            geometry_reconstructor="geom",
            gammaness_classifier="classifier",
        )
        norm_table = epp.normalise_column_names(dummy_table)


def test_events_loader(gamma_diffuse_full_reco_file):
    from ctapipe.irf import EventsLoader, Spectra

    config = Config(
        {
            "EventPreProcessor": {
                "energy_reconstructor": "ExtraTreesRegressor",
                "geometry_reconstructor": "HillasReconstructor",
                "gammaness_classifier": "ExtraTreesClassifier",
                "quality_criteria": [
                    (
                        "multiplicity 4",
                        "np.count_nonzero(tels_with_trigger,axis=1) >= 4",
                    ),
                    ("valid classifier", "ExtraTreesClassifier_is_valid"),
                    ("valid geom reco", "HillasReconstructor_is_valid"),
                    ("valid energy reco", "ExtraTreesRegressor_is_valid"),
                ],
            }
        }
    )
    loader = EventsLoader(
        config=config,
        kind="gammas",
        file=gamma_diffuse_full_reco_file,
        target_spectrum=Spectra.CRAB_HEGRA,
    )
    events, count, meta = loader.load_preselected_events(
        chunk_size=10000,
        obs_time=u.Quantity(50, u.h),
        valid_fov=u.Quantity([0, 1], u.deg),
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
    assert columns.sort() == events.colnames.sort()

    assert isinstance(meta["sim_info"], SimulatedEventsInfo)
    assert isinstance(meta["spectrum"], PowerLaw)