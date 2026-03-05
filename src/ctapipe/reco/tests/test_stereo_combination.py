import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from numpy.testing import assert_allclose, assert_array_equal

from ctapipe.containers import (
    ArrayEventContainer,
    ArrayPointingContainer,
    DispContainer,
    HillasParametersContainer,
    ImageParametersContainer,
    MorphologyContainer,
    ParticleClassificationContainer,
    ReconstructedContainer,
    ReconstructedEnergyContainer,
    ReconstructedGeometryContainer,
    TelescopeReconstructedContainer,
)
from ctapipe.core.traits import TraitError
from ctapipe.reco.reconstructor import ReconstructionProperty
from ctapipe.reco.stereo_combination import StereoDispCombiner, StereoMeanCombiner


@pytest.fixture(scope="module")
def mono_table():
    """
    Dummy table of telescope events with a
    prediction and weights.
    """
    return Table(
        {
            "obs_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "event_id": [1, 1, 1, 2, 2, 1, 2, 2, 2, 2],
            "tel_id": [1, 2, 3, 5, 7, 1, 1, 3, 4, 5],
            "hillas_intensity": [1, 2, 0, 1, 5, 9, 10, 20, 1, 2],
            "hillas_width": [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2] * u.deg,
            "hillas_length": 3
            * ([0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2] * u.deg),
            "hillas_fov_lon": [-0.5, 0, 0.5, -1, 1, 1.5, -0.5, 0, 0.5, -1] * u.deg,
            "hillas_fov_lat": [0.3, -0.3, 0.3, 0.5, 0.5, 0.2, 0.3, -0.3, 0.3, 0.5]
            * u.deg,
            "hillas_psi": [40, 85, -40, -35, 35, 55, 40, 85, -40, -35] * u.deg,
            "dummy_tel_energy": [1, 10, 4, 0.5, 0.7, 1, 1, 9, 4, 0.5] * u.TeV,
            "dummy_tel_is_valid": [
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
            ],
            "classifier_tel_prediction": [1, 0, 0.5, 0, 0.6, 1, 1, 0, 0.5, 0],
            "classifier_tel_is_valid": [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            "disp_tel_alt": [58.5, 58, 62.5, 72, 74.5, 81, 58.5, 58, 62.5, 72] * u.deg,
            "disp_tel_az": [12.5, 15, 13, 21, 20, 14.5, 12.5, 15, 13, 21] * u.deg,
            "disp_tel_is_valid": [
                True,
                False,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
            ],
            "disp_tel_parameter": [0.65, 1.1, 0.7, 0.9, 1, 0.5, 0.65, 1.1, 0.7, 0.9]
            * u.deg,
            "disp_tel_sign_score": [
                0.65,
                0.87,
                0.7,
                0.5,
                0.3,
                0.99,
                0.65,
                0.3,
                0.2,
                0.86,
            ],
            "subarray_pointing_lat": 10 * [70] * u.deg,
            "subarray_pointing_lon": 10 * [0] * u.deg,
        }
    )


@pytest.mark.parametrize("weights", ["aspect-weighted-intensity", "intensity", "none"])
def test_predict_mean_energy(weights, mono_table):
    combine = StereoMeanCombiner(
        prefix="dummy",
        property=ReconstructionProperty.ENERGY,
        weights=weights,
    )
    stereo = combine.predict_table(mono_table)
    assert stereo.colnames == [
        "obs_id",
        "event_id",
        "dummy_energy",
        "dummy_energy_uncert",
        "dummy_is_valid",
        "dummy_goodness_of_fit",
        "dummy_telescopes",
    ]
    assert_array_equal(stereo["obs_id"], np.array([1, 1, 2, 2]))
    assert_array_equal(stereo["event_id"], np.array([1, 2, 1, 2]))
    if weights == "intensity":
        assert_allclose(
            stereo["dummy_energy"].quantity,
            [7, 0.5, np.nan, 5.90909091] * u.TeV,
            atol=1e-7,
        )
        assert_allclose(
            stereo["dummy_energy_uncert"].quantity,
            [4.242641, 0, np.nan, 3.869959] * u.TeV,
            atol=1e-7,
        )
    elif weights == "none":
        assert_allclose(
            stereo["dummy_energy"].quantity, [5, 0.5, np.nan, 3.625] * u.TeV, atol=1e-7
        )
        assert_allclose(
            stereo["dummy_energy_uncert"].quantity,
            [3.741657, 0, np.nan, 3.3796265] * u.TeV,
            atol=1e-7,
        )

    assert_array_equal(stereo["dummy_telescopes"][0], np.array([1, 2, 3]))
    assert_array_equal(stereo["dummy_telescopes"][1], 5)
    assert_array_equal(stereo["dummy_telescopes"][3], np.array([1, 3, 4, 5]))


def test_predict_mean_classification(mono_table):
    combine = StereoMeanCombiner(
        prefix="classifier",
        property=ReconstructionProperty.PARTICLE_TYPE,
    )
    stereo = combine.predict_table(mono_table)
    assert stereo.colnames == [
        "obs_id",
        "event_id",
        "classifier_prediction",
        "classifier_is_valid",
        "classifier_goodness_of_fit",
        "classifier_telescopes",
    ]
    assert_array_equal(stereo["obs_id"], np.array([1, 1, 2, 2]))
    assert_array_equal(stereo["event_id"], np.array([1, 2, 1, 2]))
    assert_array_equal(
        stereo["classifier_prediction"],
        [0.5, 0.3, 1, 0.375],
    )
    tel_ids = stereo["classifier_telescopes"]
    assert_array_equal(tel_ids[0], [1, 2])
    assert_array_equal(tel_ids[1], [5, 7])
    assert_array_equal(tel_ids[2], [1])
    assert_array_equal(tel_ids[3], [1, 3, 4, 5])


def test_predict_mean_disp(mono_table):
    combine = StereoMeanCombiner(
        prefix="disp",
        property=ReconstructionProperty.GEOMETRY,
    )
    stereo = combine.predict_table(mono_table)

    for name, field in ReconstructedGeometryContainer.fields.items():
        colname = f"disp_{name}"
        assert colname in stereo.colnames
        assert stereo[colname].description == field.description

    assert "obs_id" in stereo.colnames
    assert "event_id" in stereo.colnames

    assert_array_equal(stereo["obs_id"], np.array([1, 1, 2, 2]))
    assert_array_equal(stereo["event_id"], np.array([1, 2, 1, 2]))
    assert_allclose(
        stereo["disp_alt"].quantity,
        [60.5002328, np.nan, 81, 62.773741] * u.deg,
        atol=1e-7,
    )
    assert_allclose(
        stereo["disp_az"].quantity,
        [12.7345693, np.nan, 14.5, 14.792156] * u.deg,
        atol=1e-7,
    )
    tel_ids = stereo["disp_telescopes"]
    assert_array_equal(tel_ids[0], [1, 3])
    assert_array_equal(tel_ids[1], [])
    assert_array_equal(tel_ids[2], [1])
    assert_array_equal(tel_ids[3], [1, 3, 4, 5])


@pytest.mark.parametrize("weights", ["aspect-weighted-intensity", "intensity", "none"])
def test_mean_prediction_single_event(weights):
    event = ArrayEventContainer()

    for tel_id, intensity in zip((25, 125, 130), (100, 200, 400)):
        event.dl1.tel[tel_id].parameters = ImageParametersContainer(
            hillas=HillasParametersContainer(
                intensity=intensity,
                width=0.1 * u.deg,
                length=0.3 * u.deg,
            ),
            morphology=MorphologyContainer(
                n_pixels=10,
            ),
        )

    event.dl2.tel[25] = ReconstructedContainer(
        energy={
            "dummy": ReconstructedEnergyContainer(energy=10 * u.GeV, is_valid=True)
        },
        particle_type={
            "dummy": ParticleClassificationContainer(prediction=1.0, is_valid=True)
        },
        geometry={
            "dummy": ReconstructedGeometryContainer(
                alt=60 * u.deg, az=15 * u.deg, is_valid=True
            )
        },
    )
    event.dl2.tel[125] = ReconstructedContainer(
        energy={
            "dummy": ReconstructedEnergyContainer(energy=20 * u.GeV, is_valid=True)
        },
        particle_type={
            "dummy": ParticleClassificationContainer(prediction=0.0, is_valid=True)
        },
        geometry={
            "dummy": ReconstructedGeometryContainer(
                alt=50 * u.deg, az=30 * u.deg, is_valid=True
            )
        },
    )
    event.dl2.tel[130] = ReconstructedContainer(
        energy={
            "dummy": ReconstructedEnergyContainer(energy=0.04 * u.TeV, is_valid=True)
        },
        particle_type={
            "dummy": ParticleClassificationContainer(prediction=0.8, is_valid=True)
        },
        geometry={
            "dummy": ReconstructedGeometryContainer(
                alt=45 * u.deg, az=280 * u.deg, is_valid=True
            )
        },
    )

    combine_energy = StereoMeanCombiner(
        prefix="dummy",
        property=ReconstructionProperty.ENERGY,
        weights=weights,
    )
    combine_classification = StereoMeanCombiner(
        prefix="dummy",
        property=ReconstructionProperty.PARTICLE_TYPE,
        weights=weights,
    )
    combine_geometry = StereoMeanCombiner(
        prefix="dummy",
        property=ReconstructionProperty.GEOMETRY,
        weights=weights,
    )
    combine_energy(event)
    combine_classification(event)
    combine_geometry(event)
    if weights == "none":
        assert u.isclose(event.dl2.stereo.energy["dummy"].energy, (70 / 3) * u.GeV)
        assert u.isclose(event.dl2.stereo.geometry["dummy"].alt, 63.0738383 * u.deg)
        assert u.isclose(event.dl2.stereo.geometry["dummy"].az, 348.0716693 * u.deg)
    elif weights == "intensity":
        assert u.isclose(event.dl2.stereo.energy["dummy"].energy, 30 * u.GeV)
        assert u.isclose(event.dl2.stereo.geometry["dummy"].alt, 60.9748605 * u.deg)
        assert u.isclose(event.dl2.stereo.geometry["dummy"].az, 316.0365515 * u.deg)
    assert event.dl2.stereo.particle_type["dummy"].prediction == pytest.approx(0.6)


def test_reconstructed_container_warning():
    from ctapipe.utils.deprecation import CTAPipeDeprecationWarning

    container = ReconstructedContainer()

    with pytest.warns(CTAPipeDeprecationWarning, match="renamed"):
        _ = container.classification

    with pytest.warns(CTAPipeDeprecationWarning, match="renamed"):
        container.classification = ParticleClassificationContainer()


def _make_disp_event(event_dict, prefix="dummy"):
    event = ArrayEventContainer()

    for i in range(len(event_dict["tel_id"])):
        event.dl1.tel[event_dict["tel_id"][i]].parameters = ImageParametersContainer(
            hillas=HillasParametersContainer(
                intensity=event_dict["hillas_intensity"][i],
                fov_lon=event_dict["hillas_fov_lon"][i],
                fov_lat=event_dict["hillas_fov_lat"][i],
                psi=event_dict["hillas_psi"][i],
                width=event_dict["hillas_width"][i],
                length=event_dict["hillas_length"][i],
            ),
            morphology=MorphologyContainer(
                n_pixels=10,
            ),
        )

        event.dl2.tel[event_dict["tel_id"][i]] = TelescopeReconstructedContainer(
            disp={
                prefix: DispContainer(
                    parameter=event_dict["disp_tel_parameter"][i],
                    sign_score=event_dict["disp_tel_sign_score"][i],
                )
            },
            geometry={
                prefix: ReconstructedGeometryContainer(
                    alt=event_dict["disp_tel_alt"][i],
                    az=event_dict["disp_tel_az"][i],
                    is_valid=event_dict["disp_tel_is_valid"][i],
                )
            },
        )

    event.monitoring.pointing = ArrayPointingContainer(
        array_azimuth=0 * u.deg, array_altitude=70 * u.deg
    )
    return event


def test_disp_combiner_trait_validation():
    with pytest.raises(TraitError):
        StereoDispCombiner(n_tel_combinations=1)

    with pytest.raises(TraitError):
        StereoDispCombiner(n_best_tels=1)

    with pytest.raises(TraitError):
        StereoDispCombiner(n_tel_combinations=3, n_best_tels=2)


@pytest.mark.parametrize("weights", ["aspect-weighted-intensity", "intensity", "none"])
def test_disp_combiner_single_event(weights):
    event_dict = {
        "tel_id": [1, 2, 9, 10],
        "hillas_intensity": [100, 200, 75, 30],
        "hillas_width": [0.1, 0.2, 0.1, 0.1] * u.deg,
        "hillas_length": 3 * ([0.1, 0.2, 0.1, 0.1] * u.deg),
        "hillas_fov_lon": [-0.5, 0, 0.5, 0.1] * u.deg,
        "hillas_fov_lat": [0.3, -0.3, 0.3, 0.2] * u.deg,
        "hillas_psi": [40, 85, -40, 30] * u.deg,
        "disp_tel_alt": [58.5, 58, 62.5, 20] * u.deg,
        "disp_tel_az": [12.5, 15, 13, 30] * u.deg,
        "disp_tel_parameter": [0.65, 1.1, 0.7, 1.0] * u.deg,
        "disp_tel_sign_score": [0.95, 0.98, 0.66, 0],
        "disp_tel_is_valid": [True, True, True, False],
    }

    event = _make_disp_event(event_dict)

    disp_combiner = StereoDispCombiner(
        prefix="dummy",
        weights=weights,
    )
    disp_combiner(event)
    if weights in ["intensity", "aspect-weighted-intensity"]:
        assert u.isclose(event.dl2.stereo.geometry["dummy"].alt, 70.76579427 * u.deg)
        assert u.isclose(event.dl2.stereo.geometry["dummy"].az, 0.13152707 * u.deg)
    elif weights == "none":
        assert u.isclose(event.dl2.stereo.geometry["dummy"].alt, 70.75451665 * u.deg)
        assert u.isclose(event.dl2.stereo.geometry["dummy"].az, 0.05821327 * u.deg)


def test_disp_combiner_single_event_disp_parameter():
    event = ArrayEventContainer()
    event.dl2.tel[1] = TelescopeReconstructedContainer(
        geometry={
            "dummy": ReconstructedGeometryContainer(
                is_valid=True,
            )
        },
    )

    disp_combiner = StereoDispCombiner(
        prefix="dummy",
    )

    with pytest.raises(RuntimeError, match="No valid DISP reconstruction parameter"):
        disp_combiner(event)


def test_disp_combiner_single_event_min_ang_diff():
    event_dict = {
        "tel_id": [1, 2],
        "hillas_intensity": [100, 200],
        "hillas_width": [0.1, 0.1] * u.deg,
        "hillas_length": [0.3, 0.3] * u.deg,
        "hillas_fov_lon": [0.0, 0.1] * u.deg,
        "hillas_fov_lat": [0.0, 0.1] * u.deg,
        "hillas_psi": [5, 8] * u.deg,
        "disp_tel_alt": [58.5, 58] * u.deg,
        "disp_tel_az": [12.5, 15] * u.deg,
        "disp_tel_parameter": [0.65, 1.1] * u.deg,
        "disp_tel_sign_score": [0.95, 0.98],
        "disp_tel_is_valid": [True, True],
    }
    event = _make_disp_event(event_dict)

    disp_combiner = StereoDispCombiner(
        prefix="dummy",
        min_ang_diff=10,
    )
    disp_combiner(event)

    geometry = event.dl2.stereo.geometry["dummy"]
    assert not geometry.is_valid
    assert np.isnan(geometry.alt.to_value(u.deg))
    assert np.isnan(geometry.az.to_value(u.deg))


def test_disp_combiner_n_best_tels_event():
    event_dict = {
        "tel_id": [1, 2, 3],
        "hillas_intensity": [150, 300, 50],
        "hillas_width": [0.1, 0.2, 0.1] * u.deg,
        "hillas_length": [0.3, 0.4, 0.2] * u.deg,
        "hillas_fov_lon": [-0.5, 0.1, 0.3] * u.deg,
        "hillas_fov_lat": [0.3, -0.2, 0.1] * u.deg,
        "hillas_psi": [40, 80, -20] * u.deg,
        "disp_tel_alt": [58.5, 58, 62.5] * u.deg,
        "disp_tel_az": [12.5, 15, 13] * u.deg,
        "disp_tel_parameter": [0.65, 1.1, 0.7] * u.deg,
        "disp_tel_sign_score": [0.95, 0.98, 0.66],
        "disp_tel_is_valid": [True, True, True],
    }
    event = _make_disp_event(event_dict)
    disp_combiner = StereoDispCombiner(
        prefix="dummy",
        weights="intensity",
        n_best_tels=2,
    )
    disp_combiner(event)

    expected_event = _make_disp_event(
        {key: value[:2] for key, value in event_dict.items()}
    )
    expected_combiner = StereoDispCombiner(
        prefix="dummy",
        weights="intensity",
        n_best_tels=None,
    )
    expected_combiner(expected_event)

    geometry = event.dl2.stereo.geometry["dummy"]
    expected_geometry = expected_event.dl2.stereo.geometry["dummy"]
    assert u.isclose(geometry.alt, expected_geometry.alt)
    assert u.isclose(geometry.az, expected_geometry.az)


def test_disp_combiner_n_tel_combinations_event():
    event_dict = {
        "tel_id": [1, 2, 3],
        "hillas_intensity": [150, 300, 50],
        "hillas_width": [0.1, 0.2, 0.1] * u.deg,
        "hillas_length": [0.3, 0.4, 0.2] * u.deg,
        "hillas_fov_lon": [-0.5, 0.1, 0.3] * u.deg,
        "hillas_fov_lat": [0.3, -0.2, 0.1] * u.deg,
        "hillas_psi": [40, 80, -20] * u.deg,
        "disp_tel_alt": [58.5, 58, 62.5] * u.deg,
        "disp_tel_az": [12.5, 15, 13] * u.deg,
        "disp_tel_parameter": [0.65, 1.1, 0.7] * u.deg,
        "disp_tel_sign_score": [0.95, 0.98, 0.66],
        "disp_tel_is_valid": [True, True, True],
    }
    event = _make_disp_event(event_dict)

    disp_combiner = StereoDispCombiner(
        prefix="dummy",
        n_tel_combinations=3,
    )
    disp_combiner(event)

    stereo = event.dl2.stereo.geometry["dummy"]
    assert stereo.is_valid
    assert u.isclose(stereo.alt, 70.80002984 * u.deg)
    assert u.isclose(stereo.az, 0.43926108 * u.deg)


def test_predict_disp_combiner(mono_table):
    disp_combiner = StereoDispCombiner(
        prefix="disp",
        n_tel_combinations=2,
    )
    stereo = disp_combiner.predict_table(mono_table)

    for name, field in ReconstructedGeometryContainer.fields.items():
        colname = f"disp_{name}"
        assert colname in stereo.colnames
        assert stereo[colname].description == field.description

    assert "obs_id" in stereo.colnames
    assert "event_id" in stereo.colnames

    assert_array_equal(stereo["obs_id"], np.array([1, 1, 2, 2]))
    assert_array_equal(stereo["event_id"], np.array([1, 2, 1, 2]))
    assert_allclose(
        stereo["disp_alt"].quantity,
        [70.7338725, np.nan, 81, 70.4917615] * u.deg,
        atol=1e-7,
    )
    assert_allclose(
        stereo["disp_az"].quantity,
        [359.9419634, np.nan, 14.5, 359.5978866] * u.deg,
        atol=1e-7,
    )
    tel_ids = stereo["disp_telescopes"]
    assert_array_equal(tel_ids[0], [1, 3])
    assert_array_equal(tel_ids[1], [])
    assert_array_equal(tel_ids[2], [1])
    assert_array_equal(tel_ids[3], [1, 3, 4, 5])


def test_predict_disp_combiner_n_best_tels(mono_table):
    disp_combiner = StereoDispCombiner(
        prefix="disp",
        weights="intensity",
        n_best_tels=2,
    )
    stereo = disp_combiner.predict_table(mono_table)

    expected_combiner = StereoDispCombiner(
        prefix="disp",
        weights="intensity",
        n_best_tels=None,
    )
    expected = expected_combiner.predict_table(mono_table[:-2])

    assert np.all(
        u.isclose(
            stereo["disp_alt"].quantity,
            expected["disp_alt"].quantity,
            equal_nan=True,
        )
    )
    assert np.all(
        u.isclose(
            stereo["disp_az"].quantity,
            expected["disp_az"].quantity,
            equal_nan=True,
        )
    )
    assert_array_equal(stereo["disp_telescopes"][-1], [1, 3])


def test_predict_disp_combiner_min_ang_diff(mono_table):
    disp_combiner = StereoDispCombiner(
        prefix="disp",
        min_ang_diff=85,
    )
    stereo = disp_combiner.predict_table(mono_table)

    assert np.isnan(stereo["disp_alt"][0])
    assert np.isnan(stereo["disp_az"][0])
    assert not stereo["disp_is_valid"][0]
    assert stereo["disp_telescopes"][0] == []

    disp_combiner = StereoDispCombiner(
        prefix="disp",
        min_ang_diff=75,
    )
    stereo = disp_combiner.predict_table(mono_table)

    assert np.isfinite(stereo["disp_alt"][0])
    assert np.isfinite(stereo["disp_az"][0])
    assert stereo["disp_is_valid"][0]
    assert stereo["disp_telescopes"][0] == [1, 3]


def test_predict_disp_combiner_n_tel_combinations(mono_table):
    disp_combiner = StereoDispCombiner(
        prefix="disp",
        n_tel_combinations=3,
    )
    stereo = disp_combiner.predict_table(mono_table)

    assert_allclose(
        stereo["disp_alt"].quantity,
        [70.7338725, np.nan, 81, 70.5617748] * u.deg,
        atol=1e-7,
    )
    assert_allclose(
        stereo["disp_az"].quantity,
        [359.9419634, np.nan, 14.5, 359.84586059] * u.deg,
        atol=1e-7,
    )
    tel_ids = stereo["disp_telescopes"]
    assert_array_equal(tel_ids[0], [1, 3])
    assert_array_equal(tel_ids[1], [])
    assert_array_equal(tel_ids[2], [1])
    assert_array_equal(tel_ids[3], [1, 3, 4, 5])


def test_predict_disp_combiner_empty_table(mono_table):
    disp_combiner = StereoDispCombiner(
        prefix="disp",
    )
    empty_table = mono_table[:0]
    stereo = disp_combiner.predict_table(empty_table)
    cols = [
        "obs_id",
        "event_id",
        "disp_alt",
        "disp_az",
        "disp_is_valid",
        "disp_telescopes",
    ]

    for col in cols:
        assert col in stereo.colnames
    assert len(stereo) == 0


def test_predict_disp_combiner_missing_disp_column(mono_table):
    disp_combiner = StereoDispCombiner(
        prefix="disp",
    )
    broken_table = mono_table.copy()
    del broken_table["disp_tel_parameter"]
    with pytest.raises(KeyError, match="Required DISP column"):
        disp_combiner.predict_table(broken_table)
