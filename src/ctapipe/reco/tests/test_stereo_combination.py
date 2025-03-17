import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from numpy.testing import assert_allclose, assert_array_equal

from ctapipe.containers import (
    ArrayEventContainer,
    DispContainer,
    HillasParametersContainer,
    ImageParametersContainer,
    ParticleClassificationContainer,
    PointingContainer,
    ReconstructedContainer,
    ReconstructedEnergyContainer,
    ReconstructedGeometryContainer,
    TelescopeReconstructedContainer,
)
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
            "hillas_intensity": [1, 2, 0, 1, 5, 9, 1, 2, 1, 2],
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
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            "disp_tel_parameter": [0.65, 1.1, 0.7, 0.9, 1, 0.5, 0.65, 1.1, 0.7, 0.9]
            * u.deg,
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
        assert_array_equal(stereo["dummy_energy"], [7, 0.5, np.nan, 4] * u.TeV)
        assert_allclose(
            stereo["dummy_energy_uncert"].quantity,
            [4.242641, 0, np.nan, 3.7305049] * u.TeV,
            atol=1e-7,
        )
    elif weights == "none":
        assert_array_equal(stereo["dummy_energy"], [5, 0.5, np.nan, 3.625] * u.TeV)
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
        [60.5002328, 73.2505989, 81, 62.773741] * u.deg,
        atol=1e-7,
    )
    assert_allclose(
        stereo["disp_az"].quantity,
        [12.7345693, 20.5362510, 14.5, 14.792156] * u.deg,
        atol=1e-7,
    )
    tel_ids = stereo["disp_telescopes"]
    assert_array_equal(tel_ids[0], [1, 3])
    assert_array_equal(tel_ids[1], [5, 7])
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
            )
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


def test_predict_disp_combiner(mono_table):
    disp_combiner = StereoDispCombiner(
        prefix="disp",
        property=ReconstructionProperty.GEOMETRY,
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
        [70.7338725, 69.9550623, 81, 70.4917615] * u.deg,
        atol=1e-7,
    )
    assert_allclose(
        stereo["disp_az"].quantity,
        [359.9419634, 359.8805054, 14.5, 359.5978866] * u.deg,
        atol=1e-7,
    )
    tel_ids = stereo["disp_telescopes"]
    assert_array_equal(tel_ids[0], [1, 3])
    assert_array_equal(tel_ids[1], [5, 7])
    assert_array_equal(tel_ids[2], [1])
    assert_array_equal(tel_ids[3], [1, 3, 4, 5])


@pytest.mark.parametrize("weights", ["konrad", "intensity", "none"])
def test_disp_combiner_single_event(weights):
    event = ArrayEventContainer()

    event_dict = {
        "tel_id": [1, 2, 9],
        "hillas_intensity": [100, 200, 50],
        "hillas_width": [0.1, 0.2, 0.1] * u.deg,
        "hillas_length": 3 * ([0.1, 0.2, 0.1] * u.deg),
        "hillas_fov_lon": [-0.5, 0, 0.5] * u.deg,
        "hillas_fov_lat": [0.3, -0.3, 0.3] * u.deg,
        "hillas_psi": [40, 85, -40] * u.deg,
        "disp_tel_alt": [58.5, 58, 62.5] * u.deg,
        "disp_tel_az": [12.5, 15, 13] * u.deg,
        "disp_tel_parameter": [0.65, 1.1, 0.7] * u.deg,
    }

    for i in range(3):
        event.dl1.tel[event_dict["tel_id"][i]].parameters = ImageParametersContainer(
            hillas=HillasParametersContainer(
                intensity=event_dict["hillas_intensity"][i],
                fov_lon=event_dict["hillas_fov_lon"][i],
                fov_lat=event_dict["hillas_fov_lat"][i],
                psi=event_dict["hillas_psi"][i],
                width=event_dict["hillas_width"][i],
                length=event_dict["hillas_length"][i],
            )
        )

        event.dl2.tel[event_dict["tel_id"][i]] = TelescopeReconstructedContainer(
            disp={
                "dummy": DispContainer(parameter=event_dict["disp_tel_parameter"][i])
            },
            geometry={
                "dummy": ReconstructedGeometryContainer(
                    alt=event_dict["disp_tel_alt"][i],
                    az=event_dict["disp_tel_az"][i],
                    is_valid=True,
                )
            },
        )

    event.pointing = PointingContainer(
        array_azimuth=0 * u.deg, array_altitude=70 * u.deg
    )

    disp_combiner = StereoDispCombiner(
        prefix="dummy",
        property=ReconstructionProperty.GEOMETRY,
        weights=weights,
    )
    disp_combiner(event)
    if weights in ["intensity", "konrad"]:
        assert u.isclose(event.dl2.stereo.geometry["dummy"].alt, 70.76691618 * u.deg)
        assert u.isclose(event.dl2.stereo.geometry["dummy"].az, 0.1487853 * u.deg)
    elif weights == "none":
        assert u.isclose(event.dl2.stereo.geometry["dummy"].alt, 70.75451665 * u.deg)
        assert u.isclose(event.dl2.stereo.geometry["dummy"].az, 0.05821327 * u.deg)
