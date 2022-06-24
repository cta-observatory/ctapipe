import pytest
from astropy.table import Table
import astropy.units as u
import numpy as np
from numpy.testing import assert_array_equal
from ctapipe.containers import (
    ReconstructedEnergyContainer,
    ParticleClassificationContainer,
    ArrayEventContainer,
    ReconstructedContainer,
)
from ctapipe.io import EventSource
from ctapipe.ml.stereo_combination import StereoMeanCombiner


@pytest.fixture(scope="module")
def mono_table():
    """
    Dummy table of telescope events with a
    prediction and weights.
    """
    return Table(
        {
            "obs_id": [1, 1, 1, 1, 1, 2],
            "event_id": [1, 1, 1, 2, 2, 1],
            "tel_id": [1, 2, 3, 5, 7, 1],
            "hillas_intensity": [1, 2, 0, 1, 5, 9],
            "dummy_reconstructed_energy_energy": u.Quantity(
                [1, 10, 4, 0.5, 0.7, 1], u.TeV
            ),
            "dummy_reconstructed_energy_is_valid": [
                True,
                True,
                True,
                True,
                False,
                False,
            ],
            "classifier_particle_classification_prediction": [1, 0, 0.5, 0, 0.6, 1],
            "classifier_particle_classification_is_valid": [
                True,
                True,
                False,
                True,
                True,
                True,
            ],
        }
    )


def test_predict_mean_energy(mono_table):
    combine = StereoMeanCombiner(
        algorithm="dummy", combine_property="energy", weights="intensity"
    )
    stereo = combine.predict(mono_table)
    assert stereo.colnames == [
        "obs_id",
        "event_id",
        "dummy_reconstructed_energy_tel_ids",
        "dummy_reconstructed_energy_energy",
        "dummy_reconstructed_energy_energy_uncert",
        "dummy_reconstructed_energy_is_valid",
        "dummy_reconstructed_energy_goodness_of_fit",
    ]
    assert_array_equal(stereo["obs_id"], np.array([1, 1]))
    assert_array_equal(stereo["event_id"], np.array([1, 2]))
    assert_array_equal(
        stereo["dummy_reconstructed_energy_energy"],
        u.Quantity(np.array([7, 0.5]), u.TeV),
    )
    assert_array_equal(
        stereo["dummy_reconstructed_energy_tel_ids"][0], np.array([1, 2, 3])
    )
    assert_array_equal(stereo["dummy_reconstructed_energy_tel_ids"][1], 5)


def test_predict_mean_classification(mono_table):
    combine = StereoMeanCombiner(
        algorithm="classifier", combine_property="classification"
    )
    stereo = combine.predict(mono_table)
    assert stereo.colnames == [
        "obs_id",
        "event_id",
        "classifier_particle_classification_tel_ids",
        "classifier_particle_classification_prediction",
        "classifier_particle_classification_is_valid",
        "classifier_particle_classification_goodness_of_fit",
    ]
    assert_array_equal(stereo["obs_id"], np.array([1, 1, 2]))
    assert_array_equal(stereo["event_id"], np.array([1, 2, 1]))
    assert_array_equal(
        stereo["classifier_particle_classification_prediction"],
        [0.5, 0.3, 1],
    )
    assert_array_equal(
        stereo["classifier_particle_classification_tel_ids"][0], np.array([1, 2])
    )
    assert_array_equal(stereo["classifier_particle_classification_tel_ids"][1], [5, 7])
    assert_array_equal(stereo["classifier_particle_classification_tel_ids"][2], 1)


def test_mean_prediction_single_event():
    event = ArrayEventContainer()
    event.dl2.tel[25] = ReconstructedContainer(
        energy={
            "dummy": ReconstructedEnergyContainer(energy=10 * u.GeV, is_valid=True)
        },
        classification={
            "dummy": ParticleClassificationContainer(prediction=1.0, is_valid=True)
        },
    )
    event.dl2.tel[125] = ReconstructedContainer(
        energy={
            "dummy": ReconstructedEnergyContainer(energy=20 * u.GeV, is_valid=True)
        },
        classification={
            "dummy": ParticleClassificationContainer(prediction=0.0, is_valid=True)
        },
    )
    event.dl2.tel[130] = ReconstructedContainer(
        energy={
            "dummy": ReconstructedEnergyContainer(energy=0.03 * u.TeV, is_valid=True)
        },
        classification={
            "dummy": ParticleClassificationContainer(prediction=0.8, is_valid=True)
        },
    )

    combine_energy = StereoMeanCombiner(algorithm="dummy", combine_property="energy")
    combine_classification = StereoMeanCombiner(
        algorithm="dummy", combine_property="classification"
    )
    combine_energy(event)
    combine_classification(event)
    assert event.dl2.stereo.energy["dummy"].energy == 20 * u.GeV
    assert event.dl2.stereo.classification["dummy"].prediction == 0.6
