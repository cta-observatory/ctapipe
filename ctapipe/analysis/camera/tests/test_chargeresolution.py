from ctapipe.analysis.camera.chargeresolution import ChargeResolutionCalculator
import numpy as np


def test_add_charges():
    chargeres = ChargeResolutionCalculator()
    true_charge = np.arange(100)
    measured_charge = np.arange(100)
    chargeres.add_charges(true_charge, measured_charge)
    variation_hist = chargeres.variation_hist

    assert(variation_hist[0][0] == 1)


# def test_add_source():
    # Cannot currently test - no test file has true charge


def test_get_charge_resolution():
    chargeres = ChargeResolutionCalculator( binning=None)
    true_charge = np.arange(100)
    measured_charge = np.arange(100)
    chargeres.add_charges(true_charge, measured_charge)
    true_charge, chargeres, chargeres_error, \
        scaled_chargeres, scaled_chargeres_error = \
        chargeres.get_charge_resolution()

    assert(true_charge[0] == 1)
    assert(chargeres[0] == 1)
    assert(round(chargeres_error[0], 7) == 0.7071068)
    assert(round(scaled_chargeres[0], 7) == 0.5550272)
    assert(round(scaled_chargeres_error[0], 7) == 0.3924635)


def test_get_binned_charge_resolution():
    chargeres = ChargeResolutionCalculator()
    true_charge = np.arange(100)
    measured_charge = np.arange(100)
    chargeres.add_charges(true_charge, measured_charge)
    true_charge, chargeres, chargeres_error, \
        scaled_chargeres, scaled_chargeres_error = \
        chargeres.get_charge_resolution()

    assert(true_charge[0] == 1)
    assert(chargeres[0] == 1)
    assert(round(chargeres_error[0], 7) == 0.7071068)
    assert(round(scaled_chargeres[0], 7) == 0.5550272)
    assert(round(scaled_chargeres_error[0], 7) == 0.3924635)


def test_limit_curves():
    value = ChargeResolutionCalculator.limit_curves(1, 2, 3, 4, 5)
    assert(round(value, 7) == 6.78233)


def test_requirement():
    value = ChargeResolutionCalculator.requirement(1)
    value2 = ChargeResolutionCalculator.requirement(np.arange(1, 10))[-1]
    assert(round(value, 7) == 2.0237963)
    assert(round(value2, 7) == 0.4501817)


def test_goal():
    value = ChargeResolutionCalculator.goal(1)
    value2 = ChargeResolutionCalculator.goal(np.arange(1, 10))[-1]
    assert(round(value, 7) == 1.8017134)
    assert(round(value2, 7) == 0.4066657)
