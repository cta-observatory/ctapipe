from ctapipe.analysis.camera.charge_resolution import \
    ChargeResolutionCalculator
import numpy as np
from numpy.testing import assert_almost_equal


def test_add():
    chargeres = ChargeResolutionCalculator()
    true_charge = np.arange(100)
    measured_charge = np.arange(100)
    chargeres.add(0, true_charge, measured_charge)
    assert len(chargeres._df_list) == 1
    assert chargeres._df_list[0].index.size == 100


def test_amalgamate():
    chargeres = ChargeResolutionCalculator()
    true_charge = np.arange(100)
    measured_charge = np.arange(100)
    chargeres.add(0, true_charge, measured_charge)
    chargeres.add(0, true_charge, measured_charge)
    chargeres.add(1, true_charge, measured_charge)
    assert len(chargeres._df_list) == 3
    chargeres._amalgamate()
    assert len(chargeres._df_list) == 0
    assert chargeres._df.index.size == 200


def test_memory_limit():
    chargeres = ChargeResolutionCalculator()
    chargeres._max_bytes = 1
    true_charge = np.arange(100)
    measured_charge = np.arange(100)
    chargeres.add(0, true_charge, measured_charge)
    assert len(chargeres._df_list) == 0
    chargeres.add(0, true_charge, measured_charge)
    assert len(chargeres._df_list) == 0


def test_finish():
    chargeres = ChargeResolutionCalculator()
    true_charge = np.arange(100)
    measured_charge = np.arange(100)
    chargeres.add(0, true_charge, measured_charge)
    chargeres.add(0, true_charge, measured_charge)
    df_p, df_c = chargeres.finish()
    assert np.array_equal(df_p['charge_resolution'].values,
                          df_c['charge_resolution'].values)
    chargeres.add(1, true_charge, measured_charge)
    df_p, df_c = chargeres.finish()
    assert not np.array_equal(df_p['charge_resolution'].values,
                              df_c['charge_resolution'].values)


def test_calculation():
    chargeres = ChargeResolutionCalculator()
    measured = np.array([3.5, 2.7])
    true = 3
    n = measured.size

    sum_ = np.sum(np.power(measured - true, 2))
    assert_almost_equal(sum_, 0.34, 3)
    assert_almost_equal(chargeres.rmse_abs(sum_, n), 0.412, 3)
    assert_almost_equal(chargeres.rmse(true, sum_, n), 0.137, 3)
    assert_almost_equal(chargeres.charge_res_abs(true, sum_, n), 1.780, 3)
    assert_almost_equal(chargeres.charge_res(true, sum_, n), 0.593, 3)

    assert chargeres.rmse_abs(sum_, n) == chargeres.rmse(true, sum_, n) * true
    assert (chargeres.charge_res_abs(true, sum_, n) ==
            chargeres.charge_res(true, sum_, n) * true)


def test_result():
    chargeres = ChargeResolutionCalculator(mc_true=False)
    measured = np.array([3.5, 2.7])
    true = 3
    n = measured.size
    sum_ = np.sum(np.power(measured - true, 2))

    chargeres.add(0, true, measured)
    df_p, df_c = chargeres.finish()
    assert (df_p['charge_resolution'].values[0] ==
            chargeres.rmse(true, sum_, n))
    assert (df_p['charge_resolution_abs'].values[0] ==
            chargeres.rmse_abs(sum_, n))


def test_result_mc_true():
    chargeres = ChargeResolutionCalculator()
    measured = np.array([3.5, 2.7])
    true = 3
    n = measured.size
    sum_ = np.sum(np.power(measured - true, 2))

    chargeres.add(0, true, measured)
    df_p, df_c = chargeres.finish()
    assert (df_p['charge_resolution'].values[0] ==
            chargeres.charge_res(true, sum_, n))
    assert (df_p['charge_resolution_abs'].values[0] ==
            chargeres.charge_res_abs(true, sum_, n))


# def test_get_charge_resolution():
#     chargeres = ChargeResolutionCalculator()
#     true_charge = np.arange(100)
#     measured_charge = np.arange(100)
#     chargeres.add_charges(true_charge, measured_charge)
#     true_charge, chargeres, chargeres_error, \
#         scaled_chargeres, scaled_chargeres_error = \
#         chargeres.get_charge_resolution()
#
#     assert(true_charge[0] == 1)
#     assert(chargeres[0] == 1)
#     assert(round(chargeres_error[0], 7) == 0.7071068)
#     assert(round(scaled_chargeres[0], 7) == 0.5550272)
#     assert(round(scaled_chargeres_error[0], 7) == 0.3924635)
#
#
# def test_get_binned_charge_resolution():
#     chargeres = ChargeResolutionCalculator()
#     true_charge = np.arange(100)
#     measured_charge = np.arange(100)
#     chargeres.add_charges(true_charge, measured_charge)
#     true_charge, chargeres, chargeres_error, \
#         scaled_chargeres, scaled_chargeres_error = \
#         chargeres.get_charge_resolution()
#
#     assert(true_charge[0] == 1)
#     assert(chargeres[0] == 1)
#     assert(round(chargeres_error[0], 7) == 0.7071068)
#     assert(round(scaled_chargeres[0], 7) == 0.5550272)
#     assert(round(scaled_chargeres_error[0], 7) == 0.3924635)
#
#
# def test_limit_curves():
#     value = ChargeResolutionCalculator.limit_curves(1, 2, 3, 4, 5)
#     assert(round(value, 7) == 6.78233)
#
#
# def test_requirement():
#     value = ChargeResolutionCalculator.requirement(1)
#     value2 = ChargeResolutionCalculator.requirement(np.arange(1, 10))[-1]
#     assert(round(value, 7) == 2.0237963)
#     assert(round(value2, 7) == 0.4501817)
#
#
# def test_goal():
#     value = ChargeResolutionCalculator.goal(1)
#     value2 = ChargeResolutionCalculator.goal(np.arange(1, 10))[-1]
#     assert(round(value, 7) == 1.8017134)
#     assert(round(value2, 7) == 0.4066657)
