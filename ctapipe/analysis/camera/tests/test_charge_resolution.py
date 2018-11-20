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
