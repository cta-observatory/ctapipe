from ctapipe.analysis.camera.charge_resolution import \
    ChargeResolutionCalculator
import os
import numpy as np
import pandas as pd
from ..charge_resolution import (
    sum_errors, bin_dataframe,
    ChargeResolutionPlotter, ChargeResolutionWRRPlotter
)
from numpy.testing import assert_almost_equal
import pytest


def create_temp_cr_file(directory):
    chargeres = ChargeResolutionCalculator()
    true_charge = np.arange(100)
    measured_charge = np.arange(100)
    chargeres.add(0, true_charge, measured_charge)
    chargeres.add(0, true_charge, measured_charge)
    df_p, df_c = chargeres.finish()

    output_path = os.path.join(str(directory), "cr.h5")
    with pd.HDFStore(output_path, 'w') as store:
        store['charge_resolution_pixel'] = df_p
        store['charge_resolution_camera'] = df_c
    return output_path


def test_sum_errors():
    errors = np.array([2, 5, 6, 7])
    assert_almost_equal(sum_errors(errors), 5.339, 3)


def test_bin_dataframe():
    chargeres = ChargeResolutionCalculator()
    true_charge = np.arange(100)
    measured_charge = np.arange(100)
    chargeres.add(0, true_charge, measured_charge)
    chargeres.add(0, true_charge, measured_charge)
    df_p, df_c = chargeres.finish()

    df = bin_dataframe(df_p, 20)
    assert 'bin' in df.columns
    assert np.unique(df['bin']).size <= 20


def test_file_reading(tmpdir):
    path = create_temp_cr_file(tmpdir)
    output_path = os.path.join(str(tmpdir), "cr.pdf")
    plotter = ChargeResolutionPlotter(output_path=output_path)
    plotter._set_file(path)
    assert plotter._df_pixel is not None
    assert plotter._df_camera is not None


def test_incorrect_file(tmpdir):
    path = os.path.join(str(tmpdir), "cr_incorrect.h5")
    output_path = os.path.join(str(tmpdir), "cr_incorrect.pdf")
    with pd.HDFStore(path, 'w') as store:
        store['test'] = pd.DataFrame(dict(a=[3]))

    plotter = ChargeResolutionPlotter(output_path=output_path)
    with pytest.raises(KeyError):
        plotter._set_file(path)


def test_missing_file(tmpdir):
    path = os.path.join(str(tmpdir), "cr_missing.h5")
    output_path = os.path.join(str(tmpdir), "cr_missing.pdf")

    assert not os.path.exists(path)

    plotter = ChargeResolutionPlotter(output_path=output_path)
    with pytest.raises(OSError):
        plotter._set_file(path)


def test_plotting(tmpdir):
    path = create_temp_cr_file(tmpdir)
    output_path = os.path.join(str(tmpdir), "cr.pdf")
    plotter = ChargeResolutionPlotter(output_path=output_path)
    plotter.plot_average(path, "average")
    plotter.plot_camera(path, "average")
    plotter.plot_pixel(path, 0, "average")
    q = np.arange(1, 1000)
    plotter.plot_requirement(q)
    plotter.plot_poisson(q)

    plotter.save()
    assert os.path.exists(output_path)


def test_limit_curves():
    value = ChargeResolutionPlotter.limit_curves(1, 2, 3, 4, 5, 6)
    assert_almost_equal(value, 9.798, 3)


def test_requirement():
    value = ChargeResolutionPlotter.requirement(np.arange(1, 10))[0]
    value2 = ChargeResolutionPlotter.requirement(np.arange(1, 10))[-1]
    assert_almost_equal(value, 2.020, 3)
    assert_almost_equal(value2, 0.450, 3)


def test_poisson():
    value = ChargeResolutionPlotter.poisson(np.arange(1, 10))[0]
    value2 = ChargeResolutionPlotter.poisson(np.arange(1, 10))[-1]
    assert_almost_equal(value, 1, 3)
    assert_almost_equal(value2, 0.333, 3)


def test_plotting_wrr(tmpdir):
    path = create_temp_cr_file(tmpdir)
    output_path = os.path.join(str(tmpdir), "cr_wrr.pdf")
    plotter = ChargeResolutionWRRPlotter(output_path=output_path)
    plotter.plot_average(path, "average")
    plotter.plot_camera(path, "average")
    plotter.plot_pixel(path, 0, "average")
    q = np.arange(1, 1000)
    plotter.plot_requirement(q)
    plotter.plot_poisson(q)

    plotter.save()
    assert os.path.exists(output_path)
