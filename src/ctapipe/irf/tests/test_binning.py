import logging

import astropy.units as u
import numpy as np
import pytest


def test_check_bins_in_range(tmp_path):
    from ctapipe.irf import ResultValidRange, check_bins_in_range

    valid_range = ResultValidRange(min=0.03 * u.TeV, max=200 * u.TeV)
    errormessage = "Valid range for result is 0.03 to 200., got"

    # bins are in range
    bins = u.Quantity(np.logspace(-1, 2, 10), u.TeV)
    check_bins_in_range(bins, valid_range)

    # bins are too small
    bins = u.Quantity(np.logspace(-2, 2, 10), u.TeV)
    with pytest.raises(ValueError, match=errormessage):
        check_bins_in_range(bins, valid_range)

    # bins are too big
    bins = u.Quantity(np.logspace(-1, 3, 10), u.TeV)
    with pytest.raises(ValueError, match=errormessage):
        check_bins_in_range(bins, valid_range)

    # bins are too big and too small
    bins = u.Quantity(np.logspace(-2, 3, 10), u.TeV)
    with pytest.raises(ValueError, match=errormessage):
        check_bins_in_range(bins, valid_range)

    logger = logging.getLogger("ctapipe.irf.binning")
    logpath = tmp_path / "test_check_bins_in_range.log"
    handler = logging.FileHandler(logpath)
    logger.addHandler(handler)

    check_bins_in_range(bins, valid_range, raise_error=False)
    assert "Valid range for result is" in logpath.read_text()


def test_make_bins_per_decade():
    from ctapipe.irf import make_bins_per_decade

    bins = make_bins_per_decade(100 * u.GeV, 100 * u.TeV)
    assert bins.unit == u.GeV
    assert len(bins) == 16
    assert bins[0] == 100 * u.GeV
    assert np.allclose(np.diff(np.log10(bins.to_value(u.GeV))), 0.2)

    bins = make_bins_per_decade(100 * u.GeV, 100 * u.TeV, 10)
    assert len(bins) == 31
    assert np.allclose(np.diff(np.log10(bins.to_value(u.GeV))), 0.1)

    # respect boundaries over n_bins_per_decade
    bins = make_bins_per_decade(100 * u.GeV, 105 * u.TeV)
    assert len(bins) == 17
    assert np.isclose(bins[-1], 105 * u.TeV, rtol=1e-9)


def test_true_energy_bins_base():
    from ctapipe.irf.binning import DefaultTrueEnergyBins

    binning = DefaultTrueEnergyBins(
        true_energy_min=0.02 * u.TeV,
        true_energy_max=200 * u.TeV,
        true_energy_n_bins_per_decade=7,
    )
    assert len(binning.true_energy_bins) == 29
    assert binning.true_energy_bins.unit == u.TeV
    assert np.isclose(binning.true_energy_bins[0], binning.true_energy_min, rtol=1e-9)
    assert np.isclose(binning.true_energy_bins[-1], binning.true_energy_max, rtol=1e-9)
    assert np.allclose(
        np.diff(np.log10(binning.true_energy_bins.to_value(u.TeV))), 1 / 7
    )


def test_reco_energy_bins_base():
    from ctapipe.irf.binning import DefaultRecoEnergyBins

    binning = DefaultRecoEnergyBins(
        reco_energy_min=0.02 * u.TeV,
        reco_energy_max=200 * u.TeV,
        reco_energy_n_bins_per_decade=4,
    )
    assert len(binning.reco_energy_bins) == 17
    assert binning.reco_energy_bins.unit == u.TeV
    assert np.isclose(binning.reco_energy_bins[0], binning.reco_energy_min, rtol=1e-9)
    assert np.isclose(binning.reco_energy_bins[-1], binning.reco_energy_max, rtol=1e-9)
    assert np.allclose(
        np.diff(np.log10(binning.reco_energy_bins.to_value(u.TeV))), 0.25
    )


def test_fov_offset_bins_base():
    from ctapipe.irf.binning import DefaultFoVOffsetBins

    binning = DefaultFoVOffsetBins(
        # use default for fov_offset_min
        fov_offset_max=3 * u.deg,
        fov_offset_n_bins=3,
    )
    assert len(binning.fov_offset_bins) == 4
    assert binning.fov_offset_bins.unit == u.deg
    assert np.isclose(binning.fov_offset_bins[0], binning.fov_offset_min, rtol=1e-9)
    assert np.isclose(binning.fov_offset_bins[-1], binning.fov_offset_max, rtol=1e-9)
    assert np.allclose(np.diff(binning.fov_offset_bins.to_value(u.deg)), 1)


def test_fov_phi_bins_base():
    from ctapipe.irf.binning import DefaultFoVPhiBins

    binning = DefaultFoVPhiBins(fov_offset_n_bins=4)
    assert len(binning.fov_offset_bins) == 5
    assert binning.fov_offset_bins.unit == u.deg
    assert np.allclose(
        binning.fov_offset_bins, [0.0, 90.0, 180.0, 270.0, 360.0] * u.deg
    )
