# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest
from ctapipe.utils.fitshistogram import Histogram


def compare_histograms(hist1: Histogram, hist2: Histogram):
    """ check that 2 histograms are identical in value """
    assert hist1.ndims == hist2.ndims
    assert (hist1.axis_names == hist2.axis_names).all()
    assert (hist1.data == hist2.data).all

    for ii in range(hist1.ndims):
        assert np.isclose(hist1.bin_lower_edges[ii],
                          hist2.bin_lower_edges[ii]).all()


def test_histogram_str():
    hist = Histogram(nbins=[5, 10],
                     ranges=[[-2.5, 2.5], [-1, 1]], name="testhisto")
    expected = ("Histogram(name='testhisto', axes=['axis0', 'axis1'], "
                "nbins=[ 5 10], ranges=[[-2.5  2.5]\n [-1.   1. ]])")
    assert str(hist) == expected


def test_histogram_fill_and_read():

    hist = Histogram(nbins=[5, 10], ranges=[[-2.5, 2.5], [-1, 1]])

    pa = (0.1, 0.1)
    pb = (-0.55, 0.55)

    a = np.ones((100, 2)) * pa  # point at 0.1,0.1
    b = np.ones((10, 2)) * pb  # 10 points at -0.5,0.5

    hist.fill(a)
    hist.fill(b)

    va = hist.get_value(pa)[0]
    vb = hist.get_value(pb)[0]

    assert va == 100
    assert vb == 10


def test_histogram_range_fill_and_read():
    """
    Check that the correct bin is read and written for multiple
    binnings and fill positions
    """

    num = 100

    for nxbins in np.arange(1, 50, 1):
        for xx in np.arange(-2.0, 2.0, 0.1):
            pp = (xx + 0.01829384, 0.1)
            coords = np.ones((num, 2)) * np.array(pp)
            hist = Histogram(nbins=[nxbins, 10],
                             ranges=[[-2.5, 2.5], [-1, 1]])
            hist.fill(coords)
            val = hist.get_value(pp)[0]
            assert val == num
            del hist


def test_outliers():
    """
    Check that out-of-range values work as expected
    """
    H = Histogram(nbins=[5, 10], ranges=[[-2.5, 2.5], [-1, 1]])
    H.fill(np.array([[1, 1], ]))
    val1 = H.get_value((100, 100), outlier_value=-10000)
    val2 = H.get_value((-100, 0), outlier_value=None)
    assert val1 == -10000
    assert val2 == 0


@pytest.fixture(scope='session')
def histogram_file(tmpdir_factory):
    """ a fixture that fetches a temporary output dir/file for a test
    histogram"""
    return str(tmpdir_factory.mktemp('data').join('histogram_test.fits'))


def test_histogram_fits(histogram_file):
    """
    Write to fits,read back, and check
    """

    hist = Histogram(nbins=[5, 11], ranges=[[-2.5, 2.5], [-1, 1]])
    hist.fill(np.array([[0, 0],
                        [0, 0.5]]))

    hist.to_fits().writeto(histogram_file, overwrite=True)
    newhist = Histogram.from_fits(histogram_file)

    # check that the values are the same
    compare_histograms(hist, newhist)


def test_histogram_resample_inplace():
    hist = Histogram(nbins=[5, 11], ranges=[[-2.5, 2.5], [-1, 1]])
    hist.fill(np.array([[0, 0],
                        [0, 0.5]]))

    for testpoint in [(0, 0), (0, 1), (1, 0), (3, 3)]:
        val0 = hist.get_value(testpoint)
        hist.resample_inplace((10, 22))
        hist.resample_inplace((5, 11))
        val2 = hist.get_value(testpoint)

        # at least check the resampling is undoable
        assert np.isclose(val0[0], val2[0])
