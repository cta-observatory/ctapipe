# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from astropy.table import Table
from ..hillas import hillas_parameters
from ...utils.datasets import get_path


def test_hillas_parameters():
    filename = get_path('hess_camgeom.fits.gz')
    # TODO: this test currently doesn't make sense ...
    # it's just to show how to access test files
    table = Table.read(filename, format='fits')
    x = table['PIX_POSX']
    y = table['PIX_POSY']
    assert_allclose(x.sum(), 5.486607551574707e-05)
