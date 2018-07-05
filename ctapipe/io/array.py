# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Array layout utilities
"""
from astropy.table import Table

from ctapipe.utils import get_dataset_path

_telclass_map = {0: 'SST', 1: 'MST', 2: 'LST'}


def get_array_layout(instrument_name):
    """
    Returns the array layout for the given instrument as an
    `astropy.table.Table` object. 
    """
    name = instrument_name.lower()
    try:
        layoutfile = get_dataset_path('{}_arraylayout.fits'.format(name))
    except KeyError:
        layoutfile = get_dataset_path('{}_arraylayout.fits.gz'.format(name))
    return load_array_layout_from_file(layoutfile)


def load_array_layout_from_file(array_layout_filename):
    """
    Read an array layout from a FITS file with a ``TELARRAY`` extension
    """
    return Table.read(array_layout_filename, hdu='TELARRAY')


def tel_class_name(tel_class):
    """Convert telescope class number to a string.
    """
    return _telclass_map[int(tel_class)]
