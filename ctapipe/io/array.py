"""
Array Layout utilities
"""

from astropy.table import Table
from collections import namedtuple
from ctapipe.utils.datasets import get_path
from glob import glob

_telclass_map = {0: 'SST', 1: 'MST', 2: 'LST'}

def get_array_layout(instrument_name):
    """
    Returns the array laout for the given instrument as an
    `astropy.table.Table` object. 
    """
    name = instrument_name.lower()
    layoutfile = glob(get_path('{}_arraylayout.fits*'.format(name)))[0]
    return load_array_layout_from_file(layoutfile)


def load_array_layout_from_file(array_layout_filename):
    """
    Read an array layout from a FITS file with a TELARRAY extension
    """
    return Table.read(array_layout_filename, hdu='TELARRAY')


def tel_class_name(tel_class):
    """ convert telescope class number to a string """
    return _telclass_map[int(tel_class)]
