"""
low-level utility functions for dealing with data files
"""

import os


def get_file_type(filename):
    """
    Returns a string with the type of the given file (guessed from the
    extension). The '.gz' or '.bz2' compression extensions are
    ignored.

    >>> get_file_type('myfile.fits.gz')
    'fits'

    """
    root, ext = os.path.splitext(filename)
    if ext in ['.gz', '.bz2']:
        ext = os.path.splitext(root)[1]

    ext = ext[1:]  # strip off leading '.'

    # special cases:
    if ext in ['fit', 'FITS', 'FIT']:
        ext = 'fits'

    return ext
