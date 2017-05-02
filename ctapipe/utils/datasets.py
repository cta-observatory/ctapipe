# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from pathlib import Path
from astropy.utils.decorators import deprecated

try:
    import ctapipe_resources
except:
    raise RuntimeError("Please install the 'ctapipe-extra' package, "
                       "which contains the ctapipe_resources module "
                       "needed by ctapipe. (conda install ctapipe-extra)")


__all__ = ['get_dataset'  ]


def get_dataset(filename):
    """
    Returns the full file path to an auxiliary dataset needed by ctapipe 
    
    Parameters
    ----------
    filename: str
        name of dataset to fetch

    Returns
    -------
    string with full path to the given dataset
    """
    return ctapipe_resources.get(filename)

@deprecated("ctapipe-0.4.1",alternative='get_dataset()')
def get_path(filename):
    return ctapipe_resources.get(filename)

