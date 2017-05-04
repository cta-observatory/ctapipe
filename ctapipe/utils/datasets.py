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


def find_in_path(searchpath, filename):
    """
    Search in searchpath for filename
    
    Parameters
    ----------
    searchpath: str
        colon-separated list of directories (like PATH)
    filename:
        filename to find

    Returns
    -------
    full path to file if found, None otherwise

    """

    for dir in os.path.expandvars(searchpath).split(':'):
        pathname = os.path.join(dir, filename)
        if os.path.exists(pathname):
            return pathname

    return None




def get_dataset(filename):
    """
    Returns the full file path to an auxiliary dataset needed by 
    ctapipe.
      
    This will first search for the file in directories listed in 
    tne environment variable CTAPIPE_SVC_PATH (if set), and if not found,  
    will look in the ctapipe_resources module 
    (installed with the ctapipe-extra package), which contains the defaults.
    
    Parameters
    ----------
    filename: str
        name of dataset to fetch

    Returns
    -------
    string with full path to the given dataset
    """
    searchpath = os.getenv("CTAPIPE_SVC_PATH")

    if searchpath:
        filepath = find_in_path(searchpath, filename)
        if filepath:
            return filepath

    return ctapipe_resources.get(filename)

@deprecated("ctapipe-0.5",alternative='get_dataset()')
def get_path(filename):
    return get_dataset(filename)

