import logging
from collections import defaultdict
from ctapipe.utils.datasets import get_dataset
from astropy.table import Table
from scipy.interpolate import interp1d
import numpy as np

from .camera import CameraGeometry

__all__ = ['get_atmosphere_profile_table', 'get_atmosphere_profile_functions']

log = logging.getLogger(__name__)



def get_camera_types(inst):
    """ return dict of camera names mapped to a list of tel_ids
     that use that camera
     
     Parameters
     ----------
     inst: instument Container
     
     """

    camid = defaultdict(list)

    for telid in inst.pixel_pos:
        x, y = inst.pixel_pos[telid]
        f = inst.optical_foclen[telid]
        geom = CameraGeometry.guess(x, y, f)

        camid[geom.cam_id].append(telid)

    return camid


def print_camera_types(inst, printer=log.info):
    camtypes = get_camera_types(inst)

    printer("              CAMERA  Num IDmin  IDmax")
    printer("=====================================")
    for cam, tels in camtypes.items():
        printer("{:>20s} {:4d} {:4d} ..{:4d}".format(cam, len(tels), min(tels),
                                                     max(tels)))


def get_atmosphere_profile_table(atmosphere_name='paranal'):
    return Table.read(get_dataset('{}.atmprof.fits.gz'.format(atmosphere_name)))


def get_atmosphere_profile_functions(atmosphere_name="paranal"):
    """ 
    returns the atmospheric profile as a continuous function thickness(
    altitude), and it's inverse altitude(thickness)  in m and g/cm^2
    """
    tab = get_atmosphere_profile_table(atmosphere_name)
    alt = tab['altitude'].to('m')
    thick = (tab['thickness']).to("g cm-2")

    alt_to_thickness = interp1d(x=np.array(alt), y=np.array(thick))
    thickness_to_alt = interp1d(x=np.array(thick), y=np.array(alt))


    return alt_to_thickness, thickness_to_alt