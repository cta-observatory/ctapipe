import logging
from collections import defaultdict
from ctapipe.utils.datasets import get_dataset

from .camera import CameraGeometry

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


def get_atmosphere_profile(site="paranal"):
    return get_dataset('atmprof_{}.dat'.format(site))