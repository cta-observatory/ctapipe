from copy import deepcopy
import pytest
import numpy as np
from matplotlib import pyplot as plt

from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.geometry_converter import convert_geometry_1d_to_2d, \
    convert_geometry_back
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.instrument import CameraGeometry
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset
from ctapipe.visualization import CameraDisplay
from ctapipe.image.toymodel import generate_2d_shower_model, make_toymodel_shower_image
from astropy import units as u

def test_convert_geometry():

    cam_ids = CameraGeometry.get_known_camera_names()
    cam_ids = ['FlashCam', 'NectarCam','Whipple151','DigiCam']

    model = generate_2d_shower_model(centroid=(0.4,0), width=0.01, length=0.03,
                                     psi="25d")

    for cam_id in cam_ids:
        geom = CameraGeometry.from_name(cam_id)
        if geom.pix_type=='rectangular':
            continue

        _,image,_ = make_toymodel_shower_image(geom, model.pdf,
                                                   intensity=50,
                                           nsb_level_pe=100)
        hillas_0 = hillas_parameters(geom.pix_x, geom.pix_y, image)

        geom2d, image2d = convert_geometry_1d_to_2d(geom, image,
                                                     geom.cam_id,
                                                     add_rot=-2)
        geom1d, image1d = convert_geometry_back(geom2d, image2d,
                                                 geom.cam_id,
                                                 add_rot=4)
        hillas_1 = hillas_parameters(geom1d.pix_x, geom1d.pix_y, image1d)

        if __name__ == "__main__":
            plt.viridis()
            plt.figure(figsize=(12,3))
            ax = plt.subplot(1,3,1)
            CameraDisplay(geom, image=image)
            plt.subplot(1,3,2, sharex=ax, sharey=ax)
            CameraDisplay(geom2d, image=image2d)
            plt.subplot(1,3,3, sharex=ax, sharey=ax)
            CameraDisplay(geom1d, image=image1d)
        else:
            assert np.abs(hillas_1.width - hillas_0.width).value < 1e-4


if __name__ == "__main__":
    #test_convert_geometry_from_simtel()
    import logging
    logging.basicConfig(level=logging.DEBUG)
    test_convert_geometry()
