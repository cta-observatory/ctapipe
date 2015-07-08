"""
Demo to show multiple shower images on a single figure using
`CameraDisplay` and really simple mock shower images (not
simulations). Also shows how to change the color palette. 
"""

import matplotlib.pylab as plt
from ctapipe import io, visualization
from ctapipe.reco import mock
from ctapipe.reco.hillas import hillas_parameters_2 as hillas_parameters
import numpy as np


def draw_several_cams(geom):

    ncams = 4
    cmaps = [plt.cm.jet, plt.cm.afmhot, plt.cm.terrain, plt.cm.autumn]
    fig, ax = plt.subplots(1, ncams, figsize=(15, 4), sharey=True, sharex=True)

    for ii in range(ncams):
        disp = visualization.CameraDisplay(geom, axes=ax[ii],
                                           title="CT{}".format(ii + 1))

        model = mock.generate_2d_shower_model(centroid=(0.2 - ii * 0.1,
                                                        -ii * 0.05),
                                              width=0.005 + 0.001 * ii,
                                              length=0.1 + 0.05 * ii,
                                              psi=np.radians(ii * 20))
        image, _, _ = mock.make_mock_shower_image(geom, model.pdf,
                                                  intensity=50,
                                                  nsb_level_pe=1000)

        clean = image.copy()
        clean[image <= 3.0 * image.mean()] = 0.0
        hillas = hillas_parameters(geom.pix_x.value, geom.pix_y.value, clean)

        disp.set_cmap(cmaps[ii])
        disp.set_image(image)
        disp.overlay_moments(hillas, linewidth=3, color='blue')


if __name__ == '__main__':

    hexgeom = io.get_camera_geometry("hess", 1)
    recgeom = io.make_rectangular_camera_geometry()

    draw_several_cams(recgeom)
    draw_several_cams(hexgeom)

    plt.tight_layout()
    plt.show()
