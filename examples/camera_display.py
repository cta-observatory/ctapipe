"""
Example of drawing a Camera using a mock shower image
"""

import matplotlib.pylab as plt
from ctapipe import io, visualization
from ctapipe.reco import mock
from ctapipe.utils.datasets import get_path
import numpy as np

geom = io.load_camera_geometry(1, geomfile=get_path("chercam.fits.gz"))

disp = visualization.CameraDisplay(geom)

model = mock.shower_model(centroid=(0.2, 0.0), width=0.01,
                          length=0.1, phi=np.radians(35))
image, _, _ = mock.make_mock_shower_image(geom, model, intensity=50,
                                          nsb_level_pe=1000)


disp.draw_image(image)


# show several cameras:

ncams = 4

fig, ax = plt.subplots(1, ncams, figsize=(15, 5))

for ii in range(ncams):
    disp = visualization.CameraDisplay(geom, axes=ax[ii],
                                       title="CT{}".format(ii + 1))
    model = mock.shower_model(centroid=(0.2 - ii * 0.1, -ii * 0.05), width=0.005,
                              length=0.1, phi=np.radians(ii * 20))
    image, _, _ = mock.make_mock_shower_image(geom, model, intensity=50,
                                              nsb_level_pe=1000)
    disp.draw_image(image)


plt.show()
