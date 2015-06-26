"""
Example of drawing a Camera using a mock shower image
"""

import matplotlib.pylab as plt
from ctapipe import io, visualization
from ctapipe.reco import mock
from ctapipe.utils.datasets import get_path
from ctapipe.reco import hillas_parameters
import numpy as np


# load the camera 
geom = io.get_camera_geometry("hess", 1)
disp = visualization.CameraDisplay(geom)

# create a fake camera image to display:
model = mock.shower_model(centroid=(0.2, 0.0),
                          width=0.01,
                          length=0.1,
                          phi=np.radians(35))

image, _, _ = mock.make_mock_shower_image(geom, model, intensity=50,
                                          nsb_level_pe=1000)

# apply really stupid image cleaning (single threshold):
clean = image.copy()
clean[image <= 3.0 * image.mean()] = 0.0

# calculate image parameters
hillas = hillas_parameters(geom.pix_x, geom.pix_y, clean)
print(hillas)

# show the camera image and overlay Hillas ellipse
disp.draw_image(image)
disp.add_moment_ellipse(centroid=(hillas['x'].value, hillas['y'].value),
                        length=hillas['length'].value,
                        width=hillas['width'].value, phi=np.radians(35),
                        edgecolor='white', linewidth=2)

plt.show()
