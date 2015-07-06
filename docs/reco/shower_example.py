"""Example how to make a mock shower image and plot it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab as plt
from ctapipe.io.camera import make_rectangular_camera_geometry
from ctapipe.reco import shower_model, make_mock_shower_image

geom = make_rectangular_camera_geometry()

showermodel = shower_model(centroid=[0.25, 0.0], length=0.1,
                           width=0.02, psi=np.radians(40))

image, signal, noise = make_mock_shower_image(geom, showermodel.pdf,
                                              intensity=20, nsb_level_pe=30)

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.imshow(signal, interpolation='none')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(noise, interpolation='none')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(image, interpolation='none')
plt.colorbar()
