"""Example how to make a mock shower image and plot it.
"""
import matplotlib.pyplot as plt
from ctapipe.io.camera import make_rectangular_camera_geometry
from ctapipe.image.mock import generate_2d_shower_model, make_mock_shower_image

NX = 40
NY = 40

geom = make_rectangular_camera_geometry(NX, NY)

showermodel = generate_2d_shower_model(centroid=[0.25, 0.0], length=0.1,
                                       width=0.02, psi='40d')

image, signal, noise = make_mock_shower_image(geom, showermodel.pdf,
                                              intensity=20, nsb_level_pe=30)

# make them into 2D arrays so we can plot them with imshow
image.shape = (NX, NY)
signal.shape = (NX, NY)
noise.shape = (NX, NY)

# here we just plot the images using imshow().  For a more general
# case, one should use a ctapipe.visualization.CameraDisplay
plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.imshow(signal, interpolation='nearest', origin='lower')
plt.title("Signal")
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(noise, interpolation='nearest', origin='lower')
plt.title("Noise")
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(image, interpolation='nearest', origin='lower')
plt.title("Image")
plt.colorbar()

plt.show()
