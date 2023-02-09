from matplotlib import pyplot as plt

from ctapipe.instrument import SubarrayDescription

subarray = SubarrayDescription.read("dataset://gamma_prod5.simtel.zst")
geom = subarray.tel[1].camera.geometry

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.imshow(geom.neighbor_matrix, origin="lower")
plt.title("Pixel Neighbor Matrix")

plt.subplot(1, 2, 2)
plt.scatter(geom.pix_x, geom.pix_y)
plt.title("Pixel Positions")
