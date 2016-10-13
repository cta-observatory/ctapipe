import matplotlib.pyplot as plt
from ctapipe import io, visualization
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.reco import mock

from ..hillas import hillas_parameters

"""
Test script for hillas_parameters.

DESCRIPTION:
------------
This is a very raw script for end to end test. It generates a 2D shower model in the camera, applies a basic two-level tailcuts cleaning and calculates hillas parameters from the image.

hillas_1 and hillas_2 are just the 'MomentParameters' and 'HighOrderMomentParameters' respectively.

TODO:
-----
Setup a format for proper pytest.

"""


if __name__ == '__main__':

  # Prepare the camera geometry
  geom = io.CameraGeometry.from_name('hess', 1)
  disp = visualization.CameraDisplay(geom)
  disp.set_limits_minmax(0, 350)
  disp.add_colorbar()

  # make a mock shower model
  model = mock.generate_2d_shower_model(centroid=(0.2, 0.2), width=0.01, length=0.1, psi='45d')

  # generate mock image in camera for this shower model.
  image, signal, noise = mock.make_mock_shower_image(geom, model.pdf, intensity=50, nsb_level_pe=100)

  #Image cleaning
  clean_mask = tailcuts_clean(geom, image, 1, 10, 5)      #pedvars = 1 and core and boundary threshold in pe
  image[~clean_mask] = 0


  #Pixel values in the camera
  pix_x = geom.pix_x.value
  pix_y = geom.pix_y.value

  # Hillas parameters
  hillas1, hillas2 = hillas_parameters(pix_x, pix_y, image)
  print(hillas1, hillas2)

  #Overlay moments
  disp.image = image
  disp.overlay_moments(hillas1, color = 'seagreen', linewidth = 2)
  plt.show()
