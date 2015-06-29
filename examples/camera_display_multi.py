
import matplotlib.pylab as plt
from ctapipe import io, visualization
from ctapipe.reco import mock
from ctapipe.utils.datasets import get_path
from ctapipe.reco import hillas_parameters
import numpy as np

geom = io.get_camera_geometry("hess", 1)

# show several camera images:
ncams = 4
cmaps = [plt.cm.jet, plt.cm.afmhot, plt.cm.terrain, plt.cm.autumn]

fig, ax = plt.subplots(1, ncams, figsize=(15, 5),sharey=True, sharex=True)

for ii in range(ncams):
    disp = visualization.CameraDisplay(geom, axes=ax[ii],
                                       title="CT{}".format(ii + 1))
    model = mock.shower_model(centroid=(0.2 - ii * 0.1, -ii * 0.05),
                              width=0.005,
                              length=0.1, phi=np.radians(ii * 20))

    image, _, _ = mock.make_mock_shower_image(geom, model, intensity=50,
                                              nsb_level_pe=1000)
    disp.set_cmap(cmaps[ii])
    disp.set_image(image)

plt.tight_layout()
plt.show()
