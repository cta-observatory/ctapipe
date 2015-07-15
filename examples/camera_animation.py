

"""
Example of drawing a Camera using a mock shower image
"""

import matplotlib.pylab as plt
from ctapipe import io, visualization
from ctapipe.reco import mock
from ctapipe.reco import hillas_parameters_2 as hillas_parameters
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib import pylab as plt

if __name__ == '__main__':

    plt.style.use("ggplot")
    fig, ax = plt.subplots()

    # load the camera
    geom = io.get_camera_geometry("hess", 1)
    disp = visualization.CameraDisplay(geom, axes=ax)
    disp.set_cmap(plt.cm.terrain)

    def update(frame):
        centroid = np.random.uniform(-0.5, 0.5, size=2)
        width = np.random.uniform(0, 0.01)
        length = np.random.uniform(0, 0.03) + width
        angle = np.random.uniform(0, 360)
        intens = np.random.uniform(20, 100)
        model = mock.generate_2d_shower_model(centroid=centroid,
                                              width=width,
                                              length=length,
                                              psi=np.radians(angle))
        image, sig, bg = mock.make_mock_shower_image(geom, model.pdf,
                                                     intensity=intens,
                                                     nsb_level_pe=1000)
        image /= image.max()
        disp.set_image(image)

    anim = FuncAnimation(fig, update, interval=50)
    plt.show()
