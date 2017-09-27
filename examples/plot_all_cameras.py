from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':

    plt.style.use("bmh")

    N = 3
    M = 6
    plt.figure(figsize=(15, 10))

    camera_names = CameraGeometry.get_known_camera_names()

    for ii, name in enumerate(sorted(camera_names)):
        print("plotting", name)
        geom = CameraGeometry.from_name(name)
        ax = plt.subplot(N, M, ii + 1)
        disp = CameraDisplay(geom)
        disp.image = np.random.uniform(size=geom.pix_id.shape)
        disp.cmap = 'viridis'
        plt.xlabel("")
        plt.ylabel("")

    plt.tight_layout()
    plt.show()
