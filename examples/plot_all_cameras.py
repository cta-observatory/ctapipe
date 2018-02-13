import numpy as np
from matplotlib import pyplot as plt

from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

if __name__ == '__main__':

    plt.style.use("bmh")

    camera_names = CameraGeometry.get_known_camera_names()
    n_tels = len(camera_names)
    n_rows = np.trunc(np.sqrt(n_tels)).astype(int)
    n_cols = np.ceil(n_tels / n_rows).astype(int)
    plt.figure(figsize=(15, 6))

    for ii, name in enumerate(sorted(camera_names)):
        print("plotting", name)
        geom = CameraGeometry.from_name(name)
        ax = plt.subplot(n_rows, n_cols, ii + 1)
        disp = CameraDisplay(geom)
        disp.image = np.random.uniform(size=geom.pix_id.shape)
        disp.cmap = 'viridis'
        plt.xlabel("")
        plt.ylabel("")

    plt.tight_layout()
    plt.show()
