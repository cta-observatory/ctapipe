import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from ctapipe.instrument import CameraGeometry
from ctapipe.utils import get_dataset_path
from ctapipe.visualization import CameraDisplay

if __name__ == "__main__":

    path = get_dataset_path("Whipple109.camgeom.fits.gz")
    geom = CameraGeometry.from_table(path)
    image = np.random.uniform(size=geom.pix_id.shape)

    plt.figure(figsize=(10, 4))

    N = 4

    for ii in range(N):
        plt.subplot(1, N, ii + 1)
        geom.rotate(geom.pix_rotation + ii * 30 * u.deg)
        d2 = CameraDisplay(geom, image=image, cmap="viridis")

    plt.tight_layout()
    plt.show()
