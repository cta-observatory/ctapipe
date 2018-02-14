import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

if __name__ == '__main__':

    geom = CameraGeometry.from_name("Whipple109")
    image = np.random.uniform(size=geom.pix_id.shape)

    plt.figure(figsize=(10, 4))

    N = 4

    for ii in range(N):
        plt.subplot(1, N, ii + 1)
        geom.rotate(ii * (geom.pix_rotation + 30 * u.deg))
        d2 = CameraDisplay(geom, image=image, cmap='viridis')
