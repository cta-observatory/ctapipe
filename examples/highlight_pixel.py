from matplotlib import pyplot as plt
import astropy.units as u

from ctapipe.image import toymodel
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

if __name__ == '__main__':

    plt.style.use('ggplot')

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    geom = CameraGeometry.from_name('NectarCam')
    disp = CameraDisplay(geom, ax=ax)
    disp.add_colorbar()

    model = toymodel.Gaussian(
        x=0.05 * u.m,
        y=0 * u.m,
        width=0.05 * u.m,
        length=0.15 * u.m,
        psi='35d'
    )

    image, sig, bg = model.generate_image(
        geom, intensity=1500, nsb_level_pe=5
    )

    disp.image = image

    mask = disp.image > 10
    disp.highlight_pixels(mask, linewidth=2, color='crimson')

    plt.show()
