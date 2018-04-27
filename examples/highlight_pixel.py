from matplotlib import pyplot as plt

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

    model = toymodel.generate_2d_shower_model(
        centroid=(0.05, 0.0), width=0.005, length=0.025, psi='35d'
    )

    image, sig, bg = toymodel.make_toymodel_shower_image(
        geom, model.pdf, intensity=50, nsb_level_pe=20
    )

    disp.image = image

    mask = disp.image > 15
    disp.highlight_pixels(mask, linewidth=3)

    plt.show()
