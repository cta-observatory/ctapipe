"""
Plot the same event in two camera displays showing the
different coordinate frames for camera coordinates.
"""
import astropy.units as u
import matplotlib.pyplot as plt

from ctapipe.coordinates import EngineeringCameraFrame, TelescopeFrame
from ctapipe.image.toymodel import Gaussian
from ctapipe.instrument import SubarrayDescription
from ctapipe.visualization import CameraDisplay


def main():
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 3))

    model = Gaussian(0 * u.deg, 0.1 * u.deg, 0.6 * u.deg, 0.10 * u.deg, 25 * u.deg)

    subarray = SubarrayDescription.read("dataset://gamma_prod5.simtel.zst")
    geom_cam = subarray.tel[5].camera.geometry
    geom_tel = subarray.tel[5].camera.geometry.transform_to(TelescopeFrame())

    image, *_ = model.generate_image(geom_tel, 2500)

    CameraDisplay(geom_tel.transform_to(geom_cam.frame), ax=axs[0], image=image)
    CameraDisplay(
        geom_tel.transform_to(geom_cam.frame).transform_to(EngineeringCameraFrame()),
        ax=axs[1],
        image=image,
    )

    axs[0].set_title("CameraFrame")
    axs[1].set_title("EngineeringCameraFrame")

    plt.show()


if __name__ == "__main__":
    main()
