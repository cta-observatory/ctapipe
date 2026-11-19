"""
Compare camera coordinate frames for a simulated shower image.

The same image is displayed in three frames: CameraFrame, TelescopeFrame, and
EngineeringCameraFrame.
"""

import astropy.units as u
import matplotlib.pyplot as plt

from ctapipe.coordinates import EngineeringCameraFrame, TelescopeFrame
from ctapipe.image.toymodel import Gaussian
from ctapipe.instrument import SubarrayDescription
from ctapipe.visualization import CameraDisplay


def main():
    """Generate a toy shower image and show it in three coordinate frames."""
    _, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(8, 3))

    model = Gaussian(0 * u.m, 0.1 * u.m, 0.3 * u.m, 0.05 * u.m, 25 * u.deg)

    subarray = SubarrayDescription.read("dataset://gamma_prod5.simtel.zst")
    cam = subarray.tel[5].camera.geometry

    image, *_ = model.generate_image(cam, 2500)

    CameraDisplay(cam, ax=axs[0], image=image)
    CameraDisplay(
        cam.transform_to(EngineeringCameraFrame),
        ax=axs[1],
        image=image,
    )
    CameraDisplay(
        cam.transform_to(TelescopeFrame),
        ax=axs[2],
        image=image,
    )

    axs[0].set_title("CameraFrame")
    axs[1].set_title("EngineeringCameraFrame")
    axs[2].set_title("TelescopeFrame")

    plt.show()


if __name__ == "__main__":
    main()
