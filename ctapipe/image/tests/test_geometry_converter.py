import pytest
import numpy as np
from ctapipe.image.geometry_converter import (
    convert_geometry_hex1d_to_rect2d,
    convert_geometry_rect2d_back_to_hexe1d,
    astri_to_2d_array,
    array_2d_to_astri,
    chec_to_2d_array,
    array_2d_to_chec,
)
from ctapipe.image.hillas import hillas_parameters
from ctapipe.image.toymodel import Gaussian
import astropy.units as u


def create_mock_image(geom):
    """
    creates a mock image, which parameters are adapted to the camera size
    """

    camera_r = np.max(np.sqrt(geom.pix_x ** 2 + geom.pix_y ** 2))
    model = Gaussian(
        x=0.3 * camera_r,
        y=0 * u.m,
        width=0.03 * camera_r,
        length=0.10 * camera_r,
        psi="25d",
    )

    _, image, _ = model.generate_image(
        geom, intensity=0.5 * geom.n_pixels, nsb_level_pe=3
    )
    return image


@pytest.mark.parametrize("rot", [3])
def test_convert_geometry(camera_geometry, rot):
    geom = camera_geometry
    image = create_mock_image(geom)
    hillas_0 = hillas_parameters(geom, image)

    if geom.pix_type == "hexagonal":
        convert_geometry_1d_to_2d = convert_geometry_hex1d_to_rect2d
        convert_geometry_back = convert_geometry_rect2d_back_to_hexe1d

        geom2d, image2d = convert_geometry_1d_to_2d(
            geom, image, geom.camera_name + str(rot), add_rot=rot
        )
        geom1d, image1d = convert_geometry_back(
            geom2d, image2d, geom.camera_name + str(rot), add_rot=rot
        )

    else:
        if geom.camera_name == "ASTRICam":
            convert_geometry_1d_to_2d = astri_to_2d_array
            convert_geometry_back = array_2d_to_astri
        elif geom.camera_name == "CHEC":
            convert_geometry_1d_to_2d = chec_to_2d_array
            convert_geometry_back = array_2d_to_chec
        else:
            print("camera {geom.camera_name} not implemented")
            return

        image2d = convert_geometry_1d_to_2d(image)
        image1d = convert_geometry_back(image2d)

    hillas_1 = hillas_parameters(geom, image1d)

    # if __name__ == "__main__":
    #     plot_cam(geom, geom2d, geom1d, image, image2d, image1d)
    #     plt.tight_layout()
    #     plt.pause(.1)

    assert np.abs(hillas_1.phi - hillas_0.phi).deg < 1.0
    # TODO: test other parameters


@pytest.mark.parametrize("rot", [3])
def test_convert_geometry_mock(camera_geometry, rot):
    """here we use a different key for the back conversion to trigger the mock conversion
    """
    geom = camera_geometry
    image = create_mock_image(geom)
    hillas_0 = hillas_parameters(geom, image)

    if geom.pix_type == "hexagonal":
        convert_geometry_1d_to_2d = convert_geometry_hex1d_to_rect2d
        convert_geometry_back = convert_geometry_rect2d_back_to_hexe1d

        geom2d, image2d = convert_geometry_1d_to_2d(geom, image, key=None, add_rot=rot)
        geom1d, image1d = convert_geometry_back(
            geom2d, image2d, "_".join([geom.camera_name, str(rot), "mock"]), add_rot=rot
        )
    else:
        # originally rectangular geometries don't need a buffer and therefore no mock
        # conversion
        return

    hillas_1 = hillas_parameters(geom, image1d)
    assert np.abs(hillas_1.phi - hillas_0.phi).deg < 1.0
