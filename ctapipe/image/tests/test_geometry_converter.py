import pytest
import numpy as np
from ctapipe.image.geometry_converter import (convert_geometry_hex1d_to_rect2d,
                                              convert_geometry_rect2d_back_to_hexe1d,
                                              astri_to_2d_array, array_2d_to_astri,
                                              chec_to_2d_array, array_2d_to_chec)
from ctapipe.image.hillas import hillas_parameters
from ctapipe.instrument import CameraGeometry
from ctapipe.image.toymodel import generate_2d_shower_model, make_toymodel_shower_image


cam_ids = CameraGeometry.get_known_camera_names()


def create_mock_image(geom):
    '''
    creates a mock image, which parameters are adapted to the camera size
    '''

    camera_r = np.max(np.sqrt(geom.pix_x**2 + geom.pix_y**2))
    model = generate_2d_shower_model(
        centroid=(0.3 * camera_r.value, 0),
        width=0.03 * camera_r.value,
        length=0.10 * camera_r.value,
        psi="25d"
    )

    _, image, _ = make_toymodel_shower_image(
        geom, model.pdf,
        intensity=0.5 * geom.n_pixels,
        nsb_level_pe=3,
    )
    return image


@pytest.mark.parametrize("rot", [3, ])
@pytest.mark.parametrize("cam_id", cam_ids)
def test_convert_geometry(cam_id, rot):

    geom = CameraGeometry.from_name(cam_id)
    image = create_mock_image(geom)
    hillas_0 = hillas_parameters(geom, image)

    if geom.pix_type == 'hexagonal':
        convert_geometry_1d_to_2d = convert_geometry_hex1d_to_rect2d
        convert_geometry_back = convert_geometry_rect2d_back_to_hexe1d

        geom2d, image2d = convert_geometry_1d_to_2d(geom, image,
                                                    geom.cam_id + str(rot),
                                                    add_rot=rot)
        geom1d, image1d = convert_geometry_back(geom2d, image2d,
                                                geom.cam_id + str(rot),
                                                add_rot=rot)

    else:
        if geom.cam_id == "ASTRICam":
            convert_geometry_1d_to_2d = astri_to_2d_array
            convert_geometry_back = array_2d_to_astri
        elif geom.cam_id == "CHEC":
            convert_geometry_1d_to_2d = chec_to_2d_array
            convert_geometry_back = array_2d_to_chec
        else:
            print("camera {geom.cam_id} not implemented")
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


@pytest.mark.parametrize("rot", [3, ])
@pytest.mark.parametrize("cam_id", cam_ids)
def test_convert_geometry_mock(cam_id, rot):
    """here we use a different key for the back conversion to trigger the mock conversion
    """

    geom = CameraGeometry.from_name(cam_id)
    image = create_mock_image(geom)
    hillas_0 = hillas_parameters(geom, image)

    if geom.pix_type == 'hexagonal':
        convert_geometry_1d_to_2d = convert_geometry_hex1d_to_rect2d
        convert_geometry_back = convert_geometry_rect2d_back_to_hexe1d

        geom2d, image2d = convert_geometry_1d_to_2d(geom, image, key=None,
                                                    add_rot=rot)
        geom1d, image1d = convert_geometry_back(geom2d, image2d,
                                                "_".join([geom.cam_id,
                                                          str(rot), "mock"]),
                                                add_rot=rot)
    else:
        # originally rectangular geometries don't need a buffer and therefore no mock
        # conversion
        return

    hillas_1 = hillas_parameters(geom, image1d)
    assert np.abs(hillas_1.phi - hillas_0.phi).deg < 1.0


# def plot_cam(geom, geom2d, geom1d, image, image2d, image1d):
#     # plt.viridis()
#     plt.figure(figsize=(12, 4))
#     ax = plt.subplot(1, 3, 1)
#     CameraDisplay(geom, image=image).add_colorbar()
#     plt.subplot(1, 3, 2, sharex=ax, sharey=ax)
#     CameraDisplay(geom2d, image=image2d).add_colorbar()
#     plt.subplot(1, 3, 3, sharex=ax, sharey=ax)
#     CameraDisplay(geom1d, image=image1d).add_colorbar()
#
#
# if __name__ == "__main__":
#     import logging
#     logging.basicConfig(level=logging.DEBUG)
#     for cam_id in CameraGeometry.get_known_camera_names():
#         test_convert_geometry(cam_id, 3)
#     plt.show()
