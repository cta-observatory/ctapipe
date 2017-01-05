from ctapipe.io import CameraGeometry
from ctapipe.image import tailcuts_clean, toymodel
from ctapipe.image.hillas import (hillas_parameters_1, hillas_parameters_2,
                                  hillas_parameters_3, hillas_parameters_4)
from numpy import isclose

def create_sample_image():
    # set up the sample image using a HESS camera geometry (since it's easy
    # to load)
    geom = CameraGeometry.from_name("HESS", 1)

    # make a toymodel shower model
    model = toymodel.generate_2d_shower_model(centroid=(0.2, 0.3),
                                              width=0.01, length=0.1,
                                              psi='30d')

    # generate toymodel image in camera for this shower model.
    image, signal, noise = toymodel.make_toymodel_shower_image(geom, model.pdf,
                                                               intensity=50,
                                                               nsb_level_pe=100)

    # denoise the image, so we can calculate hillas params
    clean_mask = tailcuts_clean(geom, image, 1, 10,
                                5)  # pedvars = 1 and core and boundary
    # threshold in pe
    image[~clean_mask] = 0

    # Pixel values in the camera
    pix_x = geom.pix_x.value
    pix_y = geom.pix_y.value

    return pix_x, pix_y, image

def test_hillas():
    """
    test all Hillas-parameter routines on a sample image and see if they
    agree with eachother and with the toy model (assuming the toy model code
    is correct)
    """

    px, py, image = create_sample_image()
    results = {}

    results['v1'] = hillas_parameters_1(px, py, image)
    results['v2'] = hillas_parameters_2(px, py, image)
    results['v3'] = hillas_parameters_3(px, py, image)
    results['v4'] = hillas_parameters_4(px, py, image)

    # compare each method's output
    for aa in results:
        for bb in results:
            if aa is not bb:
                print("comparing {} to {}".format(aa,bb))
                assert isclose(results[aa].length, results[bb].length)
                assert isclose(results[aa].width, results[bb].width)
                assert isclose(results[aa].r, results[bb].r)
                assert isclose(results[aa].phi.deg, results[bb].phi.deg)
                assert isclose(results[aa].psi.deg, results[bb].psi.deg)
                assert isclose(results[aa].miss, results[bb].miss)
                assert isclose(results[aa].skewness, results[bb].skewness)
                #assert isclose(results[aa].kurtosis, results[bb].kurtosis)



