from ctapipe.instrument import CameraGeometry
from ctapipe.image import tailcuts_clean, toymodel
from ctapipe.image.hillas import (hillas_parameters_1, hillas_parameters_2,
                                  hillas_parameters_3, hillas_parameters_4)
from astropy import units as u
from numpy import isclose
from numpy.random import seed

def create_sample_image(psi='-30d'):

    seed(10)

    # set up the sample image using a HESS camera geometry (since it's easy
    # to load)
    geom = CameraGeometry.from_name("LSTCam")

    # make a toymodel shower model
    model = toymodel.generate_2d_shower_model(centroid=(0.2, 0.3),
                                              width=0.001, length=0.01,
                                              psi=psi)

    # generate toymodel image in camera for this shower model.
    image, signal, noise = toymodel.make_toymodel_shower_image(geom, model.pdf,
                                                               intensity=50,
                                                               nsb_level_pe=100)

    # denoise the image, so we can calculate hillas params
    clean_mask = tailcuts_clean(geom, image, 10,
                                5)  # pedvars = 1 and core and boundary
    # threshold in pe
    image[~clean_mask] = 0

    # Pixel values in the camera
    pix_x = geom.pix_x.value
    pix_y = geom.pix_y.value

    return pix_x, pix_y, image

def compare_result(x,y):
    ux = u.Quantity(x)
    uy = u.Quantity(y)
    assert isclose(ux.value,uy.value)
    assert ux.unit == uy.unit


def do_test_hillas(withunits=True):
    """
    test all Hillas-parameter routines on a sample image and see if they
    agree with eachother and with the toy model (assuming the toy model code
    is correct)
    """

    # try all quadrants
    for psi_angle in ['30d','120d','-30d','-120d']:

        px, py, image = create_sample_image(psi_angle)
        results = {}

        if withunits:
            px = px * u.cm
            py = py * u.cm

        results['v1'] = hillas_parameters_1(px, py, image)
        results['v2'] = hillas_parameters_2(px, py, image)
        results['v3'] = hillas_parameters_3(px, py, image)
        results['v4'] = hillas_parameters_4(px, py, image)
        # compare each method's output
        for aa in results:
            for bb in results:
                if aa is not bb:
                    print("comparing {} to {}".format(aa,bb))
                    compare_result(results[aa].length, results[bb].length)
                    compare_result(results[aa].width, results[bb].width)
                    compare_result(results[aa].r, results[bb].r)
                    compare_result(results[aa].phi.deg, results[bb].phi.deg)
                    compare_result(results[aa].psi.deg, results[bb].psi.deg)
                    compare_result(results[aa].miss, results[bb].miss)
                    compare_result(results[aa].skewness, results[bb].skewness)
                    #compare_result(results[aa].kurtosis, results[bb].kurtosis)


def test_hillas_with_units():
    do_test_hillas(withunits=True)

def test_hillas_unitless():
    do_test_hillas(withunits=False)

