import numpy as np
from astropy import units as u

from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.io.hessio import hessio_event_source
from ctapipe.reco.HillasReconstructor import HillasReconstructor, GreatCircle
from ctapipe.utils import get_dataset_path


def test_fit_core():
    """
    creating some great circles pointing in different directions (two
    north-south,
    two east-west) and that have a slight position errors (+- 0.1 m in one of
    the four
    cardinal directions """
    circle1 = GreatCircle([[1, 0, 0], [0, 0, 1]])
    circle1.pos = [0, 0.1] * u.m
    circle1.trace = [1, 0, 0]

    circle2 = GreatCircle([[0, 1, 0], [0, 0, 1]])
    circle2.pos = [0.1, 0] * u.m
    circle2.trace = [0, 1, 0]

    circle3 = GreatCircle([[1, 0, 0], [0, 0, 1]])
    circle3.pos = [0, -.1] * u.m
    circle3.trace = [1, 0, 0]

    circle4 = GreatCircle([[0, 1, 0], [0, 0, 1]])
    circle4.pos = [-.1, 0] * u.m
    circle4.trace = [0, 1, 0]

    # creating the fit class and setting the the great circle member
    fit = HillasReconstructor()
    fit.circles = {1: circle1, 2: circle2, 3: circle3, 4: circle4}

    # performing the position fit with the minimisation algorithm
    # and a seed that is quite far away
    pos_fit_minimise = fit.fit_core_minimise([100, 1000] * u.m)
    print("position fit test minimise:", pos_fit_minimise)
    print()

    # performing the position fit with the geometric algorithm
    pos_fit_crosses, err_est_pos_fit_crosses = fit.fit_core_crosses()
    print("position fit test crosses:", pos_fit_crosses)
    print("error estimate:", err_est_pos_fit_crosses)
    print()

    # the results should be close to the origin of the coordinate system
    np.testing.assert_allclose(pos_fit_minimise / u.m, [0, 0], atol=1e-3)
    np.testing.assert_allclose(pos_fit_crosses / u.m, [0, 0], atol=1e-3)


def test_fit_origin():
    """
    creating some great circles pointing in different directions (two
    north-south, two east-west) and that have a slight position errors (+-
    0.1 m in one of the four cardinal directions """
    circle1 = GreatCircle([[1, 0, 0], [0, 0, 1]])
    circle1.pos = [0, 0.1] * u.m
    circle1.trace = [1, 0, 0]

    circle2 = GreatCircle([[0, 1, 0], [0, 0, 1]])
    circle2.pos = [0.1, 0] * u.m
    circle2.trace = [0, 1, 0]

    circle3 = GreatCircle([[1, 0, 0], [0, 0, 1]])
    circle3.pos = [0, -.1] * u.m
    circle3.trace = [1, 0, 0]

    circle4 = GreatCircle([[0, 1, 0], [0, 0, 1]])
    circle4.pos = [-.1, 0] * u.m
    circle4.trace = [0, 1, 0]

    # creating the fit class and setting the the great circle member
    fit = HillasReconstructor()
    fit.circles = {1: circle1, 2: circle2, 3: circle3, 4: circle4}

    # performing the direction fit with the minimisation algorithm
    # and a seed that is perpendicular to the up direction
    dir_fit_minimise = fit.fit_origin_minimise((0.1, 0.1, 1))
    print("direction fit test minimise:", dir_fit_minimise)
    print()

    # performing the direction fit with the geometric algorithm
    dir_fit_crosses = fit.fit_origin_crosses()[0]
    print("direction fit test crosses:", dir_fit_crosses)
    print()

    # the results should be close to the direction straight up
    # np.testing.assert_allclose(dir_fit_minimise, [0, 0, 1], atol=1e-1)
    np.testing.assert_allclose(dir_fit_crosses, [0, 0, 1], atol=1e-3)


def test_reconstruction():
    """
    a test of the complete fit procedure on one event including:
    • tailcut cleaning
    • hillas parametrisation
    • GreatCircle creation
    • direction fit
    • position fit

    in the end, proper units in the output are asserted """

    filename = get_dataset_path("gamma_test.simtel.gz")

    fit = HillasReconstructor()

    tel_phi = {}
    tel_theta = {}

    source = hessio_event_source(filename)

    for event in source:

        hillas_dict = {}
        for tel_id in event.dl0.tels_with_data:

            geom = event.inst.subarray.tel[tel_id].camera
            tel_phi[tel_id] = event.mc.tel[tel_id].azimuth_raw * u.rad
            tel_theta[tel_id] = (np.pi / 2 -
                                 event.mc.tel[tel_id].altitude_raw) * u.rad

            pmt_signal = event.r0.tel[tel_id].image[0]

            mask = tailcuts_clean(geom, pmt_signal,
                                  picture_thresh=10., boundary_thresh=5.)
            pmt_signal[mask == 0] = 0

            try:
                moments = hillas_parameters(geom, pmt_signal)
                hillas_dict[tel_id] = moments
            except HillasParameterizationError as e:
                print(e)
                continue

        if len(hillas_dict) < 2:
            continue

        fit_result = fit.predict(hillas_dict, event.inst, tel_phi, tel_theta)

        print(fit_result)
        fit_result.alt.to(u.deg)
        fit_result.az.to(u.deg)
        fit_result.core_x.to(u.m)
        assert fit_result.is_valid
        return
