import numpy as np
from astropy import units as u

from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.reco.HillasReconstructor import HillasReconstructor, HillasPlane
from ctapipe.utils import get_dataset_path
from astropy.coordinates import SkyCoord


def test_estimator_results():
    """
    creating some planes pointing in different directions (two
    north-south, two east-west) and that have a slight position errors (+-
    0.1 m in one of the four cardinal directions """

    p1 = SkyCoord(alt=43 * u.deg, az=45 * u.deg, frame='altaz')
    p2 = SkyCoord(alt=47 * u.deg, az=45 * u.deg, frame='altaz')
    circle1 = HillasPlane(p1=p1, p2=p2, telescope_position=[0, 1, 0] * u.m)

    p1 = SkyCoord(alt=44 * u.deg, az=90 * u.deg, frame='altaz')
    p2 = SkyCoord(alt=46 * u.deg, az=90 * u.deg, frame='altaz')
    circle2 = HillasPlane(p1=p1, p2=p2, telescope_position=[1, 0, 0] * u.m)

    p1 = SkyCoord(alt=44.5 * u.deg, az=45 * u.deg, frame='altaz')
    p2 = SkyCoord(alt=46.5 * u.deg, az=45 * u.deg, frame='altaz')
    circle3 = HillasPlane(p1=p1, p2=p2, telescope_position=[0, -1, 0] * u.m)

    p1 = SkyCoord(alt=43.5 * u.deg, az=90 * u.deg, frame='altaz')
    p2 = SkyCoord(alt=45.5 * u.deg, az=90 * u.deg, frame='altaz')
    circle4 = HillasPlane(p1=p1, p2=p2, telescope_position=[-1, 0, 0] * u.m)

    # creating the fit class and setting the the great circle member
    fit = HillasReconstructor()
    fit.hillas_planes = {1: circle1, 2: circle2, 3: circle3, 4: circle4}

    # performing the direction fit with the minimisation algorithm
    # and a seed that is perpendicular to the up direction
    dir_fit_minimise, _ = fit.estimate_direction()
    print("direction fit test minimise:", dir_fit_minimise)
    print()

    # performing the direction fit with the geometric algorithm
    fitted_core_position, _ = fit.estimate_core_position()
    print("direction fit test core position:", fitted_core_position)
    print()

    # the results should be close to the direction straight up
    np.testing.assert_allclose(dir_fit_minimise, [0, 0, 1], atol=1e-3)
    np.testing.assert_allclose(fitted_core_position.value, [0, 0], atol=1e-3)


def test_reconstruction():
    """
    a test of the complete fit procedure on one event including:
    • tailcut cleaning
    • hillas parametrisation
    • HillasPlane creation
    • direction fit
    • position fit

    in the end, proper units in the output are asserted """

    filename = get_dataset_path("gamma_test.simtel.gz")

    fit = HillasReconstructor()

    tel_azimuth = {}
    tel_altitude = {}

    source = EventSourceFactory.produce(
        input_url=filename,
        product='HESSIOEventSource',
    )

    for event in source:

        hillas_dict = {}
        for tel_id in event.dl0.tels_with_data:

            geom = event.inst.subarray.tel[tel_id].camera
            tel_azimuth[tel_id] = event.mc.tel[tel_id].azimuth_raw * u.rad
            tel_altitude[tel_id] = event.mc.tel[tel_id].altitude_raw * u.rad

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

        fit_result = fit.predict(hillas_dict, event.inst, tel_azimuth, tel_altitude)

        print(fit_result)
        fit_result.alt.to(u.deg)
        fit_result.az.to(u.deg)
        fit_result.core_x.to(u.m)
        assert fit_result.is_valid
        return
