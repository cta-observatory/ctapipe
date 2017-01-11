from astropy import units as u
import numpy as np

from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.utils.datasets import get_path

from ctapipe.reco.FitGammaHillas import FitGammaHillas, GreatCircle
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.image.cleaning import tailcuts_clean, dilate

from ctapipe.io.hessio import hessio_event_source
from ctapipe.io import CameraGeometry


def test_fit_core():

    '''
    creating some great circles pointing in different directions (two north-south,
    two east-west) and that have a slight position errors (+- 0.1 m in one of the four
    cardinal directions '''
    circle1 = GreatCircle([[1, 0, 0], [0, 0, 1]])
    circle1.pos = [0, 0.1]*u.m
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

    '''
    creating the fit class and setting the the great circle member '''
    fit = FitGammaHillas()
    fit.circles = {1: circle1, 2: circle2, 3: circle3, 4: circle4}

    ''' performing the position fit with a seed that is quite far away '''
    pos_fit = fit.fit_core([100, 1000]*u.m)
    print("position test fit:", pos_fit)

    ''' the result should be close to the origin of the coordinate system '''
    np.testing.assert_allclose(pos_fit/u.m, [0, 0], atol=1e-3)


def test_FitGammaHillas():
    '''
    a test on one event of the complete fit procedure including:
    • tailcut cleaning
    • hillas parametrisation
    • GreatCircle creation
    • direction fit
    • position fit

    in the end, proper units in the output are asserted '''

    filename = get_path("gamma_test.simtel.gz")

    fit = FitGammaHillas()

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}

    source = hessio_event_source(filename)

    for event in source:

        hillas_dict = {}
        for tel_id in event.dl0.tels_with_data:

            if tel_id not in cam_geom:
                cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])

                tel_phi[tel_id] = 180.*u.deg
                tel_theta[tel_id] = 20.*u.deg

            pmt_signal = event.r0.tel[tel_id].adc_sums[0]

            mask = tailcuts_clean(cam_geom[tel_id], pmt_signal, 1,
                                  picture_thresh=10., boundary_thresh=5.)
            pmt_signal[mask == 0] = 0

            try:
                moments = hillas_parameters(event.inst.pixel_pos[tel_id][0],
                                            event.inst.pixel_pos[tel_id][1],
                                            pmt_signal)
                hillas_dict[tel_id] = moments
            except HillasParameterizationError as e:
                print(e)
                continue

        if len(hillas_dict) < 2: continue

        fit_result = fit.predict(hillas_dict, event.inst, tel_phi, tel_theta)

        print(fit_result)
        fit_result.alt.to(u.deg)
        fit_result.az.to(u.deg)
        fit_result.core_x.to(u.m)
        assert fit_result.is_valid
        return


if __name__ == "__main__":
    test_fit_core()
    test_FitGammaHillas()
