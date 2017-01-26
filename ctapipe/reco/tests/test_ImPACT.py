from astropy import units as u
from ctapipe.utils.datasets import get_path

from ctapipe.reco.FitGammaHillas import FitGammaHillas, GreatCircle
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.image.cleaning import tailcuts_clean, dilate

from ctapipe.io.hessio import hessio_event_source
from ctapipe.io import CameraGeometry
from ctapipe.reco.ImPACT import ImPACTFitter
from ctapipe.calib.camera.calibrators import CameraDL1Calibrator

def test_ImPACT_fit():
    '''
    a test of the complete fit procedure on one event including:
    • tailcut cleaning
    • hillas parametrisation
    • GreatCircle creation
    • direction fit
    • position fit

    in the end, proper units in the output are asserted '''

    filename = get_path("gamma_test.simtel.gz")
    ImPACT = ImPACTFitter(root_dir="/Users/dparsons/Documents/Unix/CTA/ImPACT_pythontests/", fit_xmax=True)
    calibrator = CameraDL1Calibrator(None, None)

    fit = FitGammaHillas()

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}

    source = hessio_event_source(filename)

    for event in source:
        hillas_dict = {}
        calibrator.calibrate(event)
        for tel_id in event.dl0.tels_with_data:
            pmt_signal = event.dl1.tel[tel_id].image[0]

            if tel_id not in cam_geom:
                cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])

                tel_phi[tel_id] = 0.*u.deg
                tel_theta[tel_id] = 20.*u.deg

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
    test_ImPACT_fit()