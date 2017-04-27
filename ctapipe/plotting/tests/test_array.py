from astropy import units as u
import numpy as np
from ctapipe.utils.datasets import get_path

from ctapipe.reco.FitGammaHillas import FitGammaHillas, GreatCircle
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.image.cleaning import tailcuts_clean, dilate

from ctapipe.io.hessio import hessio_event_source
from ctapipe.io import CameraGeometry
from ctapipe.plotting.array import ArrayPlotter, NominalPlotter
from ctapipe.coordinates import TiltedGroundFrame, NominalFrame, CameraFrame

from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.r1 import HessioR1Calibrator

def test_array_draw():

    filename = get_path("gamma_test.simtel.gz")
    cam_geom = {}

    source = hessio_event_source(filename)
    r1 = HessioR1Calibrator(None, None)
    dl0 = CameraDL0Reducer(None, None)

    calibrator = CameraDL1Calibrator(None, None)

    for event in source:
        array_pointing =(event.mcheader.run_array_direction[1]*u.rad, event.mcheader.run_array_direction[0]*u.rad)
        #array_view = ArrayPlotter(instrument=event.inst,
        #                          system=TiltedGroundFrame(pointing_direction=array_pointing))

        hillas_dict = {}
        r1.calibrate(event)
        dl0.reduce(event)
        calibrator.calibrate(event) # calibrate the events

        # store MC pointing direction for the array

        for tel_id in event.dl0.tels_with_data:

            pmt_signal = event.dl1.tel[tel_id].image[0]

            x, y = event.inst.pixel_pos[tel_id]
            fl = event.inst.optical_foclen[tel_id]

            if tel_id not in cam_geom:
                cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])


            # Transform the pixels positions into nominal coordinates
            camera_coord = CameraFrame(x=x, y=y, z=np.zeros(x.shape) * u.m,
                                       focal_length=fl,
                                       rotation= 90*u.deg - cam_geom[tel_id].cam_rotation )

            nom_coord = camera_coord.transform_to(NominalFrame(array_direction=array_pointing,
                                                               pointing_direction=array_pointing))

            mask = tailcuts_clean(cam_geom[tel_id], pmt_signal, 1,
                                  picture_thresh=10., boundary_thresh=5.)

            try:
                moments = hillas_parameters(nom_coord.x,
                                            nom_coord.y,
                                            pmt_signal*mask)
                hillas_dict[tel_id] = moments
            except HillasParameterizationError as e:
                print(e)
                continue

        nom_coord = NominalPlotter(hillas_parameters=hillas_dict, draw_axes=True)
        nom_coord.draw_array()
        #array_view.background_image(np.ones((4,4)), ((-1500,1500),(-1500,1500)))
        #array_view.overlay_hillas(hillas_dict, draw_axes=True)
        #array_view.draw_array(range=((-1000,1000),(-1000,1000)))
        #return

if __name__ == "__main__":
    test_array_draw()