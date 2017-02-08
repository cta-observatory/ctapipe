from astropy import units as u

from ctapipe.utils.datasets import get_path

from ctapipe.reco.FitGammaHillas import FitGammaHillas, GreatCircle
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.image.cleaning import tailcuts_clean, dilate

from ctapipe.io.hessio import hessio_event_source
from ctapipe.io import CameraGeometry
from ctapipe.plotting.array import ArrayPlotter
from ctapipe.coordinates import TiltedGroundFrame

def test_array_draw():


    filename = get_path("gamma_test.simtel.gz")
    fit = FitGammaHillas()

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}

    source = hessio_event_source(filename)

    for event in source:

        array_view = ArrayPlotter(instrument=event.inst,
                                  system=TiltedGroundFrame(pointing_direction=[70*u.deg, 45*u.deg]))

        hillas_dict = {}
        for tel_id in event.dl0.tels_with_data:

            if tel_id not in cam_geom:
                cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])

            pmt_signal = event.dl0.tel[tel_id].adc_sums[0]

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
        array_view.draw_array()
        return

if __name__ == "__main__":
    test_array_draw()