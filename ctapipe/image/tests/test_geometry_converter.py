import numpy as np
from matplotlib import pyplot as plt

from ctapipe.utils import get_dataset

from ctapipe.io.hessio import hessio_event_source

from ctapipe.image.geometry_converter import CameraGeometry, \
                                             convert_geometry_1d_to_2d, \
                                             convert_geometry_back
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.image.cleaning import tailcuts_clean

from ctapipe.visualization import CameraDisplay


def apply_mc_calibration(adcs, gains, peds):
    """
    apply basic calibration
    """

    if adcs.ndim > 1:  # if it's per-sample need to correct the peds
        return ((adcs - peds[:, np.newaxis] / adcs.shape[1]) *
                gains[:, np.newaxis])

    return (adcs - peds) * gains


def test_convert_geometry():
    filename = get_dataset("gamma_test.simtel.gz")

    cam_geom = {}

    source = hessio_event_source(filename)

    # testing a few images just for the sake of being thorough
    counter = 5

    for event in source:

        for tel_id in event.r0.tels_with_data:
            if tel_id not in cam_geom:
                cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])


            # we want to test conversion of hex to rectangular pixel grid
            if cam_geom[tel_id].pix_type is not "hexagonal":
                continue

            print(tel_id, cam_geom[tel_id].pix_type)

            pmt_signal = apply_mc_calibration(
                        #event.r0.tel[tel_id].adc_samples[0],
                        event.r0.tel[tel_id].adc_sums[0],
                        event.mc.tel[tel_id].dc_to_pe[0],
                        event.mc.tel[tel_id].pedestal[0])

            new_geom, new_signal = convert_geometry_1d_to_2d(
                cam_geom[tel_id], pmt_signal, cam_geom[tel_id].cam_id, add_rot=-2)

            unrot_geom, unrot_signal = convert_geometry_back(
                new_geom, new_signal, cam_geom[tel_id].cam_id,
                event.inst.optical_foclen[tel_id], add_rot=4)

            # if run as main, do some plotting
            if __name__ == "__main__":
                fig = plt.figure()
                plt.style.use('seaborn-talk')

                ax1 = fig.add_subplot(131)
                disp1 = CameraDisplay(cam_geom[tel_id],
                                      image=np.sum(pmt_signal, axis=1)
                                      if pmt_signal.shape[-1] == 25 else pmt_signal,
                                      ax=ax1)
                disp1.cmap = plt.cm.hot
                disp1.add_colorbar()
                plt.title("original geometry")

                ax2 = fig.add_subplot(132)
                disp2 = CameraDisplay(new_geom,
                                      image=np.sum(new_signal, axis=2)
                                      if new_signal.shape[-1] == 25 else new_signal,
                                      ax=ax2)
                disp2.cmap = plt.cm.hot
                disp2.add_colorbar()
                plt.title("slanted geometry")

                ax3 = fig.add_subplot(133)
                disp3 = CameraDisplay(unrot_geom, image=np.sum(unrot_signal, axis=1)
                                      if unrot_signal.shape[-1] == 25 else unrot_signal,
                                      ax=ax3)
                disp3.cmap = plt.cm.hot
                disp3.add_colorbar()
                plt.title("geometry converted back to hex")

                plt.show()


            # do some tailcuts cleaning
            mask1 = tailcuts_clean(cam_geom[tel_id], pmt_signal,
                                   picture_thresh=10., boundary_thresh=5.)

            mask2 = tailcuts_clean(unrot_geom, unrot_signal, picture_thresh=10.,
                                   boundary_thresh=5.)
            pmt_signal[mask1==False] = 0
            unrot_signal[mask2==False] = 0

            '''
            testing back and forth conversion on hillas parameters... '''
            try:
                moments1 = hillas_parameters(cam_geom[tel_id].pix_x,
                                             cam_geom[tel_id].pix_y,
                                             pmt_signal)

                moments2 = hillas_parameters(unrot_geom.pix_x,
                                             unrot_geom.pix_y,
                                             unrot_signal)
            except (HillasParameterizationError, AssertionError) as e:
                '''
                we don't want this test to fail because the hillas code
                threw an error '''
                print(e)
                counter -= 1
                if counter < 0:
                    return
                else:
                    continue

            '''
            test if the hillas parameters from the original geometry and the
            forth-and-back rotated geometry are close '''
            assert np.allclose(
                [moments1.length.value, moments1.width.value,
                 moments1.phi.value],
                [moments2.length.value, moments2.width.value,
                 moments2.phi.value],
                rtol=1e-2, atol=1e-2)
            counter -= 1
            if counter < 0:
                return

if __name__ == "__main__":
    test_convert_geometry()
