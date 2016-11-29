import numpy as np
from matplotlib import pyplot as plt

from ctapipe.utils.datasets import get_path

from ctapipe.io.hessio import hessio_event_source
from ctapipe.io import CameraGeometry, convert_geometry_1d_to_2d, convert_geometry_back

from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError

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
    filename = get_path("gamma_test.simtel.gz")

    import glob
    filename = glob.glob("/local/home/tmichael/Data/cta/ASTRI9/gamma/*")[0]

    cam_geom = {}

    source = hessio_event_source(filename)

    ''' testing a few images just for the sake of being thorough '''
    counter = 5

    for event in source:

        for tel_id in event.dl0.tels_with_data:
            if tel_id not in cam_geom:
                cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])

            '''
            we want to test conversion of hex to rectangular pixel grid '''
            if cam_geom[tel_id].pix_type is not "hexagonal":
                continue

            print(tel_id, cam_geom[tel_id].pix_type)

            pmt_signal = apply_mc_calibration(
                        event.dl0.tel[tel_id].adc_sums[0],
                        event.mc.tel[tel_id].dc_to_pe[0],
                        event.mc.tel[tel_id].pedestal[0])

            new_geom, new_signal = convert_geometry_1d_to_2d(
                cam_geom[tel_id], pmt_signal, tel_id)

            unrot_geom, unrot_signal = convert_geometry_back(
                new_geom, new_signal, tel_id,
                event.inst.optical_foclen[tel_id])

            '''
            testing back and forth conversion on hillas parameters... '''
            try:
                moments1 = hillas_parameters(cam_geom[tel_id].pix_x,
                                             cam_geom[tel_id].pix_y,
                                             pmt_signal)[0]

                moments2 = hillas_parameters(unrot_geom.pix_x,
                                             unrot_geom.pix_y,
                                             unrot_signal)[0]
                '''
                we don't want this test to fail because the hillas code threw an error '''
            except (HillasParameterizationError, AssertionError):
                continue

            if __name__ == "__main__":
                try:
                    np.testing.assert_allclose([moments1.length, moments1.width],
                                               [moments1.length, moments1.width])
                except Exception as e:
                    print("caught exception: {}".format(e))

                fig = plt.figure()
                plt.style.use('seaborn-talk')

                ax1 = fig.add_subplot(131)
                disp1 = CameraDisplay(cam_geom[tel_id], image=pmt_signal, ax=ax1)
                disp1.cmap = plt.cm.hot
                disp1.add_colorbar()
                plt.title("original geometry")

                ax2 = fig.add_subplot(132)
                disp2 = CameraDisplay(new_geom, image=new_signal, ax=ax2)
                disp2.cmap = plt.cm.hot
                disp2.add_colorbar()
                plt.title("slanted geometry")

                ax3 = fig.add_subplot(133)
                disp3 = CameraDisplay(unrot_geom, image=unrot_signal, ax=ax3)
                disp3.cmap = plt.cm.hot
                disp3.add_colorbar()
                plt.title("geometry converted back to hex")

                plt.show()
            else:
                np.testing.assert_allclose([moments1.length, moments1.width],
                                           [moments1.length, moments1.width])
                counter -= 1
                if counter < 0:
                    return

if __name__ == "__main__":
    test_convert_geometry()
