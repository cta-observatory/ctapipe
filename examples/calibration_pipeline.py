import sys
import argparse
from matplotlib import colors, pyplot as plt
import numpy as np
from ctapipe.io.hessio import hessio_event_source
from pyhessio import *
from ctapipe.core import Container
from ctapipe.io.containers import RawData, CalibratedCameraData
from ctapipe import visualization, io
from astropy import units as u
from ctapipe.calib.camera.mc import *

fig = plt.figure(figsize=(16, 7))
cmaps = [plt.cm.jet, plt.cm.winter,
         plt.cm.ocean, plt.cm.bone, plt.cm.gist_earth, plt.cm.hot,
         plt.cm.cool, plt.cm.coolwarm]


def display_telescope(event, tel_id):
    global fig
    ntels = len(event.dl1.tels_with_data)
    fig.clear()

    plt.suptitle("EVENT {} {:.1e} TeV @({:.1f},{:.1f})deg @{:.1f} m".format(
            event.dl1.event_id, get_mc_shower_energy(),
            get_mc_shower_altitude(), get_mc_shower_azimuth(),
            np.sqrt(pow(get_mc_event_xcore(), 2) +
                    pow(get_mc_event_ycore(), 2))))
    print("\t draw cam {}...".format(tel_id))
    x, y = event.meta.pixel_pos[tel_id]
    geom = io.CameraGeometry.guess(x * u.m, y * u.m)
    npads = 1
    # Only create two pads if there is timing information extracted
    # from the calibration
    if not event.dl1.tel[tel_id].tom is None:
        npads = 2

    ax = plt.subplot(1, npads, npads-1)
    disp = visualization.CameraDisplay(geom, ax=ax,
                                       title="CT{0}".format(tel_id))

    disp.pixels.set_antialiaseds(False)
    disp.autoupdate = False
    disp.pixels.set_cmap('seismic')
    chan = 0
    signals = event.dl1.tel[tel_id].pe_charge
    disp.image = signals
    disp.add_colorbar()
    if npads == 2:
        ax = plt.subplot(1, npads, npads)
        disp = visualization.CameraDisplay(geom,
                                           ax=ax,
                                           title="CT{0}".format(tel_id))
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.pixels.set_cmap('gnuplot')
        chan = 0
        disp.image = event.dl1.tel[tel_id].tom
        disp.add_colorbar()

    if __debug__:
        print("All sum = %.3f\n" % sum(event.dl1.tel[tel_id].pe_charge))


def camera_calibration(filename, parameters, disp_args, level):
    """
    Parameters
    ----------
    filename   MC filename with raw data (in ADC samples)
    parameters Parameters to be passed to the different calibration functions
               (described in each function separately inside mc.py)
    disp_args  Either: per telescope per event or
               all telescopes of the event (currently dissabled)
    level      Output information of the calibration level results
    Returns
    -------
    A display (see function display_telescope)

    """
    TAG = sys._getframe().f_code.co_name+">"

    # Load dl1 container
    container = Container("calibrated_hessio_container")
    container.add_item("dl1", RawData())
    container.meta.add_item('pixel_pos', dict())

    # loop over all events, all telescopes and all channels and call
    # the calc_peds function defined above to do some work:
    nt = 0
    for event in hessio_event_source(filename):
        nt = nt+1
        # Fill DL1 container headers information. Clear also telescope info.
        container.dl1.run_id = event.dl0.run_id
        container.dl1.event_id = event.dl0.event_id
        container.dl1.tel = dict()  # clear the previous telescopes
        container.dl1.tels_with_data = event.dl0.tels_with_data
        if __debug__:
            print(TAG, container.dl1.run_id, "#%d" % nt,
                  container.dl1.event_id,
                  container.dl1.tels_with_data,
                  "%.3e TeV @ (%.0f,%.0f)deg @ %.3f m" %
                  (get_mc_shower_energy(), get_mc_shower_altitude(),
                   get_mc_shower_azimuth(),
                   np.sqrt(pow(get_mc_event_xcore(), 2) +
                           pow(get_mc_event_ycore(), 2))))
        for telid in event.dl0.tels_with_data:
            print(TAG, "Calibrating.. CT%d\n" % telid)

            # Get per telescope the camera geometry
            x, y = event.meta.pixel_pos[telid]
            geom = io.CameraGeometry.guess(x * u.m, y * u.m)

            # Get the calibration data sets (pedestals and single-pe)
            ped = get_pedestal(telid)
            calib = get_calibration(telid)

            # Integrate pixels traces and substract pedestal
            # See pixel_integration_mc function documentation in mc.py
            # for the different algorithms options
            int_adc_pix, peak_adc_pix = pixel_integration_mc(event,
                                                             ped, telid,
                                                             parameters)
            # Convert integrated ADC counts into p.e.
            # selecting also the HG/LG channel (currently hard-coded)
            pe_pix = calibrate_amplitude_mc(int_adc_pix, calib,
                                            telid, parameters)
            # Including per telescope metadata in the DL1 container
            if telid not in container.meta.pixel_pos:
                container.meta.pixel_pos[telid] = event.meta.pixel_pos[telid]
            container.dl1.tels_with_data = event.dl0.tels_with_data
            container.dl1.tel[telid] = CalibratedCameraData(telid)
            container.dl1.tel[telid].pe_charge = np.array(pe_pix)
            container.dl1.tel[telid].tom = np.array(peak_adc_pix[0])

            # FOR THE CTA USERS:
            # From here you can include your code.
            # It should take as input the last data level calculated here (DL1)
            # or call reconstruction algorithms (reco module) to be called.
            # For example: you could ask to calculate the tail cuts cleaning
            # using the tailcuts_clean in reco/cleaning.py module
            #
            # if 'tail_cuts' in parameters:
            #    clean_mask = tailcuts_clean(geom,
            #    image=np.array(pe_pix),pedvars=1,
            #    picture_thresh=parameters['tail_cuts'][0],
            #    boundary_thresh=parameters['tail_cuts'][1])
            #    container.dl1.tel[telid].pe_charge = np.array(pe_pix) *
            #    np.array(clean_mask)
            #    container.dl1.tel[telid].tom = np.array(peak_adc_pix[0]) *
            #    np.array(clean_mask)
            #

        sys.stdout.flush()
        # Display
        if 'event' in disp_args:
            ello = input("See evt. %d?<[n]/y/q> " % container.dl1.event_id)
            if ello == 'y':

                if 'telescope' in disp_args:
                    for telid in container.dl1.tels_with_data:
                        ello = input(
                            "See telescope/evt. %d?[CT%d]<[n]/y/q/e> " %
                            (container.dl1.event_id, telid))
                        if ello == 'y':
                            display_telescope(container, telid)
                            plt.pause(0.1)
                        elif ello == 'q':
                            break
                        elif ello == 'e':
                            return None
                        else:
                            continue
                else:
                    plt.pause(0.1)
            elif ello == 'q':
                return None


if __name__ == '__main__':
    TAG = sys._getframe().f_code.co_name+">"

    # Declare and parse command line option
    parser = argparse.ArgumentParser(
        description='Tel_id, pixel id and number of event to compute.')
    parser.add_argument('--f', dest='filename',
                        required=True, help='filename MC file name')
    args = parser.parse_args()

    plt.show(block=False)

    # Function description of camera_calibration options, given here
    # Integrator: samples integration algorithm (equivalent to hessioxxx
    # option --integration-sheme)
    #   -options: full_integration,
    #             simple_integration,
    #             global_peak_integration,
    #             local_peak_integration,
    #             nb_peak_integration
    # nsum: Number of samples to sum up (is reduced if exceeding available
    # length). (equivalent to first number in
    # hessioxxx option --integration-window)
    # nskip: Number of initial samples skipped (adapted such that interval
    # fits into what is available). Start the integration a number of
    # samples before the peak. (equivalent to second number in
    # hessioxxx option --integration-window)
    # sigamp: Amplitude in ADC counts [igain] above pedestal at which a
    # signal is considered as significant (separate for high gain/low gain).
    # (equivalent to hessioxxx option --integration-threshold)
    # clip_amp: Amplitude in p.e. above which the signal is clipped.
    # (equivalent to hessioxxx option --clip_pixel_amplitude (default 0))
    # lwt: Weight of the local pixel (0: peak from neighbours only,
    # 1: local pixel counts as much as any neighbour).
    # (option in pixel integration function in hessioxxx)
    # display: optionaly you can display events (all telescopes present on it)
    # or per telescope per event. By default the last one.
    # The first one is currently deprecated.
    # level: data level from which information is displayed.

    # The next call to camera_calibration would be equivalent of producing
    # DST0 MC file using:
    # hessioxxx/bin/read_hess -r 4 -u --integration-scheme 4
    # --integration-window 7, 3 --integration-threshold 2, 4
    # --dst-level 0 <MC_prod2_filename>

    calibrated_camera = camera_calibration(
        args.filename,
        parameters={"integrator": "nb_peak_integration",
                    "nsum": 7,
                    "nskip": 3,
                    "sigamp": [2, 4],
                    "clip_amp": 0,
                    "lwt": 0},
        disp_args={'event', 'telescope'}, level=1)

    sys.stdout.flush()

    print(TAG, "Closing file...")
    close_file()
