#!/usr/bin/env python3

import sys
import argparse
from matplotlib import colors, pyplot as plt
import numpy as np
from ctapipe.io.hessio import hessio_event_source
from pyhessio import *
from ctapipe.core import Container
from ctapipe.io.containers import RawData, CalibratedCameraData
from ctapipe.instrument import InstrumentDescription as ID
from ctapipe import visualization, io
from astropy import units as u
from ctapipe.calib.camera.mc import *
import time
import logging
from ctapipe.utils.datasets import get_path

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
if __debug__:
    logger.setLevel(logging.DEBUG)

fig = plt.figure(figsize=(16, 7))
cmaps = [plt.cm.jet, plt.cm.winter,
         plt.cm.ocean, plt.cm.bone, plt.cm.gist_earth, plt.cm.hot,
         plt.cm.cool, plt.cm.coolwarm]


def init_dl1(event):
    # Load dl1 container
    container = Container("calibrated_hessio_container")
    container.add_item("dl1", RawData())
    container.meta.add_item('pixel_pos', dict())
    # container.meta.pixel_pos = event.meta.pixel_pos
    container.meta.add_item('optical_foclen', dict())

    return container


def load_dl1_eventheader(dl0, dl1):
    dl1.run_id = dl0.run_id
    dl1.event_id = dl0.event_id
    dl1.tel = dict()  # clear the previous telescopes
    dl1.tels_with_data = dl0.tels_with_data

    return


def display_telescope(event, geom, tel_id):
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
    foclen = event.meta.optical_foclen[tel_id]
    # geom = io.CameraGeometry.guess(x * u.m, y * u.m)
    geom = io.CameraGeometry.guess(x, y, foclen)
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
    chan = 0
    signals = event.dl1.tel[tel_id].pe_charge
    disp.image = signals
    cmaxmin = (max(signals)-min(signals))
    cmap_charge = colors.LinearSegmentedColormap.from_list(
        'cmap_c', [(0/cmaxmin, 'darkblue'),
                   (np.abs(min(signals))/cmaxmin, 'black'),
                   (2.0*np.abs(min(signals))/cmaxmin, 'blue'),
                   (2.5*np.abs(min(signals))/cmaxmin, 'red'),
                   (1, 'yellow')])
    disp.pixels.set_cmap(cmap_charge)
    # disp.pixels.set_cmap('seismic')
    disp.add_colorbar()
    if npads == 2:
        ax = plt.subplot(1, npads, npads)
        disp = visualization.CameraDisplay(geom,
                                           ax=ax,
                                           title="CT{0}".format(tel_id))
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        times = event.dl1.tel[tel_id].tom
        disp.image = times
        tmaxmin = get_num_samples(tel_id)
        t_chargemax = times[signals.argmax()]
        if t_chargemax > 15:
            t_chargemax = 7
        cmap_time = colors.LinearSegmentedColormap.from_list(
            'cmap_t', [(0/tmaxmin, 'black'),
                       (0.6*t_chargemax/tmaxmin, 'red'),
                       (t_chargemax/tmaxmin, 'yellow'),
                       (1.4*t_chargemax/tmaxmin, 'blue'),
                       (1, 'black')])
        # disp.pixels.set_cmap('gnuplot')
        disp.pixels.set_cmap(cmap_time)
        chan = 0
        disp.add_colorbar()


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
    # Number of llops for each calibration function (in debug mode
    # will measure the calibration time)
    nlooptime = 1
    if __debug__:
        nlooptime = 10

    # Load dl1 container
    # container = Container("calibrated_hessio_container")
    # container.add_item("dl1", RawData())
    # container.meta.add_item('pixel_pos', dict())

    tel, cam, opt = ID.load(filename)
    # loop over all events, all telescopes and all channels and call
    # the calc_peds function defined above to do some work:
    nt = 0
    for event in hessio_event_source(filename):
        if nt == 0:
            container = init_dl1(event)

        nt = nt+1
        # Fill DL1 container headers information. Clear also telescope info.
        load_dl1_eventheader(event.dl0, container.dl1)
        if __debug__:
            logger.debug("%s> %d #%d %d %s %.3e TeV @ (%.0f,%.0f)deg @ %.3f m"
                         % (sys._getframe().f_code.co_name,
                            container.dl1.run_id, nt,
                            container.dl1.event_id,
                            str(container.dl1.tels_with_data),
                            get_mc_shower_energy(), get_mc_shower_altitude(),
                            get_mc_shower_azimuth(),
                            np.sqrt(pow(get_mc_event_xcore(), 2) +
                                    pow(get_mc_event_ycore(), 2)))
                         )

        for telid in event.dl0.tels_with_data:
            logger.info("%s> Calibrating.. CT%d\n"
                        % (sys._getframe().f_code.co_name,  telid))

            # Get per telescope the camera geometry
            x, y = event.meta.pixel_pos[telid]
            foclen = event.meta.optical_foclen[telid]
            container.meta.pixel_pos[telid] = x, y
            container.meta.optical_foclen[telid] = foclen
            # geom = io.CameraGeometry.guess(x, y, foclen)

            # Get the calibration data sets (pedestals and single-pe)
            ped = get_pedestal(telid)
            calib = get_calibration(telid)

            # Integrate pixels traces and substract pedestal
            # See pixel_integration_mc function documentation in mc.py
            # for the different algorithms options
            start = time.process_time()
            for i in range(nlooptime):
                int_adc_pix, peak_adc_pix = pixel_integration_mc(event, cam,
                                                                 ped, telid,
                                                                 parameters)
            end = time.process_time()

            logger.debug(" Time pixel integration %.3e sec",
                         (end-start)/nlooptime)

            # Convert integrated ADC counts into p.e.
            # selecting also the HG/LG channel (currently hard-coded)
            start = time.process_time()
            for i in range(nlooptime):
                pe_pix = calibrate_amplitude_mc(int_adc_pix, calib,
                                                telid, parameters)
            end = time.process_time()
            logger.debug(" pixel amplitude calibration %.3e sec",
                         (end-start)/nlooptime)
            # Including per telescope metadata in the DL1 container
            # load_dl1_results(event.dl0,container.dl1)
            # if telid not in container.meta.pixel_pos:
            #    container.meta.pixel_pos[telid] = event.meta.pixel_pos[telid]
            container.dl1.tel[telid] = CalibratedCameraData(telid)
            container.dl1.tel[telid].pe_charge = pe_pix
            container.dl1.tel[telid].tom = peak_adc_pix

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
        if disp_args is None:
            continue
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
                            display_telescope(
                                container,
                                cam['CameraTable_VersionFeb2016_TelID%s' %
                                    telid], telid)
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

    print("Total NEVT", nt)

if __name__ == '__main__':

    # Declare and parse command line option
    parser = argparse.ArgumentParser(
        description='Tel_id, pixel id and number of event to compute.')
    parser.add_argument('--f', dest='filename',
                        default=get_path('gamma_test.simtel.gz'),
                        required=False, help='filename <MC file name>')
    parser.add_argument('--d', dest='display', action='store_true',
                        required=False, help='display the camera events')
    args = parser.parse_args()

    if args.display:
        plt.show(block=False)

    # Function description of camera_calibration options, given here
    # Integrator: samples integration algorithm (equivalent to hessioxxx
    # option --integration-sheme)
    #   -options: full_integration,
    #             simple_integration,
    #             global_peak_integration,
    #             local_peak_integration,
    #             nb_peak_integration
    # window: Number of samples to sum up (is reduced if exceeding available
    # length). (equivalent to first number in
    # hessioxxx option --integration-window)
    # shift: Number of initial samples skipped (adapted such that interval
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

    parameters = {"integrator": "nb_peak_integration",
                  "window": 7,
                  "shift": 3,
                  "sigamp": [2, 4],
                  "clip_amp": 0,
                  "lwt": 0}
    if args.display:
        disp_args = {'event', 'telescope'}
    else:
        disp_args = None

    calibrated_camera = camera_calibration(args.filename,
                                           parameters,
                                           disp_args, level=1)

    sys.stdout.flush()

    logger.info("%s> Closing file..." % sys._getframe().f_code.co_name)
    close_file()
