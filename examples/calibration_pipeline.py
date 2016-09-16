#!/usr/bin/env python3
"""
Script to demonstrate how to best obtain calibrated data.
Displays the calibrated charge for each event in the file.
"""
import argparse
from matplotlib import colors, pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from ctapipe.calib.camera.calibrators import calibration_parameters, \
    calibrate_source, calibration_arguments
from astropy import log
from ctapipe.utils.datasets import get_path
from ctapipe.plotting.camera import CameraPlotter
from ctapipe.io.files import InputFile, origin_list
import os

def display_telescope(event, tel_id, display, geom_dict, pp, fig):
    fig.clear()

    cam_dimensions = (event.dl0.tel[tel_id].num_pixels,
                      event.meta.optical_foclen[tel_id])

    fig.suptitle("EVENT {} {:.1e} @({:.1f},{:.1f}) @{:.1f}"
                 .format(event.dl1.event_id, event.mc.energy,
                         event.mc.alt,
                         event.mc.az,
                         np.sqrt(pow(event.mc.core_x, 2) +
                                 pow(event.mc.core_y, 2))))

    # Select number of pads to display (will depend on the integrator):
    # charge and/or time of maximum.
    # This last one is displayed only if the integrator calculates it
    npads = 1
    if not event.dl1.tel[tel_id].peakpos[0] is None:
        npads = 2
    # Only create two pads if there is timing information extracted
    # from the calibration
    ax1 = fig.add_subplot(1, npads, 1)

    # If the geometery has not already been added to geom_dict, it will
    # be added in CameraPlotter
    plotter = CameraPlotter(event, geom_dict)
    signals = event.dl1.tel[tel_id].pe_charge
    camera1 = plotter.draw_camera(tel_id, signals, ax1)
    cmaxmin = (max(signals) - min(signals))
    cmap_charge = colors.LinearSegmentedColormap.from_list(
        'cmap_c', [(0 / cmaxmin, 'darkblue'),
                   (np.abs(min(signals)) / cmaxmin, 'black'),
                   (2.0 * np.abs(min(signals)) / cmaxmin, 'blue'),
                   (2.5 * np.abs(min(signals)) / cmaxmin, 'green'),
                   (1, 'yellow')])
    camera1.pixels.set_cmap(cmap_charge)
    camera1.add_colorbar(ax=ax1, label=" [photo-electrons]")
    ax1.set_title("CT {} ({}) - Mean pixel charge"
                  .format(tel_id, geom_dict[cam_dimensions].cam_id))
    if not event.dl1.tel[tel_id].peakpos[0] is None:
        ax2 = fig.add_subplot(1, npads, npads)
        times = event.dl1.tel[tel_id].peakpos
        camera2 = plotter.draw_camera(tel_id, times, ax2)
        tmaxmin = event.dl0.tel[tel_id].num_samples
        t_chargemax = times[signals.argmax()]
        if t_chargemax > 15:
            t_chargemax = 7
        cmap_time = colors.LinearSegmentedColormap.from_list(
            'cmap_t', [(0 / tmaxmin, 'darkgreen'),
                       (0.6 * t_chargemax / tmaxmin, 'green'),
                       (t_chargemax / tmaxmin, 'yellow'),
                       (1.4 * t_chargemax / tmaxmin, 'blue'),
                       (1, 'darkblue')])
        camera2.pixels.set_cmap(cmap_time)
        camera2.add_colorbar(ax=ax2, label="[time slice]")
        ax2.set_title("CT {} ({}) - Pixel peak position"
                      .format(tel_id, geom_dict[cam_dimensions].cam_id))

    if display:
        plt.pause(0.1)
    if pp is not None:
        pp.savefig(fig)


def main():
    script = os.path.splitext(os.path.basename(__file__))[0]
    log.info("[SCRIPT] {}".format(script))

    parser = argparse.ArgumentParser(
        description='Display each event in the file')
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        default=get_path('gamma_test.simtel.gz'),
                        help='path to the input file. '
                             'Default = gamma_test.simtel.gz')
    parser.add_argument('-O', '--origin', dest='origin', action='store',
                        default='hessio',
                        help='origin of the file: {}. Default = hessio'
                        .format(origin_list()))
    parser.add_argument('-D', dest='display', action='store_true',
                        default=False, help='display the camera events')
    parser.add_argument('--pdf', dest='output_path', action='store',
                        default=None,
                        help='path to store a pdf output of the plots')
    parser.add_argument('-t', '--telescope', dest='tel', action='store',
                        type=int, default=None, help='telecope to view. '
                                                     'Default = All')

    calibration_arguments(parser)

    logger_detail = parser.add_mutually_exclusive_group()
    logger_detail.add_argument('-q', '--quiet', dest='quiet',
                               action='store_true', default=False,
                               help='Quiet mode')
    logger_detail.add_argument('-v', '--verbose', dest='verbose',
                               action='store_true', default=False,
                               help='Verbose mode')
    logger_detail.add_argument('-d', '--debug', dest='debug',
                               action='store_true', default=False,
                               help='Debug mode')

    args = parser.parse_args()
    print('DEBUG type(args) {}'.format(type(args)))
    print('DEBUG args {}'.format(args))
    params = calibration_parameters(args)

    if args.quiet:
        log.setLevel(40)
    if args.verbose:
        log.setLevel(20)
    if args.debug:
        log.setLevel(10)

    log.debug("[file] Reading file")
    input_file = InputFile(args.input_path, args.origin)
    source = input_file.read()

    # geom_dict is a dictionary of CameraGeometry, with keys of
    # (num_pixels, focal_length), the parameters that are used to guess the
    # geometry of the telescope. By using these keys, the geometry is
    # calculated only once per telescope type as needed, reducing computation
    # time.
    # Creating a geom_dict at this point is optional, but is recommended, as
    # the same geom_dict can then be shared between the calibration and
    # CameraPlotter, again reducing computation time.
    # The dictionary becomes filled as a result of a dictionary's mutable
    # nature.
    geom_dict = {}

    # Calibrate events and fill geom_dict

    calibrated_source = calibrate_source(source, params, geom_dict)

    fig = plt.figure(figsize=(16, 7))
    if args.display:
        plt.show(block=False)
    pp = PdfPages(args.output_path) if args.output_path is not None else None
    for event in calibrated_source:
        tels = list(event.dl0.tels_with_data)
        if args.tel is None:
            tel_loop = tels
        else:
            if args.tel not in tels:
                continue
            tel_loop = [args.tel]
        log.debug(tels)
        for tel_id in tel_loop:
            display_telescope(event, tel_id, args.display, geom_dict, pp, fig)
    if pp is not None:
        pp.close()

    log.info("[COMPLETE]")

if __name__ == '__main__':
    main()
