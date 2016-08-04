"""
Create a plot of where the integration window lays on the trace for the pixel
with the highest charge, its neighbours, and the pixel with the lowest max
charge and its neighbours. Also shows a disgram of which pixels count as a
neighbour, the camera image for the max charge timeslice, the true pe camera
image, and a calibrated camera image
"""

import argparse
from ctapipe.calib.camera.calibrators import calibration_arguments, \
    calibration_parameters
from astropy import log
from ctapipe.calib.camera.calibrators import calibrate_event, calibrate_source
from ctapipe.io.files import InputFile, origin_list
import numpy as np


def get_source(filepath):
    from ctapipe.utils.datasets import get_path
    from ctapipe.io.hessio import hessio_event_source
    source = hessio_event_source(get_path(filepath))
    return source


def main():
    parser = argparse.ArgumentParser(description='Create a gif of an event')
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the input file')
    parser.add_argument('-O', '--origin', dest='origin', action='store',
                        required=True, help='origin of the file: {}'
                        .format(origin_list()))
    parser.add_argument('-o', '--output', dest='output_path', action='store',
                        required=True, help='path of the output image file')
    parser.add_argument('-e', '--event', dest='event_req', action='store',
                        required=True, type=int,
                        help='event index to plot (not id!)')
    parser.add_argument('--id', dest='event_id_f', action='store_true',
                        default=False, help='-e will specify event_id instead '
                                            'of index')
    parser.add_argument('-t', '--telescope', dest='tel', action='store',
                        type=int, default=None, help='telecope to view')
    parser.add_argument('-c', '--channel', dest='chan', action='store',
                        type=int, default=0,
                        help='channel to view (default = 0 (HG))')

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

    params = calibration_parameters(args)

    if args.quiet:
        log.setLevel(40)
    if args.verbose:
        log.setLevel(20)
    if args.debug:
        log.setLevel(10)

    log.info("[SCRIPT] write_img_fits")

    log.debug("[file] Reading file")
    input_file = InputFile(args.input_path, args.origin)
    event = input_file.get_event(args.event_req, args.event_id_f)
    calibrated_event = calibrate_event(event, params)

    tels = list(calibrated_event.dl0.tels_with_data)
    if args.tel is None or args.tel not in tels:
        log.error("[event] please specify one of the following telescopes "
                  "for this event: {}".format(tels))
        exit()

    data_ped = calibrated_event.dl1.tel[args.tel]\
                .pedestal_subtracted_adc[args.chan]
    print(data_ped.shape)
    max_charges = np.argmax(data_ped, axis=1)

    print(max_charges.shape)






if __name__ == '__main__':
    main()