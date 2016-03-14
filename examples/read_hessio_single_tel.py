"""Example of extracting a single telescope from a merged/interleaved
simtelarray data file.

Only events that contain the specified telescope are read and
displayed. Other telescopes and events are skipped over (EventIO data
files have no index table in them, so the events must be read in
sequence to find ones with the appropriate telescope, therefore this
is not a fast operation)

"""
from ctapipe.utils.datasets import get_example_simtelarray_file
from ctapipe.io.hessio import hessio_event_source
from ctapipe import visualization, io
from matplotlib import pyplot as plt
from astropy import units as u
import pyhessio

import logging
import argparse
logging.basicConfig(level=logging.DEBUG)


def get_mc_calibration_coeffs(tel_id):
    """
    Get the calibration coefficients from the MC data file to the
    data.  This is ahack (until we have a real data structure for the
    calibrated data), it should move into `ctapipe.io.hessio_event_source`.

    returns
    -------
    (peds,gains) : arrays of the pedestal and pe/dc ratios.
    """
    peds = pyhessio.get_pedestal(tel_id)[0]
    gains = pyhessio.get_calibration(tel_id)[0]
    return peds, gains


def apply_mc_calibration(adcs, tel_id):
    """
    apply basic calibration
    """
    peds, gains = get_mc_calibration_coeffs(tel_id)
    return (adcs - peds) * gains


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('tel', metavar='TEL_ID', type=int)
    parser.add_argument('filename', metavar='EVENTIO_FILE', nargs='?',
                        default=get_example_simtelarray_file())
    parser.add_argument('-m', '--max-events', type=int, default=10)
    parser.add_argument('-c', '--channel', type=int, default=0)
    parser.add_argument('-w', '--write', action='store_true',
                        help='write images to files')
    parser.add_argument('-s', '--show-samples', action='store_true',
                        help='show time-variablity, one frame at a time')
    args = parser.parse_args()

    source = hessio_event_source(args.filename,
                                 allowed_tels=[args.tel, ],
                                 max_events=args.max_events)
    disp = None

    print('SELECTING EVENTS FROM TELESCOPE {}'.format(args.tel))
    print('=' * 70)

    for event in source:

        print('Scanning input file... count = {}'.format(event.count))
        print(event.trig)
        print(event.mc)
        print(event.dl0)

        if disp is None:
            x, y = event.meta.pixel_pos[args.tel]
            geom = io.CameraGeometry.guess(x, y)
            disp = visualization.CameraDisplay(geom, title='CT%d' % args.tel)
            disp.enable_pixel_picker()
            disp.add_colorbar()
            plt.show(block=False)

        # display the event
        disp.axes.set_title('CT{:03d}, event {:010d}'
                            .format(args.tel, event.dl0.event_id))
        if args.show_samples:
            # display time-varying event
            data = event.dl0.tel[args.tel].adc_samples[args.channel]
            for ii in range(data.shape[1]):
                disp.image = apply_mc_calibration(data[:, ii], args.tel)
                disp.set_limits_percent(70)
                plt.pause(0.01)
                if args.write:
                    plt.savefig('CT{:03d}_EV{:010d}_S{:02d}.png'
                                .format(args.tel, event.dl0.event_id, ii))
        else:
            # display integrated event:
            disp.image = apply_mc_calibration(
                event.dl0.tel[args.tel].adc_sums[args.channel], args.tel)
            disp.set_limits_percent(70)
            plt.pause(0.1)
            if args.write:
                plt.savefig('CT{:03d}_EV{:010d}.png'
                            .format(args.tel, event.dl0.event_id))

    print("FINISHED READING DATA FILE")

    if disp is None:
        print('No events for tel {} were found in {}. Try a different'
              .format(args.tel, args.filename),
              'EventIO file or another telescope')
