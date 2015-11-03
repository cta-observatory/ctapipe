"""Example of extracting a single telescope from a merged/interleaved
simtelarray data file.

Only events that contain the specified telescope are read and
displayed. Other telescopes and events are skipped over (EventIO data
files have no index table in them, so the events must be read in
sequence to find ones with the appropriate telescope, therefore this
is not a fast operation)

"""
from ctapipe.utils.datasets import get_datasets_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe import visualization, io
from matplotlib import pyplot as plt
from astropy import units as u
from numpy import zeros_like

import logging
import argparse
logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('tel', metavar='TEL_ID', type=int)
    parser.add_argument('filename', metavar='EVENTIO_FILE', nargs='?',
                        default=get_datasets_path('gamma_test.simtel.gz'))
    parser.add_argument('-m', '--max-events', type=int, default=10)
    parser.add_argument('-c', '--channel', type=int, default=0)
    args = parser.parse_args()

    source = hessio_event_source(args.filename,
                                 single_tel=args.tel,
                                 max_events=args.max_events)
    disp = None

    print('SELECTING EVENTS FROM TELESCOPE {}'.format(args.tel))
    print('=' * 70)

    for event in source:

        print('Scanning input file... count = {}'.format(event.count))
        print(event.dl0)

        if disp is None:
            x, y = event.meta.pixel_pos[args.tel]
            geom = io.CameraGeometry.guess(x * u.m, y * u.m)
            disp = visualization.CameraDisplay(geom, title='CT%d' % args.tel)
            disp.enable_pixel_picker()
            disp.add_colorbar()
            plt.show(block=False)

        # display the event
        disp.image = event.dl0.tel[args.tel].adc_sums[args.channel]
        disp.axes.set_title('CT{:03d}, event {:010d}'
                            .format(args.tel, event.dl0.event_id))
        disp.set_limits_percent(70)
        plt.pause(0.1)

    print("FINISHED READING DATA FILE")

    if disp is None:
        print('No events for tel {} were found in {}. Try a different'
              .format(args.tel, args.filename),
              'EventIO file or another telescope')
