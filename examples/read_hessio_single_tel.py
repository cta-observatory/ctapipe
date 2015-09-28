"""Example of extracting a single telescope from a merged/interleaved
simtelarray data file.

Only events that contain the specified telescope are read and
displayed. Other telescopes and events are skipped over (EventIO data
files have no index table in them, so the events must be read in
sequence to find ones with the appropriate telescope, therefore this
is not a fast operation)

"""
import sys
import logging
logging.basicConfig(level=logging.DEBUG)

from ctapipe.utils.datasets import get_datasets_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe import visualization, io
from matplotlib import pyplot as plt
from astropy import units as u


if __name__ == '__main__':

    if len(sys.argv) > 1:
        filename = sys.argv.pop(1)
    else:
        filename = get_datasets_path("gamma_test.simtel.gz")

    tel = 25

    # loop over events and display menu at each event:
    source = hessio_event_source(filename, single_tel=tel, max_events=100)
    disp = None

    print("SELECTING EVENTS FROM TELESCOPE {}".format(tel))
    print("=" * 70)

    for event in source:

        print("COUNT: {}".format(event.count))
        print(event.dl0)

        if disp is None:
            x, y = event.meta.pixel_pos[tel]
            geom = io.guess_camera_geometry(x * u.m, y * u.m)
            disp = visualization.CameraDisplay(geom, title="CT{}".format(tel))

        disp.set_image(event.dl0.tel[tel].adc_sums[0])
        plt.pause(0.1)
