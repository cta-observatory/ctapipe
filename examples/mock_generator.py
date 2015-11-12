# run this example with:
#
# python read_hessio.py <filename>
#
# if no filename is given, a default example file will be used
# containing ~10 events
#

from ctapipe.io.mock import mock_event_source
from ctapipe import visualization, io
from matplotlib import pyplot as plt
from astropy import units as u
from numpy import ceil, sqrt
import random

import logging
logging.basicConfig(level=logging.DEBUG)


def display_event(event):
    """an extremely inefficient display. It creates new instances of
    CameraDisplay for every event and every camera, and also new axes
    for each event. It's hacked, but it works
    """
    print("Displaying... please wait (this is an inefficient implementation)")
    global fig
    ntels = len(event.dl0.tels_with_data)
    fig.clear()

    plt.suptitle("EVENT {}".format(event.dl0.event_id))

    disps = []

    for ii, tel_id in enumerate(event.dl0.tels_with_data):
        print("\t draw cam {}...".format(tel_id))
        nn = int(ceil(sqrt(ntels)))
        ax = plt.subplot(nn, nn, ii + 1)

        x, y = event.meta.pixel_pos[tel_id]
        geom = io.CameraGeometry.guess(x * u.m, y * u.m)
        disp = visualization.CameraDisplay(geom, ax=ax,
                                           title="CT{0}".format(tel_id))
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.cmap = 'afmhot'
        chan = 0
        signals = event.dl0.tel[tel_id].adc_sums[chan].astype(float)
        signals -= signals.mean()
        disp.image = signals
        disp.set_limits_percent(95)
        disp.add_colorbar()
        disps.append(disp)

    return disps


def get_input():
    print("============================================")
    print("n or [enter]    - go to Next event")
    print("d               - Display the event")
    print("p               - Print all event data")
    print("i               - event Info")
    print("q               - Quit")
    return input("Choice: ")

if __name__ == '__main__':

    fig = plt.figure(figsize=(12, 8))
    plt.style.use("ggplot")
    plt.show(block=False)

    n_telescopes = 20
    geom = io.CameraGeometry.from_name('hess', 1)
    geoms = [geom for i in range(n_telescopes)]
    source = mock_event_source(geoms)

    for event in source:

        print("EVENT_ID: ", event.dl0.event_id, "TELS: ",
              event.dl0.tels_with_data)

        while True:
            response = get_input()
            if response.startswith("d"):
                disps = display_event(event)
                plt.pause(0.1)
            elif response.startswith("p"):
                print("--event-------------------")
                print(event)
                print("--event.dl0---------------")
                print(event.dl0)
                print("--event.dl0.tel-----------")
                for teldata in event.dl0.tel.values():
                    print(teldata)
            elif response == "" or response.startswith("n"):
                break
            elif response.startswith('i'):
                for tel_id in sorted(event.dl0.tel):
                    for chan in event.dl0.tel[tel_id].adc_samples:
                        npix = len(event.meta.pixel_pos[tel_id][0])
                        print("CT{:4d} ch{} pixels:{} samples:{}"
                              .format(tel_id, chan, npix,
                                      event.dl0.tel[tel_id].
                                      adc_samples[chan].shape[1]))

            elif response.startswith('q'):
                break

        if response.startswith('q'):
            break
