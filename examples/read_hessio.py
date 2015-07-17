# run this example with:
#
# ipython --pylab read_hessio.py <filename>
#
import sys
import logging
logging.basicConfig(level=logging.DEBUG)

from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe import visualization, io
from matplotlib import pyplot as plt
from astropy import units as u
from numpy import ceil, sqrt
import random


fig = plt.figure(figsize=(12, 8))
cmaps = [plt.cm.jet, plt.cm.winter,
         plt.cm.ocean, plt.cm.bone, plt.cm.gist_earth, plt.cm.hot,
         plt.cm.cool, plt.cm.coolwarm]


def display_event(event):
    """an extremely inefficient display. It creates new instances of
    CameraDisplay for every event and every camera, and also new axes
    for each event. It's hacked, but it works
    """
    print("Displaying... please wait (this is an inefficient implementation)")
    global fig
    ntels = len(event.tels_with_data)
    fig.clear()

    plt.suptitle("EVENT {}".format(event.event_id))

    for ii, tel_id in enumerate(event.tels_with_data):
        print("\t draw cam {}...".format(tel_id))
        nn = int(ceil(sqrt(ntels)))
        ax = plt.subplot(nn, nn, ii + 1)

        x, y = event.pixel_pos[tel_id]
        geom = io.guess_camera_geometry(x * u.m, y * u.m)
        disp = visualization.CameraDisplay(geom, axes=ax,
                                           title="CT{0}".format(tel_id))
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.set_cmap(random.choice(cmaps))
        chan = 0
        data = event.tel_data[tel_id].adc_sums[chan]
        data -= data.mean()
        disp.set_image(data)
        disp.add_colorbar()


def get_input():
    print("============================================")
    print("n or [enter]    - go to Next event")
    print("d               - Display the event")
    print("p               - Print all event data")
    print("i               - event Info")
    print("q               - Quit")
    return input("Choice: ")

if __name__ == '__main__':

    print("If you don't see a plot, run this with "
          "'ipython -i --matplotlib read_hessio.py <filename>")

    if len(sys.argv) > 1:
        filename = sys.argv.pop(1)
    else:
        filename = get_path("gamma_test.simtel.gz")

    plt.style.use("ggplot")
    plt.show(block=False)

    # loop over events and display menu at each event:
    source = hessio_event_source(filename, max_events=1000000)

    for event in source:

        print("EVENT_ID: ", event.event_id, "TELS: ", event.tels_with_data)

        while True:
            response = get_input()
            if response.startswith("d"):
                display_event(event)
                plt.pause(0.1)
            elif response.startswith("p"):
                print(event)
                for teldata in event.tel_data.values():
                    print(teldata)
            elif response == "" or response.startswith("n"):
                break
            elif response.startswith('i'):
                for tel in event.tel_data:
                    for chan in event.tel_data[tel].adc_samples:
                        npix = len(event.pixel_pos[tel][0])
                        print("CT{:4d} ch{} pixels:{} samples:{}"
                              .format(tel, chan, npix,
                                      event.tel_data[tel].
                                      adc_samples[chan].shape[1]))

            elif response.startswith('q'):
                break

        if response.startswith('q'):
            break
