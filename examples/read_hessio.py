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


def display_event(data):
    """an extremely inefficient display. It creates new instances of
    CameraDisplay for every event and every camera, and also new axes
    for each event. It's hacked, but it works
    """
    print("Displaying... please wait (this is an inefficient implementation)")
    global fig
    ntels = len(data.dl0.tels_with_data)
    fig.clear()

    plt.suptitle("EVENT {}".format(data.dl0.event_id))

    for ii, tel_id in enumerate(data.dl0.tels_with_data):
        print("\t draw cam {}...".format(tel_id))
        nn = int(ceil(sqrt(ntels)))
        ax = plt.subplot(nn, nn, ii + 1)

        x, y = data.meta.pixel_pos[tel_id]
        geom = io.guess_camera_geometry(x * u.m, y * u.m)
        disp = visualization.CameraDisplay(geom, axes=ax,
                                           title="CT{0}".format(tel_id))
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.set_cmap(random.choice(cmaps))
        chan = 0
        signals = data.dl0.tel[tel_id].adc_sums[chan]
        signals -= signals.mean()
        disp.set_image(signals)
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

    for data in source:

        print("EVENT_ID: ", data.dl0.event_id, "TELS: ",
              data.dl0.tels_with_data)

        while True:
            response = get_input()
            if response.startswith("d"):
                display_event(data)
                plt.pause(0.1)
            elif response.startswith("p"):
                print(data)
                for teldata in data.dl0.tel.values():
                    print(teldata)
            elif response == "" or response.startswith("n"):
                break
            elif response.startswith('i'):
                for tel_id in sorted(data.dl0.tel):
                    for chan in data.dl0.tel[tel_id].adc_samples:
                        npix = len(data.meta.pixel_pos[tel_id][0])
                        print("CT{:4d} ch{} pixels:{} samples:{}"
                              .format(tel, chan, npix,
                                      event.tel[tel_id].
                                      adc_samples[chan].shape[1]))

            elif response.startswith('q'):
                break

        if response.startswith('q'):
            break
