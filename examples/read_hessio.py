# run this example with:
#
# ipython --pylab read_hessio.py <filename>
#
import sys
import logging
logging.basicConfig(level=logging.DEBUG)

from ctapipe.io.hessio import hessio_event_source
from ctapipe import visualization, io
from matplotlib import pyplot as plt
from astropy import units as u
from numpy import ceil, sqrt
import random


fig = plt.figure(figsize=(10, 10))
cmaps = [plt.cm.jet, plt.cm.winter,
         plt.cm.ocean, plt.cm.bone, plt.cm.gist_earth, plt.cm.hot,
         plt.cm.cool, plt.cm.coolwarm]


def display_event(event):
    """an extremely inefficient display. It creates new instances of
    CameraDisplay for every event and every camera, and also new axes
    for each event. It's hacked, but it works
    """

    global fig
    ntels = len(event.tels_with_data)
    fig.clear()

    plt.suptitle("EVENT {}".format(event.event_id))

    for ii, tel_id in enumerate(event.tels_with_data):
        nn = int(ceil(sqrt(ntels)))
        ax = plt.subplot(nn, nn+1, ii + 1)

        x, y = event.pixel_pos[tel_id]
        geom = io.guess_camera_geometry(x * u.m, y * u.m)
        disp = visualization.CameraDisplay(geom, axes=ax,
                                           title="CT{0}".format(tel_id))
        disp.set_cmap(random.choice(cmaps))
        data = event.sumdata[tel_id][0]
        disp.set_image(data)


def get_input():
    print("============================================")
    print("n or [enter]    - go to next event")
    print("d               - display the event")
    print("p               - print/dump event data")
    print("v               - verbose event info")
    return input("Choice: ")

if __name__ == '__main__':

    print("If you don't see a plot, run this with "
          "'ipython -i --matplotlib read_hessio.py <filename>")

    filename = "/Users/kosack/Data/CTA/Prod2/proton_20deg_180deg_run32364___cta-prod2_desert-1640m-Aar.simtel.gz"

    if len(sys.argv) > 1:
        filename = sys.argv.pop(1)

    plt.style.use("ggplot")
    source = hessio_event_source(filename, max_events=1000000)

    for event in source:

        print("EVENT_ID: ", event.event_id, "TELS: ", event.tels_with_data)

        while True:
            response = get_input()
            if response.startswith("d"):
                display_event(event)
            elif response.startswith("p"):
                print(event)
            elif response == "" or response.startswith("n"):
                break
            elif response.startswith('v'):
                for tel in event.sampledata:
                    for chan in event.sampledata[tel]:
                        npix = len(event.pixel_pos[tel][0])
                        print("CT{:4d} ch{} pixels:{}".format(tel, chan, npix) )

            elif response.startswith('q'):
                break

        if response.startswith('q'):
            break


        
