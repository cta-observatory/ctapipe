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


fig = plt.figure(figsize=(10, 10))


def display_event(event):
    """a really hacked display. It is very inefficient and slow becasue
    it creates new instances of CameraDisplay for every event and every
    camera"""

    global fig
    ntels = len(event.tels_with_data)
    fig.clear()

    plt.suptitle("EVENT {}".format(event.event_id))

    for ii, tel_id in enumerate(event.tels_with_data):
        if ntels <= 9:
            ax = plt.subplot(3, 3, ii + 1)
        else:
            nn = int(ceil(sqrt(ntels)))
            ax = plt.subplot(nn, nn, ii + 1)

        x, y = event.pixel_pos[tel_id]
        geom = io.make_camera_geometry(x * u.m, y * u.m, "hexagonal")
        disp = visualization.CameraDisplay(geom, axes=ax,
                                           title="CT{0}".format(tel_id))
        data = event.sumdata[tel_id][0]
        disp.set_image(data)


if __name__ == '__main__':

    filename = "/Users/kosack/Data/CTA/Prod2/proton_20deg_180deg_run32364___cta-prod2_desert-1640m-Aar.simtel.gz"

    if len(sys.argv) > 1:
        filename = sys.argv.pop(1)

    plt.style.use("ggplot")
    source = hessio_event_source(filename, max_events=1000000)

    for event in source:

        print("=" * 70)
        print("EVENT_ID: ", event.event_id)
        print("TELS: ", event.tels_with_data)

        for tel in event.sampledata:
            print("-" * 50)
            for chan in event.sampledata[tel]:
                npix = len(event.pixel_pos[tel][0])
                print("CT{:4d} ch{} pixels:{}".format(tel, chan, npix)
                      )

        display_event(event)
        a = input("press enter for next event")
