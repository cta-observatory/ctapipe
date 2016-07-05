from time import sleep
import threading
from ctapipe.utils.datasets import get_example_simtelarray_file
from ctapipe.core import Container
from ctapipe import visualization, io
from matplotlib import pyplot as plt
from astropy import units as u
from numpy import ceil, sqrt
import random


class DisplayEvent():
    def __init__(self,configuration=None):
        self.configuration = configuration



    def init(self,stager=None):
        print("--- ListTelda init ---")
        self.init_plt = False
        return True

    def init_matplotlib(self):
        self.fig = plt.figure()
        self.cmaps = [plt.cm.jet, plt.cm.winter,
                 plt.cm.ocean, plt.cm.bone, plt.cm.gist_earth, plt.cm.hot,
                 plt.cm.cool, plt.cm.coolwarm]
        plt.style.use("ggplot")
        plt.show(block=False)


    def run(self,event):
        """an extremely inefficient display. It creates new instances of
        CameraDisplay for every event and every camera, and also new axes
        for each event. It's hacked, but it works
        """
        if not isinstance(event,Container):
            print("Warning event is not instance of ctapipe.core.Container", event)
            return
        """
        if self.init_plt == False:
            self.init_matplotlib()
            self.init_plt = True
        print("Displaying... please wait (this is an inefficient implementation)")
        #global fig
        ntels = len(event.dl0.tels_with_data)
        self.fig.clear()

        plt.suptitle("EVENT {}".format(event.dl0.event_id))
        disps = []
        for ii, tel_id in enumerate(event.dl0.tels_with_data):
            print("\t draw cam {}...".format(tel_id))
            nn = int(ceil(sqrt(ntels)))
            ax = plt.subplot(nn, nn, ii + 1)

            x, y = event.meta.pixel_pos[tel_id]
            geom = io.CameraGeometry.guess(x, y, event.meta.optical_foclen[tel_id])
            disp = visualization.CameraDisplay(geom, ax=ax,
                                               title="CT{0}".format(tel_id))
            disp.pixels.set_antialiaseds(False)
            disp.autoupdate = False
            disp.cmap = random.choice(self.cmaps)
            chan = 0
            signals = event.dl0.tel[tel_id].adc_sums[chan].astype(float)
            signals -= signals.mean()
            disp.image = signals
            disp.set_limits_percent(95)
            disp.add_colorbar()
            disps.append(disp)
            self.fig.canvas.draw()
            plt.pause(1)
        """


    def finish(self):
        print("--- ListTelda finish ---")
