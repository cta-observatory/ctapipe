from time import sleep
import threading
from ctapipe.utils.datasets import get_example_simtelarray_file
from ctapipe.core import Container
from ctapipe import visualization, io
from matplotlib import pyplot as plt
from astropy import units as u
from numpy import ceil, sqrt
import random
import matplotlib.animation as animation
import numpy as np


class DisplayTelid():

    def __init__(self, configuration=None):
        self.configuration = configuration
        self.tel_ids = list()

    def init(self, stager=None):
        print("--- ListTelda init ---")
        self.init_plt = False
        self.nb_tel = 624
        return True

    def init_matplotlib(self):
        self.fig = plt.figure(figsize=(20, 20))
        plt.xlabel('Telescope id')
        plt.ylabel('Probability')
        plt.title(r'$\telescope id$')

        # ax = self.fig.add_subplot(111)
        self.alphab = list(range(625))
        self.frequencies = [0] * 625
        self.maxi = 30
        self.pos = np.arange(len(self.frequencies))
        self.width = 1.0     # gives histogram aspect to the bar diagram
        plt.show(block=False)
        self.fig.clear()
        self.rects = plt.bar(self.pos, self.frequencies, self.width, color='r')
        # ax.set_xticks(self.pos + (self.width / 2))
        # ax.set_xticklabels(self.alphab)
        # self.rects = plt.bar(self.pos, self.frequencies, self.width, color='r')
        # self.fig.canvas.draw()
        plt.yticks(np.arange(0, self.maxi, 1))
        # plt.hist(self.tel_ids,bins=624)
        self.last_update = 11

    def run(self, tel_ids):
        """an extremely inefficient display. It creates new instances of
        CameraDisplay for every event and every camera, and also new axes
        for each event. It's hacked, but it works
        """

        if not isinstance(tel_ids, list):
            return
        # self.tel_ids.extend(tel_ids)
        # print(self.tel_ids)

        if self.init_plt == False:
            self.init_matplotlib()
            self.init_plt = True

        # print("START")
        max_change = False
        for tel_id in tel_ids:
            self.frequencies[tel_id] = self.frequencies[tel_id] + 1
        self.last_update += 1

        if self.last_update > 10:
            for rect, f in zip(self.rects, self.frequencies):
                    rect.set_height(f)
            self.last_update = 0
            plt.pause(.001)

        # self.fig.canvas.draw()

        # print("STOP")

    def finish(self):
        print("--- ListTelda finish ---")
