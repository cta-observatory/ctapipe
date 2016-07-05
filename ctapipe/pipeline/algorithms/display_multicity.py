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

class DisplayMulticity():
    def __init__(self,configuration=None):
        self.configuration = configuration
        self.tel_ids = list()
        self.max_freq = 0
        self.max_nb_tel = 0

    def init(self,stager=None):
        print("--- ListTelda init ---")
        self.init_plt = False
        return True

    def init_matplotlib(self):
        self.fig = plt.figure(figsize=(20, 20))
        plt.xlabel('Telescope id')
        plt.ylabel('Probability')
        plt.title(r'$\telescope id$')


        #ax = self.fig.add_subplot(111)
        self.alphab = list(range(1))
        self.frequencies = [0]*1
        self.pos = np.arange(len(self.frequencies))
        self.width = 1.0     # gives histogram aspect to the bar diagram
        plt.show(block=False)
        self.fig.clear()
        self.rects = plt.bar(self.pos, self.frequencies, self.width, color='r')
        #ax.set_xticks(self.pos + (self.width / 2))
        #ax.set_xticklabels(self.alphab)
        #self.rects = plt.bar(self.pos, self.frequencies, self.width, color='r')
        #self.fig.canvas.draw()
        plt.yticks(np.arange(0, 30, 1))
        #plt.hist(self.tel_ids,bins=624)
        self.last_update = 11


    def run(self,tel_ids):
        """
        """
        if not isinstance(tel_ids,list):
            return

        if self.init_plt == False:
            self.init_matplotlib()
            self.init_plt = True

        max_change = False
        nb_tel = len(tel_ids)

        try:
            self.frequencies[nb_tel] = self.frequencies[nb_tel] + 1
        except IndexError:
            foo = [0]*(nb_tel+2)
            for idx in range(len(foo)):
                try:
                    foo[idx] = self.frequencies[idx]
                except IndexError:
                    foo[idx] = 0
            self.frequencies = list(foo)
            self.frequencies[nb_tel] = self.frequencies[nb_tel] + 1
            self.pos = np.arange(len(self.frequencies))
            self.rects = plt.bar(self.pos, self.frequencies, self.width, color='r')

        if self.last_update > 20:
            max_change = False
            for rect, f in zip(self.rects, self.frequencies):
                    rect.set_height(f)
                    if f > self.max_freq:
                        self.max_freq = f
                        max_change = True
                        plt.yticks(np.arange(0, self.max_freq,  int(self.max_freq/10)+1))
                    if nb_tel > self.max_nb_tel:
                        self.max_nb_tel = nb_tel
                        max_change = True
            self.last_update = 0
            plt.pause(.001)
        self.last_update+=1

        if max_change :
            self.pos = np.arange(len(self.frequencies))
            self.rects = plt.bar(self.pos, self.frequencies, self.width, color='r')

    def finish(self):
        print("--- ListTelda finish ---")
