from time import sleep
import threading
from ctapipe.utils.datasets import get_example_simtelarray_file
from ctapipe.core import Container
from ctapipe import visualization, io
from matplotlib import pyplot as plt
from astropy import units as u
import pickle
from numpy import ceil, sqrt
import random


class WriteFile():

    def __init__(self, configuration=None):
        self.configuration = configuration
        self.events = list()

    def init(self, stager=None):
        return True

    def run(self, _input):
        if isinstance(_input, Container):
            self.events.append(_input)
        else:
            outfile = open(_input, "wb")
            objects = self.events
            pickle.dump(objects, outfile)
            outfile.close()

    def finish(self):

        print("--- WriteFile finish ---")
