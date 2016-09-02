# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding: utf8
from time import sleep
from time import time
import zmq
import types

import pickle
from ctapipe.flow.multiprocessus.connexions import Connexions


class ConsumerSequential():

    """`ConsumerSequential` class represents a Consumer pipeline Step.
    """

    def __init__(
            self, coroutine, name=None):
        """
        Parameters
        ----------
        coroutine : Class instance that contains init, run and finish methods
        """
        self.name = name
        # Set coroutine
        self.coroutine = coroutine
        self.running = False
        self.nb_job_done = 0


    def init(self):
        """
        Initialise coroutine sockets and poller
        Returns
        -------
        True if coroutine init method returns True, otherwise False
        """
        if self.name is None:
            self.name = "Consumer"
        if self.coroutine is None:
            return False
        if self.coroutine.init() == False:
            return False
        return True

    def run(self,inputs=None):
        self.coroutine.run(inputs)
        self.nb_job_done+=1

    def finish(self):
        """
        """
        self.coroutine.finish()
        return True
