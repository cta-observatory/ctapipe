# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding: utf8

from time import time
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
        self.coroutine = coroutine
        self.running = 0
        self.nb_job_done = 0
        self.total_time = 0


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
        start_time = time()
        if self.coroutine.init() == False:
            return False
        end_time = time()
        self.total_time += (end_time - start_time)
        return True

    def run(self,inputs=None):
        """ Executes coroutine run method

        Parameters
        ----------
        inputs: input for coroutine.run
        """
        start_time = time()
        self.coroutine.run(inputs)
        end_time = time()
        self.total_time += (end_time - start_time)
        self.nb_job_done+=1

    def finish(self):
        """
        Call coroutine finish method
        """
        start_time = time()
        self.coroutine.finish()
        end_time = time()
        self.total_time += (end_time - start_time)
        return True
