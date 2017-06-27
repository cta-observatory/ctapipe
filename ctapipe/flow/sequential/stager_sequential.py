# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding: utf8
from types import GeneratorType


class StagerSequential():

    """`StagerSequential` class represents a Stager pipeline Step.
    """

    def __init__(self, coroutine, name=None, connections=None, main_connection_name=None):
        """
        Parameters
        ----------
        coroutine : Class instance that contains init, run and finish methods
        connections: list(str)
            define next available steps
        """
        self.name = name
        self.coroutine = coroutine
        self.main_connection_name = main_connection_name
        self.connections = connections or []
        self.running = 0
        self.nb_job_done = 0

    def init(self):
        """
        Initialise coroutine sockets and poller

        Returns
        -------
        True if coroutine init method returns True, otherwise False
        """
        if self.name is None:
            self.name = "STAGER"
        if self.coroutine is None:
            return False
        if self.coroutine.init() == False:
            return False
        self.coroutine.connections = self.connections
        return True

    def run(self, inputs=None):
        """ Executes coroutine run method

        Parameters
        ----------
        inputs: object
             input for coroutine.run
        """
        result = self.coroutine.run(inputs)
        if isinstance(result, GeneratorType):
            for val in result:
                msg, destination = self.get_destination_msg_from_result(val)
                yield (msg, destination)
        else:
            msg, destination = self.get_destination_msg_from_result(result)
            yield (msg, destination)
        self.nb_job_done += 1

    def get_destination_msg_from_result(self, result):
        """If result is a tuple, check if last tuple elem is a valid next
        step.  If yes, return a destination defined to the last tuple
        elem and send result without the destination If no return None
        as destination

        Parameters
        ----------
        result : any type
            value to send (can contain next step name)

        Returns
        -------
        msg, destination

        """
        destination = self.main_connection_name
        if isinstance(result, tuple):
            # look is last tuple elem is a valid next step
            if result[-1] in self.connections.keys():
                destination = result[-1]
                if len(result[:-1]) == 1:
                    msg = result[:-1][0]
                else:
                    msg = result[:-1]
                return msg, destination
            else:
                return result, destination
        else:
            return result, destination

    def finish(self):
        """
        Call coroutine finish method
        """
        self.coroutine.finish()
        return True
