# Licensed under a 3-clause BSD style license - see LICENSE.rst
from ctapipe.core import Component
from ctapipe.flow.multiprocess.connections import Connections
from multiprocessing import Process
from multiprocessing import Value
from types import GeneratorType
import zmq


class ProducerZmq(Process, Component, Connections):
    """`ProducerZmq` class represents a Producer pipeline Step.
    It is derived from Process class.
    It gets a Python generator from its coroutine run method.
    It loops overs its generator and sends input to its next stage,
    thanks to its ZMQ REQ socket,
    The process is launched by calling run method.
    init() method is call by run method.
    """

    def __init__(self, coroutine, name, main_connection_name,
                 connections=None):
        """
        Parameters
        ----------
        coroutine : Class instance
            It contains init, run and finish methods
        name: str
            Producer name
        main_connection_name : str
            Default next step name. Used to send data when destination is not provided
        connections: dict {'STEP_NANE' : (zmq STEP_NANE port in)}
            Port number for socket for each next steps
        """
        Process.__init__(self)
        Component.__init__(self, parent=None)
        self.name = name
        Connections.__init__(self, main_connection_name, connections)
        self.coroutine = coroutine
        self.other_requests = dict()
        self._nb_job_done = Value('i', 0)
        self._running = Value('i', 0)
        self.done = False

    def init(self):
        """
        Initialise coroutine
        Returns
        -------
        True if coroutine init method returns True, otherwise False
        """
        if self.coroutine is None:
            return False
        if self.coroutine.init() == False:
            return False
        self.coroutine.connections = list(self.connections)
        return self.init_connections()

    def run(self):
        """
        Method representing the process’s activity.
        It gets a Python generator from its coroutine run method.
        It loops overs its generator and sends new input to its next stage,
        thanks to its ZMQ REQ socket.
        """
        if self.init():
            generator = self.coroutine.run()
            if isinstance(generator, GeneratorType):
                for result in generator:
                    self.running = 1
                    self.nb_job_done += 1
                    if isinstance(result, tuple):
                        msg, destination = self.get_destination_msg_from_result(result)
                        self.send_msg(msg, destination)
                    else:
                        self.send_msg(result)
                self.running = 0
            else:
                self.log.warning(
                    "Warning: Productor run method was not a python generator.")
        self.finish()
        self.done = True

    def finish(self):
        """
        Executes coroutine method
        """
        self.coroutine.finish()
        return True

    def init_connections(self):
        """
        Initialise zmq sockets.
        Because this class is s Process, This method must be call in the run
         method to be hold by the correct process.
        """
        self.context = zmq.Context()
        Connections.init_connections(self)
        return True

    @property
    def nb_job_done(self):
        return self._nb_job_done.value

    @nb_job_done.setter
    def nb_job_done(self, value):
        self._nb_job_done.value = value

    @property
    def running(self):
        return self._running.value

    @running.setter
    def running(self, value):
        self._running.value = value
