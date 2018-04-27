# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding: utf8
from types import GeneratorType
from multiprocessing import Process
from multiprocessing import Value
from pickle import loads
from zmq import POLLIN
from zmq import REQ
from zmq import Poller
from zmq import Context
from ctapipe.flow.multiprocess.connections import Connections
from ctapipe.core import Component


class StagerZmq(Component, Process, Connections):

    """`StagerZmq` class represents a Stager pipeline Step.
    It is derived from Process class.
    It receives new input from its prev stage, thanks to its ZMQ REQ socket,
    and executes its coroutine objet's run method by passing
    input as parameter. Finaly it sends coroutine returned value to its next
    stage, thanks to its ZMQ REQ socket,
    The process is launched by calling run method.
    init() method is call by run method.
    The process is stopped by setting share data stop to True
    """

    def __init__(
            self, coroutine, sock_job_for_me_port,
            name=None, connections=None, main_connection_name=None):
        """
        Parameters
        ----------
        coroutine : Class instance that contains init, run and finish methods
        sock_job_for_me_port: str
            Port number for input socket url
        name: str
            Stage name
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
        self.sock_job_for_me_url = 'tcp://localhost:' + sock_job_for_me_port
        self.done = False
        self.waiting_since = Value('i', 0)
        self._nb_job_done = Value('i', 0)
        self._stop = Value('i', 0)
        self._running = Value('i', 0)

    def init(self):
        """
        Initialise coroutine sockets and poller
        Returns
        -------
        True if coroutine init and init_connections methods returns True,
         otherwise False
        """
        if self.name is None:
            self.name = "STAGER"
        if self.coroutine is None:
            return False
        if self.coroutine.init() == False:
            return False

        self.coroutine.connections = list(self.connections)

        return self.init_connections()

    def run(self):
        """
        Method representing the process's activity.
        It polls its socket and when received new input from it,
        it executes coroutine run method by passing new input.
        Then it sends coroutine return value to its next stage,
        thanks to its ZMQ REQ socket.
        The poll method's timeout is 100 ms in case of self.stop flag
        has been set to False.
        Atfer the main while loop, coroutine.finish method is called
        """
        if self.init():
            while not self.stop:
                sockets = dict(self.poll.poll(100))  # Poll or time out (100ms)
                if (self.sock_for_me in sockets and
                        sockets[self.sock_for_me] == POLLIN):
                    #  Get the input from prev_stage
                    self.waiting_since.value = 0
                    self.running = 1
                    request = self.sock_for_me.recv_multipart()
                    receiv_input = loads(request[0])
                    # do the job
                    results = self.coroutine.run(receiv_input)
                    if isinstance(results, GeneratorType):
                        for val in results:
                            msg, destination = self.get_destination_msg_from_result(val)
                            self.send_msg(msg, destination)
                    else:
                        msg, destination = self.get_destination_msg_from_result(results)
                        self.send_msg(msg, destination)
                    # send acknoledgement to prev router/queue to inform it that I
                    # am available
                    self.sock_for_me.send_multipart(request)
                    self._nb_job_done.value = self._nb_job_done.value + 1
                    self.running = 0
                else:
                    self.waiting_since.value = self.waiting_since.value + 100  # 100 ms
            self.sock_for_me.close()
        self.finish()
        self.done = True

    def finish(self):
        self.coroutine.finish()

    def init_connections(self):
        """
        Initialise zmq sockets.
        Because this class is s Process, This method must be call in the run
         method to be hold by the correct process.
        """
        Connections.init_connections(self)
        context = Context()
        self.sock_for_me = context.socket(REQ)
        self.sock_for_me.connect(self.sock_job_for_me_url)
        # Use a ZMQ Pool to get multichannel message
        self.poll = Poller()
        # Register sockets
        self.poll.register(self.sock_for_me, POLLIN)
        # Send READY to next_router to inform about my capacity to compute new
        # job
        self.sock_for_me.send_pyobj("READY")
        return True

    @property
    def wait_since(self):
        return self.waiting_since.value

    @property
    def stop(self):
        return self._stop.value

    @stop.setter
    def stop(self, value):
        self._stop.value = value

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
