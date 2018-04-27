# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding: utf8
import zmq
from multiprocessing import Process
from multiprocessing import Value
from pickle import loads
from ctapipe.core import Component


class ConsumerZMQ(Process, Component):
    """`ConsumerZMQ` class represents a Consumer pipeline Step.
    It is derived from Process class. It receives
    new input from its prev stage, thanks to its ZMQ REQ socket,
    and executes its coroutine objet's run method by passing
    input as parameter.
    The process is launched by calling run method.
    init() method is call by run method.
    The process is stopped by setting share data stop to True
    """

    def __init__(
            self, coroutine, sock_consumer_port, _name=""):
        """
        Parameters
        ----------
        coroutine : Class instance that contains init, run and finish methods
        sock_consumer_port: str
            Port number for socket url
        """
        Component.__init__(self, parent=None)
        Process.__init__(self)
        self.coroutine = coroutine
        self.sock_consumer_url = 'tcp://localhost:' + sock_consumer_port
        self.name = _name
        self._running = Value('i', 0)
        self._nb_job_done = Value('i', 0)
        self._stop = Value('i', 0)

    def init(self):
        """
        Initialise coroutine, socket and poller
        Returns
        -------
        True if coroutine init method returns True, otherwise False
        """
        # Define coroutine and executes its init method
        if self.coroutine is None:
            return False
        if self.coroutine.init() == False:
            return False
        self.done = False
        return self.init_connections()

    def run(self):
        """
        Method representing the process's activity.
        It polls its socket and when received new input from it,
        it executes coroutine run method by passing new input
        The poll method's timeout is 100 ms in case of self.stop flag
        has been set to False.
        """
        if self.init():
            while not self._stop.value:
                try:
                    sockets = dict(self.poll.poll(100))
                    if (self.sock_reply in sockets and
                            sockets[self.sock_reply] == zmq.POLLIN):
                        request = self.sock_reply.recv_multipart()
                        # do some 'work', update status
                        cmd = loads(request[0])
                        self.running = 1
                        self.coroutine.run(cmd)
                        self.running = 0
                        self.nb_job_done += 1
                        # send reply back to router/queuer
                        self.sock_reply.send_multipart(request)

                except Exception as e:
                    self.log.error('CONSUMER exception {}'.format(e))
                    break
            self.sock_reply.close()
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
        context = zmq.Context()
        self.sock_reply = context.socket(zmq.REQ)
        self.sock_reply.connect(self.sock_consumer_url)
        # Informs prev_stage that I am ready to work
        self.sock_reply.send_pyobj("READY")
        # Create and register poller
        self.poll = zmq.Poller()
        self.poll.register(self.sock_reply, zmq.POLLIN)
        return True

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
