# Licensed under a 3-clause BSD style license - see LICENSE.rst
from copy import deepcopy
from threading import Thread
from ctapipe.core import Component
import zmq
import pickle


class ProducerZmq(Thread, Component):

    """`ProducerZmq` class represents a Producer pipeline Step.
    It is derived from Thread class.
    It gets a Python generator from its coroutine run method.
    It loops overs its generator and sends new input to its next stage,
    thanks to its ZMQ REQ socket,
    The Thread is launched by calling run method, after init() method
    has been called and has returned True.
    """

    def __init__(self, coroutine, sock_request_port, _name, gui_address=None):
        """
        Parameters
        ----------
        coroutine : Class instance that contains init, run and finish methods
        sock_consumer_port: str
            Port number for socket url
        """
        # Call mother class (threading.Thread) __init__ method
        Thread.__init__(self)
        self.identity = '{}{}'.format('id_', "producer")
        self.coroutine = coroutine
        # self.port = "tcp://localhost:"+ sock_request_port
        self.port = 'inproc://' + sock_request_port
        self.name = _name
        self.running = False
        self.nb_job_done = 0
        self.gui_address = gui_address
        # Prepare our context and sockets
        self.context = zmq.Context.instance()
        # Socket to talk to Router
        self.sock_request = self.context.socket(zmq.REQ)
        self.sock_request.connect(self.port)
        self.socket_pub = self.context.socket(zmq.PUB)

        if self.gui_address is not None:
            try:
                self.socket_pub.connect("tcp://" + self.gui_address)
            except zmq.error.ZMQError as e:
                self.log.error("{} tcp://{}".format(e, self.gui_address))
                return False

    def init(self):
        """
        Initialise coroutine and socket

        Returns
                -------
                True if coroutine init method returns True, otherwise False
        """
        if self.coroutine is None:
            return False
        if self.coroutine.init() == False:
            return False

        return True

    def get_output_socket(self):
        return self.sock_request

    def run(self):
        """
        Method representing the thread’s activity.
        It gets a Python generator from its coroutine run method.
        It loops overs its generator and sends new input to its next stage,
        thanks to its ZMQ REQ socket.
        """
        generator = self.coroutine.run()
        self.running = True
        self.update_gui()
        for result in generator:
            self.sock_request.send_pyobj(result)
            # Wait for reply
            self.nb_job_done += 1
            self.update_gui()
            self.sock_request.recv()
        self.running = False
        self.update_gui()
        self.sock_request.close()
        self.socket_pub.close()

    def finish(self):
        """
        Executes coroutine method
        """
        self.coroutine.finish()

    def update_gui(self):
        msg = [self.name, self.running, self.nb_job_done]
        self.socket_pub.send_multipart(
            [b'GUI_PRODUCER_CHANGE', pickle.dumps(msg)])
