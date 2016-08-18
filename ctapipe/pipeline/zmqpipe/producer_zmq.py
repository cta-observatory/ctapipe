# Licensed under a 3-clause BSD style license - see LICENSE.rst
from ctapipe.core import Component
from ctapipe.pipeline.zmqpipe.connexions import Connexions

from threading import Thread
from time import sleep
import zmq
import pickle



class ProducerZmq(Thread, Component, Connexions):

    """`ProducerZmq` class represents a Producer pipeline Step.
    It is derived from Thread class.
    It gets a Python generator from its coroutine run method.
    It loops overs its generator and sends new input to its next stage,
    thanks to its ZMQ REQ socket,
    The Thread is launched by calling run method, after init() method
    has been called and has returned True.
    """

    def __init__(self, coroutine, name,main_connexion_name,
                connexions=dict(), gui_address=None):
        """
        Parameters
        ----------
        coroutine : Class instance that contains init, run and finish methods
        sock_consumer_port: str
            Port number for socket url
        """
        # Call mother class (threading.Thread) __init__ method
        Thread.__init__(self)
        self.name = name
        Connexions.__init__(self,main_connexion_name,connexions)

        self.identity = '{}{}'.format('id_', "producer")
        self.coroutine = coroutine
        self.running = False
        self.nb_job_done = 0
        self.gui_address = gui_address
        # Prepare our context and sockets
        self.context = zmq.Context.instance()
        self.foo = None
        self.other_requests=dict()
        self.done = False

    def init(self):
        """
        Initialise coroutine and socket

        Returns
                -------
                True if coroutine init method returns True, otherwise False
        """
        # Socket to talk to GUI
        self.socket_pub = self.context.socket(zmq.PUB)

        if self.gui_address is not None:
            try:
                self.socket_pub.connect("tcp://" + self.gui_address)
            except zmq.error.ZMQError as e:
                print("Error {} tcp://{}".format(e, self.gui_address))
                return False

        if self.coroutine is None:
            return False
        if self.coroutine.init() == False:
            return False
        self.coroutine.send_msg = self.send_msg

    def run(self):
        """
        Method representing the thread’s activity.
        It gets a Python generator from its coroutine run method.
        It loops overs its generator and sends new input to its next stage,
        thanks to its ZMQ REQ socket.
        """
        generator = self.coroutine.run()
        if generator:
            self.running = True
            self.update_gui()
            for result in generator:
                send = False
                while not send:
                    self.main_out_socket.send_pyobj(result)
                    # Wait for reply
                    reply = self.main_out_socket.recv()
                    if reply == b'OK':
                        send = True
                    else:
                        sleep(0.1)
                self.nb_job_done += 1
                self.update_gui()

            self.running = False
            self.update_gui()
        self.socket_pub.close()
        self.done = True

    def finish(self):
        """
        Executes coroutine method
        """
        while self.done != True:
            return False
        self.coroutine.finish()
        return True

    def update_gui(self):
        msg = [self.name, self.running, self.nb_job_done]
        self.socket_pub.send_multipart(
            [b'GUI_PRODUCER_CHANGE', pickle.dumps(msg)])
