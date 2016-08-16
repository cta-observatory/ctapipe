# Licensed under a 3-clause BSD style license - see LICENSE.rst
from threading import Thread
from time import sleep
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

    def __init__(self, coroutine, sock_request_port, _name, connexions=dict(),
                gui_address=None):
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
        self.port = 'inproc://' + sock_request_port
        self.name = _name
        self.running = False
        self.nb_job_done = 0
        self.gui_address = gui_address
        # Prepare our context and sockets
        self.context = zmq.Context.instance()
        self.connexions = connexions
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
        # Socket to talk to next step
        self.sock_request = self.context.socket(zmq.REQ)
        self.sock_request.connect(self.port)
        # Socket to talk to GUI
        self.socket_pub = self.context.socket(zmq.PUB)

        # Socket to talk to others steps
        for name,connexion in self.connexions.items():
            self.other_requests[name] = self.context.socket(zmq.REQ)
            try:
                self.other_requests[name].connect('inproc://' + connexion)
            except zmq.error.ZMQError as e:
                print(' {} : inproc://{}'
                               .format(e,  connexion))
                return False

        if self.gui_address is not None:
            try:
                self.socket_pub.connect("tcp://" + self.gui_address)
            except zmq.error.ZMQError as e:
                print("{} tcp://{}".format(e, self.gui_address))
                return False

        if self.coroutine is None:
            return False
        if self.coroutine.init() == False:
            return False
        self.coroutine.send_msg = self.send_msg
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
        self.done = True

    def finish(self):
        """
        Executes coroutine method
        """
        while self.done != True:
            sleep(1)
        self.coroutine.finish()
        for sock in self.other_requests.values():
            sock.close()


    def send_msg(self,destination_step_name, msg):
        sock = self.other_requests[destination_step_name]
        #print("DEBUG ------> 2 type(sock)",type(sock))
        sock.send_pyobj(msg)
        sock.recv()

    def update_gui(self):
        msg = [self.name, self.running, self.nb_job_done]
        self.socket_pub.send_multipart(
            [b'GUI_PRODUCER_CHANGE', pickle.dumps(msg)])
