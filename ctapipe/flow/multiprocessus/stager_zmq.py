# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding: utf8
from time import sleep
from time import time
from types import GeneratorType
from multiprocessing import Process
from multiprocessing import  Value
from pickle import loads
from pickle import dumps
import zmq
from ctapipe.flow.multiprocessus.connexions import Connexions

class StagerZmq(Process, Connexions):

    """`StagerZmq` class represents a Stager pipeline Step.
    It is derived from Process class.
    It receives new input from its prev stage, thanks to its ZMQ REQ socket,
    and executes its coroutine objet's run method by passing
    input as parameter. Finaly it sends coroutine returned value to its next
    stage, thanks to its ZMQ REQ socket,
    The processus is launched by calling run method.
    init() method is call by run method.
    The processus is stopped by setting share data stop to True
    """
    def __init__(
            self, coroutine, sock_job_for_me_port,
            name=None, connexions=dict(),main_connexion_name=None, gui_address=None):
        """
        Parameters
        ----------
        coroutine : Class instance that contains init, run and finish methods
        sock_job_for_me_port: str
            Port number for input socket url
        name: str
            Stage name
        main_connexion_name : str
            Default next step name. Used to send data when destination is not provided
        connexions: dict {'STEP_NANE' : (zmq STEP_NANE port in)}
            Port number for socket for each next steps
        gui_address : str
            GUI port for ZMQ 'hostname': + 'port'
        """
        Process.__init__(self)
        self.name = name
        Connexions.__init__(self,main_connexion_name,connexions)
        self.coroutine = coroutine
        self.sock_job_for_me_url = 'tcp://localhost:' + sock_job_for_me_port
        self.running = False
        self.gui_address = gui_address
        self.done = False
        self.waiting_since = Value('i',0)
        self._nb_job_done = Value('i',0)
        self._stop = Value('i',0)

    def init(self):
        """
        Initialise coroutine sockets and poller
        Returns
        -------
        True if coroutine init and init_connexions methods returns True,
         otherwise False
        """
        if self.name is None:
            self.name = "STAGER"
        if self.coroutine is None:
            return False
        if self.coroutine.init() == False:
            return False
        return self.init_connexions()

    def run(self):
        """
        Method representing the processus's activity.
        It polls its socket and when received new input from it,
        it executes coroutine run method by passing new input.
        Then it sends coroutine return value to its next stage,
        thanks to its ZMQ REQ socket.
        The poll method's timeout is 100 ms in case of self.stop flag
        has been set to False.
        Atfer the main while loop, coroutine.finish method is called
        """
        if self.init()  :
            while not self.stop:
                sockets = dict(self.poll.poll(100))  # Poll or time out (100ms)
                if (self.sock_for_me in sockets and
                        sockets[self.sock_for_me] == zmq.POLLIN):
                    #  Get the input from prev_stage
                    self.waiting_since.value = 0
                    self.running = True
                    self.update_gui()
                    request = self.sock_for_me.recv_multipart()
                    receiv_input = loads(request[0])
                    # do the job
                    results = self.coroutine.run(receiv_input)
                    if isinstance(results, GeneratorType):
                        for val in results:
                            msg,destination = self.get_destination_msg_from_result(val)
                            self.send_msg(msg,destination)
                    else:
                        msg,destination = self.get_destination_msg_from_result(results)
                        self.send_msg(msg,destination)
                    # send acknoledgement to prev router/queue to inform it that I
                    # am available
                    self.sock_for_me.send_multipart(request)
                    self._nb_job_done.value = self._nb_job_done.value + 1
                    self.running = False
                    self.update_gui()
                else:
                    self.waiting_since.value = self.waiting_since.value+100 # 100 ms
            self.sock_for_me.close()
            self.socket_pub.close()
        self.coroutine.finish()
        self.done = True


    def init_connexions(self):
        """
        Initialise zmq sockets.
        Because this class is s Process, This method must be call in the run
         method to be hold by the correct processus.
        """
        Connexions.init_connexions(self)
        context = zmq.Context()
        self.socket_pub = context.socket(zmq.PUB)
        if self.gui_address is not None:
            self.socket_pub.connect("tcp://" + self.gui_address)
        self.sock_for_me = context.socket(zmq.REQ)
        self.sock_for_me.connect(self.sock_job_for_me_url)
        # Use a ZMQ Pool to get multichannel message
        self.poll = zmq.Poller()
        # Register sockets
        self.poll.register(self.sock_for_me, zmq.POLLIN)
        # Send READY to next_router to inform about my capacity to compute new
        # job
        self.sock_for_me.send_pyobj("READY")
        return True

    def update_gui(self):
        """
        send it's status to GUI
        """
        msg = [self.name, self.running, None]
        self.socket_pub.send_multipart(
            [b'GUI_STAGER_CHANGE', dumps(msg)])

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
