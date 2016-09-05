# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding: utf8
from time import sleep
import zmq
from threading import Thread
import threading
import pickle


class StagerZmq(threading.Thread):

    """`StagerZmq` class represents a Stager pipeline Step.
    It is derived from Thread class.
    It receives new input from its prev stage, thanks to its ZMQ REQ socket,
    and executes its coroutine objet's run method by passing
    input as parameter. Finaly it sends coroutine returned value to its next
    stage, thanks to its ZMQ REQ socket,
    The Thread is launched by calling run method, after init() method
    has been called and has returned True.
    The thread is stoped by executing finish method.
    """

    def __init__(
            self, coroutine, sock_job_for_me_port,
            sock_job_for_you_port, name=None, gui_address=None):
        """
        Parameters
        ----------
        coroutine : Class instance that contains init, run and finish methods
        sock_job_for_me_port: str
            Port number for input socket url
        sock_job_for_you_port: str
            Port number for output socket url
        """
        # Call mother class (threading.Thread) __init__ method
        Thread.__init__(self)
        # Set coroutine
        self.coroutine = coroutine
        # set sockets url
        self.sock_job_for_you_url = 'inproc://' + sock_job_for_you_port
        self.sock_job_for_me_url = 'inproc://' + sock_job_for_me_port
        self.name = name
        self.running = False
        self.nb_job_done = 0
        self.gui_address = gui_address

        # Prepare our context and sockets
        context = zmq.Context.instance()
        self.sock_for_you = context.socket(zmq.REQ)

    def get_output_socket(self):
        return self.sock_for_you

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

        # Connect to GUI
        context = zmq.Context.instance()
        self.socket_pub = context.socket(zmq.PUB)
        if self.gui_address is not None:
            self.socket_pub.connect("tcp://" + self.gui_address)
        # Socket to talk to next_router
        self.sock_for_you.connect(self.sock_job_for_you_url)

        # Socket to talk to prev_router
        self.sock_for_me = context.socket(zmq.REQ)
        self.sock_for_me.connect(self.sock_job_for_me_url)

        # Use a ZMQ Pool to get multichannel message
        self.poll = zmq.Poller()
        # Register sockets
        self.poll.register(self.sock_for_me, zmq.POLLIN)
        # Send READY to next_router to inform about my capacity to compute new
        # job
        self.sock_for_me.send_pyobj("READY")
        # Stop flag
        self.stop = False
        return True

    def run(self):
        """
        Method representing the thread’s activity.
        It polls its socket and when received new input from it,
        it executes coroutine run method by passing new input.
        Then it sends coroutine return value to its next stage,
        thanks to its ZMQ REQ socket.
        The poll method's timeout is 100 ms in case of self.stop flag
        has been set to False by finish method.
        """
        while not self.stop:
            sockets = dict(self.poll.poll(100))  # Poll or time out (100ms)
            if (self.sock_for_me in sockets and
                    sockets[self.sock_for_me] == zmq.POLLIN):
                #  Get the input from prev_stage
                self.running = True
                self.update_gui()
                request = self.sock_for_me.recv_multipart()
                receiv_input = pickle.loads(request[0])
                # do the job
                send_output = self.coroutine.run(receiv_input)
                # send acknoledgement to prev router/queue to inform it that I
                # am available
                self.sock_for_me.send_multipart(request)
                # send new job to next router/queue
                self.sock_for_you.send_pyobj(send_output)
                # wait for acknoledgement form next router
                self.sock_for_you.recv()
                self.nb_job_done += 1
                self.running = False
                self.update_gui()
        self.sock_for_me.close()
        self.sock_for_you.close()
        self.socket_pub.close()

    def finish(self):
        """
        Executes coroutine method and set stop flag to True to stop
        Thread activity
        """
        self.coroutine.finish()
        self.stop = True

    def update_gui(self):
        msg = [self.name, self.running, self.nb_job_done]
        self.socket_pub.send_multipart(
            [b'GUI_STAGER_CHANGE', pickle.dumps(msg)])
