# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding: utf8


import zmq
from multiprocessing import Process
from multiprocessing import Value
import pickle
from ctapipe.core import Component


class ConsumerZMQ(Process, Component):

    """`ConsumerZMQ` class represents a Consumer pipeline Step.
    It is derived from Thread class. It receives
    new input from its prev stage, thanks to its ZMQ REQ socket,
    and executes its coroutine objet's run method by passing
    input as parameter.
    The Thread is launched by calling run method, after init() method
    has been called and has returned True.
    The thread is stoped by executing finish method.
    """

    def __init__(
            self, coroutine, sock_consumer_port, _name="",
            parent=None, gui_address=None):
        """
        Parameters
        ----------
        coroutine : Class instance that contains init, run and finish methods
        sock_consumer_port: str
            Port number for socket url
        """
        # Call mother class (threading.Thread) __init__ method
        Process.__init__(self)
        self.coroutine = coroutine
        self.gui_address = gui_address
        # self.sock_consumer_url = "tcp://localhost:"+sock_consumer_port
        self.sock_consumer_url = 'tcp://localhost:' + sock_consumer_port
        self.name = _name
        self.nb_job_done = 0
        self.running = False
        self._stop = Value('i',0)

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
        # self.stop flag is used to stop this Thread
        self.stop = False
        # self.total allows to print the number of times run method
        # has been called at end of job
        self.done = False
        print('===> {} init done'.format(self.name))
        return True

    def run(self):
        """
        Method representing the thread’s activity.
        It polls its socket and when received new input from it,
        it executes coroutine run method by passing new input
        The poll method's timeout is 100 ms in case of self.stop flag
        has been set to False by finish method.
        """
        if self.init_connexions():
            while not self.stop :
                try:
                    sockets = dict(self.poll.poll(100))
                    if (self.sock_reply in sockets and
                            sockets[self.sock_reply] == zmq.POLLIN):
                        request = self.sock_reply.recv_multipart()
                        # do some 'work'
                        cmd = pickle.loads(request[0])
                        self.running = True
                        self.update_gui()
                        self.coroutine.run(cmd)
                        self.nb_job_done += 1
                        self.running = False
                        self.update_gui()
                        # send reply back to router/queuer
                        self.sock_reply.send_multipart(request)
                except:
                    break
            self.update_gui()
            self.sock_reply.close()
            self.socket_pub.close()


        self.done = True

    """
    def finish(self):
        self.coroutine.finish()
        self.stop = True
        if self.done:
            return True
        else:
            return False
    """

    def init_connexions(self):
        context = zmq.Context()

        self.sock_reply = context.socket(zmq.REQ)
        self.sock_reply.connect(self.sock_consumer_url)

        self.socket_pub = context.socket(zmq.PUB)
        if self.gui_address is not None:
            try:
                self.socket_pub.connect("tcp://" + self.gui_address)
            except zmq.error.ZMQError as e:
                self.log.error("{} tcp://{}".format(str(e),  self.gui_address))
                return False

        # Informs prev_stage that I am ready to work
        self.sock_reply.send_pyobj("READY")
        # Create and register poller
        self.poll = zmq.Poller()
        self.poll.register(self.sock_reply, zmq.POLLIN)

        return True



    def update_gui(self):
        msg = [self.name, self.running, self.nb_job_done]
        self.socket_pub.send_multipart(
            [b'GUI_CONSUMER_CHANGE', pickle.dumps(msg)])

    @property
    def stop(self):
        return self._stop.value

    @stop.setter
    def stop(self, value):
        self._stop.value = value
