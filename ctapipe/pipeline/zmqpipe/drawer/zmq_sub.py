# Licensed under a 3-clause BSD style license - see LICENSE.rst
import zmq
from threading import Thread
import pickle
from ctapipe.core import Component

import os
import sys
import inspect
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
pipedrawerdir = os.path.dirname(currentdir)
sys.path.insert(0, pipedrawerdir)


class ZmqSub(Thread, Component):

    """
    Manages communication with pipeline thanks to ZMQ SUB message
    Transmit information to GUI when pipeline change
    Parameters
    ----------
    pipedrawer : PipelineDrawer
        Widget that draws pipeline by receiving information from this instance
    gui_port : str
        port to connect for ZMQ communication
    statusBar :  QtGui.QStatusBar
        MainWindow status bar to display information
    """

    def __init__(self, pipedrawer=None, gui_port=None, statusBar=None):
        Thread.__init__(self)
        if gui_port is not None:
            self.statusBar = statusBar
            self.context = zmq.Context.instance()
            # Socket to talk to pipeline kernel and pipeline steps and router
            self.socket = self.context.socket(zmq.SUB)
            gui_adress = "tcp://*:" + str(gui_port)
            try:
                self.socket.bind(gui_adress)
            except zmq.error.ZMQError as e:
                self.log.error("".format(str(e), gui_adress))
            # Inform about connection in statusBar
            if statusBar is not None:
                self.statusBar.showMessage("binded to " + gui_adress)
            # Register socket in a poll and register topics
            self.poll = zmq.Poller()
            self.poll.register(self.socket, zmq.POLLIN)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, 'GUI_GRAPH')
            self.socket.setsockopt_string(zmq.SUBSCRIBE, 'GUI_STAGER_CHANGE')
            self.socket.setsockopt_string(zmq.SUBSCRIBE, 'GUI_CONSUMER_CHANGE')
            self.socket.setsockopt_string(zmq.SUBSCRIBE, 'GUI_PRODUCER_CHANGE')
            self.socket.setsockopt_string(zmq.SUBSCRIBE, 'GUI_ROUTER_CHANGE')
            # self,stop flag is set by ficnish method to stop this thread
            # properly when GUI is closed
            self.stop = False
            # self.pipedrawer will receive new pipeline information
            self.pipedrawer = pipedrawer
        else:
            self.stop = False

    def run(self):
        """
        Method representing the thread’s activity.
        """
        while not self.stop:
            sockets = dict(self.poll.poll(1000))  # Poll or time out (1000ms)
            if self.socket in sockets and sockets[self.socket] == zmq.POLLIN:
                # receive a new message form pipeline
                receive = self.socket.recv_multipart()
                # decode topic and msg
                topic = receive[0]
                msg = pickle.loads(receive[1])
                # inform pipedrawer
                self.pipedrawer.pipechange(topic, msg)

    def finish(self):
        self.stop = True
