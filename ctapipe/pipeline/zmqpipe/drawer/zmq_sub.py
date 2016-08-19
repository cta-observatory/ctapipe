# Licensed under a 3-clause BSD style license - see LICENSE.rst
import zmq
from threading import Thread
import pickle
from PyQt4 import QtCore
from time import clock
from ctapipe.core import Component


import os
import sys
import inspect
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
pipedrawerdir = os.path.dirname(currentdir)
sys.path.insert(0, pipedrawerdir)


class ZmqSub(Thread, QtCore.QObject):

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
    message = QtCore.pyqtSignal(list)

    def __init__(self, pipedrawer=None, table_queue=None, gui_port=None, statusBar=None):
        Thread.__init__(self)
        QtCore.QObject.__init__(self)

        if gui_port is not None:
            self.context = zmq.Context.instance()
            # Socket to talk to pipeline kernel and pipeline steps and router
            self.socket = self.context.socket(zmq.SUB)
            gui_adress = "tcp://*:" + str(gui_port)
            try:
                self.socket.bind(gui_adress)
            except zmq.error.ZMQError as e:
                print("ERROR: ".format(str(e), gui_adress))
            # Inform about connection in statusBar
            if statusBar is not None:
                self.statusBar = statusBar
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
            # self.pipedrawer will receive new pipeline information
            self.stop = False
            self.pipedrawer = pipedrawer
            self.table_queue = table_queue
            self.steps = list()

            self.config_time = 0

            self.last_send_config = 0
        else:
            self.stop = True

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
                self.update_full_state(topic,msg)
                conf_time = clock()
                #print("DEBUG conf_time:{} self.last_send_config {}  total {}".format(conf_time,self.last_send_config,conf_time - self.last_send_config))
                if conf_time - self.last_send_config > .0416: # 24 images /sec
                     self.message.emit(self.steps)
                     self.last_send_config = conf_time
                #self.pipedrawer.pipechange(topic, msg)
                #if self.table_queue: self.table_queue.pipechange(topic, msg)
            else:
                 self.message.emit(self.steps)

    def update_full_state(self,topic,msg):
        if topic == b'GUI_GRAPH':
            config_time, receiv_steps = msg
            if config_time != self.config_time:
                self.steps = receiv_steps
                print('debug receiv_steps {}'.format(receiv_steps))
                self.config_time = config_time
        # Stager or Producer or Consumer state changes

        if self.steps is not None and (topic == b'GUI_STAGER_CHANGE' or
                                        topic == b'GUI_CONSUMER_CHANGE' or
                                        topic == b'GUI_PRODUCER_CHANGE'):
            self.step_change(msg)
        # Force to update drawing

        if topic == b'GUI_ROUTER_CHANGE':
            self.router_change(topic,msg)



    def step_change(self, msg):
        """Find which pipeline step has changed, and update its corresponding
        StagerRep
        Parameters
        ----------
        msg: list
            contains step name, step running flag and step nb_job_done
            receiv_steps: list of GUIStepInfo describing pipeline contents
        """
        name, running, nb_job_done = msg
        for step in self.steps:
            if step.name == name.split('$$thread')[0]:
                step.running = running
                step.nb_job_done = nb_job_done
                print('DEBUG step.name:{} step.nb_job_done {}'.format(step.name,step.nb_job_done))
                break

    def router_change(self, topic, msg):
        """Called by ZmqSub instance when it receives zmq message from pipeline
        Update pipeline state (self.steps) and force to update drawing
        Parameters
        ----------
        topic : str
        """
        name, queue = msg
        name = name.split('_router')[0]
        step = self.get_step_by_name(name)
        if step:
            step.queue_length = queue



    def get_step_by_name(self, name):
        ''' Find a PipeStep in self.producer_step or  self.stager_steps or
        self.consumer_step
        Return: PipeStep if found, otherwise None
        '''
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def finish(self):
        self.stop = True
