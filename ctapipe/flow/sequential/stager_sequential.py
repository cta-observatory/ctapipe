# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding: utf8
from time import sleep
from time import time
import zmq
import types

import pickle
from ctapipe.flow.multiprocessus.connexions import Connexions


class StagerSequential():

    """`StagerSequential` class represents a Stager pipeline Step.
    """

    def __init__(
            self, coroutine, name=None, connexions=list(),main_connexion_name=None):
        """
        Parameters
        ----------
        coroutine : Class instance that contains init, run and finish methods
        connexions: list(str)
            define next available steps
        """
        self.name = name
        # Set coroutine
        self.coroutine = coroutine
        self.main_connexion_name = main_connexion_name
        self.connexions = connexions
        self.running = False
        self.nb_job_done = 0


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


        return True

    def run(self,inputs=None):
        result = self.coroutine.run(inputs)
        if isinstance(result, types.GeneratorType):
            for val in result:
                msg, destination = self.get_destination_msg_from_result(val)
                yield (msg,destination)
        else:
            msg, destination = self.get_destination_msg_from_result(result)
            yield (msg,destination)
        self.nb_job_done+=1


    def get_destination_msg_from_result(self,result):
        """
        If result is a tuple, check if last tuple elem is a valid next step.
        If yes, return a destination defined to  the last tuple elem and send result without the destination
        If no return None as destination
        Parameter:
        ----------
        result : any type
            value to send (can contain next step name)
        Return:
        -------
        msg, destination

        """
        destination = self.main_connexion_name
        if isinstance(result,tuple):
            # look is last tuple elem is a valid next step
            if result[-1] in self.connexions.keys():
                destination = result[-1]
                if len(result [:-1]) == 1:
                    msg = result [:-1][0]
                else:
                    msg = result[:-1]
                return msg,destination
            else:
                return result,destination
        else:
            return result,destination

def init_connexions(self):
    """
    Initialise zmq sockets.
    Because this class is s Process, This method must be call in the run
     method to be hold by the correct processus.
    """
    self.context = zmq.Context()
    Connexions.init_connexions(self)
    # Socket to talk to GUI
    self.socket_pub = self.context.socket(zmq.PUB)
    if self.gui_address is not None:
        try:
            self.socket_pub.connect("tcp://" + self.gui_address)
        except zmq.error.ZMQError as e:
            print("Error {} tcp://{}".format(e, self.gui_address))
            return False
    return True

def update_gui(self):
    """
    send it's status to GUI
    """
    msg = [self.name, self.running, self.nb_job_done]
    self.socket_pub.send_multipart(
        [b'GUI_PRODUCER_CHANGE', dumps(msg)])

    def finish(self):
        """
        """
        self.coroutine.finish()
        return True
