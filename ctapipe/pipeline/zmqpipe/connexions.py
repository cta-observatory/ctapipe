# Licensed under a 3-clause BSD style license - see LICENSE.rst
from threading import Thread
from time import sleep
from ctapipe.core import Component
import zmq
import pickle

class Connexions():
    """
    implements ZMQ connexions between thread for PRODUCER and STAGER and CONSUMER
    """

    def __init__(self, connexions=dict()):
        """
        Parameters
        ----------
        connexions : dict
        """
        self.connexions = connexions
        self.sockets=dict()
        # Socket to talk to others steps
        self.context = zmq.Context.instance()
        self.main_out_socket = None
        for name,connexion in self.connexions.items():
            self.sockets[name] = self.context.socket(zmq.REQ)
            try:
                self.sockets[name].connect('inproc://' + connexion)
                if not self.main_out_socket:
                    self.main_out_socket = self.sockets[name]
            except zmq.error.ZMQError as e:
                print(' {} : inproc://{}'
                               .format(e,  connexion))
                return False

        return True

    def close_connexions(self):
        for sock in self.sockets.values():
            sock.close()

    def send_msg(self,destination_step_name, msg):
        sock = self.sockets[destination_step_name]
        sock.send_pyobj(msg)
        sock.recv()
