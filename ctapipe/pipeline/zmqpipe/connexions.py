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

    def __init__(self, main_connexion_name, connexions=dict()):
        """
        Parameters
        ----------
        connexions : dict
        main_connexion_name : str
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
                if main_connexion_name == name:
                    self.main_out_socket = self.sockets[name]
            except zmq.error.ZMQError as e:
                print(' {} : inproc://{}'
                               .format(e,  connexion))
                return False
        self.send_in_run = False
        return True

    def close_connexions(self):
        for sock in self.sockets.values():
            sock.close()

    def send_msg(self,msg,destination_step_name=None):
        send=False
        if not destination_step_name :
            socket  = self.main_out_socket
        else:
            socket = self.sockets[destination_step_name]
        while not send:

            socket.send_pyobj(msg)
            request = socket.recv()
            if request == b'OK':
                send = True
            else:
                sleep(0.1)
        self.send_in_run = True
