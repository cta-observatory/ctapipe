# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding: utf8


class Coroutine():

    """`Coroutine` class represents a stager/producer/consumer's coroutine
    """

    def __init__(self):
        pass

    def set_socket(self, socket):
        self.socket = socket

    def send_to_next_stage(self, object):
        if self.socket != None:
            self.socket.send_pyobj(object)
            # wait for acknoledgement form next router
            self.socket.recv()
        else:
            print("ERROR Coroutine: no socket define")
