# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Main program that lunches GUI
This GUI communicate with pipeline  elements (stager, consumer, producer,
 router) tahnks to ZMQ library PUB/SUB message
Every second, pipeline send its full configuration:
 - producer
 - levels of stages
 - consumer
 - router / queue
 Each time a producer, stager , consumer or router changes, it send its status
 to this GUI
"""

import os
import sys
import inspect
from ctapipe.core import Tool
from traitlets import (Integer, Float, List, Dict, Unicode)


currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from ctapipe.pipeline.zmqpipe.drawer import ModuleApplication


class PipeGui(Tool):
    description = "run stages in multithread pipeline"

    port = Integer(5565, help='GUI port for pipelien connexion').tag(
        config=True, allow_none=True)

    def start(self):
        app = ModuleApplication(sys.argv, self.port)

    def setup(self):
        pass

    def finish(self):
        pass


def main():
    gui = PipeGui()
    gui.run()

if __name__ == "main":
    main()
