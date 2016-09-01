from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.core import Component
from time import sleep
from traitlets import Unicode
from random import randint


class RandIntGenerator(Component):

    """RandIntGenerator class represents a Producer for pipeline.
    """
    def init(self):
        return True

    def run(self):
        for i in range(100):
            val = randint(0,99)
            self.log.debug("RandIntGenerator send {}".format(val))
            yield val



        self.log.debug("\n--- RandIntGenerator Done ---")

    def finish(self):
        self.log.debug("--- RandIntGenerator finish ---")
        pass
