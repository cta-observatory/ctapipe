from ctapipe.core import Component
from random import randint
from time import sleep
from math import sin


class Add(Component):
    """Add class represents a Stage for pipeline.
       It simply adds one to the received value and returned it.

    """
    def init(self):
        self.log.debug("--- Add init ---")
        return True

    def run(self, value):
        sleep(.5)
        self.log.debug("Add receive {}".format(value))
        value+=1
        return value

    def finish(self):
        self.log.debug("--- Add finish ---")
        pass
