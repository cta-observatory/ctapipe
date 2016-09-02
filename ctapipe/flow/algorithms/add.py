from ctapipe.core import Component
from random import randint
from time import sleep
from math import sin


class Add(Component):
    """Add class represents a Stage for pipeline.

    """
    def init(self):
        self.log.debug("--- Inverse init ---")
        return True

    def run(self, val):
        self.log.info("add receive {}".format(val))
        """
        for i in range(2000):
            foo = (val**i)
        """
        val+=1
        return val

    def finish(self):
        self.log.debug("--- Add finish ---")
        pass
