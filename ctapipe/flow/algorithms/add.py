from time import sleep

from ctapipe.core import Component


class Add(Component):
    """Add class represents a Stage for pipeline.
       It simply adds one to the received value and returned it.

    """

    def init(self):
        self.log.debug("--- Add init ---")
        return True

    def run(self, x):
        sleep(.5)
        self.log.debug("Add receive {} ".format(x))
        return x

    def finish(self):
        self.log.debug("--- Add finish ---")
        pass
