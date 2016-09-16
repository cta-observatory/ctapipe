from ctapipe.core import Component
from time import sleep

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
