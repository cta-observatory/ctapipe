from ctapipe.core import Component
from time import sleep

class Inverse(Component):
    """Add class represents a Stage for pipeline.
    It returns inverted value of received value
    """
    def init(self):
        self.log.debug("--- Add init ---")
        return True

    def run(self, input):
        sleep(.5)
        if input:
            self.log.debug("Inverse receive {}".format(input))
            return int(input) * -1

    def finish(self):
        self.log.debug("--- Add finish ---")
        pass
