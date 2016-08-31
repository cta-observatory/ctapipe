from ctapipe.core import Component
from time import sleep

class Odd(Component):
    """Odd` class represents a Stage for pipeline.

    """
    def init(self):
        self.log.debug("--- Odd init ---")
        return True

    def run(self, inputs):
        self.log.info("ODD receive {}".format(inputs))
        return inputs


    def finish(self):
        self.log.debug("--- Odd finish ---")
        pass
