from ctapipe.core import Component
from time import sleep


class Pair(Component):
    """Odd` class represents a Stage for pipeline.

    """
    def init(self):
        self.log.info("--- Pair init ---")
        return True

    def run(self, inputs):
        self.log.info("PAIR receive {}".format(inputs))
        return inputs


    def finish(self):
        self.log.debug("--- Pair finish ---")
        pass
