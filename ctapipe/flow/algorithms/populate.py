from ctapipe.core import Component
from random import randint
from time import sleep


class Populate(Component):
    """Populate class represents a Stage for pipeline.

    """
    def init(self):
        self.log.debug("--- Populate init ---")
        return True

    def run(self, val):
        #for i in range(1):
        yield (val)

    def finish(self):
        self.log.debug("--- Populate finish ---")
        pass
