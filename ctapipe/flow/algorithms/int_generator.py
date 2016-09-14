from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.core import Component
from time import sleep
from traitlets import Unicode

class IntGenerator(Component):

    """IntGenerator class represents a Producer for pipeline.
    It yields integer from 0 to 5 to Pair or Odd stages, depending of their parity
    """
    def init(self):
        return True

    def run(self):
        for i in range(20):
            sleep(.5)
            self.log.debug("IntGenerator send {}".format(i))
            if i % 2 == 0:
                yield (i,'Pair')
            else: yield(i,'Odd')

        self.log.debug("\n--- IntGenerator Done ---")

    def finish(self):
        self.log.debug("--- IntGenerator finish ---")
        pass
