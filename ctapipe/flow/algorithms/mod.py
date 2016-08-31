from ctapipe.utils.datasets import get_path
import ctapipe.instrument.InstrumentDescription as ID
from ctapipe.core import Component
from traitlets import Unicode
from time import sleep

class Mod(Component):
    """Mod` class represents a Stage for pipeline.
    """


    def init(self):
        self.log.debug("--- Mod init ---")
        return True

    def run(self, inputs):
            self.log.info("Mod receive {}".format(inputs))
            return inputs


    def finish(self):
        self.log.debug("--- Mod finish ---")
        pass
