from ctapipe.utils.datasets import get_path
import ctapipe.instrument.InstrumentDescription as ID
from ctapipe.core import Component
from traitlets import Unicode
from time import sleep


class OtherDump(Component):
    """OtherDump` class represents a Stage for pipeline.
        it dumps RawCameraData contents to a string
    """


    def init(self):
        self.log.info("--- OtherDump init ---")
        return True

    def run(self, event):
        self.log.debug("OtherDump receive {}".format(event))
        if event != None:
            return('OtherDump START-> {} <- Otherump END'.format(str(event)))



    def finish(self):
        self.log.info("--- OtherDump finish ---")
        pass
