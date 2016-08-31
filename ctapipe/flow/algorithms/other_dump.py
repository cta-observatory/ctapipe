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

    def run(self, raw_camera_data):
        if raw_camera_data != None:
            self.log.debug("OtherDump receive {}".format(raw_camera_data))
            return('OtherDump START-> {} <- Otherump END'.format(str(raw_camera_data)))

    def finish(self):
        self.log.info("--- OtherDump finish ---")
        pass
