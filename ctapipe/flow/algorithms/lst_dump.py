from ctapipe.utils.datasets import get_path
import ctapipe.instrument.InstrumentDescription as ID
from ctapipe.core import Component
from traitlets import Unicode
from time import sleep


class LSTDump(Component):
    """LSTDump` class represents a Stage for pipeline.
        it dumps RawCameraData contents to a string
    """


    def init(self):
        self.log.info("--- LSTDump init ---")
        return True

    def run(self, raw_camera_data):
        if raw_camera_data != None:
            self.log.debug("LSTDump receive {}".format(raw_camera_data))
            return('LSTDump START-> {} <- LSTDump END'.format(str(raw_camera_data)))

    def finish(self):
        self.log.info("--- LSTDump finish ---")
        pass
