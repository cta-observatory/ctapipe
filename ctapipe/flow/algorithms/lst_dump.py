from ctapipe.utils.datasets import get_path
import ctapipe.instrument.InstrumentDescription as ID
from ctapipe.calib.camera.calibrators import calibrate_event
from ctapipe.calib.camera.calibrators import calibration_parameters
from ctapipe.calib.camera.calibrators import calibration_arguments
from ctapipe.calib.camera.integrators import integrator_dict
from ctapipe.core import Component
from traitlets import Unicode
from time import sleep
from argparse import ArgumentParser

class LSTDump(Component):
    """LSTDump` class represents a Stage for pipeline.
        it dumps RawCameraData contents to a string
    """


    def init(self):
        self.log.info("--- LSTDump init ---")
        self.parameters = dict()
        self.parameters['integrator'] = 'nb_peak_integration'
        self.parameters['window'] = 7
        self.parameters['shift'] = 3
        self.parameters['sigamp'] = [2, 4]
        self.parameters['lwt'] = 0
        self.parameters['calib_scale'] = 0.92

        for key, value in self.parameters.items():
            self.log.info("[{}] {}".format(key, value))
        return True

    def run(self, event):
        if event != None:
            calibrated_event = calibrate_event(event,self.parameters)
            self.log.info(calibrated_event) 
            return('LSTDump START-> {} <- LSTDump END'.format(str(calibrated_event)))

    def finish(self):
        self.log.info("--- LSTDump finish ---")
        pass
