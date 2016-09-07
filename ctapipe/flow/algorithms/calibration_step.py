from ctapipe.utils.datasets import get_path
import ctapipe.instrument.InstrumentDescription as ID
from ctapipe.calib.camera.calibrators import calibrate_event
from ctapipe.calib.camera.integrators import integrator_dict

class CalibrationStep():
    """CalibrationStep` class represents a Stage for pipeline.
        it dumps RawCameraData contents to a string
    """
    
    def init(self):
        self.log.info("--- CalibrationStep init ---")
        self.parameters = dict()
        self.parameters['integrator'] = 'nb_peak_integration'
        self.parameters['window'] = self.integration_windows[0]
        self.parameters['shift'] =  self.integration_windows[1]
        self.parameters['sigamp'] = self.integration_sigamp
        self.parameters['lwt'] = self.integration_lwt
        self.parameters['calib_scale'] = self.integration_calib_scale
        try:
            self.parameters['clip_amp'] = self.integration_clip_amp
        except NameError as e:
            pass #integration_clip_amp not define
        for key, value in sorted(self.parameters.items()):
            self.log.info("[{}] {}".format(key, value))
        return True

    def run(self, event):
        if event != None:
            calibrated_event = calibrate_event(event,self.parameters)
            self.log.debug(calibrated_event)
            return('CalibrationStep START-> {} <- CalibrationStep END'.format(str(calibrated_event)))

    def finish(self):
        self.log.info("--- CalibrationStep finish ---")
        pass
