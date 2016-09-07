from ctapipe.flow.algorithms.calibration_step import CalibrationStep
from ctapipe.core import Component
from traitlets import Unicode
from traitlets import Int
from traitlets import Float
from traitlets import List

class LSTCalibration(Component,CalibrationStep):
    """LSTCalibration class represents a Stage for pipeline.
        it dumps RawCameraData contents to a string
    """
    integrator = Unicode('nb_peak_integration',
     help='integration scheme to be used to extract the charge').tag(
        config=True, allow_none=True)
    integration_windows = List([7, 3],help='Set integration window width \
        and offset (to before the peak) respectively' ).tag(
        config=True, allow_none=True)
    integration_sigamp = List([2, 4], help='Amplitude in ADC counts above \
        pedestal at which a signal is considered as significant, and used \
        for peak finding. (separate for high gain/low gain)').tag(
        config=True, allow_none=True)
    integration_clip_amp = Int(help = 'Amplitude in p.e. above which \
        the signal is clipped').tag(config=True, allow_none=True)
    integration_lwt = Int(0, help='Weight of the local pixel (0: peak from \
        neighbours only, 1: local pixel counts as much as any neighbour').tag(
        config=True, allow_none=True)
    integration_calib_scale = Float(0.92, help='Used for conversion from ADC \
        to pe. Identical to global variable CALIB_SCALE in reconstruct.c in \
        hessioxxx software package.').tag(config=True, allow_none=True)


    def __init__(self,parent,config):
        Component.__init__(self,parent)
        CalibrationStep.__init__(self)

    def init(self):
        CalibrationStep.init(self)

    def run(self, event):
        super().run(event)

    def finish(self):
        self.log.info("--- LSTCalibration finish ---")
        super().finish()
