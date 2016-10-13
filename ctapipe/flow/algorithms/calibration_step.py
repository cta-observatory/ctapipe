from ctapipe.utils.datasets import get_path
import ctapipe.instrument.InstrumentDescription as ID
from ctapipe.calib.camera.calibrators import calibrate_event
from ctapipe.calib.camera.integrators import integrator_dict
from ctapipe.core import Component
from traitlets import Unicode
from traitlets import Int
from traitlets import Float
from traitlets import List
import numpy as np

class CalibrationStep(Component):
    """CalibrationStep` class represents a Stage for pipeline.
        it dumps RawCameraData contents to a string
    """
    integrator = Unicode('nb_peak_integration',
     help='integration scheme to be used to extract the charge').tag(
        config=True)
    integration_windows = List([7, 3],help='Set integration window width \
        and offset (to before the peak) respectively' ).tag(
        config=True)
    integration_sigamp = List([2, 4], help='Amplitude in ADC counts above \
        pedestal at which a signal is considered as significant, and used \
        for peak finding. (separate for high gain/low gain)').tag(
        config=True)
    integration_clip_amp = Int(default_value=None,allow_none=True, help = 'Amplitude in p.e. above which \
        the signal is clipped').tag(config=True)
    integration_lwt = Int(0, help='Weight of the local pixel (0: peak from \
        neighbours only, 1: local pixel counts as much as any neighbour').tag(
        config=True)
    integration_calib_scale = Float(0.92,help='Used for conversion from ADC to \
        pe. Identical to global variable CALIB_SCALE in reconstruct.c in \
        hessioxxx software package.').tag(config=True)

    def init(self):
        self.log.info("--- CalibrationStep init ---")
        self.parameters = dict()
        self.parameters['integrator'] = 'nb_peak_integration'
        self.parameters['window'] = self.integration_windows[0]
        self.parameters['shift'] =  self.integration_windows[1]
        self.parameters['sigamp'] = self.integration_sigamp
        self.parameters['lwt'] = self.integration_lwt
        self.parameters['calib_scale'] = self.integration_calib_scale

        if self.integration_clip_amp != None:
                self.parameters['clip_amp'] = self.integration_clip_amp
        for key, value in sorted(self.parameters.items()):
            self.log.info("[{}] {}".format(key, value))
        return True

    def run(self, event):
        if event != None:
            geom_dict = {}
            calibrated_event = calibrate_event(event,self.parameters,geom_dict)
            for tel_id in calibrated_event.dl0.tels_with_data:
                signals = calibrated_event.dl1.tel[tel_id].pe_charge
                cmaxmin = (max(signals) - min(signals))
            pp = None
            return ([calibrated_event,geom_dict,pp])

    def finish(self):
        self.log.info("--- CalibrationStep finish ---")
        pass
