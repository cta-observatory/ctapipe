from ctapipe.calib.camera.dl1 import calibrate_event
from ctapipe.core import Component
from traitlets import Unicode
from traitlets import Int
from traitlets import Float
from traitlets import List


class CalibrationStep(Component):
    """CalibrationStep` class represents a Stage for pipeline.
        it executes ctapipe.calib.camera.calibrators.calibrate_event
        it return calibrated_event and geom_dict
    """
    integrator = Unicode('nb_peak_integration',
                         help=("integration scheme to be used" 
                               "to extract the charge")).tag(
                                   config=True)
    integration_window = List([7, 3], help='Set integration window width \
        and offset (to before the peak) respectively').tag(
        config=True)
    integration_sigamp = List([2, 4], help='Amplitude in ADC counts above \
        pedestal at which a signal is considered as significant, and used \
        for peak finding. (separate for high gain/low gain)').tag(
        config=True)
    integration_clip_amp = Int(default_value=None, allow_none=True, help='Amplitude in p.e. above which \
        the signal is clipped').tag(config=True)
    integration_lwt = Int(0, help='Weight of the local pixel (0: peak from \
        neighbours only, 1: local pixel counts as much as any neighbour').tag(
        config=True)
    integration_calib_scale = Float(0.92, help='Used for conversion from ADC to \
        pe. Identical to global variable CALIB_SCALE in reconstruct.c in \
        hessioxxx software package.').tag(config=True)


    def init(self):
        self.log.debug("--- CalibrationStep init ---")
        self.parameters = dict()
        self.parameters['integrator'] = self.integrator
        # 'nb_peak_integration'
        self.parameters['integration_window'] = self.integration_window
        self.parameters['integration_shift'] = self.integration_window[1]
        self.parameters['integration_sigamp'] = self.integration_sigamp
        self.parameters['integration_lwt'] = self.integration_lwt
        self.parameters['integration_calib_scale'] = self.integration_calib_scale

        if self.integration_clip_amp is not None:
            self.parameters['clip_amp'] = self.integration_clip_amp

        for key, value in sorted(self.parameters.items()):
            self.log.info("[%s] %s", key, value)
        return True

    def run(self, event):
        if event is not None:
            self.log.debug("--- CalibrationStep RUN --- %s", event.dl0.event_id)
            geom_dict = {}
            calibrated_event = calibrate_event(event, self.parameters,
                                               geom_dict)
            # for tel_id in calibrated_event.r0.tels_with_data:
            #    signals = calibrated_event.dl1.tel[tel_id].calibrated_image
            #    cmaxmin = (max(signals) - min(signals))
            self.log.debug("--- CalibrationStep STOP ---")
            return ([calibrated_event, geom_dict])

    def finish(self):
        self.log.debug("--- CalibrationStep finish ---")
