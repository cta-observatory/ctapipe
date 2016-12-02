from ctapipe.core import Tool
from ctapipe.calib.camera.charge_extraction import ChargeExtractorFactory, \
    LocalPeakIntegrator, ChargeExtractor
from ctapipe.io import CameraGeometry
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path
from traitlets import Dict, List, Int
import numpy as np


def get_test_parameters():
    parameters = {"integrator": "nb_peak_integration",
                  "integration_window": [7, 3],
                  "integration_sigamp": [2, 4],
                  "integration_lwt": 0}
    return parameters


def get_test_event():
    filename = get_path('gamma_test.simtel.gz')
    for event in hessio_event_source(filename):
        if event.dl0.event_id == 409:
            return event

# TEMP GLOBALS FOR PLAYING WITH
telid = 11
event = get_test_event()
nsamples = event.inst.num_samples[telid]
data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
ped = event.mc.tel[telid].pedestal
data_ped = data - np.atleast_3d(ped / nsamples)
data_ped = np.array([data_ped[0], data_ped[0]])
pixel_pos = event.inst.pixel_pos[telid]
optical_foclen = event.inst.optical_foclen[telid]

geom = CameraGeometry.guess(*pixel_pos, optical_foclen)
nei = geom.neighbors

params = get_test_parameters()


class CalTool(Tool):
    name = "mytool"
    description = "do some things and stuff"

    aliases = Dict(dict(extractor='ChargeExtractorFactory.extractor',
                        window_width='ChargeExtractorFactory.window_width',
                        ))
    classes = List([ChargeExtractorFactory])

    def setup(self):
        kwargs = dict(config=self.config, parent=self)

        self.extractor_factory = ChargeExtractorFactory(**kwargs)
        extractor_class = self.extractor_factory.get_class()
        self.extractor = extractor_class(waveforms=data_ped, nei=nei, **kwargs)

        pass

    def start(self):
        print(self.classes)

        print(self.extractor_factory.extractor)
        print(self.extractor_factory.window_width)
        print(self.extractor.window_width)

    def finish(self):
        print("fin")
        self.log.warning("Shutting down.")

t = CalTool()
t.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"

argv = ['--extractor','LocalPeakIntegrator', '--window_width', '10']
t.run(argv=argv)
argv = ['--extractor','SimpleIntegrator', '--window_width', '10', '--help']
t.run(argv)
