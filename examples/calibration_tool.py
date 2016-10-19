from ctapipe.calib.camera.new_tool_proposal import FactoryTool
from ctapipe.calib.camera.charge_extraction import ChargeExtractorFactory
from ctapipe.io import CameraGeometry
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path
from traitlets import Dict, List
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
nsamples = event.dl0.tel[telid].num_samples
data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
ped = event.dl0.tel[telid].pedestal
data_ped = data - np.atleast_3d(ped / nsamples)
data_ped = np.array([data_ped[0], data_ped[0]])
pixel_pos = event.meta.pixel_pos[telid]
optical_foclen = event.meta.optical_foclen[telid]

geom = CameraGeometry.guess(*pixel_pos, optical_foclen)
nei = geom.neighbors

params = get_test_parameters()


class CalTool(FactoryTool):
    name = "mytool"
    description = "do some things and stuff"

    factories = Dict(dict(extractor=ChargeExtractorFactory))
    classes = List([])
    aliases = Dict(dict(extractor='ChargeExtractorFactory.extractor'))

    def setup(self):
        self.extractor_factory = self.factories['extractor'](
            config=self.config)
        self.extractor = self.extractor_factory.get_product(data_ped,
                                                            nei=[[2]],
                                                            config=self.config)
        pass

    def start(self):
        print(self.classes)
        print(self.extractor.window_width)
        print(self.extractor.extract_charge())
        # a = self.extractor(data_ped, None, config=self.config)
        # print(a.window_width)
        # print(self.products[0].window_width)

    def finish(self):
        print("fin")
        self.log.warning("Shutting down.")

t = CalTool()
argv = ['--extractor','LocalPeakIntegrator', '--window_width', '10']
t.run(argv)

argv = ['--extractor','SimpleIntegrator', '--window_width', '10', '--help']
t.run(argv)