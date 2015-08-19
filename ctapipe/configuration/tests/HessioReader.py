from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.configuration.core import Configuration, ConfigurationException

__all_=['HessioReader']

class HessioReader:
    def __init__(self):
        self.raw_data  = None
        self.source = None
    
    def init(self):
        conf = Configuration()
        conf.read("./pipeline.ini", impl=Configuration.INI)
        self.raw_data = conf.get('source', section='HESSIO_READER')
        self.source = hessio_event_source(get_path(self.raw_data), max_events=100)

    def do_it(self):
        for event in self.source:
            yield  event

        