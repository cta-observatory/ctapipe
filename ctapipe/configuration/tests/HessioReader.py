from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.configuration.core import Configuration, ConfigurationException

__all_=['HessioReader']

class HessioReader:
    def __init__(self,_conf):
        self.raw_data  = None
        self.source = None
        self.next_instance = None
        self.conf = _conf
        self.counter = 0
    
    def init(self):
        print("--- HessioReader init ---")
        self.raw_data = self.conf.get('source', section='HESSIO_READER')
        self.source = hessio_event_source(get_path(self.raw_data), max_events=100)
        
        self.next_instance = self.__dict__["next_instance"]
        
    def do_it(self):
        for event in self.source:
            self.next_instance.do_it(event)
            self.counter+=1
            print("--< Start Event",self.counter,">--",end="\r")
        print("\n--- Done ---")

    def finish(self):
        print("--- HessioReader finish ---")
        self.next_instance.finish()
        