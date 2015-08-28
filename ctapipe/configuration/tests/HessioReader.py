from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.configuration.core import Configuration, ConfigurationException
import threading


class HessioReader():
    
    
    def __init__(self,configuration=None):
        self.configuration = configuration
        self.raw_data = None
        self.source = None
        
    def init(self):
        
        print("--- HessioReader init ---")
        if self.configuration == None:
            print("HessioReader __init__: configuration == None")
            self.configuration = Configuration()
            self.configuration.read("./pipeline.ini", impl=Configuration.INI)
        else:
            print("HessioReader __init__: configuration == ", self.configuration)
            
        self.raw_data = self.configuration.get('source', section='HESSIO_READER')
        self.source = hessio_event_source(get_path(self.raw_data), max_events=10)
        
        
        
    def run(self):
        counter = 0
        for event in self.source:
            print("HessioReader get next event")
            counter+=1
            print("--< Start Event",counter,">--")#,end="\r")
            yield event
        print("\n--- HessioReader Done ---")
        