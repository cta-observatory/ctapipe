from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.configuration.core import Configuration, ConfigurationException
import threading
from copy import deepcopy
from sys import stderr


class HessioReader():
    
    
    def __init__(self,configuration=None):
        self.configuration = configuration
        self.raw_data = None
        self.source = None
        
    def init(self):
        
        print("--- HessioReader init ---")
        if self.configuration == None:
            self.configuration = Configuration()
            success = self.configuration.read("./pipeline.ini", impl=Configuration.INI)
            if len(success) == 0: return False  
            
        self.raw_data = self.configuration.get('source', section='HESSIO_READER')
        try:
            self.source = hessio_event_source(get_path(self.raw_data), max_events=10)
            return True
        except(RuntimeError):
            print("could not open",self.raw_data, file=stderr)
            return False
        
        
        
    def run(self):
        counter = 0
        for event in self.source:
            counter+=1
            print("--<HessioReader: Start Event",counter,">--")#,end="\r")
            yield deepcopy(event)
        print("\n--- HessioReader Done ---")
        
        
        
    def finish(self):
        print ( "--- HessReader finish ---")
        pass
        
        