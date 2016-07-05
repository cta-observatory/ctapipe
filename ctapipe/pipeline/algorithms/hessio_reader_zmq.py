from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.configuration.core import Configuration, ConfigurationException
import threading
from copy import deepcopy
from sys import stderr
from ctapipe.pipeline import Coroutine

class HessioReader(Coroutine):

    def __init__(self,configuration=None):
        self.configuration = configuration
        self.raw_data = None
        self.source = None

    def init(self):
        print("--- HessioReader init ---")
        return True


    def run(self,input_file):
        try:
            self.source = hessio_event_source(input_file,max_events=10)
        except(RuntimeError):
            print("could not open",self.raw_data, file=stderr)
            return False
        counter = 0
        for event in self.source:
            event.dl0.event_id = counter
            counter+=1
            # send new job to next router/queue
            self.send_to_next_stage(event)
        print("\n--- HessioReader Done ---")
        return input_file+".bin"

    def finish(self):
        print ( "--- HessReader finish ---")
        pass
