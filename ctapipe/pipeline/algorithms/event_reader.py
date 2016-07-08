from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.configuration.core import Configuration, ConfigurationException
import threading
from copy import deepcopy
from sys import stderr
import pickle
from ctapipe.core import Container


class EventReader():

    def __init__(self, configuration=None):
        self.configuration = configuration
        self.raw_data = None

    def init(self):
        print("--- EventReader init ---")
        return True

    def run(self, input_file):
        return len(input_file)
        counter = 0
        infile = open(input_file, "rb")
        events = pickle.load(infile)
        for event in events:
            if isinstance(event, Container):
                event.dl0.event_id = counter
                counter += 1
                # send new job to next router/queue
                yield event
        print("\n--- HessioReader Done ---")
        return

    def finish(self):
        print ("--- HessReader finish ---")
        pass
