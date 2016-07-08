from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.configuration.core import Configuration, ConfigurationException
import threading
from sys import stderr


class HessioReader():

    def __init__(self, configuration=None):
        self.configuration = configuration

    def init(self):
        print("--- HessioReader init ---")
        return True

    def run(self):
        try:
            filename = get_path('gamma_test.simtel.gz')
            source = hessio_event_source(filename, max_events=10)
        except(RuntimeError):
            print("could not open gamma_test.simtel.gz", file=stderr)
            return False
        counter = 0
        for event in source:
            event.dl0.event_id = counter
            counter += 1
            # send new job to next step thanks to router
            yield event
        print("\n--- HessioReader Done ---")

    def finish(self):
        print ("--- HessReader finish ---")
        pass
