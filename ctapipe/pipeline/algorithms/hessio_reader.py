from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.configuration.core import Configuration, ConfigurationException
import threading
from sys import stderr
from ctapipe.core import Component


class HessioReader(Component):


    def __init__(self, configuration=None):
        super().__init__(parent=None)
        self.configuration = configuration

    def init(self):
        self.log.info("--- HessioReader init ---")
        return True

    def run(self):
        try:
            filename = get_path('gamma_test.simtel.gz')
            source = hessio_event_source(filename, max_events=10)
        except(RuntimeError):
            self.log.info("could not open gamma_test.simtel.gz", file=stderr)
            return False
        counter = 0
        for event in source:
            event.dl0.event_id = counter
            counter += 1
            # send new job to next step thanks to router
            yield event
        self.log.info("\n--- HessioReader Done ---")

    def finish(self):
        self.log.info ("--- HessReader finish ---")
        pass
