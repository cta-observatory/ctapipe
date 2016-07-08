from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.configuration.core import Configuration, ConfigurationException
import threading
from ctapipe.core import Component
from traitlets import Unicode


class HessioReader(Component):

    filename = Unicode('gamma_test.simtel.gz', help='simtel MC input file').tag(
        config=True, allow_none=True)

    def init(self):
        self.log.info("--- HessioReader init ---")
        print("DEBUG filename",self.filename)
        return True

    def run(self):
        try:
            in_file = get_path(self.filename)
            source = hessio_event_source(in_file, max_events=10)
        except(RuntimeError):
            self.log.error('could not open ' + in_file)
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
