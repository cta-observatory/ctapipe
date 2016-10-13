from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.core import Component
from traitlets import Unicode


class SimTelArrayReader(Component):

    """`SimTelArrayReader` class represents a Producer for pipeline.
        It opens simtelarray file and yiekld even in run method
    """
    filename = Unicode('gamma', help='simtel MC input file').tag(
        config=True, allow_none=True)
    source = None

    def init(self):
        self.log.info("--- SimTelArrayReader init {}---".format(self.filename))
        try:
            in_file = get_path(self.filename)
            self.source = hessio_event_source(in_file,max_events=10)
            self.log.info('{} successfully opened {}'.format(self.filename,self.source))
        except:
            self.log.error('could not open ' + in_file)
            return False
        return True

    def run(self):
        counter = 0

        for event in self.source:
            event.dl0.event_id = counter
            counter += 1
            # send new job to next step thanks to router

            yield (event)

        self.log.info("\n--- SimTelArrayReader Done ---")

    def finish(self):
        self.log.info("--- SimTelArrayReader finish ---")
        pass
