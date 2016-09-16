from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.core import Component
from traitlets import Unicode


class SimTelArrayReader(Component):

    """`SimTelArrayReader` class represents a Producer for pipeline.
        It opens simtelarray file and yields even to the next stage
    """
    filename = Unicode('gamma', help='simtel MC input file').tag(
        config=True, allow_none=True)
    source = None

    def init(self):
        self.log.debug("--- SimTelArrayReader init {}---".format(self.filename))
        try:
            in_file = get_path(self.filename)
            self.source = hessio_event_source(in_file,max_events=100)
            self.log.debug('{} successfully opened {}'.format(self.filename,self.source))
        except:
            self.log.error('could not open ' + in_file)
            return False
        return True

    def run(self):
        for event in self.source:
            yield (event)
        self.log.debug("\n--- SimTelArrayReader Done ---")

    def finish(self):
        self.log.debug("--- SimTelArrayReader finish ---")
        pass
