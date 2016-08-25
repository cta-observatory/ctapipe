from ctapipe.utils.datasets import get_path
import ctapipe.instrument.InstrumentDescription as ID
from ctapipe.core import Component
from traitlets import Unicode


LST=1
OTHER = 2
class ShuntTelescope(Component):
    """ShuntTelescope class represents a Stage for pipeline.
        It shunts event based on telescope type
    """


    def init(self):
        self.log.info("--- ShuntTelescope init ---")
        self.telescope_types=dict()
        for index in range(50):
            self.telescope_types[index]=LST
        for index in range(50,200,1):
            self.telescope_types[index]=OTHER
        return True

    def run(self, event):
        triggered_telescopes = event.dl0.tels_with_data
        for telescope_id in triggered_telescopes:
            if self.telescope_types[telescope_id] == LST:
                yield(event.dl0.tel[telescope_id],'LSTDump')
            if self.telescope_types[telescope_id] == OTHER:
                yield(event.dl0.tel[telescope_id],'OtherDump')



    def finish(self):
        self.log.info("--- ShuntTelescope finish ---")
        pass
