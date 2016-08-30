from time import sleep
from ctapipe.core import Component


class ListTelsWithData(Component):

    """`ListTelsWithData` class represents a Stage for pipeline.
        It receives  a hessio event and return a list containing
        telescope id of triggered telescopes for this event.
    """

    def init(self):
        self.log.info("--- ListTelsWithData init ---")

    def run(self, event):
        if event is not None:
            res = list(event.dl0.tels_with_data)
            yield((res,event))

    def finish(self):
        self.log.info("--- ListTelsWithData finish ---")
