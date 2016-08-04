from time import sleep
import threading
from ctapipe.core import Component


class ListTelsWithData(Component):

    def init(self):
        self.log.info("--- ListTelsWithData init ---")

    def run(self, event):
        if event != None:
            res = list(event.dl0.tels_with_data)
            return res

    def finish(self):
        self.log.info("--- ListTelsWithData finish ---")
