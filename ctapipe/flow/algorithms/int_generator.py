from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.core import Component
from time import sleep
from traitlets import Unicode


class IntGenerator(Component):

    """IntGenerator class represents a Producer for pipeline.
    """
    def init(self):
        return True

    def run(self):
        for i in range(500):
            self.log.info("IntGenerator send {}".format(i))
            yield i
            """
            if i%5 == 0 :
                yield(i,'Mod')
            elif i%2 == 0 :
                yield(i,'Pair')
            else :
                #self.send_msg(i,'Odd')
                yield(i,'Odd')
            """





        self.log.debug("\n--- IntGenerator Done ---")

    def finish(self):
        self.log.debug("--- IntGenerator finish ---")
        pass
