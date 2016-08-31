from ctapipe.core import Component
from random import randint
from traitlets import Int
from time import sleep



class CompareInt(Component):
    """CompareInt class represents a Stage for pipeline.

    """
    expected = Int(10, help='expected value').tag(
        config=True, allow_none=True)

    def init(self):
        self.log.debug("--- CompareInt init ---")
        return True

    def run(self, val):
        self.log.info("CompareInt receive {}".format(val))
        for i in range(2000):
            foo = (val**i)
        if int(val) >= self.expected:
            return (val,'CONSUMER')
        else:
            return (val,'AddInt')



    def finish(self):
        self.log.debug("--- CompareInt finish ---")
        pass
