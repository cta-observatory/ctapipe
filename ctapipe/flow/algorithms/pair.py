from ctapipe.core import Component
from time import sleep


class Pair(Component):
    """Odd` class represents a Stage for pipeline.
    It returns received value to Add stage except when
    received value is a multiple of 5. In this case it returns
    received value to Inverse stage
    """

    def init(self):
        self.log.debug("--- Pair init ---")
        return True

    def run(self, _input):
        sleep(.5)
        self.log.debug("Pair receive {}".format(_input))
        if _input % 5 == 0:
            return (_input, 'Inverse')
        else:
            return (_input, 'Add')

    def finish(self):
        self.log.debug("--- Pair finish ---")
        pass
