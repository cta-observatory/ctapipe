from ctapipe.core import Component
from time import sleep


class Odd(Component):
    """Odd` class represents a Stage for pipeline.
    It returns received value to Inverse stage except when
    received value is a multiple of 5. In this case it returns
    received value to Add stage
    """

    def init(self):
        self.log.debug("--- Odd init ---")
        return True

    def run(self, _input):
        sleep(.5)
        self.log.debug("ODD receive {}".format(_input))
        if _input % 5 == 0:
            return (_input, 'Add')
        else:
            return (_input, 'Inverse')

    def finish(self):
        self.log.debug("--- Odd finish ---")
        pass
