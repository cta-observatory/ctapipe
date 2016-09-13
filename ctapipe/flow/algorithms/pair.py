from ctapipe.core import Component



class Pair(Component):
    """Odd` class represents a Stage for pipeline.

    """
    def init(self):
        self.log.info("--- Pair init ---")
        return True

    def run(self, _input):
        self.log.debug("Pair receive {}".format(_input))
        if _input % 2 == 0:
            return (_input, 'Add')
        else:
            return (_input, 'Inverse')


    def finish(self):
        self.log.debug("--- Pair finish ---")
        pass
