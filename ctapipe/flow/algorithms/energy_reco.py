from ctapipe.core import Component


class EnergyReco(Component):
    """Add class represents a Stage for pipeline.
    It returns inverted value of received value
    """
    def init(self):
        self.log.debug("--- EnergyReco init ---")
        return True

    def run(self, _input):
        if input:
            self.log.info("EnergyReco receive {}".format(_input))
            return _input

    def finish(self):
        self.log.debug("--- EnergyReco finish ---")
        pass
