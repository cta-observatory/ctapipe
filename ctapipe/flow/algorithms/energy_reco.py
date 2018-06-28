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
            hillas_file = _input[0]
            reco_event_file = _input[1]
            self.log.info("EnergyReco receive hillas_file {}".format(hillas_file))
            self.log.info("EnergyReco receive reco_event_file {}".format(reco_event_file))
            return _input

    def finish(self):
        self.log.debug("--- EnergyReco finish ---")
        pass
