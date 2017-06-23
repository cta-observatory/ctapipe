from ctapipe.core import Component
from traitlets import Unicode
from subprocess import Popen
from ctapipe.flow.algorithms.build_command import build_command



class PtimeHillas2PRecoEvent(Component):

    exe = Unicode(help='executable').tag(
        config=True, allow_none=False)

    config_file = Unicode(help='configuration file').tag(
        config=True, allow_none=False)

    """CalibrationStep` class represents a Stage for pipeline.
        it executes prun2pcalibrun
    """
    def init(self):
        self.log.debug("--- CalibrationStep init ---")
        if self.exe:

            return True
        return False

    def run(self, in_filename):
        if in_filename != None:
            options = ["-c", self.config_file]
            cmd, _ = build_command(self.exe, in_filename, options=options,
                                   output_dir=".", out_extension="precoevent")
            self.log.info("--- InOutProcess cmd {} --- in_filename {}".format(cmd, in_filename))
            proc = Popen(cmd)
            proc.wait()
            self.log.debug("--- CalibrationStep STOP ---")
            return ("file {} Done".format(in_filename))

    def finish(self):
        self.log.debug("--- CalibrationStep finish ---")
        pass

