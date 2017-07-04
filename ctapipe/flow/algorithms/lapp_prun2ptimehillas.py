from ctapipe.core import Component
from traitlets import Unicode
from subprocess import Popen
from ctapipe.flow.algorithms.build_command import build_command



class Prun2PtimeHillas(Component):

    exe = Unicode(help='executable').tag(
        config=True)

    config_file = Unicode(help='configuration file').tag(
        config=True)

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
            cmd, output_file = build_command(self.exe,  in_filename,
                                             output_dir=".",
                                             out_extension="ptimehillas",
                                             options=options)
            self.log.info("--- InOutProcess cmd {} --- in_filename {}".format(cmd, in_filename))
            proc = Popen(cmd)
            proc.wait()
            self.log.debug("--- CalibrationStep STOP ---")
            return  output_file

    def finish(self):
        self.log.debug("--- CalibrationStep finish ---")
