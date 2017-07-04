from ctapipe.core import Component
from traitlets import Unicode
from subprocess import Popen
from ctapipe.flow.algorithms.build_command import build_command



class MergeTimeRecoSimu(Component):

    exe = Unicode(help='executable').tag(
        config=True)

    config_file = Unicode(help='configuration file').tag(
        config=True)

    """MergeTimeRecoSimu` class represents a Stage for pipeline.
        it executes prun2pcalibrun
    """
    def init(self):
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
            proc = Popen(cmd)
            proc.wait()
            self.log.debug("--- MergeTimeRecoSimu STOP ---")
            return  output_file

    def finish(self):
        self.log.debug("--- MergeTimeRecoSimu finish ---")
