from time import sleep
import threading
import subprocess
import os
from ctapipe.core import Component
from traitlets import Unicode
from ctapipe.pipeline.algorithms.base_process import build_command


class Prun2PtabHillas(Component):

    executable = Unicode('/tmp', help='directory contianing data files').tag(
        config=True, allow_none=False)
    output_dir = Unicode( help='directory contianing produced files').tag(
        config=True, allow_none=False)
    out_extension  = 'png'

    def init(self):
        if not os.path.exists(self.output_dir):
            try:
                os.mkdir(self.output_dir)
            except OSError:
                self.log.error(
                    "{}: could not create output directory {}"
                    .format(self.section_name,self.output_dir ))
                return False
        return True

    def run(self, input_file):
        #executable, input_file, source_dir, output_dir, out_extension, options
        cmd, output_file = build_command(self.executable, input_file,
         self.output_dir, self.out_extension)
        proc = subprocess.Popen(cmd)
        proc.wait()
        return output_file

    def finish(self):
        self.log.info('--- {} finish ---'.format(self.section_name))
