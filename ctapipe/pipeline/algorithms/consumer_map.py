from time import sleep
import threading
import subprocess
import os
from ctapipe.core import Component
from traitlets import Unicode
from ctapipe.pipeline.algorithms.base_process import build_command


class ConsumerMap(Component):

    executable = Unicode('/tmp', help='directory contianing data files').tag(
        config=True, allow_none=False)
    out_extension  = 'png'
    output_dir = Unicode('/tmp', help='directory contianing data files').tag(
            config=True, allow_none=False)

    def init(self):
        return True

    def run(self, input_file):
        cmd, output_file = build_command(self.executable, input_file,
         self.output_dir, self.out_extension)
        proc = subprocess.Popen(cmd)
        proc.wait()


    def finish(self):
        self.log.info('--- {} finish ---'.format( self.section_name))
