from traitlets import Unicode
from time import sleep
import subprocess
import os
from ctapipe.core import Component
from traitlets import Unicode


class MySort(Component):
    output_dir = Unicode('/tmp/test/out3', help='directory receving produced data').tag(
    config=True, allow_none=False)

    def init(self):
        if self.output_dir == None:
            self.log.error("MySort :output_dir is not defined")
            return False
        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
            except OSError:
                self.log.error("MySort: could not create output_dir {}".format(self.output_dir))
                return False
        return True

    def run(self,input_file):
        self.log.info('MySort{}'.format(input_file))
        output_file =  self.output_dir + '/'+input_file.split('/')[-1]
        cmd = 'sort ' + input_file + ' > '+  output_file
        proc = subprocess.Popen(cmd,shell=True, stderr=subprocess.STDOUT)
        proc.wait()

    def finish(self):
        print("--- MySort finish ---")
