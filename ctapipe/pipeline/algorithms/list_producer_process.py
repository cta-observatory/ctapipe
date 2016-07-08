from time import sleep
import threading
import subprocess
import os
from ctapipe.core import Component
from traitlets import Unicode


class ListProducerProcess(Component):

    source_dir = Unicode('/tmp', help='directory contianing data files').tag(
        config=True, allow_none=False)

    def init(self):
        return True

    def run(self):
        print('---', self.section_name, threading.get_ident(), 'start ---')
        for input_file in os.listdir(self.source_dir):
            yield self.source_dir + "/" + input_file

    def finish(self):
        print('---', self.section_name, threading.get_ident(), 'finish ---')
