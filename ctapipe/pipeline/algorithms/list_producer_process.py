from time import sleep
import threading
import subprocess
import os
from ctapipe.pipeline.algorithms.base_process import BaseProcess


class ListProducerProcess(BaseProcess):

    def __init__(self, configuration=None):
        BaseProcess.__init__(self, configuration)

    def run(self):

        print('---', self.section_name, threading.get_ident(), 'start ---')
        for input_file in os.listdir(self.source_dir):
            yield self.source_dir + "/" + input_file

    def finish(self):
        print('---', self.section_name, threading.get_ident(), 'finish ---')
