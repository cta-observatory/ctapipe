import threading
import os
from ctapipe.core import Component
from traitlets import Unicode


class ListProducerProcess(Component):

    source_dir = Unicode('/tmp', help='directory containing data files').tag(config=True)

    def init(self):
        self.log.info('----- ListProducerProcess init  source_dir {}'.format(self.source_dir))
        return True

    def run(self):
        self.log.info('ListProducerProcess --- start ---')
        for input_file in os.listdir(self.source_dir):
            self.log.info('--- ListProducerProcess send  {} ---'.format(self.source_dir + "/" + input_file))
            yield self.source_dir + "/" + input_file

    def finish(self):
        self.log.info('--- {} finish ---'.format(threading.get_ident()))
