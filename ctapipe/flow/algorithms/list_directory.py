from time import sleep
import threading
import subprocess
from glob import glob
from ctapipe.core import Component
from traitlets import Unicode


class ListDirectory(Component):
    """`ListDirectory` class represents a Producer for pipeline.
        It lists all files prensent in source_dir directory  and send them
        one by one to the next stage.
    """
    source_dir = Unicode('/tmp', help='directory contianing data files').tag(
        config=True, allow_none=False)
    regexpr = Unicode('*', help='regular expression to filter file').tag(
        config=True, allow_none=True)

    def init(self):
        return True

    def run(self):
        self.log.info('--- LIST DIRECTORY start ---')
        for full_path in glob(self.source_dir+'/'+self.regexpr):
            yield  full_path.rsplit("/",1)

    def finish(self):
        self.log.info('--- LIST DIRECTORY finish ---')
