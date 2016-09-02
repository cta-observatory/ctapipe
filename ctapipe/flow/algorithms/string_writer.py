from ctapipe.core import Component
from traitlets import Unicode
from time import sleep
import os


class StringWriter(Component):

    """`StringWriter` class represents a Stage or a Consumer for pipeline.
        It writes received objects to file
    """
    filename = Unicode('/tmp/test.txt', help='output filename').tag(
        config=True, allow_none=True)

    def init(self):
        self.file = open(self.filename, 'w')
        self.log.info("--- StringWriter init filename {}---".format(self.filename))
        return True

    def run(self, object):
        self.file.write(str(object) + "\n")
        self.log.info('StringWriter write {}'.format( object))


    def finish(self):
        self.log.info("--- StringWriter finish START ---")
        self.file.close()
        self.log.info("--- StringWriter finish STOP file close {}---".format(self.file.closed))
