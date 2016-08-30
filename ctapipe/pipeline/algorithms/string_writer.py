from ctapipe.core import Component
from traitlets import Unicode
from time import sleep


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
        self.log.info('{} receive {}'.format(self.name, object))
        self.file.write(str(object) + "\n")



    def finish(self):
        self.log.info("--- StringWriter finish ---")
        self.file.close()
