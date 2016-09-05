from ctapipe.core import Component
from traitlets import Unicode


class StringWriter(Component):

    """`StringWriter` class represents a Stage or a Consumer for pipeline.
        It writes received objects to file
    """
    filename = Unicode('/tmp/test.txt', help='output filename').tag(
        config=True, allow_none=True)

    def init(self):
        self.file = open(self.filename, 'w')
        self.log.info("--- StringWriter init ---")
        return True

    def run(self, object):
        if (object is not None):
            self.file.write(str(object) + "\n")

    def finish(self):
        self.log.info("--- StringWriter finish ---")
        self.file.close()
