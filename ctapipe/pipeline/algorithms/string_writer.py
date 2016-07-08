from ctapipe.configuration.core import Configuration, ConfigurationException
from ctapipe.core import Component

class StringWriter(Component):

        def __init__(self, configuration=None):
            super().__init__(parent=None)
            self.conf = configuration

        def init(self):
            filename = '/tmp/test.txt'
            self.file = open(filename, 'w')
            self.log.info("--- StringWriter init ---")
            return True

        def run(self, object):
            if (object != None):
                self.file.write(str(object) + "\n")

        def finish(self):
            self.log.info("--- StringWriter finish ---")
            self.file.close()
