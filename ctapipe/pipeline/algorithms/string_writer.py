from ctapipe.configuration.core import Configuration, ConfigurationException
from ctapipe.core import Component

class StringWriter(Component):

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
