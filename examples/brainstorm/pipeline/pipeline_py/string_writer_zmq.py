from ctapipe.configuration.core import Configuration, ConfigurationException
from time import sleep

class StringWriter:
        def __init__(self,configuration=None):
            self.file = None
            self.conf = configuration
            
        
        def init(self):
            filename = self.conf.get('filename', section='WRITER')
            self.file = open(filename, 'w')
            print("--- StringWriter init ---")
            return True

        
        def run(self,object):
            if ( object != None):
                print("--- Writer object ---")
                self.file.write(str(object)+"\n")
        
        
        def finish(self):
            print("--- StringWriter finish ---")
            self.file.close()