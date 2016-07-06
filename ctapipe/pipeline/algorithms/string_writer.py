from ctapipe.configuration.core import Configuration, ConfigurationException

class StringWriter:
        def __init__(self,configuration=None):
            self.conf = configuration


        def init(self):
            filename = '/tmp/test.txt'
            self.file = open(filename, 'w')
            print("--- StringWriter init ---")
            return True


        def run(self,object):
            if ( object != None):
                #print("--- Writer object ---")
                self.file.write(str(object)+"\n")


        def finish(self):
            print("--- StringWriter finish ---")
            self.file.close()
