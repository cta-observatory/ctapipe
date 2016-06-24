from traitlets.config import Application
from traitlets import Unicode


class Tool(Application):

    config_file = Unicode( help="name of configuration file with parameters")\
        .tag(config=True)

    def _setup(self, argv=None):

        self.parse_command_line(argv)
        if self.config_file:
            self.load_config_file(self.config_file)
        self.initialize()

    def finish(self):
        """ finish up (override in subclass) """
        self.log.info("Goodbye")
        
    def run(self, argv=None):
        self._setup(argv)
        self.log.info("Starting: {}".format(self.name))
        self.log.debug("CONFIG: {}".format(self.config))
        self.start()
        self.finish()
