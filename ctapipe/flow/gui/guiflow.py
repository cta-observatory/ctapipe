import sys
from ctapipe.core import Tool
from traitlets import Integer
from ctapipe.flow.gui.main_window import ModuleApplication


class PipeGui(Tool):
    """
    Main program that lunches GUI
    This GUI communicate with pipeline  elements (stager, consumer, producer,
     router) tahnks to ZMQ library PUB/SUB message
    Every second, pipeline send its full configuration:
     - producer
     - levels of stages
     - consumer
     - router / queue
     Each time a producer, stager , consumer or router changes, it send its status
     to this GUI
    """
    description = "run stages in multiprocess pipeline"
    port = Integer(5565, help='GUI port for pipelien connection').tag(
        config=True, allow_none=True)

    def start(self):
        ModuleApplication(sys.argv, self.port)

    def setup(self):
        pass

    def finish(self):
        pass


def main():
    gui = PipeGui()
    gui.run()

if __name__ == "__main__":
    main()
