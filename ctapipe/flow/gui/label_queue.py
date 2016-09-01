from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys


class LabelQueue(QLabel):
    """
    Displays steps name, queues size and numbers of job done
    """
    def __init__(self, *args):
        QLabel.__init__(self)
        self.data = dict()


    def pipechange(self, steps):
        """Called by GuiConnexion instance when it receives zmq message from pipeline
        Update pipeline state (self.steps) and force to update drawing
        Parameters
        ----------
        topic : str
        """

        if steps:
            for step in steps:
                text ='{: ^40}{: ^15}{: ^15}\n'.format('Step','Queue', 'Done')
                text+='{:-^60}\n'.format('-')
                for step in steps:
                    text+=self.formatText(step)+'\n'
                self.setText(text)

    def formatText(self,step):
        text = str()

        name = '{:^30}'.format(step.name)
        queue = '{:^10}'.format(step.queue_length)
        done = '{:^8}'.format(step.nb_job_done)
        text = name+ '\t:' + str(queue) + '\t:' + str(done)
        return text
