from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys


class InfoLabel(QLabel):
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
            text ='  STEP                                       PROCESS   QUEUE       DONE    \n\n'
            for step in steps:
                text+='{:-<80}\n'.format('-')
                text+=self.formatText(step)+'\n'
            self.setText(text)

    def formatText(self,step):
        text = str()
        if len(step.name) < 18:
            name = ' ' + step.name + ((18-len(step.name))*2)*' '
        elif len(step.name) > 18:
            name =  ' ' + step.name[0:16] + '...'
        else:
            name =  ' ' + step.name
        queue = step.queue_length
        done = step.nb_job_done
        nb_proc = step.nb_processus
        text = name+ '\t' + str(nb_proc) + '\t' + str(queue) + '\t' + str(done)
        return text


    def reset(self):
        self.setText("")
