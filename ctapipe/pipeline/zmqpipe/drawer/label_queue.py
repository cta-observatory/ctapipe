from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys


class LabelQueue(QLabel):
    def __init__(self, *args):
        QLabel.__init__(self)
        self.data = dict()


    def pipechange(self, steps):
        """Called by ZmqSub instance when it receives zmq message from pipeline
        Update pipeline state (self.steps) and force to update drawing
        Parameters
        ----------
        topic : str
        """
        for step in steps:
            text ='{: ^30}{: ^10}{: ^8}\n'.format('Step','Queue', 'Done')
            text+='{:-^50}\n'.format('-')
            for step in steps:
                text+=self.formatText(step)+'\n'
            self.setText(text)

    def formatText(self,step):
        text = str()
        #short_name = step.name.replace('$$thread_number$$','_')
        name = '{:^20}'.format(step.name)
        queue = '{:^5}'.format(step.queue_length)
        done = '{:^5}'.format(step.nb_job_done)
        run = '{:^5}'.format(step.running)
        text = name+ '\t:' + str(queue) + '\t:' + str(done)
        return text
