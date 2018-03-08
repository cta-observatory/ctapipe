#from PyQt4.QtCore import *
from PyQt4.QtGui import QLabel


class InfoLabel(QLabel):
    """
    Displays steps name, queues size and numbers of job done
    """

    def __init__(self, *args):
        QLabel.__init__(self)
        self.data = dict()
        self.mode = 'sequential'

    def pipechange(self, steps):
        """Called by GuiConnexion instance when it receives zmq message from
        pipeline Update pipeline state (self.steps) and force to
        update drawing

        """
        if steps:
            if self.mode == 'sequential':
                text = '  STEP                                       DONE    \n\n'
            else:
                text = '  STEP                                       RUNNING   QUEUE       DONE    \n\n'
            for step in steps:
                text += '{:-<80}\n'.format('-')
                text += self.formatText(step) + '\n'
            self.setText(text)

    def formatText(self, step):
        """ Format Step state

        Parameters
        ----------
        step : StagerRep to format

        Returns
        -------
        A str containg sdtep state
        """
        text = str()
        step_name = step.name.split('$$process')[0]
        if len(step_name) < 18:
            name = ' ' + step_name + ((18 - len(step_name)) * 2) * ' '
        elif len(step_name) > 18:
            name = ' ' + step_name[0:16] + '...'
        else:
            name = ' ' + step_name
        queue = step.queue_length
        done = step.nb_job_done
        nb_proc = step.nb_process
        running = step.running
        if self.mode == 'sequential':
            text = name + '\t' + str(done)
        else:
            text = name + '\t' + str(running) + '\\' + str(nb_proc) + \
                '\t' + str(queue) + '\t' + str(done)
        return text

    def reset(self):
        self.setText("")

    def mode_receive(self, mode):
        """
        Parameters
        ----------
        Flow mode (sequential or multiprocess)
        """
        self.mode = mode
