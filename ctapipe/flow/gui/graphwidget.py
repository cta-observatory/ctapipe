# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
classes used to display pipeline workload on a Qt.QWidget
"""
from graphviz import Digraph
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QWidget
from PyQt4.QtGui import QPainter
from PyQt4.QtGui import QImage
from PyQt4.QtGui import QLabel
from PyQt4.QtGui import QColor
from PyQt4.QtCore import QPoint
from PyQt4.QtGui import QPixmap
from PyQt4.QtGui import QGridLayout
from PyQt4.QtGui import QGraphicsScene
from PyQt4.QtGui import QGraphicsView
from PyQt4.QtSvg import QGraphicsSvgItem
from PyQt4.QtWebKit import QGraphicsWebView
from PyQt4.QtCore import Qt
from PyQt4.QtSvg import QSvgWidget
from PyQt4.QtSvg import QSvgRenderer
from time import sleep
import sys


class StagerRep():

    """
    class to represent a Stager, Consumer and Producer figure
    Parameters
    ----------
    name  : str
    next_steps : list(str)
    running : bool
    nb_job_done : int
    """
    STAGER = 1
    PRODUCER = 2
    CONSUMER = 3

    def __init__(self,name,next_steps=list(),running=0,
                nb_job_done=0, queue_length = 0, nb_processus = 1, step_type=STAGER):
        self.type = step_type
        self.name = name
        self.next_steps = next_steps
        self.running = running
        self.nb_job_done = nb_job_done
        self.queue_length = queue_length
        self.nb_processus = nb_processus



    def __repr__(self):
        """  called by the repr() built-in function and by string conversions
        (reverse quotes) to compute the "official" string representation of
        an object.  """
        return (self.name + ' running: '+
            str(self.running)+ '-> nb_job_done: '+
            str(self.nb_job_done) + '-> next_steps:' +
            str(self.next_steps)+ '-> queue_length:' +
            str(self.queue_length)+ '-> nb_processus:' +
            str(self.nb_processus))

    def get_statistics(self):
        return (self.name.split('$$processus')[0] + ' number of jobs done: '+ str(self.nb_job_done))


class GraphWidget(QWidget):

    """
    class that displays pipeline workload
    It receives pipeline information thanks to pipechange method
    """
    blue_cta = QColor(1, 39, 118)
    mygreen = QColor(65, 205, 85)

    def __init__(self, statusBar):
        super(GraphWidget, self).__init__()
        self.point_size = 1
        self.initUI()
        # self.steps contains all pipeline steps (Producer, Stager, consumer)
        self.steps = list()
        # self.config_time stores time when pipeline send its config
        self.config_time = 0.
        self.statusBar = statusBar
        self.zoom = 2
        self.table_queue = None
        self.changed = True
        self.last_diagram_bytes = None

    def set_table_queue(self,table_queue):
        self.table_queue = table_queue

    def initUI(self):
        #self.setGeometry(300, 300, 280, 170)
        self.setWindowTitle('PIPELINE')
        self.show()

    def paintEvent(self, e):
        """This event handler is reimplemented in this subclass to receive
        paint events passed in event. A paint event is a request to repaint all
        or part of a widget. It can happen for one of the following reasons:
        repaint() or update() was invoked,the widget was obscured and has
        now been uncovered, or many other reasons. """
        qp = QPainter()
        qp.begin(self)


        if not self.steps and self.last_diagram_bytes:
            diagram_bytes = self.last_diagram_bytes
        else:
            diagram = self.build_graph()
            diagram_bytes = diagram.pipe('svg')
            self.last_diagram_bytes = diagram_bytes
        svg = QSvgRenderer(diagram_bytes)
        svg.render(qp)
        qp.end()

    def pipechange(self, steps):
        """Called by GuiConnexion instance when it receives zmq message from pipeline
        Update pipeline state (self.steps) and force to update drawing
        Parameters
        ----------
        """
        if steps:
            self.steps = steps
            self.update()

    def reset(self):
        self.steps = list()
        self.last_diagram_bytes = None

    def build_graph(self):
        """
        Return a graphiz.Digraph
        """
        g = Digraph('test', format='svg',graph_attr={'bgcolor':'lightgrey'})
        for step in  self.steps:
            str_shape = 'octagon'
            if step.type == StagerRep.CONSUMER:
                str_shape = 'doubleoctagon'
            if step.type == StagerRep.PRODUCER:
                str_shape = 'Mdiamond'
            name = step.name.split('$$processus')[0]
            # in case of multiprocessus mode, we need to keep all processus
            # running state for a special step
            # we consider the step running if at least one processus is running

            name = self.format_name(step.name.split('$$processus')[0])
            if step.running > 0:
                g.node(name,color='lightblue', style='filled',shape=str_shape,area='0.5')
            else:
                g.node(name,shape=str_shape,color='blue',area='0.5')

        for step in self.steps:
            step_name = self.format_name(step.name.split('$$processus')[0])
            for next_step_name in step.next_steps:
                next_step = self.get_step_by_name(next_step_name.split('$$processus')[0])
                if next_step:
                    #for i in range(step.nb_processus):
                    next_step_name_formated = self.format_name(next_step.name.split('$$processus')[0])
                    g.edge(step_name, next_step_name_formated)
                    g.edge_attr.update(arrowhead='empty', arrowsize='1',color='purple')
        return g

    def format_name(self,name, max_car=15):
        if len(name) > max_car:
            name = name[0:max_car] + '...'
        return name


    def get_step_by_name(self, name):
        ''' Find a PipeStep in self.producer_step or  self.stager_steps or
        self.consumer_step
        Return: PipeStep if found, otherwise None
        '''
        for step in self.steps:
            if step.name.split('$$processus')[0] == name:
                return step
        return None
