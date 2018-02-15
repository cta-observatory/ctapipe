from graphviz import Digraph
from PyQt4.QtGui import QWidget
from PyQt4.QtGui import QPainter
from PyQt4.QtSvg import QSvgRenderer
from ctapipe.flow.stager_rep import StagerRep


class GraphWidget(QWidget):

    """
    class that displays pipeline workload.
    It receives pipeline information thanks to pipechange method
    """

    def __init__(self, statusBar):
        super(GraphWidget, self).__init__()
        self.initUI()
        # self.steps contains all pipeline steps (Producer, Stagers, consumer)
        self.steps = list()
        self.statusBar = statusBar
        self.last_diagram_bytes = None

    def initUI(self):
        self.show()

    def paintEvent(self, e):
        """This event handler is reimplemented in this subclass to receive
        paint events passed in event. A paint event is a request to repaint all
        or part of a widget. It can happen for one of the following reasons:
        repaint() or update() was invoked,the widget was obscured and has
        now been uncovered, or many other reasons. """
        qp = QPainter()
        qp.begin(self)
        # if no steos have been receive draw last_diagram_bytes
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
        """
        Called by GuiConnexion instance when it receives zmq message from Flow.
        Update pipeline state (self.steps) and force to update drawing

        Parameters
        ----------
        steps: list of (StagerRep)
        """
        if steps:
            self.steps = steps
            self.update()

    def reset(self):
        """Clear self.steps and self.last_diagram_bytes
        """
        self.steps = list()
        self.last_diagram_bytes = None

    def build_graph(self):
        """
        Returns
        -------
        graphiz.Digraph
            It contains nodes and links corresponding to self.steps
        """
        g = Digraph('test', format='svg', graph_attr={'bgcolor': 'lightgrey'})
        # Create nodes
        for step in self.steps:
            str_shape = 'octagon'
            if step.type == StagerRep.CONSUMER:
                str_shape = 'doubleoctagon'
            if step.type == StagerRep.PRODUCER:
                str_shape = 'Mdiamond'
            name = step.name.split('$$process')[0]
            name = self.format_name(step.name.split('$$process')[0])
            if step.running > 0:
                g.node(name, color='lightblue', style='filled',
                       shape=str_shape, area='0.5')
            else:
                g.node(name, shape=str_shape, color='blue', area='0.5')
        # Create edges
        for step in self.steps:
            step_name = self.format_name(step.name.split('$$process')[0])
            for next_step_name in step.next_steps:
                next_step = self.get_step_by_name(next_step_name.split('$$process')[0])
                if next_step:
                    next_step_name_formated = self.format_name(
                        next_step.name.split('$$process')[0])
                    g.edge(step_name, next_step_name_formated)
                    g.edge_attr.update(arrowhead='empty', arrowsize='1', color='purple')
        return g

    def format_name(self, name, max_car=15):
        """
        trim name if its length is more than max_car and add 3 points

        Returns
        -------
        Name triped or not
        """
        if len(name) > max_car:
            name = name[0:max_car] + '...'
        return name

    def get_step_by_name(self, name):
        ''' Find a PipeStep in self.producer_step or  self.stager_steps or
        self.consumer_step

        Parameters
        ----------
        name : str Step name

        Returns
        -------
        PipeStep if found, otherwise None
        '''
        for step in self.steps:
            if step.name.split('$$process')[0] == name:
                return step
        return None
