# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
classes used to display pipeline workload on a Qt.QWidget
"""
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from PyQt4.QtCore import QPointF, QLineF
from PyQt4.QtGui import QColor, QPen
from ctapipe.pipeline.zmqpipe.pipeline_zmq import StepInfo

GAP_Y = 60
STAGE_SIZE_X = 80
STAGE_SIZE_Y = 40
PROD_CONS_SIZE_X = 100
PROD_CONS_SIZE_Y = 40
ROUTER_SIZE_X = 80
ROUTER_SIZE_Y = 40
STAGE_GAP_X = STAGE_SIZE_X + 10


class FigureRep():

    """
    Mother class to represent a drawing figure
    Parameters
    ----------
    center : PySide.QtCore.QPointF
        figure's center
    size : int
        figure's size in pixel
    name : str
        figure's name. Used to match with pipeline producer/stager/consumer
        when receive new information dform pipeline
    """

    def __init__(self, center=QPointF(0, 0), size_x=0, size_y=0, fig_type=StepInfo.STAGER, name=None):
        self.center = center
        self.size_x = size_x
        self.size_y = size_y
        self.name = name
        self.fig_type = fig_type

    def setLength(self, size):
        """Length setter
        Parameters
        ----------
        size : list of int
            list[0] -> figure's size x in pixel
            list[1] -> figure's size y in pixel
        """
        self.size_x = size[0]
        self.size_y = size[1]

    def setCenter(self, center):
        """Center setter
        Parameters
        ----------
        center: PySide.QtCore.QPointF
            figure's center in pixel
        """
        self.center = center

    def __repr__(self):
        """ called by the repr() built-in function and by string conversions
         (reverse quotes) to compute the "official" string representation of
          an object."""
        return self.name + " -> Center: " + str(self.center)


class RouterRep(FigureRep):

    """
    class to represent a RouterQueue figure
    Parameters
    ----------
    center : PySide.QtCore.QPointF
        figure's center
    size : int
        figure's size in pixel
    queue_size : int
        queue_size to display
    name : str
        figure's name. Used to match with pipeline producer/stager/consumer
        when receive new information dform pipeline
    """

    def __init__(self, center=QPointF(0, 0), size_x=ROUTER_SIZE_X,
                 size_y=ROUTER_SIZE_Y,   queue=0, name=""):
        FigureRep.__init__(self, center=center, size_x=size_x, size_y=size_y,
                           name=name)
        self.queue_size = queue

    def draw(self, qpainter, zoom=1):
        """Draw this figure
        Parameters
        ----------
        qpainter: PySide.QtGui.QPainter
        """
        size_x = self.size_x * zoom
        size_y = self.size_y * zoom
        pensize = 3
        qpainter.setPen(
            QtGui.QPen(QtCore.Qt.black, pensize, QtCore.Qt.SolidLine))
        qpainter.drawEllipse(self.center, size_x / 2, size_y / 2)
        text_pos = QPointF(self.center)
        text_pos.setX(text_pos.x() - size_x / 2 + 10)
        text_pos.setY(text_pos.y() + pensize)
        qpainter.drawText(text_pos, str(self.queue_size))
        qpainter.setPen(
            QtGui.QPen(PipelineDrawer.blue_cta, 3, QtCore.Qt.SolidLine))


class StagerRep(FigureRep):

    """
    class to represent a Stager, Consumer and Producer figure
    Parameters
    ----------
    center : PySide.QtCore.QPointF
        figure's center
    size_x : int
        figure's size x in pixel
    size_y : int
        figure's size y in pixel
    running : bool
        flag that indicates if stager is currently running or waiting for job
    name : str
        figure's name. Used to match with pipeline producer/stager/consumer
        when receive new information dform pipeline
    nb_job_done : int
        Number of jobs already done
    """

    def __init__(self, center=QPointF(0, 0), size_x=0, size_y=0,
                 running=False, fig_type=StepInfo.STAGER, name="", nb_job_done=0):
        FigureRep.__init__(self, center=center, size_x=size_x, size_y=size_y,
                           fig_type=fig_type, name=name)
        self.running = running
        self.nb_job_done = nb_job_done

    def draw(self, qpainter, zoom=1):
        """Draw this figure
        Parameters
        ----------
        qpainter: PySide.QtGui.QPainter
        """
        size_x = self.size_x * zoom
        size_y = self.size_y * zoom
        pensize = 3
        qpainter.setPen(
            QtGui.QPen(PipelineDrawer.blue_cta, pensize, QtCore.Qt.SolidLine))
        text_pos = QPointF(self.center)
        text_pos.setX(text_pos.x() - size_x / 2 + 2)
        text_pos.setY(text_pos.y() + pensize)
        qpainter.drawText(text_pos, str(self.nb_job_done))
        pt = QPointF(self.center)
        pt.setX(5)
        pos = self.name.find("$$thread_number$$")
        if pos != -1:
            name = self.name[0:pos]
        else:
            name = self.name
        qpainter.drawText(pt, name)
        if self.running == True:
            qpainter.setPen(
                QtGui.QPen(PipelineDrawer.mygreen, 3, QtCore.Qt.SolidLine))
        else:
            qpainter.setPen(
                QtGui.QPen(PipelineDrawer.blue_cta, 3, QtCore.Qt.SolidLine))
        x1 = self.center.x() - (size_x / 2)
        y1 = self.center.y() - (size_y / 2)
        qpainter.drawRoundedRect(x1, y1, size_x, size_y, 12.0, 12.0)


class PipelineDrawer(QtGui.QWidget):

    """
    class that displays pipeline workload
    It receives pipeline information thanks to pipechange method
    """
    blue_cta = QColor(1, 39, 118)
    mygreen = QColor(65, 205, 85)

    def __init__(self, statusBar):
        super(PipelineDrawer, self).__init__()
        self.point_size = 1
        self.initUI()
        # self.levels contains all pipeline steps (Producer, Stager, consumer)
        # and RouterQueue in pipeline order.
        self.levels = list()
        # self.config_time stores time when pipeline send its config
        self.config_time = 0.
        self.statusBar = statusBar
        self.zoom = 2

    def initUI(self):
        self.setGeometry(300, 300, 280, 170)
        self.setWindowTitle('PIPELINE')
        self.show()

    def paintEvent(self, e):
        """This event handler is reimplemented in this subclass to receive
         paint events passed in event. A paint event is a request to repaint all
         or part of a widget. It can happen for one of the following reasons:
         repaint() or update() was invoked,the widget was obscured and has
         now been uncovered, or many other reasons. """
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawPipeline(qp)
        qp.end()

    def drawPipeline(self, qp):
        """Called by paintEvent, it draws figures and link between them.
        Parameters
        ----------
        qp : QtGui.QPainter
            Performs low-level painting
        """
        # If self.levels is empty, indeed, it does not make sense to draw
        # something.
        if self.levels is None:
            return
        # define Rep position because they change whan resising main windows

        width = self.size().width()
        height = self.size().height()

        if len(self.levels) != 0:
            total_height = (len(self.levels) - 1) * (
                STAGE_SIZE_Y + ROUTER_SIZE_Y)
            self.zoom = height / total_height
        else:
            self.zoom = 1

        #  center figures on screen
        last_point = QPointF(width / 2, -GAP_Y / 2 * self.zoom)
        for level in self.levels:
            total_x = len(level) * STAGE_SIZE_X * self.zoom + (
                len(level) - 1 * STAGE_GAP_X * self.zoom)
            last_point.setX(width / 2 - total_x / 2)
            last_point.setY(last_point.y() + GAP_Y * self.zoom)
            for figure in level:
                figure.setCenter(QPointF(last_point))
                last_point.setX(last_point.x() + STAGE_GAP_X * self.zoom)
        # Start to paint
        size = self.size()
        lines = list()
        last_level_pt = list()
        for level in self.levels:
            current_level_pt = list()
            for figure in level:
                figure.draw(qp, zoom=self.zoom)
                connexion_pt = QPointF(figure.center.x(), figure.center.y()
                                       - figure.size_y / 2 * self.zoom)
                current_level_pt.append(QPointF(connexion_pt.x(),
                                                connexion_pt.y() + figure.size_y * self.zoom))
                # Link to previous level connexion point(s)
                for point in last_level_pt:
                    lines.append(QLineF(point, connexion_pt))
            # Keep points for next level
            last_level_pt = list(current_level_pt)
        for line in lines:
            qp.setPen(QtGui.QPen(self.blue_cta, 1, QtCore.Qt.SolidLine))
            qp.drawLine(line)

    def pipechange(self, topic, msg):
        """Called by ZmqSub instance when it receives zmq message from pipeline
        Update pipeline state (self.levels) and force to update drawing
        Parameters
        ----------
        topic : str
            Define what has changed in pipeline:
            -GUI_GRAPH -> Full pipeline config ( step, router ...) is send
            -GUI_STAGER_CHANGE   -> A pipeline stager has changed
            -GUI_CONSUMER_CHANGE -> A pipeline consumer has changed
            -GUI_PRODUCER_CHANGE -> The pipeline producer has changed
            -GUI_ROUTER_CHANGE   -> A pipeline Router has changed
        msg : list
            contains informations to update
        """
        # Full pipeline config change
        if topic == b'GUI_GRAPH':  # and not self.receive_levels == True:
            config_time, receiv_levels = msg
            self.build_full_graph(config_time, receiv_levels)
        # Stager or Producer or Consumer state changes
        if self.levels is not None and (topic == b'GUI_STAGER_CHANGE' or
                                        topic == b'GUI_CONSUMER_CHANGE' or
                                        topic == b'GUI_PRODUCER_CHANGE'):
            self.step_change(msg)
        # Router state changes
        if self.levels is not None and topic == b'GUI_ROUTER_CHANGE':
            self.router_change(msg)
        # Force to update drawing
        self.update()

    def build_full_graph(self, config_time, receiv_levels):
        """Build pipeline representation if config_time if diferent that the
        last receive one
        Parameters
        ----------
        config_time: float
            contains pipeline's config's time
        receiv_levels: list of StepInfo describing pipeline contents
        """
        if config_time != self.config_time:
            levels = list()
            # loop overs levels and steps in level
            # Create StagerRep, ConsumerRep, ProducerRep, RouterRep according
            # to StepInfo.fig_type
            for level in receiv_levels:
                steps = list()
                for step in level:
                    if step.type == StepInfo.ROUTER:
                        steps.append(
                            RouterRep(name=step.name, queue=step.queue_size))
                    elif step.type == StepInfo.STAGER:
                        steps.append(
                            StagerRep(
                                running=False, nb_job_done=step.nb_job_done, fig_type=step.type, name=step.name,
                                size_x=STAGE_SIZE_X, size_y=STAGE_SIZE_Y))
                    else:  # PRODUCER AND CONSUMER
                        steps.append(
                            StagerRep(
                                running=False, nb_job_done=step.nb_job_done, fig_type=step.type, name=step.name,
                                size_x=PROD_CONS_SIZE_X, size_y=PROD_CONS_SIZE_Y))
                levels.append(steps)
            # Set self.levels
            self.levels = levels
            self.receive_levels = True
            self.config_time = config_time

    def step_change(self, msg):
        """Find which pipeline step has changed, and update its corresponding
        StagerRep
        Parameters
        ----------
        msg: list
            contains step name, step running flag and step nb_job_done
            receiv_levels: list of StepInfo describing pipeline contents
        """
        name, running, nb_job_done = msg
        for level in self.levels:
            for step in level:
                if step.name == name:
                    step.running = running
                    step.nb_job_done = nb_job_done
                    break

    def router_change(self, msg):
        """Find which pipeline router has changed, and update its corresponding
        RouterRep
        Parameters
        ----------
        msg: list
            contains router name and router queue
        receiv_levels: list of StepInfo describing pipeline contents
        """
        name, queue = msg
        for level in self.levels:
            for step in level:
                if step.name == name:
                    step.queue_size = queue
                    break
