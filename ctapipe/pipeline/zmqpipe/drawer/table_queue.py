from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys


class TableQueue(QTableWidget):
    def __init__(self, *args):
        QTableWidget.__init__(self, 0,3)
        self.data = dict()
        self.queues = dict()
        self.horHeaders = ['Stage','Queue','Done']

    def setmydata(self,data):
        self.data = data
        self.update_table()

    def update_table(self):
        horHeaders = []
        for n, key in enumerate(sorted(self.data.keys())):
            for m, item in enumerate(self.data[key]):
                newitem = QTableWidgetItem(item)
                self.setItem(m, n, newitem)
        self.setRowCount(len(self.data['1Stage']))
        self.setHorizontalHeaderLabels(self.horHeaders)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()


    def pipechange(self, steps):
        """Called by ZmqSub instance when it receives zmq message from pipeline
        Update pipeline state (self.steps) and force to update drawing
        Parameters
        ----------
        topic : str
        """
        """
        if topic == b'GUI_ROUTER_CHANGE':
            name, queue = msg
            self.queues[name]=queue
        """
        self.data.clear()
        self.data['1Stage']= list()
        self.data['2Queue']= list()
        self.data['3Done']= list()
        for step in steps:
            self.data['1Stage'].append(step.name)
            self.data['2Queue'].append(str(step.queue_length))
            self.data['3Done'].append(str(step.nb_job_done))
        self.update_table()
