# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Qt QApplication and QMainWindow for GUI
This requires the pyside python library to be installed
"""

import sys
from ctapipe.flow.gui import Pipelinegui
from ctapipe.flow.gui import LabelQueue
import ctapipe.flow.gui.images_rc
from PyQt4.QtGui import QMainWindow, QPushButton, QApplication, QPalette
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QPixmap
from PyQt4.QtGui import QTableWidget,QTableWidgetItem,QTextEdit
from PyQt4.QtGui import QColor
from ctapipe.flow.gui import ZmqSub



class MainWindow(QMainWindow, object):

    """
    QMainWindow displays pipeline
    Parameters
    ----------
    port : str
        used port to communicate with pipeline
        Note: The firewall must be configure to accept input/output on this port
    """

    def __init__(self, port):
        super(MainWindow, self).__init__()
        self.setupUi(port)
        # Create QtGui.QWidget that displays pipeline workload

    def setupUi(self, port):
        self.setObjectName("MainWindow")
        self.resize(600,500)
        self.centralwidget = QtGui.QWidget(self)
        p = self.centralwidget.palette()
        self.centralwidget.setAutoFillBackground(True)
        p.setColor(self.centralwidget.backgroundRole(), QColor(126, 135, 152))
        self.centralwidget.setPalette(p)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.setCentralWidget(self.centralwidget)

        self.menubar = QtGui.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 808, 25))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.setMenuBar(self.menubar)

        self.statusbar = QtGui.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.actionQuit = QtGui.QAction(self)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.actionReset = QtGui.QAction(self)
        self.actionReset.setObjectName("reset")
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionReset)
        self.menubar.addAction(self.menuFile.menuAction())
        # add other GUI objects

        self.pipeline_gui = Pipelinegui(self.statusbar)
        self.gridLayout.addWidget(self.pipeline_gui, 0, 12, 20, 11 )

        pixmap = QPixmap(':/images/cta-logo-mini.png')
        lbl = QtGui.QLabel()
        lbl.setPixmap(pixmap)
        self.gridLayout.addWidget(lbl, 0, 0, 1, 1)

        p = self.pipeline_gui.palette()
        self.pipeline_gui.setAutoFillBackground(True)
        p.setColor(
            self.pipeline_gui.backgroundRole(),QColor(255,255,255))# QColor(226, 235, 252))
        self.pipeline_gui.setPalette(p)
        self.quitButton = QtGui.QPushButton()  # self.centralwidget)
        self.quitButton.setObjectName("quitButton")
        self.quitButton.setText(QtGui.QApplication.translate
                                ("MainWindow", "Quit", None, QtGui.QApplication.UnicodeUTF8))
        self.gridLayout.addWidget(self.quitButton, 19, 0, 1, 1)

        self.label_queue = LabelQueue(0,4)
        self.gridLayout.addWidget(self.label_queue,5, 0, 1, 1)



        QtCore.QObject.connect(
            self.quitButton, QtCore.SIGNAL("clicked()"), self.stop)
        QtCore.QObject.connect(
            self.actionQuit, QtCore.SIGNAL("triggered()"), self.stop)
        QtCore.QMetaObject.connectSlotsByName(self)


        self.retranslateUi()
        QtCore.QObject.connect(
            self.actionQuit, QtCore.SIGNAL("triggered()"), self.close)
        QtCore.QMetaObject.connectSlotsByName(self)

        # Create ZmqSub for ZMQ comminucation with pipeline
        self.subscribe = ZmqSub(gui_port=port, statusBar=self.statusbar)
        self.subscribe.message.connect(self.pipeline_gui.pipechange)
        self.subscribe.message.connect(self.label_queue.pipechange)

        QtCore.QObject.connect(
            self.actionReset, QtCore.SIGNAL("triggered()"), self.subscribe.reset)
        QtCore.QMetaObject.connectSlotsByName(self)

        # start the processus
        self.subscribe.start()

    def retranslateUi(self):
        self.setWindowTitle(QtGui.QApplication.translate(
            "ctapipe", "ctapipe", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setTitle(QtGui.QApplication.translate(
            "MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.actionQuit.setText(QtGui.QApplication.translate(
            "MainWindow", "Quit", None, QtGui.QApplication.UnicodeUTF8))
        self.actionReset.setText(QtGui.QApplication.translate(
            "MainWindow", "Reset", None, QtGui.QApplication.UnicodeUTF8))


    def stop(self):
        """Method connect (via Qt slot) to exit button
        Stops self.subscribe (for ZMQ communication) processus.
        Close main_windows
        """
        self.subscribe.finish()
        self.subscribe.join()
        self.close()

    def closeEvent(self, event):
            self.subscribe.finish()
            self.subscribe.join()
            event.accept()  # let the window close


class ModuleApplication(QApplication):

    """
    QApplication

    Parameters
    ----------
    QApplication : QApplication
    """

    def __init__(self,  argv, port):
        super(ModuleApplication, self).__init__(argv)
        self.main_windows = MainWindow(port)
        self.main_windows.show()
        self.exec_()
