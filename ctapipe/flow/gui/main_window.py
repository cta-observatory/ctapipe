# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Qt QApplication and QMainWindow for GUI
This requires the pyside python library to be installed
"""

import sys
from ctapipe.flow.gui import GraphWidget
from ctapipe.flow.gui import InfoLabel
import ctapipe.flow.gui.images_rc
from PyQt4.QtGui import QMainWindow
from PyQt4.QtGui import QPushButton
from PyQt4.QtGui import QApplication
from PyQt4.QtGui import QPalette
from PyQt4.QtGui import QPixmap
from PyQt4.QtGui import QTableWidget
from PyQt4.QtGui import QTableWidgetItem
from PyQt4.QtGui import QTextEdit
from PyQt4.QtGui import QWidget
from PyQt4.QtGui import QColor
from PyQt4.QtGui import QGridLayout
from PyQt4.QtGui import QMenuBar
from PyQt4.QtGui import QMenu
from PyQt4.QtGui import QStatusBar
from PyQt4.QtGui import QAction
from PyQt4.QtGui import QLabel
from PyQt4.QtCore import Qt
from PyQt4.QtCore import QRect
from PyQt4.QtCore import QObject
from PyQt4.QtCore import QMetaObject
from PyQt4.QtCore import SIGNAL
from ctapipe.flow.gui import GuiConnexion



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

    def setupUi(self, port):
        self.setObjectName("MainWindow")
        self.resize(600,600)
        self.centralwidget = QWidget(self)
        p = self.centralwidget.palette()
        self.centralwidget.setAutoFillBackground(True)
        p.setColor(self.centralwidget.backgroundRole(), QColor(126, 135, 152))
        self.centralwidget.setPalette(p)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.setCentralWidget(self.centralwidget)

        self.menubar = QMenuBar(self)
        self.menubar.setGeometry(QRect(0, 0, 808, 25))
        self.menubar.setObjectName("menubar")
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.setMenuBar(self.menubar)

        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.actionQuit = QAction(self)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.actionReset = QAction(self)
        self.actionReset.setObjectName("reset")
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionReset)
        self.menubar.addAction(self.menuFile.menuAction())
        # add other GUI objects

        self.graph_widget = GraphWidget(self.statusbar)
        self.gridLayout.addWidget(self.graph_widget, 1, 11, 10, 10 )

        pixmap = QPixmap(':/images/cta-logo-mini.png')
        lbl = QLabel()
        lbl.setPixmap(pixmap)
        self.gridLayout.addWidget(lbl, 0, 0)

        p = self.graph_widget.palette()
        self.graph_widget.setAutoFillBackground(True)
        p.setColor(
            self.graph_widget.backgroundRole(),QColor(255,255,255))# QColor(226, 235, 252))
        self.graph_widget.setPalette(p)
        self.quitButton = QPushButton()  # self.centralwidget)
        self.quitButton.setObjectName("quitButton")
        self.quitButton.setText(QApplication.translate
                                ("MainWindow", "Quit", None, QApplication.UnicodeUTF8))
        self.gridLayout.addWidget(self.quitButton, 12, 0, 1, 1)

        self.queue_label = InfoLabel(0,4)
        self.queue_label.setAutoFillBackground(True)
        self.gridLayout.addWidget(self.queue_label,1, 0, 1, 5)
        #self.queue_label.setAlignment(PyQt4.Qt.AlignCenter);
        palette = QPalette()
        palette.setColor(self.queue_label.backgroundRole(),Qt.lightGray)
        self.queue_label.setPalette(palette)



        QObject.connect(
            self.quitButton, SIGNAL("clicked()"), self.stop)
        QObject.connect(
            self.actionQuit, SIGNAL("triggered()"), self.stop)
        QMetaObject.connectSlotsByName(self)


        self.retranslateUi()
        QObject.connect(
            self.actionQuit, SIGNAL("triggered()"), self.close)
        QMetaObject.connectSlotsByName(self)

        # Create GuiConnexion for ZMQ comminucation with pipeline
        self.guiconnexion = GuiConnexion(gui_port=port, statusBar=self.statusbar)
        self.guiconnexion.message.connect(self.graph_widget.pipechange)
        self.guiconnexion.message.connect(self.queue_label.pipechange)
        self.guiconnexion.reset_message.connect(self.graph_widget.reset)
        self.guiconnexion.reset_message.connect(self.queue_label.reset)

        QObject.connect(
            self.actionReset, SIGNAL("triggered()"), self.guiconnexion.reset)
        QMetaObject.connectSlotsByName(self)

        # start the processus
        self.guiconnexion.start()

    def retranslateUi(self):
        self.setWindowTitle(QApplication.translate(
            "ctapipe", "ctapipe", None, QApplication.UnicodeUTF8))
        self.menuFile.setTitle(QApplication.translate(
            "MainWindow", "File", None, QApplication.UnicodeUTF8))
        self.actionQuit.setText(QApplication.translate(
            "MainWindow", "Quit", None, QApplication.UnicodeUTF8))
        self.actionReset.setText(QApplication.translate(
            "MainWindow", "Reset", None, QApplication.UnicodeUTF8))


    def stop(self):
        """Method connect (via Qt slot) to exit button
        Stops self.guiconnexion (for ZMQ communication) processus.
        Close main_windows
        """
        self.guiconnexion.finish()
        self.guiconnexion.join()
        self.close()

    def closeEvent(self, event):
            self.guiconnexion.finish()
            self.guiconnexion.join()
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
