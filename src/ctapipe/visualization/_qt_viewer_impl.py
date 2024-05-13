from queue import Empty

import astropy.units as u
import numpy as np
from PySide6 import QtGui
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# import matplotlib after qt so it can detect which bindings are in use
from matplotlib.backends import backend_qtagg  # isort: skip
from matplotlib.figure import Figure  # isort: skip

from ..containers import ArrayEventContainer
from ..coordinates import EastingNorthingFrame, GroundFrame
from .mpl_array import ArrayDisplay
from .mpl_camera import CameraDisplay


class CameraDisplayWidget(QWidget):
    def __init__(self, geometry, **kwargs):
        super().__init__(**kwargs)

        self.geometry = geometry

        self.fig = Figure(layout="constrained")
        self.canvas = backend_qtagg.FigureCanvasQTAgg(self.fig)

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.display = CameraDisplay(geometry, ax=self.ax)
        self.display.add_colorbar()

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)


class TelescopeDataWidget(QWidget):
    def __init__(self, subarray, **kwargs):
        super().__init__(**kwargs)
        self.subarray = subarray
        self.current_event = None

        layout = QVBoxLayout()

        top = QHBoxLayout()
        label = QLabel(text="tel_id: ")
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignCenter)
        top.addWidget(label)
        self.tel_selector = QComboBox(self)
        self.tel_selector.currentTextChanged.connect(self.update_tel_image)
        top.addWidget(self.tel_selector)
        layout.addLayout(top)

        self.camera_displays = []
        self.widget_index = {}
        self.camera_display_stack = QStackedLayout()

        for i, tel in enumerate(self.subarray.telescope_types):
            widget = CameraDisplayWidget(tel.camera.geometry)
            self.camera_displays.append(widget)
            self.camera_display_stack.addWidget(widget)

            for tel_id in subarray.get_tel_ids_for_type(tel):
                self.widget_index[tel_id] = i

        layout.addLayout(self.camera_display_stack)
        self.setLayout(layout)

    def update_tel_image(self, tel_id):
        # tel_selector.clear also calls this, but with an empty tel_id
        if tel_id == "":
            return

        tel_id = int(tel_id)
        index = self.widget_index[tel_id]
        widget = self.camera_displays[index]

        self.camera_display_stack.setCurrentIndex(index)
        widget.display.image = self.current_event.dl1.tel[tel_id].image
        widget.display.axes.figure.canvas.draw()

    def update_event(self, event):
        self.current_event = event

        if event.dl1 is not None:
            tels_with_image = [
                str(tel_id)
                for tel_id, dl1 in event.dl1.tel.items()
                if dl1.image is not None
            ]
            self.tel_selector.clear()
            self.tel_selector.addItems(tels_with_image)
            self.tel_selector.setCurrentIndex(0)


class SubarrayDataWidget(QWidget):
    def __init__(self, subarray, **kwargs):
        super().__init__(**kwargs)
        self.subarray = subarray

        self.fig = Figure(layout="constrained")
        self.canvas = backend_qtagg.FigureCanvasQTAgg(self.fig)

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.display = ArrayDisplay(
            subarray,
            axes=self.ax,
            frame=EastingNorthingFrame(),
        )
        self.display.add_labels()
        self.display.telescopes.set_linewidth(0)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.tel_types = self.display.telescopes.get_array().astype(float)

        (self.true_impact,) = self.ax.plot(
            [],
            [],
            marker="x",
            linestyle="",
            ms=15,
            color="k",
            label="true impact",
        )
        self.display.legend_elements.append(self.true_impact)
        self.ax.legend(handles=self.display.legend_elements)
        self.reco_impacts = {}

    def update_event(self, event):
        trigger_pattern = self.tel_types.copy()
        mask = self.subarray.tel_ids_to_mask(event.trigger.tels_with_trigger)
        trigger_pattern[~mask] = np.nan
        self.display.values = trigger_pattern

        if (sim := event.simulation) is not None and (shower := sim.shower) is not None:
            impact = GroundFrame(shower.core_x, shower.core_y, 0 * u.m)
            impact = impact.transform_to(self.display.frame)
            x = impact.easting.to_value(u.m)
            y = impact.northing.to_value(u.m)
            self.true_impact.set_data(x, y)

        for key, reco in event.dl2.stereo.geometry.items():
            if key not in self.reco_impacts:
                (marker,) = self.ax.plot(
                    [],
                    [],
                    marker="x",
                    linestyle="",
                    ms=15,
                    label=key,
                )
                self.reco_impacts[key] = marker
                self.display.legend_elements.append(marker)
                self.ax.legend(handles=self.display.legend_elements)

            impact = GroundFrame(reco.core_x, reco.core_y, 0 * u.m)
            impact = impact.transform_to(self.display.frame)
            x = impact.easting.to_value(u.m)
            y = impact.northing.to_value(u.m)
            self.reco_impacts[key].set_data(x, y)

        self.canvas.draw()


class ViewerMainWindow(QMainWindow):
    new_event_signal = Signal(ArrayEventContainer)

    def __init__(self, subarray, queue, **kwargs):
        super().__init__(**kwargs)
        self.subarray = subarray
        self.queue = queue
        self.current_event = None
        self.setWindowTitle("ctapipe event display")

        layout = QVBoxLayout()

        top = QHBoxLayout()
        self.label = QLabel(self)
        self.label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignCenter
        )
        top.addWidget(self.label)
        layout.addLayout(top)

        tabs = QTabWidget()
        self.subarray_data = SubarrayDataWidget(subarray)
        tabs.addTab(self.subarray_data, "Subarray Data")
        self.tel_data = TelescopeDataWidget(subarray)
        tabs.addTab(self.tel_data, "Telescope Data")
        layout.addWidget(tabs)

        self.next_button = QPushButton("Next Event", parent=self)
        self.next_button.pressed.connect(self.next)
        layout.addWidget(self.next_button)

        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.event_thread = EventLoop(self)
        self.event_thread.start()

        self.new_event_signal.connect(self.update_event)

        # set window size slightly smaller than available desktop space
        size = QtGui.QGuiApplication.primaryScreen().availableGeometry().size()
        self.resize(0.9 * size)

    def update_event(self, event):
        if event is None:
            return

        self.current_event = event

        label = f"obs_id: {event.index.obs_id}" f", event_id: {event.index.event_id}"
        if event.simulation is not None and event.simulation.shower is not None:
            label += f", E={event.simulation.shower.energy:.3f}"

        self.label.setText(label)
        self.subarray_data.update_event(event)
        self.tel_data.update_event(event)

    def next(self):
        if self.current_event is not None:
            self.queue.task_done()

    def closeEvent(self, event):
        self.event_thread.stop_signal.emit()
        self.event_thread.wait()
        self.next()
        super().closeEvent(event)


class EventLoop(QThread):
    stop_signal = Signal()

    def __init__(self, display):
        super().__init__()
        self.display = display
        self.closed = False
        self.stop_signal.connect(self.close)

    def close(self):
        self.closed = True

    def run(self):
        while not self.closed:
            try:
                event = self.display.queue.get(timeout=0.1)
                self.display.new_event_signal.emit(event)
            except Empty:
                continue
            except ValueError:
                break


def viewer_main(subarray, queue):
    app = QApplication()
    window = ViewerMainWindow(subarray, queue)
    window.show()
    app.exec_()
