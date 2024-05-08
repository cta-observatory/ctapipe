from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

# import matplotlib after qt so it can detect which bindings are in use
from matplotlib.backends import backend_qtagg  # isort: skip
from matplotlib.figure import Figure  # isort: skip

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


class ViewerMainWindow(QMainWindow):
    new_event_signal = Signal()

    def __init__(self, subarray, queue, **kwargs):
        super().__init__(**kwargs)
        self.subarray = subarray
        self.queue = queue
        self.current_event = None

        layout = QVBoxLayout()

        top = QHBoxLayout()
        self.label = QLabel(self)
        top.addWidget(self.label)

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

        self.next_button = QPushButton("Next Event", parent=self)
        self.next_button.pressed.connect(self.next)
        layout.addWidget(self.next_button)

        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.event_thread = EventLoop(self)
        self.event_thread.start()

        self.new_event_signal.connect(self.update_event)

    def update_event(self):
        event = self.current_event
        if event is None:
            return

        if event.simulation is not None and event.simulation.shower is not None:
            self.label.setText(
                f"event_id: {event.index.event_id}"
                f", E={event.simulation.shower.energy:.3f}"
            )
        else:
            self.label.setText(f"event_id: {event.index.event_id}")

        if event.dl1 is not None:
            tels_with_image = [
                str(tel_id)
                for tel_id, dl1 in event.dl1.tel.items()
                if dl1.image is not None
            ]
            self.tel_selector.clear()
            self.tel_selector.addItems(tels_with_image)
            self.tel_selector.setCurrentIndex(0)

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

    def next(self):
        if self.current_event is not None:
            self.queue.task_done()


class EventLoop(QThread):
    def __init__(self, display):
        super().__init__()
        self.display = display

    def run(self):
        while True:
            event = self.display.queue.get()
            self.display.current_event = event
            self.display.new_event_signal.emit()


def viewer_main(subarray, queue):
    app = QApplication()
    window = ViewerMainWindow(subarray, queue)
    window.show()
    app.exec_()
