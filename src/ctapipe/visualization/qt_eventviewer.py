from multiprocessing import JoinableQueue, Process

from ..containers import ArrayEventContainer
from .viewer import EventViewer


class QtEventViewer(EventViewer):
    """
    EventViewer implementation using QT.

    Requires the ctapipe optional dependency ``pyside6``.
    On Linux using wayland, make sure to have qt6 with wayland support,
    e.g. when using conda-forge, also install ``qt6-wayland``.

    Qt requires to have the GUI thread be the main thread, so it is started
    as a subprocess and communication happens through a ``JoinableQueue``.

    Actual GUI implementation is in ``_qt_viewer_impl`` to make this class
    available always but make the qt dependency optional and error with a
    nice message in case the optional dependencies are not installed.
    """

    def __init__(self, subarray, **kwargs):
        try:
            from ._qt_viewer_impl import viewer_main
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "PySide6 is needed for this EventViewer"
            ) from None

        super().__init__(subarray=subarray, **kwargs)

        self.queue = JoinableQueue()
        # don't wait for the GUI process to consume the queue at exit
        self.queue.cancel_join_thread()

        # qt GUIs need to run as main thread -> subprocess
        self.gui_process = Process(
            target=viewer_main,
            args=(
                self.subarray,
                self.queue,
            ),
        )
        self.gui_process.daemon = True
        self.gui_process.start()

    def __call__(self, event: ArrayEventContainer):
        # if the user closed the viewer window, we just continue processing events
        # and this becomes a no-op
        if self.gui_process.is_alive():
            self.queue.join()
            self.queue.put(event)

    def close(self):
        self.queue.close()
        if self.gui_process.is_alive():
            self.gui_process.terminate()
