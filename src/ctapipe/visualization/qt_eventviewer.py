from multiprocessing import JoinableQueue, Process

from ..containers import ArrayEventContainer
from .viewer import EventViewer


class QTEventViewer(EventViewer):
    def __init__(self, subarray, **kwargs):
        try:
            from ._qt_viewer_impl import viewer_main
        except ModuleNotFoundError:
            raise ModuleNotFoundError("PySide6 is needed for this EventViewer")

        super().__init__(subarray=subarray, **kwargs)

        self.queue = JoinableQueue()
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
        self.queue.join()
        self.queue.put(event)

    def close(self):
        self.queue.join()
        self.queue.close()
        self.gui_process.terminate()
