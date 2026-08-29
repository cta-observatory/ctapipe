from abc import ABCMeta, abstractmethod

from ..containers import ArrayEventContainer
from ..core import TelescopeComponent


class EventViewer(TelescopeComponent, metaclass=ABCMeta):
    """
    A component that can display events in some form, e.g. a GUI or Web UI.
    """

    @abstractmethod
    def __call__(self, event: ArrayEventContainer):
        pass

    @abstractmethod
    def close(self):
        pass
