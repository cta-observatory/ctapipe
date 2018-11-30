from ctapipe.io.eventsource import EventSource
from ctapipe.io.hessioeventsource import HESSIOEventSource

__all__ = ['SimTelEventSource']


class SimTelEventSource(EventSource):

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        import eventio
        self.metadata['is_simulation'] = True

    @staticmethod
    def is_compatible(file_path):
        # can be copied here verbatim if HESSIOEventSource should be removed
        return HESSIOEventSource.is_compatible(file_path)

    def _generator(self):
        with eventio.EventIOFile(self.input_url) as file_:
            for o in file_:
                return None
