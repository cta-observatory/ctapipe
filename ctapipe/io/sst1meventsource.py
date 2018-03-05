# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read ZFITS data.

This requires the protozfitsreader python library to be installed
"""
from .eventsource import EventSource
from .containers import SST1MDataContainer

__all__ = ['SST1MEventSource']


class SST1MEventSource(EventSource):

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        import protozfitsreader

    def _generator(self):
        from protozfitsreader import ZFile
        for count, event in enumerate(ZFile(self.input_url)):
            data = SST1MDataContainer()
            data.count = count
            data.sst1m.fill_from_zfile_event(event)
            fill_R0Container_from_zfile_event(data.r0, event)
            yield data

    @staticmethod
    def is_compatible(file_path):
        from astropy.io import fits
        try:
            h = fits.open(file_path)[1].header
        except OSError:
            # not even a fits file
            return False

        # is it really a proto-zfits-file?
        return (
            (h['XTENSION'] == 'BINTABLE') and
            (h['EXTNAME'] == 'Events') and
            (h['ZTABLE'] is True) and
            (h['ORIGIN'] == 'CTA') and
            (h['PBFHEAD'] == 'DataModel.CameraEvent')
        )


def fill_R0Container_from_zfile_event(container, event):
    container.obs_id = -1  # I do not know what this is.
    container.event_id = event.event_number
    container.tels_with_data = [event.telescope_id, ]
    for tel_id in container.tels_with_data:
        fill_R0CameraContainer_from_zfile_event(
            container.tel[tel_id],
            event
        )


def fill_R0CameraContainer_from_zfile_event(container, event):
    container.trigger_time = event.local_time
    container.trigger_type = event.camera_event_type
    container.waveform = event.adc_samples
    # Why is this needed ???
    container.num_samples = event.adc_samples.shape[1]
