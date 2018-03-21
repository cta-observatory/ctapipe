# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for SST1M/digicam protobuf-fits.fz-files.

Needs protozfits v0.44.3 from github.com/cta-sst-1m/protozfitsreader
"""
import numpy as np
from .eventsource import EventSource
from .containers import SST1MDataContainer

__all__ = ['SST1MEventSource']


class SST1MEventSource(EventSource):

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        from protozfits import SimpleFile
        self.file = SimpleFile(self.input_url)


    def _generator(self):
        self._pixel_sort_ids = None

        for count, event in enumerate(self.file.Events):
            if self._pixel_sort_ids is None:
                self._pixel_sort_ids = np.argsort(
                    event.hiGain.waveforms.pixelsIndices)
                self.n_pixels = len(self._pixel_sort_ids)
            data = SST1MDataContainer()
            data.count = count
            data.sst1m.fill_from_zfile_event(event, self._pixel_sort_ids)
            self.fill_R0Container_from_zfile_event(data.r0, event)
            yield data


    @staticmethod
    def is_compatible(file_path):
        from astropy.io import fits
        try:
            h = fits.open(file_path)[1].header
            ttypes = [
                h[x] for x in h.keys() if 'TTYPE' in x
            ]
        except OSError:
            # not even a fits file
            return False

        is_protobuf_zfits_file = (
            (h['XTENSION'] == 'BINTABLE') and
            (h['EXTNAME'] == 'Events') and
            (h['ZTABLE'] is True) and
            (h['ORIGIN'] == 'CTA') and
            (h['PBFHEAD'] == 'DataModel.CameraEvent')
        )
        is_sst1m_file = 'trigger_input_traces' in ttypes

        return is_protobuf_zfits_file & is_sst1m_file

    def fill_R0CameraContainer_from_zfile_event(self, container, event):
        container.trigger_time = (
            event.local_time_sec * 1E9 + event.local_time_nanosec)
        container.trigger_type = event.event_type

        _samples = (
            event.hiGain.waveforms.samples
        ).reshape(self.n_pixels, -1)
        container.waveform = _samples[self._pixel_sort_ids]

        # Why is this needed ???
        # This is exactly the definition of waveforms.
        container.num_samples = container.waveform.shape[1]

    def fill_R0Container_from_zfile_event(self, container, event):
        container.obs_id = -1  # I do not know what this is.
        container.event_id = event.eventNumber
        container.tels_with_data = [event.telescopeID, ]
        r0_cam_container = container.tel[event.telescopeID]

        self.fill_R0CameraContainer_from_zfile_event(
            r0_cam_container,
            event
        )


