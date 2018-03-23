# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for NectarCam protobuf-fits.fz-files.

Needs protozfits v0.44.4 from github.com/cta-sst-1m/protozfitsreader
"""

import numpy as np
from .eventsource import EventSource
from .containers import NectarCAMDataContainer

__all__ = ['NectarCAMEventSource']


class NectarCAMEventSource(EventSource):

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        from protozfits import SimpleFile
        self.file = SimpleFile(self.input_url)
        self.header = next(self.file.RunHeader)


    def _generator(self):

        self._pixel_sort_ids = None

        for count, event in enumerate(self.file.Events):
            data = NectarCAMDataContainer()
            data.count = count
            # fill specific NectarCAM data
            data.nectarcam.fill_from_zfile_event(event, self.header.numTraces)
            # fill general R0 data
            self.fill_R0Container_from_zfile_event(data.r0, event)
            yield data


    @staticmethod
    def is_compatible(file_path):
        from astropy.io import fits
        try:
            # The file contains two tables:
            #  1: RunHeader
            #  2: Events <--- this is what we need to look at
            h = fits.open(file_path)[2].header
            ttypes = [
                h[x] for x in h.keys() if 'TTYPE' in x
            ]
        except OSError:
            # not even a fits file
            return False

        except IndexError:
            # A fits file of a different format
            return False

        is_protobuf_zfits_file = (
            (h['XTENSION'] == 'BINTABLE') and
            (h['EXTNAME'] == 'Events') and
            (h['ZTABLE'] is True) and
            (h['ORIGIN'] == 'CTA') and
            (h['PBFHEAD'] == 'DataModel.CameraEvent')
        )

        is_nectarcam_file = 'hiGain_integrals_gains' in ttypes
        return is_protobuf_zfits_file & is_nectarcam_file


    def fill_R0CameraContainer_from_zfile_event(self, container, event):
        container.trigger_time = (
            event.local_time_sec * 1E9 + event.local_time_nanosec)
        container.trigger_type = event.event_type

        container.waveform = np.array([
            (
                event.hiGain.waveforms.samples
            ).reshape(-1, self.header.numTraces),
            (
                event.loGain.waveforms.samples
            ).reshape(-1, self.header.numTraces)
        ])

        container.num_samples = container.waveform.shape[1]

    def fill_R0Container_from_zfile_event(self, container, event):
        container.obs_id = -1
        container.event_id = event.eventNumber

        container.tels_with_data = [self.header.telescopeID, ]
        r0_cam_container = container.tel[self.header.telescopeID]
        self.fill_R0CameraContainer_from_zfile_event(
            r0_cam_container,
            event
        )
