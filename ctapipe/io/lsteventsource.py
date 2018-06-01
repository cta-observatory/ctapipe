# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for LSTCam protobuf-fits.fz-files.

Needs protozfits v1.02.0 from github.com/cta-sst-1m/protozfitsreader
"""

import numpy as np
from .eventsource import EventSource
from .containers import LSTDataContainer

__all__ = ['LSTEventSource']


class LSTEventSource(EventSource):

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        from protozfits import File
        self.file = File(self.input_url)
        self.header = next(self.file.CameraConfig)


    def _generator(self):

        # container for LST data
        data = LSTDataContainer()
        data.meta['input_url'] = self.input_url

        for count, event in enumerate(self.file.Events):


            data.count = count

            # fill specific LST data
            data.lst.fill_from_zfile(self.header,event)

            # fill general R0 data
            self.fill_R0Container_from_zfile(data.r0, event)
            yield data


    @staticmethod
    def is_compatible(file_path):
        from astropy.io import fits
        try:
            # The file contains two tables:
            #  1: CameraConfiguration
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
            (h['PBFHEAD'] == 'R1.CameraEvent')
        )

        is_lst_file = 'lstcam_counters' in ttypes
        return is_protobuf_zfits_file & is_lst_file


    def fill_R0CameraContainer_from_zfile(self, container, event):


        container.num_samples = self.header.num_samples
        container.trigger_time = event.trigger_time_s
        container.trigger_type = event.trigger_type

        container.waveform = np.array(
            (
                event.waveform
            ).reshape(2, self.header.num_pixels, container.num_samples))




    def fill_R0Container_from_zfile(self, container, event):
        container.obs_id = -1
        container.event_id = event.event_id

        container.tels_with_data = [self.header.telescope_id, ]
        r0_camera_container = container.tel[self.header.telescope_id]
        self.fill_R0CameraContainer_from_zfile(
            r0_camera_container,
            event
        )
