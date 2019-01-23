# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for SST1M/digicam protobuf-fits.fz-files.

Needs protozfits v1.0.2 from github.com/cta-sst-1m/protozfitsreader
"""
import gzip
import numpy as np
from .eventsource import EventSource
from .containers import SST1MDataContainer
from ..instrument import TelescopeDescription

__all__ = ['SST1MEventSource']


def is_fits_in_header(file_path):
    '''quick check if file is a FITS file

    by looking into the first 1024 bytes and searching for the string "FITS"
    typically used in is_compatible
    '''
    # read the first 1kB
    with open(file_path, 'rb') as f:
        marker_bytes = f.read(1024)

    # if file is gzip, read the first 4 bytes with gzip again
    if marker_bytes[0] == 0x1f and marker_bytes[1] == 0x8b:
        with gzip.open(file_path, 'rb') as f:
            marker_bytes = f.read(1024)

    return b'FITS' in marker_bytes


class SST1MEventSource(EventSource):

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        from protozfits import File
        self.file = File(self.input_url)
        # TODO: Correct pixel ordering
        self._tel_desc = TelescopeDescription.from_name(
            optics_name='SST-1M',
            camera_name='DigiCam'
        )

    def _generator(self):
        pixel_sort_ids = None

        for count, event in enumerate(self.file.Events):
            if pixel_sort_ids is None:
                pixel_indices = event.hiGain.waveforms.pixelsIndices
                pixel_sort_ids = np.argsort(pixel_indices)
                self.n_pixels = len(pixel_sort_ids)
            telid = event.telescopeID
            data = SST1MDataContainer()
            data.count = count

            data.inst.subarray.tels[telid] = self._tel_desc

            # Data level Containers
            data.r0.obs_id = -1
            data.r0.event_id = event.eventNumber
            data.r0.tels_with_data = {telid}
            data.r1.obs_id = -1
            data.r1.event_id = event.eventNumber
            data.r1.tels_with_data = {telid}
            data.dl0.obs_id = -1
            data.dl0.event_id = event.eventNumber
            data.dl0.tels_with_data = {telid}

            # R0CameraContainer
            camera_time = event.local_time_sec * 1E9 + event.local_time_nanosec
            samples = event.hiGain.waveforms.samples.reshape(self.n_pixels, -1)
            data.r0.tel[telid].trigger_time = camera_time
            data.r0.tel[telid].trigger_type = event.event_type
            data.r0.tel[telid].waveform = samples[pixel_sort_ids][None, :]
            data.r0.tel[telid].num_samples = samples.shape[-1]

            # SST1MContainer
            data.sst1m.tels_with_data = {telid}

            # SST1MCameraContainer
            digicam_baseline = event.hiGain.waveforms.baselines[pixel_sort_ids]
            gps_time = (event.trig.timeSec * 1E9 + event.trig.timeNanoSec)
            container = data.sst1m.tel[telid]
            container.pixel_flags = event.pixels_flags[pixel_sort_ids]
            container.digicam_baseline = digicam_baseline
            container.local_camera_clock = camera_time
            container.gps_time = gps_time
            container.camera_event_type = event.event_type
            container.array_event_type = event.eventType
            container.trigger_input_traces = event.trigger_input_traces
            container.trigger_output_patch7 = event.trigger_output_patch7
            container.trigger_output_patch19 = event.trigger_output_patch19

            yield data

    @staticmethod
    def is_compatible(file_path):
        if not is_fits_in_header(file_path):
            return False

        from astropy.io import fits
        try:
            h = fits.open(file_path)[1].header
            ttypes = [h[x] for x in h.keys() if 'TTYPE' in x]
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
        is_sst1m_file = 'trigger_input_traces' in ttypes

        return is_protobuf_zfits_file & is_sst1m_file
