# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for NectarCam protobuf-fits.fz-files.

Needs protozfits v0.44.5 from github.com/cta-sst-1m/protozfitsreader
"""

import numpy as np
from .eventsource import EventSource
from .containers import NectarCAMDataContainer
from ctapipe.instrument import TelescopeDescription

__all__ = ['NectarCAMEventSource']


class NectarCAMEventSource(EventSource):

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        from protozfits import SimpleFile
        self.file = SimpleFile(self.input_url)
        self.header = next(self.file.RunHeader)
        # TODO: Correct pixel order
        self._tel_desc = TelescopeDescription.from_name(
            optics_name='MST',
            camera_name='NectarCam'
        )

    def _generator(self):
        telid = self.header.telescopeID

        for count, event in enumerate(self.file.Events):
            data = NectarCAMDataContainer()
            data.count = count

            data.inst.subarray.tels[telid] = self._tel_desc

            # R0Container
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
            time = event.local_time_sec * 1E9 + event.local_time_nanosec
            num_traces = self.header.numTraces
            samples_hi = event.hiGain.waveforms.samples.reshape(-1, num_traces)
            samples_lo = event.loGain.waveforms.samples.reshape(-1, num_traces)
            data.r0.tel[telid].trigger_time = time
            data.r0.tel[telid].trigger_type = event.event_type
            data.r0.tel[telid].waveform = np.array([samples_hi, samples_lo])
            data.r0.tel[telid].num_samples = samples_hi.shape[-1]

            # NectarCAMContainer
            data.nectarcam.tels_with_data = {telid}

            # NectarCAMCameraContainer
            integral_hi = event.hiGain.integrals.gains
            integral_lo = event.loGain.integrals.gains
            integrals = np.array([integral_hi, integral_lo])
            data.nectarcam.tel[telid].camera_event_type = event.eventType
            data.nectarcam.tel[telid].integrals = integrals

            yield data

    @staticmethod
    def is_compatible(file_path):
        from astropy.io import fits
        try:
            # The file contains two tables:
            #  1: RunHeader
            #  2: Events <--- this is what we need to look at
            h = fits.open(file_path)[2].header
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
        is_nectarcam_file = 'hiGain_integrals_gains' in ttypes

        return is_protobuf_zfits_file & is_nectarcam_file
