# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read ZFITS data.

This requires the protozfitsreader python library to be installed
"""
from numpy import ndarray

from ..core import Map, Field, Container
from .containers import DataContainer
from .eventsource import EventSource

__all__ = ['ZFitsEventSource']


class ZFitsEventSource(EventSource):

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        import protozfitsreader

    def _generator(self):
        from protozfitsreader import ZFile
        for count, event in enumerate(ZFile(self.input_url)):
            data = SST1M_DataContainer()
            data.count = count
            data.r0.event_id = event.event_number
            data.r0.tels_with_data = [event.telescope_id, ]

            for tel_id in data.r0.tels_with_data:
                data.sst1m.tel[tel_id].fill_from_zfile_event(event)
                data.r0.tel[tel_id].waveform = event.adc_samples
            yield data

    @staticmethod
    def is_compatible(path):
        from astropy.io import fits
        try:
            h = fits.open(path)[1].header
        except:
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


class SST1MCameraContainer(Container):
    pixel_flags = Field(ndarray, 'numpy array containing pixel flags')
    digicam_baseline = Field(ndarray, 'Baseline computed by DigiCam')
    local_camera_clock = Field(float, "camera timestamp")
    gps_time = Field(float, "gps timestamp")
    camera_event_type = Field(int, "camera event type")
    array_event_type = Field(int, "array event type")
    trigger_input_traces = Field(ndarray, "trigger patch trace (n_patches)")
    trigger_output_patch7 = Field(
        ndarray,
        "trigger 7 patch cluster trace (n_clusters)")
    trigger_output_patch19 = Field(
        ndarray,
        "trigger 19 patch cluster trace (n_clusters)")

    def fill_from_zfile_event(self, event):
        self.pixel_flags = event.pixel_flags
        self.digicam_baseline = event.baseline
        self.local_camera_clock = event.local_time
        self.gps_time = event.central_event_gps_time
        self.camera_event_type = event.camera_event_type
        self.array_event_type = event.array_event_type
        self.trigger_input_traces = event.trigger_input_traces
        self.trigger_output_patch7 = event.trigger_output_patch7
        self.trigger_output_patch19 = event.trigger_output_patch19


class SST1MContainer(Container):
    tel = Field(
        Map(SST1MCameraContainer),
        "map of tel_id to SST1MCameraContainer")


class SST1M_DataContainer(DataContainer):
    sst1m = Field(SST1MContainer(), "SST1M Specific Information")
