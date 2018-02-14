# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read ZFITS data.

This requires the protozfitsreader python library to be installed
"""
from numpy import ndarray

import logging
from ctapipe.core import Map, Field
from ctapipe.io.containers import(
    R0CameraContainer,
    R0Container,
    DataContainer
)
from ctapipe.io.eventsource import EventSource


logger = logging.getLogger(__name__)

__all__ = ['ZFitsFileReader']


class ZFitsFileReader(EventSource):
    def _generator(self):
        from protozfitsreader import ZFile
        for event in ZFile(self.input_url):

            data = SST1M_DataContainer()
            data.r0.event_id = event.event_number
            data.r0.tels_with_data = [event.telescope_id, ]

            for tel_id in data.r0.tels_with_data:
                r0 = data.r0.tel[tel_id]
                r0.fill_from_zfile_event(event)
            yield data

    def is_compatible(self, path):
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

    @property
    def is_stream(self):
        return True


class SST1M_R0CameraContainer(R0CameraContainer):
    """
    Storage of raw data from a single telescope
    """
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

        self.waveform = event.adc_samples


class SST1M_R0Container(R0Container):
    tel = Field(
        Map(SST1M_R0CameraContainer),
        "map of tel_id to SST1M_R0CameraContainer")


class SST1M_DataContainer(DataContainer):
    r0 = Field(SST1M_R0Container(), "Raw Data")
