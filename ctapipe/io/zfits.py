# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read ZFITS data.

This requires the protozfitsreader python library to be installed
"""
from numpy import ndarray

import logging
from ctapipe.core import Map, Field
from ctapipe.io.containers import(
    InstrumentContainer,
    R0CameraContainer,
    R0Container,
    R1CameraContainer,
    R1Container,
    DataContainer
)
from ctapipe.io.eventfilereader import EventFileReader


logger = logging.getLogger(__name__)

__all__ = ['ZFitsFileReader']


class ZFitsFileReader(EventFileReader):
    def _generator(self):
        from protozfitsreader import ZFile
        for event in ZFile(self.input_url):

            data = SST1M_DataContainer()
            data.r0.event_id = event.event_id
            data.r0.tels_with_data = [event.telescope_id, ]

            for tel_id in data.r0.tels_with_data:
                data.inst.num_channels[tel_id] = event.num_channels
                data.inst.num_pixels[tel_id] = event.n_pixels

                r0 = data.r0.tel[tel_id]
                r0.camera_event_number = event.event_number
                r0.pixel_flags = event.pixel_flags
                r0.local_camera_clock = event.local_time
                r0.gps_time = event.central_event_gps_time
                r0.camera_event_type = event.camera_event_type
                r0.array_event_type = event.array_event_type
                r0.adc_samples = event.adc_samples

                r0.trigger_input_traces = event.trigger_input_traces
                r0.trigger_output_patch7 = event.trigger_output_patch7
                r0.trigger_output_patch19 = event.trigger_output_patch19
                r0.digicam_baseline = event.baseline

                data.inst.num_samples[tel_id] = event.num_samples

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


class SST1M_InstrumentContainer(InstrumentContainer):
    """Storage of header info that does not change with event. This is a
    temporary hack until the Instrument module and database is fully
    implemented.  Eventually static information like this will not be
    part of the data stream, but be loaded and accessed from
    functions.
    """
    cluster_matrix_7 = Field(Map(ndarray), 'map of tel_id of cluster 7 matrix')
    cluster_matrix_19 = Field(
        Map(ndarray),
        'map of tel_id of cluster 19 matrix')
    patch_matrix = Field(Map(ndarray), 'map of tel_id of patch matrix')


class SST1M_R0CameraContainer(R0CameraContainer):
    """
    Storage of raw data from a single telescope
    """
    pixel_flags = Field(ndarray, 'numpy array containing pixel flags')
    num_pixels = Field(int, "number of pixels in camera")
    baseline = Field(ndarray, "number of time samples for telescope")
    digicam_baseline = Field(ndarray, 'Baseline computed by DigiCam')
    standard_deviation = Field(ndarray, "number of time samples for telescope")
    dark_baseline = Field(ndarray, 'dark baseline')
    hv_off_baseline = Field(ndarray, 'HV off baseline')
    camera_event_id = Field(int, 'Camera event number')
    camera_event_number = Field(int, "camera event number")
    local_camera_clock = Field(float, "camera timestamp")
    gps_time = Field(float, "gps timestamp")
    white_rabbit_time = Field(float, "precise white rabbit based timestamp")
    camera_event_type = Field(int, "camera event type")
    array_event_type = Field(int, "array event type")
    trigger_input_traces = Field(ndarray, "trigger patch trace (n_patches)")
    trigger_input_offline = Field(ndarray, "trigger patch trace (n_patches)")
    trigger_output_patch7 = Field(
        ndarray,
        "trigger 7 patch cluster trace (n_clusters)")
    trigger_output_patch19 = Field(
        ndarray,
        "trigger 19 patch cluster trace (n_clusters)")
    trigger_input_7 = Field(ndarray, 'trigger input CLUSTER7')
    trigger_input_19 = Field(ndarray, 'trigger input CLUSTER19')


class SST1M_R0Container(R0Container):
    tel = Field(
        Map(SST1M_R0CameraContainer),
        "map of tel_id to SST1M_R0CameraContainer")


class SST1M_R1CameraContainer(R1CameraContainer):
    adc_samples = Field(
        ndarray,
        "baseline subtracted ADCs, (n_pixels, n_samples)")
    nsb = Field(ndarray, "nsb rate in GHz")
    pde = Field(ndarray, "Photo Detection Efficiency at given NSB")
    gain_drop = Field(ndarray, "gain drop")


class SST1M_R1Container(R1Container):
    tel = Field(
        Map(SST1M_R1CameraContainer),
        "map of tel_id to SST1M_R1CameraContainer")


class SST1M_DataContainer(DataContainer):
    r0 = Field(SST1M_R0Container(), "Raw Data")
    r1 = Field(SST1M_R1Container(), "R1 Calibrated Data")
    inst = Field(
        SST1M_InstrumentContainer(),
        "instrumental information (deprecated)")
