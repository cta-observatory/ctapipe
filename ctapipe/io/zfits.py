# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read ZFITS data.

This requires the protozfitsreader python library to be installed
"""
from numpy import ndarray

from ..core import Map, Field, Container
from .containers import (
    DataContainer,
    R0Container,
    R0CameraContainer
)
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
            data.fill_from_zfile_event(event, count)
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
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(
        Map(SST1MCameraContainer),
        "map of tel_id to SST1MCameraContainer")

    def fill_from_zfile_event(self, event):
        self.tels_with_data = [event.telescope_id, ]
        for tel_id in self.tels_with_data:
            self.tel[tel_id].fill_from_zfile_event(event)


class SST1M_DataContainer(DataContainer):
    sst1m = Field(SST1MContainer(), "SST1M Specific Information")

    def fill_from_zfile_event(self, event, count):
        self.sst1m.fill_from_zfile_event(event)
        fill_DataContainer_from_zfile_event(self, event, count)


def fill_DataContainer_from_zfile_event(c, event, count):
    """ fill Top-level container for all event information """
    c.r0 = R0Container_from_zfile_event(event)
    c.count = count

    # comment for devs:
    # these fields are also members of DataContainer,
    # but the information to fill them is not (yet) available at this
    # point.

    # r1 = Field(R1Container(), "R1 Calibrated Data")
    # dl0 = Field(DL0Container(), "DL0 Data Volume Reduced Data")
    # dl1 = Field(DL1Container(), "DL1 Calibrated image")
    # dl2 = Field(ReconstructedContainer(), "Reconstructed Shower Information")
    # mc = Field(MCEventContainer(), "Monte-Carlo data")
    # mcheader = Field(MCHeaderContainer(), "Monte-Carlo run header data")
    # trig = Field(CentralTriggerContainer(), "central trigger information")
    # inst = Field(InstrumentContainer(), "instrumental information (deprecated")
    # pointing = Field(Map(TelescopePointingContainer), 'Telescope pointing positions')


def R0Container_from_zfile_event(event):
    c = R0Container()
    c.obs_id = -1  # I do not know what this is.
    c.event_id = event.event_number
    c.tels_with_data = [event.telescope_id, ]
    for tel_id in c.tels_with_data:
        fill_R0CameraContainer_from_zfile_event(
            c.tel[tel_id],
            event
        )
    return c


def fill_R0CameraContainer_from_zfile_event(c, event):
    c.trigger_time = event.local_time
    c.trigger_type = event.camera_event_type
    c.waveform = event.adc_samples
    # Why is this needed ???
    c.num_samples = event.adc_samples.shape[1]
