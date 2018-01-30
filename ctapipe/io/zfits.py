# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read ZFITS data.

This requires the protozfitsreader python library to be installed
"""
from os.path import isfile
import numpy as np
from numpy import ndarray

import warnings
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

__all__ = ['ZFitsFileReader', 'ZFile', 'SST1M_Event']


class ZFitsFileReader(EventFileReader):

    def _generator(self):
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


class ZFile:
    def __init__(self, fname):
        from protozfitsreader import rawzfitsreader

        if not isfile(fname):
            raise FileNotFoundError(fname)
        self.fname = fname
        self.eventnumber = 0
        self.is_events_table_open = False

        self.numrows = rawzfitsreader.getNumRows()

    def __next__(self):
        from protozfitsreader import rawzfitsreader
        from protozfitsreader import L0_pb2

        if self.eventnumber < self.numrows:
            if not self.is_events_table_open:
                rawzfitsreader.open(self.fname + ":Events")
                self.is_events_table_open = True
            event = L0_pb2.CameraEvent()
            event.ParseFromString(rawzfitsreader.readEvent())
            self.eventnumber += 1
            return SST1M_Event(event, self.eventnumber)
        else:
            raise StopIteration

    def __iter__(self):
        return self


class SST1M_Event:
    def __init__(self, event, event_id):
        self.event_id = event_id
        self._event = event

        _e = self._event                   # just to make lines shorter
        _w = self._event.hiGain.waveforms  # just to make lines shorter

        self.pixel_ids = to_numpy(_w.pixelsIndices)
        self._sort_ids = np.argsort(self.pixel_ids)
        self.n_pixels = len(self.pixel_ids)
        self._samples = to_numpy(_w.samples).reshape(self.n_pixels, -1)
        self.baseline = self.unsorted_baseline[self._sort_ids]
        self.telescope_id = _e.telescopeID
        self.event_number = _e.eventNumber
        self.central_event_gps_time = self.__calc_central_event_gps_time()
        self.local_time = self.__calc_local_time()
        self.event_number_array = _e.arrayEvtNum
        self.camera_event_type = _e.event_type
        self.array_event_type = _e.eventType
        self.num_gains = _e.num_gains
        self.num_channels = _e.head.numGainChannels
        self.num_samples = self._samples.shape[1]
        self.pixel_flags = to_numpy(_e.pixels_flags)[self._sort_ids]
        self.adc_samples = self._samples[self._sort_ids]
        self.trigger_output_patch7 = _prepare_trigger_output(
            _e.trigger_output_patch7)
        self.trigger_output_patch19 = _prepare_trigger_output(
            _e.trigger_output_patch19)
        self.trigger_input_traces = _prepare_trigger_input(
            _e.trigger_input_traces)

    @property
    def unsorted_baseline(self):
        if not hasattr(self, '__unsorted_baseline'):
            try:
                self.__unsorted_baseline = to_numpy(
                    self._event.hiGain.waveforms.baselines)
            except ValueError:
                warnings.warn((
                    "Could not read `hiGain.waveforms.baselines` for event:{0}"
                    .format(self.event_id)
                    ))
                self.__unsorted_baseline = np.ones(
                    len(self.pixel_ids)
                ) * np.nan
        return self.__unsorted_baseline

    def __calc_central_event_gps_time(self):
        time_second = self._event.trig.timeSec
        time_nanosecond = self._event.trig.timeNanoSec
        return time_second * 1E9 + time_nanosecond

    def __calc_local_time(self):
        time_second = self._event.local_time_sec
        time_nanosecond = self._event.local_time_nanosec
        return time_second * 1E9 + time_nanosecond


def _prepare_trigger_input(_a):
    _a = to_numpy(_a)
    A, B = 3, 192
    cut = 144
    _a = _a.reshape(-1, A)
    _a = _a.reshape(-1, A, B)
    _a = _a[..., :cut]
    _a = _a.reshape(_a.shape[0], -1)
    _a = _a.T
    _a = _a[np.argsort(PATCH_ID_INPUT)]
    return _a


def _prepare_trigger_output(_a):
    _a = to_numpy(_a)
    A, B, C = 3, 18, 8

    _a = np.unpackbits(_a.reshape(-1, A, B, 1), axis=-1)
    _a = _a[..., ::-1]
    _a = _a.reshape(-1, A*B*C).T
    return _a[np.argsort(PATCH_ID_OUTPUT)]


def to_numpy(a):
    '''convert a protobuf "AnyArray" to a numpy array
    '''
    any_array_type_to_npdtype = {
        1: 'i1',
        2: 'u1',
        3: 'i2',
        4: 'u2',
        5: 'i4',
        6: 'u4',
        7: 'i8',
        8: 'u8',
        9: 'f4',
        10: 'f8',
    }

    any_array_type_cannot_convert_exception_text = {
        0: "This any array has no defined type",
        11: """I have no idea if the boolean representation
            of the anyarray is the same as the numpy one"""
    }

    if a.type in any_array_type_to_npdtype:
        return np.frombuffer(
            a.data, any_array_type_to_npdtype[a.type])
    else:
        raise ValueError(
            "Conversion to NumpyArray failed with error:\n%s",
            any_array_type_cannot_convert_exception_text[a.type])


class SST1M_InstrumentContainer(InstrumentContainer):
    """Storage of header info that does not change with event. This is a
    temporary hack until the Instrument module and database is fully
    implemented.  Eventually static information like this will not be
    part of the data stream, but be loaded and accessed from
    functions.
    """

    telescope_ids = Field([], "list of IDs of telescopes used in the run")
    num_pixels = Field(Map(int), "map of tel_id to number of pixels in camera")
    num_channels = Field(Map(int), "map of tel_id to number of channels")
    num_samples = Field(Map(int), "map of tel_id to number of samples")
    geom = Field(Map(None), 'map of tel_if to CameraGeometry')
    cam = Field(Map(None), 'map of tel_id to Camera')
    optics = Field(Map(None), 'map of tel_id to CameraOptics')
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
