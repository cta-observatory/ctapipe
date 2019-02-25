import numpy as np
from astropy import units as u
from astropy.time import Time
from ctapipe.instrument import TelescopeDescription
from ctapipe.io.eventsource import EventSource
from ctapipe.io.containers import TargetIODataContainer

__all__ = ['TargetIOEventSource']


class TargetIOEventSource(EventSource):
    """
    EventSource for the targetio unofficial data format, the data
    format used by cameras containing TARGET modules, such as CHEC for
    the GCT SST.

    Extract waveform information from `target_io` to store them
    into `ctapipe.io.containers`.

    This Extractor can fill either the R0 or R1 event container, depending on
    the file being read (The header of the file is checked for a flag
    indicating that it has had R1 calibration applied to it).

    This reader requires the TARGET libraries. The instructions to install
    these libraries can be found here:
    https://forge.in2p3.fr/projects/gct/wiki/Installing_CHEC_Software

    Attributes
    ----------
    _tio_reader : target_io.TargetIOEventReader()
        C++ event reader for TargetIO files. Handles the event building into
        an array of (n_pixels, n_samples) in C++, avoiding loops in Python
    _n_events : int
        number of events in the fits file
    _n_samples : int
        number of samples in the waveform
    _r0_samples : ndarray
        three dimensional array to store the R0 level waveform for each pixel
        (n_channels, n_pixels, n_samples)
    _r1_samples : ndarray
        three dimensional array to store the R1 level waveform for each pixel
        (n_channels, n_pixels, n_samples)
    _samples : ndarray
        pointer to the first index of either r0_samples or r1_samples
        (depending if the file has been R1 calibrated) for passing to
        TargetIO to be filled
    """

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        try:
            import target_driver
            import target_io
            import target_calib
        except ImportError:
            msg = ("Cannot find TARGET libraries, please follow installation "
                   "instructions from https://forge.in2p3.fr/projects/gct/"
                   "wiki/Installing_CHEC_Software")
            self.log.error(msg)
            raise

        self._waveform_array_reader = target_io.WaveformArrayReader
        self._config_constr = target_calib.CameraConfiguration

        self._data = None
        self._event_index = None
        self._event_id = 0
        self._time_tack = None
        self._time_sec = None
        self._time_ns = None

        self._reader = self._waveform_array_reader(self.input_url, 2, 1)

        self._n_events = self._reader.fNEvents
        self._first_event_id = self._reader.fFirstEventID
        self._last_event_id = self._reader.fLastEventID
        self._obs_id = self._reader.fRunID
        n_modules = self._reader.fNModules
        n_pix = self._reader.fNPixels
        n_samples = self._reader.fNSamples
        self.camera_config = self._config_constr(self._reader.fCameraVersion)
        self._n_cells = self.camera_config.GetNCells()
        m = self.camera_config.GetMapping(n_modules == 1)

        self._optical_foclen = 2.283
        self._pixel_pos = np.vstack([m.GetXPixVector(), m.GetYPixVector()])
        self._refshape = np.zeros(10)  # TODO: Get correct values for CHEC-S
        self._refstep = 0  # TODO: Get correct values for CHEC-S
        self._time_slice = 0  # TODO: Get correct values for CHEC-S
        self._chec_tel = 0

        # Init fields
        self._r0_samples = None
        self._r1_samples = None
        self._first_cell_ids = np.zeros(n_pix, dtype=np.uint16)

        # Check if file is already r1 (Information obtained from a flag
        # in the file's header)
        is_r1 = self._reader.fR1
        if is_r1:
            self._r1_samples = np.zeros(
                (1, n_pix, n_samples),
                dtype=np.float32
            )
            self._get_tio_event = self._reader.GetR1Event
            self._samples = self._r1_samples[0]
        else:
            self._r0_samples = np.zeros(
                (1, n_pix, n_samples),
                dtype=np.uint16
            )
            self._get_tio_event = self._reader.GetR0Event
            self._samples = self._r0_samples[0]

        self._init_container()

    @staticmethod
    def is_compatible(file_path):
        return file_path.endswith('.tio')

    def _init_container(self):
        """
        Prepare the ctapipe event container, and fill it with the information
        that does not change with event, including the instrument information.
        """
        chec_tel = 0

        data = TargetIODataContainer()
        data.meta['origin'] = "targetio"

        data.meta['input'] = self.input_url
        data.meta['max_events'] = self.max_events

        # Instrument information
        pix_pos = self._pixel_pos * u.m
        foclen = self._optical_foclen * u.m
        teldesc = TelescopeDescription.guess(*pix_pos, foclen)
        data.inst.subarray.tels[chec_tel] = teldesc

        self._data = data

    def _update_container(self):
        """
        Update the ctapipe event containers with the information from the
        current event being pointed to in TargetIO.
        """
        data = self._data
        chec_tel = 0

        obs_id = self._obs_id
        event_id = self._event_id
        tels = {self._chec_tel}

        data.r0.obs_id = obs_id
        data.r0.event_id = event_id
        data.r0.tels_with_data = tels
        data.r1.obs_id = obs_id
        data.r1.event_id = event_id
        data.r1.tels_with_data = tels
        data.dl0.obs_id = obs_id
        data.dl0.event_id = event_id
        data.dl0.tels_with_data = tels

        data.trig.tels_with_trigger = [chec_tel]

        data.meta['tack'] = self._time_tack
        data.meta['sec'] = self._time_sec
        data.meta['ns'] = self._time_ns
        data.trig.gps_time = Time(self._time_sec * u.s, self._time_ns * u.ns,
                                  format='unix', scale='utc', precision=9)

        data.count = self._event_index

        data.r0.tel.clear()
        data.r1.tel.clear()
        data.dl0.tel.clear()
        data.dl1.tel.clear()
        data.mc.tel.clear()
        data.targetio.tel.clear()

        # load the data per telescope/chan
        data.r0.tel[chec_tel].waveform = self._r0_samples
        data.r1.tel[chec_tel].waveform = self._r1_samples

        # Load the TargetIO specific data per telescope/chan
        data.targetio.tel[chec_tel].first_cell_ids = self._first_cell_ids
        data.r0.tel[chec_tel].num_samples = self._samples.shape[-1]

        # Some information that currently exists in the mc container, but is
        # useful for real data (essentially the reference pulse shape,
        # which may be used in charge extraction methods)
        data.mc.tel[chec_tel].reference_pulse_shape = self._refshape
        data.mc.tel[chec_tel].meta['refstep'] = self._refstep
        data.mc.tel[chec_tel].time_slice = self._time_slice

    @property
    def _current_event_index(self):
        return self._event_index

    @_current_event_index.setter
    def _current_event_index(self, val):
        """
        Setting the event index will cause the event to be saught from
        TargetIO, and the Containers to point to
        the correct event. The ctapipe event containers are then updated
        with this new event's information.
        """
        self._event_index = val
        self._get_tio_event(val, self._samples, self._first_cell_ids)
        self._event_id = self._reader.fCurrentEventID
        self._time_tack = self._reader.fCurrentTimeTack
        self._time_sec = self._reader.fCurrentTimeSec
        self._time_ns = self._reader.fCurrentTimeNs
        self._update_container()

    def _generator(self):
        for self._current_event_index in range(self._n_events):
            yield self._data
        return

    def __len__(self):
        num = self._n_events
        if self.max_events and self.max_events < num:
            num = self.max_events
        return num

    def _get_event_by_index(self, index):
        self._current_event_index = index
        return self._data

    def _get_event_by_id(self, event_id):
        if ((event_id < self._first_event_id) |
                (event_id > self._last_event_id)):
            raise IndexError(f"Event id {event_id} not found in file")
        index = self._reader.GetEventIndex(event_id)
        return self._get_event_by_index(index)
