"""
Module to handle the storage of event information extracted with `target_io`
into the `ctapipe.io.containers`.
"""

import numpy as np
from astropy import units as u
from astropy.time import Time
from ctapipe.instrument import TelescopeDescription
from ctapipe.io.unofficial.targetio.camera import Config
from ctapipe.io.unofficial.targetio.containers import TargetioDataContainer
from target_io import TargetIOEventReader as TIOReader, \
    T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES


def get_bp_r_c(cells):
    blockphase = cells % N_BLOCKSAMPLES
    row = (cells // N_BLOCKSAMPLES) % 8
    column = (cells // N_BLOCKSAMPLES) // 8
    return blockphase, row, column


class TargetioExtractor:
    """
    Extract waveform information from `target_io` to store them
    into `ctapipe.io.containers`.

    This Extractor can fill either the R0 or R1 event container, depending on
    the file being read (The header of the file is checked for a flag
    indicating that it has had R1 calibration applied to it.)

    Parameters
    ----------
    url : str
        Filepath to the TargetIO file
    max_events : int
        Maximum number of events to read from the file

    Attributes
    ----------
    tio_reader : target_io.TargetIOEventReader()
        C++ event reader for TargetIO files. Handles the event building into
        an array of (n_pixels, n_samples) in C++, avoiding loops in Python
    n_events : int
        number of events in the fits file
    n_samples : int
        number of samples in the waveform
    r0_samples : ndarray
        three dimensional array to store the R0 level waveform for each pixel
        (n_channels, n_pixels, n_samples)
    r1_samples : ndarray
        three dimensional array to store the R1 level waveform for each pixel
        (n_channels, n_pixels, n_samples)
    samples : ndarray
        pointer to the first index of either r0_samples or r1_samples
        (depending if the file has been R1 calibrated) for passing to
        TargetIO to be filled
    cameraconfig : object
        object conataining camera-specific information (e.g. pixel positions)
        for one of the camera that use the targetio file format
        (CHEC-M, CHEC-S, or single TARGET modules)

    """
    def __init__(self, url, max_events=None):
        """
        Parameters
        ----------
        url : string
            path to the TARGET fits file
        """
        self._event_index = None

        self.url = url
        self.max_events = max_events

        self.event_id = 0
        self.time_tack = None
        self.time_sec = None
        self.time_ns = None

        self.cameraconfig = Config()

        self.tio_reader = TIOReader(self.url,
                                    self.cameraconfig.n_cells,
                                    self.cameraconfig.skip_sample,
                                    self.cameraconfig.skip_end_sample,
                                    self.cameraconfig.skip_event,
                                    self.cameraconfig.skip_end_event)
        self.n_events = self.tio_reader.fNEvents
        first_event_id = self.tio_reader.fFirstEventID
        last_event_id = self.tio_reader.fLastEventID
        self.event_id_list = np.arange(first_event_id, last_event_id)
        self.run_id = self.tio_reader.fRunID
        self.n_pix = self.tio_reader.fNPixels
        self.n_modules = self.tio_reader.fNModules
        self.n_tmpix = self.n_pix // self.n_modules
        self.n_samples = self.tio_reader.fNSamples
        self.n_cells = self.tio_reader.fNCells

        # Setup camera geom
        if self.n_pix == self.n_tmpix:
            self.cameraconfig.switch_to_single_module()
        self.pixel_pos = self.cameraconfig.pixel_pos
        self.optical_foclen = self.cameraconfig.optical_foclen

        self.n_rows = self.cameraconfig.n_rows
        self.n_columns = self.cameraconfig.n_columns
        self.n_blocks = self.cameraconfig.n_blocks
        self.refshape = self.cameraconfig.refshape
        self.refstep = self.cameraconfig.refstep
        self.time_slice = self.cameraconfig.time_slice

        # Init arrays
        self.r0_samples = None
        self.r1_samples = np.zeros((self.n_pix, self.n_samples),
                                   dtype=np.float32)[None, ...]
        self.first_cell_ids = np.zeros(self.n_pix, dtype=np.uint16)

        # Check if file is already r1 (Information obtained from a flag
        # in the file's header)
        is_r1 = self.tio_reader.fR1
        if is_r1:
            self._get_event = self.tio_reader.GetR1Event
            self.samples = self.r1_samples[0]
        else:
            self.r0_samples = np.zeros((self.n_pix, self.n_samples),
                                       dtype=np.uint16)[None, ...]
            self._get_event = self.tio_reader.GetR0Event
            self.samples = self.r0_samples[0]

        self.data = None
        self.init_container()

    @property
    def event_index(self):
        return self._event_index

    @event_index.setter
    def event_index(self, val):
        """
        Setting the event index will cause the event to be saught from
        TargetIO, and the TargetioExtractor containers to point to
        the correct event. The ctapipe event containers are then updated
        with this new event's information.
        """
        self._event_index = val
        self._get_event(self.event_index, self.samples, self.first_cell_ids)
        self.event_id = self.tio_reader.fCurrentEventID
        self.time_tack = self.tio_reader.fCurrentTimeTack
        self.time_sec = self.tio_reader.fCurrentTimeSec
        self.time_ns = self.tio_reader.fCurrentTimeNs
        self.update_container()

    def init_container(self):
        """
        Prepare the ctapipe event container, and fill it with the information
        that does not change with event, including the instrument information.
        """
        url = self.url
        max_events = self.max_events
        chec_tel = 0

        data = TargetioDataContainer()
        data.meta['origin'] = "targetio"

        data.meta['input'] = url
        data.meta['max_events'] = max_events

        # Some targetio specific parameters
        d = np.uint16
        data.meta['n_rows'] = self.n_rows
        data.meta['n_columns'] = self.n_columns
        data.meta['n_blocks'] = self.n_blocks
        data.meta['n_blockphases'] = N_BLOCKSAMPLES
        data.meta['n_cells'] = self.n_cells
        data.meta['n_modules'] = self.n_modules
        data.meta['tm'] = np.arange(self.n_pix, dtype=d) // self.n_tmpix
        data.meta['tmpix'] = np.arange(self.n_pix, dtype=d) % self.n_tmpix

        # Instrument information
        pix_pos = self.pixel_pos * u.m
        foclen = self.optical_foclen * u.m
        teldesc = TelescopeDescription.guess(*pix_pos, foclen)
        data.inst.subarray.tels[chec_tel] = teldesc
        data.inst.pixel_pos[chec_tel] = pix_pos
        data.inst.optical_foclen[chec_tel] = foclen
        data.inst.num_channels[chec_tel] = 1
        data.inst.num_pixels[chec_tel] = self.n_pix

        self.data = data

    def update_container(self):
        """
        Update the ctapipe event containers with the information from the
        current event being pointed to in TargetIO.
        """
        data = self.data
        chec_tel = 0

        event_id = self.event_id
        run_id = self.run_id

        data.r0.run_id = run_id
        data.r0.event_id = event_id
        data.r0.tels_with_data = {chec_tel}
        data.r1.run_id = run_id
        data.r1.event_id = event_id
        data.r1.tels_with_data = {chec_tel}
        data.dl0.run_id = run_id
        data.dl0.event_id = event_id
        data.dl0.tels_with_data = {chec_tel}

        data.trig.tels_with_trigger = [chec_tel]

        data.meta['tack'] = self.time_tack
        data.meta['sec'] = self.time_sec
        data.meta['ns'] = self.time_ns
        data.trig.gps_time = Time(self.time_sec * u.s, self.time_ns * u.ns,
                                  format='unix', scale='utc', precision=9)

        data.count = self.event_index

        data.r0.tel.clear()
        data.r1.tel.clear()
        data.dl0.tel.clear()
        data.dl1.tel.clear()
        data.mc.tel.clear()

        # load the data per telescope/chan
        data.r0.tel[chec_tel].adc_samples = self.r0_samples
        data.r1.tel[chec_tel].pe_samples = self.r1_samples

        # Load the TargetIO specific data per telescope/chan
        data.r0.tel[chec_tel].first_cell_ids = self.first_cell_ids
        bp, r, c = get_bp_r_c(self.first_cell_ids)
        data.r0.tel[chec_tel].blockphase = bp
        data.r0.tel[chec_tel].row = r
        data.r0.tel[chec_tel].column = c
        data.r0.tel[chec_tel].num_samples = self.n_samples

        # Some information that currently exists in the mc container, but is
        # useful for real data (essentially the reference pulse shape,
        # which may be used in charge extraction methods)
        data.mc.tel[chec_tel].reference_pulse_shape = self.refshape
        data.mc.tel[chec_tel].meta['refstep'] = self.refstep
        data.mc.tel[chec_tel].time_slice = self.time_slice

    def read_generator(self):
        """
        Create a generator which loops through the events in the TargetIO file.

        Returns
        -------
        data : generator
            Generator looping over events in the file.

        """
        data = self.data
        n_events = self.n_events
        if self.max_events and self.max_events < self.n_events:
            n_events = self.max_events
        for self.event_index in range(n_events):
            yield data

    def read_event(self, requested_event, use_event_id=False):
        """
        Obtain a particular event from the targetio file.

        Parameters
        ----------
        requested_event : int
            Seek to a paricular event index
        use_event_id : bool
            If True ,'requested_event' now seeks for a particular events id
            instead of index
        """
        index = requested_event
        if use_event_id:
            # Obtaining event id not implemented
            index = self.tio_reader.GetEventIndex(requested_event)
        n_events = self.n_events
        if self.max_events and self.max_events < self.n_events:
            n_events = self.max_events
        if (index >= n_events) | (index < 0):
            raise RuntimeError("Outside event range")
        self.event_index = index
