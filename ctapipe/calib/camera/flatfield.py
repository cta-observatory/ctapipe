"""
Factory for the estimation of the flat field coefficients
"""
from abc import abstractmethod
import numpy as np
from astropy import units as u
from ctapipe.core import Component, Factory

from ctapipe.image import ChargeExtractorFactory, WaveformCleanerFactory
from ctapipe.core.traits import Int
from ctapipe.io.containers import FlatFieldCameraContainer

__all__ = [
    'FlatFieldCalculator',
    'FlasherFlatFieldCalculator',
    'FlatFieldFactory'
]


class FlatFieldCalculator(Component):
    """
    Parent class for the flat field calculators. Fills the MON.flatfield container.

    """
    max_time_range_s = Int(60, help='Define the maximum time interval per'
           ' coefficient flat-filed calculation').tag(config=True)
    max_events = Int(10000, help='Define the maximum number of events per '
           ' coefficient flat-filed calculation').tag(config=True)
    n_channels = Int(2, help='Define the number of channels to be '
                    'treated ').tag(config=True)

    def __init__(self, config=None, tool=None, extractor_product=None, cleaner_product=None, **kwargs):
        """
        Parent class for the flat field calculators. Fills the MON.flatfield container.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        extractor_product : str
            The ChargeExtractor to use.
        cleaner_product : str
            The WaveformCleaner to use.
        kwargs

        """
        super().__init__(config=config, parent=tool, **kwargs)

        # initialize the output
        self.container = FlatFieldCameraContainer()

        # load the waveform charge extractor and cleaner
        kwargs_ = dict()
        if extractor_product:
            kwargs_['product'] = extractor_product
        self.extractor = ChargeExtractorFactory.produce(
            config=config,
            tool=tool,
            **kwargs_
        )
        self.log.info(f"extractor {self.extractor}")

        kwargs_ = dict()
        if cleaner_product:
            kwargs_['product'] = cleaner_product
        self.cleaner = WaveformCleanerFactory.produce(
            config=config,
            tool=tool,
            **kwargs_
        )
        self.log.info(f"cleaner {self.cleaner}")

    @abstractmethod
    def calculate_relative_gain(self, event):
        """
        Parameters
        ----------
        event

        """


class FlasherFlatFieldCalculator(FlatFieldCalculator):
    """
    Class for calculating flat field coefficients witht the
    flasher data. Fills the MON.flatfield container.
    """

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Parent class for the flat-field calculators.
        Fills the MON.flatfield container.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs

        """
        super().__init__(config=config, tool=tool, **kwargs)

        self.log.info("Used events statistics : %d", self.max_events)
        self.num_events_seen = 0

        # members to keep state in calculate_relative_gain()
        self.time_start = None  # trigger time of first event in sample
        self.event_median = None  # med. charge in camera per event in sample
        self.trace_integral = None  # charge per event in sample
        self.trace_time = None  # arrival time per event in sample
        self.bad_pixels_of_sample = None  # bad pixels per event in sample

    def _extract_charge(self, event, tel_id):
        """
        Extract the charge and the time from a calibration event

        Parameters
        ----------
        event : general event container

        tel_id : telescope id
        """

        waveforms = event.r0.tel[tel_id].waveform

        # Clean waveforms
        if self.cleaner:
            cleaned = self.cleaner.apply(waveforms)
        # do nothing
        else:
            cleaned = waveforms

        # Extract charge and time
        if self.extractor:
            if self.extractor.requires_neighbours():
                g = event.inst.subarray.tel[tel_id].camera
                self.extractor.neighbours = g.neighbor_matrix_where

            charge, peak_pos, window = self.extractor.extract_charge(cleaned)

        # sum all the samples
        else:
            charge = cleaned.sum(axis=2)
            peak_pos = np.argmax(cleaned, axis=2)

        return charge, peak_pos

    def calculate_relative_gain(self, event, tel_id):
        """
        calculate the relative flat field coefficients

        Parameters
        ----------
        event : general event container

        tel_id : telescope id for which we calculate the gain
        """

        # initialize the np array at each cycle
        waveform = event.r0.tel[tel_id].waveform
        trigger_time = event.r0.tel[tel_id].trigger_time
        pixel_status = event.r0.tel[tel_id].pixel_status

        if self.num_events_seen == 0:
            self.time_start = trigger_time

            if waveform.shape[0] < self.n_channels:
                self.n_channels = waveform.shape[0]

            self.event_median = np.zeros((self.max_events, self.n_channels))

            n_pix = waveform.shape[1]
            shape = (self.max_events, self.n_channels, n_pix)
            self.trace_integral = np.zeros(shape)
            self.trace_time = np.zeros(shape)
            self.bad_pixels_of_sample = np.zeros(shape)

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        integral, peakpos = self._extract_charge(event, tel_id)

        # remember the charge
        self.trace_integral[self.num_events_seen] = integral

        # remember the time
        self.trace_time[self.num_events_seen] = peakpos

        # keep the mask of not working pixels (to be improved)
        bad_pixels = np.array([pixel_status == 0, pixel_status == 0])
        self.bad_pixels_of_sample[self.num_events_seen, :] = bad_pixels

        # extract the median on all the camera per event: <x>(i)
        # (for not masked pixels)
        masked_integral = np.ma.array(integral, mask=bad_pixels)
        self.event_median[self.num_events_seen, :] = np.ma.median(
            masked_integral, axis=1)

        self.num_events_seen += 1

        sample_age = trigger_time - self.time_start
        # check if to create a calibration event
        if (
            sample_age > self.max_time_range_s
            or self.num_events_seen == self.max_events
        ):

            # consider only not masked data
            masked_trace_integral = np.ma.array(
                self.trace_integral,
                mask=self.bad_pixels_of_sample
            )
            masked_trace_time = np.ma.array(
                self.trace_time,
                mask=self.bad_pixels_of_sample
            )

            # extract for each pixel and each event : x(i,j)/<x>(i) = g(i,j)
            masked_relative_gain_event = (
                masked_trace_integral / self.event_median[:, :, np.newaxis])
            relative_gain_event = np.ma.getdata(masked_relative_gain_event)

            # extract the median, mean and std over all the events <g>j and
            # fill the container and return it
            self.container.time_mean = (
                (trigger_time - self.time_start) / 2 * u.s)
            self.container.time_range = [self.time_start, trigger_time] * u.s
            self.container.n_events = self.num_events_seen
            self.container.relative_gain_median = np.median(
                relative_gain_event, axis=0)
            self.container.relative_gain_mean = np.mean(
                relative_gain_event, axis=0)
            self.container.relative_gain_rms = np.std(
                relative_gain_event, axis=0)

            # extract the average time over the camera and the events
            camera_time_median = np.ma.median(masked_trace_time)
            camera_time_mean = np.ma.mean(masked_trace_time)
            pixel_time_median = np.ma.median(masked_trace_time, axis=0)
            pixel_time_mean = np.ma.mean(masked_trace_time, axis=0)

            # fill the container
            self.container.relative_time_median = np.ma.getdata(
                pixel_time_median - camera_time_median
            )
            self.container.relative_time_mean = np.ma.getdata(
                pixel_time_mean - camera_time_mean
            )

            # re-initialize the event count
            self.num_events_seen = 0

            return self.container

        else:

            return None


class FlatFieldFactory(Factory):
    """
    Factory to obtain flat-field coefficients
    """
    base = FlatFieldCalculator
    default = 'FlasherFlatFieldCalculator'
    custom_product_help = ('Flat-flield method to use')
