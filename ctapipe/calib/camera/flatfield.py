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
    Parent class for the flat field calculators.
    Fills the MON.flatfield container.
    """
    sample_duration = Int(
        60,
        help='sample duration in seconds'
    ).tag(config=True)
    sample_size = Int(
        10000,
        help='sample size'
    ).tag(config=True)
    n_channels = Int(
        2,
        help='number of channels to be treated'
    ).tag(config=True)

    def __init__(
        self,
        config=None,
        tool=None,
        extractor_product=None,
        cleaner_product=None,
        **kwargs
    ):
        """
        Parent class for the flat field calculators.
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
        """calculate relative gain from event
        Parameters
        ----------
        event: DataContainer

        Returns: FlatFieldCameraContainer or None

            None is returned if no new flat field coefficients were calculated
            e.g. due to insufficient statistics.
        """


class FlasherFlatFieldCalculator(FlatFieldCalculator):

    def __init__(self, config=None, tool=None, **kwargs):
        """Calculates flat field coefficients from flasher data

        based on the best algorithm described by S. Fegan in MST-CAM-TN-0060

        Parameters: see base class FlatFieldCalculator
        """
        super().__init__(config=config, tool=tool, **kwargs)

        self.log.info("Used events statistics : %d", self.sample_size)

        # members to keep state in calculate_relative_gain()
        self.num_events_seen = 0
        self.time_start = None  # trigger time of first event in sample
        self.charge_medians = None  # med. charge in camera per event in sample
        self.charges = None  # charge per event in sample
        self.arrival_times = None  # arrival time per event in sample
        self.sample_bad_pixels = None  # bad pixels per event in sample

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
            self.setup_sample_buffers(waveform, self.sample_size)

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        charge, arrival_time = self._extract_charge(event, tel_id)
        self.collect_sample(charge, pixel_status, arrival_time)

        sample_age = trigger_time - self.time_start
        # check if to create a calibration event
        if (
            sample_age > self.sample_duration
            or self.num_events_seen == self.sample_size
        ):
            relative_gain_results = calculate_relative_gain_results(
                self.charge_medians,
                self.charges,
                self.sample_bad_pixels,
            )
            time_results = calculate_time_results(
                self.arrival_times,
                self.sample_bad_pixels,
                self.time_start,
                trigger_time,
            )

            result = {
                'n_events': self.num_events_seen,
                **relative_gain_results,
                **time_results,
            }
            for key, value in result.items():
                setattr(self.container, key, value)

            self.num_events_seen = 0
            return self.container

        else:

            return None

    def setup_sample_buffers(self, waveform, sample_size):
        n_channels = waveform.shape[0]
        n_pix = waveform.shape[1]
        shape = (sample_size, n_channels, n_pix)

        self.charge_medians = np.zeros((sample_size, n_channels))
        self.charges = np.zeros(shape)
        self.arrival_times = np.zeros(shape)
        self.sample_bad_pixels = np.zeros(shape)

    def collect_sample(self, charge, pixel_status, arrival_time):

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        bad_pixels = np.zeros(charge.shape, dtype=np.bool)
        bad_pixels[:] = pixel_status == 0

        good_charge = np.ma.array(charge, mask=bad_pixels)
        charge_median = np.ma.median(good_charge, axis=1)

        self.charges[self.num_events_seen] = charge
        self.arrival_times[self.num_events_seen] = arrival_time
        self.sample_bad_pixels[self.num_events_seen] = bad_pixels
        self.charge_medians[self.num_events_seen] = charge_median
        self.num_events_seen += 1


def calculate_time_results(
    trace_time,
    bad_pixels_of_sample,
    time_start,
    trigger_time,
):
    masked_trace_time = np.ma.array(
        trace_time,
        mask=bad_pixels_of_sample
    )

    # extract the average time over the camera and the events
    camera_time_median = np.ma.median(masked_trace_time)
    camera_time_mean = np.ma.mean(masked_trace_time)
    pixel_time_median = np.ma.median(masked_trace_time, axis=0)
    pixel_time_mean = np.ma.mean(masked_trace_time, axis=0)

    return {
        'time_mean': (trigger_time - time_start) / 2 * u.s,
        'time_range': [time_start, trigger_time] * u.s,
        'relative_time_median': np.ma.getdata(
            pixel_time_median - camera_time_median),
        'relative_time_mean': np.ma.getdata(
            pixel_time_mean - camera_time_mean),
    }


def calculate_relative_gain_results(
    event_median,
    trace_integral,
    bad_pixels_of_sample,
):
    masked_trace_integral = np.ma.array(
        trace_integral,
        mask=bad_pixels_of_sample
    )
    relative_gain_event = np.ma.getdata(
        masked_trace_integral / event_median[:, :, np.newaxis]
    )

    return {
        'relative_gain_median': np.median(relative_gain_event, axis=0),
        'relative_gain_mean': np.mean(relative_gain_event, axis=0),
        'relative_gain_rms': np.std(relative_gain_event, axis=0),
    }


class FlatFieldFactory(Factory):
    """
    Factory to obtain flat-field coefficients
    """
    base = FlatFieldCalculator
    default = 'FlasherFlatFieldCalculator'
    custom_product_help = ('Flat-flield method to use')
