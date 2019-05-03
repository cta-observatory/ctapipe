"""
Factory for the estimation of the flat field coefficients
"""

from abc import abstractmethod
import numpy as np
from astropy import units as u
from ctapipe.core import Component
from ctapipe.image.extractor import ImageExtractor
from ctapipe.core.traits import Int, Unicode, Float, List


__all__ = [
    'FlatFieldCalculator',
    'FlasherFlatFieldCalculator'
]


class FlatFieldCalculator(Component):
    """
    Parent class for the flat field calculators.
    Fills the MON.flatfield container.
    """

    tel_id = Int(
        0,
        help='id of the telescope to calculate the flat-field coefficients'
    ).tag(config=True)
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
    charge_cut_outliers = List(
        [-0.3,0.3],
        help='Interval of accepted charge values (fraction with respect to camera median value)'
    ).tag(config=True)
    time_cut_outliers = List(
        [10,30],
        help='Interval (in samples) of accepted time values'
    ).tag(config=True)
    charge_product= Unicode(
        'LocalPeakWindowSum',
        help='Name of the charge extractor to be used'
    ).tag(config=True)

    def __init__(
        self,
        **kwargs
    ):
        """
        Parent class for the flat field calculators.
        Fills the flatfield container.

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
        super().__init__(**kwargs)
        # load the waveform charge extractor
        self.extractor = ImageExtractor.from_name(
            self.charge_product,
            config=self.config
        )

        self.log.info(f"extractor {self.extractor}")

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

    def __init__(self, **kwargs):
        """Calculates flat field coefficients from flasher data

        based on the best algorithm described by S. Fegan in MST-CAM-TN-0060

        Parameters: see base class FlatFieldCalculator
        """
        super().__init__(**kwargs)

        self.log.info("Used events statistics : %d", self.sample_size)

        # members to keep state in calculate_relative_gain()
        self.num_events_seen = 0
        self.time_start = None  # trigger time of first event in sample
        self.charge_medians = None  # med. charge in camera per event in sample
        self.charges = None  # charge per event in sample
        self.arrival_times = None  # arrival time per event in sample
        self.sample_masked_pixels = None  # masked pixels per event in sample

    def _extract_charge(self, event):
        """
        Extract the charge and the time from a calibration event

        Parameters
        ----------
        event : general event container

        """

        waveforms = event.r1.tel[self.tel_id].waveform

        # Extract charge and time

        if self.extractor:
            if self.extractor.requires_neighbors():
                g = event.inst.subarray.tel[self.tel_id].camera
                self.extractor.neighbours = g.neighbor_matrix_where

            charge, peak_pos = self.extractor(waveforms)

        return charge, peak_pos

    def calculate_relative_gain(self, event):
        """
        calculate the relative flat field coefficients

        Parameters
        ----------
        event : general event container

        """

        # initialize the np array at each cycle
        waveform = event.r1.tel[self.tel_id].waveform
        container = event.mon.tel[self.tel_id].flatfield

        # real data
        if not event.mcheader.simtel_version:
            trigger_time = event.r1.tel[self.tel_id].trigger_time
            hardware_or_pedestal_mask = np.logical_or(
                event.mon.tel[self.tel_id].pixel_status.hardware_mask,
                event.mon.tel[self.tel_id].pixel_status.pedestal_mask)
            pixel_mask = np.logical_or(
                hardware_or_pedestal_mask,
                event.mon.tel[self.tel_id].pixel_status.flatfield_mask)

        else: # patches for MC data
            if event.trig.tels_with_trigger:
                trigger_time = event.trig.gps_time.unix
            else:
                trigger_time = 0

            pixel_mask = np.zeros(waveform.shape[1], dtype=bool)

        if self.num_events_seen == 0:
            self.time_start = trigger_time
            self.setup_sample_buffers(waveform, self.sample_size)

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        charge, arrival_time = self._extract_charge(event)
        self.collect_sample(charge, pixel_mask, arrival_time)

        sample_age = trigger_time - self.time_start

        # check if to create a calibration event
        if (
            sample_age > self.sample_duration
            or self.num_events_seen == self.sample_size
        ):
            relative_gain_results = calculate_relative_gain_results(
                self,
                self.charge_medians,
                self.charges,
                self.sample_masked_pixels,
            )
            time_results = calculate_time_results(
                self,
                self.arrival_times,
                self.sample_masked_pixels,
                self.time_start,
                trigger_time,
            )

            result = {
                'n_events': self.num_events_seen,
                **relative_gain_results,
                **time_results,
            }
            for key, value in result.items():
                setattr(container, key, value)

            self.num_events_seen = 0
            return True

        else:

            return False

    def setup_sample_buffers(self, waveform, sample_size):
        n_channels = waveform.shape[0]
        n_pix = waveform.shape[1]
        shape = (sample_size, n_channels, n_pix)

        self.charge_medians = np.zeros((sample_size, n_channels))
        self.charges = np.zeros(shape)
        self.arrival_times = np.zeros(shape)
        self.sample_masked_pixels = np.zeros(shape)

    def collect_sample(self, charge, pixel_mask, arrival_time):

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        masked_pixels = np.zeros(charge.shape, dtype=np.bool)
        masked_pixels[:] = pixel_mask == 1

        good_charge = np.ma.array(charge, mask=masked_pixels)
        charge_median = np.ma.median(good_charge, axis=1)

        self.charges[self.num_events_seen] = charge
        self.arrival_times[self.num_events_seen] = arrival_time
        self.sample_masked_pixels[self.num_events_seen] = masked_pixels
        self.charge_medians[self.num_events_seen] = charge_median
        self.num_events_seen += 1


def calculate_time_results(
    self,
    trace_time,
    masked_pixels_of_sample,
    time_start,
    trigger_time,
):
    masked_trace_time = np.ma.array(
        trace_time,
        mask=masked_pixels_of_sample
    )

    # median over the sample per pixel
    pixel_median = np.ma.median(masked_trace_time, axis=0)

    # mean over the sample per pixel
    pixel_mean = np.ma.mean(masked_trace_time, axis=0)

    # std over the sample per pixel
    pixel_std = np.ma.std(masked_trace_time, axis=0)

    # median of the median over the camera
    median_of_pixel_median = np.ma.median(pixel_median, axis=1)


    # time outliers from median
    relative_median = pixel_median - median_of_pixel_median[:, np.newaxis]
    time_median_outliers = np.logical_or(pixel_median < self.time_cut_outliers[0],
                                         pixel_median > self.time_cut_outliers[1])

    return {
        'sample_time': (trigger_time - time_start) / 2 * u.s,
        'sample_time_range': [time_start, trigger_time] * u.s,
        'time_mean': np.ma.getdata(pixel_mean),
        'time_median': np.ma.getdata(pixel_median),
        'time_std': np.ma.getdata(pixel_std),
        'relative_time_median': np.ma.getdata(relative_median),
        'time_median_outliers': np.ma.getdata(time_median_outliers),

    }


def calculate_relative_gain_results(
    self,
    event_median,
    trace_integral,
    masked_pixels_of_sample,
):
    masked_trace_integral = np.ma.array(
        trace_integral,
        mask=masked_pixels_of_sample
    )

    # median over the sample per pixel
    pixel_median = np.ma.median(masked_trace_integral, axis=0)

    # mean over the sample per pixel
    pixel_mean = np.ma.mean(masked_trace_integral, axis=0)

    # std over the sample per pixel
    pixel_std = np.ma.std(masked_trace_integral, axis=0)

    # median of the std over the camera
    median_of_pixel_median = np.ma.median(pixel_median, axis=1)

    # relative gain
    relative_gain_event = masked_trace_integral / event_median[:, :, np.newaxis]

    # outliers from median
    charge_deviation = pixel_median - median_of_pixel_median[:, np.newaxis]

    charge_median_outliers = np.logical_or(charge_deviation < self.charge_cut_outliers[0] * median_of_pixel_median[:,np.newaxis],
                                           charge_deviation > self.charge_cut_outliers[1] * median_of_pixel_median[:,np.newaxis])



    return {
        'relative_gain_median': np.ma.getdata(np.ma.median(relative_gain_event, axis=0)),
        'relative_gain_mean': np.ma.getdata(np.ma.mean(relative_gain_event, axis=0)),
        'relative_gain_std': np.ma.getdata(np.ma.std(relative_gain_event, axis=0)),
        'charge_median': np.ma.getdata(pixel_median),
        'charge_mean': np.ma.getdata(pixel_mean),
        'charge_std': np.ma.getdata(pixel_std),
        'charge_median_outliers': np.ma.getdata(charge_median_outliers),
    }

