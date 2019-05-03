"""
Factory for the estimation of the flat field coefficients
"""

from abc import abstractmethod
import numpy as np
from astropy import units as u
from ctapipe.core import Component


from ctapipe.image.extractor import ImageExtractor
from ctapipe.core.traits import Int, Unicode, List

__all__ = [
    'calc_pedestals_from_traces',
    'PedestalCalculator',
    'PedestalIntegrator'
]


def calc_pedestals_from_traces(traces, start_sample, end_sample):
    """A very simple algorithm to calculates pedestals and pedestal
    variances from camera traces by integrating the samples over a
    fixed window for all pixels.  This assumes that the data are
    sample-mode (e.g. cameras that return time traces for each pixel).

    Parameters
    ----------

    traces: array of shape (n_pixels, n_samples)
        time-sampled camera data in a 2D array pixel x sample
    start_sample: int
        index of starting sample over which to integrate
    end_sample: int
        index of ending sample over which to integrate

    Returns
    -------

    two arrays of length n_pix (the first dimension of the input trace
    array). The first array contains the pedestal values, and the
    second is the pedestal variances over the sample window.

    """
    traces = np.asanyarray(traces)  # ensure this is an ndarray
    peds = traces[:, start_sample:end_sample].mean(axis=1)
    pedvars = traces[:, start_sample:end_sample].var(axis=1)
    return peds, pedvars


class PedestalCalculator(Component):
    """
    Parent class for the pedestal calculators.
    Fills the MON.pedestal container on the base of
    pedestal events (preliminary version)
    """

    tel_id = Int(
        0,
        help='id of the telescope to calculate the pedestal values'
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
    charge_median_cut_outliers = List(
        [-3,3],
        help='Interval (number of std) of accepted charge values around camera median value'
    ).tag(config=True)
    charge_std_cut_outliers = List(
        [-3,3],
        help='Interval (number of std) of accepted charge standard deviation around camera median value'
    ).tag(config=True)
    charge_product= Unicode(
        'FixedWindowSum',
        help='Name of the charge extractor to be used'
    ).tag(config=True)

    def __init__(
        self,
        **kwargs
    ):
        """
        Parent class for pedestal calculators.
        Fills the MON.pedestal container.

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
    def calculate_pedestals(self, event):
        """calculate relative gain from event
        Parameters
        ----------
        event: DataContainer

        Returns: PedestalCameraContainer or None

            None is returned if no new pedestal were calculated
            e.g. due to insufficient statistics.
        """


class PedestalIntegrator(PedestalCalculator):

    def __init__(self, **kwargs):
        """Calculates pedestal parameters from pedestal events

        Parameters: see base class PedestalCalculator
        """
        super().__init__(**kwargs)

        self.log.info("Used events statistics : %d", self.sample_size)

        # members to keep state in calculate_relative_gain()
        self.num_events_seen = 0
        self.time_start = None  # trigger time of first event in sample
        self.charge_medians = None  # med. charge in camera per event in sample
        self.charges = None  # charge per event in sample
        self.sample_masked_pixels = None  # pixels tp be masked per event in sample

    def _extract_charge(self, event):
        """
        Extract the charge and the time from a pedestal event

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

    def calculate_pedestals(self, event):
        """
        calculate the pedestal statistical values

        Parameters
        ----------
        event : general event container

        """
        # initialize the np array at each cycle
        waveform = event.r1.tel[self.tel_id].waveform
        container = event.mon.tel[self.tel_id].pedestal

        # real data
        if not event.mcheader.simtel_version:
            trigger_time = event.r1.tel[self.tel_id].trigger_time
            pixel_mask = event.mon.tel[self.tel_id].pixel_status.hardware_mask

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
        self.collect_sample(charge, pixel_mask)

        sample_age = trigger_time - self.time_start

        # check if to create a calibration event
        if (
            sample_age > self.sample_duration
            or self.num_events_seen == self.sample_size
        ):
            pedestal_results = calculate_pedestal_results(
                self,
                self.charges,
                self.sample_masked_pixels,
            )
            time_results = calculate_time_results(
                self.time_start,
                trigger_time,
            )

            result = {
                'n_events': self.num_events_seen,
                **pedestal_results,
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
        self.sample_masked_pixels = np.zeros(shape)

    def collect_sample(self, charge, pixel_mask):

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        masked_pixels = np.zeros(charge.shape, dtype=np.bool)
        masked_pixels[:] = pixel_mask == 1

        good_charge = np.ma.array(charge, mask=masked_pixels)
        charge_median = np.ma.median(good_charge, axis=1)

        self.charges[self.num_events_seen] = charge
        self.sample_masked_pixels[self.num_events_seen] = masked_pixels
        self.charge_medians[self.num_events_seen] = charge_median
        self.num_events_seen += 1


def calculate_time_results(
    time_start,
    trigger_time,
):

    return {
        'sample_time': (trigger_time - time_start) / 2 * u.s,
        'sample_time_range': [time_start, trigger_time] * u.s,
    }


def calculate_pedestal_results(self,
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

    # median over the camera
    median_of_pixel_median = np.ma.median(pixel_median, axis=1)

    # std of median over the camera
    std_of_pixel_median = np.ma.std(pixel_median, axis=1)

    # median of the std over the camera
    median_of_pixel_std = np.ma.median(pixel_std, axis=1)

    # std of the std over camera
    std_of_pixel_std = np.ma.std(pixel_std, axis=1)

    # outliers from standard deviation
    deviation = pixel_std - median_of_pixel_std[:, np.newaxis]
    charge_std_outliers = np.logical_or(deviation < self.charge_std_cut_outliers[0] * std_of_pixel_std[:,np.newaxis],
                                        deviation > self.charge_std_cut_outliers[1] * std_of_pixel_std[:,np.newaxis])

    # outliers from median
    deviation = pixel_median - median_of_pixel_median[:, np.newaxis]
    charge_median_outliers = np.logical_or(deviation < self.charge_median_cut_outliers[0] * std_of_pixel_median[:,np.newaxis],
                                           deviation > self.charge_median_cut_outliers[1] * std_of_pixel_median[:,np.newaxis])

    return {
        'charge_median': np.ma.getdata(pixel_median),
        'charge_mean': np.ma.getdata(pixel_mean),
        'charge_std': np.ma.getdata(pixel_std),
        'charge_std_outliers': np.ma.getdata(charge_std_outliers),
        'charge_median_outliers': np.ma.getdata(charge_median_outliers)

    }


