"""
Factory for the estimation of the flat field coefficients
"""

from abc import abstractmethod

import numpy as np
from astropy import units as u

from ctapipe.containers import DL1CameraContainer
from ctapipe.core import Component
from ctapipe.core.traits import Int, List, Unicode
from ctapipe.image.extractor import ImageExtractor

from .calibrator import _get_invalid_pixels
""
__all__ = ["FlatFieldCalculator", "FlasherFlatFieldCalculator","LSTFlasherFlatFieldCalculator"]


class FlatFieldCalculator(Component):
    """
    Parent class for the flat-field calculators.
    Fills the MonitoringCameraContainer.FlatfieldContainer on the base of a given
    flat-field event sample.
    The sample is defined by a maximal interval of time (sample_duration) or a
    minimal number of events (sample_duration).
    The calculator is supposed to be called in an event loop, extract and collect the
    event charge and fill the PedestalContainer

    Parameters
    ----------
    tel_id : int
          id of the telescope (default 0)
    sample_duration : int
         interval of time (s) used to gather the pedestal statistics
    sample_size : int
         number of pedestal events requested for the statistics
    n_channels : int
         number of waveform channel to be considered
    charge_product : str
        Name of the charge extractor to be used
    config : traitlets.loader.Config
        Configuration specified by config file or cmdline arguments.
        Used to set traitlet values.
        Set to None if no configuration to pass.

    kwargs

    """

    tel_id = Int(
        0, help="id of the telescope to calculate the flat-field coefficients"
    ).tag(config=True)
    sample_duration = Int(60, help="sample duration in seconds").tag(config=True)
    sample_size = Int(10000, help="sample size").tag(config=True)
    n_channels = Int(2, help="number of channels to be treated").tag(config=True)
    charge_product = Unicode(
        "LocalPeakWindowSum", help="Name of the charge extractor to be used"
    ).tag(config=True)

    def __init__(self, subarray, **kwargs):

        """
        Parent class for the flat-field calculators.
        Fills the MonitoringCameraContainer.FlatfieldContainer on the base of a given
        flat-field event sample.
        The sample is defined by a maximal interval of time (sample_duration) or a
        minimal number of events (sample_duration).
        The calculator is supposed to be called in an event loop, extract and collect the
        event charge and fill the PedestalContainer

        Parameters
        ----------
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray
        tel_id : int
              id of the telescope (default 0)
        sample_duration : int
             interval of time (s) used to gather the pedestal statistics
        sample_size : int
             number of pedestal events requested for the statistics
        n_channels : int
             number of waveform channel to be considered
        charge_product : str
            Name of the charge extractor to be used
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.

        kwargs

        """

        super().__init__(**kwargs)
        # load the waveform charge extractor
        self.extractor = ImageExtractor.from_name(
            self.charge_product, parent=self, subarray=subarray
        )

        self.log.info(f"extractor {self.extractor}")

    @abstractmethod
    def calculate_relative_gain(self, event):
        """
        Calculate the flat-field statistics and fill the
        mon.tel[tel_id].flatfield container

        Parameters
        ----------
        event: ctapipe.containers.ArrayEventContainer

        Returns: True if the mon.tel[tel_id].flatfield is updated,
                 False otherwise

        """


class FlasherFlatFieldCalculator(FlatFieldCalculator):
    """Calculates flat-field parameters from flasher data
      based on the best algorithm described by S. Fegan in MST-CAM-TN-0060 (eq. 19)
      Pixels are defined as outliers on the base of a cut on the pixel charge median
      over the full sample distribution and the pixel signal time inside the
      waveform time


    Parameters
    ----------
    charge_cut_outliers : List[2]
        Interval of accepted charge values (fraction with respect to camera median value)
    time_cut_outliers : List[2]
        Interval (in waveform samples) of accepted time values

    """

    charge_cut_outliers = List(
        [-0.3, 0.3],
        help="Interval of accepted charge values (fraction with respect to camera median value)",
    ).tag(config=True)
    time_cut_outliers = List(
        [0, 60], help="Interval (in waveform samples) of accepted time values"
    ).tag(config=True)

    def __init__(self, **kwargs):
        """Calculates flat-field parameters from flasher data
          based on the best algorithm described by S. Fegan in MST-CAM-TN-0060 (eq. 19)
          Pixels are defined as outliers on the base of a cut on the pixel charge median
          over the full sample distribution and the pixel signal time inside the
          waveform time


        Parameters:
        ----------
        charge_cut_outliers : List[2]
            Interval of accepted charge values (fraction with respect to camera median value)
        time_cut_outliers : List[2]
            Interval (in waveform samples) of accepted time values

        """
        super().__init__(**kwargs)

        self.log.info("Used events statistics : %d", self.sample_size)

        # members to keep state in calculate_relative_gain()
        self.n_events_seen = 0
        self.time_start = None  # trigger time of first event in sample
        self.charge_medians = None  # med. charge in camera per event in sample
        self.charges = None  # charge per event in sample
        self.arrival_times = None  # arrival time per event in sample
        self.sample_masked_pixels = None  # masked pixels per event in sample

    def _extract_charge(self, event) -> DL1CameraContainer:
        """
        Extract the charge and the time from a calibration event

        Parameters
        ----------
        event : general event container

        Returns
        -------
        DL1CameraContainer
        """

        waveforms = event.r1.tel[self.tel_id].waveform
        selected_gain_channel = event.r1.tel[self.tel_id].selected_gain_channel
        broken_pixels = _get_invalid_pixels(
            n_pixels=waveforms.shape[-2],
            pixel_status=event.mon.tel[self.tel_id].pixel_status,
            selected_gain_channel=selected_gain_channel,
        )
        # Extract charge and time
        if self.extractor:
            return self.extractor(
                waveforms, self.tel_id, selected_gain_channel, broken_pixels
            )
        else:
            return DL1CameraContainer(image=0, peak_pos=0, is_valid=False)

    def calculate_relative_gain(self, event):
        """
        calculate the flatfield statistical values
        and fill mon.tel[tel_id].flatfield container

        Parameters
        ----------
        event : general event container

        """

        # initialize the np array at each cycle
        waveform = event.r1.tel[self.tel_id].waveform
        container = event.mon.tel[self.tel_id].flatfield

        # re-initialize counter
        if self.n_events_seen == self.sample_size:
            self.n_events_seen = 0

        # real data
        trigger_time = event.trigger.time
        if event.meta["origin"] != "hessio":
            hardware_or_pedestal_mask = np.logical_or(
                event.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels,
                event.mon.tel[self.tel_id].pixel_status.pedestal_failing_pixels,
            )
            pixel_mask = np.logical_or(
                hardware_or_pedestal_mask,
                event.mon.tel[self.tel_id].pixel_status.flatfield_failing_pixels,
            )

        else:  # patches for MC data
            pixel_mask = np.zeros(waveform.shape[1], dtype=bool)

        if self.n_events_seen == 0:
            self.time_start = trigger_time
            self.setup_sample_buffers(waveform, self.sample_size)

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        dl1: DL1CameraContainer = self._extract_charge(event)

        if not dl1.is_valid:
            return False

        self.collect_sample(dl1.image, pixel_mask, dl1.peak_time)

        sample_age = (trigger_time - self.time_start).to_value(u.s)

        # check if to create a calibration event
        if sample_age > self.sample_duration or self.n_events_seen == self.sample_size:
            relative_gain_results = self.calculate_relative_gain_results(
                self.charge_medians, self.charges, self.sample_masked_pixels
            )
            time_results = self.calculate_time_results(
                self.arrival_times,
                self.sample_masked_pixels,
                self.time_start,
                trigger_time,
            )

            result = {
                "n_events": self.n_events_seen,
                **relative_gain_results,
                **time_results,
            }
            for key, value in result.items():
                setattr(container, key, value)

            return True

        else:

            return False

    def setup_sample_buffers(self, waveform, sample_size):
        """Initialize sample buffers"""

        n_channels = waveform.shape[0]
        n_pix = waveform.shape[1]
        shape = (sample_size, n_channels, n_pix)

        self.charge_medians = np.zeros((sample_size, n_channels))
        self.charges = np.zeros(shape)
        self.arrival_times = np.zeros(shape)
        self.sample_masked_pixels = np.zeros(shape)

    def collect_sample(self, charge, pixel_mask, arrival_time):
        """Collect the sample data"""

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)

        good_charge = np.ma.array(charge, mask=pixel_mask)
        charge_median = np.ma.median(good_charge, axis=1)

        self.charges[self.n_events_seen] = charge
        self.arrival_times[self.n_events_seen] = arrival_time
        self.sample_masked_pixels[self.n_events_seen] = pixel_mask
        self.charge_medians[self.n_events_seen] = charge_median
        self.n_events_seen += 1

    def calculate_time_results(
        self, trace_time, masked_pixels_of_sample, time_start, trigger_time
    ):
        """Calculate and return the time results"""
        masked_trace_time = np.ma.array(trace_time, mask=masked_pixels_of_sample)

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
        time_median_outliers = np.logical_or(
            pixel_median < self.time_cut_outliers[0],
            pixel_median > self.time_cut_outliers[1],
        )

        return {
            "sample_time": (trigger_time - time_start).to_value(u.s),
            "sample_time_min": time_start,
            "sample_time_max": trigger_time,
            "time_mean": np.ma.getdata(pixel_mean),
            "time_median": np.ma.getdata(pixel_median),
            "time_std": np.ma.getdata(pixel_std),
            "relative_time_median": np.ma.getdata(relative_median),
            "time_median_outliers": np.ma.getdata(time_median_outliers),
        }

    def calculate_relative_gain_results(
        self, event_median, trace_integral, masked_pixels_of_sample
    ):
        """Calculate and return the sample statistics"""
        masked_trace_integral = np.ma.array(
            trace_integral, mask=masked_pixels_of_sample
        )

        # median over the sample per pixel
        pixel_median = np.ma.median(masked_trace_integral, axis=0)

        # mean over the sample per pixel
        pixel_mean = np.ma.mean(masked_trace_integral, axis=0)

        # std over the sample per pixel
        pixel_std = np.ma.std(masked_trace_integral, axis=0)

        # median of the median over the camera
        median_of_pixel_median = np.ma.median(pixel_median, axis=1)

        # relative gain
        relative_gain_event = masked_trace_integral / event_median[:, :, np.newaxis]

        # outliers from median
        charge_deviation = pixel_median - median_of_pixel_median[:, np.newaxis]

        charge_median_outliers = np.logical_or(
            charge_deviation
            < self.charge_cut_outliers[0] * median_of_pixel_median[:, np.newaxis],
            charge_deviation
            > self.charge_cut_outliers[1] * median_of_pixel_median[:, np.newaxis],
        )

        return {
            "relative_gain_median": np.ma.getdata(
                np.ma.median(relative_gain_event, axis=0)
            ),
            "relative_gain_mean": np.ma.getdata(
                np.ma.mean(relative_gain_event, axis=0)
            ),
            "relative_gain_std": np.ma.getdata(np.ma.std(relative_gain_event, axis=0)),
            "charge_median": np.ma.getdata(pixel_median),
            "charge_mean": np.ma.getdata(pixel_mean),
            "charge_std": np.ma.getdata(pixel_std),
            "charge_median_outliers": np.ma.getdata(charge_median_outliers),
        }

class LSTFlasherFlatFieldCalculator(FlatFieldCalculator):
    """Calculates flat-field parameters from flasher data
       based on the best algorithm described by S. Fegan in MST-CAM-TN-0060 (eq. 19)
       Pixels are defined as outliers on the base of a cut on the pixel charge median
       over the full sample distribution and the pixel signal time inside the
       waveform time


     Parameters:
     ----------
     charge_cut_outliers : List[2]
         Interval of accepted charge values (fraction with respect to camera median value)
     time_cut_outliers : List[2]
         Interval (in waveform samples) of accepted time values

    """

    charge_median_cut_outliers = List(
        [-0.3, 0.3],
        help='Interval of accepted charge values (fraction with respect to camera median value)'
    ).tag(config=True)
    charge_std_cut_outliers = List(
        [-3, 3],
        help='Interval (number of std) of accepted charge standard deviation around camera median value'
    ).tag(config=True)
    time_cut_outliers = List(
        [0, 60], help="Interval (in waveform samples) of accepted time values"
    ).tag(config=True)

    time_sampling_correction_path = Path(
        default_value=None,
        allow_none=True,
        exists=True, directory_ok=False,
        help='Path to time sampling correction file'
    ).tag(config=True)

    sigma_clipping_max_sigma = Int(
        default_value=4,
        help="max_sigma value for the sigma clipping outlier removal",
    ).tag(config=True)

    sigma_clipping_iterations = Int(
        default_value=5,
        help="Number of iterations for the sigma clipping outlier removal",
    ).tag(config=True)


    def __init__(self, subarray, **kwargs):

        """Calculates flat-field parameters from flasher data
           based on the best algorithm described by S. Fegan in MST-CAM-TN-0060 (eq. 19)
           Pixels are defined as outliers on the base of a cut on the pixel charge median
           over the full sample distribution and the pixel signal time inside the
           waveform time


         Parameters:
         ----------
         charge_cut_outliers : List[2]
             Interval of accepted charge values (fraction with respect to camera median value)
         time_cut_outliers : List[2]
             Interval (in waveform samples) of accepted time values

        """
        super().__init__(subarray, **kwargs)

        self.log.info("Used events statistics : %d", self.sample_size)

        # members to keep state in calculate_relative_gain()
        self.num_events_seen = 0
        self.time_start = None  # trigger time of first event in sample
        self.trigger_time = None  # trigger time of present event

        self.charge_medians = None  # med. charge in camera per event in sample
        self.charges = None  # charge per event in sample
        self.arrival_times = None  # arrival time per event in sample
        self.sample_masked_pixels = None  # masked pixels per event in sample

        # declare the charge sampling corrector
        if self.time_sampling_correction_path is not None:
            self.time_sampling_corrector = TimeSamplingCorrection(
                    time_sampling_correction_path=self.time_sampling_correction_path
            )
        else:
            self.time_sampling_corrector = None

        # fix for broken extractor setup in ctapipe baseclass
        self.extractor = ImageExtractor.from_name(
            self.charge_product, parent=self, subarray=subarray
        )

    def _extract_charge(self, event):
        """
        Extract the charge and the time from a calibration event

        Parameters
        ----------
        event : general event container

        """
        # copy the waveform be cause we do not want to change it for the moment
        waveforms = np.copy(event.r1.tel[self.tel_id].waveform)

        # In case of no gain selection the selected gain channels are  [0,0,..][1,1,..]
        no_gain_selection = np.zeros((waveforms.shape[0], waveforms.shape[1]), dtype=np.int64)
        no_gain_selection[1] = 1
        n_pixels = 1855

        # correct the r1 waveform for the sampling time corrections
        if self.time_sampling_corrector:
            waveforms*= (self.time_sampling_corrector.get_corrections(event,self.tel_id)
                         [no_gain_selection, np.arange(n_pixels)])

        # Extract charge and time
        charge = 0
        peak_pos = 0
        if self.extractor:
            broken_pixels = event.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels
            dl1 = self.extractor(waveforms, self.tel_id, no_gain_selection, broken_pixels=broken_pixels)
            charge = dl1.image
            peak_pos = dl1.peak_time

        # shift the time if time shift is already defined
        # (e.g. drs4 waveform time shifts for LST)
        time_shift = event.calibration.tel[self.tel_id].dl1.time_shift
        if time_shift is not None:
                peak_pos -= time_shift

        return charge, peak_pos

    def calculate_relative_gain(self, event):
        """
         calculate the flatfield statistical values
         and fill mon.tel[tel_id].flatfield container

         Parameters
         ----------
         event : general event container

         Returns: True if the mon.tel[tel_id].flatfield is updated, False otherwise

         """

        # initialize the np array at each cycle
        waveform = event.r1.tel[self.tel_id].waveform

        # re-initialize counter
        if self.num_events_seen == self.sample_size:
            self.num_events_seen = 0

        pixel_mask = event.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels

        # time
        self.trigger_time = event.trigger.tel[self.tel_id].time
        
        if self.num_events_seen == 0:
            self.time_start = self.trigger_time
            self.setup_sample_buffers(waveform, self.sample_size)

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        charge, arrival_time = self._extract_charge(event)

        self.collect_sample(charge, pixel_mask, arrival_time)

        sample_age = (self.trigger_time - self.time_start).to_value(u.s)

        # check if to create a calibration event
        if (self.num_events_seen > 0 and
                (sample_age > self.sample_duration or
                self.num_events_seen == self.sample_size)
        ):
            # update the monitoring container
            self.store_results(event)
            return True

        else:

            return False

    def store_results(self, event):
        """
         Store statistical results in monitoring container

         Parameters
         ----------
         event : general event container
        """
        if self.num_events_seen == 0:
            raise ValueError("No flat-field events in statistics, zero results")
       
        container = event.mon.tel[self.tel_id].flatfield

        # mask the part of the array not filled
        self.sample_masked_pixels[self.num_events_seen:] = 1

        relative_gain_results = self.calculate_relative_gain_results(
            self.charge_medians,
            self.charges,
            self.sample_masked_pixels
        )
        time_results = self.calculate_time_results(
            self.arrival_times,
            self.sample_masked_pixels,
            self.time_start,
            self.trigger_time
        )

        result = {
            'n_events': self.num_events_seen,
            **relative_gain_results,
            **time_results,
        }
        for key, value in result.items():
            setattr(container, key, value)

        # update the flatfield mask
        ff_charge_failing_pixels = np.logical_or(container.charge_median_outliers,
                                                 container.charge_std_outliers)
        event.mon.tel[self.tel_id].pixel_status.flatfield_failing_pixels = \
            np.logical_or(ff_charge_failing_pixels, container.time_median_outliers)

    def setup_sample_buffers(self, waveform, sample_size):
        """Initialize sample buffers"""

        n_channels = waveform.shape[0]
        n_pix = waveform.shape[1]
        shape = (sample_size, n_channels, n_pix)

        self.charge_medians = np.zeros((sample_size, n_channels))
        self.charges = np.zeros(shape)
        self.arrival_times = np.zeros(shape)
        self.sample_masked_pixels = np.zeros(shape)

    def collect_sample(self, charge, pixel_mask, arrival_time):
        """Collect the sample data"""

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)

        good_charge = np.ma.array(charge, mask=pixel_mask)
        charge_median = np.ma.median(good_charge, axis=1)

        self.charges[self.num_events_seen] = charge
        self.arrival_times[self.num_events_seen] = arrival_time
        self.sample_masked_pixels[self.num_events_seen] = pixel_mask
        self.charge_medians[self.num_events_seen] = charge_median
        self.num_events_seen += 1

    def calculate_time_results(
        self,
        trace_time,
        masked_pixels_of_sample,
        time_start,
        trigger_time,
    ):
        """Calculate and return the time results """
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
            'sample_time': (time_start + (trigger_time - time_start) / 2).unix * u.s,
            'sample_time_min': time_start.unix*u.s,
            'sample_time_max': trigger_time.unix*u.s,
            'time_mean': pixel_mean.filled(np.nan)*u.ns,
            'time_median': pixel_median.filled(np.nan)*u.ns,
            'time_std': pixel_std.filled(np.nan)*u.ns,
            'relative_time_median': relative_median.filled(np.nan)*u.ns,
            'time_median_outliers': time_median_outliers.filled(True),

        }

    def calculate_relative_gain_results(
        self,
        event_median,
        trace_integral,
        masked_pixels_of_sample,
    ):
        """Calculate and return the sample statistics"""
        masked_trace_integral = np.ma.array(
            trace_integral,
            mask=masked_pixels_of_sample
        )

        # mean and std over the sample per pixel
        max_sigma = self.sigma_clipping_max_sigma
        pixel_mean, pixel_median, pixel_std = sigma_clipped_stats(
            masked_trace_integral,
            sigma=max_sigma,
            maxiters=self.sigma_clipping_iterations,
            cenfunc="mean",
            axis=0,
        )

        # mask pixels without defined statistical values (mainly due to hardware problems)
        pixel_mean = np.ma.array(pixel_mean, mask=np.isnan(pixel_mean))
        pixel_median = np.ma.array(pixel_median, mask=np.isnan(pixel_median))
        pixel_std = np.ma.array(pixel_std, mask=np.isnan(pixel_std))

        unused_values = np.abs(masked_trace_integral - pixel_mean) > (max_sigma * pixel_std)

        # only warn for values discard in the sigma clipping, not those from before
        outliers = unused_values & (~masked_trace_integral.mask)
        check_outlier_mask(outliers, self.log, "flatfield")

        # add outliers identified by sigma clipping for following operations
        masked_trace_integral.mask |= unused_values

        # median of the median over the camera
        median_of_pixel_median = np.ma.median(pixel_median, axis=1)

        # median of the std over the camera
        median_of_pixel_std = np.ma.median(pixel_std, axis=1)

        # std of the std over camera
        std_of_pixel_std = np.ma.std(pixel_std, axis=1)

        # relative gain
        relative_gain_event = masked_trace_integral / event_median[:, :, np.newaxis]

        # outliers from median
        charge_deviation = pixel_median - median_of_pixel_median[:, np.newaxis]

        charge_median_outliers = (
            np.logical_or(charge_deviation < self.charge_median_cut_outliers[0] * median_of_pixel_median[:,np.newaxis],
                          charge_deviation > self.charge_median_cut_outliers[1] * median_of_pixel_median[:,np.newaxis]))

        # outliers from standard deviation
        deviation = pixel_std - median_of_pixel_std[:, np.newaxis]
        charge_std_outliers = (
            np.logical_or(deviation < self.charge_std_cut_outliers[0] * std_of_pixel_std[:, np.newaxis],
                          deviation > self.charge_std_cut_outliers[1] * std_of_pixel_std[:, np.newaxis]))
        
        return {
            'relative_gain_median': np.ma.median(relative_gain_event, axis=0).filled(np.nan),
            'relative_gain_mean': np.ma.mean(relative_gain_event, axis=0).filled(np.nan),
            'relative_gain_std': np.ma.std(relative_gain_event, axis=0).filled(np.nan),
            'charge_median': pixel_median.filled(np.nan),
            'charge_mean': pixel_mean.filled(np.nan),
            'charge_std': pixel_std.filled(np.nan),
            'charge_std_outliers': charge_std_outliers.filled(True),
            'charge_median_outliers': charge_median_outliers.filled(True)
        }