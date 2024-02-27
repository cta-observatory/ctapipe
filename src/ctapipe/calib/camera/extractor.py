"""
Algorithms for the camera calibration extraction.
"""

from abc import abstractmethod

import numpy as np
from astropy import units as u

from ctapipe.containers import DL1CameraContainer
from ctapipe.core import Component
from ctapipe.core.traits import Int, List, TelescopeParameter, IntTelescopeParameter
from ctapipe.image.extractor import ImageExtractor

__all__ = ["CalibrationExtractor", "FlatFieldExtractor", "PedestalExtractor"]


class CalibrationExtractor(TelescopeComponent):
    """
    Base component for camera calibration extractors.
    Fills the MonitoringCameraContainer on the base of a given sample of interleaved
    calibration events.
    The sample is defined by a minimal number of calibration events (sample_size) and
    the statistics are calculated at a given frequency (update_frequency).
    The extractors are supposed to be called in an event loop, extract (image_extractor_type)
    and collect the event charge and arrival time. At a pre-defined iteration frequency, the
    sample statistics are calculated and the corresponding MonitoringCameraContainer is filled.

    The pixels are set as outliers on the base of a cut on the pixel charge median over the full
    sample distribution (charge_median_cut_outliers) and the pixel charge standard deviation over
    the full sample distribution with respect to the camera median values (charge_std_cut_outliers),
    as well as the pixel signal/arrival time inside the waveform time (time_cut_outliers). Sigma
    clipping

    Attributes
    ----------
    sample_size : int
        Minimal number of calibration events requested for the statistics
    update_frequency : int
        Calculation frequency of the statistics
    image_extractor_type : str
        Name of the image extractor to be used
    charge_median_cut_outliers : List[2]
        Interval of accepted charge values (fraction with respect to camera median value)
    charge_std_cut_outliers : List[2]
        Interval (number of std) of accepted charge standard deviation around camera median value
    time_cut_outliers : List[2]
        Interval (in waveform samples) of accepted time values
    sigma_clipping_max_sigma : int
        Maximal value for the sigma clipping outlier removal
    sigma_clipping_iterations : int
        Number of iterations for the sigma clipping outlier removal
    """

    sample_size = IntTelescopeParameter(2500, help="sample size").tag(config=True)
    update_frequency = IntTelescopeParameter(2500, help="update frequency").tag(
        config=True
    )
    image_extractor_type = TelescopeParameter(
        trait=ComponentName(ImageExtractor, default_value="LocalPeakWindowSum"),
        default_value="LocalPeakWindowSum",
        help="Name of the ImageExtractor subclass to be used.",
    ).tag(config=True)
    charge_median_cut_outliers = List(
        [-0.3, 0.3],
        help="Interval of accepted charge values (fraction with respect to camera median value)",
    ).tag(config=True)
    charge_std_cut_outliers = List(
        [-3, 3],
        help="Interval (number of std) of accepted charge standard deviation around camera median value",
    ).tag(config=True)
    time_cut_outliers = List(
        [0, 60], help="Interval (in waveform samples) of accepted time values"
    ).tag(config=True)
    sigma_clipping_max_sigma = Int(
        default_value=4,
        help="Maximal value for the sigma clipping outlier removal",
    ).tag(config=True)
    sigma_clipping_iterations = Int(
        default_value=5,
        help="Number of iterations for the sigma clipping outlier removal",
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        """
        Parameters
        ----------
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        kwargs
        """
        super().__init__(config=config, parent=parent, subarray=subarray, **kwargs)

        # set up the buffer
        self.n_events_in_buffer = 0  # number of events in sample
        self.trigger_time = []  # trigger time of event in sample
        self.charges = []  # charge per event in sample
        self.arrival_times = []  # arrival time per event in sample
        self.broken_pixels = []  # masked pixels per event in sample

        # load the waveform charge extractor
        self.extractor = ImageExtractor.from_name(
            self.image_extractor_type, parent=self, subarray=subarray
        )

    def __call__(self, event, tel_id) -> Bool:
        """
        Extract the charge and the time from a calibration event.
        Process sample if enough statistics is reached and
        clean up the buffer.

        Parameters
        ----------
        event : general event container
        tel_id : int

        Returns: True if the MonitoringCameraContainer is updated,
                 False otherwise
        """

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        dl1: DL1CameraContainer = self._extract_charge_and_time(event, tel_id)

        container_updated = False
        if dl1.is_valid:
            # append valid dl1 events
            self.n_events_in_buffer += 1
            self.trigger_times.append(event.trigger.tel[tel_id].time)
            self.charges.append(dl1.image)
            self.arrival_times.append(dl1.peak_time)
            self.broken_pixels.append(broken_pixels)
            # check if to create a calibration event
            if self.n_events_in_buffer == self.sample_size:
                self._process(event, tel_id)
                self._clean_buffer()
                container_updated = True

        return container_updated

    def _extract_charge_and_time(self, event, tel_id) -> DL1CameraContainer:
        """
        Extract the charge and the arrival time from a calibration event

        Parameters
        ----------
        event : general event container
        tel_id : int

        Returns
        -------
        DL1CameraContainer:
            extracted images and validity flags
        """

        # copy the waveform be cause we do not want to change it for the moment
        waveforms = np.copy(event.r1.tel[tel_id].waveform)
        broken_pixels = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels

        # TODO: Move this outside this call function in the init from subarray description
        n_channels = waveform.shape[0]
        n_pixels = waveform.shape[1]
        no_gain_selection = np.zeros((n_channels, n_pixels), dtype=np.int64)
        if n_channels == 2:
            no_gain_selection[1] = 1

        dl1 = DL1CameraContainer(image=0, peak_pos=0, is_valid=False)
        # Extract charge and arrival time
        if self.extractor:
            dl1 = self.extractor(waveforms, tel_id, no_gain_selection, broken_pixels)

        return dl1

    def _calculate_charge_stats(
        self,
        charges,
        broken_pixels,
        trigger_times,
    ):
        """
        Calculate the sample statistics regarding the charge of each camera pixel.

        Parameters
        ----------
        charges: list
        broken_pixels: list
        trigger_times: list

        Returns
        -------
        charge_stats : dict
        The sample statistics of the charge of each camera pixel.
        """

        # ensure numpy array
        masked_charges = np.ma.array(charges, mask=broken_pixels)

        # mean and std over the sample per pixel
        max_sigma = self.sigma_clipping_max_sigma
        pixel_mean, pixel_median, pixel_std = sigma_clipped_stats(
            masked_charges,
            sigma=max_sigma,
            maxiters=self.sigma_clipping_iterations,
            cenfunc="mean",
            axis=0,
        )

        # mask pixels without defined statistical values (mainly due to hardware problems)
        pixel_mean = np.ma.array(pixel_mean, mask=np.isnan(pixel_mean))
        pixel_median = np.ma.array(pixel_median, mask=np.isnan(pixel_median))
        pixel_std = np.ma.array(pixel_std, mask=np.isnan(pixel_std))

        unused_values = np.abs(masked_charges - pixel_mean) > (max_sigma * pixel_std)

        # only warn for values discard in the sigma clipping, not those from before
        outliers = unused_values & (~masked_charges.mask)

        # add outliers identified by sigma clipping for following operations
        masked_charges.mask |= unused_values

        # median of the median over the camera
        median_of_pixel_median = np.ma.median(pixel_median, axis=1)

        # median of the std over the camera
        median_of_pixel_std = np.ma.median(pixel_std, axis=1)

        # std of the std over camera
        std_of_pixel_std = np.ma.std(pixel_std, axis=1)

        # outliers from median
        charge_deviation = pixel_median - median_of_pixel_median[:, np.newaxis]

        charge_median_outliers = np.logical_or(
            charge_deviation
            < self.charge_median_cut_outliers[0]
            * median_of_pixel_median[:, np.newaxis],
            charge_deviation
            > self.charge_median_cut_outliers[1]
            * median_of_pixel_median[:, np.newaxis],
        )

        # outliers from standard deviation
        deviation = pixel_std - median_of_pixel_std[:, np.newaxis]
        charge_std_outliers = np.logical_or(
            deviation
            < self.charge_std_cut_outliers[0] * std_of_pixel_std[:, np.newaxis],
            deviation
            > self.charge_std_cut_outliers[1] * std_of_pixel_std[:, np.newaxis],
        )

        return {
            "sample_time": (
                trigger_times[0] + (trigger_times[-1] - trigger_times[0]) / 2
            ).unix
            * u.s,
            "sample_time_min": trigger_times[0].unix * u.s,
            "sample_time_max": trigger_times[-1].unix * u.s,
            "charge_median": pixel_median.filled(np.nan),
            "charge_mean": pixel_mean.filled(np.nan),
            "charge_std": pixel_std.filled(np.nan),
            "charge_std_outliers": charge_std_outliers.filled(True),
            "charge_median_outliers": charge_median_outliers.filled(True),
        }

    def _calculate_arrivaltime_stats(
        self,
        arrival_times,
        broken_pixels,
    ):
        """
        Calculate the sample statistics regarding the arrival time of each camera pixel.

        Parameters
        ----------
        arrival_times: list
        broken_pixels: list

        Returns
        -------
        arrivaltime_stats : dict
        The sample statistics of the arrival time of each camera pixel.
        """

        # ensure numpy array
        masked_arrival_times = np.ma.array(arrival_times, mask=broken_pixels)

        # median over the sample per pixel
        pixel_median = np.ma.median(masked_arrival_times, axis=0)

        # mean over the sample per pixel
        pixel_mean = np.ma.mean(masked_arrival_times, axis=0)

        # std over the sample per pixel
        pixel_std = np.ma.std(masked_arrival_times, axis=0)

        # median of the median over the camera
        median_of_pixel_median = np.ma.median(pixel_median, axis=1)

        # time outliers from median
        relative_median = pixel_median - median_of_pixel_median[:, np.newaxis]
        time_median_outliers = np.logical_or(
            pixel_median < self.time_cut_outliers[0],
            pixel_median > self.time_cut_outliers[1],
        )

        return {
            "time_mean": pixel_mean.filled(np.nan) * u.ns,
            "time_median": pixel_median.filled(np.nan) * u.ns,
            "time_std": pixel_std.filled(np.nan) * u.ns,
            "relative_time_median": relative_median.filled(np.nan) * u.ns,
            "time_median_outliers": time_median_outliers.filled(True),
        }

    def _clean_buffer(self):
        """
        Clean up the buffer by removing the events that will not participate
        in the caluclation of the next sample.
        """
        self.n_events_in_buffer -= self.update_frequency
        self.trigger_time = self.trigger_time[self.update_frequency :]
        self.charges = self.charges[self.update_frequency :]
        self.arrival_times = self.arrival_times[self.update_frequency :]
        self.broken_pixels = self.broken_pixels[self.update_frequency :]

    @abstractmethod
    def _process(self, event, tel_id):
        """
        Process the sample statistics of calibration events and fill the
        corresponding MonitoringCameraContainer.

        Parameters
        ----------
        event : general event container
        tel_id : int
        """


class FlatFieldExtractor(CalibrationExtractor):
    """
    Extractor for flat-field events.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process(self, event, tel_id):
        """
        Process the sample statistics of flat-field events and fill the
        mon.tel[tel_id].flatfield container
        """

        container = event.mon.tel[tel_id].flatfield

        charge_stats = self._calculate_charge_stats(
            self.charges,
            self.sample_masked_pixels,
            self.trigger_times,
        )
        arrivaltime_stats = self._calculate_arrivaltime_stats(
            self.arrival_times,
            self.sample_masked_pixels,
        )

        flatfield_container = {
            "n_events": self.n_events_in_buffer,
            **charge_stats,
            **arrivaltime_stats,
        }
        for key, value in flatfield_container.items():
            setattr(container, key, value)

        # update the flatfield mask
        ff_charge_failing_pixels = np.logical_or(
            container.charge_median_outliers, container.charge_std_outliers
        )
        event.mon.tel[tel_id].pixel_status.flatfield_failing_pixels = np.logical_or(
            ff_charge_failing_pixels, container.time_median_outliers
        )


class PedestalExtractor(CalibrationExtractor):
    """
    Extractor for pedestal events.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process(self, event, tel_id):
        """
        Process the sample statistics of pedestal events and fill the
        mon.tel[tel_id].pedestal container
        """

        container = event.mon.tel[tel_id].pedestal

        charge_stats = self._calculate_charge_stats(
            self.charges,
            self.sample_masked_pixels,
            self.trigger_times,
        )

        pedestal_container = {
            "n_events": self.n_events_in_buffer,
            **charge_stats,
        }
        for key, value in pedestal_container.items():
            setattr(container, key, value)

        # update pedestal mask
        event.mon.tel[tel_id].pixel_status.pedestal_failing_pixels = np.logical_or(
            container.charge_median_outliers, container.charge_std_outliers
        )
