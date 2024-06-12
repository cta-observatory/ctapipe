"""
Definition of the `CameraCalibrator` class, providing all steps needed to apply
calibration and image extraction, as well as supporting algorithms.
"""

from abc import abstractmethod
from functools import cache
import pathlib

import astropy.units as u
from astropy.table import Table
import pickle

import numpy as np
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astroquery.vizier import Vizier
from numba import float32, float64, guvectorize, int64

from ctapipe.calib.camera.extractor import StatisticsExtractor
from ctapipe.containers import DL0CameraContainer, DL1CameraContainer, PixelStatus
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    BoolTelescopeParameter,
    ComponentName,
    Dict,
    Float,
    Integer,
    TelescopeParameter,
)
from ctapipe.image.extractor import ImageExtractor
from ctapipe.image.invalid_pixels import InvalidPixelHandler
from ctapipe.image.psf_model import PSFModel
from ctapipe.image.reducer import DataVolumeReducer
from ctapipe.io import EventSource

__all__ = ["CameraCalibrator", "CalibrationCalculator"]


@cache
def _get_pixel_index(n_pixels):
    """Cached version of ``np.arange(n_pixels)``"""
    return np.arange(n_pixels)


def _get_invalid_pixels(n_channels, n_pixels, pixel_status, selected_gain_channel):
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)

    index = _get_pixel_index(n_pixels)
    masks = (
        pixel_status.hardware_failing_pixels,
        pixel_status.pedestal_failing_pixels,
        pixel_status.flatfield_failing_pixels,
    )
    for mask in masks:
        if mask is not None:
            if selected_gain_channel is not None:
                broken_pixels |= mask[selected_gain_channel, index]
            else:
                broken_pixels |= mask

    return broken_pixels


class CalibrationCalculator(TelescopeComponent):
    """
    Base component for various calibration calculators

    Attributes
    ----------
    stats_extractor: str
        The name of the StatisticsExtractor subclass to be used to calculate the statistics of an image set
    """

    stats_extractor_type = TelescopeParameter(
        trait=ComponentName(StatisticsExtractor, default_value="PlainExtractor"),
        default_value="PlainExtractor",
        help="Name of the StatisticsExtractor subclass to be used.",
    ).tag(config=True)

    # sample_size, how do i do this without copying the StatisticsExtractor traitlets? is this needed?

    def __init__(
        self,
        subarray,
        config=None,
        parent=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray. Provides information about the
            camera which are useful in calibration. Also required for
            configuring the TelescopeParameter traitlets.
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This is mutually exclusive with passing a ``parent``.
        parent: ctapipe.core.Component or ctapipe.core.Tool
            Parent of this component in the configuration hierarchy,
            this is mutually exclusive with passing ``config``
        """
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)
        self.subarray = subarray

        self.stats_extractor = StatisticsExtractor.from_name(
            self.stats_extractor_type, subarray=self.subarray, parent=self
        )

    @abstractmethod
    def __call__(self, data_url, tel_id):
        """
        Call the relevant functions to calculate the calibration coefficients
        for a given set of events

        Parameters
        ----------
        Source : EventSource
            EventSource containing the events interleaved calibration events
            from which the coefficients are to be calculated
        tel_id : int
            The telescope id. Used to obtain to correct traitlet configuration
            and instrument properties
        """

    def _check_req_data(self, url, tel_id, caltype):
        with EventSource(url, max_events=1) as source:
            event = next(iter(source))

        caldata = getattr(event.mon.tel[tel_id], caltype)

        if caldata is None:
            return False

        return True


class PedestalCalculator(CalibrationCalculator):
    """
    Component to calculate pedestals from interleaved skyfield events.

    Attributes
    ----------
    stats_extractor: str
        The name of the StatisticsExtractor subclass to be used to calculate the statistics of an image set
    """

    def __init__(
        self,
        subarray,
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

    def __call__(self, data_url, tel_id):
        pass


class GainCalculator(CalibrationCalculator):
    """
    Component to calculate the relative gain from interleaved flatfield events.

    Attributes
    ----------
    stats_extractor: str
        The name of the StatisticsExtractor subclass to be used to calculate the statistics of an image set
    """

    def __init__(
        self,
        subarray,
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

    def __call__(self, data_url, tel_id):
        if self._check_req_data(data_url, tel_id, "pedestal"):
            raise KeyError(
                "Pedestals not found. Pedestal calculation needs to be performed first."
            )


class PointingCalculator(CalibrationCalculator):
    """
    Component to calculate pointing corrections from interleaved skyfield events.

    Attributes
    ----------
    stats_extractor: str
        The name of the StatisticsExtractor subclass to be used to calculate the statistics of an image set
    telescope_location: dict
        The location of the telescope for which the pointing correction is to be calculated
    """

    telescope_location = Dict(
        {"longitude": 342.108612, "latitude": 28.761389, "elevation": 2147},
        help="Telescope location, longitude and latitude should be expressed in deg, "
        "elevation - in meters",
    ).tag(config=True)

    min_star_prominence = Integer(
        3,
        help="Minimal star prominence over the background in terms of "
        "NSB variance std deviations",
    ).tag(config=True)

    max_star_magnitude = Float(
        7.0, help="Maximal magnitude of the star to be considered in the " "analysis"
    ).tag(config=True)

    PSFModel_type = TelescopeParameter(
        trait=ComponentName(StatisticsExtractor, default_value="ComaModel"),
        default_value="PlainExtractor",
        help="Name of the PSFModel Subclass to be used.",
    ).tag(config=True)

    def __init__(
        self,
        subarray,
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

        self.psf = PSFModel.from_name(
            self.PSFModel_type, subarray=self.subarray, parent=self
        )

        self.location = EarthLocation(
            lon=self.telescope_location["longitude"] * u.deg,
            lat=self.telescope_location["latitude"] * u.deg,
            height=self.telescope_location["elevation"] * u.m,
        )

    def __call__(self, url, tel_id):
        if self._check_req_data(url, tel_id, "flatfield"):
            raise KeyError(
                "Relative gain not found. Gain calculation needs to be performed first."
            )

        self.tel_id = tel_id

        with EventSource(url, max_events=1) as src:
            self.camera_geometry = src.subarray.tel[self.tel_id].camera.geometry
            self.focal_length = src.subarray.tel[
                self.tel_id
            ].optics.equivalent_focal_length
            self.pixel_radius = self.camera_geometry.pixel_width[0]

            event = next(iter(src))

            self.pointing = SkyCoord(
                az=event.pointing.tel[self.telescope_id].azimuth,
                alt=event.pointing.tel[self.telescope_id].altitude,
                frame="altaz",
                obstime=event.trigger.time.utc,
                location=self.location,
            )

        stars_in_fov = Vizier.query_region(
            self.pointing, radius=Angle(2.0, "deg"), catalog="NOMAD"
        )[0]

        stars_in_fov = stars_in_fov[stars_in_fov["Bmag"] < self.max_star_magnitude]

    def _calibrate_varimages(self, varimages, gain):
        pass


class CameraCalibrator(TelescopeComponent):
    """
    Calibrator to handle the full camera calibration chain, in order to fill
    the DL1 data level in the event container.

    Attributes
    ----------
    data_volume_reducer_type: str
        The name of the DataVolumeReducer subclass to be used
        for data volume reduction

    image_extractor_type: str
        The name of the ImageExtractor subclass to be used for image extraction
    """

    data_volume_reducer_type = ComponentName(
        DataVolumeReducer, default_value="NullDataVolumeReducer"
    ).tag(config=True)

    image_extractor_type = TelescopeParameter(
        trait=ComponentName(ImageExtractor, default_value="NeighborPeakWindowSum"),
        default_value="NeighborPeakWindowSum",
        help="Name of the ImageExtractor subclass to be used.",
    ).tag(config=True)

    invalid_pixel_handler_type = ComponentName(
        InvalidPixelHandler,
        default_value="NeighborAverage",
        help="Name of the InvalidPixelHandler to use",
        allow_none=True,
    ).tag(config=True)

    apply_waveform_time_shift = BoolTelescopeParameter(
        default_value=False,
        help=(
            "Apply waveform time shift corrections."
            " The minimal integer shift to synchronize waveforms is applied"
            " before peak extraction if this option is True"
        ),
    ).tag(config=True)

    apply_peak_time_shift = BoolTelescopeParameter(
        default_value=True,
        help=(
            "Apply peak time shift corrections."
            " Apply the remaining absolute and fractional time shift corrections"
            " to the peak time after pulse extraction."
            " If `apply_waveform_time_shift` is False, this will apply the full time shift"
        ),
    ).tag(config=True)

    def __init__(
        self,
        subarray,
        config=None,
        parent=None,
        image_extractor=None,
        data_volume_reducer=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray. Provides information about the
            camera which are useful in calibration. Also required for
            configuring the TelescopeParameter traitlets.
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This is mutually exclusive with passing a ``parent``.
        parent: ctapipe.core.Component or ctapipe.core.Tool
            Parent of this component in the configuration hierarchy,
            this is mutually exclusive with passing ``config``
        data_volume_reducer: ctapipe.image.reducer.DataVolumeReducer
            The DataVolumeReducer to use.
            This is used to override the options from the config system
            and to enable passing a preconfigured reducer.
        image_extractor: ctapipe.image.extractor.ImageExtractor
            The ImageExtractor to use. If None, the default via the
            configuration system will be constructed.
        """
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)
        self.subarray = subarray

        self._r1_empty_warn = False
        self._dl0_empty_warn = False

        self.image_extractors = {}

        if image_extractor is None:
            for _, _, name in self.image_extractor_type:
                self.image_extractors[name] = ImageExtractor.from_name(
                    name, subarray=self.subarray, parent=self
                )
        else:
            name = image_extractor.__class__.__name__
            self.image_extractor_type = [("type", "*", name)]
            self.image_extractors[name] = image_extractor

        if data_volume_reducer is None:
            self.data_volume_reducer = DataVolumeReducer.from_name(
                self.data_volume_reducer_type, subarray=self.subarray, parent=self
            )
        else:
            self.data_volume_reducer = data_volume_reducer

        self.invalid_pixel_handler = None
        if self.invalid_pixel_handler_type is not None:
            self.invalid_pixel_handler = InvalidPixelHandler.from_name(
                self.invalid_pixel_handler_type,
                subarray=self.subarray,
                parent=self,
            )

    def _check_r1_empty(self, waveforms):
        if waveforms is None:
            if not self._r1_empty_warn:
                self.log.debug(
                    "Encountered an event with no R1 data. "
                    "DL0 is unchanged in this circumstance."
                )
                self._r1_empty_warn = True
            return True
        else:
            return False

    def _check_dl0_empty(self, waveforms):
        if waveforms is None:
            if not self._dl0_empty_warn:
                self.log.warning(
                    "Encountered an event with no DL0 data. "
                    "DL1 is unchanged in this circumstance."
                )
                self._dl0_empty_warn = True
            return True
        else:
            return False

    def _calibrate_dl0(self, event, tel_id):
        r1 = event.r1.tel[tel_id]

        if self._check_r1_empty(r1.waveform):
            return

        signal_pixels = self.data_volume_reducer(
            r1.waveform,
            tel_id=tel_id,
            selected_gain_channel=r1.selected_gain_channel,
        )

        dl0_waveform = r1.waveform.copy()
        dl0_waveform[:, ~signal_pixels] = 0

        dl0_pixel_status = r1.pixel_status.copy()
        # set dvr pixel bit in pixel_status for pixels kept by DVR
        dl0_pixel_status[signal_pixels] |= PixelStatus.DVR_STORED_AS_SIGNAL
        # unset dvr bits for removed pixels
        dl0_pixel_status[~signal_pixels] &= ~np.uint8(PixelStatus.DVR_STATUS)

        event.dl0.tel[tel_id] = DL0CameraContainer(
            event_type=r1.event_type,
            event_time=r1.event_time,
            waveform=dl0_waveform,
            selected_gain_channel=r1.selected_gain_channel,
            pixel_status=dl0_pixel_status,
            first_cell_id=r1.first_cell_id,
            calibration_monitoring_id=r1.calibration_monitoring_id,
        )

    def _calibrate_dl1(self, event, tel_id):
        waveforms = event.dl0.tel[tel_id].waveform
        if self._check_dl0_empty(waveforms):
            return

        n_channels, n_pixels, n_samples = waveforms.shape

        selected_gain_channel = event.dl0.tel[tel_id].selected_gain_channel
        broken_pixels = _get_invalid_pixels(
            n_channels,
            n_pixels,
            event.mon.tel[tel_id].pixel_status,
            selected_gain_channel,
        )
        pixel_index = _get_pixel_index(n_pixels)

        dl1_calib = event.calibration.tel[tel_id].dl1
        readout = self.subarray.tel[tel_id].camera.readout

        # subtract any remaining pedestal before extraction
        if dl1_calib.pedestal_offset is not None:
            # this copies intentionally, we don't want to modify the dl0 data
            # waveforms have shape (n_channels, n_pixel, n_samples), pedestals (n_pixels)
            waveforms = waveforms.copy()
            waveforms -= np.atleast_2d(dl1_calib.pedestal_offset)[..., np.newaxis]

        if n_samples == 1:
            # To handle ASTRI and dst
            # TODO: Improved handling of ASTRI and dst
            #   - dst with custom EventSource?
            #   - Read into dl1 container directly?
            #   - Don't do anything if dl1 container already filled
            #   - Update on SST review decision
            dl1 = DL1CameraContainer(
                image=np.squeeze(waveforms).astype(np.float32),
                peak_time=np.zeros(n_pixels, dtype=np.float32),
                is_valid=True,
            )
        else:
            # shift waveforms if time_shift calibration is available
            time_shift = dl1_calib.time_shift
            remaining_shift = None
            if time_shift is not None:
                if selected_gain_channel is not None:
                    time_shift = time_shift[selected_gain_channel, pixel_index]

                if self.apply_waveform_time_shift.tel[tel_id]:
                    sampling_rate = readout.sampling_rate.to_value(u.GHz)
                    time_shift_samples = time_shift * sampling_rate
                    waveforms, remaining_shift = shift_waveforms(
                        waveforms, time_shift_samples
                    )
                    remaining_shift /= sampling_rate
                else:
                    remaining_shift = time_shift

            extractor = self.image_extractors[self.image_extractor_type.tel[tel_id]]
            dl1 = extractor(
                waveforms,
                tel_id=tel_id,
                selected_gain_channel=selected_gain_channel,
                broken_pixels=broken_pixels,
            )

            # correct non-integer remainder of the shift if given
            if self.apply_peak_time_shift.tel[tel_id] and remaining_shift is not None:
                dl1.peak_time -= remaining_shift

        # Calibrate extracted charge
        if (
            dl1_calib.relative_factor is not None
            and dl1_calib.absolute_factor is not None
        ):
            if selected_gain_channel is None:
                dl1.image *= dl1_calib.relative_factor / dl1_calib.absolute_factor
            else:
                corr = (
                    dl1_calib.relative_factor[selected_gain_channel, pixel_index]
                    / dl1_calib.absolute_factor[selected_gain_channel, pixel_index]
                )
                dl1.image *= corr

        # handle invalid pixels
        if self.invalid_pixel_handler is not None:
            dl1.image, dl1.peak_time = self.invalid_pixel_handler(
                tel_id,
                dl1.image,
                dl1.peak_time,
                broken_pixels,
            )

        # store the results in the event structure
        event.dl1.tel[tel_id] = dl1

    def __call__(self, event):
        """
        Perform the full camera calibration from R1 to DL1. Any calibration
        relating to data levels before the data level the file is read into
        will be skipped.

        Parameters
        ----------
        event : container
            A `~ctapipe.containers.ArrayEventContainer` event container
        """
        # TODO: How to handle different calibrations depending on tel_id?
        tel = event.r1.tel or event.dl0.tel or event.dl1.tel
        for tel_id in tel.keys():
            self._calibrate_dl0(event, tel_id)
            self._calibrate_dl1(event, tel_id)


def shift_waveforms(waveforms, time_shift_samples):
    """
    Shift the waveforms by the mean integer shift to mediate
    time differences between pixels.
    The remaining shift (mean + fractional part) is returned
    to be applied later to the extracted peak time.

    Parameters
    ----------
    waveforms: ndarray of shape (n_channels, n_pixels, n_samples)
        The waveforms to shift
    time_shift_samples: ndarray
        The shift to apply in units of samples.
        Waveforms are shifted to the left by the smallest integer
        that minimizes inter-pixel differences.

    Returns
    -------
    shifted_waveforms: ndarray of shape (n_channels, n_pixels, n_samples)
        The shifted waveforms
    remaining_shift: ndarray
        The remaining shift after applying the integer shift to the waveforms.
    """
    mean_shift = time_shift_samples.mean(axis=-1, keepdims=True)
    integer_shift = np.round(time_shift_samples - mean_shift).astype("int16")
    remaining_shift = time_shift_samples - integer_shift
    shifted_waveforms = _shift_waveforms_by_integer(waveforms, integer_shift)
    return shifted_waveforms, remaining_shift


@guvectorize(
    [(float64[:], int64, float64[:]), (float32[:], int64, float32[:])],
    "(s),()->(s)",
    nopython=True,
    cache=True,
)
def _shift_waveforms_by_integer(waveforms, integer_shift, shifted_waveforms):
    n_samples = waveforms.size

    for new_sample_idx in range(n_samples):
        # repeat first value if out ouf bounds to the left
        # repeat last value if out ouf bounds to the right
        sample_idx = min(max(new_sample_idx + integer_shift, 0), n_samples - 1)
        shifted_waveforms[new_sample_idx] = waveforms[sample_idx]
