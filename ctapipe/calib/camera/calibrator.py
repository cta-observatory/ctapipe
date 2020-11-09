"""
Definition of the `CameraCalibrator` class, providing all steps needed to apply
calibration and image extraction, as well as supporting algorithms.
"""

import warnings
import numpy as np
import astropy.units as u

from ctapipe.core import TelescopeComponent
from ctapipe.image.extractor import ImageExtractor
from ctapipe.image.reducer import DataVolumeReducer
from ctapipe.core.traits import create_class_enum_trait, BoolTelescopeParameter

from numba import guvectorize, float64, float32, int64

__all__ = ["CameraCalibrator"]


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

    data_volume_reducer_type = create_class_enum_trait(
        DataVolumeReducer, default_value="NullDataVolumeReducer"
    ).tag(config=True)

    image_extractor_type = create_class_enum_trait(
        ImageExtractor, default_value="NeighborPeakWindowSum"
    ).tag(config=True)

    apply_waveform_time_shift = BoolTelescopeParameter(
        default_value=True,
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

        if image_extractor is None:
            self.image_extractor = ImageExtractor.from_name(
                self.image_extractor_type, subarray=self.subarray, parent=self
            )
        else:
            self.image_extractor = image_extractor

        if data_volume_reducer is None:
            self.data_volume_reducer = DataVolumeReducer.from_name(
                self.data_volume_reducer_type, subarray=self.subarray, parent=self
            )
        else:
            self.data_volume_reducer = data_volume_reducer

    def _check_r1_empty(self, waveforms):
        if waveforms is None:
            if not self._r1_empty_warn:
                warnings.warn(
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
                warnings.warn(
                    "Encountered an event with no DL0 data. "
                    "DL1 is unchanged in this circumstance."
                )
                self._dl0_empty_warn = True
            return True
        else:
            return False

    def _calibrate_dl0(self, event, telid):
        waveforms = event.r1.tel[telid].waveform
        selected_gain_channel = event.r1.tel[telid].selected_gain_channel
        if self._check_r1_empty(waveforms):
            return

        reduced_waveforms_mask = self.data_volume_reducer(
            waveforms, telid=telid, selected_gain_channel=selected_gain_channel
        )

        waveforms_copy = waveforms.copy()
        waveforms_copy[~reduced_waveforms_mask] = 0
        event.dl0.tel[telid].waveform = waveforms_copy
        event.dl0.tel[telid].selected_gain_channel = selected_gain_channel

    def _calibrate_dl1(self, event, telid):
        waveforms = event.dl0.tel[telid].waveform
        selected_gain_channel = event.dl0.tel[telid].selected_gain_channel
        dl1_calib = event.calibration.tel[telid].dl1

        if self._check_dl0_empty(waveforms):
            return

        selected_gain_channel = event.r1.tel[telid].selected_gain_channel
        time_shift = event.calibration.tel[telid].dl1.time_shift
        readout = self.subarray.tel[telid].camera.readout
        n_pixels, n_samples = waveforms.shape

        # subtract any remaining pedestal before extraction
        if dl1_calib.pedestal_offset is not None:
            # this copies intentionally, we don't want to modify the dl0 data
            # waveforms have shape (n_pixel, n_samples), pedestals (n_pixels, )
            waveforms = waveforms - dl1_calib.pedestal_offset[:, np.newaxis]

        if n_samples == 1:
            # To handle ASTRI and dst
            # TODO: Improved handling of ASTRI and dst
            #   - dst with custom EventSource?
            #   - Read into dl1 container directly?
            #   - Don't do anything if dl1 container already filled
            #   - Update on SST review decision
            charge = waveforms[..., 0].astype(np.float32)
            peak_time = np.zeros(n_pixels, dtype=np.float32)
        else:

            # shift waveforms if time_shift calibration is available
            if time_shift is not None:
                if self.apply_waveform_time_shift.tel[telid]:
                    sampling_rate = readout.sampling_rate.to_value(u.GHz)
                    time_shift_samples = time_shift * sampling_rate
                    waveforms, remaining_shift = shift_waveforms(
                        waveforms, time_shift_samples
                    )
                    remaining_shift /= sampling_rate
                else:
                    remaining_shift = time_shift

            charge, peak_time = self.image_extractor(
                waveforms, telid=telid, selected_gain_channel=selected_gain_channel
            )

            # correct non-integer remainder of the shift if given
            if self.apply_peak_time_shift.tel[telid] and time_shift is not None:
                peak_time -= remaining_shift

        # Calibrate extracted charge
        charge *= dl1_calib.relative_factor / dl1_calib.absolute_factor

        event.dl1.tel[telid].image = charge
        event.dl1.tel[telid].peak_time = peak_time

    def __call__(self, event):
        """
        Perform the full camera calibration from R1 to DL1. Any calibration
        relating to data levels before the data level the file is read into
        will be skipped.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """
        # TODO: How to handle different calibrations depending on telid?
        tel = event.r1.tel or event.dl0.tel or event.dl1.tel
        for telid in tel.keys():
            self._calibrate_dl0(event, telid)
            self._calibrate_dl1(event, telid)


def shift_waveforms(waveforms, time_shift_samples):
    """
    Shift the waveforms by the mean integer shift to mediate
    time differences between pixels.
    The remaining shift (mean + fractional part) is returned
    to be applied later to the extracted peak time.

    Parameters
    ----------
    waveforms: ndarray of shape (n_pixels, n_samples)
        The waveforms to shift
    time_shift_samples: ndarray of shape (n_pixels, )
        The shift to apply in units of samples.
        Waveforms are shifted to the left by the smallest integer
        that minimizes inter-pixel differences.

    Returns
    -------
    shifted_waveforms: ndarray of shape (n_pixels, n_samples)
        The shifted waveforms
    remaining_shift: ndarray of shape (n_pixels, )
        The remaining shift after applying the integer shift to the waveforms.
    """
    mean_shift = time_shift_samples.mean()
    integer_shift = np.round(time_shift_samples - mean_shift).astype("int16")
    remaining_shift = time_shift_samples - integer_shift
    shifted_waveforms = _shift_waveforms_by_integer(waveforms, integer_shift)
    return shifted_waveforms, remaining_shift


@guvectorize(
    [(float64[:], int64, float64[:]), (float32[:], int64, float32[:])],
    "(s),()->(s)",
    nopython=True,
)
def _shift_waveforms_by_integer(waveforms, integer_shift, shifted_waveforms):
    n_samples = waveforms.size

    for new_sample_idx in range(n_samples):
        # repeat first value if out ouf bounds to the left
        # repeat last value if out ouf bounds to the right
        sample_idx = min(max(new_sample_idx + integer_shift, 0), n_samples - 1)
        shifted_waveforms[new_sample_idx] = waveforms[sample_idx]
