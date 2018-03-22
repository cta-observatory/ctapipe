"""
Calibrator for the R0 -> R1 data level transition.

This module handles the calibration from the R0 data level to R1. This data
level transition will be handled by the camera servers, not in the pipeline,
however the pipeline can be used as a test-bench in the comissioning stage of
the cameras.

As the R1 calibration is camera specific, each camera (and seperately the MC)
requires their own calibrator class with inherits from `CameraR1Calibrator`.
`HessioR1Calibrator` is the calibrator for the MC data obtained from readhess.
Through the use of `CameraR1CalibratorFactory`, the correct
`CameraR1Calibrator` can be obtained based on the origin (MC/Camera format)
of the data.
"""
import os.path
from abc import abstractmethod
import numpy as np
from scipy.interpolate import interp1d
from ...core import Component, Factory
from ...core.provenance import Provenance
from ...io import EventSource
from ...core.traits import Unicode

__all__ = [
    'NullR1Calibrator',
    'HESSIOR1Calibrator',
    'TargetIOR1Calibrator',
    'CameraR1CalibratorFactory',
    'SST1MR1Calibrator',
]


class CameraR1Calibrator(Component):
    """
    The base R1-level calibrator. Fills the r1 container.

    The R1 calibrator performs the camera-specific R1 calibration that is
    usually performed on the raw data by the camera server. This calibrator
    exists in ctapipe for testing and prototyping purposes.

    Parameters
    ----------
    config : traitlets.loader.Config
        Configuration specified by config file or cmdline arguments.
        Used to set traitlet values.
        Set to None if no configuration to pass.
    tool : ctapipe.core.Tool or None
        Tool executable that is calling this component.
        Passes the correct logger to the component.
        Set to None if no Tool to pass.
    kwargs
    """

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Parent class for the r1 calibrators. Fills the r1 container.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool or None
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, parent=tool, **kwargs)
        self._r0_empty_warn = False

    @abstractmethod
    def calibrate(self, event):
        """
        Abstract method to be defined in child class.

        Perform the conversion from raw R0 data to R1 data
        (ADC Samples -> PE Samples), and fill the r1 container.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """

    def check_r0_exists(self, event, telid):
        """
        Check that r0 data exists. If it does not, then do not change r1.

        This ensures that if the containers were filled from a file containing
        r0 data, it is not overwritten by non-existant data.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        telid : int
            The telescope id.

        Returns
        -------
        bool
            True if r0.tel[telid].waveform is not None, else false.
        """
        r0 = event.r0.tel[telid].waveform
        if r0 is not None:
            return True
        else:
            if not self._r0_empty_warn:
                self.log.warning("Encountered an event with no R0 data. "
                                 "R1 is unchanged in this circumstance.")
                self._r0_empty_warn = True
            return False


class NullR1Calibrator(CameraR1Calibrator):
    """
    A dummy R1 calibrator that simply fills the r1 container with the samples
    from the r0 container.

    Parameters
    ----------
    config : traitlets.loader.Config
        Configuration specified by config file or cmdline arguments.
        Used to set traitlet values.
        Set to None if no configuration to pass.
    tool : ctapipe.core.Tool or None
        Tool executable that is calling this component.
        Passes the correct logger to the component.
        Set to None if no Tool to pass.
    kwargs
    """

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config, tool, **kwargs)
        self.log.info("Using NullR1Calibrator, if event source is at "
                      "the R0 level, then r1 samples will equal r0 samples")

    def calibrate(self, event):
        for telid in event.r0.tels_with_data:
            if self.check_r0_exists(event, telid):
                samples = event.r0.tel[telid].waveform
                event.r1.tel[telid].waveform = samples.astype('float32')


class HESSIOR1Calibrator(CameraR1Calibrator):
    """
    The R1 calibrator for hessio files. Fills the r1 container.

    This calibrator correctly applies the pedestal subtraction and conversion
    from counts to photoelectrons for the Monte-Carlo data.

    Parameters
    ----------
    config : traitlets.loader.Config
        Configuration specified by config file or cmdline arguments.
        Used to set traitlet values.
        Set to None if no configuration to pass.
    tool : ctapipe.core.Tool or None
        Tool executable that is calling this component.
        Passes the correct logger to the component.
        Set to None if no Tool to pass.
    kwargs
    """

    calib_scale = 1.05
    """
    CALIB_SCALE is only relevant for MC calibration.

    CALIB_SCALE is the factor needed to transform from mean p.e. units to
    units of the single-p.e. peak: Depends on the collection efficiency,
    the asymmetry of the single p.e. amplitude  distribution and the
    electronic noise added to the signals. Default value is for GCT.

    To correctly calibrate to number of photoelectron, a fresh SPE calibration
    should be applied using a SPE sim_telarray run with an
    artificial light source.
    """
    # TODO: Handle calib_scale differently per simlated telescope

    def calibrate(self, event):
        if event.meta['origin'] != 'hessio':
            raise ValueError('Using HESSIOR1Calibrator to calibrate a '
                             'non-hessio event.')

        for telid in event.r0.tels_with_data:
            if self.check_r0_exists(event, telid):
                samples = event.r0.tel[telid].waveform
                n_samples = samples.shape[2]
                ped = event.mc.tel[telid].pedestal / n_samples
                gain = event.mc.tel[telid].dc_to_pe * self.calib_scale
                calibrated = (samples - ped[..., None]) * gain[..., None]
                event.r1.tel[telid].waveform = calibrated


class TargetIOR1Calibrator(CameraR1Calibrator):

    pedestal_path = Unicode(
        '',
        allow_none=True,
        help='Path to the TargetCalib pedestal file'
    ).tag(config=True)
    tf_path = Unicode(
        '',
        allow_none=True,
        help='Path to the TargetCalib Transfer Function file'
    ).tag(config=True)
    pe_path = Unicode(
        '',
        allow_none=True,
        help='Path to the TargetCalib pe conversion file'
    ).tag(config=True)
    ff_path = Unicode(
        '',
        allow_none=True,
        help='Path to a TargetCalib flat field file'
    ).tag(config=True)

    def __init__(self, config=None, tool=None, **kwargs):
        """
        The R1 calibrator for targetio files (i.e. files containing data
        taken with a TARGET module, such as with CHEC)

        Fills the r1 container.

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
        try:
            import target_calib
        except ImportError:
            msg = ("Cannot find target_calib module, please follow "
                   "installation instructions from https://forge.in2p3.fr/"
                   "projects/gct/wiki/Installing_CHEC_Software")
            self.log.error(msg)
            raise

        self.tc = target_calib
        self.calibrator = None
        self.telid = 0

        self._load_calib()

    def calibrate(self, event):
        """
        Placeholder function to satisfy abstract parent, this is overloaded by
        either fake_calibrate or real_calibrate.
        """
        pass

    def _load_calib(self):
        """
        If a pedestal file has been supplied, create a target_calib
        Calibrator object. If it hasn't then point calibrate to
        fake_calibrate, where nothing is done to the waveform.
        """
        if self.pedestal_path:
            self.calibrator = self.tc.Calibrator(self.pedestal_path,
                                                 self.tf_path,
                                                 [self.pe_path, self.ff_path])
            self.calibrate = self.real_calibrate
        else:
            self.log.warning("No pedestal path supplied, "
                             "r1 samples will equal r0 samples.")
            self.calibrate = self.fake_calibrate

    def fake_calibrate(self, event):
        """
        Don't perform any calibration on the waveforms, just fill the
        R1 container.

        Parameters
        ----------
        event : `ctapipe` event-container
        """
        if event.meta['origin'] != 'targetio':
            raise ValueError('Using TargetioR1Calibrator to calibrate a '
                             'non-targetio event.')

        if self.check_r0_exists(event, self.telid):
            samples = event.r0.tel[self.telid].waveform
            event.r1.tel[self.telid].waveform = samples.astype('float32')

    def real_calibrate(self, event):
        """
        Apply the R1 calibration defined in target_calib and fill the
        R1 container.

        Parameters
        ----------
        event : `ctapipe` event-container
        """
        if event.meta['origin'] != 'targetio':
            raise ValueError('Using TargetioR1Calibrator to calibrate a '
                             'non-targetio event.')

        if self.check_r0_exists(event, self.telid):
            samples = event.r0.tel[self.telid].waveform[0]
            fci = event.targetio.tel[self.telid].first_cell_ids
            r1 = event.r1.tel[self.telid].waveform[0]
            self.calibrator.ApplyEvent(samples, fci, r1)


class SST1MR1Calibrator(CameraR1Calibrator):
    '''
    *  subtract a pixel-wise rough electronic + NSB noise pedestal value,
        so that the values fluctuate around 0 counts
        (and record it as MON or SVC data, so it can be un-applied
        or quality-checked later)
    *  apply a pixel-wise multiplicative gain correction factor to correct
        for differences in camera uniformity
    *  apply a scalar scale and shift the signal so that it efficiently fits
        in a 16-bit signed integer value with the required dynamic range
        (see Section 3.3)
    *  record all applied corrections in SVC or MON streams.
    '''
    cell_capacitance_in_farad = 85e-15
    bias_resistance = 10e3
    nsb_rate = np.logspace(-3, np.log10(50), 30)
    gain_drop = (
        1. / (
            1. + (
                nsb_rate *
                cell_capacitance_in_farad *
                bias_resistance * 1E9
            )
        )
    )

    # measured in a toy-MC for the above mentioned nsb_rate
    # the simulation takes some time, so we just put the results here
    # https://github.com/cta-sst-1m/digicamtoy
    # TODO: still have to find *the* exact source for these numbers
    # but it is somewhere in that repo.
    baseline_shift = np.array([
        500.10844, 500.13456, 500.19772, 500.2697, 500.44418, 500.6094,
        500.87226, 501.33124, 501.8768, 502.71168, 503.9574, 505.51968,
        507.92038, 511.24914, 515.572, 521.41952, 528.5894, 537.03064,
        547.04614, 557.76222, 568.03408, 577.88066, 586.55614, 593.50884,
        598.97944, 603.33174, 606.54438, 608.84342, 610.4056, 611.94482,
    ])
    baseline_shift -= baseline_shift[0]


    gain_drop_from_baseline_shift = interp1d(
        x=baseline_shift,
        y=gain_drop,
        kind='cubic',
        fill_value="extrapolate",
    )

    dark_baseline_path = Unicode(
        "",
        help='Path to a SST1M dark-baseline file (at the moment .npz)'
    ).tag(config=True)

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        if not os.path.isfile(self.dark_baseline_path):
            raise AttributeError('dark_baseline_path must point to a file')

        self.dark_baseline = np.load(self.dark_baseline_path)['baseline']
        Provenance().add_input_file(
            self.dark_baseline_path,
            role='r1.tel.svc.sst1m_dark_baseline'
        )


    def calibrate(self, event):

        for telescope_id in event.r0.tels_with_data:
            r0 = event.r0.tel[telescope_id]
            sst1m = event.sst1m.tel[telescope_id]
            r1 = event.r1.tel[telescope_id]

            baseline_subtracted = r0.waveform - sst1m.digicam_baseline[:, np.newaxis]
            baseline_shift = sst1m.digicam_baseline - self.dark_baseline
            gain_drop = self.gain_drop_from_baseline_shift(baseline_shift)

            uniform_over_camera = baseline_subtracted * gain_drop[:, np.newaxis]
            r1.waveform = uniform_over_camera.round().astype(np.int16)
        return event



class CameraR1CalibratorFactory(Factory):
    """
    The R1 calibrator `ctapipe.core.factory.Factory`. This
    `ctapipe.core.factory.Factory` allows the correct
    `CameraR1Calibrator` to be obtained for the data investigated.

    Additional filepaths are required by some cameras for R1 calibration. Due
    to the current inplementation of `ctapipe.core.factory.Factory`, every
    trait that could
    possibly be required for a child `ctapipe.core.component.Component` of
    `CameraR1Calibrator` is
    included in this `ctapipe.core.factory.Factory`. The
    `CameraR1Calibrator` specific to a
    camera type should then define how/if that filepath should be used. The
    format of the file is not restricted, and the file can be read from inside
    ctapipe, or can call a different library created by the camera teams for
    the calibration of their camera.
    """
    base = CameraR1Calibrator
    custom_product_help = ('R1 Calibrator to use. If None then a '
                           'calibrator will either be selected based on the '
                           'supplied EventSource, or will default to '
                           '"NullR1Calibrator".')

    def __init__(self, config=None, tool=None, eventsource=None, **kwargs):
        """
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
        eventsource : ctapipe.io.eventsource.EventSource
            EventSource that is being used to read the events. The EventSource
            contains information (such as metadata or inst) which indicates
            the appropriate R1Calibrator to use.
        kwargs

        """

        super().__init__(config, tool, **kwargs)
        if eventsource and not issubclass(type(eventsource), EventSource):
            raise TypeError(
                "eventsource must be a ctapipe.io.eventsource.EventSource"
            )
        self.eventsource = eventsource

    def _get_product_name(self):
        try:
            return super()._get_product_name()
        except AttributeError:
            es = self.eventsource
            if es:
                if es.metadata['is_simulation']:
                    return 'HESSIOR1Calibrator'
                elif es.__class__.__name__ == "TargetIOEventSource":
                    return 'TargetIOR1Calibrator'
            return 'NullR1Calibrator'
