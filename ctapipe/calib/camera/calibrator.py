"""
Conveniance calibrator to handle the full camera calibration.

This calibrator will apply the calibrations found in r1.py, dl0.py and dl1.py.
"""

from ctapipe.core import Component
from ctapipe.calib.camera import (
    CameraR1Calibrator,
    CameraDL0Reducer,
    CameraDL1Calibrator,
)
from ctapipe.image import ChargeExtractor, WaveformCleaner


__all__ = ['CameraCalibrator']


class CameraCalibrator(Component):
    """
    Conveniance calibrator to handle the full camera calibration.

    This calibrator will apply the calibrations found in r1.py, dl0.py and
    dl1.py.

    The following traitlet alias configuration is suggestion for configuring
    the calibration inside a `ctapipe.core.Tool`:

    .. code-block:: python

        aliases = Dict(dict(
        ped='CameraR1CalibratorFactory.pedestal_path',
        tf='CameraR1CalibratorFactory.tf_path',
        pe='CameraR1CalibratorFactory.adc2pe_path',
        extractor='ChargeExtractorFactory.extractor',
        extractor_t0='ChargeExtractorFactory.t0',
        window_width='ChargeExtractorFactory.window_width',
        window_shift='ChargeExtractorFactory.window_shift',
        sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
        sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
        lwt='ChargeExtractorFactory.lwt',
        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
        radius='CameraDL1Calibrator.radius',
        cleaner='WaveformCleanerFactory.cleaner',
        cleaner_t0='WaveformCleanerFactory.t0',
        ))

    """
    def __init__(self, config=None, tool=None,
                 r1_product=None,
                 extractor_product='NeighbourPeakIntegrator',
                 cleaner_product='NullWaveformCleaner',
                 eventsource=None,
                 **kwargs):
        """
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
        r1_product : str
            The R1 calibrator to use. Manually overrides the Factory.
        extractor_product : str
            The ChargeExtractor to use. Manually overrides the Factory.
        cleaner_product : str
            The WaveformCleaner to use. Manually overrides the Factory.
        eventsource : ctapipe.io.eventsource.EventSource
            EventSource that is being used to read the events. The EventSource
            contains information (such as metadata or inst) which indicates
            the appropriate R1Calibrator to use.
        kwargs
        """
        super().__init__(config=config, tool=tool, **kwargs)

        extractor = ChargeExtractor.from_name(
            extractor_product,
            config=config,
            tool=tool
        )

        cleaner = WaveformCleaner.from_name(
            cleaner_product,
            config=config,
            tool=tool,
        )

        if r1_product:
            self.r1 = CameraR1Calibrator.from_name(
                r1_product,
                config=config,
                tool=tool,
            )
        else:
            self.r1 = CameraR1Calibrator.from_eventsource(
                eventsource,
                config=config,
                tool=tool,
            )

        self.dl0 = CameraDL0Reducer(config=config, tool=tool)
        self.dl1 = CameraDL1Calibrator(config=config, tool=tool,
                                       extractor=extractor,
                                       cleaner=cleaner)

    def calibrate(self, event):
        """
        Perform the full camera calibration from R0 to DL1. Any calibration
        relating to data levels before the data level the file is read into
        will be skipped.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """
        self.r1.calibrate(event)
        self.dl0.reduce(event)
        self.dl1.calibrate(event)
