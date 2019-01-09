"""
Conveniance calibrator to handle the full camera calibration.

This calibrator will apply the calibrations found in r1.py, dl0.py and dl1.py.
"""

from ctapipe.core import Component
from ctapipe.calib.camera.r1 import CameraR1Calibrator

from ctapipe.calib.camera import (
    CameraDL0Reducer,
    CameraDL1Calibrator,
)
from ctapipe.image.charge_extractors import ChargeExtractor
from ctapipe.image.waveform_cleaning import WaveformCleaner

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
        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
        radius='CameraDL1Calibrator.radius',
        ))

    """
    def __init__(self, config=None, tool=None,
                 r1_name=None,
                 extractor_name=None,
                 cleaner_name=None,
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
        r1_name : str
            The R1 calibrator to use. Manually overrides the Factory.
        extractor_name : str
            The ChargeExtractor to use. Manually overrides the Factory.
        cleaner_name : str
            The WaveformCleaner to use. Manually overrides the Factory.
        eventsource : ctapipe.io.eventsource.EventSource
            EventSource that is being used to read the events. The EventSource
            contains information (such as metadata or inst) which indicates
            the appropriate R1Calibrator to use.
        kwargs
        """
        super().__init__(config=config, parent=tool, **kwargs)

        extractor = ChargeExtractor.from_name(
            extractor_name,
            config=config,
            tool=tool
        )

        cleaner = WaveformCleaner.from_name(
            cleaner_name,
            config=config,
            tool=tool,
        )

        if r1_name:
            self.r1 = CameraR1Calibrator.from_name(
                r1_name,
                config=config,
                tool=tool,
            )
        else:
            self.r1 = CameraR1Calibrator.for_eventsource(
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
