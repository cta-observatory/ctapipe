"""
Conveniance calibrator to handle the full camera calibration.

This calibrator will apply the calibrations found in r1.py, dl0.py and dl1.py.
"""

from ctapipe.core import Component
from ctapipe.calib.camera import (
    CameraDL0Reducer,
    CameraDL1Calibrator,
)
from ctapipe.image import ImageExtractor


__all__ = ['CameraCalibrator']


class CameraCalibrator(Component):
    """
    Conveniance calibrator to handle the full camera calibration.

    This calibrator will apply the calibrations found in r1.py, dl0.py and
    dl1.py.

    """
    def __init__(self, config=None, parent=None,
                 extractor_name='NeighborPeakWindowSum',
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
        extractor_name : str
            The name of the ImageExtractor to use.
        kwargs
        """
        super().__init__(config=config, parent=parent, **kwargs)

        extractor = ImageExtractor.from_name(
            extractor_name,
            parent=self,
        )

        self.dl0 = CameraDL0Reducer(parent=parent)
        self.dl1 = CameraDL1Calibrator(
            parent=self,
            extractor=extractor,
        )

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
        self.dl0.reduce(event)
        self.dl1.calibrate(event)
