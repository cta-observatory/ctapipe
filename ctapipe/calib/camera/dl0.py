"""
Calibrator for the R1 -> DL0 data level transition.

This module handles the calibration from the R1 data level to DL0. This
transition exists as a conveniance in the pipepline and can be used to test
data volume reduction methods inside the pipeline. By default, no data volume
reduction is applied, and the DL0 samples are identical to the R1. However,
if a reducer from `ctapipe.image.reducers` is passed to the
`CameraDL0Reducer`, then the reduction will be applied.
"""
from ctapipe.core import Component

__all__ = ['CameraDL0Reducer']


class CameraDL0Reducer(Component):
    """
    Parent class for the dl0 data volume reducers. Fills the dl0 container.

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
    reducer : ctapipe.calib.camera.reducer.Reducer
        The reducer to use to reduce the waveforms in the event.
        By default no data volume reduction is applied, and the dl0 samples
        will equal the r1 samples.
    kwargs
    """
    def __init__(self, config=None, tool=None, reducer=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        if reducer is None:
            self.log.info("Applying no data volume reduction in the "
                          "conversion from R1 to DL0")
        self._reducer = reducer
        self._r1_empty_warn = False

    def check_r1_exists(self, event, telid):
        """
        Check that r1 data exists. If it does not, then do not change dl0.

        This ensures that if the containers were filled from a file containing
        r1 data, it is not overwritten by non-existant data.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        telid : int
            The telescope id.

        Returns
        -------
        bool
            True if r1.tel[telid].waveform is not None, else false.
        """
        r1 = event.r1.tel[telid].waveform
        if r1 is not None:
            return True
        else:
            if not self._r1_empty_warn:
                self.log.warning("Encountered an event with no R1 data. "
                                 "DL0 is unchanged in this circumstance.")
                self._r1_empty_warn = True
            return False

    def reduce(self, event):
        """
        Perform the conversion from raw R1 data to dl0 data
        (PE Samples -> Reduced PE Samples), and fill the dl0 container.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """
        tels = event.r1.tels_with_data
        for telid in tels:
            r1 = event.r1.tel[telid].waveform
            if self.check_r1_exists(event, telid):
                if self._reducer is None:
                    event.dl0.tel[telid].waveform = r1
                else:
                    reduction = self._reducer.reduce_waveforms(r1)
                    event.dl0.tel[telid].waveform = reduction
