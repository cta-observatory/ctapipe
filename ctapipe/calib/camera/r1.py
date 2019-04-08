"""
Calibrator for the R0 -> R1 data level transition.

This module handles the calibration from the R0 data level to R1. This data
level transition will be handled by the camera servers, not in the pipeline,
however the pipeline can be used as a test-bench in the comissioning stage of
the cameras.

As the R1 calibration is camera specific, each camera (and seperately the MC)
requires their own calibrator class with inherits from `CameraR1Calibrator`.
`HessioR1Calibrator` is the calibrator for the MC data obtained from readhess.
Through the use of `CameraR1Calibrator.from_eventsource()`, the correct
`CameraR1Calibrator` can be obtained based on the origin (MC/Camera format)
of the data.
"""
from abc import abstractmethod

from ...core import Component

__all__ = [
    'NullR1Calibrator',
    'HESSIOR1Calibrator',
    'CameraR1Calibrator',
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
    def __init__(self, config=None, parent=None, **kwargs):
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
        super().__init__(config=config, parent=parent, **kwargs)
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

    @classmethod
    def from_eventsource(cls, eventsource, **kwargs):
        """
        Obtain the correct `CameraR1Calibrator` according the the `EventSource`
        of the file.

        Parameters
        ----------
        eventsource : ctapipe.io.EventSource
            Subclass of `ctapipe.io.EventSource`
        kwargs
            Named arguments to pass to the `CameraR1Calibrator`

        Returns
        -------
        calibrator
            Subclass of `CameraR1Calibrator`
        """
        if (hasattr(eventsource, 'metadata') and
                eventsource.metadata['is_simulation']):
            return HESSIOR1Calibrator(**kwargs)
        elif eventsource.__class__.__name__ == "TargetIOEventSource":
            from ctapipe_io_targetio import TargetIOR1Calibrator
            return TargetIOR1Calibrator(**kwargs)

        return NullR1Calibrator(**kwargs)


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

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config, parent, **kwargs)
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
