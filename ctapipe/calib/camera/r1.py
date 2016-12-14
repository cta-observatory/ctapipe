"""
Module containing the r1 calibration for the MC. This could be extended to have
the r1 calibration for each telescope, if you want to be able to read in raw
r0 telescope data.
"""
from traitlets import CaselessStrEnum

from ctapipe.core import Component, Factory
from abc import abstractmethod

CALIB_SCALE = 1.05
"""
CALIB_SCALE is only relevant for MC calibration.

CALIB_SCALE is the factor needed to transform from mean p.e. units to units of
the single-p.e. peak: Depends on the collection efficiency, the asymmetry of
the single p.e. amplitude  distribution and the electronic noise added to the
signals. Default value is for GCT.

To correctly calibrate to number of photoelectron, a fresh SPE calibration
should be applied using a SPE sim_telarray run with an artificial light source.
"""


class CameraR1Calibrator(Component):
    name = 'CameraR1Calibrator'
    origin = None

    def __init__(self, config, tool, **kwargs):
        """
        Parent class for the r1 calibrators. Fills the r1 container.

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
        super().__init__(config=config, parent=tool, **kwargs)
        if self.origin is None:
            raise ValueError("Subclass of CameraR1Calibrator should specify "
                             "an origin")

    @abstractmethod
    def calibrate(self, event, telid):
        """
        Abstract method to be defined in child class.

        Perform the conversion from raw R0 data to R1 data
        (ADC Samples -> PE Samples), and fill the r1 container.
        """


class MCR1Calibrator(CameraR1Calibrator):
    name = 'MCR1Calibrator'
    origin = 'hessio'

    def calibrate(self, event, telid):
        if event.meta['source'] != 'hessio':
            raise ValueError('Using MCR1Calibrator to calibrate a non-hessio '
                             'event.')

        samples = event.dl0.tel[telid].adc_samples
        n_samples = samples.shape[2]
        pedestal = event.mc.tel[telid].pedestal / n_samples
        gain = event.mc.tel[telid].dc_to_pe * CALIB_SCALE
        calibrated = (samples - pedestal[..., None]) * gain[..., None]
        event.r1.tel[telid].pe_samples = calibrated


class CameraR1CalibratorFactory(Factory):
    name = "CameraR1CalibratorFactory"
    description = "Obtain CameraR1Calibrator based on file origin"

    subclasses = Factory.child_subclasses(CameraR1Calibrator)
    subclass_origins = [c.origin for c in subclasses]

    origin = CaselessStrEnum(subclass_origins, 'hessio',
                             help='Origin of events to be '
                                  'calibration.').tag(config=True)

    # Product classes traits

    def get_factory_name(self):
        return self.name

    def get_product_name(self):
        return self.origin
