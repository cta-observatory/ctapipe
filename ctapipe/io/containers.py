"""
"""

from ctapipe.core import Container
import numpy as np


__all__ = ['RawData', 'RawCameraData', 'MCShowerData', 'MCEvent', 'MCCamera', 'CalibratedCameraData']


class RawData(Container):
    """
    Storage of a Merged Raw Data Event

    Parameters
    ----------

    run_id : int
        run number
    event_id : int
        event number
    tels_with_data : list
        list of which telescope IDs are present
    pixel_pos : dict of ndarrays by tel_id
        (deprecated)
    tel : dict of `RawCameraData` by tel_id
        dictionary of the data for each telescope
    """

    def __init__(self, name="RawData"):
        super().__init__(name)
        self.add_item('run_id')
        self.add_item('event_id')
        self.add_item('tels_with_data')
        self.add_item('tel', dict())


class MCShowerData(Container):
    def __init__(self, name='MCShowerData'):
        super().__init__(name)
        self.add_item('energy')
        self.add_item('alt')
        self.add_item('az')
        self.add_item('core_x')
        self.add_item('core_y')
        self.add_item('h_first_int')
    def __str__(self):
        return_string  = self._name+":\n"
        return_string += "energy:   {0:.2}\n".format(self.energy)
        return_string += "altitude: {0:.2}\n".format(self.alt)
        return_string += "azimuth:  {0:.2}\n".format(self.az)
        return_string += "core x:   {0:.4}\n".format(self.core_x)
        return_string += "core y:   {0:.4}"  .format(self.core_y)
        return return_string

class MCEvent(MCShowerData):
    """
    Storage of MC event data

    Parameters
    ----------

    tel : dict of `RawCameraData` by tel_id
        dictionary of the data for each telescope

    """
    def __init__(self, name='MCEvent'):
        super().__init__(name)
        self.add_item('tel',dict())
    def __str__(self):
        return_string = super().__str__()+"\n"
        npix = np.sum([np.sum(t.photo_electrons > 0) for t in self.tel.values()])
        return_string += "total photo_electrons: {}".format( npix )
        return return_string


class CentralTriggerData(Container):
    def __init__(self, name='CentralTriggerData'):
        super().__init__(name)
        self.add_item('gps_time')
        self.add_item('tels_with_trigger')


class MCCamera(Container):
    """
    Storage of mc data used for a single telescope

    Parameters
    ----------

    pe_count : dict by channel
        (masked) arrays of true (mc) pe count in each pixel (n_pixels)

    """
    def __init__(self, tel_id):
        super().__init__("CT{:03d}".format(tel_id))
        self.add_item('photo_electrons', dict())
        # Some parameters used in calibration
        self.add_item('refshapes', dict())
        self.add_item('refstep')
        self.add_item('lrefshape')
        self.add_item('time_slice')



class RawCameraData(Container):
    """
    Storage of raw data from a single telescope

    Parameters
    ----------

    adc_sums : dict by channel
        (masked) arrays of all integrated ADC data (n_pixels)
    adc_samples : dict by channel
        (masked) arrays of non-integrated ADC sample data (n_pixels, n_samples)
    num_channels : int
        number of gain channels in camera

    """
    def __init__(self, tel_id):
        super().__init__("CT{:03d}".format(tel_id))
        self.add_item('adc_sums', dict())
        self.add_item('adc_samples', dict())
        self.add_item('calibration')
        self.add_item('pedestal')
        self.add_item('num_channels')
        self.add_item('num_pixels')
        self.add_item('num_samples')


class CalibratedCameraData(Container):
    """
    Storage of calibrated (p.e.) data from a single telescope

    Parameters
    ----------

    pe_charge : dict
        ndarrays of all calibrated data (npix)
    integration_window : dict
        bool ndarrays of shape [npix][nsamples] indicating the samples used in
        the obtaining of the charge, dependant on the integration method used

    """
    def __init__(self, tel_id):
        super().__init__("CT{:03d}".format(tel_id))
        self.add_item('pe_charge', dict())
        self.add_item('integration_window', dict())
        self.add_item('pedestal_subtracted_adc', dict())
        self.add_item('num_channels')
        self.add_item('num_pixels')
        self.add_item('calibration_parameters', dict())
