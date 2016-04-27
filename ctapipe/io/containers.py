"""
"""

from ctapipe.core import Container


__all__ = ['RawData', 'RawCameraData', 'MCShowerData', 'MCEvent', 'CalibratedCameraData']


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
    def __str__(self):
        return_string  = self._name+":\n"
        return_string += "energy:   {0:.2}\n".format(self.energy)
        return_string += "altitude: {0:.2}\n".format(self.alt)
        return_string += "azimuth:  {0:.2}\n".format(self.az)
        return_string += "core x:   {0:.4}\n".format(self.core_x)
        return_string += "core y:   {0:.4}"  .format(self.core_y)
        return return_string

class MCEvent(MCShowerData):
    def __init__(self, name='MCEvent'):
        super().__init__(name)
        self.add_item('photo_electrons',dict())
    def __str__(self):
        return_string = super().__str__()+"\n"
        NPix=0
        for tel in self.photo_electrons.values():
            for pix in tel.values():
                if pix > 0:
                    NPix += 1
        return_string += "hit pixels: {}".format( NPix )
        return return_string


class CentralTriggerData(Container):
    def __init__(self, name='CentralTriggerData'):
        super().__init__(name)
        self.add_item('gps_time')
        self.add_item('tels_with_trigger')


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
        self.add_item('num_channels')


class CalibratedCameraData(RawData):
    """
    Storage of calibrated (p.e.) data from a single telescope

    Parameters
    ----------

    pe_charge : dict (only one channel)
        arrays of all calibrated data (n_pixels)
    tom : time of maximum

    """
    def __init__(self, tel_id):
        super(CalibratedCameraData, self).__init__("CT{:03d}".format(tel_id))
        #super().__init__("CT{:03d}".format(tel_id))
        #self.add_item('run_id')
        #self.add_item('event_id')
        #self.add_item('tels_with_data')
        self.add_item('pe_charge', dict())
        self.add_item('tom', dict())
