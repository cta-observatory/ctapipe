"""
"""

from ctapipe.core import Container


__all__ = ['RawData', 'RawCameraData']


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
        super(RawData, self).__init__(name)
        self.add_item('run_id')
        self.add_item('event_id')
        self.add_item('tels_with_data')
        self.add_item('tel', dict())


class RawCameraData(Container):
    """ 
    Storage of raw data from a single telescope 

    Parameters
    ----------

    adc_sums : dict by channel
        arrays of all integrated ADC data (n_pixels)
    adc_samples : dict by channel
        arrays of non-integrated ADC sample data (n_pixels, n_samples)
    num_channels : int
        number of gain channels in camera

    """
    def __init__(self, tel_id):
        super(RawCameraData, self).__init__("CT{:03d}".format(tel_id))
        self.add_item('adc_sums', dict())
        self.add_item('adc_samples', dict())
        self.add_item('num_channels')
