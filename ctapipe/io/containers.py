"""
"""

from ctapipe.core import Container


__all__ = ['RawData', 'RawCameraData']


class RawData(Container):

    def __init__(self, name="RawData"):
        self.add_item('run_id')
        super(RawData, self).__init__(name)
        self.add_item('event_id')
        self.add_item('tels_with_data')
        self.add_item('pixel_pos', dict())
        self.add_item('tel_data', dict())


class RawCameraData(Container):

    def __init__(self, tel_id):
        super(RawCameraData, self).__init__("CT{:03d}".format(tel_id))
        self.add_item('adc_sums', dict())
        self.add_item('adc_samples', dict())
        self.add_item('num_channels')
