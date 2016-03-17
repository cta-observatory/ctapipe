class CameraData():
    """
    Storage of data (as in adc_samples, adc_sums, eventnumber, trigger_type, )

    """
    def __init__(self, table):
        self.table = table

    @property
    def adc_samples(self):
        return self.table['adc_samples']

    @property
    def adc_sums(self):
        return self.table['adc_sums']

    @property
    def trigger_type(self):
        return self.table['trigger_type']

    @property
    def timestamp(self):
        return self.table['timestamp']

    @property
    def telescope_id(self):
        return self.table.meta['telescope_id']

    @property
    def telescope_name(self):
        return self.table.meta['telescope_name']

    @property
    def telescope_geometry(self):
        return self.table.meta['geometry']

    def __iter__(self):
        for row in self.table:
            yield row
