from astropy.table import Table
from astropy.time import Time
from ctapipe.instrument.CameraDescription import make_rectangular_camera_geometry

import numpy as np
from numpy import ma

def create_dummy_chunk(N=10, pixels=1440, samples=50):
    data = []
    telescope_id = 1
    geometry = make_rectangular_camera_geometry()
    for eventnumber in range(N):
        adc_samples = np.random.normal(loc=0, scale=1, size=(pixels, samples))
        mask = np.random.randint(low=0, high=1, size=(pixels, samples))
        tm = Time('2017-01-01T00:00:00', format='isot', scale='utc')

        trigger_type = np.random.randint(low=0, high=10)
        adc_sum = np.sum(adc_samples, axis=1)
        data.append({'adc_samples':ma.array(adc_samples, mask=mask), 'adc_sums':adc_sum, 'trigger_type':trigger_type, 'timestamp':tm})

    meta_dict={ 'name':'CTA Data Chunk'}


    return CameraData(Table(data, meta=meta_dict), telescope_id=telescope_id, telescope_name='LST_1', camera_description=geometry)


class CameraData():
    """
    A class containing CTA telescope data. The underlying datastructure is an astropy.table
    This table holds a chunk of events from ONE telescope. The exact contents of the
    events depends on the data model.

    """
    def __init__(self, table, telescope_id, telescope_name, camera_description):
        '''
        Parameters
        ----------
        self: type
            description
        table: astropy.Table
            the underlying table containing the data
        telescope_id:
            the unique id of the telescope
        telescope_name:
            as in LST_1 or similar
        camera_description:
            an instance of the ctapipe.instrument.CameraDescription class.
        '''

        self.table = table
        self.table.meta['telescope_id'] = 1
        self.table.meta['telescope_name'] = telescope_name
        self.table.meta['camera_description'] = camera_description

    #overwrite the attribute access and just return the attribute of the underlying table
    #this way we can add more columns to the astropy.table without changing stuff in here
    def __getattr__(self, key):
        return self.table[key]

    def __len__(self):
        '''
        returns the number of events contained in this chunk.
        '''
        return len(self.table)

    @property
    def telescope_id(self):
        return self.table.meta['telescope_id']

    @property
    def telescope_name(self):
        return self.table.meta['telescope_name']

    @property
    def camera_description(self):
        return self.table.meta['camera_description']


    def __iter__(self):
        for row in self.table:
            yield row
