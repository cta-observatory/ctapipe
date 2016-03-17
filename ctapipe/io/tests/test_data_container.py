from astropy.table import Table
from astropy.time import Time
from ..camera import make_rectangular_camera_geometry
from ..data_container import CameraData

import numpy as np
from numpy import ma


def test_iteration():
    #create 1 chunk and iterate over events
    for event in create_dummy_chunk():
        print(event)

def test_length():
    #test whether the CameraData class overwrites __len__ correctly
    chunk = create_dummy_chunk(N=128)
    assert len(chunk) == 128



def test_dimensions():
    N =10
    pixels = 40*40
    samples = 50
    chunk = create_dummy_chunk(N=N, pixels=pixels, samples=samples)
    chunk.table.pprint(max_width=-1)

    assert chunk.adc_sums.shape == (N, pixels), 'adc_sums does not have the correct dimensions.'

    assert chunk.adc_samples.shape == (N, pixels, samples), 'adc_samples does not have the correct dimensions.'

def create_dummy_chunk(N=10, pixels=1440, samples=50):
    data = []
    tm = Time(['2000:002', '2002:345'])
    telescope_id = 1
    geometry = make_rectangular_camera_geometry()
    for eventnumber in range(N):
        adc_samples = np.random.normal(loc=0, scale=1, size=(pixels, samples))
        mask = np.random.randint(low=0, high=1, size=(pixels, samples))


        trigger_type = np.random.randint(low=0, high=10)
        adc_sum = np.sum(adc_samples, axis=1)
        data.append({'adc_samples':ma.array(adc_samples, mask=mask), 'adc_sums':adc_sum, 'trigger_type':trigger_type, 'timestamp':tm})

    meta_dict={ 'telescope_id':telescope_id,
                'name':'CTA Data Chunk',
                'telescope_name':'LST_1',
                'telescope_type':'LST',
                'geometry':geometry}

    return CameraData(Table(data, meta=meta_dict))
