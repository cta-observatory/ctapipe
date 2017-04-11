from ctapipe.io.containers import R0CameraContainer
from ctapipe.io.hdfserializer import SimpleHDF5TableWriter
import numpy as np

def test_write_container():

    r0tel = R0CameraContainer()
    r0tel.adc_samples = np.random.uniform(size=(50,10))
    r0tel.adc_sums = np.random.uniform(size=50)
    r0tel.num_samples = 10


    writer = SimpleHDF5TableWriter('test.h5', group_name='R0')


    for ii in range(10):
        r0tel.adc_samples[:] = np.random.uniform(size=(50,10))
        r0tel.adc_sums[:] = np.random.uniform(size=50)

        writer.write("tel_001", r0tel)
        writer.write("tel_002", r0tel)  # write a second table too
