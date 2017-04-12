from ctapipe.io.containers import R0CameraContainer, MCEventContainer
from ctapipe.io.hdftableio import SimpleHDF5TableWriter
import numpy as np
from astropy import units as u

def test_write_container():

    r0tel = R0CameraContainer()
    mc = MCEventContainer()
    mc.reset()
    r0tel.adc_samples = np.random.uniform(size=(50,10))
    r0tel.adc_sums = np.random.uniform(size=50)
    r0tel.num_samples = 10
    r0tel.meta['test_attribute'] = 3.14159
    r0tel.meta['date'] = "2020-10-10"

    writer = SimpleHDF5TableWriter('test.h5', group_name='R0')

    for ii in range(100):
        r0tel.adc_samples[:] = np.random.uniform(size=(50,10))
        r0tel.adc_sums[:] = np.random.uniform(size=50)
        r0tel.num_samples = 10
        mc.energy = 10**np.random.uniform(1,2) * u.TeV

        writer.write("tel_001", r0tel)
        writer.write("tel_002", r0tel)  # write a second table too
        writer.write("MC", mc)

    return writer

if __name__ == '__main__':

    import logging
    logging.basicConfig(level=logging.DEBUG)

    writer = test_write_container()