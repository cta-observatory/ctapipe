from ctapipe.io.containers import R0CameraContainer, MCEventContainer
from ctapipe.io.hdftableio import HDF5TableWriter, HDF5TableReader
import numpy as np
from astropy import units as u
import tables
import pytest

@pytest.fixture(scope='session')
def temp_h5_file(tmpdir_factory):
    """ a fixture that fetches a temporary output dir/file for a test
    file that we want to read or write (so it doesn't clutter up the test 
    directory when the automated tests are run)"""
    return str(tmpdir_factory.mktemp('data').join('test.h5'))


def test_write_container(temp_h5_file):
    r0tel = R0CameraContainer()
    mc = MCEventContainer()
    mc.reset()
    r0tel.adc_samples = np.random.uniform(size=(50, 10))
    r0tel.adc_sums = np.random.uniform(size=50)
    r0tel.num_samples = 10
    r0tel.meta['test_attribute'] = 3.14159
    r0tel.meta['date'] = "2020-10-10"

    writer = HDF5TableWriter(str(temp_h5_file), group_name='R0',
                             filters=tables.Filters(
        complevel=7))
    writer.exclude("tel_002",".*samples")  # test exclusion of columns

    for ii in range(100):
        r0tel.adc_samples[:] = np.random.uniform(size=(50, 10))
        r0tel.adc_sums[:] = np.random.uniform(size=50)
        r0tel.num_samples = 10
        mc.energy = 10**np.random.uniform(1,2) * u.TeV
        mc.core_x = np.random.uniform(-1, 1) * u.m
        mc.core_y = np.random.uniform(-1, 1) * u.m

        writer.write("tel_001", r0tel)
        writer.write("tel_002", r0tel)  # write a second table too
        writer.write("MC", mc)


def test_read_container(temp_h5_file):
    r0tel1 = R0CameraContainer()
    r0tel2 = R0CameraContainer()
    mc = MCEventContainer()

    reader = HDF5TableReader(str(temp_h5_file))

    # get the generators for each table
    mctab = reader.read('/R0/MC', mc)
    r0tab1 = reader.read('/R0/tel_001', r0tel1)
    r0tab2 = reader.read('/R0/tel_002', r0tel2)

    # read all 3 tables in sync
    for ii in range(3):
        print("MC:", next(mctab))
        print("t0:", next(r0tab1).adc_sums)
        print("t1:", next(r0tab2).adc_sums)
        print("---------------------------")

def test_read_whole_table(temp_h5_file):

    mc = MCEventContainer()

    reader = HDF5TableReader(str(temp_h5_file))

    for cont in reader.read('/R0/MC', mc):
        print(cont)

if __name__ == '__main__':

    import logging
    logging.basicConfig(level=logging.DEBUG)

    test_write_container("test.h5")
    test_read_container("test.h5")
    test_read_whole_table("test.h5")