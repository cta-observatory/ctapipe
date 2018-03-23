import tempfile

import numpy as np
import pytest
import tables
from astropy import units as u

from ctapipe.core.container import Container, Field
from ctapipe.io.containers import R0CameraContainer, MCEventContainer
from ctapipe.io.hdf5tableio import HDF5TableWriter, HDF5TableReader


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
    r0tel.waveform = np.random.uniform(size=(50, 10))
    r0tel.image = np.random.uniform(size=50)
    r0tel.num_samples = 10
    r0tel.meta['test_attribute'] = 3.14159
    r0tel.meta['date'] = "2020-10-10"

    writer = HDF5TableWriter(str(temp_h5_file), group_name='R0',
                             filters=tables.Filters(
        complevel=7))
    writer.exclude("tel_002", ".*samples")  # test exclusion of columns

    for ii in range(100):
        r0tel.waveform[:] = np.random.uniform(size=(50, 10))
        r0tel.image[:] = np.random.uniform(size=50)
        r0tel.num_samples = 10
        mc.energy = 10**np.random.uniform(1, 2) * u.TeV
        mc.core_x = np.random.uniform(-1, 1) * u.m
        mc.core_y = np.random.uniform(-1, 1) * u.m

        writer.write("tel_001", r0tel)
        writer.write("tel_002", r0tel)  # write a second table too
        writer.write("MC", mc)

    writer.close()


def test_write_containers(temp_h5_file):

    class C1(Container):
        a = Field('a', None)
        b = Field('b', None)

    class C2(Container):
        c = Field('c', None)
        d = Field('d', None)

    with tempfile.NamedTemporaryFile() as f:
        writer = HDF5TableWriter(f.name, 'test')
        for i in range(20):
            c1 = C1()
            c2 = C2()
            c1.a, c1.b, c2.c, c2.d = np.random.normal(size=4)
            c1.b = np.random.normal()

            writer.write("tel_001", [c1, c2])

        writer.close()


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

        m = next(mctab)
        r0_1 = next(r0tab1)
        r0_2 = next(r0tab2)

        print("MC:", m)
        print("t0:", r0_1.image)
        print("t1:", r0_2.image)
        print("---------------------------")

    assert 'test_attribute' in r0_1.meta
    assert r0_1.meta['date'] == "2020-10-10"

    reader.close()


def test_read_whole_table(temp_h5_file):

    mc = MCEventContainer()

    reader = HDF5TableReader(str(temp_h5_file))

    for cont in reader.read('/R0/MC', mc):
        print(cont)

    reader.close()


def test_with_context_writer(temp_h5_file):

    class C1(Container):
        a = Field('a', None)
        b = Field('b', None)

    with tempfile.NamedTemporaryFile() as f:

        with HDF5TableWriter(f.name, 'test') as h5_table:

            for i in range(5):
                c1 = C1()
                c1.a, c1.b = np.random.normal(size=2)

                h5_table.write("tel_001", c1)


def test_writer_closes_file(temp_h5_file):

    with tempfile.NamedTemporaryFile() as f:
        with HDF5TableWriter(f.name, 'test') as h5_table:

            assert h5_table._h5file.isopen == True

    assert h5_table._h5file.isopen == False


def test_reader_closes_file(temp_h5_file):

    with HDF5TableReader(str(temp_h5_file)) as h5_table:

        assert h5_table._h5file.isopen == True

    assert h5_table._h5file.isopen == False


def test_with_context_reader(temp_h5_file):

    mc = MCEventContainer()

    with HDF5TableReader(str(temp_h5_file)) as h5_table:

        assert h5_table._h5file.isopen == True

        for cont in h5_table.read('/R0/MC', mc):
            print(cont)

    assert h5_table._h5file.isopen == False


def test_closing_reader(temp_h5_file):

    f = HDF5TableReader(str(temp_h5_file))
    f.close()


def test_closing_writer(temp_h5_file):

    with tempfile.NamedTemporaryFile() as f:
        h5_table = HDF5TableWriter(f.name, 'test')
        h5_table.close()


def test_cannot_read_with_writer(temp_h5_file):

    with pytest.raises(IOError):

        with HDF5TableWriter(temp_h5_file, 'test', mode='r'):

            pass


def test_cannot_write_with_reader(temp_h5_file):

    with HDF5TableReader(temp_h5_file, mode='w') as h5:

        assert h5._h5file.mode == 'r'


def test_cannot_append_with_reader(temp_h5_file):

    with HDF5TableReader(temp_h5_file, mode='a') as h5:

        assert h5._h5file.mode == 'r'


def test_cannot_r_plus_with_reader(temp_h5_file):

    with HDF5TableReader(temp_h5_file, mode='r+') as h5:

        assert h5._h5file.mode == 'r'


def test_append_mode(temp_h5_file):

    class ContainerA(Container):

        a = Field(int)

    a = ContainerA()
    a.a = 1

    # First open with 'w' mode to clear the file and add a Container
    with HDF5TableWriter(temp_h5_file, 'group') as h5:

        h5.write('table_1', a)

    # Try to append A again
    with HDF5TableWriter(temp_h5_file, 'group', mode='a') as h5:

        h5.write('table_2', a)

    # Check if file has two tables with a = 1
    with HDF5TableReader(temp_h5_file) as h5:

        for a in h5.read('/group/table_1', ContainerA()):

            assert a.a == 1

        for a in h5.read('/group/table_2', ContainerA()):

            assert a.a == 1


@pytest.mark.xfail
def test_write_to_any_location(temp_h5_file):

    loc = '/path/path_1'

    class ContainerA(Container):

        a = Field(int)

    a = ContainerA()
    a.a = 1

    with HDF5TableWriter(temp_h5_file, 'group_1', root_uep=loc) as h5:

        for _ in range(5):

            h5.write('table', a)

    with HDF5TableReader(temp_h5_file) as h5:

        for a in h5.read(loc + '/group_1/table', ContainerA()):

            assert a.a == 1


if __name__ == '__main__':

    import logging
    logging.basicConfig(level=logging.DEBUG)

    test_write_container("test.h5")
    test_read_container("test.h5")
    test_read_whole_table("test.h5")
