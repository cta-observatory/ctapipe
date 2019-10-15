import tempfile

import enum
import numpy as np
import pytest
import tables
import pandas as pd
from astropy import units as u

from ctapipe.core.container import Container, Field
from ctapipe.io.containers import (
    R0CameraContainer,
    MCEventContainer,
    HillasParametersContainer,
    LeakageContainer,
)
from ctapipe.io.hdf5tableio import HDF5TableWriter, HDF5TableReader


@pytest.fixture(scope="session")
def temp_h5_file(tmpdir_factory):
    """ a fixture that fetches a temporary output dir/file for a test
    file that we want to read or write (so it doesn't clutter up the test
    directory when the automated tests are run)"""
    return str(tmpdir_factory.mktemp("data").join("test.h5"))


def test_write_container(temp_h5_file):
    r0tel = R0CameraContainer()
    mc = MCEventContainer()
    mc.reset()
    r0tel.waveform = np.random.uniform(size=(50, 10))
    r0tel.meta["test_attribute"] = 3.14159
    r0tel.meta["date"] = "2020-10-10"

    with HDF5TableWriter(
        temp_h5_file, group_name="R0", filters=tables.Filters(complevel=7)
    ) as writer:
        writer.exclude("tel_002", ".*samples")  # test exclusion of columns

        for ii in range(100):
            r0tel.waveform[:] = np.random.uniform(size=(50, 10))
            mc.energy = 10 ** np.random.uniform(1, 2) * u.TeV
            mc.core_x = np.random.uniform(-1, 1) * u.m
            mc.core_y = np.random.uniform(-1, 1) * u.m

            writer.write("tel_001", r0tel)
            writer.write("tel_002", r0tel)  # write a second table too
            writer.write("MC", mc)


def test_prefix(tmp_path):
    tmp_file = tmp_path / "test_prefix.hdf5"
    hillas_parameter_container = HillasParametersContainer(
        x=1 * u.m, y=1 * u.m, length=1 * u.m, width=1 * u.m
    )

    leakage_container = LeakageContainer(
        pixels_width_1=0.1,
        pixels_width_2=0.1,
        intensity_width_1=0.1,
        intensity_width_2=0.1,
    )

    with HDF5TableWriter(tmp_file.name, group_name="blabla", add_prefix=True) as writer:
        writer.write("events", [hillas_parameter_container, leakage_container])

    df = pd.read_hdf(tmp_file.name, key="/blabla/events")
    assert "hillas_x" in df.columns
    assert "leakage_pixels_width_1" in df.columns


def test_write_containers(temp_h5_file):
    class C1(Container):
        a = Field(None, "a")
        b = Field(None, "b")

    class C2(Container):
        c = Field(None, "c")
        d = Field(None, "d")

    with tempfile.NamedTemporaryFile() as f:
        with HDF5TableWriter(f.name, "test") as writer:
            for i in range(20):
                c1 = C1()
                c2 = C2()
                c1.a, c1.b, c2.c, c2.d = np.random.normal(size=4)
                c1.b = np.random.normal()

                writer.write("tel_001", [c1, c2])


def test_write_bool():
    class C(Container):
        boolean = Field(True, 'Boolean value')

    with tempfile.NamedTemporaryFile() as f:
        with HDF5TableWriter(f.name, "test") as writer:
            for i in range(2):
                c = C(boolean=(i % 2 == 0))
                writer.write("c", c)

        c = C()
        with HDF5TableReader(f.name) as reader:
            c_reader = reader.read('/test/c', c)
            for i in range(2):
                cur = next(c_reader)
                expected = (i % 2) == 0
                assert isinstance(cur.boolean, np.bool_)
                assert cur.boolean == expected


def test_write_large_integer():
    class C(Container):
        value = Field(True, 'Integer value')

    exps = [15, 31, 63]
    with tempfile.NamedTemporaryFile() as f:
        with HDF5TableWriter(f.name, "test") as writer:
            for exp in exps:
                c = C(value=2**exp - 1)
                writer.write("c", c)

        c = C()
        with HDF5TableReader(f.name) as reader:
            c_reader = reader.read('/test/c', c)
            for exp in exps:
                cur = next(c_reader)
                assert cur.value == 2**exp - 1


def test_read_container(temp_h5_file):
    r0tel1 = R0CameraContainer()
    r0tel2 = R0CameraContainer()
    mc = MCEventContainer()

    with HDF5TableReader(temp_h5_file) as reader:

        # get the generators for each table
        mctab = reader.read("/R0/MC", mc)
        r0tab1 = reader.read("/R0/tel_001", r0tel1)
        r0tab2 = reader.read("/R0/tel_002", r0tel2)

        # read all 3 tables in sync
        for ii in range(3):

            m = next(mctab)
            r0_1 = next(r0tab1)
            r0_2 = next(r0tab2)

            print("MC:", m)
            print("t0:", r0_1.waveform)
            print("t1:", r0_2.waveform)
            print("---------------------------")

        assert "test_attribute" in r0_1.meta
        assert r0_1.meta["date"] == "2020-10-10"


def test_read_whole_table(temp_h5_file):

    mc = MCEventContainer()

    with HDF5TableReader(temp_h5_file) as reader:

        for cont in reader.read("/R0/MC", mc):
            print(cont)


def test_with_context_writer(temp_h5_file):
    class C1(Container):
        a = Field("a", None)
        b = Field("b", None)

    with tempfile.NamedTemporaryFile() as f:

        with HDF5TableWriter(f.name, "test") as h5_table:

            for i in range(5):
                c1 = C1()
                c1.a, c1.b = np.random.normal(size=2)

                h5_table.write("tel_001", c1)


def test_writer_closes_file(temp_h5_file):

    with tempfile.NamedTemporaryFile() as f:
        with HDF5TableWriter(f.name, "test") as h5_table:

            assert h5_table._h5file.isopen == 1

    assert h5_table._h5file.isopen == 0


def test_reader_closes_file(temp_h5_file):

    with HDF5TableReader(temp_h5_file) as h5_table:

        assert h5_table._h5file.isopen == 1

    assert h5_table._h5file.isopen == 0


def test_with_context_reader(temp_h5_file):

    mc = MCEventContainer()

    with HDF5TableReader(temp_h5_file) as h5_table:

        assert h5_table._h5file.isopen == 1

        for cont in h5_table.read("/R0/MC", mc):
            print(cont)

    assert h5_table._h5file.isopen == 0


def test_closing_reader(temp_h5_file):

    f = HDF5TableReader(temp_h5_file)
    f.close()

    assert f._h5file.isopen == 0


def test_closing_writer(temp_h5_file):

    with tempfile.NamedTemporaryFile() as f:
        h5_table = HDF5TableWriter(f.name, "test")
        h5_table.close()

        assert h5_table._h5file.isopen == 0


def test_cannot_read_with_writer(temp_h5_file):

    with pytest.raises(IOError):

        with HDF5TableWriter(temp_h5_file, "test", mode="r"):
            pass


def test_cannot_write_with_reader(temp_h5_file):

    with HDF5TableReader(temp_h5_file, mode="w") as h5:
        assert h5._h5file.mode == "r"


def test_cannot_append_with_reader(temp_h5_file):

    with HDF5TableReader(temp_h5_file, mode="a") as h5:
        assert h5._h5file.mode == "r"


def test_cannot_r_plus_with_reader(temp_h5_file):

    with HDF5TableReader(temp_h5_file, mode="r+") as h5:
        assert h5._h5file.mode == "r"


def test_append_mode(temp_h5_file):
    class ContainerA(Container):

        a = Field(int)

    a = ContainerA()
    a.a = 1

    # First open with 'w' mode to clear the file and add a Container
    with HDF5TableWriter(temp_h5_file, "group") as h5:

        h5.write("table_1", a)

    # Try to append A again
    with HDF5TableWriter(temp_h5_file, "group", mode="a") as h5:

        h5.write("table_2", a)

    # Check if file has two tables with a = 1
    with HDF5TableReader(temp_h5_file) as h5:

        for a in h5.read("/group/table_1", ContainerA()):

            assert a.a == 1

        for a in h5.read("/group/table_2", ContainerA()):

            assert a.a == 1


def test_write_to_any_location(temp_h5_file):

    loc = "path/path_1"

    class ContainerA(Container):

        a = Field(0, "some integer field")

    a = ContainerA()
    a.a = 1

    with HDF5TableWriter(temp_h5_file, group_name=loc + "/group_1") as h5:

        for _ in range(5):

            h5.write("table", a)
            h5.write("deeper/table2", a)

    with HDF5TableReader(temp_h5_file) as h5:

        for a in h5.read("/" + loc + "/group_1/table", ContainerA()):

            assert a.a == 1

    with HDF5TableReader(temp_h5_file) as h5:

        for a in h5.read("/" + loc + "/group_1/deeper/table2", ContainerA()):

            assert a.a == 1


class WithNormalEnum(Container):
    class EventType(enum.Enum):
        pedestal = 1
        physics = 2
        calibration = 3

    event_type = Field(
        EventType.calibration,
        f"type of event, one of: {list(EventType.__members__.keys())}",
    )


def test_read_write_container_with_enum(tmp_path):
    tmp_file = tmp_path / "container_with_enum.hdf5"

    def create_stream(n_event):
        data = WithNormalEnum()
        for i in range(n_event):
            data.event_type = data.EventType(i % 3 + 1)
            yield data

    with HDF5TableWriter(tmp_file, group_name="data") as h5_table:
        for data in create_stream(10):
            h5_table.write("table", data)

    with HDF5TableReader(tmp_file, mode="r") as h5_table:
        for group_name in ["data/"]:
            group_name = "/{}table".format(group_name)
            for data in h5_table.read(group_name, WithNormalEnum()):
                assert isinstance(data.event_type, WithNormalEnum.EventType)


class WithIntEnum(Container):
    class EventType(enum.IntEnum):
        pedestal = 1
        physics = 2
        calibration = 3

    event_type = Field(
        EventType.calibration,
        f"type of event, one of: {list(EventType.__members__.keys())}",
    )


def test_read_write_container_with_int_enum(tmp_path):
    tmp_file = tmp_path / "container_with_int_enum.hdf5"

    def create_stream(n_event):
        data = WithIntEnum()
        for i in range(n_event):
            data.event_type = data.EventType(i % 3 + 1)
            yield data

    with HDF5TableWriter(tmp_file, group_name="data") as h5_table:
        for data in create_stream(10):
            h5_table.write("table", data)

    with HDF5TableReader(tmp_file, mode="r") as h5_table:
        for group_name in ["data/"]:
            group_name = "/{}table".format(group_name)
            for data in h5_table.read(group_name, WithIntEnum()):
                assert isinstance(data.event_type, WithIntEnum.EventType)


def test_column_transforms(tmp_path):
    """ ensure a user-added column transform is applied """
    tmp_file = tmp_path / "test_column_transforms.hdf5"

    class SomeContainer(Container):
        value = Field(-1, "some value that should be transformed")

    cont = SomeContainer()

    def my_transform(x):
        """ makes a length-3 array from x"""
        return np.ones(3) * x

    with HDF5TableWriter(tmp_file, group_name="data") as writer:
        # add user generated transform for the "value" column
        cont.value = 6.0
        writer.add_column_transform("mytable", "value", my_transform)
        writer.write("mytable", cont)

    # check that we get a length-3 array when reading back
    with HDF5TableReader(tmp_file, mode="r") as reader:
        for data in reader.read("/data/mytable", SomeContainer()):
            print(data)
            assert data.value.shape == (3,)
            assert np.allclose(data.value, [6.0, 6.0, 6.0])


if __name__ == "__main__":

    import logging

    logging.basicConfig(level=logging.DEBUG)

    test_write_container("test.h5")
    test_read_container("test.h5")
    test_read_whole_table("test.h5")
