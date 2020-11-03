import tempfile

import enum
import numpy as np
import pytest
import tables
import pandas as pd
from astropy import units as u

from ctapipe.core.container import Container, Field
from ctapipe import containers
from ctapipe.containers import (
    R0CameraContainer,
    SimulatedShowerContainer,
    HillasParametersContainer,
    LeakageContainer,
)
from ctapipe.io.hdf5tableio import HDF5TableWriter, HDF5TableReader


@pytest.fixture(scope="session")
def temp_h5_file(tmpdir_factory):
    """a fixture that fetches a temporary output dir/file for a test
    file that we want to read or write (so it doesn't clutter up the test
    directory when the automated tests are run)"""
    return str(tmpdir_factory.mktemp("data").join("test.h5"))


def test_write_container(temp_h5_file):
    r0tel = R0CameraContainer()
    simshower = SimulatedShowerContainer()
    simshower.reset()
    r0tel.waveform = np.random.uniform(size=(50, 10))
    r0tel.meta["test_attribute"] = 3.14159
    r0tel.meta["date"] = "2020-10-10"

    with HDF5TableWriter(
        temp_h5_file, group_name="R0", filters=tables.Filters(complevel=7)
    ) as writer:
        writer.exclude("tel_002", ".*samples")  # test exclusion of columns

        for ii in range(100):
            r0tel.waveform[:] = np.random.uniform(size=(50, 10))
            simshower.energy = 10 ** np.random.uniform(1, 2) * u.TeV
            simshower.core_x = np.random.uniform(-1, 1) * u.m
            simshower.core_y = np.random.uniform(-1, 1) * u.m

            writer.write("tel_001", r0tel)
            writer.write("tel_002", r0tel)  # write a second table too
            writer.write("sim_shower", simshower)


def test_read_multiple_containers():
    hillas_parameter_container = HillasParametersContainer(
        x=1 * u.m, y=1 * u.m, length=1 * u.m, width=1 * u.m
    )

    leakage_container = LeakageContainer(
        pixels_width_1=0.1,
        pixels_width_2=0.1,
        intensity_width_1=0.1,
        intensity_width_2=0.1,
    )
    with tempfile.NamedTemporaryFile() as f:
        with HDF5TableWriter(f.name, group_name="dl1", add_prefix=True) as writer:
            writer.write("params", [hillas_parameter_container, leakage_container])

        df = pd.read_hdf(f.name, key="/dl1/params")
        assert "hillas_x" in df.columns
        assert "leakage_pixels_width_1" in df.columns

        # test reading both containers separately
        with HDF5TableReader(f.name) as reader:
            generator = reader.read(
                "/dl1/params", HillasParametersContainer(), prefixes=True
            )
            hillas = next(generator)
        for value, read_value in zip(
            hillas_parameter_container.as_dict().values(), hillas.as_dict().values()
        ):
            np.testing.assert_equal(value, read_value)

        with HDF5TableReader(f.name) as reader:
            generator = reader.read("/dl1/params", LeakageContainer(), prefixes=True)
            leakage = next(generator)
        for value, read_value in zip(
            leakage_container.as_dict().values(), leakage.as_dict().values()
        ):
            np.testing.assert_equal(value, read_value)

        # test reading both containers simultaneously
        with HDF5TableReader(f.name) as reader:
            generator = reader.read(
                "/dl1/params",
                [HillasParametersContainer(), LeakageContainer()],
                prefixes=True,
            )
            hillas_, leakage_ = next(generator)

        for value, read_value in zip(
            leakage_container.as_dict().values(), leakage_.as_dict().values()
        ):
            np.testing.assert_equal(value, read_value)

        for value, read_value in zip(
            hillas_parameter_container.as_dict().values(), hillas_.as_dict().values()
        ):
            np.testing.assert_equal(value, read_value)


def test_read_without_prefixes():
    hillas_parameter_container = HillasParametersContainer(
        x=1 * u.m, y=1 * u.m, length=1 * u.m, width=1 * u.m
    )

    leakage_container = LeakageContainer(
        pixels_width_1=0.1,
        pixels_width_2=0.1,
        intensity_width_1=0.1,
        intensity_width_2=0.1,
    )
    with tempfile.NamedTemporaryFile() as f:
        with HDF5TableWriter(f.name, group_name="dl1", add_prefix=False) as writer:
            writer.write("params", [hillas_parameter_container, leakage_container])

        df = pd.read_hdf(f.name, key="/dl1/params")
        assert "x" in df.columns
        assert "pixels_width_1" in df.columns

        # call with prefixes=False
        with HDF5TableReader(f.name) as reader:
            generator = reader.read(
                "/dl1/params",
                [HillasParametersContainer(), LeakageContainer()],
                prefixes=False,
            )
            hillas_, leakage_ = next(generator)

        for value, read_value in zip(
            leakage_container.as_dict().values(), leakage_.as_dict().values()
        ):
            np.testing.assert_equal(value, read_value)

        for value, read_value in zip(
            hillas_parameter_container.as_dict().values(), hillas_.as_dict().values()
        ):
            np.testing.assert_equal(value, read_value)

        # call with manually removed prefixes
        with HDF5TableReader(f.name) as reader:
            generator = reader.read(
                "/dl1/params",
                [HillasParametersContainer(prefix=""), LeakageContainer(prefix="")],
                prefixes=True,
            )
            hillas_, leakage_ = next(generator)

        for value, read_value in zip(
            leakage_container.as_dict().values(), leakage_.as_dict().values()
        ):
            np.testing.assert_equal(value, read_value)

        for value, read_value in zip(
            hillas_parameter_container.as_dict().values(), hillas_.as_dict().values()
        ):
            np.testing.assert_equal(value, read_value)


def test_read_duplicated_container_types():
    hillas_config_1 = HillasParametersContainer(
        x=1 * u.m, y=2 * u.m, length=3 * u.m, width=4 * u.m, prefix="hillas_1"
    )
    hillas_config_2 = HillasParametersContainer(
        x=2 * u.m, y=3 * u.m, length=4 * u.m, width=5 * u.m, prefix="hillas_2"
    )

    with tempfile.NamedTemporaryFile() as f:
        with HDF5TableWriter(f.name, group_name="dl1", add_prefix=True) as writer:
            writer.write("params", [hillas_config_1, hillas_config_2])

        df = pd.read_hdf(f.name, key="/dl1/params")
        assert "hillas_1_x" in df.columns
        assert "hillas_2_x" in df.columns

        with HDF5TableReader(f.name) as reader:
            generator = reader.read(
                "/dl1/params",
                [HillasParametersContainer(), HillasParametersContainer()],
                prefixes=["hillas_1", "hillas_2"],
            )
            hillas_1, hillas_2 = next(generator)

        for value, read_value in zip(
            hillas_config_1.as_dict().values(), hillas_1.as_dict().values()
        ):
            np.testing.assert_equal(value, read_value)

        for value, read_value in zip(
            hillas_config_2.as_dict().values(), hillas_2.as_dict().values()
        ):
            np.testing.assert_equal(value, read_value)


def test_custom_prefix():
    container = HillasParametersContainer(
        x=1 * u.m, y=1 * u.m, length=1 * u.m, width=1 * u.m
    )
    container.prefix = "custom"
    with tempfile.NamedTemporaryFile() as f:
        with HDF5TableWriter(f.name, group_name="dl1", add_prefix=True) as writer:
            writer.write("params", container)

        with HDF5TableReader(f.name) as reader:
            generator = reader.read(
                "/dl1/params", HillasParametersContainer(), prefixes="custom"
            )
            read_container = next(generator)
        assert isinstance(read_container, HillasParametersContainer)
        for value, read_value in zip(
            container.as_dict().values(), read_container.as_dict().values()
        ):
            np.testing.assert_equal(value, read_value)


def test_units():
    class WithUnits(Container):
        inverse_length = Field(5 / u.m, "foo")
        time = Field(1 * u.s, "bar", unit=u.s)
        grammage = Field(2 * u.g / u.cm ** 2, "baz", unit=u.g / u.cm ** 2)

    c = WithUnits()

    with tempfile.NamedTemporaryFile() as f:
        with HDF5TableWriter(f.name, "data") as writer:
            writer.write("units", c)

        with tables.open_file(f.name, "r") as f:

            assert f.root.data.units.attrs["inverse_length_UNIT"] == "m-1"
            assert f.root.data.units.attrs["time_UNIT"] == "s"
            assert f.root.data.units.attrs["grammage_UNIT"] == "cm-2 g"


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
        boolean = Field(True, "Boolean value")

    with tempfile.NamedTemporaryFile() as f:
        with HDF5TableWriter(f.name, "test") as writer:
            for i in range(2):
                c = C(boolean=(i % 2 == 0))
                writer.write("c", c)

        c = C()
        with HDF5TableReader(f.name) as reader:
            c_reader = reader.read("/test/c", c)
            for i in range(2):
                cur = next(c_reader)
                expected = (i % 2) == 0
                assert isinstance(cur.boolean, np.bool_)
                assert cur.boolean == expected


def test_write_large_integer():
    class C(Container):
        value = Field(True, "Integer value")

    exps = [15, 31, 63]
    with tempfile.NamedTemporaryFile() as f:
        with HDF5TableWriter(f.name, "test") as writer:
            for exp in exps:
                c = C(value=2 ** exp - 1)
                writer.write("c", c)

        c = C()
        with HDF5TableReader(f.name) as reader:
            c_reader = reader.read("/test/c", c)
            for exp in exps:
                cur = next(c_reader)
                assert cur.value == 2 ** exp - 1


def test_read_container(temp_h5_file):
    r0tel1 = R0CameraContainer()
    r0tel2 = R0CameraContainer()
    sim_shower = SimulatedShowerContainer()

    with HDF5TableReader(temp_h5_file) as reader:

        # get the generators for each table
        # test supplying a single container as well as an
        # iterable with one entry only
        simtab = reader.read("/R0/sim_shower", (sim_shower,))
        r0tab1 = reader.read("/R0/tel_001", r0tel1)
        r0tab2 = reader.read("/R0/tel_002", r0tel2)

        # read all 3 tables in sync
        for ii in range(3):

            m = next(simtab)[0]
            r0_1 = next(r0tab1)
            r0_2 = next(r0tab2)

            print("sim_shower:", m)
            print("t0:", r0_1.waveform)
            print("t1:", r0_2.waveform)
            print("---------------------------")

        assert "test_attribute" in r0_1.meta
        assert r0_1.meta["date"] == "2020-10-10"


def test_read_whole_table(temp_h5_file):

    sim_shower = SimulatedShowerContainer()

    with HDF5TableReader(temp_h5_file) as reader:

        for cont in reader.read("/R0/sim_shower", sim_shower):
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

    sim_shower = SimulatedShowerContainer()

    with HDF5TableReader(temp_h5_file) as h5_table:

        assert h5_table._h5file.isopen == 1

        for cont in h5_table.read("/R0/sim_shower", sim_shower):
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

    foo = Field(5, "foo field to test if this still works with more fields")

    event_type = Field(
        EventType.calibration,
        f"type of event, one of: {list(EventType.__members__.keys())}",
    )

    bar = Field(10, "bar field to test if this still works with more fields")


def test_read_write_container_with_enum(tmp_path):
    tmp_file = tmp_path / "container_with_enum.hdf5"

    def create_stream(n_event):
        data = WithNormalEnum()
        for i in range(n_event):
            data.event_type = data.EventType(i % 3 + 1)
            data.foo = i
            data.bar = i * 2
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


def test_filters():
    from tables import Filters, open_file

    class TestContainer(Container):
        value = Field(-1, "test")

    no_comp = Filters(complevel=0)
    zstd = Filters(complevel=5, complib="blosc:zstd")

    with tempfile.NamedTemporaryFile(suffix=".hdf5") as f:
        with HDF5TableWriter(
            f.name, group_name="data", mode="w", filters=no_comp
        ) as writer:
            assert writer._h5file.filters.complevel == 0

            c = TestContainer(value=5)
            writer.write("default", c)

            writer.filters = zstd
            writer.write("zstd", c)

            writer.filters = no_comp
            writer.write("nocomp", c)

        with open_file(f.name) as h5file:
            assert h5file.root.data.default.filters.complevel == 0
            assert h5file.root.data.zstd.filters.complevel == 5
            assert h5file.root.data.zstd.filters.complib == "blosc:zstd"
            assert h5file.root.data.nocomp.filters.complevel == 0


def test_column_order():
    """ Test that columns are written in the order the containers define them"""

    class Container1(Container):
        b = Field(1, "b")
        a = Field(2, "a")

    class Container2(Container):
        d = Field(3, "d")
        c = Field(4, "c")

    # test with single container
    with tempfile.NamedTemporaryFile(suffix=".hdf5") as f:
        with HDF5TableWriter(f.name, mode="w") as writer:
            c = Container1()
            writer.write("foo", c)

        with tables.open_file(f.name, "r") as f:
            assert f.root.foo[:].dtype.names == ("b", "a")

    # test with two containers
    with tempfile.NamedTemporaryFile(suffix=".hdf5") as f:
        with HDF5TableWriter(f.name, mode="w") as writer:
            c1 = Container1()
            c2 = Container2()
            writer.write("foo", [c2, c1])
            writer.write("bar", [c1, c2])

        with tables.open_file(f.name, "r") as f:
            assert f.root.foo[:].dtype.names == ("d", "c", "b", "a")
            assert f.root.bar[:].dtype.names == ("b", "a", "d", "c")


def test_writing_nan_defaults():
    from ctapipe.containers import ImageParametersContainer

    params = ImageParametersContainer()

    with tempfile.NamedTemporaryFile(suffix=".hdf5") as f:
        with HDF5TableWriter(f.name, mode="w") as writer:
            writer.write("params", params.values())


ALL_CONTAINERS = []
for name in dir(containers):
    try:
        obj = getattr(containers, name)
        if issubclass(obj, Container):
            ALL_CONTAINERS.append(obj)
    except TypeError:
        pass


@pytest.mark.parametrize("cls", ALL_CONTAINERS)
def test_write_default_container(cls):

    with tempfile.NamedTemporaryFile(suffix=".hdf5") as f:
        with HDF5TableWriter(f.name, mode="w") as writer:
            try:
                writer.write("params", cls())
            except ValueError as e:
                # some containers do not have writable members,
                # only subcontainers. For now, ignore them.
                if "cannot create an empty data type" in str(e):
                    pytest.xfail()
                else:
                    raise


if __name__ == "__main__":

    import logging

    logging.basicConfig(level=logging.DEBUG)

    test_write_container("test.h5")
    test_read_container("test.h5")
    test_read_whole_table("test.h5")
