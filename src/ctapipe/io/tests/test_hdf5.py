import enum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tables
from astropy import units as u
from astropy.time import Time

from ctapipe import containers
from ctapipe.containers import (
    HillasParametersContainer,
    LeakageContainer,
    R0CameraContainer,
    SimulatedShowerContainer,
    TelEventIndexContainer,
)
from ctapipe.core.container import Container, Field
from ctapipe.io import read_table
from ctapipe.io.datalevels import DataLevel
from ctapipe.io.hdf5tableio import HDF5TableReader, HDF5TableWriter


@pytest.fixture(scope="session")
def test_h5_file(tmp_path_factory):
    """Test hdf5 file with some tables for the reader tests"""
    path = tmp_path_factory.mktemp("hdf5") / "test.h5"

    r0 = R0CameraContainer()
    shower = SimulatedShowerContainer()
    r0.waveform = np.random.uniform(size=(50, 10))
    r0.meta["test_attribute"] = 3.14159
    r0.meta["date"] = "2020-10-10"

    with HDF5TableWriter(
        path, group_name="R0", filters=tables.Filters(complevel=7)
    ) as writer:
        for _ in range(100):
            r0.waveform[:] = np.random.uniform(size=(50, 10))
            shower.energy = 10 ** np.random.uniform(1, 2) * u.TeV
            shower.core_x = np.random.uniform(-1, 1) * u.m
            shower.core_y = np.random.uniform(-1, 1) * u.m

            writer.write("tel_001", r0)
            writer.write("tel_002", r0)  # write a second table too
            writer.write("sim_shower", shower)

    return path


def test_read_meta(test_h5_file):
    """Test reading meta information"""
    from ctapipe import __version__
    from ctapipe.io.hdf5tableio import get_node_meta

    with tables.open_file(test_h5_file, "r") as f:
        meta = get_node_meta(f.root["/R0/tel_001"])

        # check we don't have anything else
        # system attributes and column metadata should be excluded
        assert len(meta) == 3
        assert isinstance(meta["CTAPIPE_VERSION"], str)
        assert meta["CTAPIPE_VERSION"] == __version__

        assert isinstance(meta["date"], str)
        assert meta["date"] == "2020-10-10"

        assert isinstance(meta["test_attribute"], float)
        assert meta["test_attribute"] == 3.14159


def test_read_column_attrs(test_h5_file):
    """Test reading meta information"""
    from ctapipe.io.hdf5tableio import get_column_attrs

    with tables.open_file(test_h5_file, "r") as f:
        column_attrs = get_column_attrs(f.root["/R0/sim_shower"])
        assert len(column_attrs) == len(SimulatedShowerContainer.fields)
        assert column_attrs["energy"]["POS"] == 0
        assert column_attrs["energy"]["TRANSFORM"] == "quantity"
        assert column_attrs["energy"]["UNIT"] == "TeV"
        assert column_attrs["energy"]["DTYPE"] == np.float64

        assert column_attrs["alt"]["POS"] == 1
        assert column_attrs["alt"]["TRANSFORM"] == "quantity"
        assert column_attrs["alt"]["UNIT"] == "deg"
        assert column_attrs["alt"]["DTYPE"] == np.float64


def test_append_container(tmp_path):
    path = tmp_path / "test_append.h5"
    with HDF5TableWriter(path, mode="w") as writer:
        for event_id in range(10):
            hillas = HillasParametersContainer()
            index = TelEventIndexContainer(obs_id=1, event_id=event_id, tel_id=1)
            writer.write("data", [index, hillas])

    with HDF5TableWriter(path, mode="a") as writer:
        for event_id in range(10):
            index = TelEventIndexContainer(obs_id=2, event_id=event_id, tel_id=1)
            hillas = HillasParametersContainer()
            writer.write("data", [index, hillas])

    table = read_table(path, "/data")
    assert len(table) == 20
    assert np.all(table["obs_id"] == np.repeat([1, 2], 10))
    assert np.all(table["event_id"] == np.tile(np.arange(10), 2))


def test_reader_container_reuse(test_h5_file):
    """Test the reader does not reuse the same container instance"""
    with HDF5TableReader(test_h5_file) as reader:
        iterator = reader.read("/R0/sim_shower", SimulatedShowerContainer)
        assert next(iterator) is not next(iterator)


def test_read_multiple_containers(tmp_path):
    path = tmp_path / "test_append.h5"
    hillas_parameter_container = HillasParametersContainer(
        fov_lon=1 * u.deg, fov_lat=1 * u.deg, length=1 * u.deg, width=1 * u.deg
    )

    leakage_container = LeakageContainer(
        pixels_width_1=0.1,
        pixels_width_2=0.1,
        intensity_width_1=0.1,
        intensity_width_2=0.1,
    )
    with HDF5TableWriter(path, group_name="dl1", add_prefix=True) as writer:
        writer.write("params", [hillas_parameter_container, leakage_container])

    df = pd.read_hdf(path, key="/dl1/params")
    assert "hillas_fov_lon" in df.columns
    assert "leakage_pixels_width_1" in df.columns

    # test reading both containers separately
    with HDF5TableReader(path) as reader:
        generator = reader.read("/dl1/params", HillasParametersContainer, prefixes=True)
        hillas = next(generator)
    for value, read_value in zip(
        hillas_parameter_container.as_dict().values(), hillas.as_dict().values()
    ):
        np.testing.assert_equal(value, read_value)

    with HDF5TableReader(path) as reader:
        generator = reader.read("/dl1/params", LeakageContainer, prefixes=True)
        leakage = next(generator)
    for value, read_value in zip(
        leakage_container.as_dict().values(), leakage.as_dict().values()
    ):
        np.testing.assert_equal(value, read_value)

    # test reading both containers simultaneously
    with HDF5TableReader(path) as reader:
        generator = reader.read(
            "/dl1/params",
            (HillasParametersContainer, LeakageContainer),
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


def test_read_without_prefixes(tmp_path):
    path = tmp_path / "test.h5"

    hillas_parameter_container = HillasParametersContainer(
        fov_lon=1 * u.deg, fov_lat=1 * u.deg, length=1 * u.deg, width=1 * u.deg
    )

    leakage_container = LeakageContainer(
        pixels_width_1=0.1,
        pixels_width_2=0.1,
        intensity_width_1=0.1,
        intensity_width_2=0.1,
    )

    with HDF5TableWriter(path, group_name="dl1", add_prefix=False) as writer:
        writer.write("params", (hillas_parameter_container, leakage_container))

    df = pd.read_hdf(path, key="/dl1/params")
    assert "fov_lon" in df.columns
    assert "pixels_width_1" in df.columns

    # call with prefixes=False
    with HDF5TableReader(path) as reader:
        generator = reader.read(
            "/dl1/params",
            (HillasParametersContainer, LeakageContainer),
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
    with HDF5TableReader(path) as reader:
        generator = reader.read(
            "/dl1/params",
            (HillasParametersContainer, LeakageContainer),
            prefixes=["", ""],
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


def test_read_duplicated_container_types(tmp_path):
    path = tmp_path / "test.h5"

    hillas_config_1 = HillasParametersContainer(
        fov_lon=1 * u.deg,
        fov_lat=2 * u.deg,
        length=3 * u.deg,
        width=4 * u.deg,
        prefix="hillas_1",
    )
    hillas_config_2 = HillasParametersContainer(
        fov_lon=2 * u.deg,
        fov_lat=3 * u.deg,
        length=4 * u.deg,
        width=5 * u.deg,
        prefix="hillas_2",
    )

    with HDF5TableWriter(path, group_name="dl1", add_prefix=True) as writer:
        writer.write("params", [hillas_config_1, hillas_config_2])

    df = pd.read_hdf(path, key="/dl1/params")
    assert "hillas_1_fov_lon" in df.columns
    assert "hillas_2_fov_lon" in df.columns

    with HDF5TableReader(path) as reader:
        generator = reader.read(
            "/dl1/params",
            (HillasParametersContainer, HillasParametersContainer),
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


def test_custom_prefix(tmp_path):
    path = tmp_path / "test.h5"

    container = HillasParametersContainer(
        fov_lon=1 * u.deg, fov_lat=1 * u.deg, length=1 * u.deg, width=1 * u.deg
    )
    container.prefix = "custom"
    with HDF5TableWriter(path, group_name="dl1", add_prefix=True) as writer:
        writer.write("params", container)

    with HDF5TableReader(path) as reader:
        generator = reader.read(
            "/dl1/params", HillasParametersContainer, prefixes="custom"
        )
        read_container = next(generator)
    assert isinstance(read_container, HillasParametersContainer)
    for value, read_value in zip(
        container.as_dict().values(), read_container.as_dict().values()
    ):
        np.testing.assert_equal(value, read_value)


def test_units(tmp_path):
    path = tmp_path / "test.h5"

    class WithUnits(Container):
        inverse_length = Field(5 / u.m, "foo")
        time = Field(1 * u.s, "bar", unit=u.s)
        grammage = Field(2 * u.g / u.cm**2, "baz", unit=u.g / u.cm**2)

    c = WithUnits()

    with HDF5TableWriter(path, "data") as writer:
        writer.write("units", c)

    with tables.open_file(path, "r") as f:
        assert f.root.data.units.attrs["CTAFIELD_0_UNIT"] == "m**-1"
        assert f.root.data.units.attrs["CTAFIELD_1_UNIT"] == "s"
        # order of the units does not matter
        assert f.root.data.units.attrs["CTAFIELD_2_UNIT"] in {"cm**-2.g", "g.cm**-2"}


def test_write_containers(tmp_path):
    class C1(Container):
        a = Field(None, "a")
        b = Field(None, "b")

    class C2(Container):
        c = Field(None, "c")
        d = Field(None, "d")

    with HDF5TableWriter(tmp_path / "test.h5", "test") as writer:
        for _ in range(20):
            c1 = C1()
            c2 = C2()
            c1.a, c1.b, c2.c, c2.d = np.random.normal(size=4)
            writer.write("tel_001", [c1, c2])


def test_write_bool(tmp_path):
    path = tmp_path / "test.h5"

    class C(Container):
        boolean = Field(True, "Boolean value")

    with HDF5TableWriter(path, "test") as writer:
        for i in range(2):
            c = C(boolean=(i % 2 == 0))
            writer.write("c", c)

    c = C()
    with HDF5TableReader(path) as reader:
        c_reader = reader.read("/test/c", C)
        for i in range(2):
            cur = next(c_reader)
            expected = (i % 2) == 0
            assert isinstance(cur.boolean, np.bool_)
            assert cur.boolean == expected


def test_write_large_integer(tmp_path):
    path = tmp_path / "test.h5"

    class C(Container):
        value = Field(True, "Integer value")

    exps = [15, 31, 63]
    with HDF5TableWriter(path, "test") as writer:
        for exp in exps:
            c = C(value=2**exp - 1)
            writer.write("c", c)

    c = C()
    with HDF5TableReader(path) as reader:
        c_reader = reader.read("/test/c", C)
        for exp in exps:
            cur = next(c_reader)
            assert cur.value == 2**exp - 1


def test_read_container(test_h5_file):
    with HDF5TableReader(test_h5_file) as reader:
        # get the generators for each table
        # test supplying a single container as well as an
        # iterable with one entry only
        simtab = reader.read("/R0/sim_shower", (SimulatedShowerContainer,))
        r0tab1 = reader.read("/R0/tel_001", R0CameraContainer)
        r0tab2 = reader.read("/R0/tel_002", R0CameraContainer)

        # read all 3 tables in sync
        for _ in range(3):
            m = next(simtab)[0]
            r0_1 = next(r0tab1)
            r0_2 = next(r0tab2)

            print("sim_shower:", m)
            print("t0:", r0_1.waveform)
            print("t1:", r0_2.waveform)
            print("---------------------------")

        assert "test_attribute" in r0_1.meta
        assert r0_1.meta["date"] == "2020-10-10"


def test_read_whole_table(test_h5_file):
    with HDF5TableReader(test_h5_file) as reader:
        for cont in reader.read("/R0/sim_shower", SimulatedShowerContainer):
            print(cont)


def test_with_context_writer(tmp_path):
    path = tmp_path / "test.h5"

    class C1(Container):
        a = Field("a", None)
        b = Field("b", None)

    with HDF5TableWriter(path, "test") as h5_table:
        for i in range(5):
            c1 = C1()
            c1.a, c1.b = np.random.normal(size=2)

            h5_table.write("tel_001", c1)


def test_writer_closes_file(tmp_path):
    with HDF5TableWriter(tmp_path / "test.h5", "test") as h5_table:
        assert h5_table.h5file.isopen == 1

    assert h5_table.h5file.isopen == 0


def test_reader_closes_file(test_h5_file):
    with HDF5TableReader(test_h5_file) as h5_table:
        assert h5_table._h5file.isopen == 1

    assert h5_table._h5file.isopen == 0


def test_with_context_reader(test_h5_file):
    with HDF5TableReader(test_h5_file) as h5_table:
        assert h5_table._h5file.isopen == 1

        for cont in h5_table.read("/R0/sim_shower", SimulatedShowerContainer):
            print(cont)

    assert h5_table._h5file.isopen == 0


def test_closing_reader(test_h5_file):
    f = HDF5TableReader(test_h5_file)
    f.close()

    assert f._h5file.isopen == 0


def test_closing_writer(tmp_path):
    h5_table = HDF5TableWriter(tmp_path / "test.h5", "test")
    h5_table.close()

    assert h5_table.h5file.isopen == 0


def test_cannot_read_with_writer(tmp_path):
    with pytest.raises(OSError):
        with HDF5TableWriter(tmp_path / "test.h5", "test", mode="r"):
            pass


def test_cannot_write_with_reader(test_h5_file):
    with HDF5TableReader(test_h5_file, mode="w") as h5:
        assert h5._h5file.mode == "r"


def test_cannot_append_with_reader(test_h5_file):
    with HDF5TableReader(test_h5_file, mode="a") as h5:
        assert h5._h5file.mode == "r"


def test_cannot_r_plus_with_reader(test_h5_file):
    with HDF5TableReader(test_h5_file, mode="r+") as h5:
        assert h5._h5file.mode == "r"


def test_append_mode(tmp_path):
    path = tmp_path / "test.h5"

    class ContainerA(Container):
        a = Field(int)

    container = ContainerA(a=1)

    # First open with 'w' mode to clear the file and add a Container
    with HDF5TableWriter(path, "group") as h5:
        h5.write("table_1", container)

    # Try to append A again
    with HDF5TableWriter(path, "group", mode="a") as h5:
        h5.write("table_2", container)

    # Check if file has two tables with a = 1
    with HDF5TableReader(path) as h5:
        for container in h5.read("/group/table_1", ContainerA):
            assert container.a == 1

        for container in h5.read("/group/table_2", ContainerA):
            assert container.a == 1


def test_write_to_any_location(tmp_path):
    path = tmp_path / "test.h5"
    loc = "path/path_1"

    class ContainerA(Container):
        a = Field(0, "some integer field")

    container = ContainerA(a=1)

    with HDF5TableWriter(path, group_name=loc + "/group_1") as h5:
        for _ in range(5):
            h5.write("table", container)
            h5.write("deeper/table2", container)

    with HDF5TableReader(path) as h5:
        for container in h5.read(f"/{loc}/group_1/table", ContainerA):
            assert container.a == 1

    with HDF5TableReader(path) as h5:
        for container in h5.read(f"/{loc}/group_1/deeper/table2", ContainerA):
            assert container.a == 1


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
            for data in h5_table.read(group_name, WithNormalEnum):
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
            for data in h5_table.read(group_name, WithIntEnum):
                assert isinstance(data.event_type, WithIntEnum.EventType)


def test_column_exclusions(tmp_path):
    """test if we can exclude columns using regexps for the table and column name"""
    tmp_file = tmp_path / "test_column_exclusions.hdf5"

    class SomeContainer(Container):
        default_prefix = ""
        hillas_x = Field(None)
        hillas_y = Field(None)
        impact_x = Field(None)
        impact_y = Field(None)

    cont = SomeContainer(hillas_x=10, hillas_y=10, impact_x=15, impact_y=15)

    with HDF5TableWriter(tmp_file) as writer:
        # don't write hillas columns for any table
        writer.exclude(".*table", "hillas_.*")

        # exclude a specific column
        writer.exclude("data/anothertable", "impact_x")
        print(writer._exclusions)

        writer.write("data/mytable", cont)
        writer.write("data/anothertable", cont)

    # check that we get back the transformed values (note here a round trip will
    # not work, as there is no inverse transform in this test)
    with HDF5TableReader(tmp_file, mode="r") as reader:
        data = next(reader.read("/data/mytable", SomeContainer))
        assert data.hillas_x is None
        assert data.hillas_y is None
        assert data.impact_x == 15
        assert data.impact_y == 15

        data = next(reader.read("/data/anothertable", SomeContainer))
        assert data.hillas_x is None
        assert data.hillas_y is None
        assert data.impact_x is None
        assert data.impact_y == 15


def test_column_transforms(tmp_path):
    """ensure a user-added column transform is applied"""
    from ctapipe.containers import NAN_TIME
    from ctapipe.io.tableio import FixedPointColumnTransform

    tmp_file = tmp_path / "test_column_transforms.hdf5"

    class SomeContainer(Container):
        default_prefix = ""

        current = Field(1 * u.A, unit=u.uA)
        time = Field(NAN_TIME)
        image = Field(np.array([1.234, 123.456]))

    cont = SomeContainer()

    with HDF5TableWriter(tmp_file, group_name="data") as writer:
        writer.add_column_transform(
            "mytable", "image", FixedPointColumnTransform(100, 0, np.float64, np.int32)
        )
        # add user generated transform for the "value" column
        writer.write("mytable", cont)

    # check that we get a length-3 array when reading back
    with HDF5TableReader(tmp_file, mode="r") as reader:
        data = next(reader.read("/data/mytable", SomeContainer))
        assert data.current.value == 1e6
        assert data.current.unit == u.uA
        assert isinstance(data.time, Time)
        assert data.time == NAN_TIME
        # rounded to two digits
        assert np.all(data.image == np.array([1.23, 123.46]))


def test_fixed_point_column_transform(tmp_path):
    """ensure a user-added column transform is applied"""
    from ctapipe.io.tableio import FixedPointColumnTransform

    tmp_file = tmp_path / "test_column_transforms.hdf5"

    class SomeContainer(Container):
        default_prefix = ""
        image = Field(np.array([np.nan, np.inf, -np.inf]))

    cont = SomeContainer()

    with HDF5TableWriter(tmp_file, group_name="data") as writer:
        writer.add_column_transform(
            "signed", "image", FixedPointColumnTransform(100, 0, np.float64, np.int32)
        )
        writer.add_column_transform(
            "unsigned",
            "image",
            FixedPointColumnTransform(100, 0, np.float64, np.uint32),
        )
        # add user generated transform for the "value" column
        writer.write("signed", cont)
        writer.write("unsigned", cont)

    with HDF5TableReader(tmp_file, mode="r") as reader:
        signed = next(reader.read("/data/signed", SomeContainer))
        unsigned = next(reader.read("/data/unsigned", SomeContainer))

        for data in (signed, unsigned):
            # check we get our original nans back
            assert np.isnan(data.image[0])
            assert np.isposinf(data.image[1])
            assert np.isneginf(data.image[2])


def test_column_transforms_regexps(tmp_path):
    """ensure a user-added column transform is applied when given as a regexp"""

    tmp_file = tmp_path / "test_column_transforms.hdf5"

    def multiply_by_10(x):
        return x * 10

    class SomeContainer(Container):
        default_prefix = ""
        hillas_x = Field(1)
        hillas_y = Field(1)

    cont = SomeContainer()

    with HDF5TableWriter(tmp_file, group_name="data") as writer:
        writer.add_column_transform_regexp("my.*", "hillas_.*", multiply_by_10)
        writer.add_column_transform_regexp("anothertable", "hillas_y", multiply_by_10)

        writer.write("mytable", cont)
        writer.write("anothertable", cont)

    # check that we get back the transformed values (note here a round trip will
    # not work, as there is no inverse transform in this test)
    with HDF5TableReader(tmp_file, mode="r") as reader:
        data = next(reader.read("/data/mytable", SomeContainer))
        assert data.hillas_x == 10
        assert data.hillas_y == 10

        data = next(reader.read("/data/anothertable", SomeContainer))
        assert data.hillas_x == 1
        assert data.hillas_y == 10


def test_time(tmp_path):
    tmp_file = tmp_path / "test_time.hdf5"

    class TimeContainer(Container):
        time = Field(None, "an astropy time")

    time = Time("2012-01-01T20:00:00", format="isot", scale="utc")
    container = TimeContainer(time=time)

    with HDF5TableWriter(tmp_file, group_name="data") as writer:
        # add user generated transform for the "value" column
        writer.write("table", container)

    with HDF5TableReader(tmp_file, mode="r") as reader:
        for data in reader.read("/data/table", TimeContainer):
            assert isinstance(data.time, Time)
            assert data.time.scale == "tai"
            assert data.time.format == "mjd"
            assert (data.time - time).to(u.s).value < 1e-7


def test_filters(tmp_path):
    from tables import Filters, open_file

    path = tmp_path / "test_time.hdf5"

    class TestContainer(Container):
        value = Field(-1, "test")

    no_comp = Filters(complevel=0)
    zstd = Filters(complevel=5, complib="blosc:zstd")

    with HDF5TableWriter(path, group_name="data", mode="w", filters=no_comp) as writer:
        assert writer.h5file.filters.complevel == 0

        c = TestContainer(value=5)
        writer.write("default", c)

        writer.filters = zstd
        writer.write("zstd", c)

        writer.filters = no_comp
        writer.write("nocomp", c)

    with open_file(path) as h5file:
        assert h5file.root.data.default.filters.complevel == 0
        assert h5file.root.data.zstd.filters.complevel == 5
        assert h5file.root.data.zstd.filters.complib == "blosc:zstd"
        assert h5file.root.data.nocomp.filters.complevel == 0


def test_column_order_single_container(tmp_path):
    """Test that columns are written in the order the containers define them"""
    path = tmp_path / "test.h5"

    class Container1(Container):
        b = Field(1, "b")
        c = Field("foo", "a", type=str, max_length=20)
        a = Field(2, "a")

    # test with single container
    with HDF5TableWriter(path, mode="w") as writer:
        c = Container1()
        writer.write("foo", c)

    with tables.open_file(path, "r") as f:
        assert f.root.foo[:].dtype.names == ("b", "c", "a")


def test_column_order_multiple_containers(tmp_path):
    """Test that columns are written in the order the containers define them"""
    path = tmp_path / "test.h5"

    class Container1(Container):
        b = Field(1, "b")
        a = Field(2, "a")

    class Container2(Container):
        d = Field(3, "d")
        c = Field(4, "c")

    # test with two containers
    with HDF5TableWriter(path, mode="w") as writer:
        c1 = Container1()
        c2 = Container2()
        writer.write("foo", [c2, c1])
        writer.write("bar", [c1, c2])

    with tables.open_file(path, "r") as f:
        assert f.root.foo[:].dtype.names == ("d", "c", "b", "a")
        assert f.root.bar[:].dtype.names == ("b", "a", "d", "c")


def test_writing_nan_defaults(tmp_path):
    from ctapipe.containers import ImageParametersContainer

    path = tmp_path / "test.h5"

    params = ImageParametersContainer()

    with HDF5TableWriter(path, mode="w") as writer:
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
def test_write_default_container(cls, tmp_path):
    path = tmp_path / "test.h5"

    with HDF5TableWriter(path, mode="w") as writer:
        try:
            writer.write("params", cls())
        except ValueError as e:
            # some containers do not have writable members,
            # only subcontainers. For now, ignore them.
            if "cannot create an empty data type" not in str(e):
                raise


def test_strings(tmp_path):
    """Test we can write unicode strings"""
    from ctapipe.core import Container
    from ctapipe.io import read_table

    # when not giving a max_len, should be taken from the first container
    class Container1(Container):
        default_prefix = ""
        string = Field("", "test string")

    path = tmp_path / "test.h5"

    strings = ["Hello", "öäα"]

    with HDF5TableWriter(path, mode="w") as writer:
        for string in strings:
            writer.write("strings", Container1(string=string))

    table = read_table(path, "/strings")

    # the α is above the max length estimated from the first element
    assert table["string"].tolist() == ["Hello", "öä"]

    class Container2(Container):
        default_prefix = ""
        string = Field("", "test string", max_length=10)

    path = tmp_path / "test.h5"

    strings = ["Hello", "öäα", "12345678910"]
    expected = ["Hello", "öäα", "1234567891"]

    with HDF5TableWriter(path, mode="w") as writer:
        for string in strings:
            writer.write("strings", Container2(string=string))

    table = read_table(path, "/strings")

    # the α is above the max length estimated from the first element
    assert table["string"].tolist() == expected

    # test this also works with table reader
    with HDF5TableReader(path) as reader:
        generator = reader.read("/strings", Container2)
        for string in expected:
            c = next(generator)
            assert c.string == string


def test_prefix_in_output_container(tmp_path):
    """Test that output containers retain the used prefix"""

    class Container1(Container):
        default_prefix = ""
        value = Field(-1, "value")

    path = tmp_path / "prefix.h5"
    with HDF5TableWriter(path, mode="w", add_prefix=True) as writer:
        for value in (1, 2, 3):
            writer.write("values", Container1(value=value, prefix="custom_prefix"))

    with HDF5TableReader(path) as reader:
        generator = reader.read("/values", Container1, prefixes="custom_prefix")

        for value in (1, 2, 3):
            c = next(generator)
            assert c.prefix == "custom_prefix"
            assert c.value == value


def test_can_read_without_prefix_given(tmp_path):
    """Test that output containers retain the used prefix"""

    class Container1(Container):
        default_prefix = ""
        value = Field(-1, "value")

    path = tmp_path / "prefix.h5"
    with HDF5TableWriter(path, mode="w", add_prefix=True) as writer:
        for value in (1, 2, 3):
            writer.write("values", Container1(value=value, prefix="custom_prefix"))

    # test we can read back the data without knowing the "custom_prefix"
    with HDF5TableReader(path) as reader:
        generator = reader.read("/values", Container1)

        for value in (1, 2, 3):
            c = next(generator)
            assert c.value == value
            assert c.prefix == "custom_prefix"


def test_can_read_same_containers(tmp_path):
    """Test we can read two identical containers with different prefixes"""

    class Container1(Container):
        default_prefix = ""
        value = Field(-1, "value")

    # test with two of the same container with different prefixes
    path = tmp_path / "two_containers.h5"
    with HDF5TableWriter(path, mode="w", add_prefix=True) as writer:
        for value in (1, 2, 3):
            writer.write(
                "values",
                [
                    Container1(value=value, prefix="foo"),
                    Container1(value=5 * value, prefix="bar"),
                ],
            )

    # This needs to fail since the mapping is not unique
    with HDF5TableReader(path) as reader:
        with pytest.raises(OSError):
            generator = reader.read("/values", [Container1, Container1])
            next(generator)

    # But when explicitly giving the prefixes, this works and order
    # should not be important
    with HDF5TableReader(path) as reader:
        generator = reader.read(
            "/values",
            [Container1, Container1],
            prefixes=["bar", "foo"],
        )

        for value in (1, 2, 3):
            c1, c2 = next(generator)
            assert c1.value == 5 * value
            assert c1.prefix == "bar"
            assert c2.value == value
            assert c2.prefix == "foo"


@pytest.mark.parametrize("input_type", (str, Path, tables.File))
def test_hdf5_datalevels(input_type, dl2_shower_geometry_file):
    from ctapipe.io import get_hdf5_datalevels

    if input_type is tables.File:
        with tables.open_file(dl2_shower_geometry_file) as h5file:
            datalevels = get_hdf5_datalevels(h5file)
    else:
        path = input_type(dl2_shower_geometry_file)
        datalevels = get_hdf5_datalevels(path)

    assert set(datalevels) == {
        DataLevel.DL1_IMAGES,
        DataLevel.DL1_PARAMETERS,
        DataLevel.DL2,
    }
