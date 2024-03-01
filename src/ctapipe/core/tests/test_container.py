# Licensed under a 3-clause BSD style license - see LICENSE.rst
from functools import partial

import numpy as np
import pytest
from astropy import units as u

from ctapipe.core import Container, DeprecatedField, Field, FieldValidationError, Map


def test_prefix():
    class AwesomeContainer(Container):
        pass

    # make sure the default prefix is class name without container
    assert AwesomeContainer.default_prefix == "awesome"
    assert AwesomeContainer().prefix == "awesome"

    # make sure we can set the class level prefix at definition time
    class ReallyAwesomeContainer(Container):
        default_prefix = "test"

    assert ReallyAwesomeContainer.default_prefix == "test"
    r = ReallyAwesomeContainer()
    assert r.prefix == "test"

    # new instance should have the new prefix,
    # old instance the one it was created with
    ReallyAwesomeContainer.default_prefix = "test2"
    assert ReallyAwesomeContainer().prefix == "test2"
    assert r.prefix == "test"

    # make sure we can assign instance level prefixes
    c1 = ReallyAwesomeContainer()
    c2 = ReallyAwesomeContainer()
    c3 = ReallyAwesomeContainer(prefix="c3")
    c2.prefix = "c2"

    assert c1.prefix == "test2"
    assert c2.prefix == "c2"
    assert c3.prefix == "c3"


def test_inheritance():
    class ExampleContainer(Container):
        a = Field(None)

    class SubclassContainer(ExampleContainer):
        b = Field(None)

    assert "a" in SubclassContainer.fields

    c = SubclassContainer()
    assert c.a is None
    assert c.b is None

    c.a = 5
    c.b = 10

    assert c.a == 5
    assert c.b == 10


def test_multiple_inheritance():
    class ContainerA(Container):
        a = Field(None)

    class ContainerB(ContainerA):
        b = Field(None)

    class ContainerC(ContainerB):
        c = Field(None)

    assert "a" in ContainerC.fields
    assert "b" in ContainerC.fields


def test_override_inheritance():
    class ContainerA(Container):
        a = Field(1)

    class ContainerB(ContainerA):
        a = Field(2)

    a = ContainerA()
    assert a.a == 1

    b = ContainerB()
    assert b.a == 2


def test_container():
    class ExampleContainer(Container):
        x = Field(-1, "x value")
        y = Field(-1, "y value")

    cont = ExampleContainer()
    cont2 = ExampleContainer()

    assert cont.x == -1
    assert cont.y == -1

    # test setting a value
    cont.x = 10
    assert cont.x == 10

    # make sure the value is set in the instance not the class
    assert cont2.x == -1

    # make sure you can't set an attribute that isn't defined
    with pytest.raises(AttributeError):
        cont.z = 100

    # test resetting values to defaults:
    cont.reset()
    assert cont.x == -1

    # test adding metadata
    cont.meta["stuff"] = "things"
    assert "stuff" in cont.meta and cont.meta["stuff"] == "things"


def test_child_containers():
    class ChildContainer(Container):
        z = Field(1, "sub-item")

    class ParentContainer(Container):
        x = Field(0, "some value")
        child = Field(default_factory=ChildContainer, description="a child")

    cont = ParentContainer()
    assert cont.child.z == 1


def test_map_containers():
    class ChildContainer(Container):
        z = Field(1, "sub-item")

    class ParentContainer(Container):
        x = Field(0, "some value")
        children = Field(default_factory=Map, description="map of tel_id to child")

    cont = ParentContainer()
    cont.children[10] = ChildContainer()
    cont.children[5] = ChildContainer()

    cont.children[5].z = 99
    assert cont.children[5].z == 99

    cont.reset()
    assert 5 not in cont.children


def test_container_as_dict():
    class ChildContainer(Container):
        z = Field(1, "sub-item")

    class ParentContainer(Container):
        x = Field(0, "some value")
        child = Field(default_factory=ChildContainer, description="a child")

    class GrandParentContainer(Container):
        y = Field(2, "some other value")
        child = Field(default_factory=ParentContainer, description="child")
        map = Field(default_factory=partial(Map, ChildContainer))

    cont = ParentContainer()

    assert cont.as_dict() == {"x": 0, "child": cont.child}
    assert cont.as_dict(recursive=True) == {"x": 0, "child": {"z": 1}}
    assert cont.as_dict(recursive=True, add_prefix=True) == {
        "parent_x": 0,
        "parent_child": {"child_z": 1},
    }

    assert cont.as_dict(recursive=True, flatten=True, add_prefix=False) == {
        "x": 0,
        "z": 1,
    }

    assert cont.as_dict(recursive=True, flatten=True, add_prefix=True) == {
        "parent_x": 0,
        "child_z": 1,
    }

    cont = GrandParentContainer()
    cont.map["foo"] = ChildContainer(z=3)
    cont.map["bar"] = ChildContainer(z=4)
    d = cont.as_dict(recursive=True, flatten=True, add_prefix=True, add_key=True)
    assert d == {
        "parent_x": 0,
        "child_z": 1,
        "grandparent_y": 2,
        "foo_child_z": 3,
        "bar_child_z": 4,
    }


def test_container_brackets():
    class ExampleContainer(Container):
        answer = Field(-1, "The answer to all questions")

    t = ExampleContainer()

    t["answer"] = 42

    with pytest.raises(AttributeError):
        t["foo"] = 5


def test_deprecated_field():
    with pytest.warns(DeprecationWarning, match="answer to all questions"):

        class ExampleContainer(Container):
            answer = DeprecatedField(
                -1, "The answer to all questions", reason="because"
            )

        cont = ExampleContainer()
        cont.answer = 6


def test_field_validation():
    # test units
    field_u = Field(None, "float with units", unit="m")
    field_u.validate(3.0 * u.m)
    field_u.validate(3.0 * u.km)
    with pytest.raises(FieldValidationError):
        field_u.validate(3.0)

    # test numpy arrays

    field_f = Field(None, "float array", dtype=np.float64, ndim=2)
    field_f.validate(np.ones((2, 2), dtype=np.float64))

    #   test dtype
    with pytest.raises(FieldValidationError):
        field_f.validate(np.ones((2, 2), dtype=np.int32))

    #   test ndims
    with pytest.raises(FieldValidationError):
        field_f.validate(np.ones((2), dtype=np.float32))

    # test scalars
    with pytest.raises(FieldValidationError):
        field_f.validate(7.0)

    field_s = Field(None, "scalar with type", dtype="int32")
    field_s.validate(np.int32(3))

    field_s2 = Field(None, "scalar with type", dtype="float32")
    field_s2.validate(np.float32(3.2))
    with pytest.raises(FieldValidationError):
        field_s2.validate(3.3)

    # test scalars with units and dtypes:
    field_s3 = Field(1.0, "scalar with dtype and unit", dtype="float32", unit="m")
    field_s3.validate(np.float32(6) * u.m)

    # test with no restrictions:
    field_all = Field(None, "stuff")
    field_all.validate(3.0)
    field_all.validate(np.ones((3, 3, 3)))
    field_all.validate(3.0 * u.kg)

    # test allow_none:
    field_n = Field(None, "test", allow_none=True, dtype="float32", ndim=6)
    field_n.validate(None)

    field_n2 = Field(None, "test", allow_none=False, dtype="float32")
    with pytest.raises(FieldValidationError):
        field_n2.validate(None)

    field_type = Field(None, "foo", type=str)
    field_type.validate("foo")
    with pytest.raises(FieldValidationError):
        field_type.validate(5)


def test_container_validation():
    """check that we can validate all fields in a container"""

    class MyContainer(Container):
        x = Field(3.2, "test", unit="m")
        z = Field(np.int64(1), "sub-item", dtype="int64")

    with pytest.raises(FieldValidationError):
        MyContainer().validate()  # fails since 3.2 has no units

    with pytest.raises(FieldValidationError):
        MyContainer(x=10 * u.s).validate()  # seconds is not convertible to meters

    MyContainer(x=6.4 * u.m).validate()  # works


def test_recursive_validation():
    """
    Check both sub-containers and Maps work with recursive validation
    """

    class ChildContainer(Container):
        x = Field(3.2 * u.m, "test", unit="m")

    class ParentContainer(Container):
        cont = Field(None, "test sub", type=ChildContainer)
        map = Field(Map(ChildContainer), "many children")

    with pytest.raises(FieldValidationError):
        ParentContainer(cont=ChildContainer(x=1 * u.s)).validate()

    with pytest.raises(FieldValidationError):
        cont = ParentContainer(cont=ChildContainer(x=1 * u.m))
        cont.map[0] = ChildContainer(x=1 * u.m)
        cont.map[1] = ChildContainer(x=1 * u.s)
        cont.validate()


def test_long_field_repr():
    field = Field(default_factory=lambda: np.geomspace(1.0, 1e4, 5))
    assert repr(field) == "Field(default=[1.e+00 1.e+01 ... 1.e+03 1.e+04])"

    field = Field(default_factory=lambda: np.linspace(1.0, 2.0, 5))
    assert repr(field) == "Field(default=[1.   1.25 ... 1.75 2.  ])"
