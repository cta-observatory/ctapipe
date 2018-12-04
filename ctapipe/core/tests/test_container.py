# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from ctapipe.core import Container, Field, Map


def test_prefix():
    class AwesomeContainer(Container):
        pass

    # make sure the default prefix is class name without container
    assert AwesomeContainer.container_prefix == 'awesome'
    assert AwesomeContainer().prefix == 'awesome'

    # make sure we can set the class level prefix at definition time
    class ReallyAwesomeContainer(Container):
        container_prefix = 'test'

    assert ReallyAwesomeContainer.container_prefix == 'test'
    r = ReallyAwesomeContainer()
    assert r.prefix == 'test'

    ReallyAwesomeContainer.container_prefix = 'test2'
    # new instance should have the old prefix, old instance
    # the one it was created with
    assert ReallyAwesomeContainer().prefix == 'test2'
    assert r.prefix == 'test'

    # Make sure we can set the class level prefix at runtime
    ReallyAwesomeContainer.container_prefix = 'foo'
    assert ReallyAwesomeContainer().prefix == 'foo'

    # make sure we can assign instance level prefixes
    c1 = ReallyAwesomeContainer()
    c2 = ReallyAwesomeContainer()
    c2.prefix = 'c2'

    assert c1.prefix == 'foo'
    assert c2.prefix == 'c2'


def test_inheritance():

    class ExampleContainer(Container):
        a = Field(None)

    class SubclassContainer(ExampleContainer):
        b = Field(None)

    assert 'a' in SubclassContainer.fields

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

    assert 'a' in ContainerC.fields
    assert 'b' in ContainerC.fields


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
    cont.meta['stuff'] = 'things'
    assert 'stuff' in cont.meta and cont.meta['stuff'] == 'things'


def test_child_containers():
    class ChildContainer(Container):
        z = Field(1, "sub-item")

    class ParentContainer(Container):
        x = Field(0, "some value")
        child = Field(ChildContainer(), "a child")

    cont = ParentContainer()
    assert cont.child.z == 1


def test_map_containers():

    class ChildContainer(Container):
        z = Field(1, "sub-item")

    class ParentContainer(Container):
        x = Field(0, "some value")
        children = Field(Map(), "map of tel_id to child")

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
        child = Field(ChildContainer(), "a child")

    cont = ParentContainer()

    the_flat_dict = cont.as_dict(recursive=True, flatten=True)
    the_dict = cont.as_dict(recursive=True, flatten=False)

    assert 'child_z' in the_flat_dict
    assert 'child' in the_dict and 'z' in the_dict['child']


def test_container_brackets():

    class TestContainer(Container):
        answer = Field(-1, "The answer to all questions")

    t = TestContainer()

    t['answer'] = 42

    with pytest.raises(AttributeError):
        t['foo'] = 5
