# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from ctapipe.core import Container, Item, Map


def test_container():

    class ExampleContainer(Container):
        x = Item(-1, "x value")
        y = Item(-1, "y value")

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
        z = Item(1, "sub-item")

    class ParentContainer(Container):
        x = Item(0, "some value")
        child = Item(ChildContainer(), "a child")

    cont = ParentContainer()
    assert cont.child.z == 1


def test_map_containers():

    class ChildContainer(Container):
        z = Item(1, "sub-item")

    class ParentContainer(Container):
        x = Item(0, "some value")
        children = Item(Map(), "map of tel_id to child")

    cont = ParentContainer()
    cont.children[10] = ChildContainer()
    cont.children[5] = ChildContainer()

    cont.children[5].z = 99
    assert cont.children[5].z == 99

    cont.reset()
    assert 5 not in cont.children


def test_container_as_dict():

    class ChildContainer(Container):
        z = Item(1, "sub-item")

    class ParentContainer(Container):
        x = Item(0, "some value")
        child = Item(ChildContainer(), "a child")

    cont = ParentContainer()

    the_flat_dict = cont.as_dict(recursive=True, flatten=True)
    the_dict = cont.as_dict(recursive=True, flatten=False)

    assert 'child_z' in the_flat_dict
    assert 'child' in the_dict and 'z' in the_dict['child']
