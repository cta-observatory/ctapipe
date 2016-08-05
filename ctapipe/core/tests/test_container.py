# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from ctapipe.core import Container


def test_container():

    cont = Container("data")

    with pytest.raises(AttributeError):
        cont.x

    with pytest.raises(KeyError):
        cont["x"]

    with pytest.raises(AttributeError):
        cont.x = 10

    with pytest.raises(TypeError):
        cont['x'] = 10

    # test adding an item
    cont.add_item('y')
    assert cont.y is None

    # test adding item with value:
    cont.add_item('z', 10)
    assert cont.z == 10
    assert cont['z'] == 10

    # test iteration (turn it into a list)
    assert len(list(cont)) == 2


def test_container_metadata():

    cont = Container("data")
    cont.meta.add_item("version", 2.0)

    assert cont.meta.version == 2.0
