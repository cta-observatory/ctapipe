# Licensed under a 3-clause BSD style license - see LICENSE.rst
from ..datasets import get_datasets_path


def test_get_datasets_path():
    path = get_datasets_path()
    assert path is not None
