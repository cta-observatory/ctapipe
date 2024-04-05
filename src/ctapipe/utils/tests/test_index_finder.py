import numpy as np

from ctapipe.utils import IndexFinder


def test_zerolength():
    values = [42]
    finder = IndexFinder(values)
    assert finder.closest(40) == 0
    assert finder.closest(-1) == 0


def test_array():
    values = np.arange(10) + 5
    finder = IndexFinder(values)
    assert finder.closest(12) == 7


def test_unsorted():
    values = [1, 10, 5, 9, 4, -2]
    finder = IndexFinder(values)
    assert finder.closest(8) == 3
    assert finder.closest(-10) == 5
