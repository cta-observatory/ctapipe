# Licensed under a 3-clause BSD style license - see LICENSE.rst
from ..datasets import get_ctapipe_extra_path

# TODO: we should have a decorator that skips tests if
# the datasets are not available
# (and probably a warning should be printed to the test log
# that many tests will not be run.)
def test_get_datasets_path():
    path = get_ctapipe_extra_path()
    assert path is not None
