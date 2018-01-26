"""
common pytest fixtures for tests in ctapipe
"""

import pytest
from ctapipe.utils import get_dataset
from ctapipe.io.eventseeker import EventSeeker
from ctapipe.io.hessiofilereader import HessioFileReader

@pytest.fixture(scope='module')
def test_event():
    """ an example event for algorithm testing"""
    filename = get_dataset('gamma_test.simtel.gz')

    with HessioFileReader(None, None, input_url=filename) as reader:
        seeker = EventSeeker(None, None, reader=reader )
        event = seeker['409']

    yield event



