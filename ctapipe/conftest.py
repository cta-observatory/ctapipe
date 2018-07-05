"""
common pytest fixtures for tests in ctapipe
"""

import pytest

from ctapipe.io.eventseeker import EventSeeker
from ctapipe.io.hessioeventsource import HESSIOEventSource
from ctapipe.utils import get_dataset_path


@pytest.fixture(scope='module')
def test_event():
    """ an example event for algorithm testing"""
    filename = get_dataset_path('gamma_test.simtel.gz')

    with HESSIOEventSource(input_url=filename) as reader:
        seeker = EventSeeker(reader)
        event = seeker['409']

    yield event
