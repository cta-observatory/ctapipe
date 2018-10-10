"""
common pytest fixtures for tests in ctapipe
"""

import pytest
from ctapipe.utils import get_dataset_path
from ctapipe.io.eventseeker import EventSeeker
from ctapipe.io.hessioeventsource import HESSIOEventSource
from copy import deepcopy

@pytest.fixture(scope='session')
def test_event():
    """ an example event for algorithm testing"""
    filename = get_dataset_path('gamma_test.simtel.gz')

    print("******************** LOAD TEST EVENT ***********************")

    with HESSIOEventSource(input_url=filename) as reader:
        seeker = EventSeeker(reader)
        event = seeker['409']

    return event




