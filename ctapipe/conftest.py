"""
common pytest fixtures for tests in ctapipe
"""

import pytest

from copy import deepcopy

from ctapipe.io.eventseeker import EventSeeker
from ctapipe.io.hessioeventsource import HESSIOEventSource
from ctapipe.utils import get_dataset_path


@pytest.fixture(scope='session')
def _global_example_event():
    """
    helper to get a single event from a MC file. Don't use this fixture
    directly, rather use `test_event`
    """
    filename = get_dataset_path('gamma_test.simtel.gz')

    print("******************** LOAD TEST EVENT ***********************")

    with HESSIOEventSource(input_url=filename) as reader:
        seeker = EventSeeker(reader)
        event = seeker['409']

    return event


@pytest.fixture(scope='function')
def example_event(_global_example_event):
    """
    Use this fixture anywhere you need a test event read from a MC file. For
    example:

    .. code-block::
        def test_my_thing(test_event):
            assert len(test_event.r0.tels_with_data)>0

    """
    return deepcopy(_global_example_event)