import os
import pytest
from pkg_resources import resource_filename
from ctapipe.io import event_source
from ctapipe.io.containers import DataContainer


def test_sst1m_eventsource_plugin():
    ctapipe_io_sst1m = pytest.importorskip('ctapipe_io_sst1m')
    pytest.importorskip('protozfits')

    sst1m_test_file = resource_filename(
        'protozfits',
        os.path.join(
            'tests',
            'resources',
            'example_10evts.fits.fz'
        )
    )

    source = event_source(sst1m_test_file)
    assert isinstance(source, ctapipe_io_sst1m.SST1MEventSource)

    for event in source:
        # of course
        assert isinstance(
            event,
            ctapipe_io_sst1m.containers.SST1MDataContainer
        )
        # but also:
        assert isinstance(
            event,
            DataContainer
        )

        # if the event was not a DataContainer, this eventsource cannot
        # claim to be ctapipe-compatible right?
