from traitlets.config import Config

from ctapipe.containers import EventType, SubarrayEventContainer


def test_event_filter():
    from ctapipe.utils import EventTypeFilter

    event_filter = EventTypeFilter(
        allowed_types={EventType.SUBARRAY, EventType.FLATFIELD}
    )

    e = SubarrayEventContainer()
    e.dl0.trigger.event_type = EventType.SUBARRAY
    assert event_filter(e)
    e.dl0.trigger.event_type = EventType.FLATFIELD
    assert event_filter(e)
    e.dl0.trigger.event_type = EventType.DARK_PEDESTAL
    assert not event_filter(e)


def test_event_filter_none():
    from ctapipe.utils import EventTypeFilter

    event_filter = EventTypeFilter(allowed_types=None)

    # all event types should pass
    e = SubarrayEventContainer()
    for value in EventType:
        e.dl0.trigger.event_type = value
        assert event_filter(e)


def test_event_filter_config():
    from ctapipe.utils import EventTypeFilter

    config = Config(
        {
            "EventTypeFilter": {
                "allowed_types": [
                    EventType.SUBARRAY.value,
                    "FLATFIELD",
                    1,
                ]
            }
        }
    )
    event_filter = EventTypeFilter(config=config)

    assert event_filter.allowed_types == {
        EventType.SUBARRAY,
        EventType.FLATFIELD,
        EventType.SINGLE_PE,
    }

    e = SubarrayEventContainer()
    e.dl0.trigger.event_type = EventType.DARK_PEDESTAL
    assert not event_filter(e)

    e.dl0.trigger.event_type = EventType.SUBARRAY
    assert event_filter(e)
