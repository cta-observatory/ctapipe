from traitlets.config import Config

from ctapipe.containers import ArrayEventContainer, EventType


def test_event_filter():
    from ctapipe.utils import EventTypeFilter

    event_filter = EventTypeFilter(
        allowed_types={EventType.SUBARRAY, EventType.FLATFIELD}
    )

    e = ArrayEventContainer()
    e.trigger.event_type = EventType.SUBARRAY
    assert event_filter(e)
    e.trigger.event_type = EventType.FLATFIELD
    assert event_filter(e)
    e.trigger.event_type = EventType.DARK_PEDESTAL
    assert not event_filter(e)


def test_event_filter_none():
    from ctapipe.utils import EventTypeFilter

    event_filter = EventTypeFilter(allowed_types=None)

    # all event types should pass
    e = ArrayEventContainer()
    for value in EventType:
        e.trigger.event_type = value
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

    e = ArrayEventContainer()
    e.trigger.event_type = EventType.DARK_PEDESTAL
    assert not event_filter(e)

    e.trigger.event_type = EventType.SUBARRAY
    assert event_filter(e)
