def test_available_sources():
    from ctapipe.io.eventsource import EventSource
    from ctapipe.core import non_abstract_children

    # make this before the explicit imports to make sure
    # all classes are avaialble even if not explicitly imported
    children = non_abstract_children(EventSource)

    from ctapipe.io.simteleventsource import SimTelEventSource

    assert SimTelEventSource in children
