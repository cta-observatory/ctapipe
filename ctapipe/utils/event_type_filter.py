from ..core import Component
from ..core.traits import Enum, Set
from ..containers import EventType


__all__ = ["EventTypeFilter"]


class EventTypeFilter(Component):
    """Check that an event has one of the allowed types"""

    allowed_types = Set(
        # add both the enum instance and the integer values to support
        # giving the integers in config files.
        trait=Enum(list(EventType) + [t.value for t in EventType]),
        default_value=None,
        allow_none=True,
        help="The allowed types. Set to None to allow all types.",
    ).tag(config=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convert ints to enum type
        if self.allowed_types is not None:
            self.allowed_types = {EventType(e) for e in self.allowed_types}

    def __call__(self, event):
        """Returns True if the event should be kept"""
        if self.allowed_types is None:
            return True

        return event.trigger.event_type in self.allowed_types
