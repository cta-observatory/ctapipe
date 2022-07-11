from ..containers import EventType
from ..core import Component
from ..core.traits import Set, UseEnum

__all__ = ["EventTypeFilter"]


_values = ", ".join([f"{e.name} or {e.value}" for e in EventType])


class EventTypeFilter(Component):
    """Check that an event has one of the allowed types"""

    allowed_types = Set(
        # add both the enum instance and the integer values to support
        # giving the integers in config files.
        trait=UseEnum(EventType),
        default_value=None,
        allow_none=True,
        help=(
            "The allowed types. Set to None to allow all types."
            f"Possible values: {_values}."
        ),
    ).tag(config=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, event):
        """Returns True if the event should be kept"""
        if self.allowed_types is None:
            return True

        return event.trigger.event_type in self.allowed_types
