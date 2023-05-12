import numpy as np

from ctapipe.containers import ArrayEventContainer
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import Integer, IntTelescopeParameter


class SoftwareTrigger(TelescopeComponent):
    """
    A stereo trigger that can remove telescope events from subarray events.

    This class is needed to correctly handle super-arrays as simulated for
    CTA and still handle the LST hardware stereo trigger and the normal stereo
    trigger correctly.

    When selecting subarrays from simulations that contain many more telescopes,
    as is done in all major CTA productions to date, the stereo trigger is not
    correctly simulated as in that after selecting a realistic subarray, events
    are still in the data stream where only one telescope of the selected subarray
    triggered, which would in reality not trigger the stereo trigger.

    An additional complexity is the LST hardware stereo trigger, that forces that
    an array event has always to contain no or at least two LST telescope events.

    This means that after selectig a subarray, we need to:
    - Remove LST telescope events from the subarray if only one LST triggered
    - Ignore events with only 1 telescope after this has been applied

    With the default settings, this class is a no-op. To get the correct behavior
    for CTA simulations, use the following configuration:

    ..
        SoftwareTrigger:
            min_telescopes: 2
            min_telescopes_of_type:
                - ["type", "*", 0]
                - ["type", "LST*", 2]
    """

    min_telescopes = Integer(
        default_value=1,
        help=(
            "Minimum number of telescopes required globally."
            " Events with fewer telescopes will be filtered out completely."
        ),
    ).tag(config=True)

    min_telescopes_of_type = IntTelescopeParameter(
        default_value=0,
        help=(
            "Minimum number of telescopes required for a specific type."
            " In events with fewer telescopes of that type"
            " , those telescopes will be removed from the array event."
            " This might result in the event not fullfilling ``min_telescopes`` anymore"
            " and thus being filtered completely."
        ),
    ).tag(config=True)

    def __init__(self, subarray, *args, **kwargs):
        super().__init__(subarray, *args, **kwargs)

        self._ids_by_type = {
            str(type): set(self.subarray.get_tel_ids_for_type(type))
            for type in self.subarray.telescope_types
        }

    def __call__(self, event: ArrayEventContainer) -> bool:
        """
        Remove telescope events that have not the required number of telescopes of
        a given type from the subarray event and decide if the event would
        have triggered the stereo trigger.

        Data is cleared from events that did not trigger.

        Returns
        -------
        triggered : bool
            Whether or not this event would have triggered the stereo trigger
        """

        tels_removed = set()
        for tel_type in self.subarray.telescope_types:
            tel_type_str = str(tel_type)
            min_tels = self.min_telescopes_of_type.tel[tel_type_str]

            # no need to check telescopes for which we have no min requirement
            if min_tels == 0:
                continue

            tels_with_trigger = set(event.trigger.tels_with_trigger)
            tel_ids = self._ids_by_type[tel_type_str]
            tels_in_event = tels_with_trigger.intersection(tel_ids)

            if len(tels_in_event) < min_tels:
                for tel_id in tels_in_event:
                    self.log.debug(
                        "Removing tel_id %d of type %s from event due to type requirement",
                        tel_id,
                        tel_type_str,
                    )

                    # remove from tels_with_trigger
                    tels_removed.add(tel_id)

                    # remove any related data
                    for container in ("trigger", "r0", "r1", "dl0", "dl1", "dl2"):
                        tel_map = getattr(event, container).tel
                        if tel_id in tel_map:
                            del tel_map[tel_id]

        if len(tels_removed) > 0:
            # convert to array with correct dtype to have setdiff1d work correctly
            tels_removed = np.fromiter(tels_removed, np.uint16, len(tels_removed))
            event.trigger.tels_with_trigger = np.setdiff1d(
                event.trigger.tels_with_trigger, tels_removed, assume_unique=True
            )

        if len(event.trigger.tels_with_trigger) < self.min_telescopes:
            event.trigger.tels_with_trigger = []
            for container in ("trigger", "r0", "r1", "dl0", "dl1", "dl2"):
                tel_map = getattr(event, container).tel
                tel_map.clear()
            return False
        return True
