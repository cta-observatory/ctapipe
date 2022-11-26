from ctapipe.containers import ArrayEventContainer


def test_software_trigger(subarray_prod5_paranal):
    from ctapipe.instrument.trigger import SoftwareTrigger

    subarray = subarray_prod5_paranal
    trigger = SoftwareTrigger(
        subarray=subarray,
        min_telescopes=2,
        min_telescopes_of_type=[
            ("type", "*", 0),
            ("type", "LST*", 2),
        ],
    )

    # only one telescope, no SWAT
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = [5]
    assert trigger(event) == False
    assert event.trigger.tels_with_trigger == []

    # 1 LST + 1 MST, 1 LST would not have triggered LST hardware trigger
    # and after LST is removed, we only have 1 telescope, so no SWAT either
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = [1, 6]
    assert trigger(event) == False
    assert event.trigger.tels_with_trigger == []

    # two MSTs and 1 LST, -> remove single LST
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = [1, 5, 6]
    assert trigger(event) == True
    assert event.trigger.tels_with_trigger == [5, 6]

    # two MSTs, nothing to change
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = [5, 6]
    assert trigger(event) == True
    assert event.trigger.tels_with_trigger == [5, 6]

    # three LSTs, nothing to change
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = [1, 2, 3]
    assert trigger(event) == True
    assert event.trigger.tels_with_trigger == [1, 2, 3]

    # thee LSTs, plus MSTs, nothing to change
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = [1, 2, 3, 5, 6, 7]
    assert trigger(event) == True
    assert event.trigger.tels_with_trigger == [1, 2, 3, 5, 6, 7]
