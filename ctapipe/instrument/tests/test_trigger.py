import numpy as np
import pytest
from numpy.testing import assert_equal

from ctapipe.containers import ArrayEventContainer
from ctapipe.io import EventSource


@pytest.mark.parametrize("data_type", (list, np.array))
def test_software_trigger(subarray_prod5_paranal, data_type):
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
    event.trigger.tels_with_trigger = data_type([5])
    assert trigger(event) == False
    assert_equal(event.trigger.tels_with_trigger, data_type([]))

    # 1 LST + 1 MST, 1 LST would not have triggered LST hardware trigger
    # and after LST is removed, we only have 1 telescope, so no SWAT either
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = data_type([1, 6])
    assert trigger(event) == False
    assert_equal(event.trigger.tels_with_trigger, data_type([]))

    # two MSTs and 1 LST, -> remove single LST
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = data_type([1, 5, 6])
    assert trigger(event) == True
    assert_equal(event.trigger.tels_with_trigger, data_type([5, 6]))

    # two MSTs, nothing to change
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = data_type([5, 6])
    assert trigger(event) == True
    assert_equal(event.trigger.tels_with_trigger, data_type([5, 6]))

    # three LSTs, nothing to change
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = data_type([1, 2, 3])
    assert trigger(event) == True
    assert_equal(event.trigger.tels_with_trigger, data_type([1, 2, 3]))

    # thee LSTs, plus MSTs, nothing to change
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = data_type([1, 2, 3, 5, 6, 7])
    assert trigger(event) == True
    assert_equal(event.trigger.tels_with_trigger, data_type([1, 2, 3, 5, 6, 7]))


@pytest.mark.parametrize("allowed_tels", (None, list(range(1, 20))))
def test_software_trigger_simtel(allowed_tels):
    from ctapipe.instrument.trigger import SoftwareTrigger

    path = "dataset://gamma_divergent_LaPalma_baseline_20Zd_180Az_prod3_test.simtel.gz"

    expected = [
        [12, 16],
        [],
        [1, 2, 3, 4],
        [1, 4],
        [],
        [1, 3],
        [1, 2, 3, 4, 5, 6, 7, 12, 15, 16, 17, 18],
        [13, 14],
        [],
        [2, 3, 7, 12],
        [1, 2, 5, 17],
        [],
        [13, 19],
        [],
        [],
        [1, 2, 4, 5, 11, 18],
        [17, 18],
        [7, 12],
        [],
    ]

    with EventSource(
        path, focal_length_choice="EQUIVALENT", allowed_tels=allowed_tels
    ) as source:

        trigger = SoftwareTrigger(
            subarray=source.subarray,
            min_telescopes=2,
            min_telescopes_of_type=[
                ("type", "*", 0),
                ("type", "LST*", 2),
            ],
        )

        for e, expected_tels in zip(source, expected):
            trigger(e)
            assert_equal(e.trigger.tels_with_trigger, expected_tels)
