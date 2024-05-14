import json
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_equal

from ctapipe.containers import ArrayEventContainer
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import EventSource


def assert_all_tel_keys(event, expected, ignore=None):
    if ignore is None:
        ignore = set()

    expected = tuple(expected)
    for name, container in event.items():
        if hasattr(container, "tel"):
            actual = tuple(container.tel.keys())
            if name not in ignore and actual != expected:
                raise AssertionError(
                    f"Unexpected tel_ids in container {name}:" f"{actual} != {expected}"
                )


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
    assert not trigger(event)
    assert_equal(event.trigger.tels_with_trigger, data_type([]))

    # 1 LST + 1 MST, 1 LST would not have triggered LST hardware trigger
    # and after LST is removed, we only have 1 telescope, so no SWAT either
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = data_type([1, 6])
    assert not trigger(event)
    assert_equal(event.trigger.tels_with_trigger, data_type([]))

    # two MSTs and 1 LST, -> remove single LST
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = data_type([1, 5, 6])
    assert trigger(event)
    assert_equal(event.trigger.tels_with_trigger, data_type([5, 6]))

    # two MSTs, nothing to change
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = data_type([5, 6])
    assert trigger(event)
    assert_equal(event.trigger.tels_with_trigger, data_type([5, 6]))

    # three LSTs, nothing to change
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = data_type([1, 2, 3])
    assert trigger(event)
    assert_equal(event.trigger.tels_with_trigger, data_type([1, 2, 3]))

    # thee LSTs, plus MSTs, nothing to change
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = data_type([1, 2, 3, 5, 6, 7])
    assert trigger(event)
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
            assert_all_tel_keys(e, expected_tels, ignore={"dl0", "dl1", "dl2", "muon"})


def test_software_trigger_simtel_single_lsts():
    from ctapipe.instrument.trigger import SoftwareTrigger

    path = "dataset://gamma_divergent_LaPalma_baseline_20Zd_180Az_prod3_test.simtel.gz"

    # remove 3 LSTs, so that we trigger the 1-LST condition
    allowed_tels = [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    expected = [
        [12, 16],
        [],
        [],
        [],
        [],
        [],
        [5, 6, 7, 12, 15, 16, 17, 18],
        [13, 14],
        [],
        [7, 12],
        [5, 17],
        [13, 19],
        [],
        [],
        [5, 11, 18],
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
            print(e.trigger.tels_with_trigger)
            trigger(e)
            print(e.trigger.tels_with_trigger, expected_tels)
            assert_equal(e.trigger.tels_with_trigger, expected_tels)
            assert_all_tel_keys(e, expected_tels, ignore={"dl0", "dl1", "dl2", "muon"})


def test_software_trigger_simtel_process(tmp_path):
    from ctapipe.core import run_tool
    from ctapipe.io import TableLoader
    from ctapipe.tools.process import ProcessorTool

    path = "dataset://gamma_divergent_LaPalma_baseline_20Zd_180Az_prod3_test.simtel.gz"
    config = dict(
        ProcessorTool=dict(
            EventSource=dict(
                focal_length_choice="EQUIVALENT",
                # remove 3 LSTs, so that we trigger the 1-LST condition
                allowed_tels=(1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
            ),
            SoftwareTrigger=dict(
                min_telescopes=2,
                min_telescopes_of_type=[
                    ("type", "*", 0),
                    ("type", "LST*", 2),
                ],
            ),
        )
    )

    output_path = tmp_path / "software_trigger.dl1.h5"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    run_tool(
        ProcessorTool(),
        [f"--input={path}", f"--output={output_path}", f"--config={config_path}"],
    )

    del config["ProcessorTool"]["SoftwareTrigger"]
    output_path_no_software_trigger = tmp_path / "no_software_trigger.dl1.h5"
    config_path = tmp_path / "config_no_software_trigger.json"
    config_path.write_text(json.dumps(config))

    run_tool(
        ProcessorTool(),
        [
            f"--input={path}",
            f"--output={output_path_no_software_trigger}",
            f"--config={config_path}",
        ],
    )

    with TableLoader(
        output_path,
        focal_length_choice="EQUIVALENT",
    ) as loader:
        events_trigger = loader.read_telescope_events(
            "LST_LST_LSTCam",
            dl2=False,
            true_parameters=False,
        )

    with TableLoader(
        output_path_no_software_trigger,
        focal_length_choice="EQUIVALENT",
    ) as loader:
        events_no_trigger = loader.read_telescope_events(
            "LST_LST_LSTCam",
            dl2=False,
            true_parameters=False,
        )

    assert len(events_no_trigger) > len(events_trigger)


def test_different_telescope_type_same_string(subarray_prod5_paranal):
    """
    Test that for a subarray that contains two different telescope types
    with the same string representation, the subarray trigger is working
    as expected, treating these telescopes as a single group.
    """
    from ctapipe.instrument.trigger import SoftwareTrigger

    lst234 = deepcopy(subarray_prod5_paranal.tel[2])
    # make LST-1 slightly different from LSTs 2-4
    lst1 = deepcopy(subarray_prod5_paranal.tel[1])
    lst1.optics.mirror_area *= 0.9

    assert lst1 != lst234

    subarray = SubarrayDescription(
        name="test",
        tel_positions={
            tel_id: subarray_prod5_paranal.positions[tel_id] for tel_id in (1, 2, 3, 4)
        },
        tel_descriptions={
            1: lst1,
            2: lst234,
            3: lst234,
            4: lst234,
        },
        reference_location=subarray_prod5_paranal.reference_location,
    )

    trigger = SoftwareTrigger(
        subarray=subarray,
        min_telescopes=2,
        min_telescopes_of_type=[
            ("type", "*", 0),
            ("type", "LST*", 2),
        ],
    )

    # all four LSTs, nothing to change
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = [1, 2, 3, 4]
    assert trigger(event)
    assert_equal(event.trigger.tels_with_trigger, [1, 2, 3, 4])

    # two LSTs, nothing to change
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = [1, 2]
    assert trigger(event)
    assert_equal(event.trigger.tels_with_trigger, [1, 2])

    # two LSTs, nothing to change
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = [1, 3]
    assert trigger(event)
    assert_equal(event.trigger.tels_with_trigger, [1, 3])

    # one LST, remove
    event = ArrayEventContainer()
    event.trigger.tels_with_trigger = [
        1,
    ]
    assert not trigger(event)
    assert_equal(event.trigger.tels_with_trigger, [])
