from collections import OrderedDict
import numpy as np
from pytest import raises
from ctapipe.utils.CutFlow import CutFlow, UndefinedCut, PureCountingCut


def smaller2(x):
    return x < 2


def smaller3(x):
    return x < 3


def smaller5(x):
    return x < 5


def test_CutFlow():
    flow = CutFlow("TestFlow")
    # set_cut and add_cut a aliases
    flow.set_cut("smaller5", smaller5)
    flow.add_cut("smaller3", lambda x: x < 3)

    for i in range(2, 6):
        flow.count("noCuts")
        # .keep counts if the function returns True,
        # i.e. when we "keep" the event
        if flow.keep("smaller5", i):
            # .cut counts if the function returns False,
            # i.e. when we do NOT "cut" the event
            if flow.cut("smaller3", i):
                pass
            else:
                # do something else that could fail or be rejected
                try:
                    assert i == 3
                    flow.count("something")
                except:
                    pass


    t = flow(sort_column=1)
    assert np.all(t["selected Events"] == [4, 3, 2, 1])

    with raises(UndefinedCut):
        flow.cut("undefined", 5)

    with raises(PureCountingCut):
        flow.cut("noCuts")


def test_set_cuts_clear():
    flow = CutFlow("TestFlow")
    flow.set_cut("smaller5", smaller5)

    flow.set_cuts(OrderedDict([
        ("smaller3", smaller3),
        ("smaller2", smaller2)
    ]), clear=True)

    assert flow.cuts == OrderedDict([
        ("smaller3", [smaller3, 0]),
        ("smaller2", [smaller2, 0])
    ])


def test_set_cuts_no_clear():
    flow = CutFlow("TestFlow")
    flow.set_cut("smaller5", smaller5)

    flow.set_cuts(OrderedDict([
        ("smaller3", smaller3),
        ("smaller2", smaller2)
    ]), clear=False)


    assert flow.cuts == OrderedDict([
        ("smaller5", [smaller5, 0]),
        ("smaller3", [smaller3, 0]),
        ("smaller2", [smaller2, 0])
    ])


if __name__ == "__main__":
    test_CutFlow()
    test_set_cuts_clear()
    test_set_cuts_no_clear()
