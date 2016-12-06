from ctapipe.utils.CutFlow import CutFlow, \
                                  UndefinedCutException, \
                                  PureCountingCutException
import numpy as np
from pytest import raises


def test_CutFlow():
    flow = CutFlow("TestFlow")
    flow.set_cut("smaller5", lambda x: x < 5)
    flow.add_cut("smaller3", lambda x: x < 3)

    flow.count("noCuts")
    if flow.cut("smaller5", 3):
        flow.count("something")
        flow.cut("smaller3", 3)

    flow.count("noCuts")
    if flow.cut("smaller5", 1):
        flow.count("something")
        flow.cut("smaller3", 1)

    flow["noCuts"]
    if flow.cut("smaller5", 6):
        # note: not counted since previous cut fails
        flow.count("something")
        flow.cut("smaller3", 6)

    flow.count("noCuts")
    if flow.cut("smaller5", 4):
        pass

    t = flow(sort_column=1)
    assert np.all(t["selected Events"] == [4, 3, 2, 1])

    with raises(UndefinedCutException):
        flow.cut("undefined", 5)

    with raises(PureCountingCutException):
        flow.cut("noCuts")


if __name__ == "__main__":
    test_CutFlow()


