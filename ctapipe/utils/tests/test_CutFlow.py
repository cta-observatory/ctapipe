from ctapipe.utils.CutFlow import CutFlow, \
                                  UndefinedCutException, \
                                  PureCountingCutException
import numpy as np
from pytest import raises


def test_CutFlow():
    flow = CutFlow("TestFlow")
    # set_cut and add_cut a aliases
    flow.set_cut("smaller5", lambda x: x < 5)
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

    with raises(UndefinedCutException):
        flow.cut("undefined", 5)

    with raises(PureCountingCutException):
        flow.cut("noCuts")


if __name__ == "__main__":
    test_CutFlow()


