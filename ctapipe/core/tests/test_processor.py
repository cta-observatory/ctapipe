from ctapipe.core import Processor
from pytest import raises


def test_abstract():

    with raises(TypeError):
        Processor()


def test_callable():

    class MyProcessor(Processor):

        def __call__(self, a, b):
            return a + b

    p = MyProcessor()
    p(1, 2)
