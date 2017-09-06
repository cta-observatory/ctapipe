from pytest import raises


def test_none():
    from ctapipe.core import Container, Field

    class TestContainer(Container):
        test = Field('test')


    TestContainer(test=5)

    with raises(ValueError):
        TestContainer()

    class TestContainer(Container):
        test = Field('test', allow_none=True)

    TestContainer()


def test_none_existent():
    from ctapipe.core import Container, Field

    class TestContainer(Container):
        test = Field('test', allow_none=True)

    t = TestContainer()

    with raises(AttributeError):
        t.x = 10


def test_array():
    from ctapipe.core import Container, ArrayField
    import numpy as np

    class TestContainer(Container):
        x = ArrayField('x')

    t = TestContainer(x=np.zeros(5))

    t.x = 5

    assert t.x == np.array(5)

    class TestContainer(Container):
        x = ArrayField('x', dtype=int, shape=(2, ))

    t = TestContainer(x=[1.0, 2.4])

    assert t.x.dtype == np.dtype(int)
    assert np.all(t.x == np.array([1, 2]))

    with raises(ValueError):
        t.x = [1, 2, 3]


def test_quantity():
    from ctapipe.core import Container, QuantityField
    import astropy.units as u
    import numpy as np

    class TestContainer(Container):
        x = QuantityField('x', default=5 * u.m, unit=u.m)

    t = TestContainer()

    t.x = 4 * u.cm

    with raises(ValueError):
        t.x = 5 * u.A

    class TestContainer(Container):
        x = QuantityField('x', dtype=int, shape=(2, ), unit=u.m)

    t = TestContainer(x=[1, 2] * u.m)

    assert t.x.dtype == np.dtype(int)

    with raises(ValueError):
        t = TestContainer(x=[1, 2, 3] * u.m)


def test_time():
    from ctapipe.core import Container, TimeField
    from astropy.time import Time

    class TestContainer(Container):
        time = TimeField('observation time')

    t = TestContainer(time=Time.now())

    t.time = '2017-01-01 20:00'

    with raises(ValueError):
        t.time = 5
