import numpy as np
from numpy.testing import assert_array_equal


def test_fixed_point_unsigned():
    from ctapipe.io.hdf5tableio import FixedPointColumnTransform

    tr = FixedPointColumnTransform(
        scale=10, offset=0, source_dtype=np.float32, target_dtype=np.uint16
    )

    iinfo = np.iinfo(tr.target_dtype)
    assert tr.posinf == iinfo.max
    assert tr.nan == iinfo.max - 1
    assert tr.neginf == iinfo.max - 2

    values = {
        0: 0,
        0.1: 1,
        0.16: 2,
        -np.inf: tr.neginf,
        np.nan: tr.nan,
        np.inf: tr.posinf,
        (tr.maxval + 1) / 10: tr.posinf,
        -1: tr.neginf,
    }

    for v, e in values.items():
        transformed = tr(v)
        assert transformed.dtype == tr.target_dtype
        assert transformed == e

    # test array
    v = np.array(list(values.keys()))
    e = np.array(list(values.values()))
    assert_array_equal(tr(v), e)


def test_fixed_point_unsigned_offset():
    from ctapipe.io.hdf5tableio import FixedPointColumnTransform

    tr = FixedPointColumnTransform(
        scale=10, offset=400, source_dtype=np.float32, target_dtype=np.uint16
    )

    values = {
        -40: 0,
        -30: 100,
        0: 400,
        0.1: 401,
        0.16: 402,
        -np.inf: tr.neginf,
        np.nan: tr.nan,
        np.inf: tr.posinf,
        (tr.maxval - 400) / 10: tr.maxval,
        (tr.maxval - 399) / 10: tr.posinf,
        -40.1: tr.neginf,
    }

    # test single values
    for v, e in values.items():
        transformed = tr(v)
        assert transformed.dtype == tr.target_dtype
        assert transformed == e, f"Unexpected outcome transforming {v}"

    # test array
    v = np.array(list(values.keys()))
    e = np.array(list(values.values()))
    assert_array_equal(tr(v), e)


def test_fixed_point_signed():
    from ctapipe.io.hdf5tableio import FixedPointColumnTransform

    tr = FixedPointColumnTransform(
        scale=10, offset=0, source_dtype=np.float32, target_dtype=np.int16
    )

    iinfo = np.iinfo(tr.target_dtype)
    assert tr.posinf == iinfo.max
    assert tr.nan == iinfo.min + 1
    assert tr.neginf == iinfo.min

    values = {
        -np.inf: tr.neginf,
        (tr.minval - 1) / 10: tr.neginf,
        tr.minval / 10: tr.minval,
        -50: -500,
        0: 0,
        50: 500,
        tr.maxval / 10: tr.maxval,
        (tr.maxval + 1) / 10: tr.posinf,
        np.inf: tr.posinf,
    }

    # test single values
    for v, e in values.items():
        transformed = tr(v)
        assert transformed.dtype == tr.target_dtype
        assert transformed == e, f"Unexpected outcome transforming {v}"

    # test array
    v = np.array(list(values.keys()))
    e = np.array(list(values.values()))
    assert_array_equal(tr(v), e)


def test_utf8_max_max_len():
    from ctapipe.io.tableio import encode_utf8_max_len

    assert encode_utf8_max_len("hello", 4) == b"hell"

    # each of these is 2 bytes
    assert encode_utf8_max_len("αβ", 4) == "αβ".encode("utf-8")
    assert encode_utf8_max_len("αβγ", 4) == "αβ".encode("utf-8")
    assert encode_utf8_max_len("αβγ", 3) == "α".encode("utf-8")

    emoji = "foo \N{FACE WITH MEDICAL MASK}"
    assert encode_utf8_max_len(emoji, 8) == emoji.encode("utf-8")
    assert encode_utf8_max_len(emoji, 5) == b"foo "


def test_string_transform():
    from ctapipe.io.tableio import StringTransform

    trafo = StringTransform(10)

    assert trafo("hello") == b"hello"
    assert trafo("12345678910") == b"1234567891"

    strings = np.array(["foo", "bar"])
    assert np.all(trafo(strings) == np.array([b"foo", b"bar"]))

    assert trafo.inverse(b"hello") == "hello"
    assert trafo.inverse("α".encode("utf-8")) == "α"

    nonascii = np.array(["αβßä", "abc"])
    assert np.all(trafo.inverse(trafo(nonascii)) == nonascii)


def test_string_truncated():
    from ctapipe.io.tableio import StringTransform

    trafo = StringTransform(5)
    assert trafo("hello world") == b"hello"

    # 6 bytes, make sure we only write valid utf-8
    assert trafo("ααα") == "αα".encode("utf-8")
