from ctapipe.compat import StrEnum


def test_str_enum():
    class Foo(StrEnum):
        A = "A"
        B = "B"

    assert f"{Foo.A}" == "A"
