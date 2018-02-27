from ctapipe.core import Component
from ctapipe.core.traits import Path, TraitError

from pytest import raises
import tempfile


def test_path_exists():

    class C1(Component):
        p = Path(exists=False)

    c1 = C1()
    c1.p = 'test'

    with tempfile.NamedTemporaryFile() as f:
        with raises(TraitError):
            c1.p = f.name

    class C2(Component):
        p = Path(exists=True)

    c2 = C2()

    with tempfile.TemporaryDirectory() as d:
        c2.p = d

    with tempfile.NamedTemporaryFile() as f:
        c2.p = f.name


def test_path_directory_ok():

    class C(Component):
        p = Path(exists=True, directory_ok=False)

    c = C()

    with raises(TraitError):
        c.p = 'lknasdlakndlandslknalkndslakndslkan'

    with tempfile.TemporaryDirectory() as d:
        with raises(TraitError):
            c.p = d

    with tempfile.NamedTemporaryFile() as f:
        c.p = f.name


def test_path_file_ok():

    class C(Component):
        p = Path(exists=True, file_ok=False)

    c = C()

    with raises(TraitError):
        c.p = 'lknasdlakndlandslknalkndslakndslkan'

    with tempfile.TemporaryDirectory() as d:
        c.p = d

    with tempfile.NamedTemporaryFile() as f:
        with raises(TraitError):
            c.p = f.name
