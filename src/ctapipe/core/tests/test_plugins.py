from importlib.metadata import EntryPoint, EntryPoints

import pytest

from ctapipe.core import plugins


class _RealTargetContainer:
    """Module-level so it's a real importable target for EntryPoint.load()."""


class _FakeEntryPoint:
    """Minimal stand-in for importlib.metadata.EntryPoint."""

    def __init__(self, name, value, obj, group="some_group"):
        self.name = name
        self.value = value
        self.group = group
        self._obj = obj

    def load(self):
        return self._obj


class _FakeEntryPoints:
    """Stand-in for importlib.metadata.EntryPoints backed by an explicit
    list of fake entry points, filtered the same way .select() would be.
    """

    def __init__(self, entry_points=()):
        self._entry_points = list(entry_points)

    def select(self, *, group, name=None):
        return [
            ep
            for ep in self._entry_points
            if ep.group == group and (name is None or ep.name == name)
        ]


@pytest.fixture(autouse=True)
def _clear_entry_point_cache():
    plugins.resolve_entry_point.cache_clear()
    yield
    plugins.resolve_entry_point.cache_clear()


@pytest.fixture
def real_drive_entry_point(monkeypatch):
    """Register a real importlib.metadata.EntryPoint pointing at
    _RealTargetContainer, under group "some_group" / name "drive"."""
    ep = EntryPoint(
        name="drive",
        value="ctapipe.core.tests.test_plugins:_RealTargetContainer",
        group="some_group",
    )
    monkeypatch.setattr(plugins, "installed_entry_points", EntryPoints([ep]))


def test_resolve_entry_point_none_when_unregistered(monkeypatch):
    monkeypatch.setattr(plugins, "installed_entry_points", _FakeEntryPoints())

    assert plugins.resolve_entry_point("some_group", "some_name") is None


def test_resolve_entry_point_returns_loaded_object(monkeypatch):
    sentinel = object()
    fake_entry_points = _FakeEntryPoints(
        [_FakeEntryPoint("drive", "some.module:Drive", sentinel)]
    )
    monkeypatch.setattr(plugins, "installed_entry_points", fake_entry_points)

    assert plugins.resolve_entry_point("some_group", "drive") is sentinel


def test_resolve_entry_point_raises_on_duplicate_registration(monkeypatch):
    ep_a = _FakeEntryPoint("drive", "pkg_a:Drive", object())
    ep_b = _FakeEntryPoint("drive", "pkg_b:Drive", object())
    monkeypatch.setattr(
        plugins, "installed_entry_points", _FakeEntryPoints([ep_a, ep_b])
    )

    with pytest.raises(RuntimeError, match="Multiple plugins registered"):
        plugins.resolve_entry_point("some_group", "drive")


def test_resolve_entry_point_loads_real_entry_point(real_drive_entry_point):
    resolved = plugins.resolve_entry_point("some_group", "drive")

    assert resolved is _RealTargetContainer


def test_lazy_entry_point_returns_none_when_unregistered(monkeypatch):
    monkeypatch.setattr(plugins, "installed_entry_points", _FakeEntryPoints())

    factory = plugins.lazy_entry_point("some_group", "drive")

    assert factory() is None


def test_lazy_entry_point_instantiates_registered_class(monkeypatch):
    class Target:
        pass

    ep = _FakeEntryPoint("drive", "some.module:Target", Target)
    monkeypatch.setattr(plugins, "installed_entry_points", _FakeEntryPoints([ep]))

    factory = plugins.lazy_entry_point("some_group", "drive")

    assert isinstance(factory(), Target)


def test_lazy_entry_point_raises_on_duplicate_registration(monkeypatch):
    ep_a = _FakeEntryPoint("drive", "pkg_a:Drive", object())
    ep_b = _FakeEntryPoint("drive", "pkg_b:Drive", object())
    monkeypatch.setattr(
        plugins, "installed_entry_points", _FakeEntryPoints([ep_a, ep_b])
    )

    factory = plugins.lazy_entry_point("some_group", "drive")

    with pytest.raises(RuntimeError, match="Multiple plugins registered"):
        factory()


def test_lazy_entry_point_does_not_resolve_until_called(monkeypatch):
    """Building the factory must not touch installed_entry_points at all --
    resolution happens on factory(), not on lazy_entry_point()."""

    class _ExplodingEntryPoints:
        def select(self, **kwargs):
            raise AssertionError("resolve_entry_point should not run yet")

    monkeypatch.setattr(plugins, "installed_entry_points", _ExplodingEntryPoints())

    factory = plugins.lazy_entry_point("some_group", "drive")  # must not raise

    monkeypatch.setattr(plugins, "installed_entry_points", _FakeEntryPoints())
    assert factory() is None  # now it's safe to resolve, and finds nothing


def test_lazy_entry_point_instantiates_real_entry_point(real_drive_entry_point):
    factory = plugins.lazy_entry_point("some_group", "drive")
    instance = factory()

    assert isinstance(instance, _RealTargetContainer)
