import pytest
import traitlets


def test_plugin():
    """Test we can use the dummy event source provided by the test plugin"""
    from ctapipe.io import EventSource

    try:
        es_name = EventSource("test.plugin").__class__.__name__
    except traitlets.traitlets.TraitError:
        pytest.fail(
            "plugin event source not found, did you run `pip install -e ./test_plugin`?"
        )

    assert es_name == "PluginEventSource"
