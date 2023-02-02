import pytest
import traitlets


def test_plugin():
    """Test we can use the dummy event source provided by the test plugin"""
    from ctapipe.io import EventSource

    try:
        EventSource("test.plugin").__class__.__name__ == "PluginEventSource"
    except traitlets.traitlets.TraitError:
        pytest.fail(
            "plugin event source not found, did you run pip install -e ./test_plugin"
        )
