def test_plugin():
    """Test we can use the dummy event source provided by the test plugin"""
    from ctapipe.io import EventSource

    assert EventSource("test.plugin").__class__.__name__ == "PluginEventSource"
