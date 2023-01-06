"""
Test for Reconstructor base class
"""
import logging


def test_plugin(subarray_prod5_paranal, caplog):
    from ctapipe.reco import Reconstructor

    subarray = subarray_prod5_paranal
    with caplog.at_level(logging.INFO, logger="ctapipe.core.plugins"):
        reconstructor = Reconstructor.from_name("PluginReconstructor", subarray)

    assert reconstructor.__module__ == "ctapipe_test_plugin"
    assert reconstructor.__class__.__name__ == "PluginReconstructor"

    assert caplog.record_tuples == [
        (
            "ctapipe.core.plugins",
            logging.INFO,
            "Loading ctapipe_reco plugin: ctapipe_test_plugin:PluginReconstructor",
        ),
        (
            "ctapipe.core.plugins",
            logging.INFO,
            "Entrypoint provides: <class 'ctapipe_test_plugin.PluginReconstructor'>",
        ),
    ]
