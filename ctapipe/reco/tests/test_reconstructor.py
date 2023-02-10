"""
Test for Reconstructor base class
"""
import pytest


def test_plugin(subarray_prod5_paranal):
    from ctapipe.reco import Reconstructor

    subarray = subarray_prod5_paranal

    try:
        reconstructor = Reconstructor.from_name("PluginReconstructor", subarray)
    except KeyError:
        pytest.fail(
            "plugin event source not found, did you run `pip install -e ./test_plugin`?"
        )

    assert reconstructor.__module__ == "ctapipe_test_plugin"
    assert reconstructor.__class__.__name__ == "PluginReconstructor"
