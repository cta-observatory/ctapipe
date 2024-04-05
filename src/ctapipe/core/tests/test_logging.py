def test_colors():
    """Test applying colors"""
    from ctapipe.core.logging import apply_colors

    assert apply_colors("DEBUG") == "\x1b[1;34mDEBUG\x1b[0m"
    assert apply_colors("INFO") == "\x1b[1;32mINFO\x1b[0m"
    assert apply_colors("WARNING") == "\x1b[1;33mWARNING\x1b[0m"
    assert apply_colors("ERROR") == "\x1b[1;31mERROR\x1b[0m"
    assert apply_colors("CRITICAL") == "\x1b[1;35mCRITICAL\x1b[0m"

    # unknown log-level, regression test for #2504
    assert apply_colors("FOO") == "\x1b[1mFOO\x1b[0m"
