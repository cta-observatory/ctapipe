"""Test ctapipe-info functionality."""


def test_info_version(script_runner):

    result = script_runner.run(
        "ctapipe-info",
        "--version",
    )

    assert result.success
    assert result.stderr == ""


def test_info_tools(script_runner):

    result = script_runner.run(
        "ctapipe-info",
        "--tools",
    )

    assert result.success
    assert result.stderr == ""


def test_info_dependencies(script_runner):

    result = script_runner.run(
        "ctapipe-info",
        "--dependencies",
    )

    assert result.success
    assert result.stderr == ""


def test_info_system(script_runner):

    result = script_runner.run(
        "ctapipe-info",
        "--system",
    )

    assert result.success
    assert result.stderr == ""


def test_info_plugins(script_runner):

    result = script_runner.run(
        "ctapipe-info",
        "--plugins",
    )

    assert result.success
    assert result.stderr == ""


def test_info_eventsources(script_runner):

    result = script_runner.run(
        "ctapipe-info",
        "--event-sources",
    )

    assert result.success
    assert result.stderr == ""


def test_info_all(script_runner):

    result = script_runner.run(
        "ctapipe-info",
        "--all",
    )

    assert result.success
    assert result.stderr == ""
