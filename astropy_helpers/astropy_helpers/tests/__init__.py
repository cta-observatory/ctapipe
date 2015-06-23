import os
import subprocess as sp
import sys

from setuptools import sandbox

import pytest

from ..utils import extends_doc

PACKAGE_DIR = os.path.dirname(__file__)


def run_cmd(cmd, args, path=None, raise_error=True):
    """
    Runs a shell command with the given argument list.  Changes directory to
    ``path`` if given, otherwise runs the command in the current directory.

    Returns a 3-tuple of (stdout, stderr, exit code)

    If ``raise_error=True`` raise an exception on non-zero exit codes.
    """

    if path is not None:
        # Transparently support py.path objects
        path = str(path)

    p = sp.Popen([cmd] + list(args), stdout=sp.PIPE, stderr=sp.PIPE,
                 cwd=path)
    streams = tuple(s.decode('latin1').strip() for s in p.communicate())
    return_code = p.returncode

    if raise_error and return_code != 0:
        raise RuntimeError(
            "The command `{0}` with args {1!r} exited with code {2}.\n"
            "Stdout:\n\n{3}\n\nStderr:\n\n{4}".format(
                cmd, list(args), return_code, streams[0], streams[1]))

    return streams + (return_code,)


@extends_doc(sandbox.run_setup)
def run_setup(*args, **kwargs):
    """
    In Python 3, on MacOS X, the import cache has to be invalidated otherwise
    new extensions built with ``run_setup`` do not always get picked up.
    """

    try:
        return sandbox.run_setup(*args, **kwargs)
    finally:
        if sys.version_info[:2] >= (3, 3):
            import importlib
            importlib.invalidate_caches()


@pytest.fixture(scope='function', autouse=True)
def reset_setup_helpers(request):
    """
    Saves and restores the global state of the astropy_helpers.setup_helpers
    module between tests.
    """

    mod = __import__('astropy_helpers.setup_helpers', fromlist=[''])

    old_state = mod._module_state.copy()

    def finalizer(old_state=old_state):
        mod = sys.modules.get('astropy_helpers.setup_helpers')
        if mod is not None:
            mod._module_state.update(old_state)

    request.addfinalizer(finalizer)


@pytest.fixture(scope='function', autouse=True)
def reset_distutils_log():
    """
    This is a setup/teardown fixture that ensures the log-level of the
    distutils log is always set to a default of WARN, since different
    settings could affect tests that check the contents of stdout.
    """

    from distutils import log
    log.set_threshold(log.WARN)


@pytest.fixture(scope='module', autouse=True)
def fix_hide_setuptools():
    """
    Workaround for https://github.com/astropy/astropy-helpers/issues/124

    In setuptools 10.0 run_setup was changed in such a way that it sweeps
    away the existing setuptools import before running the setup script.  In
    principle this is nice, but in the practice of testing astropy_helpers
    this is problematic since we're trying to test code that has already been
    imported during the testing process, and which relies on the setuptools
    module that was already in use.
    """

    if hasattr(sandbox, 'hide_setuptools'):
        sandbox.hide_setuptools = lambda: None


TEST_PACKAGE_SETUP_PY = """\
#!/usr/bin/env python

from setuptools import setup

NAME = 'astropy-helpers-test'
VERSION = {version!r}

setup(name=NAME, version=VERSION,
      packages=['_astropy_helpers_test_'],
      zip_safe=False)
"""


@pytest.fixture
def testpackage(tmpdir, version='0.1'):
    """
    This fixture creates a simplified package called _astropy_helpers_test_
    used primarily for testing ah_boostrap, but without using the
    astropy_helpers package directly and getting it confused with the
    astropy_helpers package already under test.
    """

    source = tmpdir.mkdir('testpkg')

    with source.as_cwd():
        source.mkdir('_astropy_helpers_test_')
        init = source.join('_astropy_helpers_test_', '__init__.py')
        init.write('__version__ = {0!r}'.format(version))
        setup_py = TEST_PACKAGE_SETUP_PY.format(version=version)
        source.join('setup.py').write(setup_py)

        # Make the new test package into a git repo
        run_cmd('git', ['init'])
        run_cmd('git', ['add', '--all'])
        run_cmd('git', ['commit', '-m', 'test package'])

    return source


def cleanup_import(package_name):
    """Remove all references to package_name from sys.modules"""

    for k in list(sys.modules):
        if not isinstance(k, str):
            # Some things will actually do this =_=
            continue
        elif k.startswith('astropy_helpers.tests'):
            # Don't delete imported test modules or else the tests will break,
            # badly
            continue
        if k == package_name or k.startswith(package_name + '.'):
            del sys.modules[k]
