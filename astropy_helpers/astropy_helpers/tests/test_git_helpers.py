import glob
import imp
import os
import pkgutil
import re
import sys
import tarfile

from . import *


PY3 = sys.version_info[0] == 3

if PY3:
    _text_type = str
else:
    _text_type = unicode


_DEV_VERSION_RE = re.compile(r'\d+\.\d+(?:\.\d+)?\.dev(\d+)')


TEST_VERSION_SETUP_PY = """\
#!/usr/bin/env python

from setuptools import setup

NAME = '_eva_'
VERSION = {version!r}
RELEASE = 'dev' not in VERSION

from astropy_helpers.git_helpers import get_git_devstr
from astropy_helpers.version_helpers import generate_version_py

if not RELEASE:
    VERSION += get_git_devstr(False)

generate_version_py(NAME, VERSION, RELEASE, False, uses_git=not RELEASE)

setup(name=NAME, version=VERSION, packages=['_eva_'])
"""


TEST_VERSION_INIT = """\
try:
    from .version import version as __version__
    from .version import githash as __githash__
except ImportError:
    __version__ = __githash__ = ''
"""


@pytest.fixture
def version_test_package(tmpdir, request):
    def make_test_package(version='42.42.dev'):
        test_package = tmpdir.mkdir('test_package')
        test_package.join('setup.py').write(
            TEST_VERSION_SETUP_PY.format(version=version))
        test_package.mkdir('_eva_').join('__init__.py').write(TEST_VERSION_INIT)
        with test_package.as_cwd():
            run_cmd('git', ['init'])
            run_cmd('git', ['add', '--all'])
            run_cmd('git', ['commit', '-m', 'test package'])

        if '' in sys.path:
            sys.path.remove('')

        sys.path.insert(0, '')

        def finalize():
            cleanup_import('_eva_')

        request.addfinalizer(finalize)

        return test_package

    return make_test_package


def test_update_git_devstr(version_test_package, capsys):
    """Tests that the commit number in the package's version string updates
    after git commits even without re-running setup.py.
    """

    # We have to call version_test_package to actually create the package
    test_pkg = version_test_package()

    with test_pkg.as_cwd():
        run_setup('setup.py', ['--version'])

        stdout, stderr = capsys.readouterr()
        version = stdout.strip()

        m = _DEV_VERSION_RE.match(version)
        assert m, (
            "Stdout did not match the version string pattern:"
            "\n\n{0}\n\nStderr:\n\n{1}".format(stdout, stderr))
        revcount = int(m.group(1))

        import _eva_
        assert _eva_.__version__ == version

        # Make a silly git commit
        with open('.test', 'w'):
            pass

        run_cmd('git', ['add', '.test'])
        run_cmd('git', ['commit', '-m', 'test'])

        import _eva_.version
        imp.reload(_eva_.version)

    # Previously this checked packagename.__version__, but in order for that to
    # be updated we also have to re-import _astropy_init which could be tricky.
    # Checking directly that the packagename.version module was updated is
    # sufficient:
    m = _DEV_VERSION_RE.match(_eva_.version.version)
    assert m
    assert int(m.group(1)) == revcount + 1

    # This doesn't test astropy_helpers.get_helpers.update_git_devstr directly
    # since a copy of that function is made in packagename.version (so that it
    # can work without astropy_helpers installed).  In order to get test
    # coverage on the actual astropy_helpers copy of that function just call it
    # directly and compare to the value in packagename
    from astropy_helpers.git_helpers import update_git_devstr

    newversion = update_git_devstr(version, path=str(test_pkg))
    assert newversion == _eva_.version.version


def test_version_update_in_other_repos(version_test_package, tmpdir):
    """
    Regression test for https://github.com/astropy/astropy-helpers/issues/114
    and for https://github.com/astropy/astropy-helpers/issues/107
    """

    test_pkg = version_test_package()

    with test_pkg.as_cwd():
        run_setup('setup.py', ['build'])

    # Add the path to the test package to sys.path for now
    sys.path.insert(0, str(test_pkg))
    try:
        import _eva_
        m = _DEV_VERSION_RE.match(_eva_.__version__)
        assert m
        correct_revcount = int(m.group(1))

        with tmpdir.as_cwd():
            testrepo = tmpdir.mkdir('testrepo')
            testrepo.chdir()
            # Create an empty git repo
            run_cmd('git', ['init'])

            import _eva_.version
            imp.reload(_eva_.version)
            m = _DEV_VERSION_RE.match(_eva_.version.version)
            assert m
            assert int(m.group(1)) == correct_revcount
            correct_revcount = int(m.group(1))

            # Add several commits--more than the revcount for the _eva_ package
            for idx in range(correct_revcount + 5):
                test_filename = '.test' + str(idx)
                testrepo.ensure(test_filename)
                run_cmd('git', ['add', test_filename])
                run_cmd('git', ['commit', '-m', 'A message'])

            import _eva_.version
            imp.reload(_eva_.version)
            m = _DEV_VERSION_RE.match(_eva_.version.version)
            assert m
            assert int(m.group(1)) == correct_revcount
            correct_revcount = int(m.group(1))
    finally:
        sys.path.remove(str(test_pkg))


@pytest.mark.parametrize('version', ['1.0.dev', '1.0'])
def test_installed_git_version(version_test_package, version, tmpdir, capsys):
    """
    Test for https://github.com/astropy/astropy-helpers/issues/87

    Ensures that packages installed with astropy_helpers have a correct copy
    of the git hash of the installed commit.
    """

    # To test this, it should suffice to build a source dist, unpack it
    # somewhere outside the git repository, and then do a build and import
    # from the build directory--no need to "install" as such

    test_pkg = version_test_package(version)

    with test_pkg.as_cwd():
        run_setup('setup.py', ['build'])

        try:
            import _eva_
            githash = _eva_.__githash__
            assert githash and isinstance(githash, _text_type)
            # Ensure that it does in fact look like a git hash and not some
            # other arbitrary string
            assert re.match(r'[0-9a-f]{40}', githash)
        finally:
            cleanup_import('_eva_')

        run_setup('setup.py', ['sdist', '--dist-dir=dist', '--formats=gztar'])

        tgzs = glob.glob(os.path.join('dist', '*.tar.gz'))
        assert len(tgzs) == 1

        tgz = test_pkg.join(tgzs[0])

    build_dir = tmpdir.mkdir('build_dir')
    tf = tarfile.open(str(tgz), mode='r:gz')
    tf.extractall(str(build_dir))

    with build_dir.as_cwd():
        pkg_dir = glob.glob('_eva_-*')[0]
        os.chdir(pkg_dir)
        run_setup('setup.py', ['build'])

        try:
            import _eva_
            loader = pkgutil.get_loader('_eva_')
            # Ensure we are importing the 'packagename' that was just unpacked
            # into the build_dir
            if sys.version_info[:2] != (3, 3):
                # Skip this test on Python 3.3 wherein the SourceFileLoader
                # has a bug where get_filename() does not return an absolute
                # path
                assert loader.get_filename().startswith(str(build_dir))
            assert _eva_.__githash__ == githash
        finally:
            cleanup_import('_eva_')
