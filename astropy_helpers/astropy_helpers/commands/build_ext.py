import errno
import os
import shutil

from distutils.core import Extension
from setuptools.command.build_ext import build_ext as SetuptoolsBuildExt

from ..utils import get_numpy_include_path, invalidate_caches
from ..version_helpers import get_pkg_version_module


def should_build_with_cython(package, release=None):
    """Returns the previously used Cython version (or 'unknown' if not
    previously built) if Cython should be used to build extension modules from
    pyx files.  If the ``release`` parameter is not specified an attempt is
    made to determine the release flag from `astropy.version`.
    """

    from ..setup_helpers import _module_state

    try:
        version_module = __import__(package + '.cython_version',
                                    fromlist=['release', 'cython_version'])
    except ImportError:
        version_module = None

    if release is None and version_module is not None:
        try:
            release = version_module.release
        except AttributeError:
            pass

    try:
        cython_version = version_module.cython_version
    except AttributeError:
        cython_version = 'unknown'

    # Only build with Cython if, of course, Cython is installed, we're in a
    # development version (i.e. not release) or the Cython-generated source
    # files haven't been created yet (cython_version == 'unknown'). The latter
    # case can happen even when release is True if checking out a release tag
    # from the repository
    if (_module_state['have_cython'] and
            (not release or cython_version == 'unknown')):
        return cython_version
    else:
        return False


# TODO: I think this can be reworked without having to create the class
# programmatically.
def generate_build_ext_command(packagename, release):
    """
    Creates a custom 'build_ext' command that allows for manipulating some of
    the C extension options at build time.  We use a function to build the
    class since the base class for build_ext may be different depending on
    certain build-time parameters (for example, we may use Cython's build_ext
    instead of the default version in distutils).

    Uses the default distutils.command.build_ext by default.
    """

    uses_cython = should_build_with_cython(packagename, release)

    if uses_cython:
        from Cython.Distutils import build_ext as basecls
    else:
        basecls = SetuptoolsBuildExt

    attrs = dict(basecls.__dict__)
    orig_run = getattr(basecls, 'run', None)
    orig_finalize = getattr(basecls, 'finalize_options', None)

    def finalize_options(self):
        # Add a copy of the _compiler.so module as well, but only if there are
        # in fact C modules to compile (otherwise there's no reason to include
        # a record of the compiler used)
        # Note, self.extensions may not be set yet, but
        # self.distribution.ext_modules is where any extension modules passed
        # to setup() can be found
        extensions = self.distribution.ext_modules
        if extensions:
            src_path = os.path.relpath(
                os.path.join(os.path.dirname(__file__), 'src'))
            shutil.copy2(os.path.join(src_path, 'compiler.c'),
                         os.path.join(self.package_name, '_compiler.c'))
            ext = Extension(self.package_name + '._compiler',
                            [os.path.join(self.package_name, '_compiler.c')])
            extensions.insert(0, ext)

        if orig_finalize is not None:
            orig_finalize(self)

        # Generate
        if self.uses_cython:
            try:
                from Cython import __version__ as cython_version
            except ImportError:
                # This shouldn't happen if we made it this far
                cython_version = None

            if (cython_version is not None and
                    cython_version != self.uses_cython):
                self.force_rebuild = True
                # Update the used cython version
                self.uses_cython = cython_version

        # Regardless of the value of the '--force' option, force a rebuild if
        # the debug flag changed from the last build
        if self.force_rebuild:
            self.force = True

    def run(self):
        # For extensions that require 'numpy' in their include dirs, replace
        # 'numpy' with the actual paths
        np_include = get_numpy_include_path()
        for extension in self.extensions:
            if 'numpy' in extension.include_dirs:
                idx = extension.include_dirs.index('numpy')
                extension.include_dirs.insert(idx, np_include)
                extension.include_dirs.remove('numpy')

            # Replace .pyx with C-equivalents, unless c files are missing
            for jdx, src in enumerate(extension.sources):
                if src.endswith('.pyx'):
                    pyxfn = src
                    cfn = src[:-4] + '.c'
                elif src.endswith('.c'):
                    pyxfn = src[:-2] + '.pyx'
                    cfn = src

                if not os.path.isfile(pyxfn):
                    continue

                if self.uses_cython:
                    extension.sources[jdx] = pyxfn
                else:
                    if os.path.isfile(cfn):
                        extension.sources[jdx] = cfn
                    else:
                        msg = (
                            'Could not find C file {0} for Cython file {1} '
                            'when building extension {2}. Cython must be '
                            'installed to build from a git checkout.'.format(
                                cfn, pyxfn, extension.name))
                        raise IOError(errno.ENOENT, msg, cfn)

        if orig_run is not None:
            # This should always be the case for a correctly implemented
            # distutils command.
            orig_run(self)

        # Update cython_version.py if building with Cython
        try:
            cython_version = get_pkg_version_module(
                    packagename, fromlist=['cython_version'])[0]
        except (AttributeError, ImportError):
            cython_version = 'unknown'
        if self.uses_cython and self.uses_cython != cython_version:
            package_dir = os.path.relpath(packagename)
            cython_py = os.path.join(package_dir, 'cython_version.py')
            with open(cython_py, 'w') as f:
                f.write('# Generated file; do not modify\n')
                f.write('cython_version = {0!r}\n'.format(self.uses_cython))

            if os.path.isdir(self.build_lib):
                # The build/lib directory may not exist if the build_py command
                # was not previously run, which may sometimes be the case
                self.copy_file(cython_py,
                               os.path.join(self.build_lib, cython_py),
                               preserve_mode=False)

            invalidate_caches()

    attrs['run'] = run
    attrs['finalize_options'] = finalize_options
    attrs['force_rebuild'] = False
    attrs['uses_cython'] = uses_cython
    attrs['package_name'] = packagename
    attrs['user_options'] = basecls.user_options[:]
    attrs['boolean_options'] = basecls.boolean_options[:]

    return type('build_ext', (basecls, object), attrs)
