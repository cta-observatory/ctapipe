# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains a number of utilities for use during
setup/build/packaging that are useful to astropy as a whole.
"""

from __future__ import absolute_import, print_function

import collections
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import traceback

from distutils import log, ccompiler, sysconfig
from distutils.dist import Distribution
from distutils.errors import DistutilsOptionError, DistutilsModuleError
from distutils.core import Extension
from distutils.core import Command
from distutils.command.sdist import sdist as DistutilsSdist

from setuptools import find_packages as _find_packages

from .distutils_helpers import *
from .version_helpers import get_pkg_version_module
from .test_helpers import AstropyTest
from .utils import (silence, walk_skip_hidden, import_file, extends_doc,
                    resolve_name)


from .commands.build_ext import generate_build_ext_command
from .commands.build_py import AstropyBuildPy
from .commands.install import AstropyInstall
from .commands.install_lib import AstropyInstallLib
from .commands.register import AstropyRegister

# This import is not used in this module, but it is included for backwards
# compat with version 0.4, which included this function in the public API
# for this module
from .utils import get_numpy_include_path, write_if_different
from .commands.build_ext import should_build_with_cython

_module_state = {
    'adjusted_compiler': False,
    'registered_commands': None,
    'have_cython': False,
    'have_sphinx': False,
    'package_cache': None,
    'compiler_version_cache': {}
}

try:
    import Cython
    _module_state['have_cython'] = True
except ImportError:
    pass

try:
    import sphinx
    _module_state['have_sphinx'] = True
except ValueError as e:
    # This can occur deep in the bowels of Sphinx's imports by way of docutils
    # and an occurrence of this bug: http://bugs.python.org/issue18378
    # In this case sphinx is effectively unusable
    if 'unknown locale' in e.args[0]:
        log.warn(
            "Possible misconfiguration of one of the environment variables "
            "LC_ALL, LC_CTYPES, LANG, or LANGUAGE.  For an example of how to "
            "configure your system's language environment on OSX see "
            "http://blog.remibergsma.com/2012/07/10/"
            "setting-locales-correctly-on-mac-osx-terminal-application/")
except ImportError:
    pass
except SyntaxError:
    # occurs if markupsafe is recent version, which doesn't support Python 3.2
    pass


PY3 = sys.version_info[0] >= 3


# This adds a new keyword to the setup() function
Distribution.skip_2to3 = []


def adjust_compiler(package):
    """
    This function detects broken compilers and switches to another.  If
    the environment variable CC is explicitly set, or a compiler is
    specified on the commandline, no override is performed -- the purpose
    here is to only override a default compiler.

    The specific compilers with problems are:

        * The default compiler in XCode-4.2, llvm-gcc-4.2,
          segfaults when compiling wcslib.

    The set of broken compilers can be updated by changing the
    compiler_mapping variable.  It is a list of 2-tuples where the
    first in the pair is a regular expression matching the version
    of the broken compiler, and the second is the compiler to change
    to.
    """

    compiler_mapping = [
        (b'i686-apple-darwin[0-9]*-llvm-gcc-4.2', 'clang')
        ]

    if _module_state['adjusted_compiler']:
        return

    # Whatever the result of this function is, it only needs to be run once
    _module_state['adjusted_compiler'] = True

    if 'CC' in os.environ:

        # Check that CC is not set to llvm-gcc-4.2
        c_compiler = os.environ['CC']

        try:
            version = get_compiler_version(c_compiler)
        except OSError:
            msg = textwrap.dedent(
                    """
                    The C compiler set by the CC environment variable:

                        {compiler:s}

                    cannot be found or executed.
                    """.format(compiler=c_compiler))
            log.warn(msg)
            sys.exit(1)

        for broken, fixed in compiler_mapping:
            if re.match(broken, version):
                msg = textwrap.dedent(
                    """Compiler specified by CC environment variable
                    ({compiler:s}:{version:s}) will fail to compile {pkg:s}.
                    Please set CC={fixed:s} and try again.
                    You can do this, for example, by running:

                        CC={fixed:s} python setup.py <command>

                    where <command> is the command you ran.
                    """.format(compiler=c_compiler, version=version,
                               pkg=package, fixed=fixed))
                log.warn(msg)
                sys.exit(1)

        # If C compiler is set via CC, and isn't broken, we are good to go. We
        # should definitely not try accessing the compiler specified by
        # ``sysconfig.get_config_var('CC')`` lower down, because this may fail
        # if the compiler used to compile Python is missing (and maybe this is
        # why the user is setting CC). For example, the official Python 2.7.3
        # MacOS X binary was compiled with gcc-4.2, which is no longer available
        # in XCode 4.
        return

    if get_distutils_build_option('compiler'):
        return

    compiler_type = ccompiler.get_default_compiler()

    if compiler_type == 'unix':

        # We have to get the compiler this way, as this is the one that is
        # used if os.environ['CC'] is not set. It is actually read in from
        # the Python Makefile. Note that this is not necessarily the same
        # compiler as returned by ccompiler.new_compiler()
        c_compiler = sysconfig.get_config_var('CC')

        try:
            version = get_compiler_version(c_compiler)
        except OSError:
            msg = textwrap.dedent(
                    """
                    The C compiler used to compile Python {compiler:s}, and
                    which is normally used to compile C extensions, is not
                    available. You can explicitly specify which compiler to
                    use by setting the CC environment variable, for example:

                        CC=gcc python setup.py <command>

                    or if you are using MacOS X, you can try:

                        CC=clang python setup.py <command>
                    """.format(compiler=c_compiler))
            log.warn(msg)
            sys.exit(1)


        for broken, fixed in compiler_mapping:
            if re.match(broken, version):
                os.environ['CC'] = fixed
                break


def get_compiler_version(compiler):
    if compiler in _module_state['compiler_version_cache']:
        return _module_state['compiler_version_cache'][compiler]

    # Different flags to try to get the compiler version
    # TODO: It might be worth making this configurable to support
    # arbitrary odd compilers; though all bets may be off in such
    # cases anyway
    flags = ['--version', '--Version', '-version', '-Version',
             '-v', '-V']

    def try_get_version(flag):
        process = subprocess.Popen(
            shlex.split(compiler) + [flag],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            return 'unknown'

        output = stdout.strip()
        if not output:
            # Some compilers return their version info on stderr
            output = stderr.strip()

        if not output:
            output = 'unknown'

        return output

    for flag in flags:
        version = try_get_version(flag)
        if version != 'unknown':
            break

    # Cache results to speed up future calls
    _module_state['compiler_version_cache'][compiler] = version

    return version


def get_debug_option(packagename):
    """ Determines if the build is in debug mode.

    Returns
    -------
    debug : bool
        True if the current build was started with the debug option, False
        otherwise.

    """

    try:
        current_debug = get_pkg_version_module(packagename,
                                               fromlist=['debug'])[0]
    except (ImportError, AttributeError):
        current_debug = None

    # Only modify the debug flag if one of the build commands was explicitly
    # run (i.e. not as a sub-command of something else)
    dist = get_dummy_distribution()
    if any(cmd in dist.commands for cmd in ['build', 'build_ext']):
        debug = bool(get_distutils_build_option('debug'))
    else:
        debug = bool(current_debug)

    if current_debug is not None and current_debug != debug:
        build_ext_cmd = dist.get_command_class('build_ext')
        build_ext_cmd.force_rebuild = True

    return debug


def register_commands(package, version, release, srcdir='.'):
    if _module_state['registered_commands'] is not None:
        return _module_state['registered_commands']

    if _module_state['have_sphinx']:
        from .commands.build_sphinx import AstropyBuildSphinx
    else:
        AstropyBuildSphinx = FakeBuildSphinx

    _module_state['registered_commands'] = registered_commands = {
        'test': generate_test_command(package),

        # Use distutils' sdist because it respects package_data.
        # setuptools/distributes sdist requires duplication of information in
        # MANIFEST.in
        'sdist': DistutilsSdist,

        # The exact form of the build_ext command depends on whether or not
        # we're building a release version
        'build_ext': generate_build_ext_command(package, release),

        # We have a custom build_py to generate the default configuration file
        'build_py': AstropyBuildPy,

        # Since install can (in some circumstances) be run without
        # first building, we also need to override install and
        # install_lib.  See #2223
        'install': AstropyInstall,
        'install_lib': AstropyInstallLib,

        'register': AstropyRegister,
        'build_sphinx': AstropyBuildSphinx
    }

    # Need to override the __name__ here so that the commandline options are
    # presented as being related to the "build" command, for example; normally
    # this wouldn't be necessary since commands also have a command_name
    # attribute, but there is a bug in distutils' help display code that it
    # uses __name__ instead of command_name. Yay distutils!
    for name, cls in registered_commands.items():
        cls.__name__ = name

    # Add a few custom options; more of these can be added by specific packages
    # later
    for option in [
            ('use-system-libraries',
             "Use system libraries whenever possible", True)]:
        add_command_option('build', *option)
        add_command_option('install', *option)

    add_command_hooks(registered_commands, srcdir=srcdir)

    return registered_commands


def add_command_hooks(commands, srcdir='.'):
    """
    Look through setup_package.py modules for functions with names like
    ``pre_<command_name>_hook`` and ``post_<command_name>_hook`` where
    ``<command_name>`` is the name of a ``setup.py`` command (e.g. build_ext).

    If either hook is present this adds a wrapped version of that command to
    the passed in ``commands`` `dict`.  ``commands`` may be pre-populated with
    other custom distutils command classes that should be wrapped if there are
    hooks for them (e.g. `AstropyBuildPy`).
    """

    hook_re = re.compile(r'^(pre|post)_(.+)_hook$')

    # Distutils commands have a method of the same name, but it is not a
    # *classmethod* (which probably didn't exist when distutils was first
    # written)
    def get_command_name(cmdcls):
        if hasattr(cmdcls, 'command_name'):
            return cmdcls.command_name
        else:
            return cmdcls.__name__

    packages = filter_packages(find_packages(srcdir))
    dist = get_dummy_distribution()

    hooks = collections.defaultdict(dict)

    for setuppkg in iter_setup_packages(srcdir, packages):
        for name, obj in vars(setuppkg).items():
            match = hook_re.match(name)
            if not match:
                continue

            hook_type = match.group(1)
            cmd_name = match.group(2)

            cmd_cls = dist.get_command_class(cmd_name)

            if hook_type not in hooks[cmd_name]:
                hooks[cmd_name][hook_type] = []

            hooks[cmd_name][hook_type].append((setuppkg.__name__, obj))

    for cmd_name, cmd_hooks in hooks.items():
        commands[cmd_name] = generate_hooked_command(
                cmd_name, dist.get_command_class(cmd_name), cmd_hooks)


def generate_hooked_command(cmd_name, cmd_cls, hooks):
    """
    Returns a generated subclass of ``cmd_cls`` that runs the pre- and
    post-command hooks for that command before and after the ``cmd_cls.run``
    method.
    """

    def run(self, orig_run=cmd_cls.run):
        self.run_command_hooks('pre_hooks')
        orig_run(self)
        self.run_command_hooks('post_hooks')

    return type(cmd_name, (cmd_cls, object),
                {'run': run, 'run_command_hooks': run_command_hooks,
                 'pre_hooks': hooks.get('pre', []),
                 'post_hooks': hooks.get('post', [])})


def run_command_hooks(cmd_obj, hook_kind):
    """Run hooks registered for that command and phase.

    *cmd_obj* is a finalized command object; *hook_kind* is either
    'pre_hook' or 'post_hook'.
    """

    hooks = getattr(cmd_obj, hook_kind, None)

    if not hooks:
        return

    for modname, hook in hooks:
        if isinstance(hook, str):
            try:
                hook_obj = resolve_name(hook)
            except ImportError as exc:
                raise DistutilsModuleError(
                        'cannot find hook {0}: {1}'.format(hook, err))
        else:
            hook_obj = hook

        if not callable(hook_obj):
            raise DistutilsOptionError('hook {0!r} is not callable' % hook)

        log.info('running {0} from {1} for {2} command'.format(
                 hook_kind.rstrip('s'), modname, cmd_obj.get_command_name()))

        try :
            hook_obj(cmd_obj)
        except Exception as exc:
            log.error('{0} command hook {1} raised an exception: %s\n'.format(
                hook_obj.__name__, cmd_obj.get_command_name()))
            log.error(traceback.format_exc())
            sys.exit(1)


def generate_test_command(package_name):
    """
    Creates a custom 'test' command for the given package which sets the
    command's ``package_name`` class attribute to the name of the package being
    tested.
    """

    return type(package_name.title() + 'Test', (AstropyTest,),
                {'package_name': package_name})


def update_package_files(srcdir, extensions, package_data, packagenames,
                         package_dirs):
    """
    This function is deprecated and maintained for backward compatibility
    with affiliated packages.  Affiliated packages should update their
    setup.py to use `get_package_info` instead.
    """

    info = get_package_info(srcdir)
    extensions.extend(info['ext_modules'])
    package_data.update(info['package_data'])
    packagenames = list(set(packagenames + info['packages']))
    package_dirs.update(info['package_dir'])


def get_package_info(srcdir='.', exclude=()):
    """
    Collates all of the information for building all subpackages
    subpackages and returns a dictionary of keyword arguments that can
    be passed directly to `distutils.setup`.

    The purpose of this function is to allow subpackages to update the
    arguments to the package's ``setup()`` function in its setup.py
    script, rather than having to specify all extensions/package data
    directly in the ``setup.py``.  See Astropy's own
    ``setup.py`` for example usage and the Astropy development docs
    for more details.

    This function obtains that information by iterating through all
    packages in ``srcdir`` and locating a ``setup_package.py`` module.
    This module can contain the following functions:
    ``get_extensions()``, ``get_package_data()``,
    ``get_build_options()``, ``get_external_libraries()``,
    and ``requires_2to3()``.

    Each of those functions take no arguments.

    - ``get_extensions`` returns a list of
      `distutils.extension.Extension` objects.

    - ``get_package_data()`` returns a dict formatted as required by
      the ``package_data`` argument to ``setup()``.

    - ``get_build_options()`` returns a list of tuples describing the
      extra build options to add.

    - ``get_external_libraries()`` returns
      a list of libraries that can optionally be built using external
      dependencies.

    - ``get_entry_points()`` returns a dict formatted as required by
      the ``entry_points`` argument to ``setup()``.

    - ``requires_2to3()`` should return `True` when the source code
      requires `2to3` processing to run on Python 3.x.  If
      ``requires_2to3()`` is missing, it is assumed to return `True`.

    """
    ext_modules = []
    packages = []
    package_data = {}
    package_dir = {}
    skip_2to3 = []

    # Use the find_packages tool to locate all packages and modules
    packages = filter_packages(find_packages(srcdir, exclude=exclude))

    # For each of the setup_package.py modules, extract any
    # information that is needed to install them.  The build options
    # are extracted first, so that their values will be available in
    # subsequent calls to `get_extensions`, etc.
    for setuppkg in iter_setup_packages(srcdir, packages):
        if hasattr(setuppkg, 'get_build_options'):
            options = setuppkg.get_build_options()
            for option in options:
                add_command_option('build', *option)
        if hasattr(setuppkg, 'get_external_libraries'):
            libraries = setuppkg.get_external_libraries()
            for library in libraries:
                add_external_library(library)
        if hasattr(setuppkg, 'requires_2to3'):
            requires_2to3 = setuppkg.requires_2to3()
        else:
            requires_2to3 = True
        if not requires_2to3:
            skip_2to3.append(
                os.path.dirname(setuppkg.__file__))

    for setuppkg in iter_setup_packages(srcdir, packages):
        # get_extensions must include any Cython extensions by their .pyx
        # filename.
        if hasattr(setuppkg, 'get_extensions'):
            ext_modules.extend(setuppkg.get_extensions())
        if hasattr(setuppkg, 'get_package_data'):
            package_data.update(setuppkg.get_package_data())

    # Locate any .pyx files not already specified, and add their extensions in.
    # The default include dirs include numpy to facilitate numerical work.
    ext_modules.extend(get_cython_extensions(srcdir, packages, ext_modules,
                                             ['numpy']))

    # Now remove extensions that have the special name 'skip_cython', as they
    # exist Only to indicate that the cython extensions shouldn't be built
    for i, ext in reversed(list(enumerate(ext_modules))):
        if ext.name == 'skip_cython':
            del ext_modules[i]

    # On Microsoft compilers, we need to pass the '/MANIFEST'
    # commandline argument.  This was the default on MSVC 9.0, but is
    # now required on MSVC 10.0, but it doesn't seem to hurt to add
    # it unconditionally.
    if get_compiler_option() == 'msvc':
        for ext in ext_modules:
            ext.extra_link_args.append('/MANIFEST')

    return {
        'ext_modules': ext_modules,
        'packages': packages,
        'package_dir': package_dir,
        'package_data': package_data,
        'skip_2to3': skip_2to3
        }


def iter_setup_packages(srcdir, packages):
    """ A generator that finds and imports all of the ``setup_package.py``
    modules in the source packages.

    Returns
    -------
    modgen : generator
        A generator that yields (modname, mod), where `mod` is the module and
        `modname` is the module name for the ``setup_package.py`` modules.

    """

    for packagename in packages:
        package_parts = packagename.split('.')
        package_path = os.path.join(srcdir, *package_parts)
        setup_package = os.path.relpath(
            os.path.join(package_path, 'setup_package.py'))

        if os.path.isfile(setup_package):
            module = import_file(setup_package,
                                 name=packagename + '.setup_package')
            yield module


def iter_pyx_files(package_dir, package_name):
    """
    A generator that yields Cython source files (ending in '.pyx') in the
    source packages.

    Returns
    -------
    pyxgen : generator
        A generator that yields (extmod, fullfn) where `extmod` is the
        full name of the module that the .pyx file would live in based
        on the source directory structure, and `fullfn` is the path to
        the .pyx file.
    """
    for dirpath, dirnames, filenames in walk_skip_hidden(package_dir):
        for fn in filenames:
            if fn.endswith('.pyx'):
                fullfn = os.path.relpath(os.path.join(dirpath, fn))
                # Package must match file name
                extmod = '.'.join([package_name, fn[:-4]])
                yield (extmod, fullfn)

        break  # Don't recurse into subdirectories


def get_cython_extensions(srcdir, packages, prevextensions=tuple(),
                          extincludedirs=None):
    """
    Looks for Cython files and generates Extensions if needed.

    Parameters
    ----------
    srcdir : str
        Path to the root of the source directory to search.
    prevextensions : list of `~distutils.core.Extension` objects
        The extensions that are already defined.  Any .pyx files already here
        will be ignored.
    extincludedirs : list of str or None
        Directories to include as the `include_dirs` argument to the generated
        `~distutils.core.Extension` objects.

    Returns
    -------
    exts : list of `~distutils.core.Extension` objects
        The new extensions that are needed to compile all .pyx files (does not
        include any already in `prevextensions`).
    """

    # Vanilla setuptools and old versions of distribute include Cython files
    # as .c files in the sources, not .pyx, so we cannot simply look for
    # existing .pyx sources in the previous sources, but we should also check
    # for .c files with the same remaining filename. So we look for .pyx and
    # .c files, and we strip the extension.
    prevsourcepaths = []
    ext_modules = []

    for ext in prevextensions:
        for s in ext.sources:
            if s.endswith(('.pyx', '.c')):
                sourcepath = os.path.realpath(os.path.splitext(s)[0])
                prevsourcepaths.append(sourcepath)

    for package_name in packages:
        package_parts = package_name.split('.')
        package_path = os.path.join(srcdir, *package_parts)

        for extmod, pyxfn in iter_pyx_files(package_path, package_name):
            sourcepath = os.path.realpath(os.path.splitext(pyxfn)[0])
            if sourcepath not in prevsourcepaths:
                ext_modules.append(Extension(extmod, [pyxfn],
                                             include_dirs=extincludedirs))

    return ext_modules


class DistutilsExtensionArgs(collections.defaultdict):
    """
    A special dictionary whose default values are the empty list.

    This is useful for building up a set of arguments for
    `distutils.Extension` without worrying whether the entry is
    already present.
    """
    def __init__(self, *args, **kwargs):
        def default_factory():
            return []

        super(DistutilsExtensionArgs, self).__init__(
            default_factory, *args, **kwargs)

    def update(self, other):
        for key, val in other.items():
            self[key].extend(val)


def pkg_config(packages, default_libraries, executable='pkg-config'):
    """
    Uses pkg-config to update a set of distutils Extension arguments
    to include the flags necessary to link against the given packages.

    If the pkg-config lookup fails, default_libraries is applied to
    libraries.

    Parameters
    ----------
    packages : list of str
        A list of pkg-config packages to look up.

    default_libraries : list of str
        A list of library names to use if the pkg-config lookup fails.

    Returns
    -------
    config : dict
        A dictionary containing keyword arguments to
        `distutils.Extension`.  These entries include:

        - ``include_dirs``: A list of include directories
        - ``library_dirs``: A list of library directories
        - ``libraries``: A list of libraries
        - ``define_macros``: A list of macro defines
        - ``undef_macros``: A list of macros to undefine
        - ``extra_compile_args``: A list of extra arguments to pass to
          the compiler
    """

    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries',
                '-D': 'define_macros', '-U': 'undef_macros'}
    command = "{0} --libs --cflags {1}".format(executable, ' '.join(packages)),

    result = DistutilsExtensionArgs()

    try:
        pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        output = pipe.communicate()[0].strip()
    except subprocess.CalledProcessError as e:
        lines = [
            "{0} failed.  This may cause the build to fail below.".format(executable),
            "  command: {0}".format(e.cmd),
            "  returncode: {0}".format(e.returncode),
            "  output: {0}".format(e.output)
            ]
        log.warn('\n'.join(lines))
        result['libraries'].extend(default_libraries)
    else:
        if pipe.returncode != 0:
            lines = [
                "pkg-config could not lookup up package(s) {0}.".format(
                    ", ".join(packages)),
                "This may cause the build to fail below."
                ]
            log.warn('\n'.join(lines))
            result['libraries'].extend(default_libraries)
        else:
            for token in output.split():
                # It's not clear what encoding the output of
                # pkg-config will come to us in.  It will probably be
                # some combination of pure ASCII (for the compiler
                # flags) and the filesystem encoding (for any argument
                # that includes directories or filenames), but this is
                # just conjecture, as the pkg-config documentation
                # doesn't seem to address it.
                arg = token[:2].decode('ascii')
                value = token[2:].decode(sys.getfilesystemencoding())
                if arg in flag_map:
                    if arg == '-D':
                        value = tuple(value.split('=', 1))
                    result[flag_map[arg]].append(value)
                else:
                    result['extra_compile_args'].append(value)

    return result


def add_external_library(library):
    """
    Add a build option for selecting the internal or system copy of a library.

    Parameters
    ----------
    library : str
        The name of the library.  If the library is `foo`, the build
        option will be called `--use-system-foo`.
    """

    for command in ['build', 'build_ext', 'install']:
        add_command_option(command, str('use-system-' + library),
                           'Use the system {0} library'.format(library),
                           is_bool=True)


def use_system_library(library):
    """
    Returns `True` if the build configuration indicates that the given
    library should use the system copy of the library rather than the
    internal one.

    For the given library `foo`, this will be `True` if
    `--use-system-foo` or `--use-system-libraries` was provided at the
    commandline or in `setup.cfg`.

    Parameters
    ----------
    library : str
        The name of the library

    Returns
    -------
    use_system : bool
        `True` if the build should use the system copy of the library.
    """
    return (
        get_distutils_build_or_install_option('use_system_{0}'.format(library))
        or get_distutils_build_or_install_option('use_system_libraries'))


@extends_doc(_find_packages)
def find_packages(where='.', exclude=(), invalidate_cache=False):
    """
    This version of ``find_packages`` caches previous results to speed up
    subsequent calls.  Use ``invalide_cache=True`` to ignore cached results
    from previous ``find_packages`` calls, and repeat the package search.
    """

    if not invalidate_cache and _module_state['package_cache'] is not None:
        return _module_state['package_cache']

    packages = _find_packages(where=where, exclude=exclude)
    _module_state['package_cache'] = packages

    return packages


def filter_packages(packagenames):
    """
    Removes some packages from the package list that shouldn't be
    installed on the current version of Python.
    """

    if PY3:
        exclude = '_py2'
    else:
        exclude = '_py3'

    return [x for x in packagenames if not x.endswith(exclude)]


class FakeBuildSphinx(Command):
    """
    A dummy build_sphinx command that is called if Sphinx is not
    installed and displays a relevant error message
    """

    #user options inherited from sphinx.setup_command.BuildDoc
    user_options = [
         ('fresh-env', 'E', '' ),
         ('all-files', 'a', ''),
         ('source-dir=', 's', ''),
         ('build-dir=', None, ''),
         ('config-dir=', 'c', ''),
         ('builder=', 'b', ''),
         ('project=', None, ''),
         ('version=', None, ''),
         ('release=', None, ''),
         ('today=', None, ''),
         ('link-index', 'i', ''),
     ]

    #user options appended in astropy.setup_helpers.AstropyBuildSphinx
    user_options.append(('warnings-returncode', 'w',''))
    user_options.append(('clean-docs', 'l', ''))
    user_options.append(('no-intersphinx', 'n', ''))
    user_options.append(('open-docs-in-browser', 'o',''))

    def initialize_options(self):
        try:
            raise RuntimeError("Sphinx must be installed for build_sphinx")
        except:
            log.error('error : Sphinx must be installed for build_sphinx')
            sys.exit(1)
