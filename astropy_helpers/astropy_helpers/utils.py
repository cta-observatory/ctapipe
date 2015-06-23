# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, unicode_literals

import contextlib
import functools
import imp
import inspect
import os
import sys
import textwrap
import types
import warnings

try:
    from importlib import machinery as import_machinery
    # Python 3.2 does not have SourceLoader
    if not hasattr(import_machinery, 'SourceLoader'):
        import_machinery = None
except ImportError:
    import_machinery = None


# Python 3.3's importlib caches filesystem reads for faster imports in the
# general case. But sometimes it's necessary to manually invalidate those
# caches so that the import system can pick up new generated files.  See
# https://github.com/astropy/astropy/issues/820
if sys.version_info[:2] >= (3, 3):
    from importlib import invalidate_caches
else:
    invalidate_caches = lambda: None


# Note: The following Warning subclasses are simply copies of the Warnings in
# Astropy of the same names.
class AstropyWarning(Warning):
    """
    The base warning class from which all Astropy warnings should inherit.

    Any warning inheriting from this class is handled by the Astropy logger.
    """


class AstropyDeprecationWarning(AstropyWarning):
    """
    A warning class to indicate a deprecated feature.
    """


class AstropyPendingDeprecationWarning(PendingDeprecationWarning,
                                       AstropyWarning):
    """
    A warning class to indicate a soon-to-be deprecated feature.
    """


def _get_platlib_dir(cmd):
    """
    Given a build command, return the name of the appropriate platform-specific
    build subdirectory directory (e.g. build/lib.linux-x86_64-2.7)
    """

    plat_specifier = '.{0}-{1}'.format(cmd.plat_name, sys.version[0:3])
    return os.path.join(cmd.build_base, 'lib' + plat_specifier)


def get_numpy_include_path():
    """
    Gets the path to the numpy headers.
    """
    # We need to go through this nonsense in case setuptools
    # downloaded and installed Numpy for us as part of the build or
    # install, since Numpy may still think it's in "setup mode", when
    # in fact we're ready to use it to build astropy now.

    if sys.version_info[0] >= 3:
        import builtins
        if hasattr(builtins, '__NUMPY_SETUP__'):
            del builtins.__NUMPY_SETUP__
        import imp
        import numpy
        imp.reload(numpy)
    else:
        import __builtin__
        if hasattr(__builtin__, '__NUMPY_SETUP__'):
            del __builtin__.__NUMPY_SETUP__
        import numpy
        reload(numpy)

    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()
    return numpy_include


class _DummyFile(object):
    """A noop writeable object."""

    errors = ''  # Required for Python 3.x

    def write(self, s):
        pass

    def flush(self):
        pass


@contextlib.contextmanager
def silence():
    """A context manager that silences sys.stdout and sys.stderr."""

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _DummyFile()
    sys.stderr = _DummyFile()
    exception_occurred = False
    try:
        yield
    except:
        exception_occurred = True
        # Go ahead and clean up so that exception handling can work normally
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        raise

    if not exception_occurred:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


if sys.platform == 'win32':
    import ctypes

    def _has_hidden_attribute(filepath):
        """
        Returns True if the given filepath has the hidden attribute on
        MS-Windows.  Based on a post here:
        http://stackoverflow.com/questions/284115/cross-platform-hidden-file-detection
        """
        if isinstance(filepath, bytes):
            filepath = filepath.decode(sys.getfilesystemencoding())
        try:
            attrs = ctypes.windll.kernel32.GetFileAttributesW(filepath)
            assert attrs != -1
            result = bool(attrs & 2)
        except (AttributeError, AssertionError):
            result = False
        return result
else:
    def _has_hidden_attribute(filepath):
        return False


def is_path_hidden(filepath):
    """
    Determines if a given file or directory is hidden.

    Parameters
    ----------
    filepath : str
        The path to a file or directory

    Returns
    -------
    hidden : bool
        Returns `True` if the file is hidden
    """

    name = os.path.basename(os.path.abspath(filepath))
    if isinstance(name, bytes):
        is_dotted = name.startswith(b'.')
    else:
        is_dotted = name.startswith('.')
    return is_dotted or _has_hidden_attribute(filepath)


def walk_skip_hidden(top, onerror=None, followlinks=False):
    """
    A wrapper for `os.walk` that skips hidden files and directories.

    This function does not have the parameter `topdown` from
    `os.walk`: the directories must always be recursed top-down when
    using this function.

    See also
    --------
    os.walk : For a description of the parameters
    """

    for root, dirs, files in os.walk(
            top, topdown=True, onerror=onerror,
            followlinks=followlinks):
        # These lists must be updated in-place so os.walk will skip
        # hidden directories
        dirs[:] = [d for d in dirs if not is_path_hidden(d)]
        files[:] = [f for f in files if not is_path_hidden(f)]
        yield root, dirs, files


def write_if_different(filename, data):
    """Write `data` to `filename`, if the content of the file is different.

    Parameters
    ----------
    filename : str
        The file name to be written to.
    data : bytes
        The data to be written to `filename`.
    """

    assert isinstance(data, bytes)

    if os.path.exists(filename):
        with open(filename, 'rb') as fd:
            original_data = fd.read()
    else:
        original_data = None

    if original_data != data:
        with open(filename, 'wb') as fd:
            fd.write(data)


def import_file(filename, name=None):
    """
    Imports a module from a single file as if it doesn't belong to a
    particular package.

    The returned module will have the optional ``name`` if given, or else
    a name generated from the filename.
    """
    # Specifying a traditional dot-separated fully qualified name here
    # results in a number of "Parent module 'astropy' not found while
    # handling absolute import" warnings.  Using the same name, the
    # namespaces of the modules get merged together.  So, this
    # generates an underscore-separated name which is more likely to
    # be unique, and it doesn't really matter because the name isn't
    # used directly here anyway.
    mode = 'U' if sys.version_info[0] < 3 else 'r'

    if name is None:
        basename = os.path.splitext(filename)[0]
        name = '_'.join(os.path.relpath(basename).split(os.sep)[1:])

    if import_machinery:
        loader = import_machinery.SourceFileLoader(name, filename)
        mod = loader.load_module()
    else:
        with open(filename, mode) as fd:
            mod = imp.load_module(name, fd, filename, ('.py', mode, 1))

    return mod


def resolve_name(name):
    """Resolve a name like ``module.object`` to an object and return it.

    Raise `ImportError` if the module or name is not found.
    """

    parts = name.split('.')
    cursor = len(parts) - 1
    module_name = parts[:cursor]
    attr_name = parts[-1]

    while cursor > 0:
        try:
            ret = __import__('.'.join(module_name), fromlist=[attr_name])
            break
        except ImportError:
            if cursor == 0:
                raise
            cursor -= 1
            module_name = parts[:cursor]
            attr_name = parts[cursor]
            ret = ''

    for part in parts[cursor:]:
        try:
            ret = getattr(ret, part)
        except AttributeError:
            raise ImportError(name)

    return ret


if sys.version_info[0] >= 3:
    def iteritems(dictionary):
        return dictionary.items()
else:
    def iteritems(dictionary):
        return dictionary.iteritems()


def extends_doc(extended_func):
    """
    A function decorator for use when wrapping an existing function but adding
    additional functionality.  This copies the docstring from the original
    function, and appends to it (along with a newline) the docstring of the
    wrapper function.

    Example
    -------

        >>> def foo():
        ...     '''Hello.'''
        ...
        >>> @extends_doc(foo)
        ... def bar():
        ...     '''Goodbye.'''
        ...
        >>> print(bar.__doc__)
        Hello.

        Goodbye.

    """

    def decorator(func):
        func.__doc__ = '\n\n'.join([extended_func.__doc__.rstrip('\n'),
                                    func.__doc__.lstrip('\n')])
        return func

    return decorator


# Duplicated from astropy.utils.decorators.deprecated
# When fixing issues in this function fix them in astropy first, then
# port the fixes over to astropy-helpers
def deprecated(since, message='', name='', alternative='', pending=False,
               obj_type=None):
    """
    Used to mark a function or class as deprecated.

    To mark an attribute as deprecated, use `deprecated_attribute`.

    Parameters
    ------------
    since : str
        The release at which this API became deprecated.  This is
        required.

    message : str, optional
        Override the default deprecation message.  The format
        specifier ``func`` may be used for the name of the function,
        and ``alternative`` may be used in the deprecation message
        to insert the name of an alternative to the deprecated
        function. ``obj_type`` may be used to insert a friendly name
        for the type of object being deprecated.

    name : str, optional
        The name of the deprecated function or class; if not provided
        the name is automatically determined from the passed in
        function or class, though this is useful in the case of
        renamed functions, where the new function is just assigned to
        the name of the deprecated function.  For example::

            def new_function():
                ...
            oldFunction = new_function

    alternative : str, optional
        An alternative function or class name that the user may use in
        place of the deprecated object.  The deprecation warning will
        tell the user about this alternative if provided.

    pending : bool, optional
        If True, uses a AstropyPendingDeprecationWarning instead of a
        AstropyDeprecationWarning.

    obj_type : str, optional
        The type of this object, if the automatically determined one
        needs to be overridden.
    """

    method_types = (classmethod, staticmethod, types.MethodType)

    def deprecate_doc(old_doc, message):
        """
        Returns a given docstring with a deprecation message prepended
        to it.
        """
        if not old_doc:
            old_doc = ''
        old_doc = textwrap.dedent(old_doc).strip('\n')
        new_doc = (('\n.. deprecated:: %(since)s'
                    '\n    %(message)s\n\n' %
                    {'since': since, 'message': message.strip()}) + old_doc)
        if not old_doc:
            # This is to prevent a spurious 'unexpected unindent' warning from
            # docutils when the original docstring was blank.
            new_doc += r'\ '
        return new_doc

    def get_function(func):
        """
        Given a function or classmethod (or other function wrapper type), get
        the function object.
        """
        if isinstance(func, method_types):
            try:
                func = func.__func__
            except AttributeError:
                # classmethods in Python2.6 and below lack the __func__
                # attribute so we need to hack around to get it
                method = func.__get__(None, object)
                if isinstance(method, types.FunctionType):
                    # For staticmethods anyways the wrapped object is just a
                    # plain function (not a bound method or anything like that)
                    func = method
                elif hasattr(method, '__func__'):
                    func = method.__func__
                elif hasattr(method, 'im_func'):
                    func = method.im_func
                else:
                    # Nothing we can do really...  just return the original
                    # classmethod, etc.
                    return func
        return func

    def deprecate_function(func, message):
        """
        Returns a wrapped function that displays an
        ``AstropyDeprecationWarning`` when it is called.
        """

        if isinstance(func, method_types):
            func_wrapper = type(func)
        else:
            func_wrapper = lambda f: f

        func = get_function(func)

        def deprecated_func(*args, **kwargs):
            if pending:
                category = AstropyPendingDeprecationWarning
            else:
                category = AstropyDeprecationWarning

            warnings.warn(message, category, stacklevel=2)

            return func(*args, **kwargs)

        # If this is an extension function, we can't call
        # functools.wraps on it, but we normally don't care.
        # This crazy way to get the type of a wrapper descriptor is
        # straight out of the Python 3.3 inspect module docs.
        if type(func) != type(str.__dict__['__add__']):
            deprecated_func = functools.wraps(func)(deprecated_func)

        deprecated_func.__doc__ = deprecate_doc(
            deprecated_func.__doc__, message)

        return func_wrapper(deprecated_func)

    def deprecate_class(cls, message):
        """
        Returns a wrapper class with the docstrings updated and an
        __init__ function that will raise an
        ``AstropyDeprectationWarning`` warning when called.
        """
        # Creates a new class with the same name and bases as the
        # original class, but updates the dictionary with a new
        # docstring and a wrapped __init__ method.  __module__ needs
        # to be manually copied over, since otherwise it will be set
        # to *this* module (astropy.utils.misc).

        # This approach seems to make Sphinx happy (the new class
        # looks enough like the original class), and works with
        # extension classes (which functools.wraps does not, since
        # it tries to modify the original class).

        # We need to add a custom pickler or you'll get
        #     Can't pickle <class ..>: it's not found as ...
        # errors. Picklability is required for any class that is
        # documented by Sphinx.

        members = cls.__dict__.copy()

        members.update({
            '__doc__': deprecate_doc(cls.__doc__, message),
            '__init__': deprecate_function(get_function(cls.__init__),
                                           message),
        })

        return type(cls.__name__, cls.__bases__, members)

    def deprecate(obj, message=message, name=name, alternative=alternative,
                  pending=pending):
        if obj_type is None:
            if isinstance(obj, type):
                obj_type_name = 'class'
            elif inspect.isfunction(obj):
                obj_type_name = 'function'
            elif inspect.ismethod(obj) or isinstance(obj, method_types):
                obj_type_name = 'method'
            else:
                obj_type_name = 'object'
        else:
            obj_type_name = obj_type

        if not name:
            name = get_function(obj).__name__

        altmessage = ''
        if not message or type(message) == type(deprecate):
            if pending:
                message = ('The %(func)s %(obj_type)s will be deprecated in a '
                           'future version.')
            else:
                message = ('The %(func)s %(obj_type)s is deprecated and may '
                           'be removed in a future version.')
            if alternative:
                altmessage = '\n        Use %s instead.' % alternative

        message = ((message % {
            'func': name,
            'name': name,
            'alternative': alternative,
            'obj_type': obj_type_name}) +
            altmessage)

        if isinstance(obj, type):
            return deprecate_class(obj, message)
        else:
            return deprecate_function(obj, message)

    if type(message) == type(deprecate):
        return deprecate(message)

    return deprecate


def deprecated_attribute(name, since, message=None, alternative=None,
                         pending=False):
    """
    Used to mark a public attribute as deprecated.  This creates a
    property that will warn when the given attribute name is accessed.
    To prevent the warning (i.e. for internal code), use the private
    name for the attribute by prepending an underscore
    (i.e. ``self._name``).

    Parameters
    ----------
    name : str
        The name of the deprecated attribute.

    since : str
        The release at which this API became deprecated.  This is
        required.

    message : str, optional
        Override the default deprecation message.  The format
        specifier ``name`` may be used for the name of the attribute,
        and ``alternative`` may be used in the deprecation message
        to insert the name of an alternative to the deprecated
        function.

    alternative : str, optional
        An alternative attribute that the user may use in place of the
        deprecated attribute.  The deprecation warning will tell the
        user about this alternative if provided.

    pending : bool, optional
        If True, uses a AstropyPendingDeprecationWarning instead of a
        AstropyDeprecationWarning.

    Examples
    --------

    ::

        class MyClass:
            # Mark the old_name as deprecated
            old_name = misc.deprecated_attribute('old_name', '0.1')

            def method(self):
                self._old_name = 42
    """
    private_name = '_' + name

    @deprecated(since, name=name, obj_type='attribute')
    def get(self):
        return getattr(self, private_name)

    @deprecated(since, name=name, obj_type='attribute')
    def set(self, val):
        setattr(self, private_name, val)

    @deprecated(since, name=name, obj_type='attribute')
    def delete(self):
        delattr(self, private_name)

    return property(get, set, delete)


def minversion(module, version, inclusive=True, version_path='__version__'):
    """
    Returns `True` if the specified Python module satisfies a minimum version
    requirement, and `False` if not.

    By default this uses `pkg_resources.parse_version` to do the version
    comparison if available.  Otherwise it falls back on
    `distutils.version.LooseVersion`.

    Parameters
    ----------

    module : module or `str`
        An imported module of which to check the version, or the name of
        that module (in which case an import of that module is attempted--
        if this fails `False` is returned).

    version : `str`
        The version as a string that this module must have at a minimum (e.g.
        ``'0.12'``).

    inclusive : `bool`
        The specified version meets the requirement inclusively (i.e. ``>=``)
        as opposed to strictly greater than (default: `True`).

    version_path : `str`
        A dotted attribute path to follow in the module for the version.
        Defaults to just ``'__version__'``, which should work for most Python
        modules.

    Examples
    --------

    >>> import astropy
    >>> minversion(astropy, '0.4.4')
    True
    """

    if isinstance(module, types.ModuleType):
        module_name = module.__name__
    elif isinstance(module, six.string_types):
        module_name = module
        try:
            module = resolve_name(module_name)
        except ImportError:
            return False
    else:
        raise ValueError('module argument must be an actual imported '
                         'module, or the import name of the module; '
                         'got {0!r}'.format(module))

    if '.' not in version_path:
        have_version = getattr(module, version_path)
    else:
        have_version = resolve_name('.'.join([module.__name__, version_path]))

    try:
        from pkg_resources import parse_version
    except ImportError:
        from distutils.version import LooseVersion as parse_version

    if inclusive:
        return parse_version(have_version) >= parse_version(version)
    else:
        return parse_version(have_version) > parse_version(version)
