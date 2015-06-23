"""
This module contains various utilities for introspecting the distutils
module and the setup process.

Some of these utilities require the
`astropy_helpers.setup_helpers.register_commands` function to be called first,
as it will affect introspection of setuptools command-line arguments.  Other
utilities in this module do not have that restriction.
"""

import os
import sys

from distutils import ccompiler
from distutils.dist import Distribution
from distutils.errors import DistutilsError

from .utils import silence


# This function, and any functions that call it, require the setup in
# `astropy_helpers.setup_helpers.register_commands` to be run first.
def get_dummy_distribution():
    """
    Returns a distutils Distribution object used to instrument the setup
    environment before calling the actual setup() function.
    """

    from .setup_helpers import _module_state

    if _module_state['registered_commands'] is None:
        raise RuntimeError(
            'astropy_helpers.setup_helpers.register_commands() must be '
            'called before using '
            'astropy_helpers.setup_helpers.get_dummy_distribution()')

    # Pre-parse the Distutils command-line options and config files to if
    # the option is set.
    dist = Distribution({'script_name': os.path.basename(sys.argv[0]),
                         'script_args': sys.argv[1:]})
    dist.cmdclass.update(_module_state['registered_commands'])

    with silence():
        try:
            dist.parse_config_files()
            dist.parse_command_line()
        except (DistutilsError, AttributeError, SystemExit):
            # Let distutils handle DistutilsErrors itself AttributeErrors can
            # get raise for ./setup.py --help SystemExit can be raised if a
            # display option was used, for example
            pass

    return dist


def get_distutils_option(option, commands):
    """ Returns the value of the given distutils option.

    Parameters
    ----------
    option : str
        The name of the option

    commands : list of str
        The list of commands on which this option is available

    Returns
    -------
    val : str or None
        the value of the given distutils option. If the option is not set,
        returns None.
    """

    dist = get_dummy_distribution()

    for cmd in commands:
        cmd_opts = dist.command_options.get(cmd)
        if cmd_opts is not None and option in cmd_opts:
            return cmd_opts[option][1]
    else:
        return None


def get_distutils_build_option(option):
    """ Returns the value of the given distutils build option.

    Parameters
    ----------
    option : str
        The name of the option

    Returns
    -------
    val : str or None
        The value of the given distutils build option. If the option
        is not set, returns None.
    """
    return get_distutils_option(option, ['build', 'build_ext', 'build_clib'])


def get_distutils_install_option(option):
    """ Returns the value of the given distutils install option.

    Parameters
    ----------
    option : str
        The name of the option

    Returns
    -------
    val : str or None
        The value of the given distutils build option. If the option
        is not set, returns None.
    """
    return get_distutils_option(option, ['install'])


def get_distutils_build_or_install_option(option):
    """ Returns the value of the given distutils build or install option.

    Parameters
    ----------
    option : str
        The name of the option

    Returns
    -------
    val : str or None
        The value of the given distutils build or install option. If the
        option is not set, returns None.
    """
    return get_distutils_option(option, ['build', 'build_ext', 'build_clib',
                                         'install'])


def get_compiler_option():
    """ Determines the compiler that will be used to build extension modules.

    Returns
    -------
    compiler : str
        The compiler option specified for the build, build_ext, or build_clib
        command; or the default compiler for the platform if none was
        specified.

    """

    compiler = get_distutils_build_option('compiler')
    if compiler is None:
        return ccompiler.get_default_compiler()

    return compiler


def add_command_option(command, name, doc, is_bool=False):
    """
    Add a custom option to a setup command.

    Issues a warning if the option already exists on that command.

    Parameters
    ----------
    command : str
        The name of the command as given on the command line

    name : str
        The name of the build option

    doc : str
        A short description of the option, for the `--help` message

    is_bool : bool, optional
        When `True`, the option is a boolean option and doesn't
        require an associated value.
    """

    dist = get_dummy_distribution()
    cmdcls = dist.get_command_class(command)

    if (hasattr(cmdcls, '_astropy_helpers_options') and
            name in cmdcls._astropy_helpers_options):
        return

    attr = name.replace('-', '_')

    if hasattr(cmdcls, attr):
        raise RuntimeError(
            '{0!r} already has a {1!r} class attribute, barring {2!r} from '
            'being usable as a custom option name.'.format(cmdcls, attr, name))

    for idx, cmd in enumerate(cmdcls.user_options):
        if cmd[0] == name:
            log.warn('Overriding existing {0!r} option '
                     '{1!r}'.format(command, name))
            del cmdcls.user_options[idx]
            if name in cmdcls.boolean_options:
                cmdcls.boolean_options.remove(name)
            break

    cmdcls.user_options.append((name, None, doc))

    if is_bool:
        cmdcls.boolean_options.append(name)

    # Distutils' command parsing requires that a command object have an
    # attribute with the same name as the option (with '-' replaced with '_')
    # in order for that option to be recognized as valid
    setattr(cmdcls, attr, None)

    # This caches the options added through add_command_option so that if it is
    # run multiple times in the same interpreter repeated adds are ignored
    # (this way we can still raise a RuntimeError if a custom option overrides
    # a built-in option)
    if not hasattr(cmdcls, '_astropy_helpers_options'):
        cmdcls._astropy_helpers_options = set([name])
    else:
        cmdcls._astropy_helpers_options.add(name)


def get_distutils_display_options():
    """ Returns a set of all the distutils display options in their long and
    short forms.  These are the setup.py arguments such as --name or --version
    which print the project's metadata and then exit.

    Returns
    -------
    opts : set
        The long and short form display option arguments, including the - or --
    """

    short_display_opts = set('-' + o[1] for o in Distribution.display_options
                             if o[1])
    long_display_opts = set('--' + o[0] for o in Distribution.display_options)

    # Include -h and --help which are not explicitly listed in
    # Distribution.display_options (as they are handled by optparse)
    short_display_opts.add('-h')
    long_display_opts.add('--help')

    # This isn't the greatest approach to hardcode these commands.
    # However, there doesn't seem to be a good way to determine
    # whether build *will be* run as part of the command at this
    # phase.
    display_commands = set([
        'clean', 'register', 'setopt', 'saveopts', 'egg_info',
        'alias'])

    return short_display_opts.union(long_display_opts.union(display_commands))


def is_distutils_display_option():
    """ Returns True if sys.argv contains any of the distutils display options
    such as --version or --name.
    """

    display_options = get_distutils_display_options()
    return bool(set(sys.argv[1:]).intersection(display_options))
