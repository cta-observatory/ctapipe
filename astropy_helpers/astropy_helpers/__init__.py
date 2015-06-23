try:
    from .version import version as __version__
    from .version import githash as __githash__
except ImportError:
    __version__ = ''
    __githash__ = ''


# If we've made it as far as importing astropy_helpers, we don't need
# ah_bootstrap in sys.modules anymore.  Getting rid of it is actually necessary
# if the package we're installing has a setup_requires of another package that
# uses astropy_helpers (and possibly a different version at that)
# See https://github.com/astropy/astropy/issues/3541
import sys
if 'ah_bootstrap' in sys.modules:
    del sys.modules['ah_bootstrap']


# Note, this is repeated from ah_bootstrap.py, but is here too in case this
# astropy-helpers was upgraded to from an older version that did not have this
# check in its ah_bootstrap.
# matplotlib can cause problems if it is imported from within a call of
# run_setup(), because in some circumstances it will try to write to the user's
# home directory, resulting in a SandboxViolation.  See
# https://github.com/matplotlib/matplotlib/pull/4165
# Making sure matplotlib, if it is available, is imported early in the setup
# process can mitigate this (note importing matplotlib.pyplot has the same
# issue)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot
except:
    # Ignore if this fails for *any* reason*
    pass
