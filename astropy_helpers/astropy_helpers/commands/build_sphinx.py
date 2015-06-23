import inspect
import os
import pkgutil
import re
import shutil
import subprocess
import sys
import textwrap

from distutils import log
from distutils.cmd import DistutilsOptionError

import sphinx
from sphinx.setup_command import BuildDoc as SphinxBuildDoc

from ..utils import minversion


PY3 = sys.version_info[0] >= 3


class AstropyBuildSphinx(SphinxBuildDoc):
    """
    A version of the ``build_sphinx`` command that uses the version of Astropy
    that is built by the setup ``build`` command, rather than whatever is
    installed on the system.  To build docs against the installed version, run
    ``make html`` in the ``astropy/docs`` directory.

    This also automatically creates the docs/_static directories--this is
    needed because GitHub won't create the _static dir because it has no
    tracked files.
    """

    description = 'Build Sphinx documentation for Astropy environment'
    user_options = SphinxBuildDoc.user_options[:]
    user_options.append(('warnings-returncode', 'w',
            'Parses the sphinx output and sets the return code to 1 if there '
            'are any warnings. Note that this will cause the sphinx log to '
            'only update when it completes, rather than continuously as is '
            'normally the case.'))
    user_options.append(('clean-docs', 'l',
            'Completely clean previous builds, including '
            'automodapi-generated files before building new ones'))
    user_options.append(('no-intersphinx', 'n',
            'Skip intersphinx, even if conf.py says to use it'))
    user_options.append(('open-docs-in-browser', 'o',
            'Open the docs in a browser (using the webbrowser module) if the '
            'build finishes successfully.'))

    boolean_options = SphinxBuildDoc.boolean_options[:]
    boolean_options.append('warnings-returncode')
    boolean_options.append('clean-docs')
    boolean_options.append('no-intersphinx')
    boolean_options.append('open-docs-in-browser')

    _self_iden_rex = re.compile(r"self\.([^\d\W][\w]+)", re.UNICODE)

    def initialize_options(self):
        SphinxBuildDoc.initialize_options(self)
        self.clean_docs = False
        self.no_intersphinx = False
        self.open_docs_in_browser = False
        self.warnings_returncode = False

    def finalize_options(self):
        #Clear out previous sphinx builds, if requested
        if self.clean_docs:
            dirstorm = [os.path.join(self.source_dir, 'api')]
            if self.build_dir is None:
                dirstorm.append('docs/_build')
            else:
                dirstorm.append(self.build_dir)

            for d in dirstorm:
                if os.path.isdir(d):
                    log.info('Cleaning directory ' + d)
                    shutil.rmtree(d)
                else:
                    log.info('Not cleaning directory ' + d + ' because '
                             'not present or not a directory')

        SphinxBuildDoc.finalize_options(self)

    def run(self):
        # TODO: Break this method up into a few more subroutines and
        # document them better
        import webbrowser

        if PY3:
            from urllib.request import pathname2url
        else:
            from urllib import pathname2url

        # This is used at the very end of `run` to decide if sys.exit should
        # be called. If it's None, it won't be.
        retcode = None

        # If possible, create the _static dir
        if self.build_dir is not None:
            # the _static dir should be in the same place as the _build dir
            # for Astropy
            basedir, subdir = os.path.split(self.build_dir)
            if subdir == '':  # the path has a trailing /...
                basedir, subdir = os.path.split(basedir)
            staticdir = os.path.join(basedir, '_static')
            if os.path.isfile(staticdir):
                raise DistutilsOptionError(
                    'Attempted to build_sphinx in a location where' +
                    staticdir + 'is a file.  Must be a directory.')
            self.mkpath(staticdir)

        # Now make sure Astropy is built and determine where it was built
        build_cmd = self.reinitialize_command('build')
        build_cmd.inplace = 0
        self.run_command('build')
        build_cmd = self.get_finalized_command('build')
        build_cmd_path = os.path.abspath(build_cmd.build_lib)

        ah_importer = pkgutil.get_importer('astropy_helpers')
        ah_path = os.path.abspath(ah_importer.path)

        # Now generate the source for and spawn a new process that runs the
        # command.  This is needed to get the correct imports for the built
        # version
        runlines, runlineno = inspect.getsourcelines(SphinxBuildDoc.run)
        subproccode = textwrap.dedent("""
            from sphinx.setup_command import *

            os.chdir({srcdir!r})
            sys.path.insert(0, {build_cmd_path!r})
            sys.path.insert(0, {ah_path!r})

        """).format(build_cmd_path=build_cmd_path, ah_path=ah_path,
                    srcdir=self.source_dir)
        # runlines[1:] removes 'def run(self)' on the first line
        subproccode += textwrap.dedent(''.join(runlines[1:]))

        # All "self.foo" in the subprocess code needs to be replaced by the
        # values taken from the current self in *this* process
        subproccode = self._self_iden_rex.split(subproccode)
        for i in range(1, len(subproccode), 2):
            iden = subproccode[i]
            val = getattr(self, iden)
            if iden.endswith('_dir'):
                # Directories should be absolute, because the `chdir` call
                # in the new process moves to a different directory
                subproccode[i] = repr(os.path.abspath(val))
            else:
                subproccode[i] = repr(val)
        subproccode = ''.join(subproccode)

        # This is a quick gross hack, but it ensures that the code grabbed from
        # SphinxBuildDoc.run will work in Python 2 if it uses the print
        # function
        if minversion(sphinx, '1.3'):
            subproccode = 'from __future__ import print_function' + subproccode

        if self.no_intersphinx:
            # the confoverrides variable in sphinx.setup_command.BuildDoc can
            # be used to override the conf.py ... but this could well break
            # if future versions of sphinx change the internals of BuildDoc,
            # so remain vigilant!
            subproccode = subproccode.replace('confoverrides = {}',
                'confoverrides = {\'intersphinx_mapping\':{}}')

        log.debug('Starting subprocess of {0} with python code:\n{1}\n'
                  '[CODE END])'.format(sys.executable, subproccode))

        # To return the number of warnings, we need to capture stdout. This
        # prevents a continuous updating at the terminal, but there's no
        # apparent way around this.
        if self.warnings_returncode:
            proc = subprocess.Popen([sys.executable],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            stdo, stde = proc.communicate(subproccode.encode('utf-8'))

            print(stdo)

            stdolines = stdo.splitlines()

            if b'build succeeded.' in stdolines:
                retcode = 0
            else:
                retcode = 1

            if retcode != 0:
                if os.environ.get('TRAVIS', None) == 'true':
                    #this means we are in the travis build, so customize
                    #the message appropriately.
                    msg = ('The build_sphinx travis build FAILED '
                           'because sphinx issued documentation '
                           'warnings (scroll up to see the warnings).')
                else:  # standard failure message
                    msg = ('build_sphinx returning a non-zero exit '
                           'code because sphinx issued documentation '
                           'warnings.')
                log.warn(msg)

        else:
            proc = subprocess.Popen([sys.executable], stdin=subprocess.PIPE)
            proc.communicate(subproccode.encode('utf-8'))

        if proc.returncode == 0:
            if self.open_docs_in_browser:
                if self.builder == 'html':
                    absdir = os.path.abspath(self.builder_target_dir)
                    index_path = os.path.join(absdir, 'index.html')
                    fileurl = 'file://' + pathname2url(index_path)
                    webbrowser.open(fileurl)
                else:
                    log.warn('open-docs-in-browser option was given, but '
                             'the builder is not html! Ignoring.')
        else:
            log.warn('Sphinx Documentation subprocess failed with return '
                     'code ' + str(proc.returncode))

        if retcode is not None:
            # this is potentially dangerous in that there might be something
            # after the call to `setup` in `setup.py`, and exiting here will
            # prevent that from running.  But there's no other apparent way
            # to signal what the return code should be.
            sys.exit(retcode)
