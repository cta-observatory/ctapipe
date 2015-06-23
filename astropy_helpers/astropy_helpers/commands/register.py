from setuptools.command.register import register as SetuptoolsRegister


class AstropyRegister(SetuptoolsRegister):
    """Extends the built in 'register' command to support a ``--hidden`` option
    to make the registered version hidden on PyPI by default.

    The result of this is that when a version is registered as "hidden" it can
    still be downloaded from PyPI, but it does not show up in the list of
    actively supported versions under http://pypi.python.org/pypi/astropy, and
    is not set as the most recent version.

    Although this can always be set through the web interface it may be more
    convenient to be able to specify via the 'register' command.  Hidden may
    also be considered a safer default when running the 'register' command,
    though this command uses distutils' normal behavior if the ``--hidden``
    option is omitted.
    """

    user_options = SetuptoolsRegister.user_options + [
        ('hidden', None, 'mark this release as hidden on PyPI by default')
    ]
    boolean_options = SetuptoolsRegister.boolean_options + ['hidden']

    def initialize_options(self):
        SetuptoolsRegister.initialize_options(self)
        self.hidden = False

    def build_post_data(self, action):
        data = SetuptoolsRegister.build_post_data(self, action)
        if action == 'submit' and self.hidden:
            data['_pypi_hidden'] = '1'
        return data

    def _set_config(self):
        # The original register command is buggy--if you use .pypirc with a
        # server-login section *at all* the repository you specify with the -r
        # option will be overwritten with either the repository in .pypirc or
        # with the default,
        # If you do not have a .pypirc using the -r option will just crash.
        # Way to go distutils

        # If we don't set self.repository back to a default value _set_config
        # can crash if there was a user-supplied value for this option; don't
        # worry, we'll get the real value back afterwards
        self.repository = 'pypi'
        SetuptoolsRegister._set_config(self)
        options = self.distribution.get_option_dict('register')
        if 'repository' in options:
            source, value = options['repository']
            # Really anything that came from setup.cfg or the command line
            # should override whatever was in .pypirc
            self.repository = value
