from setuptools.command.install_lib import install_lib as SetuptoolsInstallLib

from ..utils import _get_platlib_dir


class AstropyInstallLib(SetuptoolsInstallLib):
    user_options = SetuptoolsInstallLib.user_options[:]
    boolean_options = SetuptoolsInstallLib.boolean_options[:]

    def finalize_options(self):
        build_cmd = self.get_finalized_command('build')
        platlib_dir = _get_platlib_dir(build_cmd)
        self.build_dir = platlib_dir
        SetuptoolsInstallLib.finalize_options(self)
