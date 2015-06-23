from setuptools.command.install import install as SetuptoolsInstall

from ..utils import _get_platlib_dir


class AstropyInstall(SetuptoolsInstall):
    user_options = SetuptoolsInstall.user_options[:]
    boolean_options = SetuptoolsInstall.boolean_options[:]

    def finalize_options(self):
        build_cmd = self.get_finalized_command('build')
        platlib_dir = _get_platlib_dir(build_cmd)
        self.build_lib = platlib_dir
        SetuptoolsInstall.finalize_options(self)
