from setuptools.command.build_py import build_py as SetuptoolsBuildPy

from ..utils import _get_platlib_dir


class AstropyBuildPy(SetuptoolsBuildPy):
    user_options = SetuptoolsBuildPy.user_options[:]
    boolean_options = SetuptoolsBuildPy.boolean_options[:]

    def finalize_options(self):
        # Update build_lib settings from the build command to always put
        # build files in platform-specific subdirectories of build/, even
        # for projects with only pure-Python source (this is desirable
        # specifically for support of multiple Python version).
        build_cmd = self.get_finalized_command('build')
        platlib_dir = _get_platlib_dir(build_cmd)

        build_cmd.build_purelib = platlib_dir
        build_cmd.build_lib = platlib_dir
        self.build_lib = platlib_dir

        SetuptoolsBuildPy.finalize_options(self)

    def run_2to3(self, files, doctests=False):
        # Filter the files to exclude things that shouldn't be 2to3'd
        skip_2to3 = self.distribution.skip_2to3
        filtered_files = []
        for filename in files:
            for package in skip_2to3:
                if filename[len(self.build_lib) + 1:].startswith(package):
                    break
            else:
                filtered_files.append(filename)

        SetuptoolsBuildPy.run_2to3(self, filtered_files, doctests)

    def run(self):
        # first run the normal build_py
        SetuptoolsBuildPy.run(self)
