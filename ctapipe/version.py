"""
Get version identification from git

See the documentation of get_version for more information

This code was (only slightly) adapted from here:
https://github.com/aebrahim/python-git-version

Combining ideas from
http://blogs.nopcode.org/brainstorm/2013/05/20/pragmatic-python-versioning-via-setuptools-and-git-tags/
and Python Versioneer
https://github.com/warner/python-versioneer
but being much more lightwheight

"""
from subprocess import check_output, CalledProcessError
from os import path, name, devnull, environ, listdir

__all__ = ("get_version",)

CURRENT_DIRECTORY = path.dirname(path.abspath(__file__))
VERSION_FILE = path.join(CURRENT_DIRECTORY, "VERSION")

GIT_COMMAND = "git"

if name == "nt":
    def find_git_on_windows():
        """find the path to the git executable on windows"""
        # first see if git is in the path
        try:
            check_output(["where", "/Q", "git"])
            # if this command succeeded, git is in the path
            return "git"
        # catch the exception thrown if git was not found
        except CalledProcessError:
            pass
        # There are several locations git.exe may be hiding
        possible_locations = []
        # look in program files for msysgit
        if "PROGRAMFILES(X86)" in environ:
            possible_locations.append("%s/Git/cmd/git.exe" %
                                      environ["PROGRAMFILES(X86)"])
        if "PROGRAMFILES" in environ:
            possible_locations.append("%s/Git/cmd/git.exe" %
                                      environ["PROGRAMFILES"])
        # look for the github version of git
        if "LOCALAPPDATA" in environ:
            github_dir = "%s/GitHub" % environ["LOCALAPPDATA"]
            if path.isdir(github_dir):
                for subdir in listdir(github_dir):
                    if not subdir.startswith("PortableGit"):
                        continue
                    possible_locations.append("%s/%s/bin/git.exe" %
                                              (github_dir, subdir))
        for possible_location in possible_locations:
            if path.isfile(possible_location):
                return possible_location
        # git was not found
        return "git"

    GIT_COMMAND = find_git_on_windows()


def call_git_describe(abbrev=7):
    """return the string output of git desribe"""
    try:
        with open(devnull, "w") as fnull:
            arguments = [GIT_COMMAND, "describe", "--tags",
                         "--abbrev=%d" % abbrev]
            return check_output(arguments, cwd=CURRENT_DIRECTORY,
                                stderr=fnull).decode("ascii").strip()
    except (OSError, CalledProcessError):
        return None


def format_git_describe(git_str, pep440=False):
    """format the result of calling 'git describe' as a python version"""
    if git_str is None:
        return None
    if "-" not in git_str:  # currently at a tag
        return git_str
    else:
        # formatted as version-N-githash
        # want to convert to version.postN-githash
        git_str = git_str.replace("-", ".post", 1)
        if pep440:  # does not allow git hash afterwards
            return git_str.split("-")[0]
        else:
            return git_str.replace("-g", "+git")


def get_version(pep440=False):
    """Tracks the version number.

    pep440: bool
        When True, this function returns a version string suitable for
        a release as defined by PEP 440. When False, the githash (if
        available) will be appended to the version string.


    If the script is located within an active git repository,
    git-describe is used to get the version information.
    """

    git_version = format_git_describe(call_git_describe(), pep440=pep440)
    if git_version is None:  # not a git repository
        raise FileNotFoundError('Not a git repository')
    return git_version


if __name__ == "__main__":
    print(get_version())
