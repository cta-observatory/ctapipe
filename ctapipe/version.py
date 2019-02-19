"""
Get version identification from git.


The update_release_version() function writes the current version to the
VERSION file. This function should be called before packaging a release version.

Use the get_version() function to get the version string, including the latest
commit,  from git.
If git is not available the VERSION file will be read.

Heres an example of such a version string:

    v0.2.0.post58+git57440dc


This code was taken from here:
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
VERSION_FILE = path.join(CURRENT_DIRECTORY, "_version_cache.py")

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


def get_git_describe_version(abbrev=7):
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

    if "-" not in git_str:  # currently at a tag
        formatted_str = git_str
    else:
        # formatted as version-N-githash
        # want to convert to version.postN-githash
        git_str = git_str.replace("-", ".post", 1)
        if pep440:  # does not allow git hash afterwards
            formatted_str = git_str.split("-")[0]
        else:
            formatted_str = git_str.replace("-g", "+git")

    # need to remove the "v" to have a proper python version
    if formatted_str.startswith('v'):
        formatted_str = formatted_str[1:]

    return formatted_str


def read_release_version():
    """Read version information from VERSION file"""
    try:
        from ._version_cache import version
        if len(version) == 0:
            version = None
        return version
    except ImportError:
        return "unknown"


def update_release_version(pep440=False):
    """Release versions are stored in a file called VERSION.
    This method updates the version stored in the file.
    This function should be called when creating new releases.
    It is called by setup.py when building a package.


    pep440: bool
        When True, this function returns a version string suitable for
        a release as defined by PEP 440. When False, the githash (if
        available) will be appended to the version string.

    """
    version = get_version(pep440=pep440)
    with open(VERSION_FILE, "w") as outfile:
        outfile.write(f"version='{version}'")
        outfile.write("\n")


def get_version(pep440=False):
    """Tracks the version number.

    pep440: bool
        When True, this function returns a version string suitable for
        a release as defined by PEP 440. When False, the githash (if
        available) will be appended to the version string.

    The file VERSION holds the version information. If this is not a git
    repository, then it is reasonable to assume that the version is not
    being incremented and the version returned will be the release version as
    read from the file.

    However, if the script is located within an active git repository,
    git-describe is used to get the version information.

    The file VERSION will need to be changed manually.
    """

    raw_git_version = get_git_describe_version()
    if not raw_git_version:  # not a git repository
        return  read_release_version()

    git_version = format_git_describe(raw_git_version, pep440=pep440)

    return git_version


if __name__ == "__main__":
    print(get_version())
