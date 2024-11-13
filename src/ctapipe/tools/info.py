# Licensed under a 3-clause BSD style license - see LICENSE.rst
""" print information about ctapipe and its command-line tools. """
import logging
import os
import sys
from importlib.metadata import PackageNotFoundError, requires, version
from importlib.resources import files

from packaging.markers import default_environment
from packaging.requirements import Requirement

from ..core import Provenance, get_module_version
from ..core.plugins import detect_and_import_plugins
from ..utils import datasets
from .utils import get_parser

__all__ = ["info"]


def main(args=None):
    parser = get_parser(info)
    parser.add_argument("--version", action="store_true", help="Print version number")
    parser.add_argument(
        "--tools", action="store_true", help="Print available command line tools"
    )
    parser.add_argument(
        "--dependencies",
        action="store_true",
        help="Print available versions of dependencies",
    )
    parser.add_argument(
        "--resources",
        action="store_true",
        help="Print available versions of dependencies",
    )
    parser.add_argument("--system", action="store_true", help="Print system info")
    parser.add_argument(
        "--all", dest="show_all", action="store_true", help="show all info"
    )
    parser.add_argument("--plugins", action="store_true", help="Print plugin info")
    parser.add_argument(
        "--datamodel", action="store_true", help="Print data model info"
    )
    parser.add_argument(
        "--event-sources",
        action="store_true",
        help="Print available EventSource implementations",
    )
    parser.add_argument(
        "--reconstructors",
        action="store_true",
        help="Print available Reconstructor implementations",
    )

    args = parser.parse_args(args)

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)

    info(**vars(args))


def info(
    version=False,
    tools=False,
    dependencies=False,
    resources=False,
    system=False,
    plugins=False,
    show_all=False,
    datamodel=False,
    event_sources=False,
    reconstructors=False,
):
    """
    Display information about the current ctapipe installation.
    """
    logging.basicConfig(level=logging.CRITICAL, format="%(levelname)s - %(message)s")

    if version or show_all:
        _info_version()

    if datamodel or show_all:
        _info_datamodel()

    if tools or show_all:
        _info_tools()

    if dependencies or show_all:
        _info_dependencies()

    if resources or show_all:
        _info_resources()

    if system or show_all:
        _info_system()

    if plugins or show_all:
        _info_plugins()

    if event_sources or show_all:
        _info_sources()

    if reconstructors or show_all:
        _info_reconstructors()


def _info_version():
    """Print version info."""
    import ctapipe

    print("\n*** ctapipe version info ***\n")
    print(f"version: {ctapipe.__version__}")
    # print('release: {0}'.format(version.release))
    # print('githash: {0}'.format(version.githash))
    print("")


def _info_tools():
    """Print info about command line tools."""
    print("\n*** ctapipe tools ***\n")
    print("the following can be executed by typing ctapipe-<toolname>:")
    print("")

    # TODO: how to get a one-line description or
    # full help text from the docstring or ArgumentParser?
    # This is the function names, we want the command-line names
    # that are defined in setup.py !???
    from textwrap import TextWrapper

    from ctapipe.tools.utils import get_all_descriptions

    wrapper = TextWrapper(width=80, subsequent_indent=" " * 35)

    scripts = get_all_descriptions()
    for name, desc in sorted(scripts.items()):
        text = f"{name:<30s}  - {desc}"
        print(wrapper.fill(text))
        print("")
    print("")


def _info_dependencies():
    """Print info about dependencies."""
    print("\n*** ctapipe core dependencies ***\n")

    env = default_environment()
    requirements = [Requirement(r) for r in requires("ctapipe")]
    dependencies = [
        r.name for r in requirements if r.marker is None or r.marker.evaluate(env)
    ]

    env["extra"] = "all"
    optional_dependencies = [
        r.name for r in requirements if r.marker is not None and r.marker.evaluate(env)
    ]

    for name in dependencies:
        print(f"{name:>20s} -- {version(name)}")

    print("\n*** ctapipe optional dependencies ***\n")

    for name in optional_dependencies:
        try:
            v = version(name)
        except PackageNotFoundError:
            v = "not installed"
        print(f"{name:>20s} -- {v}")


def _info_resources():
    """display all known resources"""

    print("\n*** ctapipe resources ***\n")
    print("CTAPIPE_SVC_PATH: (directories where resources are searched)")
    if os.getenv("CTAPIPE_SVC_PATH") is not None:
        for directory in datasets.get_searchpath_dirs():
            print(f"\t * {directory}")
    else:
        print("\t no path is set")
    print("")

    all_resources = sorted(datasets.find_all_matching_datasets(r"\w.*"))
    home = os.path.expanduser("~")
    try:
        resource_dir = files("ctapipe_resources")
    except ImportError:
        resource_dir = None

    fmt = "{name:<30.30s} : {loc:<30.30s}"
    print(fmt.format(name="RESOURCE NAME", loc="LOCATION"))
    print("-" * 70)
    for resource in all_resources:
        if resource.suffix == ".py" or resource.name.startswith("_"):
            continue
        loc = str(resource)
        if resource_dir is not None:
            loc = loc.replace(resource_dir, "[ctapipe_resources]")
        loc = loc.replace(home, "~")
        print(fmt.format(name=resource.name, loc=loc))


def _info_system():
    # collect system info using the ctapipe provenance system :

    print("\n*** ctapipe system environment ***\n")

    prov = Provenance()
    prov.start_activity("ctapipe-info")
    system_prov = prov.current_activity.provenance["system"]

    for section in ["platform", "python"]:
        print("\n====== ", section, " ======== \n")
        sysinfo = system_prov[section]

        for name, val in sysinfo.items():
            print("{:>20.20s} -- {:<60.60s}".format(name, str(val)))


def _info_plugins():
    for kind in ("io", "reco"):
        plugins = detect_and_import_plugins(f"ctapipe_{kind}")
        print(f"\n*** ctapipe_{kind} plugins ***\n")

        if not plugins:
            print(f"No {kind} plugins installed")
            return

        for name in plugins:
            version = get_module_version(name)
            print(f"{name:>20s} -- {version}")


def _info_sources():
    from ctapipe.io import EventSource

    print("\n*** ctapipe available EventSource implementations ***\n")
    sources = EventSource.non_abstract_subclasses()
    maxlen = max(len(source) for source in sources)
    for source in sources.values():
        print(
            f"{source.__name__:>{maxlen}s} -- {source.__module__}.{source.__qualname__}"
        )


def _info_reconstructors():
    from ctapipe.reco import Reconstructor

    print("\n*** ctapipe Reconstructor implementations ***\n")
    sources = Reconstructor.non_abstract_subclasses()
    maxlen = max(len(source) for source in sources)
    for source in sources.values():
        print(
            f"{source.__name__:>{maxlen}s} -- {source.__module__}.{source.__qualname__}"
        )


def _info_datamodel():
    from ctapipe.io.datawriter import DATA_MODEL_CHANGE_HISTORY, DATA_MODEL_VERSION
    from ctapipe.io.hdf5eventsource import COMPATIBLE_DATA_MODEL_VERSIONS

    print("\n*** ctapipe data model ***\n")
    print(f"output version: {DATA_MODEL_VERSION}")
    print(f"compatible input versions: {', '.join(COMPATIBLE_DATA_MODEL_VERSIONS)}")
    print("change history:")
    print(DATA_MODEL_CHANGE_HISTORY)


if __name__ == "__main__":
    main()
