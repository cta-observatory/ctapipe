"""
Provenance-related functionality

TODO: have this register whenever ctapipe is loaded

"""

import json
import logging
import os
import platform
import sys
import uuid
import warnings
from collections import UserList
from contextlib import contextmanager
from functools import cache
from importlib import import_module
from importlib.metadata import Distribution, distributions
from os.path import abspath
from pathlib import Path
from types import ModuleType

import psutil
from astropy.time import Time

from ..version import __version__
from .support import Singleton

log = logging.getLogger(__name__)

__all__ = ["Provenance"]

_interesting_env_vars = [
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "CONDA_PYTHON_EXE",
    "CONDA_EXE",
    "CONDA_PROMPT_MODIFIER",
    "CONDA_SHLVL",
    "PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "USER",
    "HOME",
    "SHELL",
]


@cache
def modules_of_distribution(distribution: Distribution):
    modules = distribution.read_text("top_level.txt")
    if modules is None:
        return None
    return set(modules.splitlines())


@cache
def get_distribution_of_module(module: ModuleType | str):
    """Get the package distribution for an imported module"""
    if isinstance(module, str):
        name = module
        module = import_module(module)
    else:
        name = module.__name__

    path = Path(module.__file__).absolute()

    for dist in distributions():
        modules = modules_of_distribution(dist)
        if modules is None:
            base = dist.locate_file("")
            if dist.files is not None and any(path == base / f for f in dist.files):
                return dist
        elif name in modules:
            return dist

    raise ValueError(f"Could not find a distribution for module: {module}")


def get_module_version(name):
    """
    Get the version of a python *module*, something you can import.

    If the module does not expose a ``__version__`` attribute, this function
    will try to determine the *distribution* of the module and return its
    version.
    """
    # we try first with module.__version__
    # to support editable installs
    try:
        module = import_module(name)
    except ModuleNotFoundError:
        return "not installed"

    try:
        return module.__version__
    except AttributeError:
        try:
            return get_distribution_of_module(module).version
        except Exception:
            return "unknown"


class MissingReferenceMetadata(UserWarning):
    """Warning raised if reference metadata could not be read from input file."""


class _ActivityProvenance:
    """
    Low-level helper class to collect provenance information for a given
    *activity*.  Users should use `Provenance` as a top-level API,
    not this class directly.
    """

    def __init__(self, activity_name=sys.executable):
        self._prov = {
            "activity_name": activity_name,
            "activity_uuid": str(uuid.uuid4()),
            "status": "running",
            "start": {},
            "stop": {},
            "system": {},
            "input": [],
            "output": [],
            "exit_code": None,
        }
        self.name = activity_name

    def start(self):
        """begin recording provenance for this activity. Set's up the system
        and startup provenance data. Generally should be called at start of a
        program."""
        self._prov["start"].update(_sample_cpu_and_memory())
        self._prov["system"].update(_get_system_provenance())

    def _register(self, what, url, role, add_meta, reference_meta):
        if what not in {"input", "output"}:
            raise ValueError("what must be 'input' or 'output'")

        reference_meta = self._get_reference_meta(
            url=url, reference_meta=reference_meta, read_meta=add_meta
        )
        self._prov[what].append(dict(url=url, role=role, reference_meta=reference_meta))

    def register_input(self, url, role=None, add_meta=True, reference_meta=None):
        """
        Add a URL of a file to the list of inputs (can be a filename or full
        url, if no URL specifier is given, assume 'file://')

        Parameters
        ----------
        url: str
            filename or url of input file
        role: str
            role name that this input satisfies
        add_meta: bool
            If true, try to load reference metadata from input file
            and add to provenance.
        """
        self._register(
            "input", url, role=role, add_meta=add_meta, reference_meta=reference_meta
        )

    def register_output(self, url, role=None, add_meta=True, reference_meta=None):
        """
        Add a URL of a file to the list of outputs (can be a filename or full
        url, if no URL specifier is given, assume 'file://')

        Should only be called once the file is finalized, so that reference metadata
        can be read.

        Parameters
        ----------
        url: str
            filename or url of output file
        role: str
            role name that this output satisfies
        add_meta: bool
            If true, try to load reference metadata from input file
            and add to provenance.
        """
        self._register(
            "output", url, role=role, add_meta=add_meta, reference_meta=reference_meta
        )

    def register_config(self, config):
        """add a dictionary of configuration parameters to this activity"""
        self._prov["config"] = config

    def finish(self, status="success", exit_code=0):
        """record final provenance information, normally called at shutdown."""
        self._prov["stop"].update(_sample_cpu_and_memory())

        # record the duration (wall-clock) for this activity
        t_start = Time(self._prov["start"]["time_utc"], format="isot")
        t_stop = Time(self._prov["stop"]["time_utc"], format="isot")
        self._prov["status"] = status
        self._prov["exit_code"] = exit_code
        self._prov["duration_min"] = (t_stop - t_start).to("min").value

    @property
    def output(self):
        return self._prov.get("output", None)

    @property
    def input(self):
        return self._prov.get("input", None)

    def sample_cpu_and_memory(self):
        """
        Record a snapshot of current CPU and memory information.
        """
        if "samples" not in self._prov:
            self._prov["samples"] = []
        self._prov["samples"].append(_sample_cpu_and_memory())

    @property
    def provenance(self):
        return self._prov

    def _get_reference_meta(
        self, url, reference_meta=None, read_meta=True
    ) -> dict | None:
        # here to prevent circular imports / top-level cross-dependencies
        from ..io.metadata import read_reference_metadata

        if reference_meta is not None or read_meta is False:
            return reference_meta

        try:
            return read_reference_metadata(url).to_dict()
        except Exception:
            warnings.warn(
                f"Could not read reference metadata for input file: {url}",
                MissingReferenceMetadata,
            )
            return None


class Provenance(metaclass=Singleton):
    """
    Manage the provenance info for a stack of *activities*

    use `start_activity(name) <start_activity>`_ to start an activity. Any calls to
    `add_input_file` or `add_output_file` will register files within
    that activity. Finish the current activity with `finish_activity`.

    Nested activities are allowed, and handled as a stack. The final output
    is not hierarchical, but a flat list of activities (however hierarchical
    activities could easily be implemented if necessary)
    """

    def __init__(self):
        self._activities = []  # stack of active activities
        self._finished_activities = []

    def start_activity(self, activity_name=sys.executable):
        """push activity onto the stack"""
        activity = _ActivityProvenance(activity_name)
        activity.start()
        self._activities.append(activity)
        log.debug(f"started activity: {activity_name}")
        return activity

    def _get_current_or_start_activity(self) -> _ActivityProvenance:
        if self.current_activity is None:
            log.info(
                "No activity has been explicitly started, starting new default activity."
                " Consider calling Provenance().start_activity(<name>) explicitly."
            )
            return self.start_activity()
        return self.current_activity

    def add_input_file(self, filename, role=None, add_meta=True, reference_meta=None):
        """register an input to the current activity

        Parameters
        ----------
        filename: str
            name or url of file
        role: str
            role this input file satisfies (optional)
        """
        activity = self._get_current_or_start_activity()
        activity.register_input(
            abspath(filename),
            role=role,
            add_meta=add_meta,
            reference_meta=reference_meta,
        )
        log.debug(
            "added input entity '%s' to activity: '%s'",
            filename,
            activity.name,
        )

    def add_output_file(self, filename, role=None, add_meta=True):
        """
        register an output to the current activity

        Parameters
        ----------
        filename: str
            name or url of file
        role: str
            role this output file satisfies (optional)

        """
        activity = self._get_current_or_start_activity()
        activity.register_output(abspath(filename), role=role, add_meta=add_meta)
        log.debug(
            "added output entity '%s' to activity: '%s'",
            filename,
            activity.name,
        )

    def add_config(self, config):
        """
        add configuration parameters to the current activity

        Parameters
        ----------
        config: dict
            configuration parameters
        """
        activity = self._get_current_or_start_activity()
        activity.register_config(config)
        log.debug(
            "added config entity '%s' to activity: '%s'",
            config,
            activity.name,
        )

    def finish_activity(self, status="completed", exit_code=0, activity_name=None):
        """end the current activity"""
        activity = self._activities.pop()
        if activity_name is not None and activity_name != activity.name:
            raise ValueError(
                "Tried to end activity '{}', but '{}' is current " "activity".format(
                    activity_name, activity.name
                )
            )

        activity.finish(status, exit_code)
        self._finished_activities.append(activity)
        log.debug(f"finished activity: {activity.name}")

    @contextmanager
    def activity(self, name):
        """context manager for activities"""
        self.start_activity(name)
        yield
        self.finish_activity(name)

    @property
    def current_activity(self):
        if len(self._activities) == 0:
            return None
        return self._activities[-1]  # current activity is at the top of stack

    @property
    def finished_activities(self):
        return self._finished_activities

    @property
    def provenance(self):
        """returns provenence for full list of activities"""
        return [x.provenance for x in self._finished_activities]

    def as_json(self, **kwargs):
        """return all finished provenance as JSON.  Kwargs for `json.dumps`
        may be included, e.g. ``indent=4``"""

        def set_default(obj):
            """handle sets (not part of JSON) by converting to list"""
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, UserList):
                return list(obj)
            if isinstance(obj, Path):
                return str(obj)

        return json.dumps(self.provenance, default=set_default, **kwargs)

    @property
    def active_activity_names(self):
        return [x.name for x in self._activities]

    @property
    def finished_activity_names(self):
        return [x.name for x in self._finished_activities]

    def clear(self):
        """remove all tracked activities"""
        self._activities = []
        self._finished_activities = []


def _get_python_packages():
    def _sortkey(dist):
        """Sort packages by name, case insensitive"""
        # get is needed to avoid errors / deprecation warning
        # in case packages with broken metadata are in the system
        # see e.g. https://github.com/pypa/setuptools/issues/4482
        return dist.metadata.get("Name", "").lower()

    return [
        {"name": p.name, "version": p.metadata.get("Version", "<unknown>")}
        for p in sorted(distributions(), key=_sortkey)
        if p.metadata.get("Name") is not None
    ]


def _get_system_provenance():
    """return a dict containing provenance for all things that are
    fixed during the runtime"""

    bits, linkage = platform.architecture()

    return dict(
        ctapipe_version=__version__,
        ctapipe_svc_path=os.getenv("CTAPIPE_SVC_PATH"),
        executable=sys.executable,
        platform=dict(
            architecture_bits=bits,
            architecture_linkage=linkage,
            machine=platform.machine(),
            processor=platform.processor(),
            node=platform.node(),
            version=platform.version(),
            system=platform.system(),
            release=platform.release(),
            libcver=platform.libc_ver(),
            n_cpus=psutil.cpu_count(),
            boot_time=Time(psutil.boot_time(), format="unix").isot,
        ),
        python=dict(
            version_string=sys.version,
            version=platform.python_version_tuple(),
            compiler=platform.python_compiler(),
            implementation=platform.python_implementation(),
            packages=_get_python_packages(),
        ),
        environment=_get_env_vars(),
        arguments=sys.argv,
        start_time_utc=Time.now().isot,
    )


def _get_env_vars():
    envvars = {}
    for var in _interesting_env_vars:
        envvars[var] = os.getenv(var, None)
    return envvars


def _sample_cpu_and_memory():
    # times = np.asarray(psutil.cpu_times(percpu=True))
    # mem = psutil.virtual_memory()

    return dict(
        time_utc=Time.now().utc.isot,
        # memory=dict(total=mem.total,
        #             inactive=mem.inactive,
        #             available=mem.available,
        #             free=mem.free,
        #             wired=mem.wired),
        # cpu=dict(ncpu=psutil.cpu_count(),
        #          user=list(times[:, 0]),
        #          nice=list(times[:, 1]),
        #          system=list(times[:, 2]),
        #          idle=list(times[:, 3])),
    )
