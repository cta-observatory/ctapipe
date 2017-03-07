"""
Provenance-related functionality

TODO: have this register whenever ctapipe is loaded

"""

import platform
import sys
import logging

import ctapipe
import numpy as np
import psutil
from astropy.time import Time

log = logging.getLogger(__name__)

__all__ = ['Provenance','prov']


class Provenance:
    """
    Manage the provenance info for a stack of *activities*

    use `start_activity(name)` to start an activity. Any calls to
    `add_input_entity()`

    """

    def __init__(self):
        self._activities = []  # stack of active activities
        self._finished_activities = []

    def start_activity(self, activity_name):
        """ push activity onto the stack"""
        activity = _ActivityProvenance(activity_name)
        activity.start()
        self._activities.append(activity)
        log.debug("started activity: {}".format(activity_name))

    def add_input_entity(self, url):
        """ register an input to the current activity """
        self.current_activity.register_input(url)
        log.debug("added input entity '{}' to activity: '{}'".format(
            url, self.current_activity.name))

    def add_output_entity(self,url):
        """ register an output to the current activity """
        self.current_activity.register_output(url)
        log.debug("added output entity '{}' to activity: '{}'".format(
            url, self.current_activity.name))

    def finish_activity(self, activity_name=None):
        """ end the current activity """
        activity = self._activities.pop()
        if activity_name is not None and activity_name != activity.name:
            raise ValueError("Tried to end activity '{}', but '{}' is current "
                             "activity".format(activity_name, activity.name))

        activity.finish()
        self._finished_activities.append(activity)
        log.debug("finished activity: {}".format(activity.name))

    @property
    def current_activity(self):
        if len(self._activities) == 0:
            raise IndexError("No activities in progress")

        return self._activities[-1] # current activity as at the top of stack

    @property
    def provenance(self):
        return [x.provenance for x in self._finished_activities]

    @property
    def active_activity_names(self):
        return [x.name for x in self._activities]

    @property
    def finished_activity_names(self):
        return [x.name for x in self._finished_activities]


class _ActivityProvenance:
    """
    Low-level helper class to ollect provenance information for a given
    *activity*.  Users should use `Provenance` as a top-level API, not this
    class directly.
    """

    def __init__(self, activity_name=sys.executable):
        self._prov = {
            'activity_name': activity_name,
            'start': {},
            'stop' :{},
            'system': {},
            'input': [],
            'output': []
        }
        self.name = activity_name

    def start(self):
        """ begin recording provenance for this activity. Set's up the system
        and startup provenance data. Generally should be called at start of a program."""
        self._prov['start'].update(self._sample_cpu_and_memory())
        self._prov['system'].update(self._get_system_provenance())

    def register_input(self, url):
        """
        Add a URL of a file to the list of inputs (can be a filename or full
        url, if no URL specifier is given, assume 'file://')

        Parameters
        ----------
        url: str
            filename or url of input file
        """
        self._prov['input'].append(url)

    def register_output(self, url):
        """
        Add a URL of a file to the list of outputs (can be a filename or full
        url, if no URL specifier is given, assume 'file://')

        Parameters
        ----------
        url: str
            filename or url of output file
        """
        self._prov['output'].append(url)

    def finish(self):
        """ record final provenance information, normally called at shutdown."""
        self._prov['stop'].update(self._sample_cpu_and_memory())
        t_start = Time(self._prov['start']['time_utc'], format='isot')
        t_stop = Time(self._prov['stop']['time_utc'], format='isot')
        self._prov['duration'] = (t_stop-t_start).to('min').value

    def sample_cpu_and_memory(self):
        """
        Record a snapshot of current CPU and memory information.
        """
        if 'samples' not in self._prov:
            self._prov['samples'] = []
        self._prov['samples'].append(self._sample_cpu_and_memory())

    @property
    def provenance(self):
        return self._prov

    def _get_system_provenance(self):
        """ return JSON string containing provenance for all things that are
        fixed during the runtime"""

        bits, linkage = platform.architecture()

        return dict(
            ctapipe_version=ctapipe.__version__,
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
                num_cpus=psutil.cpu_count(),
                boot_time=Time(psutil.boot_time(), format='unix').isot,
            ),
            python=dict(
                version_string=sys.version,
                version=platform.python_version_tuple(),
                compiler=platform.python_compiler(),
                implementation=platform.python_implementation(),
            ),
            arguments=sys.argv,
            start_time_utc=Time.now().isot,
        )

    def _sample_cpu_and_memory(self):
        times = np.asarray(psutil.cpu_times(percpu=True))
        mem = psutil.virtual_memory()

        return dict(
            time_utc=Time.now().utc.isot,
            memory=dict(total=mem.total,
                        inactive=mem.inactive,
                        available=mem.available,
                        free=mem.free,
                        wired=mem.wired),
            cpu=dict(ncpu=psutil.cpu_count(),
                     user=list(times[:, 0]),
                     nice=list(times[:, 1]),
                     system=list(times[:, 2]),
                     idle=list(times[:, 3])),
        )



prov = Provenance()
