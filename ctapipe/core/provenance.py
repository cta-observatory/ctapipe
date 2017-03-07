"""
Provenance-related functionality

TODO: have this register whenever ctapipe is loaded

"""

import platform
import sys

import ctapipe
import numpy as np
import psutil
from astropy.time import Time

__all__ = ['Provenance']

class Provenance:
    """
    Collect local provenance information for later storage.


    This class is used accumulate information on system execution
    environment, inputs, and outputs for a given Tool or script.
    """

    def __init__(self):
        self._prov = {'STARTUP': {},
                      'SHUTDOWN': {},
                      'SYSTEM': {},
                      'INPUT': [],
                      'OUTPUT': []}

    def start(self):
        """ begin recording provenance. Set's up the system and startup
        provenance data. Generally should be called at start of a program."""
        self._prov['STARTUP'].update(self._sample_cpu_and_memory())
        self._prov['SYSTEM'].update(self._get_system_provenance())

    def register_input(self, url):
        """
        Add a URL of a file to the list of inputs (can be a filename or full
        url, if no URL specifier is given, assume 'file://')

        Parameters
        ----------
        url: str
            filename or url of input file
        """
        self._prov['INPUT'].append(url)

    def register_output(self, url):
        """
        Add a URL of a file to the list of outputs (can be a filename or full
        url, if no URL specifier is given, assume 'file://')

        Parameters
        ----------
        url: str
            filename or url of output file
        """
        self._prov['OUTPUT'].append(url)

    def finish(self):
        """ record final provenance information, normally called at shutdown."""
        self._prov['SHUTDOWN'] = self._sample_cpu_and_memory()

    def sample(self):
        """
        Record a snapshot of current CPU and memory information.
        """
        if 'SAMPLES' not in self._prov:
            self._prov['SAMPLES'] = []
        self._prov['SAMPLES'].append(self._sample_cpu_and_memory())

    @property
    def provenance(self):
        return self._prov

    def _get_system_provenance(self):
        """ return JSON string containing provenance for all things that are
        fixed during the runtime"""

        uname = platform.uname()
        bits, linkage = platform.architecture()
        pyver = sys.version_info

        return dict(
            ctapipe_version=ctapipe.__version__,
            executable=sys.executable,
            platform=dict(
                architecture_bits=bits,
                architecture_linkage=linkage,
                machine=uname.machine,
                processor=uname.processor,
                node=uname.node,
                version=uname.version,
                system=uname.system,
                release=uname.release,
                libcver=platform.libc_ver(),
                num_cpus=psutil.cpu_count(),
                boot_time=Time(psutil.boot_time(), format='unix').isot,
            ),
            python=dict(
                version_string=sys.version,
                version=[
                    pyver.major,
                    pyver.minor,
                    pyver.micro,
                    pyver.releaselevel
                ]),
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
