"""
Provenance-related functionality

TODO: have this register whenever ctapipe is loaded

"""

import sys
from astropy.time import Time
import json
import platform
import psutil
import numpy as np
import ctapipe


class Provenance:
    """
    Collect provenance information for later storage.
    """

    def __init__(self):
        self._prov = {}

    def start(self):
        """ begin recording provenance. Set's up the system and startup
        provenance data. Generally should be called at start of a program."""
        self._prov['START'] = self._sample_provenance()
        self._prov['SYSTEM'] = self._get_system_provenance()

    def finish(self):
        """ record final provenance information, normally called at shutdown."""
        self._prov['END'] = self._sample_provenance()

    def sample(self):
        """take a sample of provenance information, including current cpu
        time and other stats."""
        if 'SAMPLES' not in self._prov:
            self._prov['SAMPLES'] = []
        self._prov['SAMPLES'].append(self._sample_provenance())

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
                version = [
                    pyver.major,
                    pyver.minor,
                    pyver.micro,
                    pyver.releaselevel
                ]),
            arguments=sys.argv,
            start_time_utc=Time.now().isot,
        )



    def _sample_provenance(self):
        times = np.asarray(psutil.cpu_times(percpu=True))
        mem = psutil.virtual_memory()

        return dict(
            time_utc = Time.now().utc,
            memory = dict(total=mem.total,
                          inactive=mem.inactive,
                          available=mem.available,
                          free=mem.free,
                          wired=mem.wired),
            cpu = dict(ncpu=psutil.cpu_count(),
                           user=list(times[:, 0]),
                           nice=list(times[:, 1]),
                           system=list(times[:, 2]),
                           idle=list(times[:, 3])),
        )





