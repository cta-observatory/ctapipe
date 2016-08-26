# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Definition of Observation Configurations.

The ObsConfig class defines a hierarchy of classes that contain the
configuration information related to an observation. Each sub-config
should additionally be loadable on its own

This may be extended later to make a SimObsConfig that also includes extra
Monte-Carlo information

When possible, items inside each class should be loaded dynamically
(e.g. on first access)

"""

from functools import partial
from collections import defaultdict
import numpy as np

__all__ = [
    'get_site_id_for_run',
    'get_site_id_for_time',
    'BaseConfig',
    'CameraConfig',
    'ArrayTriggerConfig',
    'TelescopeTriggerConfig',
    'TelescopeConfig',
    'ArrayConfig',
    'OpticsConfig',
    'SubarrayConfig',
    'ObsConfig',
    'SimObsConfig',
]


def get_site_id_for_run(run_id):
    """ lookup which array and version was used for a given run """
    
    return site_id, version

def get_site_id_for_time(obstime):
    """lookup which array and version was used for a given obstime
    (astropy.time.Time)

    Example::
        aid,aver = get_site_id_for_time( Time("2015-01-01", scale='utc') ) 
    """
    
    return site_id, version


class BaseConfig:
    pass


class CameraConfig(BaseConfig):
    """Definition of a Camera, including its pixel geometry and other
    characteristics
    """
    def __init__(self, site_id, version, tel_id):
        self.site_id = site_id
        self.version = version
        self.tel_id = tel_id
        
        # the following are to be loaded:
        self.version_id =version_id
        self.pixel_type = "hexagonal"
        self.pix_x = None
        self.pix_y = None
        self.pix_z = None
        self.pix_area = None
        self.focal_plane_offset = 0.0 # offset from focal_length

    @staticmethod
    def load_from_file():
        raise NotImplementedError


class ArrayTriggerConfig(BaseConfig):
    """ Contains trigger info for a given run_type """
    pass


class TelescopeTriggerConfig(BaseConfig):
    """ Contains trigger info for a given run_type """
    pass

        
class TelescopeConfig(BaseConfig):

    """Configuration of a single Telescope, including it's optics and
    camera
    """

    def __init__(self, tel_id , site_id, version):
        self._site_id = site_id
        self._version = version
        self.tel_id = tel_id

        # the following are to be loaded:
        self._optics = OpticsConfig(site_id, version, tel_id)
        self._camera = CameraConfig(site_id, version, tel_id)


class ArrayConfig(BaseConfig):
    """Overall description of an array (including all telescope that are
    built or forseen. This is not a subarray"""
    
    def __init__(self, site_id, version):

        self._site_id = site_id
        self._version = version
        # make a defaultdict of TelescopeConfigs, with everything but
        # tel_id as frozen parameters. Then it will load
        # "on-the-fly"
        TelescopeConfigPartial = partial(TelescopeConfig, site_id=site_id,
                                         version=version)
        self._telconfig = defaultdict(TelescopeConfigPartial)
        
        # the telescope positions and ids indexed by index (0-N)
        self.tel_x = []
        self.tel_y = []
        self.tel_z = []
        self.tel_ids = []

    def tel(self,tel_id):
        if tel_id not in self.tel_ids:
            raise ValueError("tel_id out of range")

        if tel_id not in self._telconfig:
            self._telconfig[tel_id] = TelescopeConfig(tel_id,
                                                      self._site_id,
                                                      self._version)

        return self._telconfig[tel_id]

    @property
    def num_tels(self):
        return len(self.tel_id)


class OpticsConfig(BaseConfig):
    """Definition of Telescope Optics, including information on the
    overall mirror characteristics and the fascets
    """
    
    def __init__(self, site_id, version, tel_id):
        super(OpticsConfig, self).__init__()
        self.site_id = site_id
        self.version = version
        self.tel_id = tel_id

        # must load the following
        self.optics_type = "davies-cotton"  # or have a DaviesCotton subclass
        self.mirror_area = 0.0
        self.focal_length = 0.0
        self.facet_x = np.array()
        self.facet_y = np.array()
        self.facet_z = np.array()
        self.facet_area = np.array()


class SubarrayConfig(BaseConfig):
    """Description of a particular Subarray used during an observation or
    during a monte-carlo production
    """
    def __init__(self, run_id):
        super(SubarrayConfig, self).__init__()
        self.run_id = run_id


class ObsConfig(BaseConfig):
    """All configuration information related to an observation run.
    """

    def __init__(self, run_id):
        super(ObsConfig, self).__init__()
        self._run_id = run_id

        # the overall array information (not just the subarray)
        self._array = ArrayConfig(site_id, version)

        # the subarray information (e.g. a list of active telescopes, etc)
        self._subarray = SubarrayConfig(run_id)

        # trigger type info
        self._trigger = TriggerConfig(run_id)

        
class SimObsConfig(ObsConfig):
    """ObsConfig from a simulation run.
    """

    def __init__(self, mc_run_id):
        self._mc_run_id = mc_run_id
