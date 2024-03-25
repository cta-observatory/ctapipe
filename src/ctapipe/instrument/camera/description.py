"""
Classes pertaining to the description of a Cherenkov camera
"""

from ctapipe.utils import find_all_matching_datasets

from ..warnings import warn_from_name
from .geometry import CameraGeometry
from .readout import CameraReadout

__all__ = ["CameraDescription"]


class CameraDescription:
    """
    Describes a Cherenkov camera and its associated
    `~ctapipe.instrument.CameraGeometry` and `~ctapipe.instrument.CameraReadout`

    Parameters
    ----------
    name: str
        Camera name (e.g. NectarCam, LSTCam, ...)
    geometry: CameraGeometry
       The pixel geometry of this camera
    readout: CameraReadout
       The readout properties for this camera
    """

    __slots__ = (
        "name",
        "geometry",
        "readout",
    )

    def __init__(self, name, geometry: CameraGeometry, readout: CameraReadout):
        self.name = name
        self.geometry = geometry
        self.readout = readout

    def __hash__(self):
        """Make this hashable, so it can be used as dict keys or in sets"""
        return hash((self.geometry, self.readout))

    def __eq__(self, other):
        return self.geometry == other.geometry and self.readout == other.readout

    @classmethod
    def get_known_camera_names(cls):
        """
        Returns a list of camera names that are available currently on the system.
        Beware that the `from_name` method also tries to download camera descriptions
        from the data server, so this list might not be exhaustive.

        Returns
        -------
        list(str)
        """

        pattern = r"(.*)\.camgeom\.fits(\.gz)?"
        return find_all_matching_datasets(pattern, regexp_group=1)

    @classmethod
    def from_name(cls, name):
        """Construct a CameraDescription from a camera name

        Parameters
        ----------
        name: str
            Camera name (e.g. NectarCam, LSTCam, ...)


        Notes
        -----

        Warning: This method loads a pre-generated ``CameraDescription`` and is
        thus not guaranteed to be the same pixel ordering or even positions that
        correspond with event data! Therefore if you are analysing data, you
        should not rely on this method, but rather open the data with an
        ``EventSource`` and use the ``CameraDescription`` that is provided by
        ``source.subarray.tel[i].camera`` or by
        ``source.subarray.camera_types[type_name]``. This will guarantee that
        the pixels in the event data correspond with the ``CameraDescription``

        Returns
        -------
        CameraDescription

        """
        warn_from_name()

        geometry = CameraGeometry.from_name(name)
        try:
            readout = CameraReadout.from_name(name)
        except FileNotFoundError:
            readout = None
        return cls(name=name, geometry=geometry, readout=readout)

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return "{}(name={}, geometry={}, readout={})".format(
            self.__class__.__name__,
            self.name,
            str(self.geometry),
            str(self.readout),
        )
