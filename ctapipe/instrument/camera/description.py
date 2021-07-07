"""
Classes pertaining to the description of a Cherenkov camera
"""

from .geometry import CameraGeometry
from .readout import CameraReadout
from ctapipe.utils import find_all_matching_datasets

__all__ = ["CameraDescription"]


class CameraDescription:
    """
    Describes a Cherenkov camera and its associated
    `~ctapipe.instrument.CameraGeometry` and `~ctapipe.instrument.CameraReadout`

    Parameters
    ----------
    camera_name: str
        Camera name (e.g. NectarCam, LSTCam, ...)
    geometry: CameraGeometry
       The pixel geometry of this camera
    readout: CameraReadout
       The readout properties for this camera
    """

    def __init__(self, camera_name, geometry: CameraGeometry, readout: CameraReadout):

        self.camera_name = camera_name
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
    def from_name(cls, camera_name):
        """
        Construct a CameraDescription from a camera name

        Parameters
        ----------
        camera_name: str
            Camera name (e.g. NectarCam, LSTCam, ...)

        Returns
        -------
        CameraDescription

        """

        geometry = CameraGeometry.from_name(camera_name)
        try:
            readout = CameraReadout.from_name(camera_name)
        except FileNotFoundError:
            readout = None
        return cls(camera_name=camera_name, geometry=geometry, readout=readout)

    def __str__(self):
        return f"{self.camera_name}"

    def __repr__(self):
        return "{}(camera_name={}, geometry={}, readout={})".format(
            self.__class__.__name__,
            self.camera_name,
            str(self.geometry),
            str(self.readout),
        )
