"""
Classes pertaining to the description of a Cherenkov Telescope

Todo:
-----

- add more info in OpticsDescription (mirror area, facets, etc). How to guess
  this?
- add ability to write to/from tables (like that written by
  ctapipe-dump-instrument)
- add ability to construct by names TelescopeDescription.from_name(
  camera='LSTCam', optics=('SST','1M')) (which would create a very unbalanced
  telescope :-))

"""
from .camera import CameraGeometry
from .guess import UNKNOWN_TELESCOPE, guess_telescope
from .optics import OpticsDescription


class TelescopeDescription:
    """
    Describes a Cherenkov Telescope and it's associated `OpticsDescription` and
    `CameraGeometry`

    The string representation is a combination of the optics and
    camera, separated by a colon: "optics:camera" (e.g. "SST-1m:DigiCam")

    The `TelescopeDescription.guess()` constructor can be used to fill in
    info from metadata, e.g. for Monte-Carlo files.

    Parameters
    ----------
    name: str
        Telescope name
    tel_type: str
        Telescope type
    optics: OpticsDescription
       the optics associated with this telescope
    camera: CameraGeometry
       the camera associated with this telescope
    """

    def __init__(self, name, tel_type, optics: OpticsDescription, camera: CameraGeometry):

        self.name = name
        self.type = tel_type
        self.optics = optics
        self.camera = camera

    def __hash__(self):
        """Make this hashable, so it can be used as dict keys or in sets"""
        return hash((self.optics, self.camera))

    def __eq__(self, other):
        """Make this hashable, so it can be used as dict keys or in sets"""
        return hash(self) == hash(other)

    @classmethod
    def from_name(cls, optics_name, camera_name):
        """
        construct a TelescopeDescription from a name (telescope description
        string)

        Parameters
        ----------
        camera_name: str
           camera name
        optics_name: str
           optics name (e.g. LST, or SST-ASTRI), also called
           telescope_description

        Returns
        -------
        TelescopeDescription

        """

        camera = CameraGeometry.from_name(camera_name)
        optics = OpticsDescription.from_name(optics_name)

        try:
            result = guess_telescope(camera.n_pixels, optics.equivalent_focal_length)
        except ValueError:
            result = UNKNOWN_TELESCOPE

        return cls(name=result.name, tel_type=result.type, optics=optics, camera=camera)

    def __str__(self):
        return f"{self.type}_{self.optics}_{self.camera}"

    def __repr__(self):
        return "{}(type={}, name={}, optics={}, camera={})".format(
            self.__class__.__name__,
            self.type,
            self.name,
            str(self.optics),
            str(self.camera),
        )
