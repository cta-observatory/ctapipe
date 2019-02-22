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
from .optics import OpticsDescription
from .camera import CameraGeometry
from .guess import guess_telescope


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
    type: str
        Telescope type
    optics: OpticsDescription
       the optics associated with this telescope
    camera: CameraGeometry
       the camera associated with this telescope
    """

    def __init__(
        self,
        name,
        type,
        optics: OpticsDescription,
        camera: CameraGeometry
    ):

        self.name = name
        self.type = type
        self.optics = optics
        self.camera = camera

    def __hash__(self):
        '''Make this hashable, so it can be used as dict keys or in sets'''
        return hash((self.optics, self.camera))

    def __eq__(self, other):
        '''Make this hashable, so it can be used as dict keys or in sets'''
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

        t = guess_telescope(camera.n_pixels, optics.equivalent_focal_length)

        return cls(name=t.name, type=t.type, optics=optics, camera=camera)

    def __str__(self):
        return str(self.optics) + ":" + str(self.camera)

    def __repr__(self):
        return "{}({}, type={}, optics={}, camera={})".format(
            self.name,
            self.type,
            self.__class__.__name__,
            str(self.optics),
            str(self.camera),
        )
