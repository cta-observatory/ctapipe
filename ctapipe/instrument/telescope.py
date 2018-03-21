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
    optics: OpticsDescription
       the optics associated with this telescope
    camera: CameraGeometry
       the camera associated with this telescope
    """


    def __init__(self,
                 optics: OpticsDescription,
                 camera: CameraGeometry):

        self._optics = optics
        self._camera = camera

    @property
    def optics(self):
        """ OpticsDescription for this telescope """
        return self._optics

    @property
    def camera(self):
        """ CameraGeometry for this telescope"""
        return self._camera

    @classmethod
    def guess(cls, pix_x, pix_y, equivalent_focal_length):
        """
        Construct a TelescopeDescription from metadata, filling in the
        missing information using a lookup table.

        Parameters
        ----------
        pix_x: array
           array of pixel x-positions with units
        pix_y: array
           array of pixel y-positions with units
        equivalent_focal_length: float
           effective focal length of telescope with units (m)
        """
        camera = CameraGeometry.guess(pix_x, pix_y, equivalent_focal_length)
        optics = OpticsDescription.guess(equivalent_focal_length)
        return cls(optics=optics, camera=camera)

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
        return cls(optics=optics, camera=camera)

    def __str__(self):
        return str(self.optics) + ":" + str(self.camera)

    def __repr__(self):
        return "{}(optics={}, camera={})".format(self.__class__.__name__,
                                                 str(self.optics),
                                                 str(self.camera))
