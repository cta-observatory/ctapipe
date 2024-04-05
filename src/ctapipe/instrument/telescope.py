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
from ..coordinates import CameraFrame
from .camera import CameraDescription
from .guess import guess_telescope, unknown_telescope
from .optics import OpticsDescription
from .warnings import warn_from_name

__all__ = ["TelescopeDescription"]


class TelescopeDescription:
    """
    Describes a Cherenkov Telescope and its associated
    `~ctapipe.instrument.OpticsDescription` and `~ctapipe.instrument.CameraDescription`

    Attributes
    ----------
    name: str
        Telescope name
    tel_type: str
        Telescope type
    optics: OpticsDescription
       the optics associated with this telescope
    camera: CameraDescription
       the camera associated with this telescope
    """

    __slots__ = (
        "name",
        "optics",
        "camera",
    )

    def __init__(
        self,
        name: str,
        optics: OpticsDescription,
        camera: CameraDescription,
    ):
        if not isinstance(name, str):
            raise TypeError("`name` must be a str")

        if not isinstance(optics, OpticsDescription):
            raise TypeError("`optics` must be an instance of `OpticsDescription`")

        if not isinstance(camera, CameraDescription):
            raise TypeError("`camera` must be an instance of `CameraDescription`")

        self.name = name
        self.optics = optics
        self.camera = camera

    def __hash__(self):
        """Make this hashable, so it can be used as dict keys or in sets"""
        return hash((self.optics, self.camera))

    def __eq__(self, other):
        return self.optics == other.optics and self.camera == other.camera

    @classmethod
    def from_name(cls, optics_name, camera_name):
        """
        construct a TelescopeDescription from a name (telescope description
        string)

        Parameters
        ----------
        camera_name : str
           camera name
        optics_name : str
           optics name (e.g. LST, or SST-ASTRI), also called
           telescope_description

        Notes
        -----

        Warning: This method loads a pre-generated ``TelescopeDescription`` and is
        thus not guaranteed to be the same pixel ordering or even positions that
        correspond with event data! Therefore if you are analysing data, you
        should not rely on this method, but rather open the data with an
        ``EventSource`` and use the ``TelescopeDescription`` that is provided by
        ``source.subarray.tel[i]`` or by
        ``source.subarray.telescope_types[type_name]``. This will guarantee that
        the pixels in the event data correspond with the ``TelescopeDescription``


        Returns
        -------
        TelescopeDescription

        """
        warn_from_name()

        camera = CameraDescription.from_name(camera_name)
        optics = OpticsDescription.from_name(optics_name)
        camera.geometry.frame = CameraFrame(focal_length=optics.equivalent_focal_length)

        try:
            result = guess_telescope(
                camera.geometry.n_pixels, optics.equivalent_focal_length
            )
        except ValueError:
            result = unknown_telescope(optics.mirror_area, camera.geometry.n_pixels)

        return cls(name=result.name, optics=optics, camera=camera)

    @property
    def camera_name(self):
        """Name of the camera"""
        return self.camera.name

    @property
    def optics_name(self):
        """Name of the optics"""
        return self.optics.name

    @property
    def type(self):
        """Size classification"""
        return self.optics.size_type

    def __str__(self):
        return f"{self.type}_{self.optics_name}_{self.camera_name}"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"type={self.type.value!r}"
            f", optics_name={self.optics_name!r}"
            f", camera_name={self.camera_name!r}"
            ")"
        )
