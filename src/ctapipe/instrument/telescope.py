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


def get_lst_mirror_vertices():
    vertices_LST_i = [
            [-1001.0, -45.32],
            [-1076.5, -88.91],
            [-1076.5, -176.09],
            [-1001.0, -219.68],
            [-1001.0, -312.06],
            [-1076.5, -355.65],
            [-1076.5, -442.83],
            [-1001.0, -486.42],
            [-999.5, -576.2],
            [-924.0, -619.79],
            [-922.5, -709.56],
            [-847.0, -753.15],
            [-845.5, -842.93],
            [-770.0, -886.52],
            [-768.5, -976.3],
            [-693.0, -1019.89],
            [-617.5, -976.3],
            [-537.5, -1022.49],
            [-537.5, -1109.67],
            [-462.0, -1153.26],
            [-386.5, -1109.67],
            [-308.0, -1153.26],
            [-232.5, -1109.67],
            [-154.0, -1153.26],
            [-78.5, -1109.67],
            [-0.0, -1153.26],
            [75.5, -1109.67],
            [154.0, -1153.26],
            [229.5, -1109.67],
            [308.0, -1153.26],
            [383.5, -1109.67],
            [462.0, -1153.26],
            [537.5, -1109.67],
            [537.5, -1022.49],
            [617.5, -976.3],
            [693.0, -1019.89],
            [768.5, -976.3],
            [768.5, -889.12],
            [845.5, -842.93],
            [845.5, -755.75],
            [922.5, -709.56],
            [922.5, -622.38],
            [999.5, -576.2],
            [999.5, -489.02],
            [1076.5, -442.83],
            [1076.5, -355.65],
            [1001.0, -312.06],
            [1001.0, -219.68],
            [1076.5, -176.09],
            [1076.5, -88.91],
            [1001.0, -45.32],
            [1001.0, 47.05],
            [1076.5, 90.64],
            [1076.5, 177.82],
            [1153.5, 224.01],
            [1153.5, 311.19],
            [1078.0, 354.78],
            [1076.5, 444.56],
            [1001.0, 488.15],
            [999.5, 577.93],
            [924.0, 621.52],
            [922.5, 711.3],
            [847.0, 754.89],
            [845.5, 844.66],
            [770.0, 888.25],
            [694.5, 844.66],
            [614.5, 890.85],
            [614.5, 978.03],
            [539.0, 1021.62],
            [463.5, 978.03],
            [383.5, 1024.22],
            [383.5, 1111.4],
            [308.0, 1154.99],
            [232.5, 1111.4],
            [154.0, 1154.99],
            [78.5, 1111.4],
            [0.0, 1154.99],
            [-75.5, 1111.4],
            [-154.0, 1154.99],
            [-229.5, 1111.4],
            [-308.0, 1154.99],
            [-383.5, 1111.4],
            [-383.5, 1024.22],
            [-463.5, 978.03],
            [-539.0, 1021.62],
            [-614.5, 978.03],
            [-614.5, 890.85],
            [-694.5, 844.66],
            [-770.0, 888.25],
            [-845.5, 844.66],
            [-845.5, 757.48],
            [-922.5, 711.3],
            [-922.5, 624.12],
            [-999.5, 577.93],
            [-999.5, 490.75],
            [-1076.5, 444.56],
            [-1076.5, 357.38],
            [-1153.5, 311.19],
            [-1153.5, 224.01],
            [-1078.0, 180.42],
            [-1076.5, 90.64],
            [-1001.0, 47.05],
        ]

    return vertices_LST_i
