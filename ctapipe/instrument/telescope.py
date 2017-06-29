from .optics import OpticsDescription
from .camera import CameraGeometry


class TelescopeDescription:
    """
    Describes a Cherenkov Telescope

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

        self.optics = optics
        self.camera = camera


    @classmethod
    def guess(cls, pix_x, pix_y, effective_focal_length):
        """
        Construct a TelescopeDescription from metadata, filling in the
        missing information using a lookup table.

        Parameters
        ----------
        pix_x: array
           array of pixel x-positions with units
        pix_y: array
           array of pixel y-positions with units
        effective_focal_length: float
           effective focal length of telescope with units (m)
        """
        camera = CameraGeometry.guess(pix_x, pix_y, effective_focal_length)
        optics = OpticsDescription.guess(effective_focal_length)
        return cls(optics=optics, camera=camera)



