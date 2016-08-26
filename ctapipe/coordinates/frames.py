# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
TODO / Questions
----------------

- Is part of this implemented in SOFA? (exposed via Astropy?)
- Check against which other code(s)? HESS?

Coordinate frames
-----------------

- AltAz (1)
- GroundFrame (1)
- TiltedTelescopeFrame (N) - attributes: tel locations + tel pointings
- TelescopeFrame (N)
- CameraFrame (N)

Attributes:

- Camera focal length f (in meters)
- Pointing direction (Alt, Az)
- Telescope location (X, Y, Z) in ground system (array center is at (0, 0, 0)
- IACT array location on Earth

How to implement this?
----------------------

Step 0
++++++

Define CameraFrame (attributes: pointing Alt, Az + focal length) and connect it to AltAzFrame

Step 1
++++++

Goal: Starting with Hillas parameters from two showers, use the major axis line intersection
to compute the event direction and impact location (it's in the tilted system).
Use the HESS shower example images and reco parameters in the FITS file for testing ...

References
----------

* Gillessen PhD thesis section 4.1 (http://archiv.ub.uni-heidelberg.de/volltextserver/4754/)
* Isabel Braun PhD thesis (http://archiv.ub.uni-heidelberg.de/volltextserver/7354/)
* Function cam_to_ref in rec_tools.c in hessioxxx

"""
import numpy as np
import astropy.units as u
from astropy.coordinates import (BaseCoordinateFrame, FrameAttribute, SphericalRepresentation,
                                 CartesianRepresentation, RepresentationMapping, FunctionTransform,
                                 )
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.angles import rotation_matrix

__all__ = [
    'CameraFrame',
    'TelescopeFrame',
    'TiltedTelescopeFrame',
    'GroundFrame',
]


class CameraFrame(BaseCoordinateFrame):
    """Camera coordinate frame.
    """
    default_representation = CartesianRepresentation


class TelescopeFrame(BaseCoordinateFrame):
    """Telescope coordinate frame.

    Frame attributes:

    - Focal length
    """
    default_representation = CartesianRepresentation

    # TODO: probably this should not have a default!
    focal_length = FrameAttribute(default=15.*u.m)
    # focal_length = FrameAttribute(default=None)


class TiltedTelescopeFrame(BaseCoordinateFrame):
    """Tilted telescope coordinate frame.
    """
    default_representation = CartesianRepresentation

    # Is this the right frame for these attributes?
    pointing_direction = FrameAttribute(default=None)
    telescope_location = FrameAttribute(default=None)
    # time?


class GroundFrame(BaseCoordinateFrame):
    """Ground coordinate frame.
    """
    default_representation = CartesianRepresentation

    observatory_location = FrameAttribute(default=None)


@frame_transform_graph.transform(FunctionTransform, CameraFrame, TelescopeFrame)
def camera_to_telescope(camera_coord, telescope_frame):

    xyz_camera = camera_coord.cartesian.xyz
    f = telescope_frame.focal_length

    xyz_telescope = xyz_camera / f

    representation = CartesianRepresentation(xyz_telescope)

    return telescope_frame.realize_frame(representation)


@frame_transform_graph.transform(FunctionTransform, TelescopeFrame, CameraFrame)
def telescope_to_camera(telescope_coord, camera_frame):

    xyz_telescope = telescope_coord.cartesian.xyz
    f = telescope_coord.focal_length

    xyz_camera = f * xyz_telescope

    representation = CartesianRepresentation(xyz_camera)

    return camera_frame.realize_frame(representation)
