"""Tests for MirrorFacetsDescription."""

import numpy as np
from astropy import units as u

from ctapipe.instrument.optics import MirrorFacetShape, MirrorFacetsDescription


def test_mirror_facets_description():
    """Create a mirror facet description and convert its shapes to enums."""
    description = MirrorFacetsDescription(
        id=np.array([1, 2, 3]),
        x=np.array([0.0, 1.0, 2.0]) * u.m,
        y=np.array([0.0, 0.5, 1.0]) * u.m,
        z=np.array([10.0, 10.1, 10.2]) * u.m,
        nx=np.array([0.0, 0.0, 0.0]),
        ny=np.array([0.0, 0.0, 0.0]),
        nz=np.array([1.0, 1.0, 1.0]),
        diameter=np.array([1.2, 1.2, 1.2]) * u.m,
        mirror_shape=np.array(["CIRCLE", "HEXAGON", "SQUARE"]),
    )

    print (description.diameter)
    
    #np.testing.assert_array_equal(description.id, [1, 2, 3])
    #np.testing.assert_array_equal(description.x, [0, 1, 2] * u.m)
    #np.testing.assert_array_equal(description.y, [0, 0.5, 1] * u.m)
    #np.testing.assert_array_equal(description.z, [10, 10.1, 10.2] * u.m)
    #np.testing.assert_array_equal(description.nx, [0, 0, 0])
    #np.testing.assert_array_equal(description.ny, [0, 0, 0])
    #np.testing.assert_array_equal(description.nz, [1, 1, 1])
    #np.testing.assert_array_equal(description.diameter, [1.2, 1.2, 1.2] * u.m)
    #np.testing.assert_array_equal(
    #    description.mirror_shape,
    #    [
    #        MirrorFacetShape.CIRCLE,
    #        MirrorFacetShape.HEXAGON,
    #        MirrorFacetShape.SQUARE,
    #    ],
    #)
