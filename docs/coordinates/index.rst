.. _coordinates:

==============================
Coordinates (`coordinates`)
==============================

.. currentmodule:: ctapipe.coordinates

Introduction
============

`ctapipe.coordinates` contains coordinate frame definitions and
coordinate transformation routines that are associated with the
reconstruction of Cherenkov telescope events.  It is built on
`astropy.coordinates`, which internally use the ERFA coordinate
transformation library, the open-source-licensed fork of the IAU SOFA
system.

Maybe we'll use the `gwcs <http://gwcs.readthedocs.org/en/latest/>`__ package
for generalised world coordinate transformations to implement some coordinate
transformations, e.g. precision pointing corrections using a pointing model.


TODO:
-----

- include time transform examples (using `astropy.time`), particularly
  to the CTA-standard Mission-Elapsed (MET) time system, which is a
  Terrestrial Time with a specific MJD_offset.

Getting Started
===============

Coordinate frames and transformations are implemented using the
`astropy.coordinates` framework, which uses the ERFA (open-license form
of the IAU SOFA library) internalyl for astronomical math.

The coordinate library defines a set of *frames*, which represent
different coordinate reprentations. Coordiantes sohuld be described
using an `astropy.coordinates.SkyCoord` in the appropriate frame.

Example
-------

.. code-block:: python

    >>> import astropy.units as u
    >>> from ctapipe.coordinates import CameraFrame, TelescopeFrame
    >>> camera_coord = CameraFrame(x=1*u.m, y=2*u.m, z=0*u.m)
    >>> print(camera_coord)
    <CameraFrame Coordinate: (x, y, z) in m
        (1.0, 2.0, 0.0)>
    >>> telescope_coord = camera_coord.transform_to(TelescopeFrame)
    >>> print(telescope_coord)
    <TelescopeFrame Coordinate (focal_length=15.0 m): (x, y, z) [dimensionless]
        (0.06666667, 0.13333333, 0.0)>


Note that Coordinate objects can accept arrays as input, so if one
wants to transform a large list of coordinates from one frame to
another, it is most efficient to pass them all at once. 

.. code-block:: python
                
    import astropy.units as u
    from ctapipe.coordinates import CameraFrame, TelescopeFrame
    x = np.array([0.1,0.2,0.3,0.4,0.5]) * u.m
    y = np.array([-0.1,0.4,-0.4,0.6,-0.6]) * u.m   
    camera_coord = CameraFrame(x=x, y=y, z=0*u.m)
    telescope_coord = camera_coord.transform_to(TelescopeFrame)
    print(telescope_coord)
    print("X:",telescope_coord.x)
    print("Y:",telescope_coord.y)

The result is a set of coordinates::

    <TelescopeFrame Coordinate (focal_length=15.0 m): (x, y, z) [dimensionless]
       [(0.00666667, -0.00666667, 0.0), (0.01333333, 0.02666667, 0.0),
       (0.02, -0.02666667, 0.0), (0.02666667, 0.04, 0.0),
       (0.03333333, -0.04, 0.0)]>
    X: [ 0.00666667  0.01333333  0.02        0.02666667  0.03333333]
    Y: [-0.00666667  0.02666667 -0.02666667  0.04       -0.04      ]


        
Reference/API
=============

.. automodapi:: ctapipe.coordinates
    :no-inheritance-diagram:
