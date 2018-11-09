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

.. note::

  Due to the underlying implementation of astropy coordinates and time
  representations, the transformation of a single coordinate between
  frames or time systems has a small overhead in speed.
  This can be mitigated by transforming many coordinates at once, which is
  much faster. All routines can accept arrays instead of single values.


Earth Orientation Corrections
==============================

Polar motion and leap-second corrections are taken into account automatically
by the use of `astropy.coordinates` and `astropy.time`.  Updates to the IERS
tables are loaded automatically if an internet connection is available.  If
you want to see what corrections are being used, use the following example:

.. plot::

   import matplotlib.pyplot as plt
   from astropy.utils import iers

   table = iers.IERS_Auto.open() # table containing all parameters
   px = table['PM_x']
   py = table['PM_y']
   mjd = table['MJD']
   tcorr = table['UT1_UTC_A']

   plt.figure(figsize=(9,4))
   plt.subplot(1,2,1)
   plt.plot(px, py)
   plt.xlabel("polar motion X ({})".format(px.unit.to_string('latex')))
   plt.ylabel("polar motion Y ({})".format(py.unit.to_string('latex')))

   plt.subplot(1,2,2)
   plt.plot(mjd, tcorr)
   plt.xlabel("MJD")
   plt.ylabel("UT to UTC time correction")
   plt.tight_layout()



TODO:
-----

- include time transform examples (using `astropy.time`), particularly
  to the CTA-standard Mission-Elapsed (MET) time system, which is a
  Terrestrial Time with a specific MJD_offset.

Getting Started
===============

The coordinate library defines a set of *frames*, which represent
different coordinate reprentations. Coordinates should be described
using an `astropy.coordinates.SkyCoord` in the appropriate frame.

The following special frames are defined for CTA:

* `CameraFrame`
* `GroundFrame`
* `TiltedGroundFrame`
* `NominalFrame`
* `HorizonFrame`
* `TelescopeFrame`

they can be transformed to and from any other `astropy.coordinates` frame, like
`astropy.coordinates.AltAz` or `astropy.coordinates.ICRS` (RA/Dec)

Examples
--------

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
