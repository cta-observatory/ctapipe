.. _atmosphere:

==================================================
Atmosphere Models (`~ctapipe.atmosphere`)
==================================================

.. currentmodule:: ctapipe.atmosphere

Models of the atmosphere useful for transforming between *column densities* (X
in grammage units) and *heights* (in distance above sea-level units).

Example Usage:

.. code-block:: python

    from ctapipe.atmosphere import ExponentialAtmosphereDensityProfile

    density_profile = ExponentialAtmosphereDensityProfile(
          h0 = 7.8 * u.m
          rho0 = 0.000125 * u.g/u.cm**3
    )

    # convert a h_max  in meters to an x_max in g/cm2 units:

    x_max = density_profile.line_of_sight_integral(
        distance=8 * u.km,
        zenith_angle=40 * u.deg
    )


You can also get a quick plot of any profile:

.. code-block:: python

    from ctapipe.atmosphere import ExponentialAtmosphereDensityProfile
    density_profile = ExponentialAtmosphereDensityProfile()
    density_profile.peek()


Reading an atmosphere from a file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A `~ctapipe.atmosphere.TableAtmosphereDensityProfile` can be constructed from an input
`~astropy.table.Table` that has columns named ``HEIGHT``, ``DENSITY`` and
``COLUMN_DENSITY``.

.. code-block:: python

    from astropy.table import Table
    from astropy import units as u

    from ctapipe.atmosphere import TableAtmosphereDensityProfile

    table = Table(
        dict(
            HEIGHT=[1,10,20] * u.km,
            DENSITY=[0.00099,0.00042, 0.00009] * u.g / u.cm**3
            COLUMN_DENSITY=[1044.0, 284.0, 57.0] * u.g / u.cm**2
        )
    )

    TableAtmosphereDensityProfile(table=table).peek()



Some `~ctapipe.io.EventSource` implementations also support reading and creating
a `~ctapipe.atmosphere.TableAtmosphereDensityProfile` from a file automatically. They provide a
`~ctapipe.io.EventSource.atmosphere_density_profile` property as follows:

.. code-block:: python

    from ctapipe.io import EventSource

    url = "dataset://gamma_20deg_0deg_run2___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10-100evts.simtel.zst"

    with EventSource(url) as source:
        profile = source.atmosphere_density_profile

    if profile:
        profile.peek()


Reference/API
=============

.. automodapi:: ctapipe.atmosphere
