.. _atmosphere:

==================================================
Atmosphere Models (`~ctapipe.atmosphere`)
==================================================

.. currentmodule:: ctapipe.atmosphere

Models of the atmosphere useful for tranforming between *column densities* (X in
grammage units) and *heights* (in distance above sea-level units).

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

.. example:: python

    from ctapipe.atmosphere import ExponentialAtmosphereDensityProfile
    density_profile = ExponentialAtmosphereDensityProfile()
    density_profile.peek()


Reference/API
=============

.. automodapi:: ctapipe.atmosphere
