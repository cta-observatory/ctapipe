.. _atmosphere:

==================================================
Atmosphere Models (`~ctapipe.atmosphere`)
==================================================

.. currentmodule:: ctapipe.atmosphere

Models of the atmosphere useful for tranforming between *column densities* (X in grammage units) and *heights* (in distance above sea-level units).

Example Usage:

.. code-block:: python

    from ctapipe.atmosphere import ExponentialAtmosphereDensityProfile

    density_profile = ExponentialAtmosphereDensityProfile(
          h0 = 7.8 * u.m
          rho0 = 0.000125 * u.g/u.cm**3
    )

    # convert a h_max to an x_max:

    h_max = 8 * u.km
    x_max = density_profile.integral(h_max)  # in g/cm2 units


Reference/API
=============

.. automodapi:: ctapipe.atmosphere
