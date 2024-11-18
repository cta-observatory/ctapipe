
.. _getting_started_users:


*************************
Getting Started for Users
*************************

.. warning::

   The following guide is for *users*. If you want to contribute to
   ctapipe as a developer, see :ref:`getting_started_dev`.


Installation
============


How To Get the Latest Version
-----------------------------

We recommend using the ``mamba`` package manager, which is a C++ reimplementation of ``conda``.
It can be found `here <https://github.com/mamba-org/mamba>`_.

To install ``ctapipe`` into an existing conda environment, use:

.. code-block:: console

   $ mamba install -c conda-forge ctapipe

You can also directly create a new environment like this (add more packages as you like):

.. code-block:: console

   $ mamba create -n ctapipe -c conda-forge python ctapipe

or with pip:

.. code-block:: console

   $ pip install ctapipe


``ctapipe`` has a number of optional dependencies that are not automatically installed
when just installing ``ctapipe``.
These are:

- `matplotlib <https://matplotlib.org/>`_ for visualization, used in `~ctapipe.visualization` and several other places.
- `bokeh <https://bokeh.org/>`_ for web-based, interactive visualizations, see `~ctapipe.visualization.bokeh`.
- `iminuit <https://scikit-hep.org/iminuit/>`_ for fitting, used in `~ctapipe.reco.impact` and `~ctapipe.image.muon`.
- `eventio <https://github.com/cta-observatory/pyeventio>`_ used for reading ``sim_telarray`` files in `~ctapipe.io.SimTelEventSource`.

You can install them individually, or if you just want to get ``ctapipe`` with all optional dependencies, use:

.. code-block:: console

   $ pip install 'ctapipe[all]'

The ``conda`` package ``ctapipe`` includes all optional dependencies, if you want to install
a minimal version of ``ctapipe`` only including required dependencies, you can use the
``ctapipe-base`` package:

.. code-block:: console

   $ mamba install -c conda-forge ctapipe-base


How To Get a Specific Version
-----------------------------

To install a specific version of ``ctapipe`` you can use the following command:

.. code-block:: console

   $ mamba install -c conda-forge ctapipe=0.17.0

or with pip:

.. code-block:: console

   $ pip install ctapipe==0.17.0
