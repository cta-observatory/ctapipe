
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


How To Get a Specific Version
-----------------------------

To install a specific version of ``ctapipe`` you can use the following command:

.. code-block:: console

   $ mamba install -c conda-forge ctapipe=0.17.0

or with pip:

.. code-block:: console

   $ pip install ctapipe==0.17.0
