
.. _getting_started_users:

*************************
Getting Started For Users
*************************

.. warning::

   The following guide is for *users*. As of now this will just cover
   the basic installation procedure. This guide will be extended in
   the future.


------------------------
Get the ctapipe software
------------------------

+++++++++++++++++++++++++++++
How to get the latest version
+++++++++++++++++++++++++++++

We recommend using the ``mamba`` package manager, which is a C++ reimplementation of ``conda``.
It can be found `here <https://github.com/mamba-org/mamba>`_.
Once you created a new virtual environment, you can install ``ctapipe`` with the following command::

  mamba install -c conda-forge ctapipe

or with pip::

  pip install ctapipe


+++++++++++++++++++++++++++++
How to get a specific version
+++++++++++++++++++++++++++++

To install a specific version of ``ctapipe`` you can use the following command::
  
  mamba install -c conda-forge ctapipe==0.17.0

or with pip::

  pip install ctapipe==0.17.0
