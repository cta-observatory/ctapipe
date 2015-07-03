.. _datasets:

==========
 Datasets
==========

.. currentmodule:: ctapipe.datasets

Introduction
============

`ctapipe.datasets` contains functions to download and generate test
and example datasets.

To keep the ``ctapipe`` code repo small, we have decided to put the
test and example files as well as the IPython notebooks in a separate
repo: https://github.com/cta-observatory/ctapipe-extra

This sounds easy, but in detail is tricky because ``ctapipe-extra``
has to be versioned and several possible locations of ``ctapipe`` and
``ctapipe-extra`` on the user's machine have to be supported (e.g. git
repo or site-packages or extra folder in home directory).

The ``ctapipe-extra`` repo is included as a git submodule in the
``ctapipe`` repo for two reasons:

1. Make it clear which version of ``ctapipe-extra`` corresponds to a
   given version of ``ctapipe``.
2. Make it easy for developers to get the right version of ``ctapipe-extra`` by typing

.. code-block:: bash

    git submodule init
    git submodule update

This setup is inspired by the one used by Sherpa (see the `sherpa repo
<https://github.com/sherpa/sherpa>`__ and the `sherpa-test-data repo
<https://github.com/sherpa/sherpa-test-data>`__ as well as their
reasoning / explanation of the setup `here
<https://github.com/sherpa/sherpa/pull/31>`__).

We will also implement utility functions in `ctapipe.datasets` and a
command line utility to fetch the correct version of
``ctapipe-extra``, even if not working out of the ``ctapipe`` git
repo.

Note that our versioning solution is coupled to git (hashes and tags
for stable versions)!


Getting Started
===============

TODO: add examples.

Reference/API
=============

TODO
