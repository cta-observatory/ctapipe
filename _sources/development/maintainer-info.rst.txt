***************
Maintainer info
***************

This is a collection of some notes for maintainers.

Python / numpy versions to support
----------------------------------

ctapipe follows `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`.

This means ctapipe will require the following minimum python / numpy releases
vs. time:

After 2020-06-23 drop support for Python 3.6 (initially released on Dec 23, 2016)
After 2021-07-26 drop support for Numpy 1.17 (initially released on Jul 26, 2019)
After 2021-12-22 drop support for Numpy 1.18 (initially released on Dec 22, 2019)
After 2021-12-26 drop support for Python 3.7 (initially released on Jun 27, 2018)
After 2022-06-21 drop support for Numpy 1.19 (initially released on Jun 20, 2020)
After 2023-04-14 drop support for Python 3.8 (initially released on Oct 14, 2019)

However, for specific features, ctapipe could require more recent versions
of numpy. E.g. for the astropy quantity interoperability, we required 1.17 earlier than 2021.


How to update the online docs?
------------------------------

The docs are automatically build by travis and uploaded to github pages.

To do it manually, follow these instructions:

First install `ghp-import <https://github.com/davisp/ghp-import>`__

.. code-block:: bash

    pip install ghp-import

Then build the docs:

.. code-block:: bash

    python setup.py build_docs --clean-docs

If there's no warnings, you can publish the docs with this command

.. code-block:: bash

    ghp-import -n -p -m 'Update gh-pages docs' docs/_build/html

which is equivalent to this make target so that you don't have to remember it:

.. code-block:: bash

    make doc-publish

Only ctapipe maintainers can do this, because you need write permission to the main repo.

How to make a release?
----------------------

1. Create a new github release, a good starting point should already be made by the
   release drafter plugin.

2. The PyPI upload will be done automatically by travis

3. Unfortunately, building the conda packages is a bit harder.
   Please see `the cta-conda-recipes repo <https://github.com/cta-observatory/cta-conda-recipes>`
   for instructions.
