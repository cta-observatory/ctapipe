***************
Maintainer info
***************

This is a collection of some notes for maintainers.

Python / numpy versions to support
----------------------------------

ctapipe follows `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_.

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

The docs are automatically using readthedocs and deployed there.


How to make a release?
----------------------
1. Open a new pull request to prepare the release.
   This should be the last pull request to be merged before making the actual release.

   Run ``towncrier`` in order to do this:

   .. code-block:: bash

      towncrier build --version=<VERSION NUMBER>


2. Create a new github release, a good starting point should already be made by the
   release drafter plugin.

3. The PyPI upload will be done automatically by travis

4. conda packages are built by conda-forge, the recipe is maintained here: https://github.com/conda-forge/ctapipe-feedstock/
   An pull request to update the recipe should be opened automatically by a conda-forge bot when a new version is published to PyPi.
