***************
Maintainer Info
***************

This is a collection of some notes for maintainers.


Python / NumPy Versions To Support
==================================

ctapipe follows `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_.

This means ctapipe will require the following minimum python / numpy releases
vs. time:

- After 2023-01-31 drop support for NumPy 1.20 (initially released on 2021-01-31)
- After 2023-04-14 drop support for Python 3.8 (initially released on 2019-10-14)
- After 2023-06-23 drop support for NumPy 1.21 (initially released on 2021-06-22)
- After 2024-01-01 drop support for NumPy 1.22 (initially released on 2021-12-31)
- After 2024-04-05 drop support for Python 3.9 (initially released on 2020-10-05)
- After 2024-06-22 drop support for NumPy 1.23 (initially released on 2022-06-22)
- After 2024-12-18 drop support for NumPy 1.24 (initially released on 2022-12-18)
- After 2025-04-04 drop support for Python 3.10 (initially released on 2021-10-04)
- After 2026-04-24 drop support for Python 3.11 (initially released on 2022-10-24)


However, for specific features, ctapipe could require more recent versions
of numpy. E.g. for the astropy quantity interoperability, we required 1.17 earlier than 2021.


How To Update the Online Docs?
==============================

The docs are automatically built and deployed using readthedocs.


How To Make a Release?
======================

#. Open a new pull request to prepare the release.
   This should be the last pull request to be merged before making the actual release.

   #. Run ``towncrier`` to render the changelog:

      .. code-block:: console

         $ git fetch
         $ git switch -c prepare_<VERSION NUMBER> origin/main
         $ towncrier build --version=<VERSION NUMBER>

   #. Add the planned new version to the ``docs/_static/switcher.json`` file, so it will be
      available from the version dropdown once the documentation is built.

   #. Update the ``AUTHORS`` file using the following command:

      .. code-block:: console

         $ bash -c "git shortlog -sne | grep -v dependabot | sed -E 's/^\s+[0-9]+\s+//' > AUTHORS"

#. Create a new GitHub release.
   A good starting point should already be made by the release drafter plugin.

#. The PyPI upload will be done automatically by GitHub Actions.

#. conda packages are built by conda-forge, the recipe is maintained here: https://github.com/conda-forge/ctapipe-feedstock/.
   A pull request to update the recipe should be opened automatically by a conda-forge bot when a new version is published to PyPI. This can take a couple of hours.
   You can make it manually to speed things up.
