***************
Maintainer Info
***************

This is a collection of some notes for maintainers.


Python / NumPy Versions To Support
==================================

ctapipe follows `SPEC-0 <https://scientific-python.org/specs/spec-0000/>`_.

This means ctapipe will require the following minimum python / numpy releases
vs. time:

- In 2025 Q1, drop support for Python 3.10
- In 2025 Q4, drop support for Python 3.11

SPEC 0 aims to keep compatibility with 3 consecutive Python releases.
Python now releases on a yearly schedule, which means that we drop the oldest
Python version at the end of each year after support has been added for the next one.


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
