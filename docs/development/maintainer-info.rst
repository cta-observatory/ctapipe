***************
Maintainer info
***************

This is a collection of some notes for maintainers.

How to update the online docs?
------------------------------

Building the ctapipe docs on readthedocs doesn't work (see `GH-1 <https://github.com/cta-observatory/ctapipe/issues/1>`__).
So for now from time to time maintainers should build the docs locally and then upload them to
`Github pages <https://help.github.com/articles/creating-project-pages-manually/>`__.

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

Making a ctapipe release is easy. Just follow these step-by-step instructions:


TODO