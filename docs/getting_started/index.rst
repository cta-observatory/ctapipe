.. _getting_started:

***************
Getting Started
***************

If you want to access SimTelArrray data files (recommended), you must first install the `pyhessio` package.  This can be done by:

.. code-block:: bash

		git clone https://github.com/cta-observatory/pyhessio
		conda build pyhessio
		conda install --use-local pyhessio

Next, check out the `ctapipe <https://github.com/cta-observatory/ctapipe>`__ repo:

.. code-block:: bash

    git clone https://github.com/cta-observatory/ctapipe.git   # if you like HTTPS
 
.. code-block:: bash

    cd ctapipe
    make init     # will fetch required sub-repos and set up package 
    make develop  # will make symlinks in your python library dir


Make sure the tests and examples code finds the test and example files.
Run the tests to make sure everything is OK:

.. code-block:: bash

   make test

Build the HTML docs locally and open them in your web browser:

.. code-block:: bash

   make doc-show

Run the example Python scripts:

.. code-block:: bash

    cd examples
    python xxx_example.py

Run the command line tools:

.. code-block:: bash

    python setup.py install
    ctapipe-info --tools

**Start hacking and contributing:**  The following command will put
the package in "developer mode", meaning that it will make a symlink
of your checked-out working directory in your local (user) python
package library directory (which is usually something like
`$HOME/.local/lib/python/site-packages/`. Then you can access the development
`ctapipe` from anywhere on your system.

.. code-block:: bash

    make develop
    edit .

To update to the latest development version (merging in remote changes
to your local working copy):

.. code-block:: bash

   git pull               
               
For further information, see http://astropy.readthedocs.org/en/latest/
... most things in ctapipe work the same.
