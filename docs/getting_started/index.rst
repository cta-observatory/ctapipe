.. _getting_started:

***************
Getting Started
***************

Check out the `ctapipe <https://github.com/cta-observatory/ctapipe>`__ repo from Github:

.. code-block:: bash

    git clone https://github.com/cta-observatory/ctapipe.git   # if you like HTTPS
    git clone git@github.com:cta-observatory/ctapipe.git       # if you like SSH

Clone the `ctapipe-extra <https://github.com/cta-observatory/ctapipe-extra>`__ git submodule repo from Github:

.. code-block:: bash

    cd ctapipe
    git submodule init
    git submodule update

Make sure the tests and examples code finds the test and example files.
Run the tests to make sure everything is OK:

.. code-block:: bash

   make test

Build the HTML docs locally and open them in your web browser:

.. code-block:: bash

   make docshow

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

    python setup.py develop
    edit .

To update to the latest development version (merging in remote changes
to your local working copy):

.. code-block:: bash

   git pull               
               
For further information, see http://astropy.readthedocs.org/en/latest/
... most things in ctapipe work the same.
