.. _getting_started:

***************
Getting Started
***************

This guide assumes you are using the *Anaconda* python distribution, installed locally (*miniconda* should also work).

First you should create a virtual environment in which to do your developement (this will allow you to control the dependency libraries independently of other projects). You can use either python 3.4 or python 3.5. Here we will create a virtualenv called "cta" (but you can choose another name if you like)

.. code-block:: bash

	conda env create -n cta python=3.5 astropy matplotlib scipy scikit-learn


Next, switch to this new virtual environment
	
.. code-block:: bash

	source activate cta

Next, you should create a directory where you can store the software you check out. For example:

.. code-block:: bash
    
    mkdir ctasoft
    cd ctasoft

If you want to access SimTelArrray data files (recommended), you must first install the `pyhessio` package.  This can be done by:

.. code-block:: bash

	git clone https://github.com/cta-observatory/pyhessio
	conda build pyhessio
	conda install --use-local pyhessio

Next, check out the `ctapipe <https://github.com/cta-observatory/ctapipe>`__ repo:

.. code-block:: bash

    git clone https://github.com/cta-observatory/ctapipe.git   # if you like HTTPS
    
Now setup this checked out version for development:
 
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
