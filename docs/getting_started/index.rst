.. _getting_started:

***************
Getting Started
***************

This guide assumes you are using the *Anaconda* python distribution, installed locally (*miniconda* should also work).

First you should create a virtual environment in which to do your developement (this will allow you to control the dependency libraries independently of other projects). You can use either python 3.4 or python 3.5. Here we will create a virtualenv called "cta" (but you can choose another name if you like)

-----------------------
Set up your environment
-----------------------

.. code-block:: bash

	conda create -n cta python=3.5 astropy matplotlib scipy scikit-learn numba cython 

Next, switch to this new virtual environment and install some other useful tools for development:
	
.. code-block:: bash

	source activate cta
	
	conda install ipython ipython-notebook ipython-qtconsole spyder pyflakes
	pip install autopep8

Next, you should create a directory where you can store the software you check out. For example:

.. code-block:: bash
    
    mkdir ctasoft
    cd ctasoft

If you want to access SimTelArrray data files (recommended), you must first install the `pyhessio` package.  This can be done by:

.. code-block:: bash

	git clone https://github.com/cta-observatory/pyhessio
	conda build pyhessio
	conda install --use-local pyhessio

------------------------
Get the ctapipe software
------------------------

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

To update to the latest development version (merging in remote changes
to your local working copy):

.. code-block:: bash

   git pull               
            
---------------
Developing Code
---------------
 
Checking out ctapipe in the manner described above is read-only, meaning that if you want to commit a change, you cannot (the master repo is locked to only the managers). Therefore, in order to develop, you need to make a personal fork on GitHub. 
This is described in the AstroPy documentation http://astropy.readthedocs.org/en/latest/development/workflow/get_devel_version.html#get-devel .  You would need to of course change any reference to "astropy" the package to "ctapipe" and "astropy" the organization to "cta-observatory", but the instructions should work.

Even easier (if you are on a Mac computer) is to use the `github-desktop GUI <https://desktop.github.com/>`_, which can do all of it for you automatically. It will handle the forking, syncing, and even allow you to issue pull-requests. 
