.. _getting_started:

******************************
Getting Started For Developers
******************************

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

In order to checkout the software in such a way that you can read *and
commit* changes, you need to `Fork and Clone
<https://help.github.com/articles/fork-a-repo/>`_ the main ctapipe
repository (cta-observatory/ctapipe).

+++++++++++
Step 1: Fork
+++++++++++

Follow the instructions in the link above to make a *fork* of the
ctapipe repo in your own GitHub userspace. That fork will be then
called *yourusername*/ctapipe (it's as simple as clicking the fork button on `main ctapipe github page <https://github.com/cta-observatory/ctapipe>`_.

+++++++++++++
Step 2: clone
+++++++++++++

Next, you need to clone (copy to your local machine) the newly forked
ctapipe repo (make sure you put in your own username there):

.. code-block:: bash

    git clone https://github.com/[YOUR-GITHUB-USERNAME]/ctapipe.git  
    cd ctapipe


You now need to tell Git that this repo where the master CTA version is:


.. code-block:: bash
		
	git remote add cta-observatory https://github.com/cta-observatory/ctapipe.git

If that worked, then you should see a *cta-observatory* target in
addition to *origin* when typing `git remote -v`.  Later if you want
to pull in any changes from the master repo, you just need to type
`git pull cta-observatory/master`.

+++++++++++++
Step 3: Setup
+++++++++++++

Now setup this cloned version for development:
 
.. code-block:: bash

    make init     # will fetch required sub-repos and set up package 
    make develop  # will make symlinks in your python library dir

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

    ctapipe-info --tools

To update to the latest development version (merging in remote changes
to your local working copy):

.. code-block:: bash

   git pull cta-observatory/master
            
---------------------
More Development help
---------------------
 
More information on how to develop code using the GitHub-FLow workflow
(which is what we are using) can be found in the AstroPy documentation
http://astropy.readthedocs.org/en/latest/development/workflow/get_devel_version.html#get-devel
.  You would need to of course change any reference to "astropy" the
package to "ctapipe" and "astropy" the organization to
"cta-observatory", but the instructions should work.

Even easier (if you are on a Mac computer) is to use the
`github-desktop GUI <https://desktop.github.com/>`_, which can do most
of the fork/clone and remote git commands above automatically. It
provides a graphical view of your fork and the upstream
cta-observatory repository, so you can see easily what version you are
working on. It will handle the forking, syncing, and even allow you to
issue pull-requests.
