.. _getting_started:

******************************
Getting Started For Developers
******************************

.. warning::

   the following guide is used only if you want to *develop* the
   `ctapipe` package, if you just want to write code that uses it
   externally, you can install `ctapipe` as a conda package
   with `conda install -c cta-observatory ctapipe`.

This guide assumes you are using the *Anaconda* python distribution,
installed locally (*miniconda* should also work).

First you should create a virtual environment in which to do your
development (this will allow you to control the dependent libraries
independently of other projects). You can use either python 3.4 or
python 3.5. Here we will create a virtualenv called "cta" (but you can
choose another name if you like)

-----------------------
Set up your environment
-----------------------

.. code-block:: bash

	conda create -n cta python=3.5 astropy matplotlib scipy scikit-learn numba cython 

Next, switch to this new virtual environment and install some other useful tools for development:
	
.. code-block:: bash

	source activate cta
	
	conda install -c cta-observatory pyhessio ipython ipython-notebook pyflakes traitlets
	conda install -c conda-forge autopep8 graphviz

Next, you should create a directory where you can store the software you check out. For example:

.. code-block:: bash
    
    mkdir ctasoft
    cd ctasoft

------------------------
Get the ctapipe software
------------------------

In order to checkout the software in such a way that you can read *and
commit* changes, you need to `Fork and Clone
<https://help.github.com/articles/fork-a-repo/>`_ the main ctapipe
repository (cta-observatory/ctapipe).


++++++++++++
Step 1: Fork
++++++++++++

Follow the instructions in the link above to make a *fork* of the
ctapipe repo in your own GitHub userspace. That fork will be then
called *yourusername*/ctapipe (it's as simple as clicking the fork button on `main ctapipe github page <https://github.com/cta-observatory/ctapipe>`_.

You only need to make this fork once, when you first start developing, and
you can use it from then on.

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
		
	git remote add upstream https://github.com/cta-observatory/ctapipe.git

If that worked, then you should see a *upstream* target in
addition to *origin* when typing `git remote -v`.  Later if you want
to pull in any changes from the master repo, you just need to type
`git pull upstream master`.

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

   git pull upstream master

---------------------------------------
Developing a new feature or code change
---------------------------------------

You should always create a branch when developing some new code (unless it is
a very small change).  Genearlly make a new branch for each new feature, so
that you can make pull-requests for each one separately and not mix code
from each.  Remember that `git checkout <name>` switches between branches,
`git checkout -b <name>` creates a new branch, and `git branch` on it's own
will tell you which branches are available and which one you are currently on.

First think of a name for your code change, here we'll use
*implement_feature_1* as an example.

+++++++++++++++++++++++++++
1. Create a feature branch:
+++++++++++++++++++++++++++

.. code-block:: sh

    git checkout -b implement_feature_1

++++++++++++++++
2. Edit the code
++++++++++++++++
and make as many commits as you want (more than one is generally
better for large changes!).

.. code-block:: sh

    git add some_changed_file.py another_file.py
    git commit
      [type descriptive message in window that pops up]

and repeat. Of course, make sure you frequently test via `make test` (or
`py.test` in a sub-module), check the style, and make sure the docs  render
correctly (both code and top-level) using `make doc`.

++++++++++++++++++++++++++++++++++++++++++
3. Push your branch to your fork on github
++++++++++++++++++++++++++++++++++++++++++

(sometimes refered to as
"publishing" since it becomes public, but only in your fork) by running

.. code-block:: sh

    git push

You can do this at any time and more than once. It just moves the changes
from your local branch on your development machine to your fork on github.


++++++++++++++++++++++++
4. make a *Pull Request*
++++++++++++++++++++++++

When you're happy, you make  PR on on your github fork page by clicking "pull
request".  You can also do this via the GitHub GUI if you have that installed.

Make sure to describe all the changes and give examples and use cases!

See the :ref:`pullrequests` section for more info.

+++++++++++++++++++++++++
5. Wait for a code review
+++++++++++++++++++++++++

At least one reviewer must accept your request.

If the reviewer asks for changes, all you need to do is make them, `git
commit` them and then run `git push` and the reviewer will see the changes.
When they accept them, they will be merged into the *master* repo on
cta-observatory's account.

+++++++++++++++++++++++++++++
6. delete your feature branch
+++++++++++++++++++++++++++++

since it is no longer needed:

.. code-block:: sh

    git checkout master   # switch back to your master branch

pull in the upstream changes, which should include your new features, and
remove the branch from the local and remote (github).

.. code-block:: sh

    git pull upstream master
    git branch --delete --remotes implement_feature_1

Note the last step can also be done on the GitHub website.

---------------------
More Development help
---------------------

For coding details, read the :ref:`guidelines` section of this
documentation.

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

