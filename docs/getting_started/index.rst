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

You can use *python 3.5* or above (we currently test on 3.5 and 3.6)

------------------------
Get the ctapipe software
------------------------

In order to checkout the software in such a way that you can read *and
commit* changes, you need to `Fork and Clone
<https://help.github.com/articles/fork-a-repo/>`_ the main ctapipe
repository (cta-observatory/ctapipe).

First, it's useful to make a directory where you have can check out
cta GIT repos (this is optinal - you can put it anywhere)

.. code-block:: console
    
    $ mkdir ctasoft
    $ cd ctasoft

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Step 1: Fork the Master CTA-Observatory ctapipe repository
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Follow the instructions in the link above to make a *fork* of the
ctapipe repo in your own GitHub userspace. That fork will be then
called *yourusername*/ctapipe (it's as simple as clicking the fork button on `main ctapipe github page <https://github.com/cta-observatory/ctapipe>`_.

You only need to make this fork once, when you first start developing, and
you can use it from then on.

+++++++++++++++++++++++++++++++++++++++++
Step 2: clone your forked version locally
+++++++++++++++++++++++++++++++++++++++++

Next, you need to clone (copy to your local machine) the newly forked
ctapipe repo (make sure you put in your own username there):

.. code-block:: console

    $ git clone https://github.com/[YOUR-GITHUB-USERNAME]/ctapipe.git  
    $ cd ctapipe


You now need to tell Git that this repo where the master CTA version is:

.. code-block:: console
		
    $ git remote add upstream https://github.com/cta-observatory/ctapipe.git

If that worked, then you should see a *upstream* target in
addition to *origin* when typing `git remote -v`.  Later if you want
to pull in any changes from the master repo, you just need to type
`git pull upstream master`.


+++++++++++++++++++++++++++++++++++++++
Step 4: Set up your package environment
+++++++++++++++++++++++++++++++++++++++

Change to the directory where you cloned `ctapipe`, and type:

.. code-block:: console

       
    $ conda env create -n cta-dev -f environment.yml

	
This will create a conda virtual environment called `cta-dev` with all
the ctapipe dependencies and a few useful packages for development and
interaction. Next, switch to this new virtual environment:
	
.. code-block:: console

    $ source activate cta-dev
	

You will need to type that last command any time you open a new
terminal to actiavate the virtual environment (you can of course
install everything into the base Anaconda environment without creating
a virtual environment, but then you may have trouble if you want to
install other packages with different requirements on the
dependencies)

+++++++++++++++++++++++++++++++++++++
Step 5: Setup ctapipe for development
+++++++++++++++++++++++++++++++++++++

Now setup this cloned version for development. The following command
will make symlinks in your python library directory to your ctapipe
installation (it creates a `.pth` file, there is no need to set
PYTHONPATH, in fact it should be blank to avoid other problems). From
then on, all the ctapipe binaries and the library itself will be
usable from anywhere.

.. code-block:: console

    $ make develop  

.. note::

   If the previous command fails with an error similar to "*no module
   named ctapipe._version_cache*", it is because the version tags are
   missing in your git repo (a problem with older versions of git
   before 1.9.0). To fix the problem, simply type `git fetch upstream
   --tags` and try `make develop` again)

    
Run the tests to make sure everything is OK:

.. code-block:: console

    $ pytest    # if using an older version of python, type py.test instead

Build the HTML docs locally and open them in your web browser:

.. code-block:: console

    $ make doc   

Run the example Python scripts:

.. code-block:: console

    $ cd examples
    $ python xxx_example.py

try running some command line tools:

.. code-block:: console

    $ ctapipe-info --all
    $ ctapipe-camdemo --camera=NectarCam  # try --help for more info

To update to the latest development version (merging in remote changes
to your local working copy):

.. code-block:: console

   $ git pull upstream master

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

    $ git checkout -b implement_feature_1

++++++++++++++++
2. Edit the code
++++++++++++++++

and make as many commits as you want (more than one is generally
better for large changes!).

.. code-block:: sh

    $ git add some_changed_file.py another_file.py
    $ git commit
      [type descriptive message in window that pops up]

and repeat. The commit message should follow the *GIT conventions*:
the first line is a short description, followed by a blank line,
followed by details if needed (in a bullet list if applicable). You
may even refer to GitHub issue ids, and they will be automatically
linked to the commit in the issue tracker.  An example commit message::
  
  fixed bug #245 

  - changed the order of if statements to avoid logical error
  - added unit test to check for regression

Of course,make sure you frequently test via `make test` (or `py.test` in a
sub-module), check the style, and make sure the docs render correctly
(both code and top-level) using `make doc`.

.. note::

   A git commit should ideally contain one and only one feature change
   (e.g it should not mix changes that are logically different
   together). Therefore it's best to group related changes with `git
   add <files>`. You may even commit only *parts* of a changed file
   using and `git add -p`.  If you want to keep your git commit
   history clean, learn to use commands like `git commit --ammend`
   (append to previous commit without creating a new one, e.g. when
   you find a typo or something small)

   A clean history and a chain of well-written commit messages will
   make it easier on code reviews to see what you did.

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

When you're happy, you make PR on on your github fork page by clicking
"pull request".  You can also do this via *GitHub Desktop* if you have
that installed, by pushing the pull-request button in the
upper-right-hand corner.

Make sure to describe all the changes and give examples and use cases!

See the :ref:`pullrequests` section for more info.

+++++++++++++++++++++++++
5. Wait for a code review
+++++++++++++++++++++++++

Keep in mind the following:

* At least one reviewer must look at your code and accept your
  request. They may ask for changes before accepting.
* All unit tests must pass.  They are automatically run by Travis when
  you submit or update your pull request and you can monitor the
  results on the pull-request page.  If there is a test that you added
  that should *not* pass because the feature is not yet implemented,
  you may `mark it as skipped temporarily
  <https://docs.pytest.org/en/latest/skipping.html>`_ until the
  feature is complete.
* All documentation must build without errors. Again, this is checked
  by Travis.  It is your responsibility to run "make doc" and check
  that you don't have any syntax errors in your docstrings.
* All code you have written should follow the style guide (e.g. no
  warnings when you run the `flake8` syntax checker)

If the reviewer asks for changes, all you need to do is make them, `git
commit` them and then run `git push` and the reviewer will see the changes.

When the PR is accepted, the reviewer will merge your branch into the
*master* repo on cta-observatory's account.  

+++++++++++++++++++++++++++++
6. delete your feature branch
+++++++++++++++++++++++++++++

since it is no longer needed (assuming it was accepted and merged in):

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

To make git a bit easier (if you are on a Mac computer) you may want
to use the `github-desktop GUI <https://desktop.github.com/>`_, which
can do most of the fork/clone and remote git commands above
automatically. It provides a graphical view of your fork and the
upstream cta-observatory repository, so you can see easily what
version you are working on. It will handle the forking, syncing, and
even allow you to issue pull-requests.

