
.. _getting_started_dev:

******************************
Getting Started for Developers
******************************

We strongly recommend using the `mambaforge conda distribution <https://github.com/conda-forge/miniforge#mambaforge>`_.

.. warning::

   The following guide is used only if you want to *develop* the
   ``ctapipe`` package, if you just want to write code that uses it
   as a dependency, you can install ``ctapipe`` from PyPI or conda-forge.
   See :ref:`getting_started_users`


Forking vs. Working in the Main Repository
==========================================

If you are a member of CTAO (Consortium or Central Organization), or
otherwise a regular contributor to ctapipe, the maintainers can give you
access to the main repository at ``cta-observatory/ctapipe``.
Working directly in the main repository has two main advantages
over using a personal fork:

- No need for synchronisation between the fork and the main repository
- Easier collaboration on the same branch with other developers

If you are an external contributor and don't plan to contribute regularly,
you need to go for the fork.

The instructions below have versions for both approaches, select the tab that applies to your
setup.


Cloning the Repository
======================

The examples below use ssh, assuming you setup an ssh key to access GitHub.
See `the GitHub documentation <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_ if you haven't done so already.

.. tab-set::

   .. tab-item:: Working in the main repository
      :sync: main

      Clone the repository:

      .. code-block:: console

          $ git clone git@github.com:cta-observatory/ctapipe.git
          $ cd ctapipe


   .. tab-item:: Working a fork
      :sync: fork

      In order to checkout the software so that you can push changes to GitHub without
      having write access to the main repository at ``cta-observatory/ctapipe``, you
      `need to fork <https://help.github.com/articles/fork-a-repo/>`_ it.

      After that, clone your fork of the repository and add the main reposiory as a second
      remote called ``upstream``, so that you can keep your fork synchronized with the main repository.

      .. code-block:: console

          $ git clone https://github.com/[YOUR-GITHUB-USERNAME]/ctapipe.git
          $ cd ctapipe
          $ git remote add upstream https://github.com/cta-observatory/ctapipe.git


Setting Up the Development Environment
======================================

We provide a conda environment with all packages needed for development of ctapipe and a couple of additional helpful packages (like ipython, jupyter and vitables):

.. code-block:: console

    $ mamba env create -f environment.yml


Next, switch to this new virtual environment:

.. code-block:: console

    $ mamba activate cta-dev

You will need to run that last command any time you open a new
terminal to activate the conda environment.


Installing ctapipe in Development Mode
======================================

Now setup this cloned version for development.
The following command will use the editable installation feature of python packages.
From then on, all the ctapipe executables and the library itself will be
usable from anywhere, given you have activated the ``cta-dev`` conda environment.

.. code-block:: console

    $ pip install -e '.[dev]'

Using the editable installation means you won't have to rerun the installation for
simple code changes to take effect.
However, for things like adding new submodules or new entry points, rerunning the above
step might still be needed.

ctapipe supports adding new ``EventSource`` and ``Reconstructor`` implementations
through plugins. In order for the respective tests to pass you have to install the
test plugin via

.. code-block:: console

    $ pip install -e ./test_plugin


We are using the ``pre-commit``, ``code-spell`` and ``ruff`` tools
for automatic adherence to the code style
(see our :doc:`/developer-guide/style-guide`).
To enforce running these tools whenever you make a commit, setup the
`pre-commit hook <https://pre-commit.com/>`_:

.. code-block:: console

    $ pre-commit install

The pre-commit hook will then execute the tools with the same settings as when a pull request is checked on GitHub,
and if any problems are reported the commit will be rejected.
You then have to fix the reported issues before tying to commit again.
Note that a common problem is code not complying with the style guide, and that whenever this was the only problem found,
simply adding the changes resulting from the pre-commit hook to the commit will result in your changes being accepted.

Run the tests to make sure everything is OK:

.. code-block:: console

    $ pytest

Build the HTML docs locally and open them in your web browser:

.. code-block:: console

    $ make doc

Run the example Python scripts:

.. code-block:: console

    $ cd examples
    $ python xxx_example.py

Try running some command line tools:

.. code-block:: console

    $ ctapipe-info --all
    $ ctapipe-process -i dataset://gamma_prod5.simtel.zst -o test.h5  # try --help for more info

To update to the latest development version (merging in remote changes
to your local working copy):


.. tab-set::

   .. tab-item:: Working in the main repository
      :sync: main

      .. code-block:: console

         $ git pull

   .. tab-item:: Working a fork
      :sync: fork

      .. code-block:: console

         $ git fetch upstream
         $ git merge upstream/main --ff-only
         $ git push

      Note: you can also press the "Sync fork" button on the main page of your fork on the github
      and then just use ``git pull``.


Developing a New Feature or Code Change
=======================================

You should always create a new branch when developing some new code.
Make a new branch for each new feature, so that you can make pull-requests
for each one separately and not mix code from each.
It is much easier to review and merge small, well-defined contributions than
a collection of multiple, unrelated changes.

Most importantly, you should *never* add commits to the ``main`` branch of your fork,
as the main branch will often be updated in the main ``cta-observatory`` repository
and having a diverging history in the main branch of a fork will create issues when trying
to keep forks in sync.

Remember that ``git switch <name>`` [#switch]_ switches between branches,
``git switch -c <name>`` creates a new branch and switches to it,
and ``git branch`` on it's own will tell you which branches are available
and which one you are currently on.


Create a Feature Branch
-----------------------

First think of a name for your code change, here we'll use
*implement_feature_1* as an example.


To ensure you are starting your work from an up-to-date ``main`` branch,
we recommend starting a new branch like this:


.. tab-set::

   .. tab-item:: Working in the main repository
      :sync: main

      .. code-block:: console

         $ git fetch  # get the latest changes
         $ git switch -c <new branch name> origin/main  # start a new branch from main

   .. tab-item:: Working a fork
      :sync: fork

      .. code-block:: console

         $ git fetch upstream  # get latest changes from main repository
         $ git switch -c <new branch name> upstream/main # start new branch from upstream/main


Edit the Code
-------------

and make as many commits as you want (more than one is generally
better for large changes!).

.. code-block:: sh

    $ git add some_changed_file.py another_file.py
    $ git commit
      [type descriptive message in window that pops up]

and repeat. The commit message should follow the *Git conventions*:
use the imperative, the first line is a short description, followed by a blank line,
followed by details if needed (in a bullet list if applicable). You
may even refer to GitHub issue ids, and they will be automatically
linked to the commit in the issue tracker.  An example commit message::

  fix bug #245

  - changed the order of if statements to avoid logical error
  - added unit test to check for regression

Of course, make sure you frequently test via ``make test`` (or ``pytest`` in a
sub-module), check the style, and make sure the docs render correctly
(both code and top-level) using ``make doc``.

.. note::

   A git commit should ideally contain one and only one feature change
   (e.g it should not mix changes that are logically different).
   Therefore it's best to group related changes with ``git
   add <files>``. You may even commit only *parts* of a changed file
   using and ``git add -p``.  If you want to keep your git commit
   history clean, learn to use commands like ``git commit --ammend``
   (append to previous commit without creating a new one, e.g. when
   you find a typo or something small).

   A clean history and a chain of well-written commit messages will
   make it easier on code reviews to see what you did.


Push Your Changes
-----------------

The first time you push a new branch, you need to specify to which remote the branch
should be pushed [#push]_. Normally this will be ``origin``:

.. code-block:: console

   $ git push -u origin implement_feature_1

After that first setup, you can push new changes using a simple

.. code-block:: console

   $ git push


You can do this at any time and more than once. It just moves the changes
from your local branch on your development machine to your fork on github.


Integrating Changes From the ``main`` Branch
--------------------------------------------

In case of updates to the ``main`` branch during your development,
it might be necessary to update your branch to integrate those changes,
especially in case of conflicts.

To get the latest changes, run:

.. tab-set::

   .. tab-item:: Working in the main repository
      :sync: main

      .. code-block:: console

         $ git fetch

   .. tab-item:: Working a fork
      :sync: fork

      .. code-block:: console

         $ git fetch upstream

Then, update a local branch using:

.. tab-set::

   .. tab-item:: Working in the main repository
      :sync: main

      .. code-block:: console

         $ git rebase origin/main

      or

      .. code-block:: console

         $ git merge origin/main

   .. tab-item:: Working a fork
      :sync: fork

      .. code-block:: console

         $ git rebase upstream/main

      or

      .. code-block:: console

         $ git merge upstream/main

For differences between rebasing and merging and when to use which, see `this tutorial <https://www.atlassian.com/git/tutorials/merging-vs-rebasing>`_.


Create a *Pull Request*
-----------------------

When you're happy, you create PR on on your github fork page by clicking
"pull request".  You can also do this via *GitHub Desktop* if you have
that installed, by pushing the pull-request button in the
upper-right-hand corner.

Make sure to describe all the changes and give examples and use cases!

See the :ref:`pullrequests` section for more info.


Wait for a Code Review
----------------------

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
  warnings when you run the ``flake8`` syntax checker)

If the reviewer asks for changes, all you need to do is make them, ``git
commit`` them and then run ``git push`` and the reviewer will see the changes.

When the PR is accepted, the reviewer will merge your branch into the
*master* repo on cta-observatory's account.


Delete Your Feature Branch
--------------------------

since it is no longer needed (assuming it was accepted and merged in):

.. code-block:: console

    $ git switch main  # switch back to your master branch

pull in the upstream changes, which should include your new features, and
remove the branch from the local and remote (github).

.. tab-set::

   .. tab-item:: Working in the main repository
      :sync: main

      .. code-block:: console

         $ git pull

   .. tab-item:: Working a fork
      :sync: fork

      .. code-block:: console

         $ git fetch upstream
         $ git merge upstream/main --ff-only

And then delete your branch:

.. code-block:: console

   $ git branch --delete --remotes implement_feature_1


Debugging Your Code
===================

More often than not your tests will fail or your algorithm will
show strange behaviour. **Debugging** is one of the power tools each
developer should know. Since using ``print`` statements is **not** debugging and does
not give you access to runtime variables at the point where your code fails, we recommend
using ``pdb`` or ``ipdb`` for an IPython shell.
A nice introduction can be found `here <https://hasil-sharma.github.io/2017-05-13-python-ipdb/>`_.


More Development Help
=====================

For coding details, read the :ref:`guidelines` section of this
documentation.

To make git a bit easier (if you are on a Mac computer) you may want
to use the `github-desktop GUI <https://desktop.github.com/>`_, which
can do most of the fork/clone and remote git commands above
automatically. It provides a graphical view of your fork and the
upstream cta-observatory repository, so you can see easily what
version you are working on. It will handle the forking, syncing, and
even allow you to issue pull-requests.

.. rubric:: Footnotes

.. [#switch] ``git switch`` is a relatively new addition to git. If your version of git does not have it, update or use ``git checkout`` instead. The equivalent old command to ``git switch -c`` is ``git checkout -b``.

.. [#push] As of git version 2.37, you can set these options so that ``git push`` will just work,
    also for the first push:

    .. code-block:: console

       $ git config --global branch.autoSetupMerge simple
       $ git config --global push.autoSetupRemote true
