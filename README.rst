=======
ctapipe |teststatus| |healthstatus| |coverage| |conda| 
=======
.. |teststatus| image:: http://img.shields.io/travis/cta-observatory/ctapipe.svg?branch=master
    :target: https://travis-ci.org/cta-observatory/ctapipe
    :alt: Test Status
.. |conda| image:: https://anaconda.org/cta-observatory/ctapipe/badges/installer/conda.svg
.. |healthstatus| image:: https://landscape.io/github/cta-observatory/ctapipe/master/landscape.svg?style=flat
   :target: https://landscape.io/github/cta-observatory/ctapipe/master
   :alt: Code Health
.. |coverage| image:: https://codecov.io/gh/cta-observatory/ctapipe/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/cta-observatory/ctapipe

Low-level data processing pipeline software for
`CTA <www.cta-observatory.org>`_ (the Cherenkov Telescope Array)

This is code is a prototype data processing framework and is under rapid
development. It is not recommended for production use unless you are an
expert or developer!

* Code: https://github.com/cta-observatory/ctapipe
* Docs: https://cta-observatory.github.io/ctapipe/
* Example notebooks: https://github.com/cta-observatory/ctapipe/tree/master/examples/notebooks

Installation for Users
----------------------

*ctapipe* and its dependencies may be installed using the *Anaconda* or
*Miniconda* package system. We recommend creating a conda virtual environment
first, to isolate the installed version and dependencies from your master
environment (this is optional).


The following command will set up a conda virtual environment, add the
necessary package channels, and download ctapipe and its dependencies. The
file *environment.yml* can be found in this repo.

::

  conda env create -n cta -f environment.yml
  source activate cta
  conda install ctapipe


Developers should follow the development install instructions found in the
`documentation <https://cta-observatory.github
.io/ctapipe/getting_started>`_.

