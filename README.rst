=======
ctapipe
=======

CTA Python pipeline experimental version.

This is code for exploring a CTA data processing framework. It is not
official and not recommended for use unless you are an expert or developer!

* Code: https://github.com/cta-observatory/ctapipe
* Docs: https://cta-observatory.github.io/ctapipe/
* Example notebooks: https://github.com/cta-observatory/ctapipe/tree/master/examples/notebooks

* .. image:: http://img.shields.io/travis/cta-observatory/ctapipe.svg?branch=master
    :target: https://travis-ci.org/cta-observatory/ctapipe
    :alt: Test Status


Examples can be found in ~ctapipe/examples~ and also the notebooks in
 ~ctapipe/examples/notebooks/~ The README file in that directory will
help you run the notebook examples.

Installation for Users
----------------------

*ctapipe* and its dependencies may be installed using the *Anaconda* or
*Miniconda* package system. We recommend creating a conda virtual environment
first, to isolate the installed version and dependencies from your master
environment (this is optional).


Optionally create the virtual env:

::

  conda create -n cta ipython ipython-notebook
  source activate cta

Then, install the packages via:

::

  conda install -c cta-observatory ctapipe  

Developers should follow the development install instructions found in the
documentation above.