=======
ctapipe
=======

CTA Python pipeline experimental version.

This is code for exploring a CTA data processing framework. It is not
official and not recommended for use!

* Code: https://github.com/cta-observatory/ctapipe
* Docs: https://cta-observatory.github.io/ctapipe/
* Example notebooks: https://github.com/cta-observatory/ctapipe/tree/master/examples/notebooks

  
* .. image:: http://img.shields.io/travis/cta-observatory/ctapipe.svg?branch=master
    :target: https://travis-ci.org/cta-observatory/ctapipe
    :alt: Test Status

====================
Quick Start for Devs
====================

First install Anaconda python distribution for Python3.4
http://continuum.io/downloads#py34

If you already use anaconda, but with python 2.7, you can create a
second anaconda virtual environment for python3 following the instructions here:
http://continuum.io/blog/anaconda-python-3)::
  
    # only if you already run anaconda: install a new virtualenv for
    # cta development:
    conda create -n cta python=3.4 anaconda
    source activate cta  # to switch to the cta python virtualenv

later you can switch back to your the root environment (or another) by running::
    
    source activate root  
    
Anaconda's distribution doesn't interfere with your local python
distribution if you have one, and can be installed without root
privileges. It contains all the required packages. To "activate"
Anaconda's python, just put it's bin directory in your path: (e.g.
`export PATH=$HOME/anaconda/bin:$PATH`).

After installing anaconda and setting your PATH, run the following to update the packages (for now we have no version restrictions, so the latest ones usually work)::

    conda update --all

Next you need to check out the ~ctapipe~ module and initialize it.

    git clone https://github.com/cta-observatory/ctapipe.git
    cd ctapipe
    git submodule init
    git submodule update

The last two commands fetch the example data files. The following
command should be run once to enable development mode (by making
symlinks to the package in your local python package directory). After
that the package will be importable anywhere on your system::

    make develop

next steps::

    make docshow   # build and show the documentation
    make test      # run the tests

 look at the examples in ~ctapipe/examples~ and also the notebooks in
 ~ctapipe/examples/notebooks/~ The README file in that directory will
 help you run the notebook examples.
