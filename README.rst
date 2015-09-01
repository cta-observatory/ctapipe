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
    conda create -n pycta python=3.4 anaconda
    source activate pycta  # to switch to the cta python virtualenv
    
Anaconda's distribution doesn't interfere with your local python
distribution if you have one, and can be installed without root
privileges. It contains all the required packages. To "activate"
Anaconda's python, just put it's bin directory in your path: (e.g.
`export PATH=$HOME/anaconda/bin:$PATH`).

You can use a shell macro to
enable that to be able to swtch between your system python and this
one. After installing anaconda and setting your PATH, run::

    conda update --all

then for the CTA pipe module::

    git clone https://github.com/cta-observatory/ctapipe.git
    cd ctapipe
    git submodule init
    git submodule update

The last two commands fetch the example data files. Then the following
command will enable development mode (by making symlinks to the
package in your local python package directory). After that the
package will be importable ::

    python setup.py develop  

next steps::

    make docshow   # build and show the documentation
    make test      # run the tests
