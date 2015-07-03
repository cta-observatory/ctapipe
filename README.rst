=========
 ctapipe
=========

CTA Python pipeline experimental version.

This is code for exploring a CTA data processing framework. It is not
official and not recommended for use!

* Code: https://github.com/cta-observatory/ctapipe
* Docs: https://ctapipe.readthedocs.org/


====================
Quick Start for Devs
====================

First install Anaconda python distribution for Python3.4
http://continuum.io/downloads#py34

Anaconda's distribution doesn't interfere with your local python
distribution if you have one, and can be installed without root
privileges. It contains all the required packages. To "activate"
Anaconda's python, just put it's bin directory in your path: (e.g.
`export PATH=$HOME/anaconda/bin:$PATH`). You can use a shell macro to
enable that to be able to swtch between your system python and this
one. After installing anaconda and setting your PATH, run

.. codeblock:: bash
               
    conda update --all

then for the CTA pipe module:

.. codeblock:: bash
               
    git clone https://github.com/cta-observatory/ctapipe.git
    cd ctapipe
    git submodule init
    git submodule update

The last two commands fetch the example data files. Then the following
command will enable development mode (by making symlinks to the
package in your local python package directory). After that the
package will be importable:

.. codeblock:: bash
               
    python setup.py develop  

Inside the ctapipe directory, you can type "make docshow" to
build and display the docs and code browser. or "make test" to run the
tests. 

Status shields
==============

(mostly useful for developers)

* .. image:: http://img.shields.io/travis/cta-observatory/ctapipe.svg?branch=master
    :target: https://travis-ci.org/cta-observatory/ctapipe
    :alt: Test Status

* .. image:: https://img.shields.io/coveralls/cta-observatory/ctapipe.svg
    :target: https://coveralls.io/r/cta-observatory/ctapipe
    :alt: Code Coverage
