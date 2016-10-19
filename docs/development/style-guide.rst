Style Guide
==================

Coding Style
------------

Code should follow the Python PEP8 style guide. You can check if your
code has style problems using any one of the following shell commmands (you
may need to install the relevant package):

.. code-block:: sh

  % flake8  file.py   # recommended
  % pyflakes file.py
  % pep8 file.py

The `autopep8` command-line utility can be used to automatically reformat
non-conforming code to the PEP8 style.


API Documentation Style
-----------------------

All functions, classes, and modules should contain appropriate API
documentation in their *docstrings*.  The *docstrings* should be
written in ReStructuredText format (same as the Sphinx high-level
documentation), and should follow the `NumPy Docstring Standards
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard>`_


Interactive Development Environment
-----------------------------------

It is recommended that a fully python-aware *interactive development
environment* (IDE) is used to develop code, rather than a basic text
editor. IDEs will automatically mark lines that have style
problems. The recommended IDEs are:

* PyCharm CE (Jetbrains)
* emacs with the *elpy* package installed
* PyDev (Eclipse)

The IDEs provide a lot of support for avoiding common style and coding
mistakes, and automatic re-formatting (e.g. `M-x py-autopep8-buffer`
in emacs)



