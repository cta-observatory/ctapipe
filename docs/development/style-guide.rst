Style Guide
==================

Coding Style
------------

Code should follow the Python
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ style guide. You
can check if your code has style problems using any one of the
following shell commmands (you may need to install the relevant
package):

.. code-block:: sh

  % pylint file.py 
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

Documentation for all algorithms should contain citations to external
works, which should be collected in `bibliography.rst`. An example of
citing a reference in that file::

  this algorithm is an implementaton of [author2003]_



Interactive Development Environment
-----------------------------------

It is recommended that a fully python-aware *interactive development
environment* (IDE) is used to develop code, rather than a basic text
editor. IDEs will automatically mark lines that have style
problems. The recommended IDEs are:

* `PyCharm CE <http://www.jetbrains.com/pycharm>`_ (Jetbrains)
* emacs with the `elpy <http://elpy.readthedocs.io/en/latest/>`_
  package installed
* `PyDev <http://www.pydev.org>`_ (Eclipse)

The IDEs provide a lot of support for avoiding common style and coding
mistakes, and automatic re-formatting (e.g. `M-x py-autopep8-buffer`
in emacs)



