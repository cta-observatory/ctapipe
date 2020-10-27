Style Guide
==================

Coding Style
------------

Code follows the Python
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ style guide.
This is enforced via the
`black formatter <https://black.readthedocs.io/en/stable/>`_
and the pre-commit hook set up in :doc:`/getting_started/index`.

You can also use `black \<filename\>` to reformat your code by hand or install
editor plugins.


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



