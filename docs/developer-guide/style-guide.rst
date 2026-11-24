***********
Style Guide
***********


Coding Style
============

Code follows the Python
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ style guide.
This is enforced via the
`Ruff linter and formatter <https://docs.astral.sh/ruff/>`_
and the pre-commit hook set up in :ref:`getting_started_dev`.

You can also use ``ruff format <filename>`` to reformat your code by hand or install
editor plugins.


API Documentation Style
=======================

All functions, classes, and modules should contain appropriate API
documentation in their *docstrings*.  The *docstrings* should be
written in ReStructuredText format (same as the Sphinx high-level
documentation), and should follow the `NumPy Docstring Standards
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_

Docstring style is checked primarily by Ruff, using the NumPy convention.
Selected numpydoc validation checks complement Ruff for requirements it does
not cover. Both tools are configured in ``pyproject.toml`` and run via
pre-commit. Sphinx uses the same numpydoc validation settings.
Run ``pre-commit run numpydoc-validation --all-files`` to check all Python files.

Documentation for all algorithms should contain citations to external
works, which should be collected in ``bibliography.rst``. An example of
citing a reference in that file::

  this algorithm is an implementation of [author2003]_


Interactive Development Environment
===================================

It is recommended that a fully python-aware *interactive development
environment* (IDE) is used to develop code, rather than a basic text
editor. IDEs will automatically mark lines that have style
problems. The recommended IDEs are:

* `PyCharm CE <https://www.jetbrains.com/pycharm>`_ (Jetbrains)
* emacs with the `elpy <https://elpy.readthedocs.io/en/latest/>`_
  package installed
* `PyDev <https://www.pydev.org>`_ (Eclipse)

The IDEs provide a lot of support for avoiding common style and coding
mistakes, and automatic re-formatting (e.g. ``M-x py-autopep8-buffer``
in emacs).
