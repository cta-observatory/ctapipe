.. include:: references.rst

:html_theme.sidebar_secondary.remove: true
:html_theme.sidebar_primary.remove: true

.. _ctapipe:

##############################################
Prototype CTA Pipeline Framework (``ctapipe``)
##############################################

.. currentmodule:: ctapipe

|

.. image:: _static/ctapipe_logo.webp
   :class: only-light
   :align: center
   :width: 90%
   :alt: The ctapipe logo.

.. image:: _static/ctapipe_logo_dark.webp
   :class: only-dark
   :align: center
   :width: 90%
   :alt: The ctapipe logo.



**Version**: |version| **Date**: |today|

**Useful links**:
`Source Repository <https://github.com/cta-observatory/ctapipe>`__ |
`Issue Tracker <https://github.com/cta-observatory/ctapipe/issues>`__ |
`Discussions <https://github.com/cta-observatory/ctapipe/discussions>`__

**License**: BSD-3

**Python**: |python_requires|



``ctapipe`` is a framework for prototyping the low-level data processing
algorithms for the Cherenkov Telescope Array Observatory (CTAO).

.. _ctapipe_docs:

.. toctree::
  :maxdepth: 1
  :hidden:

  user-guide/index
  developer-guide/index
  api-reference/index
  changelog
  bibliography



.. grid:: 1 2 2 3

    .. grid-item-card::

        :octicon:`book;40px`

        User Guide
        ^^^^^^^^^^

        Learn how to get started as a user. This guide
        will help you install ctapipe.

        +++

        .. button-ref:: user-guide/index
            :expand:
            :color: primary
            :click-parent:

            To the user guide


    .. grid-item-card::

        :octicon:`person-add;40px`

        Developer Guide
        ^^^^^^^^^^^^^^^

        Learn how to get started as a developer.
        This guide will help you install ctapipe for development
        and explains how to contribute.

        +++

        .. button-ref:: developer-guide/index
            :expand:
            :color: primary
            :click-parent:

            To the developer guide


    .. grid-item-card::

        :octicon:`code;40px`

        API Docs
        ^^^^^^^^

        The API docs contain detailed descriptions of
        of the various modules, classes and functions
        included in ctapipe.

        +++

        .. button-ref:: api-reference/index
            :expand:
            :color: primary
            :click-parent:

            To the API docs
