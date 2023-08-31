.. include:: references.txt

:html_theme.sidebar_secondary.remove: true
:html_theme.sidebar_primary.remove: true

.. _ctapipe:

==============================================
Prototype CTA Pipeline Framework (``ctapipe``)
==============================================

.. currentmodule:: ctapipe

|

.. image:: ctapipe_logo.webp
   :class: only-light
   :align: center
   :width: 90%
   :alt: The ctapipe logo.

.. image:: ctapipe_logo_dark.webp
   :class: only-dark
   :align: center
   :width: 90%
   :alt: The ctapipe logo.

|

**Date**: |today| **Version**: |version|

**Useful links**:
`Source Repository <https://github.com/cta-observatory/ctapipe>`__ |
`Issue Tracker <https://github.com/cta-observatory/ctapipe/issues>`__ |
`Discussions <https://github.com/cta-observatory/ctapipe/discussions>`__

**License**: BSD-3 **Python**: |python_requires|

|

``ctapipe`` is a framework for prototyping the low-level data processing algorithms for the Cherenkov Telescope Array.

.. _ctapipe_docs:

.. toctree::
  :maxdepth: 1
  :hidden:

  User Guide <getting_started_users/index>
  Developer Guide <getting_started/index>
  development/index
  ctapipe_api/index
  tutorials/index
  examples/index
  tools/index
  FAQ
  data_models/index
  bibliography
  changelog



.. grid:: 1 2 2 3

    .. grid-item-card::

        :octicon:`book;40px`
        
        User Guide
        ^^^^^^^^^^

        Learn how to get started as a user. This guide
        will help you install ctapipe.

        +++

        .. button-ref:: getting_started_users/index
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

        .. button-ref:: getting_started/index
            :expand:
            :color: primary
            :click-parent:

            To the developer guide

    
    .. grid-item-card::

        :octicon:`git-pull-request;40px`

        Coding Guidelines
        ^^^^^^^^^^^^^^^^^

        These guidelines explain the coding style and the workflow. The ctapipe
        enhancement proposals (CEPs) can also be found here.

        +++

        .. button-ref:: development/index
            :expand:
            :color: primary
            :click-parent:

            To the development guidelines


    .. grid-item-card::
        
        :octicon:`code;40px`

        API Docs
        ^^^^^^^^

        The API docs contain detailed descriptions of
        of the various modules and functions included
        in ctapipe.

        +++

        .. button-ref:: ctapipe_api/index
            :expand:
            :color: primary
            :click-parent:

            To API docs


    .. grid-item-card::

        :octicon:`mortar-board;40px`
        
        Tutorials
        ^^^^^^^^^

        A collection of tutorials aimed at new users
        and developers to familiarize with ctapipe.

        +++

        .. button-ref:: tutorials/index
            :expand:
            :color: primary
            :click-parent:

            To the tutorials


    .. grid-item-card::

        :octicon:`light-bulb;40px`

        Examples
        ^^^^^^^^

        Some lower-level examples of features included in the ctapipe API.

        +++

        .. button-ref:: examples/index
            :expand:
            :color: primary
            :click-parent:

            To the examples

