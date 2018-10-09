.. include:: references.txt

.. _ctapipe:

=====================================================
 Prototype CTA Pipeline Framework (:mod:`ctapipe`)
=====================================================

.. currentmodule:: ctapipe

**version**:  |version|

.. image:: ctapipe_logo.png
   :align: center
   :width: 90%

What is ctapipe?
================

`ctapipe` is a framework for prototyping the low-level data processing algorithms for the Cherenkov Telescope Array. 

.. CAUTION::
   This is not yet stable code, so expect large and rapid changes to
   structure and functionality as we explore various design choices before
   the 1.0 release.

* Code, feature requests, bug reports, pull requests: https://github.com/cta-observatory/ctapipe
* Docs: https://cta-observatory.github.io/ctapipe/
* License: BSD-3
* Python 3.6 or later (Python 2 is not supported)



.. _ctapipe_docs:

General documentation
=====================

.. toctree::
  :maxdepth: 1
  :glob:

  getting_started/index
  development/index
  tools/index
  */index
  FAQ
  bibliography
  changelog

Module API Status (relative to next release)
============================================

* **stable** = should not change drastically in next release
* **caution** = mostly stable, but expect some changes
* **unstable** = expect large changes and avoid heavy reliance
* **experimental** = stable feature, but under evaluation
* **deprecated** = do not use

================  ===============
 Module           Status
================  ===============
`analysis`        **stable**
`calib`           **stable**
`coordinates`     caution
`core`            **stable**
`flow`            experimental
`instrument`      **stable**
`io`              **stable** 
`reco`            **stable**
`utils`           **stable**
`visualization`   **stable**
================  ===============


Development Help
================

* Development workflow examples from AstroPy: http://astropy.readthedocs.org/en/latest/development/workflow/development_workflow.html
* GIT tutorial: https://www.atlassian.com/git/tutorials/syncing/git-pull
* Code distribution and Packaging https://packaging.python.org/en/latest/
