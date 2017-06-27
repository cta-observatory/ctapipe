.. include:: references.txt

.. _ctapipe:

=====================================================
 CTA Experimental Pipeline Framework (:mod:`ctapipe`)
=====================================================

.. currentmodule:: ctapipe

**version**:  |version|

.. image:: event.png
   :align: center
   :width: 90%

What is ctapipe?
================

`ctapipe` is an experimental framework for the data processing
pipelines for CTA.

.. CAUTION::
   This is not yet stable code, so expect large and rapid changes to
   structure and functionality as we explore various design choices.

* Code, feature requests, bug reports, pull requests: https://github.com/cta-observatory/ctapipe
* Docs: https://cta-observatory.github.io/ctapipe/
* License: BSD-3
* Python 3.4 or later (Python 2 is not supported)



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
`analysis`        empty
`calib`           caution
`coordinates`     **stable**
`core`            **stable**
`flow`            experimental
`instrument`      unstable
`pipeline`        deprecated
`plotting`        caution
`reco`            caution
`utils`           **stable**
`visualization`   **stable**
================  ===============


Authors (as of 2017-04-28)
==========================
(computed from the list of GIT commits)

- Karl Kosack (project lead)
- Abelardo Moralejo Olaizola
- Alison Mitchell
- Christoph Deil              
- Dan Parsons                 
- David Landriu               
- Gernot Maier                 
- Jason Watson                
- Jean Jacquemier             
- Jeremie Decock              
- Justus Zorn
- Kai Bruegge                 
- Maximilian Noethe
- Markus Gaug
- Michael Punch               
- Michele Mastropietro        
- Miguel Nievas
- Paulo Cumani
- Pierre Aubert
- Raquel de los Reyes         
- Tarek Hassan                
- Tino Michael                
- Wrijupan Bhattacharyya      
- Raquel de los Reyes



Development Help
================

* Development workflow examples from AstroPy: http://astropy.readthedocs.org/en/latest/development/workflow/development_workflow.html
* GIT tutorial: https://www.atlassian.com/git/tutorials/syncing/git-pull
* Code distribution and Packaging https://packaging.python.org/en/latest/
