==================================================================================
 ctapipe |pypi| |conda| |doilatest| |ci| |sonarqube_coverage| |sonarqube_quality|
==================================================================================

.. |ci| image:: https://github.com/cta-observatory/ctapipe/actions/workflows/ci.yml/badge.svg?branch=main
    :target: https://github.com/cta-observatory/ctapipe/actions/workflows/ci.yml
    :alt: Test Status
.. |sonarqube_quality| image:: https://sonar-cta-dpps.zeuthen.desy.de/api/project_badges/measure?project=cta-observatory_ctapipe_AY52EYhuvuGcMFidNyUs&metric=alert_status&token=sqb_a533204f382b350568e922385cab7c2394587458
    :target: https://sonar-cta-dpps.zeuthen.desy.de/dashboard?id=cta-observatory_ctapipe_AY52EYhuvuGcMFidNyUs
    :alt: sonarqube quality gate
.. |sonarqube_coverage| image:: https://sonar-cta-dpps.zeuthen.desy.de/api/project_badges/measure?project=cta-observatory_ctapipe_AY52EYhuvuGcMFidNyUs&metric=coverage&token=sqb_a533204f382b350568e922385cab7c2394587458
    :target: https://sonar-cta-dpps.zeuthen.desy.de/api/project_badges/measure?project=cta-observatory_ctapipe_AY52EYhuvuGcMFidNyUs&metric=coverage&token=sqb_a533204f382b350568e922385cab7c2394587458)](https://sonar-cta-dpps.zeuthen.desy.de/dashboard?id=cta-observatory_ctapipe_AY52EYhuvuGcMFidNyUs
    :alt: sonarqube code coverage
.. |conda| image:: https://anaconda.org/conda-forge/ctapipe/badges/version.svg
  :target: https://anaconda.org/conda-forge/ctapipe
.. |doilatest| image:: https://zenodo.org/badge/37927055.svg
  :target: https://zenodo.org/badge/latestdoi/37927055
.. |pypi| image:: https://badge.fury.io/py/ctapipe.svg
    :target: https://pypi.org/project/ctapipe

Low-level data processing pipeline software for the
`CTAO (Cherenkov Telescope Array Observatory) <https://www.ctao.org>`__.

This is code is a prototype data processing framework and is under rapid
development. It is not recommended for production use unless you are an
expert or developer!

* Code: https://github.com/cta-observatory/ctapipe
* Docs: https://ctapipe.readthedocs.io/
* Slack: Contact Karl Kosack for invite

Citing this software
====================

If you use this software for a publication, please cite the Zenodo Record
for the specific version you are using and our latest publication.

You can find all ctapipe Zenodo records here: `List of ctapipe Records on Zenodo <https://zenodo.org/search?q=conceptrecid:%223372210%22&sort=-version&all_versions=True>`__.

There is also a Zenodo DOI always pointing to the latest version: |doilatest|

At this point, our latest publication is the `2023 ICRC proceeding <https://doi.org/10.22323/1.444.0703>`_, which you can
cite using this bibtex entry:

.. code::

   @inproceedings{ctapipe-icrc-2023,
     author = {Linhoff, Maximilian and Beiske, Lukas and Biederbeck, Noah and Fröse, Stefan and Kosack, Karl and Nickel, Lukas},
     title = {ctapipe -- Prototype Open Event Reconstruction Pipeline for the Cherenkov Telescope Array},
     usera = {for the CTA Consortium and Observatory},
     doi = {10.22323/1.444.0703},
     booktitle = {Proceedings, 38th International Cosmic Ray Conference},
     year=2023,
     volume={444},
     number={703},
     location={Nagoya, Japan},
   }


Installation for Users
======================

*ctapipe* and its dependencies may be installed using the *Anaconda* or
*Miniconda* package system. We recommend creating a conda virtual environment
first, to isolate the installed version and dependencies from your main
environment (this is optional).


The latest version of ``ctapipe`` can be installed via::

  mamba install -c conda-forge ctapipe

or via::

  pip install ctapipe

**Note**: to install a specific version of ctapipe take look at the documentation `here <https://ctapipe.readthedocs.io/en/latest/user-guide/index.html>`__.

**Note**: ``mamba`` is a C++ reimplementation of conda and can be found `here <https://github.com/mamba-org/mamba>`__.

Note this is *pre-alpha* software and is not yet stable enough for end-users (expect large API changes until the first stable 1.0 release).

Developers should follow the development install instructions found in the
`documentation <https://ctapipe.readthedocs.io/en/latest/developer-guide/getting-started.html>`__.
