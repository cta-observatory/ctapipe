============================================================
ctapipe |pypi| |conda| |doilatest| |ci| |coverage| |codacy|
============================================================

.. |ci| image:: https://github.com/cta-observatory/ctapipe/workflows/CI/badge.svg?branch=master
    :target: https://github.com/cta-observatory/ctapipe/actions?query=workflow%3ACI+branch%3Amaster
    :alt: Test Status
.. |codacy|  image:: https://api.codacy.com/project/badge/Grade/6192b471956b4cc684130c80c8214115
  :target: https://www.codacy.com/gh/cta-observatory/ctapipe?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cta-observatory/ctapipe&amp;utm_campaign=Badge_Grade
.. |conda| image:: https://anaconda.org/conda-forge/ctapipe/badges/version.svg
  :target: https://anaconda.org/conda-forge/ctapipe
.. |coverage| image:: https://codecov.io/gh/cta-observatory/ctapipe/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/cta-observatory/ctapipe
.. |doilatest| image:: https://zenodo.org/badge/37927055.svg
  :target: https://zenodo.org/badge/latestdoi/37927055
.. |pypi| image:: https://badge.fury.io/py/ctapipe.svg
    :target: https://pypi.org/project/ctapipe

Low-level data processing pipeline software for
`CTA <https://www.cta-observatory.org>`__ (the Cherenkov Telescope Array)

This is code is a prototype data processing framework and is under rapid
development. It is not recommended for production use unless you are an
expert or developer!

* Code: https://github.com/cta-observatory/ctapipe
* Docs: https://ctapipe.readthedocs.io/
* Slack: Contact Karl Kosack for invite

Citing this software
--------------------

If you use this software for a publication, please cite the Zenodo Record
for the specific version you are using and our latest publication.

You can find all ctapipe Zenodo records here: `List of ctapipe Records on Zenodo <https://zenodo.org/search?q=conceptrecid:%223372210%22&sort=-version&all_versions=True>`__.

There is also a Zenodo DOI always pointing to the latest version: |doilatest|

At this point, our latest publication is the 2021 ICRC proceeding, which you can
cite using this bibtex entry:

.. code::

  @inproceedings{ctapipe-icrc-2021,
      author = {Nöthe, Maximilian  and  Kosack, Karl  and  Nickel, Lukas  and  Peresano, Michele},
      title = {Prototype Open Event Reconstruction Pipeline for the Cherenkov Telescope Array},
      doi = {10.22323/1.395.0744},
      booktitle = {Proceedings, 37th International Cosmic Ray Conference},
      year=2021,
      volume={395},
      number={744},
      location={Berlin, Germany},
    }


Installation for Users
----------------------

*ctapipe* and its dependencies may be installed using the *Anaconda* or
*Miniconda* package system. We recommend creating a conda virtual environment
first, to isolate the installed version and dependencies from your master
environment (this is optional).


The latest version of ``ctapipe`` can be installed via::

  mamba install -c conda-forge ctapipe

or via::

  pip install ctapipe

**Note**: to install a specific version of ctapipe take look at the documentation `here <https://ctapipe.readthedocs.org/en/latest/getting_started_users/>`__.

**Note**: ``mamba`` is a C++ reimplementation of conda and can be found `here <https://github.com/mamba-org/mamba>`__.

Note this is *pre-alpha* software and is not yet stable enough for end-users (expect large API changes until the first stable 1.0 release).

Developers should follow the development install instructions found in the
`documentation <https://ctapipe.readthedocs.org/en/latest/getting_started/>`__.
