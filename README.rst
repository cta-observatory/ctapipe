============================================================
ctapipe |ci| |codacy| |coverage| |conda| |doilatest|
============================================================

.. |ci| image:: https://github.com/cta-observatory/ctapipe/workflows/CI/badge.svg?branch=master
    :target: https://github.com/cta-observatory/ctapipe/actions?query=workflow%3ACI+branch%3Amaster
    :alt: Test Status
.. |codacy|  image:: https://api.codacy.com/project/badge/Grade/6192b471956b4cc684130c80c8214115
  :target: https://www.codacy.com/gh/cta-observatory/ctapipe?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cta-observatory/ctapipe&amp;utm_campaign=Badge_Grade
.. |conda| image:: https://anaconda.org/conda-forge/ctapipe/badges/installer/conda.svg
  :target: https://anaconda.org/conda-forge/ctapipe
.. |coverage| image:: https://codecov.io/gh/cta-observatory/ctapipe/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/cta-observatory/ctapipe
.. |doilatest| image:: https://zenodo.org/badge/37927055.svg
  :target: https://zenodo.org/badge/latestdoi/37927055
.. |doiv07| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3372211.svg
   :target: https://doi.org/10.5281/zenodo.3372211
.. |doiv08| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3837306.svg
   :target: https://doi.org/10.5281/zenodo.3837306
.. |doiv09| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4084989.svg
   :target: https://doi.org/10.5281/zenodo.4084989
.. |doiv010| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4581045.svg
   :target: https://doi.org/10.5281/zenodo.4581045
.. |doiv011| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4746317.svg
   :target: https://doi.org/10.5281/zenodo.4746317

Low-level data processing pipeline software for
`CTA <www.cta-observatory.org>`_ (the Cherenkov Telescope Array)

This is code is a prototype data processing framework and is under rapid
development. It is not recommended for production use unless you are an
expert or developer!

* Code: https://github.com/cta-observatory/ctapipe
* Docs: https://ctapipe.readthedocs.io/
* Slack: Contact Karl Kosack for invite

Citing this software
--------------------
If you use this software for a publication, please cite the proper version using the following DOIs:

- v0.11.0 : |doiv011|
- v0.10.5 : |doiv010|
- v0.9.1 : |doiv09|
- v0.8.0 : |doiv08|
- v0.7.0 : |doiv07|

Installation for Users
----------------------

*ctapipe* and its dependencies may be installed using the *Anaconda* or
*Miniconda* package system. We recommend creating a conda virtual environment
first, to isolate the installed version and dependencies from your master
environment (this is optional).


The following command will set up a conda virtual environment, add the
necessary package channels, and install ctapipe specified version and its dependencies::

    CTAPIPE_VER=0.11.0
    wget https://raw.githubusercontent.com/cta-observatory/ctapipe/v$CTAPIPE_VER/environment.yml
    conda env create -n cta -f environment.yml
    conda activate cta
    conda install -c conda-forge ctapipe=$CTAPIPE_VER


**Note**: To prevent packages you installed with `pip install --user` from taking precedence over the conda environment, run::

    conda env config vars set PYTHONNOUSERSITE=1 -n cta

  
**Note**: this environment contains many useful packages that are not strictly requirements of ctapipe.
To get only ctapipe and its direct dependencies, just do ``conda install -c conda-forge ctapipe[=<version>]`` in an environment
of your choice. 

**Note**: If you encounter long ``Solving environment`` times with conda, try using ``mamba`` (https://github.com/mamba-org/mamba) instead.

The file *environment.yml* can be found in this repo. 
Note this is *pre-alpha* software and is not yet stable enough for end-users (expect large API changes until the first stable 1.0 release).

Developers should follow the development install instructions found in the
`documentation <https://cta-observatory.github
.io/ctapipe/getting_started>`_.

