.. _cep-003:

************************************************************
CEP 4 - Changing the module structure of the ctapipe package
************************************************************

* Status: draft
* Discussion: NA
* Date accepted: NA
* Last revised: 2024-01-08
* Author: Maximilian Linhoff
* Created: 2024-01-08


Abstract
========

``ctapipe``'s module structure has organically grown over its development
in the last couple of years and has some suboptimal properties.

* Large inter-dependencies between modules, sometimes even circular. E.g. importing ``ctapipe.calib`` will import ``ctapipe.image`` because the image extraction code lives in ``ctapipe.image``.
* Deep hierarchies, resulting in issues with references in the sphinx documentation due to the same class being documentated at more than two locations.

A better structure would resolve these issues and would also make it easier to
assign maintainers / code owners to specific submodules.


Proposed new structure
======================

In the new structure, we want to use a flatter hierarchy of smaller modules,
that will resolve the above mentioned issues.

We expect no disadvantages other than:
* Breaking compatibility of imports once
* More namespaces for developers to remember

The second point should be offset by the more logical structure of the namespaces.

.. code::

   ctapipe
        .atmosphere
        .camera_calibration
        .config
        .coordinates
        .core
        .data_model
        .data_volume_reduction
        .gain_selection
        .image_cleaning
        .image_features
        .instrument
        .muon
        .processors
        .reconstruction
        .tools
        .visualization


Looking at it from the current structure, we'd have the following changes:

- ``.image`` is split into ``.image_cleaning`` and ``.image_features``
- ``.calib`` is split into ``.camera_calibration``, ``.data_volume_reduction``, ``.gain_selection``
- ``.image.extractor`` is moved to ``.camera_calibration``
- ``.image.muon`` is moved to ``.muon``
- ``.containers`` is renamed to ``.data_model``
- the configuration related (``Tool``, ``Component``, ``.traits``) parts of ``.core`` are moved to ``.config``
- ``.coordinates``, ``.instrument``, ``.visualization``, ``.reconstruction`` remain as is
- The high-level API Components (``ShowerProcessor``, ``ImageProcessor``, ``DataWriter``) are moved to ``.processors``.


Related Issues
==============

* `Importing ctapipe.io is taking very long (>5s) #2476 <https://github.com/cta-observatory/ctapipe/issues/2476>`_
* `Rethink calib and image module scope and coupling #1037 <https://github.com/cta-observatory/ctapipe/issues/1037>`_
* `Move data volume reduction from image to own submodule #1394  <https://github.com/cta-observatory/ctapipe/issues/1394>`_
* `Split-up ctapipe into multiple smaller packages #2090 <https://github.com/cta-observatory/ctapipe/issues/2090>`
