.. _cep-003:

************************************************************
CEP 3 - Dropping Support for Image Parameters in CameraFrame
************************************************************

* Status: proposed
* Discussion: DataPipe F2F Meeting and [`#2405 <https://github.com/cta-observatory/ctapipe/pull/2405>`__], ctapipe developer meeting 2023-11-24
* Date accepted:
* Last revised: 2023-09-22
* Author: Maximilian Linhoff
* Created: 2023-09-22


Abstract
========

Currently, ctapipe supports computing all image parameters in two variants:

* Using a ``CameraGeometry`` where pixel coordinates are expressed in ``CameraFrame``, i.e.
  in length units (most commonly meters) on the camera focal plane.
* Using a ``CameraGeometry`` where pixel coordinates are expressed in ``TelescopeFrame``, i.e.
  in angular units (most commonly degree) on sky.

We propose to drop support for the first, to simplify code in multiple places and reduce
possibility for confusing the two similar variants of the image parameters.

The overhead of supporting both ``TelescopeFrame`` and ``CameraFrame`` representations
of the image parameters is quite significant, as it e.g. requires dealing with both
possible definitions in all Hillas-style dl2 reconstructors.


Advantages of Computation in TelescopeFrame
===========================================

Computing the image parameters in ``TelescopeFrame`` – angular units on the sky –
has the following advantages:

* Parameters are easier to compare across different telescope types.
* Pointing corrections can directly be applied in the conversion from ``CameraFrame``
  to ``TelescopeFrame`` and are then automatically included in the image parameters,
  which is much more straight forward than trying to correct image parameters that
  are affected to different degrees after they have been computed.
* Conversion from ``CameraFrame`` to ``TelescopeFrame`` will include any necessary
  special handling of the curved cameras of dual mirror telescopes.


Previous Discussions
====================

* Issue discussing the removal of the camera frame image parameters: `#2061 <https://github.com/cta-observatory/ctapipe/issues/2061>`_
* Original issue for introducing the computation of image parameters in telescope frame: `#1090 <https://github.com/cta-observatory/ctapipe/issues/1090>`_
* Pull Request implementing image parameters in telescope frame, also setting it as the default: `#1591 <https://github.com/cta-observatory/ctapipe/pull/1591>`_
* Adapting the reconstructors to also work with image parameters in telescope frame: `#1408 <https://github.com/cta-observatory/ctapipe/pull/1408>`_
