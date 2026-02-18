.. _irf:

**********************************************
Instrument Response Functions (`~ctapipe.irf`)
**********************************************

.. currentmodule:: ctapipe.irf

This module contains functionalities for generating instrument response functions.
The simulated events used for this have to be selected based on their associated "gammaness"
value and (optionally) their reconstructed angular offset from their point of origin.
The code for doing this can found in :ref:`cut_optimization` and is intended for use via the
`~ctapipe.tools.optimize_event_selection.EventSelectionOptimizer` tool.

The generation of the irf components themselves is implemented in :ref:`irfs` and is intended for
use via the `~ctapipe.tools.compute_irf.IrfTool` tool.
This tool can optionally also compute some common benchmarks, which are implemented in :ref:`benchmarks`.

The cut optimization as well as the calculations of the irf components and the benchmarks
are done using the `pyirf <https://pyirf.readthedocs.io/en/stable/>`_ package.

:ref:`binning`, :ref:`preprocessing`, and :ref:`spectra` contain helper functions and classes used by many of the
other components in this module.


Submodules
==========

.. toctree::
    :maxdepth: 1

    optimize
    irfs
    benchmarks
    binning
    spectra
    event_weighter


Reference/API
=============

.. automodapi:: ctapipe.irf
    :no-inheritance-diagram:
