ctapipe v0.29.0 (2026-01-14)
============================


Bug Fixes
---------

- Fix naming mismatch between ``classification`` and ``particle_type`` group
  in ``HDF5EventSource``.
  Rename the ``classification`` field from ``ReconstructedContainer`` to
  ``particle_type``. [`#2763 <https://github.com/cta-observatory/ctapipe/pull/2763>`__]

- Fix a unit handling bug in the toymodel that resulted in wrong image intensities. [`#2898 <https://github.com/cta-observatory/ctapipe/pull/2898>`__]

- Fix trigger tables when merging monitoring data of same event types. [`#2901 <https://github.com/cta-observatory/ctapipe/pull/2901>`__]

- Always fill available monitoring information in ``ctapipe-process``.
  Previously, monitoring was only filled if applying camera calibration was
  part of the configured processing steps. [`#2911 <https://github.com/cta-observatory/ctapipe/pull/2911>`__]

- Fixed bug that could in some cases lead to incorrect effective areas. In
  simulated files with no simulation distribution present, the fallback was to use
  the simulation configuration, but applying the number of reused showers was
  missing in that computation. The fallback computation is now corrected, and more
  checks were added to ensure consistency when both the simulation configuration
  and shower distribution are present. [`#2913 <https://github.com/cta-observatory/ctapipe/pull/2913>`__]


Data Model Changes
------------------

- Add more calibration and monitoring groups to HDF data format [`#2881 <https://github.com/cta-observatory/ctapipe/pull/2881>`__]


New Features
------------

- Add merge_strategy option to HDF5Merger component to control merging behavior. The HDF5Merger component supports the following strategies:
  - 'events-multiple-obs': allows merging event files (with and without monitoring data) from different observation blocks.
  - 'events-single-ob': for merging events in consecutive chunks of the same observation block.
  - 'monitoring-only': attaches horizontally monitoring data from the same observation block (requires monitoring=True).
  Allow monitoring data to be merged to the HDF5 event data files. [`#2901 <https://github.com/cta-observatory/ctapipe/pull/2901>`__]


Maintenance
-----------

- Add support for python 3.14, raise minimum supported python version to 3.12. [`#2855 <https://github.com/cta-observatory/ctapipe/pull/2855>`__]


ctapipe v0.28.0 (2025-11-13)
============================


API Changes
-----------

- ``CameraGeometry.neighbor_matrix_sparse`` is now a ``scipy.sparse.csr_array`` instead
  of a ``scipy.sparse.csr_matrix``, see https://docs.scipy.org/doc/scipy/reference/sparse.migration_to_sparray.html
  for more information. [`#2821 <https://github.com/cta-observatory/ctapipe/pull/2821>`__]


Bug Fixes
---------

- Fix ``ctapipe-optimize-event-selection`` raising an error for compressed (.fits.gz) output files. [`#2829 <https://github.com/cta-observatory/ctapipe/pull/2829>`__]

- Fix a bug in the ``chord_length`` function used by the  ``MuonIntensityFitter`` likelihood function
  that resulted in incorrect periodicity. The function returned 0 outside of the domain phi [-π, π]. [`#2873 <https://github.com/cta-observatory/ctapipe/pull/2873>`__]

- Fix a bug, where the ``ImageProcessor`` passes ``TelescopeMonitoringContainer``
  instead of ``CameraMonitoringContainer`` to the ``ImageCleaner`` which
  results in an AttributeError for the ``NSBImageCleaner``. [`#2875 <https://github.com/cta-observatory/ctapipe/pull/2875>`__]

- Add a check in ``lts_linear_regression`` to ensure that the input data has
  at least 2 valid points. Less could lead to segmentation faults. [`#2878 <https://github.com/cta-observatory/ctapipe/pull/2878>`__]

- Add default check in ``ImageQualityQuery`` to check for images
  with at least 2 surviving pixels after image cleaning. This prevents a segfault in the
  timing parameters if not enough points are available for the linear regression. [`#2877 <https://github.com/cta-observatory/ctapipe/pull/2877>`__]

New Features
------------

- Enhanced ``StatisticsAggregator`` classes to support time-based chunking in addition to event-based chunking.

  Refactored with component-based chunking system using ``BaseChunking``, ``SizeChunking``, and ``TimeChunking`` components.
  Statistics can now be aggregated over time intervals (seconds) rather than fixed event counts, enabling time-series analysis.
  Time-based chunking supports ``chunk_duration`` and configurable overlap via ``chunk_shift``.

  Breaking change: ``StatisticsAggregator`` now requires a ``chunking`` component to define chunking behavior. [`#2848 <https://github.com/cta-observatory/ctapipe/pull/2848>`__]

- Add a documentation page to show Hillas parameters displayed on camera image as schematic. [`#2868 <https://github.com/cta-observatory/ctapipe/pull/2868>`__]

- Add phi0 as argument to the chord length function. [`#2872 <https://github.com/cta-observatory/ctapipe/pull/2872>`__]

- Read all pixel statistics tables from the HDF5 file and instantiate the appropriate interpolators for each table in the MonitoringSource. [`#2888 <https://github.com/cta-observatory/ctapipe/pull/2888>`__]

- Allow subsequent aggregation of pre-aggregated time ranges (time_start, time_end) instead of just time column. [`#2889 <https://github.com/cta-observatory/ctapipe/pull/2889>`__]

Maintenance
-----------

- Updated colormaps in "Array Display" and "Use N-dimensional Histogram functionality and Interpolation" tutorials. [`#2883 <https://github.com/cta-observatory/ctapipe/pull/2883>`__]

- Add compatibility with eventio 2.0. [`#2857 <https://github.com/cta-observatory/ctapipe/pull/2857>`__]

ctapipe v0.27.1 (2025-10-22)
============================

Bug Fixes
---------

- Fix the normalization for muon analysis to account for different PixelShapes. [`#2865 <https://github.com/cta-observatory/ctapipe/pull/2865>`__]

- Fix the missing selection of the correct broken pixels for gain selected data
  in the image extractors. [`#2867 <https://github.com/cta-observatory/ctapipe/pull/2867>`__]

- Fix the sign of phi0, previously, it was giving the complementary angle of the impact point. [`#2869 <https://github.com/cta-observatory/ctapipe/pull/2869>`__]

- Allow ``peak_time=None`` in ``InvalidPixelHandler``. Some extractors may not
  provide a valid peak time (e.g. the ``VarianceExtractor``). [`#2867 <https://github.com/cta-observatory/ctapipe/pull/2867>`__]


ctapipe v0.27.0 (2025-10-10)
============================

This release contains a major breaking change and introduction of a new plugin interface,
both concerned with how monitoring data is handled in ctapipe.

The existing data structures have been reworked and a new ``MonitoringSource`` Interface
and plugin entry point are now defined.

See below for details and other smaller new features and bug fixes.


API Changes
-----------

- Update definition of the ``ctapipe.containers.EventType`` enum
  to match an updated definition in the CTAO R1 and DL0 data models.

  ``SKY_PEDESTAL=2`` is changed to the generic ``PEDESTAL``,
  which is expected from most cameras in real data.

  New types added are ``SKY_PEDESTAL=5`` and ``RANDOM_MONO=18``.

  These types were previously assigned at the subarray event level
  but in fact are per telescope event in the data model.
  A new field ``event_type`` was added to the ``TelescopeTriggerContainer``
  and will be filled by the ``EventSource``-implementations in ctapipe. [`#2769 <https://github.com/cta-observatory/ctapipe/pull/2769>`__]

- Refactor old containers (breaking changes) and implement calibpipe-datapipe interface using a new ``MonitoringSource()`` mechanism.

  * Remove containers ``DL1CameraCalibrationContainer``, ``FlatFieldContainer``, ``PedestalContainer``, ``PixelStatusContainer``, and ``WaveformCalibrationContainer``
  * Add containers ``CameraCalibrationContainer``, ``MonitoringTelescopeContainer``, ``ChunkStatisticsContainer``, ``PixelStatisticsContainer``, and ``ArrayPointingContainer``
  * Modify containers ``MonitoringContainer``, ``MonitoringCameraContainer``, ``PointingContainer``, ``TelescopePointingContainer``, and ``StatisticsContainer``,
  * Move pointing tree of ``ArrayEventContainer`` into the monitoring tree.
  * Remove calibration tree of ``ArrayEventContainer``. Information is retrieved from the ``monitoring.tel[tel_id].camera.coefficients`` tree.
  * Implement ``MonitoringSource()`` to handle different monitoring data sources
  * Implement the monitoring sources ``HDF5MonitoringSource`` that handles HDF5 file input
  * Implement Table I/O for camera calibration coefficients following the agreed HDF5 table naming and hierarchy
  * Remove ``ctapipe.calib.camera.flatfield`` and ``ctapipe.calib.camera.pedestal`` modules. Functionality is implemented in the ``ctapipe.monitoring`` module. [`#2802 <https://github.com/cta-observatory/ctapipe/pull/2802>`__]

- Remove the option ``DataWriter.write_index_tables`` along with its alias
  in ``ctapipe-process --write-index-tables``.
  The option seems to corrupt output files for large data sizes, is not used
  in ctapipe itself and was not tested. [`#2853 <https://github.com/cta-observatory/ctapipe/pull/2853>`__]


Bug Fixes
---------

- Fix image conversion to cartesian in case geometry units
  are not in meters. [`#2779 <https://github.com/cta-observatory/ctapipe/pull/2779>`__]

- Raise a clear error message in the ``DispReconstructor``
  in case of non-supported coordinate frames for the pointing
  directions instead of silently assuming AltAz. [`#2785 <https://github.com/cta-observatory/ctapipe/pull/2785>`__]

- Avoid astropy warnings in ``ctapipe.utils.astro.get_bright_stars``. [`#2801 <https://github.com/cta-observatory/ctapipe/pull/2801>`__]

- In the muon intensity fitting class filters out non-physical solutions for chord vs phi function. [`#2803 <https://github.com/cta-observatory/ctapipe/pull/2803>`__]

- Fix HDF5EventSource not properly reading telescope-wise DL2 fields written by ``ctapipe-apply-models``. [`#2824 <https://github.com/cta-observatory/ctapipe/pull/2824>`__]

- Fix misleading error messages if a non-existing file is passed
  via the command-line to a Tool that has another configuration option
  that uses ``Path(default_value=None, allow_none=False)``.

  This pattern is now deprecated as it results in these hard to understand
  error messages due to behavior deep in traitlets.

  A warning will be raised if traitlets are defined like this and
  all ctapipe tools have been adapted to explicitly check
  for None if a ``Path`` is required. [`#2826 <https://github.com/cta-observatory/ctapipe/pull/2826>`__]

- Return the number of events used to compute the statistics in SigmaClippingAggregator. [`#2841 <https://github.com/cta-observatory/ctapipe/pull/2841>`__]

- Fixed bug in `ctapipe.core.Tool.get_current_config` where ``Tools`` containing lists of
  ``Components`` did not get registered, and thus their configs did not end up in the
  provenance log. [`#2851 <https://github.com/cta-observatory/ctapipe/pull/2851>`__]


New Features
------------

- Add short-form alternatives to the cli aliases of ``ctapipe-optimize-event-selection``. [`#2770 <https://github.com/cta-observatory/ctapipe/pull/2770>`__]

- Use the HDF5Merger to add the observation block and scheduling block info
  to the DL1 files produced by ctapipe-calculate-pixel-statistics tool. [`#2776 <https://github.com/cta-observatory/ctapipe/pull/2776>`__]

- Enable ``ctapipe-merge`` to be used on camera calibration data and pixel statistics aggregation. [`#2787 <https://github.com/cta-observatory/ctapipe/pull/2787>`__]

- Generalise the DL2 table processing implemented in IRF tools in order to be used by other usecases that ingest, filter and merge DL2 tables.
  The module was moved from IRF to IO. [`#2791 <https://github.com/cta-observatory/ctapipe/pull/2791>`__]

- Add new cli tool ``ctapipe-store-astropy-cache`` to download and store
  astropy data needed for coordinate transformations. [`#2823 <https://github.com/cta-observatory/ctapipe/pull/2823>`__]

- ``ctapipe.io.read_table`` and ``ctapipe.io.write_table`` now support custom column metadata. [`#2824 <https://github.com/cta-observatory/ctapipe/pull/2824>`__]

- Add common names of stars to table output of ``ctapipe.utils.astro.get_bright_stars``. [`#2827 <https://github.com/cta-observatory/ctapipe/pull/2827>`__]

- Add support for disabling numba caching via ``CTAPIPE_DISABLE_NUMBA_CACHE=1`` environment variable.
  See https://github.com/numba/numba/issues/10128 for why this might be needed. [`#2836 <https://github.com/cta-observatory/ctapipe/pull/2836>`__]


Maintenance
-----------

- - fixed confusing (and incorrect) debug log message in download.py
  - removed all references to ``ctapipe_resources`` in the code and docs, since
    that module is no longer supported. Updated documentation to mention the
    remote test data server when applicable. [`#2782 <https://github.com/cta-observatory/ctapipe/pull/2782>`__]

- Updated the developer environment in environment.yml to be in sync with what is in pyproject.toml. [`#2849 <https://github.com/cta-observatory/ctapipe/pull/2849>`__]



ctapipe v0.26.0 (2025-06-05)
============================


API Changes
-----------

- The algorithms for muon ring fitting have been improved

  - Upgrade the taubin_circle_fit with weights to avoid bias toward rings with larger radii
  - Add the optional initial parameters : ring center and radius.
  - Add a combined ring fitting method (kundu_chaudhuri_taubin).
  - This new combined method is set to be the default method.
  - Updated the test with an additional fitting method, each method is tested separately.
  - Realistic muon ring simulation for all camera types implemented in the test with toy model.
  - Include the parameter fit errors in the tolerance metric.
  - Add error fields to the muon container.
  - The API for MuonFitter has been modified.
  - Minor documentation updates. [`#2736 <https://github.com/cta-observatory/ctapipe/pull/2736>`__]


Bug Fixes
---------

- Fix ``CameraDisplay`` throwing an error when used in a matplotlib
  subfigure. [`#2762 <https://github.com/cta-observatory/ctapipe/pull/2762>`__]


Data Model Changes
------------------

- Add error fields to the muon container.


New Features
------------

- Add option ``single_ob`` to the ``HDF5Merger`` to support merging
  chunks of the same observation block into the same. [`#2436 <https://github.com/cta-observatory/ctapipe/pull/2436>`__]

- Add option to skip the simtel R1 calibration [`#2753 <https://github.com/cta-observatory/ctapipe/pull/2753>`__]

- Add support for chunked reading over subsets of a file
  by passing ``start=`` and/or ``stop=`` to the
  ``TableLoader.read_..._chunked`` methods. [`#2757 <https://github.com/cta-observatory/ctapipe/pull/2757>`__]


ctapipe 0.25.1 (2025-04-28)
===========================

Bug Fixes
---------

- Fallback to "Unknown User" in case no username is available
  from the system for provenance. [`#2741 <https://github.com/cta-observatory/ctapipe/pull/2741>`__]

- Fix ``HDF5EventSource`` raising an exception in case of missing
  subarray pointing in the input file. [`#2744 <https://github.com/cta-observatory/ctapipe/pull/2744>`__]


ctapipe 0.25.0 (2025-04-16)
===========================

Bug Fixes
---------

- Fix units (absence of) for skewness and excess kurtosis in muon analysis [`#2732 <https://github.com/cta-observatory/ctapipe/pull/2732>`__]

- Improve heuristic in ``SimTelEventSource`` for when true Cherenkov
  images are expected to be present in the input file.
  This fixes missing true images in output files when the first event of a run
  had missing true images. [`#2735 <https://github.com/cta-observatory/ctapipe/pull/2735>`__]

- Fix inverted sign of ``DispReconstructor`` prediction. [`#2738 <https://github.com/cta-observatory/ctapipe/pull/2738>`__]


Data Model Changes
------------------
- Two new fields in the Hillas parameters:
  1. psi_uncertainty (uncertainty on the psi angle of the image)
  2. transverse_cog_uncertainty (uncertainty on the center of gravity along the transverse axis of the image) [`#2629 <https://github.com/cta-observatory/ctapipe/pull/2629>`__]


New Features
------------

- Add two new fields in the Hillas parameters computations:
  1. psi_uncertainty (uncertainty on the psi angle of the image)
  2. transverse_cog_uncertainty (uncertainty on the center of gravity along the transverse axis of the image) [`#2629 <https://github.com/cta-observatory/ctapipe/pull/2629>`__]


ctapipe v0.24.0 (2025-03-31)
============================


Bug Fixes
---------

- Fix ``SubarrayDescription.info()`` in cases
  of different ``TelescopeDescription`` with the same string representation. [`#2673 <https://github.com/cta-observatory/ctapipe/pull/2673>`__]

- Make sure that the configuration stored in the provenance
  information actually contains the full configuration
  and can be used to re-create the application. [`#2688 <https://github.com/cta-observatory/ctapipe/pull/2688>`__]

- Fix the function grouping telescope ids into ranges for
  the case of unsigned integer telescope ids. [`#2692 <https://github.com/cta-observatory/ctapipe/pull/2692>`__]

- Fix ``Tool`` not exposing all options defined by ``traitlets.Application``
  by default.

  Fix ``--show-config`` and ``--show-config-json`` by not running ``setup`` and
  ``finish`` steps in case those options are given. [`#2703 <https://github.com/cta-observatory/ctapipe/pull/2703>`__]

- Fix processing gain-selected dl1 data with the ``PixelStatisticsCalculatorTool``. [`#2715 <https://github.com/cta-observatory/ctapipe/pull/2715>`__]


Data Model Changes
------------------
- Change definition of DVR bits in PixelStatus to new definition in DL0 data model. [`#2725 <https://github.com/cta-observatory/ctapipe/pull/2725>`__]

- Change format in which event timestamps are written to HDF5 files.
  Instead of a single float64 MJD value which has ~µs precision,
  the CTAO high precision time format is used now.
  This stores the timestamp as two uint32 values: seconds
  and quarter nanoseconds since ``1970-01-01T00:00:00.0 TAI``.

  This only affects the storage format and the precision,
  the in-memory API is unchanged as it relies on ``astropy.time.Time``
  and values are converted when reading/writing from/to HDF5. [`#2707 <https://github.com/cta-observatory/ctapipe/pull/2707>`__]


New Features
------------

- Add a ``ctapipe-optimize-event-selection`` tool to produce cut-selection files,
  based on a gamma, and optionally a proton and an electron DL2 file.
  Two components for calculating G/H and optionally theta cuts are added:
  ``PercentileCuts`` keeps a certain percentage of gamma events in each bin and
  ``PointSourceSensitivityOptimizer`` optimizes G/H cuts for maximum point source sensitivity and
  optionally calculates percentile theta cuts.

- Add a ``ctapipe-compute-irf`` tool to produce irfs given a cut-selection file, a gamma,
  and optionally a proton, and an electron DL2 input file.
  Given only a gamma file, the energy dispersion, effective area, and point spread function are calculated.
  Optionally, the bias and resolution of the energy reconstruction and the angular resolution can be calculated
  and saved in a separate output file.
  If a proton or a proton and an electron file is also given, a background model can be calculated,
  as well as the point source sensitivity.

  Irfs can be calculated with and without applying a direction cut.
  Only radially symmetric parameterizations of the irf components are implemented so far. [`#2473 <https://github.com/cta-observatory/ctapipe/pull/2473>`__]

- Add a generic stats-calculation tool utilizing the PixelStatisticsCalculator. [`#2628 <https://github.com/cta-observatory/ctapipe/pull/2628>`__]

- Add ChunkInterpolator to ctapipe.monitoring.interpolation as a tool to select data from chunks. The planned use for this is to select calibration data. [`#2634 <https://github.com/cta-observatory/ctapipe/pull/2634>`__]

- Adds psf models to ``ctapipe.instrument.optics`` with the parent class ``PSFModel`` and a psf model based on pure coma aberration called ``ComaPSFModel``
  - The function ``PSFModel.pdf`` gives the value of the PSF in a given location [`#2643 <https://github.com/cta-observatory/ctapipe/pull/2643>`__]

- Add new fields to the MuonParametersContainer.

  Implement the computation of the new features listed below
  and refactor the code using the existing containers.

  Added fields:
  - ring_intensity
  - intensity_outside_ring
  - n_pixels_in_ring
  - mean_intensity_outside_ring
  - radial_std_dev
  - skeweness
  - excess_kurtosis [`#2670 <https://github.com/cta-observatory/ctapipe/pull/2670>`__]

- Store also the SubarrayDescription in the camera monitoring data produced by the stats tool [`#2696 <https://github.com/cta-observatory/ctapipe/pull/2696>`__]

- Add a method ``read_scheduling_blocks`` in the ``TableLoader`` class to read scheduling block information. [`#2709 <https://github.com/cta-observatory/ctapipe/pull/2709>`__]

- Adding a ``meta_convention`` option to ``SubarrayDescription.to_table()`` method to choose
  between hdf and fits conventions mainly for the reference location. Default is set to 'hdf'.
  In Addition ``TableLoader`` does not join metadata of instrument table when
  ``instrument=true``. [`#2722 <https://github.com/cta-observatory/ctapipe/pull/2722>`__]


Maintenance
-----------

- Add a tutorial to the docs on how to use the machine learning tools
  to process dl1b (image parameters) files to dl2 for a mono analysis.

  Perform minor updates to other parts of the api docs and the user-guide. [`#2691 <https://github.com/cta-observatory/ctapipe/pull/2691>`__]


ctapipe v0.23.2 (2025-01-21)
============================

Bug Fixes
---------

- Fill ``ctapipe.containers.SimulatedCameraContainer.true_image_sum`` in
  ``HDF5EventSource``. Always returned default value of -1 before the fix. [`#2680 <https://github.com/cta-observatory/ctapipe/pull/2680>`__]

Maintenance
-----------

- Add compatibility with numpy>=2.1. [`#2682 <https://github.com/cta-observatory/ctapipe/pull/2682>`__]


ctapipe v0.23.1 (2024-12-04)
============================

Bug Fixes
---------

- Fix ``<reconstruction_property>_uncert`` calculations in ``ctapipe.reco.StereoMeanCombiner``.
  Add helper functions for vectorized numpy calculations as new ``ctapipe.reco.telescope_event_handling`` module. [`#2658 <https://github.com/cta-observatory/ctapipe/pull/2658>`__]

- Fix error in ``ctapipe-process`` when in the middle of a simtel file
  that has true images available, a telescope event is missing the true image.
  This can happen rarely in case a telescope triggered on pure NSB or
  is oversaturated to the point where the true pe didn't fit into memory constraints.

  The error was due to the ``DataWriter`` trying to write a ``None`` into an
  already setup table for the true images.

  The ``SimTelEventSource`` will now create an invalid true image filled with ``-1``
  for such events. [`#2659 <https://github.com/cta-observatory/ctapipe/pull/2659>`__]

- In ``SimTelEventSource``, ignore telescope events that did not take part in the stereo event trigger.
  This happens rarely in Prod6 files in conjunction with the random mono trigger system.

- Fix the order in which ``Tool`` runs final operations to fix an issue
  of provenance not being correctly recorded. [`#2662 <https://github.com/cta-observatory/ctapipe/pull/2662>`__]

- Fix data type of ``tel_id`` in the output of ``SubarrayDescription.to_table``

- Fixed a bug where if a configuration file with unknown file extension was passed
  to a tool, e.g. ``--config myconf.conf`` instead of ``--config myconf.yaml``, it
  was silently ignored, despite an info log saying "Loading config file
  myconf.conf". Configuration files must now have one of the following extensions
  to be recognized: yml, yaml, toml, json, py. If not a ``ToolConfigurationError``
  is raised. [`#2666 <https://github.com/cta-observatory/ctapipe/pull/2666>`__]

Maintenance
-----------

- Add support for astropy 7.0. [`#2639 <https://github.com/cta-observatory/ctapipe/pull/2639>`__]
- Change data server for test datasets from in2p3 to DESY hosted server. [`#2664 <https://github.com/cta-observatory/ctapipe/pull/2664>`__]

ctapipe v0.23.0 (2024-11-18)
============================


API Changes
-----------

- Add possibility to use ``HIPPARCOS`` catalog to get star positions

  - add catalogs enumerator to ``ctapipe.utils.astro``
  - update ``get_bright_stars`` in ``ctapipe.utils.astro`` to allow catalog selection
  - ensure application of proper motion
  - add possibility to select stars on proximity to a given position and on apparent magnitude
  - Bundle star catalogs in the python package
  - API change: ``ctapipe.utils.astro.get_bright_stars`` now requires a timestamp to apply proper motion.
    In the prevision revision, the time argument was missing and the proper motion was not applied. [`#2625 <https://github.com/cta-observatory/ctapipe/pull/2625>`__]

- Move the simulated shower distribution from something
  that was specific to ``SimTelEventSource`` to a general interface
  of ``EventSource``. Implement the new interface in both ``SimTelEventSource``
  and ``HDF5EventSource`` and adapt writing of this information in ``DataWriter``.

  This makes sure that the ``SimulatedShowerDistribution`` information is always
  included, also when running ``ctapipe-process`` consecutively. [`#2633 <https://github.com/cta-observatory/ctapipe/pull/2633>`__]

- The following dependencies are now optional:

  * eventio, used for ``ctapipe.io.SimTelEventSource``.
  * matplotlib, used ``ctapipe.visualization.CameraDisplay``, ``ctapipe.visualization.ArrayDisplay``,
    and most default visualization tasks, e.g. ``.peek()`` methods.
  * iminuit, used for the ``ctapipe.image.muon`` and ``ctapipe.reco.impact`` fitting routines.
  * bokeh, for ``ctapipe.visualiation.bokeh``

  Code that needs these dependencies will now raise ``ctapipe.exceptions.OptionalDependencyMissing``
  in case such functionality is used and the dependency in question is not installed.

  These packages will now longer be installed by default when using e.g. ``pip install ctapipe``.

  If you want to install ctapipe with all optional dependencies included, do ``pip install "ctapipe[all]"``.

  For ``conda``, we will publish to packages: ``ctapipe`` will include all optional dependencies
  and a new ``ctapipe-base`` package will only include the required dependencies.

  [`#2641 <https://github.com/cta-observatory/ctapipe/pull/2641>`__]

- * Add possibility to directly pass the reference metadata to
    ``Provenance.add_input_file``.
  * Remove the call to ``Provenace.add_input_file`` from the
    ``EventSource`` base class.
  * Add the proper calls to ``Provenance.add_input_file`` in
    ``HDF5EventSource`` (providing the metadata) and
    ``SimTelEventSource`` (not providing metadata yet, but avoiding a warning)
  * Plugin implementations of ``EventSource`` should make sure they
    register their input files using ``Provenance.add_input_file``, preferably
    providing also the reference metadata. [`#2648 <https://github.com/cta-observatory/ctapipe/pull/2648>`__]


Bug Fixes
---------

- Fix ensuring that hdf5 files created with older versions of ctapipe, e.g.
  the public dataset created with 0.17 can be read by ctapipe-process.
  These files contain pointing information at a different location and
  are missing the subarray reference location, which was introduced
  in later versions of ctapipe. A dummy location (lon=0, lat=0)
  is used for these now, the same value is already used for simtel files
  lacking this information. [`#2627 <https://github.com/cta-observatory/ctapipe/pull/2627>`__]

New Features
------------

- Add option ``override_obs_id`` to ``SimTelEventSource`` which allows
  assigning new, unique ``obs_ids`` in case productions reuse CORSIKA run
  numbers. [`#2411 <https://github.com/cta-observatory/ctapipe/pull/2411>`__]

- Add calibration calculators which aggregates statistics, detects outliers, handles faulty data chunks. [`#2609 <https://github.com/cta-observatory/ctapipe/pull/2609>`__]

- Update ``CameraCalibrator`` in ``ctapipe.calib.camera.calibrator`` allowing it to correctly calibrate variance images generated with the ``VarianceExtractor``.
    - If the ``VarianceExtractor`` is used for the ``CameraCalibrator`` the element-wise square of the relative and absolute gain calibration factors are applied to the image;
    - For other image extractors the plain factors are still applied.
    - The ``VarianceExtractor`` provides no peak time and the calibrator will skip shifting the peak time for extractors like the ``VarianceExtractor`` that similarly do not provide a peak time [`#2636 <https://github.com/cta-observatory/ctapipe/pull/2636>`__]

- Add ``__repr__`` methods to all objects that were missing
  them in ``ctapipe.io.metadata``, update the existing ones
  for consistency. [`#2650 <https://github.com/cta-observatory/ctapipe/pull/2650>`__]


ctapipe v0.22.0 (2024-09-12)
============================

API Changes
-----------

- The ``PointingInterpolator`` was moved from ``ctapipe.io`` to ``ctapipe.monitoring``. [`#2615 <https://github.com/cta-observatory/ctapipe/pull/2615>`__]


Bug Fixes
---------

- Fix a redundant error message in ``Tool`` caused by normal ``SystemExit(0)`` [`#2575 <https://github.com/cta-observatory/ctapipe/pull/2575>`__]

- Fix error message for non-existent config files. [`#2591 <https://github.com/cta-observatory/ctapipe/pull/2591>`__]


New Features
------------

- ctapipe is now compatible with numpy 2.0. [`#2580 <https://github.com/cta-observatory/ctapipe/pull/2580>`__]
  Note: not all new behaviour of numpy 2.0 is followed, as the core dependency ``numba`` does not yet implement
  all changes from numpy 2.0. See `the numba announcement for more detail <https://numba.discourse.group/t/communicating-numpy-2-0-changes-to-numba-users/2457>`_.

- Add lstchains image cleaning procedure including its pedestal cleaning method. [`#2541 <https://github.com/cta-observatory/ctapipe/pull/2541>`__]

- A new ImageExtractor called ``VarianceExtractor`` was added
  An Enum class was added to containers.py that is used in the metadata of the VarianceExtractor output [`#2543 <https://github.com/cta-observatory/ctapipe/pull/2543>`__]

- Add API to extract the statistics from a sequence of images. [`#2554 <https://github.com/cta-observatory/ctapipe/pull/2554>`__]

- The provenance system now records the reference metadata
  of input and output files, if available. [`#2598 <https://github.com/cta-observatory/ctapipe/pull/2598>`__]

- Add Interpolator class to generalize the PointingInterpolator in the monitoring collection. [`#2600 <https://github.com/cta-observatory/ctapipe/pull/2600>`__]

- Add outlier detection components to identify faulty pixels. [`#2604 <https://github.com/cta-observatory/ctapipe/pull/2604>`__]

- The ``ctapipe-merge`` tool now checks for duplicated input files and
  raises an error in that case.

  The ``HDF5Merger`` class, and thus also the ``ctapipe-merge`` tool,
  now checks for duplicated obs_ids during merging, to prevent
  invalid output files. [`#2611 <https://github.com/cta-observatory/ctapipe/pull/2611>`__]

- The ``Instrument.site`` metadata item now accepts any string,
  not just a pre-defined list of sites. [`#2616 <https://github.com/cta-observatory/ctapipe/pull/2616>`__]

Refactoring and Optimization
----------------------------

- Update exception handling in tools

  - Add a possibility to handle custom exception in ``Tool.run()``
    with the preservation of the exit code. [`#2594 <https://github.com/cta-observatory/ctapipe/pull/2594>`__]


ctapipe v0.21.2 (2024-06-26)
============================

A small bugfix release to add support for scipy 1.14.

Also contains a small new feature regarding exit code handling in ``Tool``.

Bug Fixes
---------

- Replace deprecated usage of scipy sparse matrices, adds support for scipy 1.14. [`#2569 <https://github.com/cta-observatory/ctapipe/pull/2569>`__]


New Features
------------

- Add ``SystemExit`` handling at the ``ctapipe.core.Tool`` level

  If a ``SystemExit`` with a custom error code is generated during the tool execution,
  the tool will be terminated gracefully and the error code will be preserved and propagated.

  The ``Activity`` statuses have been updated to ``["running", "success", "interrupted", "error"]``.
  The ``"running"`` status is assigned at init. [`#2566 <https://github.com/cta-observatory/ctapipe/pull/2566>`__]


Maintenance
-----------

- made plugin detection less verbose in logs: DEBUG level used instead of INFO [`#2560 <https://github.com/cta-observatory/ctapipe/pull/2560>`__]


ctapipe v0.21.1 (2024-05-15)
============================

This is a small bug fix and maintenance release for 0.21.0.


Bug Fixes
---------

- Fix ``SoftwareTrigger`` not correctly handling different telescope
  types that have the same string representation, e.g. the four LSTs
  in prod6 files.

  Telescopes that have the same string representation now always are treated
  as one group in ``SoftwareTrigger``. [`#2552 <https://github.com/cta-observatory/ctapipe/pull/2552>`__]


Maintenance
-----------

- A number of simple code cleanups in the ImPACT reconstructor code. [`#2551 <https://github.com/cta-observatory/ctapipe/pull/2551>`__]


ctapipe v0.21.0 (2024-04-25)
============================


API Changes
-----------

- ``reference_location`` is now a required argument for  ``SubarrayDescription``
  [`#2402 <https://github.com/cta-observatory/ctapipe/pull/2402>`__]

- ``CameraGeometry.position_to_pix_index`` will now return the minimum integer value for invalid
  pixel coordinates instead of -1 due to the danger of using -1 as an index in python accessing
  the last element of a data array for invalid pixels.
  The function will now also no longer raise an error if the arguments are empty arrays and instead
  just return an empty index array.
  The function will also no longer log a warning in case of coordinates that do not match a camera pixel.
  The function is very low-level and if not finding a pixel at the tested position warrants a warning or
  is expected will depend on the calling code. [`#2397 <https://github.com/cta-observatory/ctapipe/pull/2397>`__]

- Change the definition of the ``leakage_pixels_width_{1,2}`` image features
  to give the ratio of pixels at the border to the pixels after cleaning
  instead of to the total number of pixels of the camera. [`#2432 <https://github.com/cta-observatory/ctapipe/pull/2432>`__]

- Change how the ``DataWriter`` writes pointing information.
  Before, each unique pointing position was written in a table
  with the event time as index column into ``dl1/monitoring/telescope/pointing``.

  This has two issues: For observed data, each pointing will be unique
  in horizontal coordinates due to tracking a fixed ICRS coordinate.
  Resulting in a pointing position written for each event, although the
  resolution of the monitoring is much lower.
  For simulated events, the event time is the timestamp of the simulation
  and pointing is fixed in ``AltAz``.
  ``ctapipe`` was using the closest point in time for simulated events when
  reading data back in, however, this is problematic in case of many
  simulation runs processed in parallel.

  We now store the first received pointing information
  in the ``configuration/telescope/pointing`` table per obs id,
  only for simulation events. [`#2438 <https://github.com/cta-observatory/ctapipe/pull/2438>`__]

- Replace ``n_signal`` and ``n_background`` options in ``ctapipe-train-particle-classifier``
  with ``n_events`` and ``signal_fraction``, where ``signal_fraction`` = n_signal / (n_signal + n_background). [`#2465 <https://github.com/cta-observatory/ctapipe/pull/2465>`__]

- Move the ``TableLoader`` options from being traitlets to
  each ``read_...`` method allowing to load different data with the
  same TableLoader-Instance.

  In addition the default values for the options have changed. [`#2482 <https://github.com/cta-observatory/ctapipe/pull/2482>`__]

- Adding monitoring: MonitoringCameraContainer as keyword argument to
  the ``ImageCleaner`` API so cleaning algorithms can now access
  relevant information for methods that e.g. require monitoring information. [`#2511 <https://github.com/cta-observatory/ctapipe/pull/2511>`__]

- Unified the options for DataWriter and the data level names:

  +-------------------------+--------------------------+
  | Old                     | New                      |
  +=========================+==========================+
  | ``write_raw_waveforms`` | ``write_r0_waveforms``   |
  +-------------------------+--------------------------+
  | ``write_waveforms``     | ``write_r1_waveforms``   |
  +-------------------------+--------------------------+
  | ``write_images``        | ``write_dl1_images``     |
  +-------------------------+--------------------------+
  | ``write_parameters``    | ``write_dl1_parameters`` |
  +-------------------------+--------------------------+
  | ``write_showers``       | ``write_dl2``            |
  +-------------------------+--------------------------+

  This changes requires that existing configuration files are updated
  if they use these parameters [`#2520 <https://github.com/cta-observatory/ctapipe/pull/2520>`__]


Bug Fixes
---------

- Ensure that ``SubarrayDescription.reference_location`` is always generated by
  ```SimTelEventSource``, even if the metadata is missing. In that case, construct a
  dummy location with the correct observatory height and latitude and longitude
  equal to zero ("Null Island").

- Fixed the definition of ``h_max``, which was both inconsistent between
  `~ctapipe.reco.HillasReconstructor` and `~ctapipe.reco.HillasIntersection`
  implementations, and was also incorrect since it was measured from the
  observatory elevation rather than from sea level.

  The value of ``h_max`` is now defined as the height above sea level of the
  shower-max point (in meters), not the distance to that point. Therefore it is
  not corrected for the zenith angle of the shower. This is consistent with the
  options currently used for *CORSIKA*, where the *SLANT* option is set to false,
  meaning heights are actual heights not distances from the impact point, and
  ``x_max`` is a *depth*, not a *slant depth*. Note that this definition may be
  inconsistent with other observatories where slant-depths are used, and also note
  that the slant depth or distance to shower max are the more useful quantities
  for shower physics. [`#2403 <https://github.com/cta-observatory/ctapipe/pull/2403>`__]

- Add the example config for ctapipe-train-disp-reconstructor
  to the list of configs generated by ctapipe-quickstart. [`#2414 <https://github.com/cta-observatory/ctapipe/pull/2414>`__]

- Do not use a hidden attribute of ``SKLearnReconstructor`` in ``ctapipe-apply-models``. [`#2418 <https://github.com/cta-observatory/ctapipe/pull/2418>`__]

- Add docstring for ``ctapipe-train-disp-reconstructor``. [`#2420 <https://github.com/cta-observatory/ctapipe/pull/2420>`__]

- Remove warnings about missing R1 or DL0 data when using the CameraCalibrator.
  These were previously emitted directly as python warnings and did not use the
  component logging system, which they now do.
  As we do not actually expect R1 to be present it was also moved down to
  debug level. [`#2421 <https://github.com/cta-observatory/ctapipe/pull/2421>`__]

- Check that the array pointing is given in horizontal coordinates
  before training a ``DispReconstructor``. [`#2431 <https://github.com/cta-observatory/ctapipe/pull/2431>`__]

- Fix additional, unwanted columns being written into disp prediction output. [`#2440 <https://github.com/cta-observatory/ctapipe/pull/2440>`__]

- Properly transform pixel coordinates between ``CameraFrame``
  and ``TelescopeFrame`` in ``MuonIntensityFitter`` taking.
  Before, ``MuonIntensityFitter`` always used the equivalent focal
  length for transformations, now it is using the focal length
  attached to the ``CameraGeometry``, thus respecting the
  ``focal_length_choice`` options of the event sources. [`#2464 <https://github.com/cta-observatory/ctapipe/pull/2464>`__]

- Fix colored logging in case of custom log levels being defined. [`#2505 <https://github.com/cta-observatory/ctapipe/pull/2505>`__]

- Fix a possible out-of-bounds array access in the FlashCamExtractor. [`#2544 <https://github.com/cta-observatory/ctapipe/pull/2544>`__]


Data Model Changes
------------------

- Remove redundant ``is_valid`` field in ``DispContainer`` and rename the remaining field.
  Use the same prefix for both containers filled by ``DispReconstructor``.

  Fix default name of ``DispReconstructor`` target column.

  Let ``HDF5EventSource`` load ``DispContainer``. [`#2443 <https://github.com/cta-observatory/ctapipe/pull/2443>`__]

- Change R1- and DL0-waveforms datamodel shape from (n_pixels, n_samples)
  to be always (n_channels, n_pixels, n_samples). ``HDF5EventSource`` was adjusted
  accordingly to support also older datamodel versions.

  Re-introduce also the possibility of running ``ImageExtractor``\s on data
  consisting of multiple gain channels. [`#2529 <https://github.com/cta-observatory/ctapipe/pull/2529>`__]


New Features
------------

- Large updates to the Image Pixel-wise fit for Atmospheric Cherenkov Telescopes reconstruction method (https://doi.org/10.48550/arXiv.1403.2993)

  * ImPACT - General code clean up and optimisation. Now updated to work similarly to other reconstructors using the standardised interface, such that it can be used ctapipe-process. Significant improvements to tests too
  * ImPACT_utilities - Created new file to hold general usage functions, numba used in some areas for speedup
  * template_network_interpolator - Now works with templates with different zenith and azimuth angles
  * unstructured_interpolator - Significant speed improvements
  * pixel_likelihood - Constants added back to neg_log_likelihood_approx, these are quite important to obtaining a well normalised goodness of fit.
  * hillas_intersection - Fixed bug in core position being incorrectly calculated, fixed tests too [`#2305 <https://github.com/cta-observatory/ctapipe/pull/2305>`__]

- Allow passing the matplotlib axes to the ``SubarrayDescription.peek`` function,
  fix warnings in case of layout engine being already defined. [`#2369 <https://github.com/cta-observatory/ctapipe/pull/2369>`__]

- Add support for interpolating a monitoring pointing table
  in ``TableLoader``. The corresponding table is not yet written by ``ctapipe``,
  but can be written by external tools.
  This is to enable analysis of real observations, where the pointing changes over time in
  alt/az. [`#2409 <https://github.com/cta-observatory/ctapipe/pull/2409>`__]

- Implement the overburden-to height a.s.l. transformation function in the atmosphere module
  and test that round-trip returns original value. [`#2422 <https://github.com/cta-observatory/ctapipe/pull/2422>`__]

- In case no configuration is found for a telescope in ``TelescopeParameter``,
  it is now checked whether the telescope exists at all to provide a better
  error message. [`#2429 <https://github.com/cta-observatory/ctapipe/pull/2429>`__]

- Allow setting n_jobs on the command line for the
  train_* and apply_models tools using a new ``n_jobs`` flag.
  This temporarily overwrites any settings in the (model) config(s). [`#2430 <https://github.com/cta-observatory/ctapipe/pull/2430>`__]

- Add support for using ``str`` and ``Path`` objects as input
  to ``ctapipe.io.get_hdf5_datalevels``. [`#2451 <https://github.com/cta-observatory/ctapipe/pull/2451>`__]

- The recommended citation for ctapipe has been updated to the ICRC 2023 proceeding,
  please update. [`#2470 <https://github.com/cta-observatory/ctapipe/pull/2470>`__]

- Support astropy 6.0. [`#2475 <https://github.com/cta-observatory/ctapipe/pull/2475>`__]

- The ``DispReconstructor`` now computes a score for how certain the prediction of the disp sign is. [`#2479 <https://github.com/cta-observatory/ctapipe/pull/2479>`__]

- Also load the new fixed pointing information in ``TableLoader``.

  Add option ``keep_order`` to ``ctapipe.io.astropy_helpers.join_allow_empty``
  that will keep the original order of rows when performing left or right joins. [`#2481 <https://github.com/cta-observatory/ctapipe/pull/2481>`__]

- Add an ``AstroQuantity`` trait which can hold any ``astropy.units.Quantity``. [`#2524 <https://github.com/cta-observatory/ctapipe/pull/2524>`__]

- Add function ``ctapipe.coordinates.get_point_on_shower_axis``
  that computes a point on the shower axis in alt/az as seen
  from a telescope. [`#2537 <https://github.com/cta-observatory/ctapipe/pull/2537>`__]

- Update bokeh dependency to version 3.x. [`#2549 <https://github.com/cta-observatory/ctapipe/pull/2549>`__]


Maintenance
-----------

- The CI system now reports to the CTA SonarQube instance for code quality tracking [`#2214 <https://github.com/cta-observatory/ctapipe/pull/2214>`__]

- Updated some numpy calls to not use deprecated functions. [`#2406 <https://github.com/cta-observatory/ctapipe/pull/2406>`__]

- The ``ctapipe`` source code repository now uses the ``src/``-based layout.
  This fixes the editable installation of ctapipe. [`#2459 <https://github.com/cta-observatory/ctapipe/pull/2459>`__]

- Fix headings in docs. Change occurrences of ``API Reference`` to ``Reference/API`` for consistency.
  Change capitalization of some headings for consistency. [`#2474 <https://github.com/cta-observatory/ctapipe/pull/2474>`__]

- The ``from_name`` methods of instrument description classes now raise a warning
  that it is better to access instrument information via a ``SubarrayDescription``.

  Also improve documentation in instrument module to explain when not to use the
  various ``from_name()`` methods. These are provided for the case when no event
  data is available, e.g. for unit testing or demos, but do not guarantee that the
  returned instrument information corresponds to a particular set of event data. [`#2485 <https://github.com/cta-observatory/ctapipe/pull/2485>`__]

- Support and test on python 3.12. [`#2486 <https://github.com/cta-observatory/ctapipe/pull/2486>`__]

- Drop support for python 3.9. [`#2526 <https://github.com/cta-observatory/ctapipe/pull/2526>`__]


Refactoring and Optimization
----------------------------

- Load data and apply event and column selection in chunks in ``ctapipe-train-*``
  before merging afterwards.
  This reduces memory usage. [`#2423 <https://github.com/cta-observatory/ctapipe/pull/2423>`__]

- Make default ML config files more readable and add comments. [`#2455 <https://github.com/cta-observatory/ctapipe/pull/2455>`__]

- Update and add missing docstrings related to the ML functionalities. [`#2456 <https://github.com/cta-observatory/ctapipe/pull/2456>`__]

- Add ``true_impact_distance`` to the output of ``CrossValidator``. [`#2468 <https://github.com/cta-observatory/ctapipe/pull/2468>`__]

- Add ``cache=True`` to some numba-compiled functions which were missing it. [`#2477 <https://github.com/cta-observatory/ctapipe/pull/2477>`__]

- Write cross validation results for each model out immediately after validation to free up memory earlier. [`#2483 <https://github.com/cta-observatory/ctapipe/pull/2483>`__]

- Compute deconvolution parameters in FlashCamExtractor only as needed. [`#2545 <https://github.com/cta-observatory/ctapipe/pull/2545>`__]

ctapipe v0.20.0 (2023-09-11)
============================


API Changes
-----------

- The ``ctapipe-dump-triggers`` tool was removed, since it wrote a custom data format
  not compatible with e.g. the output of the ``DataWriter`` and ``ctapipe-process``.
  If you only want to store trigger and simulation information from simulated / DL0
  input files into the ctapipe format HDF5 files, you can now use
  ``ctapipe-process -i <input> -o <output> --no-write-parameters``. [`#2375 <https://github.com/cta-observatory/ctapipe/pull/2375>`__]

- Change the fill value for invalid telescope ids in ``SubarrayDescription.tel_index_array``
  from ``-1`` to ``np.iinfo(int).minval`` to prevent ``-1`` being used as an index resulting in the last element being used for invalid telescope ids. [`#2376 <https://github.com/cta-observatory/ctapipe/pull/2376>`__]

- Remove ``EventSource.from_config``, simply use ``EventSource(config=config)`` or
  ``EventSource(parent=parent)``. [`#2384 <https://github.com/cta-observatory/ctapipe/pull/2384>`__]


Data Model Changes
------------------

- Added missing fields defined in the CTAO R1 and DL0 data models to the corresponding containers. [`#2338 <https://github.com/cta-observatory/ctapipe/pull/2338>`__]

- Remove the ``injection_height`` field from the ``SimulationConfigContainer``,
  this field was always empty and is never filled by ``sim_telarray``.

  Add the corresponding ``starting_grammage`` field to the ``SimulatedShowerContainer``,
  where it is actually available. [`#2343 <https://github.com/cta-observatory/ctapipe/pull/2343>`__]

- Added new fields to the ``MuonEfficiencyContainer`` - ``is_valid`` to check if fit converged successfully, ``parameters_at_limit`` to check if parameters were fitted close to a bound and ``likelihood_value`` which represents cost function value atthe minimum. These fields were added to the output of the ``MuonIntensityFitter``. [`#2381 <https://github.com/cta-observatory/ctapipe/pull/2381>`__]


New Features
------------

- Remove writing the full provenance information to the log  and instead simply refer the reader to the actual provenance file. [`#2328 <https://github.com/cta-observatory/ctapipe/pull/2328>`__]

- Add support for including r1 and r0 waveforms in the ``ctapipe-merge`` tool. [`#2386 <https://github.com/cta-observatory/ctapipe/pull/2386>`__]


Bug Fixes
---------

- The ```HillasIntersection``` method used to fail when individual events were reconstructed to originate from a FoV offset of more than 90 degrees.
  This is now fixed by returning an INVALID container for a reconstructed offset of larger than 45 degrees. [`#2265 <https://github.com/cta-observatory/ctapipe/pull/2265>`__]

- Fixed a bug in the calculation of the full numeric pixel likelihood and the corresponding tests. [`#2388 <https://github.com/cta-observatory/ctapipe/pull/2388>`__]


Maintenance
-----------

- Drop support for python 3.8 in accordance with the NEP 29 schedule. [`#2342 <https://github.com/cta-observatory/ctapipe/pull/2342>`__]

- * Switched to ``PyData`` theme for docs
  * Updated ``Sphinx`` to version 6.2.1
  * Updated front page of docs [`#2373 <https://github.com/cta-observatory/ctapipe/pull/2373>`__]



ctapipe 0.19.3 (2023-06-20)
===========================

This is a bugfix release fixing a number of bugs, mainly one preventing the processing of divergent pointing
prod6 data due to a bug in ``SoftwareTrigger``, see below for details.


Bug Fixes
---------

- Fix peak time units of FlashCamExtractor (See https://github.com/cta-observatory/ctapipe/issues/2336) [`#2337 <https://github.com/cta-observatory/ctapipe/pull/2337>`__]

- Fix shape of mask returned by ``NullDataVolumeReducer``. [`#2340 <https://github.com/cta-observatory/ctapipe/pull/2340>`__]

- Fix definition of the ``--dl2-subarray`` flag of ``ctapipe-merge``. [`#2341 <https://github.com/cta-observatory/ctapipe/pull/2341>`__]

- Fix ``ctapipe-train-disp-reconstructor --help`` raising an exception. [`#2352 <https://github.com/cta-observatory/ctapipe/pull/2352>`__]

- Correctly fill ``reference_location`` for ``SubarrayDescription.tel_coords``. [`#2354 <https://github.com/cta-observatory/ctapipe/pull/2354>`__]

- Fix ``SoftwareTrigger`` not removing all parts of a removed telescope event
  from the array event leading to invalid files produced by ``DataWriter``. [`#2357 <https://github.com/cta-observatory/ctapipe/pull/2357>`__]

- Fix that the pixel picker of the matplotlib ``CameraDisplay`` triggers
  also for clicks on other ``CameraDisplay`` instances in the same figure. [`#2358 <https://github.com/cta-observatory/ctapipe/pull/2358>`__]


New Features
------------

- Add support for Hillas parameters in ``TelescopeFrame`` to
  ``CameraDisplay.overlay_moments`` and make sure that the
  label text does not overlap with the ellipse. [`#2347 <https://github.com/cta-observatory/ctapipe/pull/2347>`__]

- Add support for using ``ctapipe.image.toymodel`` features in ``TelescopeFrame``. [`#2349 <https://github.com/cta-observatory/ctapipe/pull/2349>`__]


Maintenance
-----------

- Improve docstring and validation of parameters of ``CameraGeometry``. [`#2361 <https://github.com/cta-observatory/ctapipe/pull/2361>`__]



ctapipe v0.19.2 (2023-05-17)
============================

This release contains a critical bugfix for the ``FlashCamExtractor`` that resulted
in non-sensical peak time values in DL1, see below.

Bug Fixes
---------

- Fix a bug in the peak_time estimation of ``FlashCamExtractor`` (See issue `#2332 <https://github.com/cta-observatory/ctapipe/issues/2332>`_) [`#2333 <https://github.com/cta-observatory/ctapipe/pull/2333>`__]


ctapipe v0.19.1 (2023-05-11)
============================

This release is a small bugfix release for v0.19.0, that also includes a new feature enabling computing different
telescope multiplicities in the machine learning feature generation.

Thanks to the release of numba 0.57 and some minor fixes, ctapipe is now also compatible with Python 3.11.

Bug Fixes
---------

- Fix ``ApplyModels.overwrite``. [`#2311 <https://github.com/cta-observatory/ctapipe/pull/2311>`__]

- Fix for config files not being included as inputs in provenance log. [`#2312 <https://github.com/cta-observatory/ctapipe/pull/2312>`__]

- Fix calculation of the neighbor matrix of ``CameraGeometry`` for empty and single-pixel geometries. [`#2317 <https://github.com/cta-observatory/ctapipe/pull/2317>`__]

- Fix HDF5Writer not working on windows due to using pathlib for hdf5 dataset names. [`#2319 <https://github.com/cta-observatory/ctapipe/pull/2319>`__]

- Fix StereoTrigger assuming the wrong data type for ``tels_with_trigger``, resulting in
  it not working for actual events read from an EventSource. [`#2320 <https://github.com/cta-observatory/ctapipe/pull/2320>`__]

- Allow disabling the cross validation (by setting ``CrossValidator.n_cross_validations = 0``)
  for the train tools. [`#2310 <https://github.com/cta-observatory/ctapipe/pull/2310>`__]


New Features
------------

- Add ``SubarrayDescription.multiplicity`` method that can compute
  telescope multiplicity for a given telescope boolean mask, either for
  all telescope or a given telescope type.

  Enable adding additional keyword arguments to ``FeatureGenerator``.

  Pass the ``SubarrayDescription`` to ``FeatureGenerator`` in sklearn classes. [`#2308 <https://github.com/cta-observatory/ctapipe/pull/2308>`__]


Maintenance
-----------

- Add support for python 3.11. [`#2107 <https://github.com/cta-observatory/ctapipe/pull/2107>`__]


ctapipe v0.19.0 (2023-03-30)
============================

API Changes
-----------

- Renamed ``GeometryReconstructor`` to ``HillasGeometryReconstructor`` [`#2293 <https://github.com/cta-observatory/ctapipe/pull/2293>`__]


Bug Fixes
---------


Data Model Changes
------------------


New Features
------------

- Add signal extraction algorithm for the FlashCam. [`#2188 <https://github.com/cta-observatory/ctapipe/pull/2188>`__]


Maintenance
-----------

- The ``examples/`` subdirectory was removed as most scripts there were out of date. Useful information in those examples was moved to example notebooks in docs/examples [`#2266 <https://github.com/cta-observatory/ctapipe/pull/2266>`__]

- The tools to train ml models now provide better error messages in case
  the input files did not contain any events for specific telescope types. [`#2295 <https://github.com/cta-observatory/ctapipe/pull/2295>`__]


Refactoring and Optimization
----------------------------


ctapipe v0.18.1 (2023-03-16)
============================


Bug Fixes
---------

- Ensure the correct activity metadata is written into output files. [`#2261 <https://github.com/cta-observatory/ctapipe/pull/2261>`__]

- Fix ``--overwrite`` option not taking effect for ``ctapipe-apply-models``. [`#2287 <https://github.com/cta-observatory/ctapipe/pull/2287>`__]

- Fix ``TableLoader.read_subarray_events`` raising an exception when
  ``load_observation_info=True``. [`#2288 <https://github.com/cta-observatory/ctapipe/pull/2288>`__]



ctapipe v0.18.0 (2023-02-09)
============================


API Changes
-----------

- ctapipe now uses entry points for plugin discovery. ``EventSource`` implementations
  now need to advertise a ``ctapipe_io`` entry point, to be discovered by ctapipe.
  Additionally, ctapipe now includes preliminary support for discovering ``Reconstructor``
  implementations via the ``ctapipe_reco`` entry_point. [`#2101 <https://github.com/cta-observatory/ctapipe/pull/2101>`__]

- Migrate muon analysis into the ``ctapipe-process`` tool:

  1. The former ``muon_reconstruction`` tool is dropped and all functionalities are transferred
     into the ``ctapipe-process`` tool.

  2. The ``process`` tool now has a ``write_muon_parameters`` flag which defaults to ``false``.
     Muons are only analyzed and written if the flag is set. Analyzing muons requires DL1 image
     parameters, so they are computed in case they are not available from the input even
     if the user did not explicitly ask for the computation of image parameters.

  3. Two instances of ``QualityQuery``, ``MuonProcessor.ImageParameterQuery`` and ``MuonProcessor.RingQuery``
     are added to the muon analysis to either preselect images according to image parameters and
     to select images according to the initial, geometrical ring fit for further processing.
     Deselected events or those where the muon analysis fails are being returned and written
     filled with invalid value markers instead of being ignored.
     Base configure options for the muon analysis were added to the ``base_config.yaml``.

  4. The ``DataWriter`` now writes the results of a muon analysis into ``/dl1/event/telescope/muon/tel_id``,
     given ``write_moun_parameters`` is set to ``true``.

  5. Muon nodes were added to the ``HDF5EventSource``, the ``TableLoader`` and the ``ctapipe-merge`` tool. [`#2168 <https://github.com/cta-observatory/ctapipe/pull/2168>`__]

- Change default behaviour of ``run_rool``:

  1. The default value of ``raises`` is now ``True``. That means, when using
     ``run_tool``, the Exceptions raised by a Tool will be re-raised. The raised
     exceptions can be tested for their type and content.
     If the Tool must fail and only the non-zero error case is important to test,
     set ``raises=False`` (as it was before).

  2. If the ``cwd`` parameter is ``None`` (as per default), now a temporary directory
     is used instead of the directory, where ``run_tool`` is called (typically via
     pytest). This way, log-files and other output files don't clutter your
     working space. [`#2175 <https://github.com/cta-observatory/ctapipe/pull/2175>`__]

- Remove ``-f`` flag as alias for ``--overwrite`` and fail early if output exists, but overwrite is not set [`#2213 <https://github.com/cta-observatory/ctapipe/pull/2213>`__]

- The ``_chunked`` methods of the ``TableLoader`` now return
  an Iterator over namedtuples with start, stop, data. [`#2241 <https://github.com/cta-observatory/ctapipe/pull/2241>`__]

- Remove debug-logging and early-exits in ``hdf5eventsource`` so broken files raise errors. [`#2244 <https://github.com/cta-observatory/ctapipe/pull/2244>`__]

New Features
------------

- Implement Components and Tools to perform training and application of
  machine learning models based on scikit-learn.

  Four new tools are implemented:
  - ``ctapipe-train-energy-regressor``
  - ``ctapipe-train-particle-classifier``
  - ``ctapipe-train-disp-reconstructor``
  - ``ctapipe-apply-models``

  The first two tools are used to train energy regression and particle classification
  respectively. The third tool trains two models for geometrical reconstruction using the disp
  method and the fourth tool can apply those models in bulk to input files.
  ``ctapipe-process`` can now also apply these trained models directly in the event loop.

  The intended workflow is to process training files to a combined dl1 / dl2 level
  using ``ctapipe-process``, merging those to large training files using ``ctapipe-merge``
  and then train the models.
  [`#1767 <https://github.com/cta-observatory/ctapipe/pull/1767>`__,
  `#2121 <https://github.com/cta-observatory/ctapipe/pull/2121>`__,
  `#2133 <https://github.com/cta-observatory/ctapipe/pull/2133>`__,
  `#2138 <https://github.com/cta-observatory/ctapipe/pull/2138>`__,
  `#2217 <https://github.com/cta-observatory/ctapipe/pull/2217>`__,
  `#2229 <https://github.com/cta-observatory/ctapipe/pull/2229>`__,
  `#2140 <https://github.com/cta-observatory/ctapipe/pull/2140>`__]

- ``Tool`` now comes with an ``ExitStack`` that enables proper
  handling of context-manager members inside ``Tool.run``.
  Things that require a cleanup step should be implemented
  as context managers and be added to the tool like this:

  .. code::

      self.foo = self.enter_context(Foo())

  This will ensure that ``Foo.__exit__`` is called when the
  ``Tool`` terminates, for whatever reason. [`#1926 <https://github.com/cta-observatory/ctapipe/pull/1926>`__]

- Implement atmospheric profiles for conversions from h_max to X_max.
  The new module ``ctapipe.atmosphere`` has classes for the most common cases
  of a simple ``ExponentialAtmosphereDensityProfile``, a ``TableAtmosphereDensityProfile``
  and CORSIKA's ``FiveLayerAtmosphereDensityProfile``. [`#2000 <https://github.com/cta-observatory/ctapipe/pull/2000>`__]

- ``TableLoader`` can now also load observation and scheduling block configuration. [`#2096 <https://github.com/cta-observatory/ctapipe/pull/2096>`__]

- The ``ctapipe-info`` tool now supports printing information about
  the available ``EventSource`` and ``Reconstructor`` implementations
  as well as io and reco plugins. [`#2101 <https://github.com/cta-observatory/ctapipe/pull/2101>`__]

- Allow lookup of ``TelescopeParameter`` values by telescope type. [`#2120 <https://github.com/cta-observatory/ctapipe/pull/2120>`__]

- Implement a ``SoftwareTrigger`` component to handle the effect of
  selecting sub-arrays from larger arrays in the simulations.
  The component can remove events where the stereo trigger would not have
  decided to record an event and also remove single telescopes from events
  for cases like the CTA LSTs, that have their own hardware stereo trigger
  that requires at least two LSTs taking part in an event. [`#2136 <https://github.com/cta-observatory/ctapipe/pull/2136>`__]


- It's now possible to transform between ``GroundFrame`` coordinates
  and ``astropy.coordinates.EarthLocation``, enabling the conversion
  between relative array coordinates (used in the simulation) and
  absolute real-world coordinates. [`#2167 <https://github.com/cta-observatory/ctapipe/pull/2167>`__]

- The ``ctapipe-display-dl1`` tool now has a ``QualityQuery`` instance which can be used
  to select which images should be displayed. [`#2172 <https://github.com/cta-observatory/ctapipe/pull/2172>`__]

- Add a new ``ctapipe.io.HDF5Merger`` component that can selectively merge
  HDF5 files produced with ctapipe. The new component is now used in the
  ``ctapipe-merge`` tool but can also be used on its own.
  This component is also used by ``ctapipe-apply-models`` to selectively copy
  data from the input file to the output file.
  Through using this new component, ``ctapipe-merge`` gained support for
  fine-grained control which information should be included in the output file
  and for appending to existing output files. [`#2179 <https://github.com/cta-observatory/ctapipe/pull/2179>`__]

- ``CameraDisplay.overlay_coordinate`` can now be used to
  plot coordinates into the camera display, e.g. to show
  the source position or the position of stars in the FoV. [`#2203 <https://github.com/cta-observatory/ctapipe/pull/2203>`__]


Bug Fixes
---------

- Fix for Hillas lines in ``ArrayDisplay`` being wrong in the new ``EastingNorthingFrame``. [`#2134 <https://github.com/cta-observatory/ctapipe/pull/2134>`__]

- Replace usage of ``$HOME`` with ``Path.home()`` for cross-platform compatibility. [`#2155 <https://github.com/cta-observatory/ctapipe/pull/2155>`__]

- Fix for ``TableLoader`` having the wrong data types for ``obs_id``,
  ``event_id`` and ``tel_id``. [`#2163 <https://github.com/cta-observatory/ctapipe/pull/2163>`__]

- Fix ``Tool`` printing a large traceback in case of certain configuration errors. [`#2171 <https://github.com/cta-observatory/ctapipe/pull/2171>`__]

- The string representation of ``Field`` now sets numpy print options
  to prevent large arrays in the docstrings of ``Container`` classes. [`#2173 <https://github.com/cta-observatory/ctapipe/pull/2173>`__]

- Fix missing comma in eventio version requirement in setup.cfg (#2185). [`#2187 <https://github.com/cta-observatory/ctapipe/pull/2187>`__]

- Move reading of stereo data before skipping empty events in HDF5EventSource,
  this fixes a bug where the stereo data and simulation data get out of sync
  with the other event data when using ``allowed_tels``. [`#2189 <https://github.com/cta-observatory/ctapipe/pull/2189>`__]

- Fix mixture of quantity and unit-less values passed to ``np.histogram``
  in ``ctapipe.image.muon.ring_completeness``, which raises an error with
  astropy 5.2.1. [`#2197 <https://github.com/cta-observatory/ctapipe/pull/2197>`__]


Maintenance
-----------

- Use towncrier for the generation of change logs [`#2144 <https://github.com/cta-observatory/ctapipe/pull/2144>`__]

- Replace usage of deprecated astropy matrix function. [`#2166 <https://github.com/cta-observatory/ctapipe/pull/2166>`__]

- Use ``weakref.proxy(parent)`` in ``Component.__init__``.

  Due to the configuration systems, children need to reference their parent(s).
  When parents get out of scope, their children still hold the reference to them.
  That means that python cannot garbage-collect the parents (which are Tools, most of the time).

  This change uses weak-references (which do not increase the reference count),
  which means parent-Tools can get garbage collected by python.

  This decreases the memory consumption of the tests by roughly 50%. [`#2223 <https://github.com/cta-observatory/ctapipe/pull/2223>`__]


Refactoring and Optimization
----------------------------

- Speed-up table loader by using ``hstack`` instead of ``join`` where possible. [`#2126 <https://github.com/cta-observatory/ctapipe/pull/2126>`__]


v0.7.0 – 0.17.0
===============

For changelogs for these releases, please visit the `github releases page <https://github.com/cta-observatory/ctapipe/releases>`__


v0.6.1
======

* Fix broken build (#743) @kosack
* Add example script for a simple event writer (#746) @jjlk
* Fix camera axis alignment in HillasReconstructor (#741) @mackaiver
* Lst reader (#749) @FrancaCassol
* replace deprecated astropy broadcast (#754) @mackaiver
* A few more example notebooks (#757) @kosack
* Add MC xmax info (#759) @mackaiver
* Use Astropy Coordinate Transofmations For Reconstruction (#758) @mackaiver
* Trigger pixel reader (#745) @thomasarmstrong
* Change requested in #742: init Hillas skewness and kurtosis to NaN (#744) @STSpencer
* Fix call to np.linalg.leastsq (#760) @kosack
* Fix/muon bugs (#762) @kosack
* Implement hillas features usen eigh (#748) @MaxNoe
* Use HillasParametersContainer only (#763) @MaxNoe
* Regression features in ``RegressorClassifierBase`` (#764) @vuillaut
* Adding an example notebook no how to convert hex geometry to square and back (#767) @vuillaut
* Wrong angle in ArrayDisplay. changed phi to psi. (#771) @thomasgas
* Unstructured interpolator (#770) @ParsonsRD
* Lst reader (#776) @FrancaCassol
* Fixing core reconstruction (#777) @kpfrang
* Leakage (#783) @MaxNoe
* Revert "Fixing core reconstruction" (#789) @kosack
* Fixing the toy image generator (#790) @MaxNoe
* Fix bad builds by changing channel name (missing pyqt package) (#793) @kosack
* Implement concentration image features (#791) @MaxNoe
* updated main documentation page (#792) @kosack
* Impact intersection (#778) @mackaiver
* add test for sliced geometries for hillas calculation (#781) @mackaiver
* Simple HESS adaptations (#794) @ParsonsRD
* added a config file for github release-drafter plugin (#795) @kosack
* Array plotting (#784) @thomasgas
* Minor changes: mostly deprecationwarning fixes (#787) @mireianievas
* Codacy code style improvements (#796) @dneise
* Add unit to h_max in HillasReconstructor (#797) @jjlk
* speed up unit tests that use test_event fixture (#798) @kosack
* Update Timing Parameters (#799) @LukasNickel

v0.6.0
======

This is an interim release, after some major refactoring, and before we add
the automatic gain selection and refactored container classes. It's not
intended yet for production.

Some Major changes since last release:

* new ``EventSource`` class hierarchy for reading event data, which now supports simulation and testbench data from multiple camera prototypes (notably CHEC, SST-1M, NectarCam)
* new ``EventSeeker`` class for (inefficient) random event access.
* a much improved ``Factory`` class
* re-organized event data structure (still evolving) - all scripts not in ctapipe must be changed to work with the new data items that were re-named  (a migration guide will be given in the 0.7 release)
* better HDF5 table output, supporting merging multiple ``Containers`` into a single output table
* improvements to Muon analysis, and the muon example script
* improvements to the calibration classes
* big improvements to the Instrument classes
* lots of cleanups and bug fixes
* much more...

v0.5.3 (unreleased)
===================

* Major speed improvements to calibration code, particularly
   ``NeighborPeakIntegrator`` (Jason Watson, #490), which now uses some
   compiled c-code for speed.

* ``GeometryConverter`` now works for all cameras (Tino Michael, #)

* Plotting improvements when overlays are used (MaxNoe, #489)

* Fixes to coordinate ``PlanarRepresentation`` (MaxNoe, #506)

* HDF5 output for charge resolution calculation (Jason Watons, #488)

* Stastical errors added to sensitivity calculation (Tino Michel, #508)

* Error estimator for direction and h_max fits in
  ``HillasReconstructor`` (Tino Michael, #509, #510)


v0.5.2 (2017-07-31)
===================

* improvements to ``core.Container`` (MaxNoe)

* ``TableWriter`` correctly handles units and metadata

* ``ctapipe.instrument`` now has much more rich functionality
  (SubarrayDescription, TelescopeDescription, OpticsDescription
  classes added)

* no more need to construct ``CameraGeometry`` manually, they are
  created in the ``hessio_event_source``, all new code should use
  ``event.inst.subarray``. The old inst.tel_pos, inst.optics_foclen,
  etc, will be phased out in the next point release (but still exist
  in this release) (K. Kosack)

* ``ctapipe-dump-instrument`` script added

* improvements to ``Regressor`` and Classifier code (Tino Michael)

* provenance system includes actor roles

* fixes to likelihood tests (Dan Parsons)



v0.5.1 (2016-07-20)
===================


* TQDM and iminuit are now accepted dependencies

* Implementation of ImPACT reconstruction and ``TableInterpolator``
  class (Dan Parsons)

* improved handling of atmosphere profiles

* Implementation of Muon detection and reconstruction algorithms
  (Alison Mitchell)

* unified camera and telescope names

* better dataset handling (``ctapipe.utils.datasets``), and now
  automatically find datasets and tables in ``ctapipe-extra`` or in any
  directory listed in the user-defined ``$CTAPIPE_SVC_PATH`` path.

* TableWriter class (HDF5TableWriter) for writing out any
  ``core.Container`` to an HDF5 table via ``pytables`` (Karl Kosack)

* Improvements to ``flow`` framework (Jean Jacquemier)

* Travis CI now builds automatically for multiply python versions and
  uploads latest documentation

* use Lanscape.io for code quality

* code for calculating sensitivity curves using event-weighting method
  (Tino Michael)
