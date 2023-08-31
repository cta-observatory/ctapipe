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

- Add ``SubarrayDescription.mulitplicity`` method that can compute
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

* Major speed improvements to calibration code, particuarly
   ``NeighborPeakIntegrator`` (Jason Watson, #490), which now uses some
   compiled c-code for speed.

* ``GeometryConverter`` now works for all cameras (Tino Michael, #)

* Plotting improvements when overlays are used (Max Noe, #489)

* Fixes to coordinate ``PlanarRepresentation`` (Max Noe, #506)

* HDF5 output for charge resolution calculation (Jason Watons, #488)

* Stastical errors added to sensitivity calculation (Tino Michel, #508)

* Error estimator for direction and h_max fits in
  ``HillasReconstructor`` (Tino Michael, #509, #510)


v0.5.2 (2017-07-31)
===================

* improvements to ``core.Container`` (Max Noe)

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
  uploads lates documentation

* use Lanscape.io for code quality

* code for calculating sensitivity curves using event-weighting method
  (Tino Michael)
