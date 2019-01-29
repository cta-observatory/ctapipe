==========
Change Log
==========


v0.6.1
------

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
* Regression features in `RegressorClassifierBase` (#764) @vuillaut
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
------

This is an interim release, after some major refactoring, and before we add
the automatic gain selection and refactored container classes. It's not
intended yet for production.

Some Major changes since last release:

* new `EventSource` class hierarchy for reading event data, which now supports simulation and testbench data from multiple camera prototypes (notably CHEC, SST-1M, NectarCam)
* new `EventSeeker` class for (inefficient) random event access.
* a much improved `Factory` class
* re-organized event data structure (still evolving) - all scripts not in ctapipe must be changed to work with the new data items that were re-named  (a migration guide will be given in the 0.7 release)
* better HDF5 table output, supporting merging multiple `Containers` into a single output table
* improvements to Muon analysis, and the muon example script
* improvements to the calibration classes
* big improvements to the Instrument classes
* lots of cleanups and bug fixes
* much more...

v0.5.3 (unlreleased)
--------------------

* Major speed improvements to calibration code, particuarly
   `NeighborPeakIntegrator` (Jason Watson, #490), which now uses some
   compiled c-code for speed.

* `GeometryConverter` now works for all cameras (Tino Michael, #)

* Plotting improvements when overlays are used (Max Noe, #489)

* Fixes to coordinate `PlanarRepresentation` (Max Noe, #506)

* HDF5 output for charge resolution calculation (Jason Watons, #488)

* Stastical errors added to sensitivity calculation (Tino Michel, #508)

* Error estimator for direction and h_max fits in
  `HillasReconstructor` (Tino Michael, #509, #510)
  

v0.5.2 (2017-07-31)
-------------------

* improvements to `core.Container` (Max Noe)

* `TableWriter` correctly handles units and metadata

* `ctapipe.instrument` now has much more rich functionality
  (SubarrayDescription, TelescopeDescription, OpticsDescription
  classes added)

* no more need to construct `CameraGeometry` manually, they are
  created in the `hessio_event_source`, all new code should use
  `event.inst.subarray`. The old inst.tel_pos, inst.optics_foclen,
  etc, will be phased out in the next point release (but still exist
  in this release) (K. Kosack)

* `ctapipe-dump-instrument` script added

* improvements to `Regressor` and Classifier code (Tino Michael)

* provenance system includes actor roles

* fixes to likelihood tests (Dan Parsons)


  
v0.5.1 (2016-07-20)
-------------------


* TQDM and iminuit are now accepted dependencies

* Implementation of ImPACT reconstruction and `TableInterpolator`
  class (Dan Parsons)

* improved handling of atmosphere profiles

* Implementation of Muon detection and reconstruction algorithms
  (Alison Mitchell)

* unified camera and telescope names
  
* better dataset handling (`ctapipe.utils.datasets`), and now
  automatically find datasets and tables in `ctapipe-extra` or in any
  directory listed in the user-defined `$CTAPIPE_SVC_PATH` path.

* TableWriter class (HDF5TableWriter) for writing out any
  `core.Container` to an HDF5 table via `pytables` (Karl Kosack)

* Improvements to `flow` framework (Jean Jacquemier)

* Travis CI now builds automatically for multiply python versions and
  uploads lates documentation

* use Lanscape.io for code quality

* code for calculating sensitivity curves using event-weighting method
  (Tino Michael)

