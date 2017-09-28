==========
Change Log
==========

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

