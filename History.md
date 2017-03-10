0.3.5 / 2017-03-09
==================

  * some updates to example notebooks
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe
  * Merge pull request #337 from watsonjj/small_example_fix
  * Corrected r0 to dl0
  * Merge remote-tracking branch 'cta-observatory/master'
  * Merge pull request #336 from kosack/watsonjj-r0container
  * Merge branch 'master' into watsonjj-r0container
  * Merge pull request #284 from watsonjj/r0container
  * merge and apply fixes to PR#284
  * Merge remote-tracking branch 'remotes/upstream/master' into r0container
  * Replaced event.dl0 calls to event.r0
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe into watsonjj-r0container
  * Merge branch 'r0container' of https://github.com/watsonjj/ctapipe into watsonjj-r0container
  * Merge pull request #335 from kosack/master
  * small fix to example
  * Merge pull request #334 from kosack/master
  * add --all option to ctapipe-info
  * fix broken --dependencies option in ctapipe-info
  * Merge pull request #333 from kosack/docupdate
  * Merge remote-tracking branch 'cta-observatory/master' into docupdate
  * Merge pull request #331 from kosack/master
  * fix bug in CameraDisplay
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe into feature/--help
  * Merge remote-tracking branch 'remotes/upstream/master' into r0container
  * minor camera display update
  * Merge pull request #326 from kosack/master
  * use cta-observatory conda channel for pyhessio install
  * Merge pull request #325 from kosack/master
  * remove version cache from git repo and make sure it's in .gitignore
  * Merge pull request #324 from kosack/master
  * improve camdemo tool
  * add version cache to gitignore
  * Merge remote-tracking branch 'remotes/upstream/master' into r0container
  * Added pedestal_path and tf_path to CameraR1CalibratorFactor, as real cameras require these separate files for calibration
  * Corrected bug in Factory
  * Merge remote-tracking branch 'remotes/upstream/master' into r0container
  * Merge branch 'filereader' into r0container
  * Merge branch 'filereader' into r0container
  * Small change to how integration window is plotted
  * Included extra external children
  * Corrected container
  * Corrected @observe logging
  * Corrected container for DL0
  * Updated examples/display_integrator.py to use new calibration steps
  * Updated calibration pipeline to use new calibration steps
  * Renamed examples/dl1_calibration.py back to examples/calibration_pipeline.py as it now contains the whole calibration chain from r0 to dl1
  * Corrections to pass tests
  * Added tests for calibrators
  * Removed calibrate_source
  * Corrected check_*_exists
  * Renamed test files Renamed MCR1Calibrator to HessioR1Calibrator
  * Created function check_*_exists in each calibration step to allow them to be ran even if the data has been read at a later calibration step
  * Created dl0.py to handle to data volume reduction in the conversion from r1 to dl0. Created reductors.py to contain the date volume reductors that can be passed as an argument to CameraDL0Reducer in dl0.py.
  * Corrected r1.py to loop through all telescopes that have data in the event
  * Removed dl0 correction for mc files - this is now handled by r1.py
  * Removed mc.py and mycam.py as they do not fit in the new calib methodology
  * Created r1 calibrator component - should replace mc.py
  * Added origin attribute to EventFileReader class - useful for components that depend on the file type
  * Updated docstring
  * Renamed ctapipe/calib/camera/calibrators.py to ctapipe/calib/camera/dl1.py
  * Removed clearing of dl2
  * Imported external classes that should be included in the factory
  * Refactored dl0 container to r0 container Created new containers: r1 (containing r1 calibrated data) and dl0 (containing data volume reduced data)
