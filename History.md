
n.n.n / 2017-05-04
==================

  * add pytables as requirement in setup.py
  * added TQDM (progress bar) to environment, since needed by some tools
  * added pytables back to env
  * remove accidentally committed file
  * apply PEP8 formatting to ImPact, and remove dependency on matplotlib
  * fixed docs for utils
  * simplify import of get_dataset (import from utils not utils.datasets_
  * small refactoring and fix for IMPACT use of ShowerMaxEstimator
  * better utils.datasets.get_dataset
  * updated ctapipe-dump-instrument to allow multiple output types
  * fixed bad __all__ attribute in reductors and some doc problem
  * more doc fixes
  * fix some doc warnings
  * fixed some docs in image/muon module
  * update to image documentation
  * provide get_known_camera_type() method
  * unified loading of all camera types in CameraGeometry.from_name
  * more doc fixes
  * fixed lots of typos in documentation
  * few more doc cleanups
  * fix documentation problems
  * use CameraCalibrator in simple_pipeline example
  * removed files that shouldn't be in package
  * improve docs
  * clean up simple pipeline imports
  * refactor location of atmosphere profile functions
  * add docs, fix units, and use inverse transform rather than solver
  * minor improvement to documentation for datasets module
  * interpolate atmosphere file in instrument
  * Correcting coordinate treatment
  * Updated documentation for image module
  * Created test and docs for CameraCalibrator
  * fix impact tests to work with new code and add instrument.get_atmosphere_profile
  * Created conveniance calibrator (CameraCalibrator) that applies r1, dl0 and dl1 calibration Updated calibration_pipeline with CameraCalibrator
  * Fixed tests for calibration
  * fix example to use get_dataset
  * moved table_interpolator from reco to utils
  * Moved charge_extraction.py, waveform_cleaning.py and reductors.py to the images module
  * Update calib.camera docs Removed test_mycam.py Change to absolute python import in dl1.py
  * added tqdm and iminuit as dependencies
  * Update of docs and docstrings for the calibration module. Deleted old docs.
  * removed files accidentally committed to Muon Master
  * Removed example script (will be replaced later)
  * added FAQ and author list
  * Reverted the dl1 image back to a normal numpy array instead of a masked array Updated documentation for charge extractors
  * Added unit tests
  * Changed code to work with new coordinates scheme
  * added author list
  * fix size hillas parameter display overlay
  * minor bug fixes preparing for merge to master
  * fix #248, now have option to do image cleaning where single pixels above pic thresh are removed
  * Adding muon fitting tests - still need to make image not a masked array
  * add a tool to dump instrument info to FITS files from simtel files
  * Cleanup
  * Corrected integration_correction for telescopes with bin widths not equal to 1
  * fix entry points for chargeresolution
  * change makefile to run pytest directly instead of via setup.py
  * remove psutil dependence in provenance
  * fix import in example
  * add print level and pedantic to minuit
  * change format of time read from monte-carlo
  * Fixed help message for calibration_pipeline example
  * use fixtures in test_serializer to avoid writing files to random places
  * fix provenance test
  * updated provenance test
  * fix loading of array data
  * remove need for CTAPIPE_EXTRAS_DIR, and use ctapipe_resources package
  * add UUID for each activity
  * correctly return from generator in SimpleHDF5TableReader
  * Added comment
  * Added comment
  * Added comments and improved layout of viewer
  * Changed colour scale and removed print statement
  * Created component for a full event display including the individual camera images and the projection of the Hillas parameters in different frames
  * Added support for the inclusion of multiple axes in the plot
  * Added camera rotation as this is needed for correct reconstruction
  * doc update
  * Fix dimensions problem of correction array
  * SimpleHDF5TableReader.read() now returns a generator
  * more reader functionality
  * basic reader functionality
  * started implementing a reader
  * move more common functionality into base class
  * refactor and add column exclusions
  * Changed pointing direction to take the HorizonFrame (or AltAz) object as a definition of an array pointing direction
  * Update docstrings Change integration_correction arguments to be more generic
  * add more translation functions
  * more docs for hdf5 tables
  * add numpy and tables as requirements
  * better docs and test case
  * improve SimpleHDF5TableWriter
  * add __getitem__ to container to assist with serializers
  * added simplistic hdf5 table writer class
  * Changes to handle ASTRI
  * added a test script
  * use CameraGeometry to get neighbors instead of custom code (large speedup)
  * small cleanups
  * remove unnecessary call to hash() in guess()
  * start brainstorm
  * pass format and other file-related options in `CameraGeometry.from_table`
  * clean up example
  * clean up image docs too
  * minor doc update
  * minor reformatting
  * improve CameraGeometry documentation
  * replace progress bar from tqdm package with astropy one to avoid extra dependency
  * change test to use tmpdir
  * add CameraGeometry.from_table() functionality
  * added a few utils funcs to help with debugging
  * simplify neighbor pixel calculation (failed for some cameras with gaps)
  * removed extraneous global _geometry_cache
  * simplify tailcuts_clean and dilate
  * undo last commit
  * removed some unused code from neighbor attribute
  * add some comments about the caching of CameraGeometry
  * fix bug in CameraGeometry that broke geometry_converter
  * update docs for cleaning
  * rewrite cleaning.dilate to use neighbor matrix (much faster)
  * fix examples for new cleaning functions
  * fix example
  * change tailcuts_clean algorithm to be faster
  * use astropy.utils.lazyproperty for neighbors
  * Traitlet for adc2pe file
  * memoize CameraGeometry.guess for speed
  * fix some more imports and tests
  * update notebook to use instrument.CameraGeometry
  * clean up
  * refactor to use instrument.CameraGeometry
  * use instrument.CameraGeometry
  * moved io/camera.py to instrument and removed duplicate CameraDescription
  * Updated comment
  * improve hillas tests
  * Renamed instances of GATE to GCT
  * Cleanup
  * Created tests for waveform_cleaning
  * Corrected AverageWfPeakIntegrator Created test for AverageWfPeakIntegrator
  * Cleanup of charge_extractors Renamed window_start to t0 in charge_extractors. Window shift is now considered in SimpleIntegrator. Created waveform_cleaning.py for waveform cleaning methods to be applied in the dl1 calibration. Currently contains NullWaveformCleaner, CHECMWaveformCleaner, and a factory.
  * Renamed entrypoints for charge resolution
  * set title of displays to cam_id by default, fix pixel spacing
  * use PIXEL_EPSILON with hex cameras too
  * added small epsilon value to pixel size to fix aliasing
  * Added entrypoints for charge resolution scripts
  * Moved charge resolution scripts into tools
  * Added viewer for nominal system
  * add tool example to sum images for similiar cameras and display
  * minor fix to file saving
  * add contextmanager for activities
  * Added option for drawing the extrapolation of the image axes to the array view. Useful for checking Hillas reconstruction
  * Fixed telescope colours when drawing Hillas params. Reduces default size of markers
  * minor changes for plotting purposes
  * Added scaling factor to ImPACT Templates
  * calibration updates
  * Adapting to recent calibration changes
  * minor changes
  * update notebook
  * add example provenance notebook
  * add Provenance.clear() method
  * add automatic provenance in Tool and io
  * use Singleton metaclass for provenance
  * refactoring
  * separate activity provenance from global provenance
  * cleanups to proveance class and core init
  * update provenence
  * first attempt at system provenance and time sampling
  * Increased the requirement on the tolerance (should be more accurate)
  * Modified print outs and file names in ImPACT
  * increasing tail cuts thresholds
  * Updated code for use with newer image templates
  * making muon code compatible with recent ctapipe updates
  * Added new charge extractor: AverageWfPeakIntegrator
  * Improved functions from drawing a single telescope image
  * Added comments and changed the style in which the telescopes are drawn, now only outlines are drawn with a colour appropriate for the telescope type
  * Added funtionality to the Array drawer to overlay points, and centre the map on that point (useful for checking reconstruction)
  * Adding functionality to array drawing
  * Added functions for overlaying Hillas parameters on the array level view
  * Added simple classes for drawing array-level events
  * Updated ImPACT code to ensure units are handled consistantly within the class
  * Simple fix for mismatch of array dimensions (should be fixed in master soon)
  * Added comments
  * Updated the prediction function such that is takes the shower and energy classes as an arguement
  * Major refactoring of the code. Lists used internally all replaced with dictionaries, so that we keep track of the telescopes in the event at all times. Output switched to being the new reconstructed shower container.
  * Added use of new pixel likelihood class
  * Added first test for ImPACT
  * removing scipy
  * Added additional cuts, limited default reconstruction to only SST-GCT
  * Changed binning of image templates
  * Major changes to the code to deal with the new format of ImPACT tables. The biggest change is the difference in units stored, with the templates now being directly stored as the expected numerb of p.e. for a given telescopes type. Option for fitting the first interaction point has also been added. Scheme for loading of image templates has been altered to load template as required.
  * Change from iminuit to scipy.optimise
  * Updated cuts and position at which fitting seeds are thrown
  * Added printing of errors
  * Added option for specifying the root directory of the lookup tables, such that this is not hardcoded to a location on a given machine
  * Removed function included my mistake
  * Removed function included my mistake
  * Removed function included my mistake
  * Removed function included my mistake
  * Tried to santise the rotation angle in the nominal system conversion, still needs some work
  * Removed factors in angle calculation we are unsure about
  * Reverted changes made by mistake
  * Updated reader to deal with new dictionary based table format
  * Added compatability for older sim_telarray file extensions
  * Removed pyhessio directory from inside ctapipe
  * Renamed instances of meta.pixel_pos and meta.optical_foclen to inst.pixel_pos and inst.optical_foclen zfits.py requires changing to match the new hessio.py
  * Plotting Additions
  * Gaus fit and astropy table
  * Modified structure of stored ImPACT templates
  * Adding muon cut dictionary
  * clean up
  * Changed interpolator name to something more general
  * Updated for use with new container classes and pyhessio interface
  * Added prototype script for the generation of image templates using camera images. Rotates and translates images into the correct system, then you can just take the average.
  * Fixed unit conversion in rotation function
  * Added example analysis script for prod 3. This runs the first pass at the ImPACT analysis implmentation, essentially this is a direct translation of the code implemented within the H.E.S.S. analysis to the CTA framework.
  * Added interpolation features to better deal with edge cases when performing interpolation
  * Added camera rotation value to the output object when guessing the camera type. This is useful later when performing coordinate conversions
  * Fixed error that was thrown due to features now unsupported by python 3.5
  * Fixed broken call to lower edges
  * First pass at creating a template fitting class using iminuit and the template interpolator
  * Created am image template interpolator class. This class loads templates from pickle files and create a scipy interpolator class. This can then be used for providing the model for ImPACT fitting.
  * Fixed display of muon image and likelihood fit
  * Changes to chord_length and minor fixes
  * Changing some cuts
  * Compatibility to updated coordinates class
  * updating plotting diagnostics
  * Resolved camera rotation issue with muon ring overlay
  * Overlay fitted ring and debugging
  * Adding muon event display
  * first plotting and fewer hard coded values
  * More docs
  * Add more docstrings and comments
  * Move muon features to image
  * Return nan if fit not successfull
  * Simplify efficiency estimation by using the other fit methods
  * Fix call to norm.pdf in efficiency fit
  * Add efficiency fit function
  * Fix impact_parameter fit: Use bounds and fix integral function
  * Use a binned chisq fit to determine impact
  * Add import for new fit in muon __init__ and fix return value
  * Add impact parameter likelihood calculation
  * Fix calculation of ring_completeness
  * Add function to calculate completness of muon rings
  * Move feature calculation to own file
  * Fix mean_squared_error in muon
  * Add two goodness of fit features for muons
  * Add gaussian likelihood ring fit to calib.array.muon
  * Add support and test for astropy units muon fit
  * Add first ring fitting algorithm in muon submodule
  * Fix tests
  * Fix PEP8 stuff
  * Use scipy.stats.norm instead of self implemented gaussian pdf
  * Use numpy.pi, one import less
  * Pep 8 stuff
  * Do not override __init__ of component (This would break everything)
  * Revert "adding astropy_helpers"
  * adding astropy_helpers
  * reco.cleaning moved to image.cleaning
  * Minor parameter adjustments
  * Getting true mirror radius (not hardcoded)
  * Getting intensity fitting (hess based algorithm) working on CTA MC
  * Added test to  test_muon_ring_finder, minor modif to muon_ring finder
  * Corrected a bug with the parameters passed to fit_muon function
  * Cleaner code
  * First draft of functions to run the muon reconstruction pipeline. DOES  NOT WORK at the moment. The hard coded cuts must be modified first
  * force units in ChaudhuriKunduRingFitter.fit return the MuonRingParameter container from ChaudhuriKunduRingFitter.fit
  * Example of how to use the muon fit functions and muon reconstruction pipeline (including the image calibration).
  * Fix indentation and container description for MuonRingParameter and MuonIntensityParameter
  * Divide MuonParameter container into two containers: -one for the ring fit output -one for the intensity fit output
  * Added EventContainer to __all__ (was missing in documentation)
  * Implemented reading in of telescope mirror dish area and number of mirror tiles from simtelarray files
  * Modify some names in the MuonParameter container, made them more intuitive
  * Add position on the mirror of the muon impact
  * Add MuonParameter container
  * new base class for the intensity fitters
  * Adding __init__ files to new modules. Changes to muon_reco.py for compatibility with changed RingFitter class
  * yet another fix
  * Make the test meaningful by creating a real circle
  * Added a test directory Fixed bugs in imports
  * Fixed the documentation format to the standard numpy one in the base class Made the Chaudhuri Kundu ring fitter function derive from the base class
  * add the ring fitter base class ,thanks to Kai
  * Adding muon calibration algorithm files from RDP (H.E.S.S. based algorithm)
  * Added test functions
  * Added usints to cuts
  * Added support for astropy units
  * Added lots of comments
  * Fix now working, values currently hardcoded to HESS I
  * Working minimisation with sensible likelihood
  * Fixed minimisation and added print statements
  * Added minimisation function
  * Fixed Unints
  * Added worker classes and example scripts for muon fitting

n.n.n / 2017-05-04
==================

  * Merge pull request #403 from kosack/fix_impact_formating
  * add pytables as requirement in setup.py
  * added TQDM (progress bar) to environment, since needed by some tools
  * added pytables back to env
  * remove accidentally committed file
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe into fix_impact_formating
  * Merge pull request #402 from kosack/master
  * apply PEP8 formatting to ImPact, and remove dependency on matplotlib
  * fixed docs for utils
  * Merge pull request #401 from kosack/master
  * simplify import of get_dataset (import from utils not utils.datasets_
  * Merge pull request #400 from kosack/master
  * small refactoring and fix for IMPACT use of ShowerMaxEstimator
  * better utils.datasets.get_dataset
  * Merge pull request #399 from AMWMitchell/master
  * Merge pull request #398 from kosack/master
  * updated ctapipe-dump-instrument to allow multiple output types
  * fixed bad __all__ attribute in reductors and some doc problem
  * more doc fixes
  * fix some doc warnings
  * Merge pull request #395 from kosack/master
  * fixed some docs in image/muon module
  * update to image documentation
  * Merge pull request #394 from kosack/unify_camera_geom_by_name
  * provide get_known_camera_type() method
  * unified loading of all camera types in CameraGeometry.from_name
  * more doc fixes
  * fixed lots of typos in documentation
  * Merge pull request #393 from kosack/master
  * few more doc cleanups
  * Merge pull request #392 from kosack/master
  * fix documentation problems
  * Merge pull request #391 from kosack/master
  * use CameraCalibrator in simple_pipeline example
  * Merge pull request #390 from kosack/master
  * removed files that shouldn't be in package
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe
  * Merge pull request #388 from watsonjj/additional_dl1
  * Merge pull request #389 from kosack/fix_atmosphere_profile
  * improve docs
  * clean up simple pipeline imports
  * refactor location of atmosphere profile functions
  * add docs, fix units, and use inverse transform rather than solver
  * minor improvement to documentation for datasets module
  * interpolate atmosphere file in instrument
  * Correcting coordinate treatment
  * Merge remote-tracking branch 'remotes/upstream/master' into additional_dl1
  * Updated documentation for image module
  * Created test and docs for CameraCalibrator
  * Merge pull request #368 from kosack/remove_ctapipe_extra_submodule
  * fix impact tests to work with new code and add instrument.get_atmosphere_profile
  * Created conveniance calibrator (CameraCalibrator) that applies r1, dl0 and dl1 calibration Updated calibration_pipeline with CameraCalibrator
  * Fixed tests for calibration
  * fix example to use get_dataset
  * Merge remote-tracking branch 'cta-observatory/master' into remove_ctapipe_extra_submodule
  * Merge pull request #387 from kosack/master
  * moved table_interpolator from reco to utils
  * Merge pull request #384 from ParsonsRD/ImPACT_dev
  * Moved charge_extraction.py, waveform_cleaning.py and reductors.py to the images module
  * Update calib.camera docs Removed test_mycam.py Change to absolute python import in dl1.py
  * Merge remote-tracking branch 'remotes/upstream/master' into additional_dl1
  * Merge pull request #386 from kosack/master
  * added tqdm and iminuit as dependencies
  * Update of docs and docstrings for the calibration module. Deleted old docs.
  * Merge pull request #385 from kosack/master
  * removed files accidentally committed to Muon Master
  * Merge pull request #381 from AMWMitchell/muonmaster
  * Merge branch 'array_viewer' into ImPACT_dev
  * Merge pull request #383 from watsonjj/additional_dl1
  * Merge branch 'master' into array_viewer
  * Merge pull request #382 from kosack/master
  * Removed example script (will be replaced later)
  * added FAQ and author list
  * Reverted the dl1 image back to a normal numpy array instead of a masked array Updated documentation for charge extractors
  * Merge branch 'master' into ImPACT_dev
  * Added unit tests
  * Changed code to work with new coordinates scheme
  * added author list
  * Merge pull request #380 from kosack/master
  * fix size hillas parameter display overlay
  * minor bug fixes preparing for merge to master
  * Merge pull request #379 from kosack/master
  * fix #248, now have option to do image cleaning where single pixels above pic thresh are removed
  * Merge remote-tracking branch 'remotes/upstream/master' into muonmaster
  * Merge remote-tracking branch 'remotes/upstream/muonmaster' into muonmaster
  * Merge branch 'muonsimple' into muonmaster
  * Adding muon fitting tests - still need to make image not a masked array
  * Merge pull request #378 from kosack/master
  * add a tool to dump instrument info to FITS files from simtel files
  * Merge pull request #371 from watsonjj/fix_help
  * Merge pull request #377 from watsonjj/additional_dl1
  * Cleanup
  * Corrected integration_correction for telescopes with bin widths not equal to 1
  * Merge branch 'master' into ImPACT_dev
  * Merge pull request #376 from kosack/master
  * fix entry points for chargeresolution
  * Merge pull request #375 from kosack/master
  * change makefile to run pytest directly instead of via setup.py
  * Merge pull request #374 from kosack/master
  * remove psutil dependence in provenance
  * Merge pull request #373 from kosack/master
  * fix import in example
  * add print level and pedantic to minuit
  * Merge pull request #372 from kosack/master
  * change format of time read from monte-carlo
  * Fixed help message for calibration_pipeline example
  * use fixtures in test_serializer to avoid writing files to random places
  * Merge pull request #369 from kosack/master
  * fix provenance test
  * updated provenance test
  * fix loading of array data
  * remove need for CTAPIPE_EXTRAS_DIR, and use ctapipe_resources package
  * Merge pull request #343 from kosack/add_provenence_system
  * add UUID for each activity
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe into add_provenence_system
  * Merge pull request #366 from kosack/implement_pytables_output
  * correctly return from generator in SimpleHDF5TableReader
  * Merge pull request #362 from kosack/implement_pytables_output
  * Merge branch 'master' into array_viewer
  * Added comment
  * Added comment
  * Added comments and improved layout of viewer
  * Changed colour scale and removed print statement
  * Created component for a full event display including the individual camera images and the projection of the Hillas parameters in different frames
  * Added support for the inclusion of multiple axes in the plot
  * Added camera rotation as this is needed for correct reconstruction
  * doc update
  * Merge pull request #361 from kosack/remove_tqdm_dependency
  * Merge pull request #364 from ParsonsRD/coordinates_fix_astropy
  * Fix dimensions problem of correction array
  * SimpleHDF5TableReader.read() now returns a generator
  * Merge branch 'master' into array_viewer
  * more reader functionality
  * basic reader functionality
  * started implementing a reader
  * move more common functionality into base class
  * refactor and add column exclusions
  * Changed pointing direction to take the HorizonFrame (or AltAz) object as a definition of an array pointing direction
  * Update docstrings Change integration_correction arguments to be more generic
  * add more translation functions
  * more docs for hdf5 tables
  * add numpy and tables as requirements
  * better docs and test case
  * improve SimpleHDF5TableWriter
  * add __getitem__ to container to assist with serializers
  * added simplistic hdf5 table writer class
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe into implement_pytables_output
  * Changes to handle ASTRI
  * Merge pull request #360 from kosack/refactor_camera_geometry
  * added a test script
  * use CameraGeometry to get neighbors instead of custom code (large speedup)
  * small cleanups
  * remove unnecessary call to hash() in guess()
  * start brainstorm
  * pass format and other file-related options in `CameraGeometry.from_table`
  * clean up example
  * clean up image docs too
  * minor doc update
  * minor reformatting
  * improve CameraGeometry documentation
  * replace progress bar from tqdm package with astropy one to avoid extra dependency
  * change test to use tmpdir
  * add CameraGeometry.from_table() functionality
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe into refactor_camera_geometry
  * Merge pull request #353 from watsonjj/charge_resolution_mv
  * Merge pull request #359 from watsonjj/additional_dl1
  * added a few utils funcs to help with debugging
  * simplify neighbor pixel calculation (failed for some cameras with gaps)
  * removed extraneous global _geometry_cache
  * simplify tailcuts_clean and dilate
  * undo last commit
  * removed some unused code from neighbor attribute
  * add some comments about the caching of CameraGeometry
  * fix bug in CameraGeometry that broke geometry_converter
  * update docs for cleaning
  * rewrite cleaning.dilate to use neighbor matrix (much faster)
  * fix examples for new cleaning functions
  * fix example
  * change tailcuts_clean algorithm to be faster
  * use astropy.utils.lazyproperty for neighbors
  * Traitlet for adc2pe file
  * Merge remote-tracking branch 'remotes/upstream/master' into additional_dl1
  * memoize CameraGeometry.guess for speed
  * fix some more imports and tests
  * update notebook to use instrument.CameraGeometry
  * clean up
  * refactor to use instrument.CameraGeometry
  * use instrument.CameraGeometry
  * moved io/camera.py to instrument and removed duplicate CameraDescription
  * Updated comment
  * Merge pull request #354 from watsonjj/additional_dl1
  * Merge pull request #357 from kosack/master
  * improve hillas tests
  * Merge pull request #355 from watsonjj/gate_gct_rename
  * Renamed instances of GATE to GCT
  * Cleanup
  * Merge remote-tracking branch 'remotes/upstream/master' into additional_dl1
  * Created tests for waveform_cleaning
  * Corrected AverageWfPeakIntegrator Created test for AverageWfPeakIntegrator
  * Cleanup of charge_extractors Renamed window_start to t0 in charge_extractors. Window shift is now considered in SimpleIntegrator. Created waveform_cleaning.py for waveform cleaning methods to be applied in the dl1 calibration. Currently contains NullWaveformCleaner, CHECMWaveformCleaner, and a factory.
  * Renamed entrypoints for charge resolution
  * Merge pull request #352 from kosack/master
  * set title of displays to cam_id by default, fix pixel spacing
  * Merge pull request #351 from kosack/master
  * use PIXEL_EPSILON with hex cameras too
  * Merge pull request #349 from watsonjj/charge_resolution_mv
  * Merge pull request #350 from kosack/master
  * added small epsilon value to pixel size to fix aliasing
  * Added entrypoints for charge resolution scripts
  * Moved charge resolution scripts into tools
  * Added viewer for nominal system
  * Merge pull request #348 from kosack/image_sum_tool_example
  * add tool example to sum images for similiar cameras and display
  * Merge remote-tracking branch 'refs/remotes/origin/muonsimple' into muonsimple
  * minor fix to file saving
  * add contextmanager for activities
  * Merged with HEAD
  * Added option for drawing the extrapolation of the image axes to the array view. Useful for checking Hillas reconstruction
  * Fixed telescope colours when drawing Hillas params. Reduces default size of markers
  * minor changes for plotting purposes
  * Added scaling factor to ImPACT Templates
  * Merge pull request #5 from AMWMitchell/muonmaster
  * calibration updates
  * Merge pull request #347 from AMWMitchell/muonmaster
  * Merge pull request #4 from AMWMitchell/muonmaster_update
  * Adapting to recent calibration changes
  * Merge branch 'master' into muonmaster_update
  * minor changes
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe into add_provenence_system
  * update notebook
  * add example provenance notebook
  * add Provenance.clear() method
  * add automatic provenance in Tool and io
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe into add_provenence_system
  * use Singleton metaclass for provenance
  * refactoring
  * separate activity provenance from global provenance
  * cleanups to proveance class and core init
  * update provenence
  * first attempt at system provenance and time sampling
  * Merge branch 'master' into ImPACT_dev
  * Increased the requirement on the tolerance (should be more accurate)
  * Modified print outs and file names in ImPACT
  * increasing tail cuts thresholds
  * Updated code for use with newer image templates
  * Merge pull request #330 from AMWMitchell/muonmaster
  * Merge pull request #3 from AMWMitchell/muonsimple
  * making muon code compatible with recent ctapipe updates
  * Added new charge extractor: AverageWfPeakIntegrator
  * Merge pull request #2 from watsonjj/alison_muonsimple
  * Merge remote-tracking branch 'remotes/AMWMitchell/muonmaster' into alison_muonsimple
  * Merge pull request #328 from AMWMitchell/muonmaster
  * Merge pull request #1 from watsonjj/alison_muonmaster
  * Merge remote-tracking branch 'remotes/upstream/master' into alison_muonmaster
  * Improved functions from drawing a single telescope image
  * Merge branch 'array_viewer' into ImPACT_dev
  * Added comments and changed the style in which the telescopes are drawn, now only outlines are drawn with a colour appropriate for the telescope type
  * Added funtionality to the Array drawer to overlay points, and centre the map on that point (useful for checking reconstruction)
  * Adding functionality to array drawing
  * Added functions for overlaying Hillas parameters on the array level view
  * Added simple classes for drawing array-level events
  * Merge branch 'master' into ImPACT
  * Updated ImPACT code to ensure units are handled consistantly within the class
  * Simple fix for mismatch of array dimensions (should be fixed in master soon)
  * Added comments
  * Merge branch 'master' into ImPACT_interpolator
  * Updated the prediction function such that is takes the shower and energy classes as an arguement
  * Major refactoring of the code. Lists used internally all replaced with dictionaries, so that we keep track of the telescopes in the event at all times. Output switched to being the new reconstructed shower container.
  * Added use of new pixel likelihood class
  * Merge branch 'master' into ImPACT_interpolator
  * Added first test for ImPACT
  * Merge branch 'master' into ImPACT_interpolator
  * removing scipy
  * Added additional cuts, limited default reconstruction to only SST-GCT
  * Changed binning of image templates
  * Major changes to the code to deal with the new format of ImPACT tables. The biggest change is the difference in units stored, with the templates now being directly stored as the expected numerb of p.e. for a given telescopes type. Option for fitting the first interaction point has also been added. Scheme for loading of image templates has been altered to load template as required.
  * Change from iminuit to scipy.optimise
  * fixed merge issue
  * Updated cuts and position at which fitting seeds are thrown
  * Added printing of errors
  * Added option for specifying the root directory of the lookup tables, such that this is not hardcoded to a location on a given machine
  * Removed function included my mistake
  * Removed function included my mistake
  * Removed function included my mistake
  * Removed function included my mistake
  * Merge branch 'master' into ImPACT_interpolator
  * Tried to santise the rotation angle in the nominal system conversion, still needs some work
  * Removed factors in angle calculation we are unsure about
  * Reverted changes made by mistake
  * Updated reader to deal with new dictionary based table format
  * Added compatability for older sim_telarray file extensions
  * Update muonsimple to muonmaster
  * Merge pull request #270 from watsonjj/muonmaster_mergemaster
  * Removed pyhessio directory from inside ctapipe
  * Renamed instances of meta.pixel_pos and meta.optical_foclen to inst.pixel_pos and inst.optical_foclen zfits.py requires changing to match the new hessio.py
  * Merge remote-tracking branch 'remotes/upstream/master' into muonmaster
  * Plotting Additions
  * Gaus fit and astropy table
  * Merge branch 'master' into ImPACT_interpolator
  * Modified structure of stored ImPACT templates
  * Adding muon cut dictionary
  * clean up
  * Changed interpolator name to something more general
  * Updated for use with new container classes and pyhessio interface
  * Merge branch 'master' into ImPACT_interpolator
  * Merged with head
  * Added prototype script for the generation of image templates using camera images. Rotates and translates images into the correct system, then you can just take the average.
  * Fixed unit conversion in rotation function
  * Added example analysis script for prod 3. This runs the first pass at the ImPACT analysis implmentation, essentially this is a direct translation of the code implemented within the H.E.S.S. analysis to the CTA framework.
  * Added interpolation features to better deal with edge cases when performing interpolation
  * Added camera rotation value to the output object when guessing the camera type. This is useful later when performing coordinate conversions
  * Fixed error that was thrown due to features now unsupported by python 3.5
  * Fixed broken call to lower edges
  * First pass at creating a template fitting class using iminuit and the template interpolator
  * Created am image template interpolator class. This class loads templates from pickle files and create a scipy interpolator class. This can then be used for providing the model for ImPACT fitting.
  * Merge pull request #253 from AMWMitchell/muonsimple
  * Fixed display of muon image and likelihood fit
  * Changes to chord_length and minor fixes
  * Changing some cuts
  * Compatibility to updated coordinates class
  * Bring muonsimple branch up-to-date with master
  * updating plotting diagnostics
  * Resolved camera rotation issue with muon ring overlay
  * Overlay fitted ring and debugging
  * Adding muon event display
  * first plotting and fewer hard coded values
  * Merge pull request #237 from MaxNoe/muon_features
  * More docs
  * updating fork with muonmaster
  * Add more docstrings and comments
  * Move muon features to image
  * Return nan if fit not successfull
  * Simplify efficiency estimation by using the other fit methods
  * Fix call to norm.pdf in efficiency fit
  * Add efficiency fit function
  * Fix impact_parameter fit: Use bounds and fix integral function
  * Use a binned chisq fit to determine impact
  * Add import for new fit in muon __init__ and fix return value
  * Add impact parameter likelihood calculation
  * Fix calculation of ring_completeness
  * Add function to calculate completness of muon rings
  * Move feature calculation to own file
  * Fix mean_squared_error in muon
  * Add two goodness of fit features for muons
  * Add gaussian likelihood ring fit to calib.array.muon
  * Add support and test for astropy units muon fit
  * Add first ring fitting algorithm in muon submodule
  * Merge pull request #232 from MaxNoe/muonmaster
  * Fix tests
  * Fix PEP8 stuff
  * Use scipy.stats.norm instead of self implemented gaussian pdf
  * Use numpy.pi, one import less
  * Pep 8 stuff
  * Merge pull request #230 from MaxNoe/muonmaster
  * Do not override __init__ of component (This would break everything)
  * Revert "adding astropy_helpers"
  * adding astropy_helpers
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe into muonmaster
  * reco.cleaning moved to image.cleaning
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe into muonmaster
  * Minor parameter adjustments
  * Getting true mirror radius (not hardcoded)
  * Merge branch 'muonmaster' of https://github.com/cta-observatory/ctapipe into muonmaster
  * Getting intensity fitting (hess based algorithm) working on CTA MC
  * Merge pull request #220 from mdpunch/muonmaster
  * Added test to  test_muon_ring_finder, minor modif to muon_ring finder
  * Corrected a bug with the parameters passed to fit_muon function
  * Merge remote-tracking branch 'cta-observatory/muonmaster' into muonmaster
  * Cleaner code
  * First draft of functions to run the muon reconstruction pipeline. DOES  NOT WORK at the moment. The hard coded cuts must be modified first
  * Merge pull request #207 from pcumani/muonmaster
  * Merge pull request #197 from mgaug/muonmaster
  * force units in ChaudhuriKunduRingFitter.fit return the MuonRingParameter container from ChaudhuriKunduRingFitter.fit
  * Example of how to use the muon fit functions and muon reconstruction pipeline (including the image calibration).
  * Fix indentation and container description for MuonRingParameter and MuonIntensityParameter
  * Merge pull request #203 from moralejo/muonmaster
  * Merge branch 'muonmaster' of https://github.com/cta-observatory/ctapipe into muonmaster
  * Merge pull request #201 from pcumani/muonmaster
  * Divide MuonParameter container into two containers: -one for the ring fit output -one for the intensity fit output
  * Added EventContainer to __all__ (was missing in documentation)
  * Merge branch 'muonmaster' of https://github.com/cta-observatory/ctapipe into muonmaster
  * Implemented reading in of telescope mirror dish area and number of mirror tiles from simtelarray files
  * Modify some names in the MuonParameter container, made them more intuitive
  * Merge branch 'muonmaster' of https://github.com/cta-observatory/ctapipe into muonmaster
  * Merge pull request #199 from pcumani/muonmaster
  * Add position on the mirror of the muon impact
  * Add MuonParameter container
  * new base class for the intensity fitters
  * Merge branch 'mgaug-muonmaster' into muonmaster
  * Merge branch 'muonmaster' of https://github.com/mgaug/ctapipe into mgaug-muonmaster
  * Adding __init__ files to new modules. Changes to muon_reco.py for compatibility with changed RingFitter class
  * Merge branch 'mgaug-muonmaster' into muonmaster
  * yet another fix
  * Make the test meaningful by creating a real circle
  * Merge branch 'muonmaster' of https://github.com/mgaug/ctapipe into mgaug-muonmaster
  * Added a test directory Fixed bugs in imports
  * Fixed the documentation format to the standard numpy one in the base class Made the Chaudhuri Kundu ring fitter function derive from the base class
  * Merge pull request #190 from mgaug/muonmaster
  * add the ring fitter base class ,thanks to Kai
  * Adding muon calibration algorithm files from RDP (H.E.S.S. based algorithm)
  * Added test functions
  * Added usints to cuts
  * merged with head
  * Added support for astropy units
  * Added lots of comments
  * Fix now working, values currently hardcoded to HESS I
  * Working minimisation with sensible likelihood
  * Merge branch 'master' into muon
  * Fixed minimisation and added print statements
  * Added minimisation function
  * Fixed Unints
  * Added worker classes and example scripts for muon fitting

0.4.0 / 2017-03-13
==================

  * updated conda environment.yml and setup instructions to use it
  * add History.md
  * update notebook
  * add example provenance notebook
  * add Provenance.clear() method
  * revert .9 -> .1
  * forgot adding this file to staging
  * removed rotation of astri cameras
  * typo
  * moved uncertainty from predict to fit_core_crosses
  * add automatic provenance in Tool and io
  * added uncertainty on position fit
  * fixed missing camera rotation
  * fix bug in ctapipe-dump-triggers tool
  * use Singleton metaclass for provenance
  * refactoring
  * separate activity provenance from global provenance
  * cleanups to proveance class and core init
  * update provenence
  * mini change in doc string
  * first attempt at system provenance and time sampling
  * Corrected tests for charge_resolution
  * Corrected example cmdline arguments
  * some updates to example notebooks
  * Corrected r0 to dl0
  * merge and apply fixes to PR#284
  * Replaced event.dl0 calls to event.r0
  * small fix to example
  * add --all option to ctapipe-info
  * fix broken --dependencies option in ctapipe-info
  * fix in camera_rotation
  * fix bug in CameraDisplay
  * mask in geometry now uses bools instead of int
  * minor camera display update
  * use cta-observatory conda channel for pyhessio install
  * remove version cache from git repo and make sure it's in .gitignore
  * improve camdemo tool
  * add version cache to gitignore
  * remove import of unused npLinAlgError
  * replaced least chi2 calculation with numpy build-in
  * stuff
  * Added pedestal_path and tf_path to CameraR1CalibratorFactor, as real cameras require these separate files for calibration
  * Corrected bug in Factory
  * When saving the pickle file, the directory will created if it doesn't exist
  * Some refactoring
  * Update Charge Resolution
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

n.n.n / 2017-03-13
==================

  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe into add_provenence_system
  * Merge pull request #344 from kosack/master
  * updated conda environment.yml and setup instructions to use it
  * add History.md
  * update notebook
  * Merge pull request #341 from watsonjj/charge_resolution_component
  * Merge pull request #321 from tino-michael/shower_reco
  * add example provenance notebook
  * add Provenance.clear() method
  * revert .9 -> .1
  * forgot adding this file to staging
  * removed rotation of astri cameras
  * typo
  * moved uncertainty from predict to fit_core_crosses
  * add automatic provenance in Tool and io
  * added uncertainty on position fit
  * fixed missing camera rotation
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe into add_provenence_system
  * Merge pull request #342 from kosack/fix_dump_trigger_tool
  * fix bug in ctapipe-dump-triggers tool
  * use Singleton metaclass for provenance
  * refactoring
  * separate activity provenance from global provenance
  * cleanups to proveance class and core init
  * update provenence
  * mini change in doc string
  * first attempt at system provenance and time sampling
  * Merge pull request #329 from tino-michael/geom_mask
  * Merge pull request #339 from watsonjj/small_example_fix
  * Corrected tests for charge_resolution
  * Merge remote-tracking branch 'remotes/upstream/master' into charge_resolution_component
  * Corrected example cmdline arguments
  * Merge pull request #338 from kosack/master
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
  * fix in camera_rotation
  * Merge pull request #333 from kosack/docupdate
  * Merge remote-tracking branch 'cta-observatory/master' into docupdate
  * Merge pull request #331 from kosack/master
  * fix bug in CameraDisplay
  * Merge branch 'master' of https://github.com/cta-observatory/ctapipe into feature/--help
  * mask in geometry now uses bools instead of int
  * Merge remote-tracking branch 'remotes/upstream/master' into r0container
  * minor camera display update
  * Merge pull request #326 from kosack/master
  * use cta-observatory conda channel for pyhessio install
  * Merge pull request #325 from kosack/master
  * remove version cache from git repo and make sure it's in .gitignore
  * Merge pull request #324 from kosack/master
  * improve camdemo tool
  * add version cache to gitignore
  * Merge branch 'r0container' into charge_resolution_component
  * remove import of unused npLinAlgError
  * replaced least chi2 calculation with numpy build-in
  * Merge branch 'master' into shower_reco
  * Merge remote-tracking branch 'remotes/upstream/master' into r0container
  * stuff
  * Added pedestal_path and tf_path to CameraR1CalibratorFactor, as real cameras require these separate files for calibration
  * Corrected bug in Factory
  * When saving the pickle file, the directory will created if it doesn't exist
  * Some refactoring
  * Update Charge Resolution
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
